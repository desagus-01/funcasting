from functools import reduce
from typing import Literal, Mapping

import numpy as np
import polars as pl
from numpy._typing import NDArray
from polars import DataFrame

from maths.sampling import weighted_bootstrapping_idx
from methods.cma import CopulaMarginalModel
from methods.model_selection_pipeline import (
    UnivariateRes,
    get_univariate_results,
)
from methods.preprocess_pipeline import run_univariate_preprocess
from methods.simulation_forecasting import (
    ForecastModel,
    simulate_asset_paths,
)
from models.types import ProbVector
from utils.helpers import drop_nulls_and_compensate_prob


def _build_innovations_df_from_models(
    post: DataFrame,
    model_map: dict[str, UnivariateRes],
    assets=None,
) -> DataFrame:
    if assets is None:
        assets = [c for c in post.columns if c != "date"]

    base = post.select("date")
    patches: list[pl.DataFrame] = []

    for asset in assets:
        model = model_map[asset]
        used_dates = post.filter(pl.col(asset).is_not_null()).select("date")

        non_null_values = post.select(asset).drop_nulls().to_numpy().ravel()
        invariant = model.invariant(non_null_values)

        patch = used_dates.with_columns(pl.Series(asset, invariant))

        patches.append(patch)

    innovations_full = reduce(
        lambda acc, p: acc.join(p, on="date", how="left"), patches, base
    )
    return innovations_full


def draw_innovations(
    invariants_df: DataFrame,
    assets: list[str],
    prob_vector: ProbVector,
    horizon: int,
    n_sims: int,
    seed: int | None,
    method: Literal["bootstrap", "historical", "cma"] = "bootstrap",
    *,
    target_copula: Literal["t", "norm"] | None = None,
    copula_fit_method: Literal["ml", "irho", "itau"] | None = None,
    target_marginals: dict[str, Literal["t", "norm"]] | None = None,
) -> NDArray[np.floating]:
    """
    Draw joint invariant shocks for multiple assets and return a shock tensor.

    Selects `assets` columns from `invariants_df`, drops null rows (with
    probability re-normalization), optionally updates the joint distribution
    using CMA, and then produces shocks using either historical scenarios or
    weighted bootstrap resampling.

    Parameters
    ----------
    invariants_df : polars.DataFrame
        Historical invariants (one column per asset).
    assets : list[str]
        Asset columns to draw jointly. Output asset dimension follows this order.
    prob_vector : ProbVector
        Scenario probabilities for rows of `invariants_df`.
    horizon : int
        Forecast horizon (>= 1).
    n_sims : int
        Number of simulated paths (used for bootstrap/cma).
    seed : int | None
        RNG seed.
    method : {"bootstrap", "historical", "cma"}
        - "historical": return all scenarios as-is
        - "bootstrap": resample scenarios with replacement using `prob_vector`
        - "cma": CMA-update distribution, then bootstrap
    target_copula : {"t", "norm"} | None
        CMA-only: target copula family.
    copula_fit_method : {"ml", "irho", "itau"} | None
        CMA-only: copula fit method (defaults to "itau" if None).
    target_marginals : dict[str, {"t", "norm"}] | None
        CMA-only: per-asset marginal targets.

    Returns
    -------
    simulated_draws : ndarray[float]
        Shock tensor:
        - method="historical": (n_scenarios, 1, n_assets)
        - method in {"bootstrap","cma"}: (n_sims, horizon, n_assets)
    prob : ProbVector
        Probabilities aligned to the scenario rows used internally (after
        null-dropping and CMA, if applied).

    Raises
    ------
    ValueError
        If horizon < 1, or CMA targets are provided when method != "cma".
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    invariants_df = invariants_df.select(assets)
    invariants, prob = drop_nulls_and_compensate_prob(invariants_df, prob_vector)

    if (target_copula is not None or target_marginals is not None) and (
        method != "cma"
    ):
        raise ValueError(
            "You can only have target_marginal and/or target_copula when method is cma!"
        )

    if method == "cma":
        if copula_fit_method is None:
            print("No copula fit method selected, using itau as default")
            copula_fit_method = "itau"

        invariants_cma = CopulaMarginalModel.from_data_and_prob(
            data=invariants, prob=prob
        )

        invariants, prob = invariants_cma.update_distribution(
            seed=seed,
            target_marginals=target_marginals,
            target_copula=target_copula,
            copula_fit_method=copula_fit_method,
        )

    invariants_vector = invariants.to_numpy()

    if method == "historical":
        simulated_draws = invariants_vector[:, None, :]
        return simulated_draws

    n_draws = n_sims * horizon
    idx = weighted_bootstrapping_idx(invariants, prob, n_samples=n_draws, seed=seed)
    simulated_draws = invariants_vector[idx].reshape(n_sims, horizon, len(assets))
    return simulated_draws


def get_assets_models(
    post_process_df: DataFrame,
    assets_univariate_result: Mapping[str, UnivariateRes],
    assets: list[str] | None = None,
):
    assets_ = (
        assets
        if assets is not None
        else [c for c in post_process_df.columns if c != "date"]
    )

    forecast_models: dict[str, ForecastModel] = {}
    for asset in assets_:
        post_series_non_null = (
            post_process_df.select(asset).drop_nulls().to_numpy().ravel()
        )
        forecast_models[asset] = ForecastModel.from_res_and_series(
            fitting_results=assets_univariate_result[asset],
            post_series_non_null=post_series_non_null,
        )

    return forecast_models


# TODO: make better name, this is shit
def run_n_steps_forecast(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    horizon: int = 100,
    n_sims: int = 1000,
    seed: int | None = None,
    method: Literal["bootstrap", "historical", "cma"] = "bootstrap",
    *,
    target_copula: Literal["t", "norm"] | None = None,
    copula_fit_method: Literal["ml", "irho", "itau"] | None = None,
    target_marginals: dict[str, Literal["t", "norm"]] | None = None,
):

    if (horizon > 1) and (method == "historical"):
        raise ValueError(
            "Historical method for innovations can only be used for one step forecasts."
        )

    post_process = run_univariate_preprocess(data=data, assets=assets)

    univariate_results = get_univariate_results(
        data=post_process.post_data,
        assets_to_model=post_process.needs_further_modelling,
    )

    assets_models = get_assets_models(
        post_process_df=post_process.post_data,
        assets_univariate_result=univariate_results,
    )

    invariants_df = _build_innovations_df_from_models(
        post=post_process.post_data,
        model_map=univariate_results,
    )

    innovations = draw_innovations(
        invariants_df=invariants_df,
        assets=assets,
        prob_vector=prob,
        horizon=horizon,
        n_sims=n_sims,
        seed=seed,
        method=method,
        target_copula=target_copula,
        target_marginals=target_marginals,
        copula_fit_method=copula_fit_method,
    )

    assets_forecasts = {}
    for i, asset in enumerate(assets):
        assets_forecasts[asset] = simulate_asset_paths(
            forecast_model=assets_models[asset], innovations=innovations[:, :, i]
        )

    return assets_forecasts

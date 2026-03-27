from __future__ import annotations

import logging
from functools import reduce
from typing import Literal, Mapping

import numpy as np
import polars as pl
from numpy._typing import NDArray
from polars import DataFrame

from models.types import ProbVector
from pipelines.model_selection import (
    get_univariate_results,
)
from pipelines.preprocess import (
    run_univariate_preprocess,
)
from scenarios.copula_marginal import CopulaMarginalModel
from scenarios.resampling import weighted_bootstrapping_idx
from simulation.simulate_paths import (
    simulate_asset_paths,
)
from simulation.state import SimulationForecast
from time_series.models.fitted_types import UnivariateRes
from time_series.transforms.inverses import apply_inverse_transforms
from utils.helpers import drop_nulls_and_compensate_prob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_innovations_df_from_models(
    post: DataFrame,
    model_map: dict[str, UnivariateRes],
    assets: list[str] | None = None,
) -> DataFrame:
    if assets is None:
        assets = [c for c in post.columns if c != "date"]

    logger.info("Building innovations dataframe for assets=%s", assets)

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

    logger.info(
        "Built innovations dataframe with shape=%s",
        innovations_full.shape,
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
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    logger.info(
        "Drawing innovations with method=%s, horizon=%d, n_sims=%d, assets=%s",
        method,
        horizon,
        n_sims,
        assets,
    )

    invariants_df = invariants_df.select(assets)
    invariants, prob = drop_nulls_and_compensate_prob(invariants_df, prob_vector)

    logger.info(
        "Prepared invariant scenarios after null handling: n_scenarios=%d",
        invariants.height,
    )

    if (target_copula is not None or target_marginals is not None) and method != "cma":
        raise ValueError(
            "You can only have target_marginal and/or target_copula when method is cma!"
        )

    if method == "cma":
        if copula_fit_method is None:
            copula_fit_method = "itau"
            logger.info(
                "No copula_fit_method provided; defaulting to '%s'", copula_fit_method
            )

        logger.info(
            "Applying CMA update with target_copula=%s target_marginals=%s",
            target_copula,
            target_marginals,
        )

        invariants_cma = CopulaMarginalModel.from_data_and_prob(
            data=invariants,
            prob=prob,
        )

        invariants, prob = invariants_cma.update_distribution(
            seed=seed,
            target_marginals=target_marginals,
            target_copula=target_copula,
            copula_fit_method=copula_fit_method,
        )

        logger.info("CMA update complete: n_scenarios=%d", invariants.height)

    invariants_vector = invariants.to_numpy()

    if method == "historical":
        logger.info("Returning historical innovations without resampling")
        simulated_draws = invariants_vector[:, None, :]
        return simulated_draws

    n_draws = n_sims * horizon
    logger.info("Bootstrapping %d innovation draws", n_draws)

    idx = weighted_bootstrapping_idx(
        invariants,
        prob,
        n_samples=n_draws,
        seed=seed,
    )
    simulated_draws = invariants_vector[idx].reshape(n_sims, horizon, len(assets))

    logger.info("Innovation draw complete with output shape=%s", simulated_draws.shape)
    return simulated_draws


def get_assets_models(
    data: DataFrame,
    assets_univariate_result: Mapping[str, UnivariateRes],
    assets: list[str] | None = None,
) -> dict[str, SimulationForecast]:
    assets_ = assets if assets is not None else [c for c in data.columns if c != "date"]

    forecast_models: dict[str, SimulationForecast] = {}
    for asset in assets_:
        post_series_non_null = data.select(asset).drop_nulls().to_numpy().ravel()
        forecast_models[asset] = SimulationForecast.from_res_and_series(
            fitting_results=assets_univariate_result[asset],
            post_series_non_null=post_series_non_null,
        )

    return forecast_models


def run_n_steps_forecast(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    horizon: int = 100,
    n_sims: int = 1000,
    seed: int | None = None,
    method: Literal["bootstrap", "historical", "cma"] = "bootstrap",
    *,
    back_to_price: bool = True,
    target_copula: Literal["t", "norm"] | None = None,
    copula_fit_method: Literal["ml", "irho", "itau"] | None = None,
    target_marginals: dict[str, Literal["t", "norm"]] | None = None,
):
    logger.info(
        "Starting n-step forecast: assets=%s horizon=%d n_sims=%d method=%s seed=%s",
        assets,
        horizon,
        n_sims,
        method,
        seed,
    )

    if (horizon > 1) and (method == "historical"):
        raise ValueError(
            "Historical method for innovations can only be used for one step forecasts."
        )
    if (len(assets) <= 1) and (method == "cma"):
        raise ValueError(
            "Must have more than one asset in order to use the copula method."
        )

    post_process = run_univariate_preprocess(data=data, prob=prob, assets=assets)
    logger.info(
        "Preprocessing complete: post_data_shape=%s assets_to_model=%s",
        post_process.post_data.shape,
        post_process.needs_further_modelling,
    )

    univariate_results = get_univariate_results(
        data=post_process.post_data,
        assets_to_model=post_process.needs_further_modelling,
    )
    logger.info(
        "Univariate model selection complete for %d assets",
        len(post_process.needs_further_modelling),
    )

    assets_models = get_assets_models(
        data=post_process.post_data,
        assets_univariate_result=univariate_results,
        assets=assets,
    )

    invariants_df = _build_innovations_df_from_models(
        post=post_process.post_data,
        model_map=univariate_results,
        assets=assets,
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

    logger.info("Simulating asset paths for %d assets", len(assets))

    assets_forecasts = {}
    for i, asset in enumerate(assets):
        assets_forecasts[asset] = simulate_asset_paths(
            forecast_model=assets_models[asset],
            innovations=innovations[:, :, i],
        )

    logger.info("Forecast complete - will apply inverse transforms now")
    transformed = apply_inverse_transforms(
        asset_data_dict=assets_forecasts,
        n_original=data.height,
        inverse_specs=post_process.inverse_specs,
        back_to_price=True if back_to_price else False,
    )

    return assets_forecasts, transformed

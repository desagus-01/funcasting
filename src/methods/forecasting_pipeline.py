from dataclasses import dataclass
from functools import reduce
from typing import Literal

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
from models.types import ProbVector
from utils.helpers import drop_nulls_and_compensate_prob

MeanKind = Literal["none", "demean", "arma"]
VolKind = Literal["none", "garch"]


@dataclass(frozen=True, slots=True)
class UnivariateModel:
    mean_kind: MeanKind
    mean_order: tuple[int, int] = (0, 0)
    mean_params: dict[str, float] | None = None
    vol_kind: VolKind = "none"
    vol_order: tuple[int, int, int] = (0, 0, 0)
    vol_params: dict[str, float] | None = None

    @classmethod
    def from_fitting_results(
        cls,
        fitting_results: UnivariateRes,
    ):
        if fitting_results.mean_res is None:
            mean_kind: MeanKind = "none"
            mean_order = (0, 0)
            mean_params = None
        else:
            mean_kind = fitting_results.mean_res.kind
            mean_order = fitting_results.mean_res.model_order
            if mean_order is None:
                mean_order = (0, 0)
            mean_params = fitting_results.mean_res.params

        if fitting_results.volatility_res is None:
            vol_kind: VolKind = "none"
            vol_order = (0, 0, 0)
            vol_params = None
        else:
            vol_kind = "garch"
            vol_order = fitting_results.volatility_res.model_order
            vol_params = fitting_results.volatility_res.params

        return cls(
            mean_kind=mean_kind,
            mean_order=mean_order,
            mean_params=mean_params,
            vol_kind=vol_kind,
            vol_order=vol_order,
            vol_params=vol_params,
        )


@dataclass(slots=True)
class UnivariateState:
    series_hist: NDArray[np.floating]
    ma_residual_lags: NDArray[np.floating] | None = None
    vol_residual_lags: NDArray[np.floating] | None = None
    var_hist: NDArray[np.floating] | None = None

    @classmethod
    def from_fitting_results(
        cls,
        fitting_results: UnivariateRes,
        univariate_model: UnivariateModel,
        post_series_non_null: NDArray[np.floating],
        x_hist_len: int = 10,
    ):
        x_hist = post_series_non_null[
            -min(x_hist_len, post_series_non_null.size) :
        ].copy()

        eps_mean_hist = None
        if (
            univariate_model.mean_kind == "arma"
            and fitting_results.mean_res is not None
        ):
            p, q = univariate_model.mean_order
            if x_hist.size < p:
                x_hist = post_series_non_null[-p:].copy()
            eps = fitting_results.mean_res.residuals
            eps_mean_hist = eps[-q:].copy() if q > 0 else None

        eps_vol_hist = None
        var_hist = None
        if (
            univariate_model.vol_kind == "garch"
            and fitting_results.volatility_res is not None
        ):
            p_g, o_g, q_g = univariate_model.vol_order
            m = max(p_g, o_g, 1)
            sig2 = fitting_results.volatility_res.conditional_volatility**2
            eps_vol_hist = (
                fitting_results.volatility_res.residuals[-m:].copy() if m > 0 else None
            )
            var_hist = sig2[-q_g:].copy() if q_g > 0 else None

        return cls(
            series_hist=x_hist,
            ma_residual_lags=eps_mean_hist,
            vol_residual_lags=eps_vol_hist,
            var_hist=var_hist,
        )


@dataclass(frozen=True, slots=True)
class ForecastModel:
    model: UnivariateModel
    state0: UnivariateState

    # TODO: Break out the below, confusing atm
    @classmethod
    def from_res_and_series(
        cls,
        fitting_results: UnivariateRes,
        post_series_non_null: NDArray[np.floating],
        x_hist_len: int = 10,
    ):
        if post_series_non_null.size == 0:
            raise ValueError("post_series_non_null is empty")

        model = UnivariateModel.from_fitting_results(fitting_results=fitting_results)
        return cls(
            model=model,
            state0=UnivariateState.from_fitting_results(
                fitting_results=fitting_results,
                univariate_model=model,
                post_series_non_null=post_series_non_null,
                x_hist_len=x_hist_len,
            ),
        )


@dataclass(frozen=True, slots=True)
class MultivariateForecastInfo:
    models: dict[str, ForecastModel]
    invariants: DataFrame


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


# TODO: make this more specific
def multivariate_forecasting_info(
    data: DataFrame, assets: list[str] | None = None
) -> MultivariateForecastInfo:
    post_process = run_univariate_preprocess(data=data, assets=assets)

    univariate_results = get_univariate_results(
        data=post_process.post_data,
        assets_to_model=post_process.needs_further_modelling,
    )

    invariants = _build_innovations_df_from_models(
        post=post_process.post_data,
        model_map=univariate_results,
    )

    assets_ = (
        assets
        if assets is not None
        else [c for c in post_process.post_data.columns if c != "date"]
    )

    forecast_models: dict[str, ForecastModel] = {}
    for asset in assets_:
        post_series_non_null = (
            post_process.post_data.select(asset).drop_nulls().to_numpy().ravel()
        )
        forecast_models[asset] = ForecastModel.from_res_and_series(
            fitting_results=univariate_results[asset],
            post_series_non_null=post_series_non_null,
        )

    return MultivariateForecastInfo(
        models=forecast_models,
        invariants=invariants,
    )


def _param_get(params: dict[str, float], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in params:
            return float(params[k])
    return float(default)


def _get_lag(params: dict[str, float], base: str, lag: int) -> float:
    return _param_get(params, f"{base}[{lag}]", f"{base}.L{lag}", default=0.0)


def _arma_recursive_mean(
    arma_order: tuple[int, int],
    arma_params: dict[str, float],
    state: UnivariateState,
) -> float:
    p, q = arma_order
    mean = 0.0

    # AR
    for i in range(1, p + 1):
        mean += _get_lag(arma_params, "ar", i) * float(state.series_hist[-i])

    # MA
    if q > 0:
        if state.ma_residual_lags is None or state.ma_residual_lags.size < q:
            raise ValueError(f"Need {q} mean residual lags for MA part.")
        for j in range(1, q + 1):
            mean += _get_lag(arma_params, "ma", j) * float(state.ma_residual_lags[-j])

    return mean


def conditional_mean_next(form: UnivariateModel, state: UnivariateState) -> float:
    if form.mean_kind == "none":
        return 0.0
    if form.mean_kind == "demean":
        return float((form.mean_params or {}).get("mean", 0.0))
    if form.mean_kind == "arma":
        return _arma_recursive_mean(form.mean_order, form.mean_params or {}, state)
    raise ValueError(f"Unknown mean_kind: {form.mean_kind}")


def _get_vol_param_coeffs(params: dict[str, float], base: str, n: int) -> np.ndarray:
    """
    Extract coefficients [base[1],...,base[n]] as a vector.
    Missing keys default to 0.0 (robust to absent gamma terms, etc).
    """
    if n <= 0:
        return np.array([], dtype=float)
    out = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        out[i - 1] = float(params.get(f"{base}[{i}]", 0.0))
    return out


def _get_volatility_shock_lags(
    order: int, shock_hist: NDArray[np.floating]
) -> NDArray[np.floating]:
    return shock_hist[-order:][::-1]


def garch_recursion(
    garch_params: dict[str, float],
    garch_order: tuple[int, int, int],
    shock_hist: NDArray[np.floating],
    variance_hist: NDArray[np.floating],
) -> float:
    p, o, q = garch_order
    omega = float(garch_params["omega"])
    alpha = _get_vol_param_coeffs(garch_params, "alpha", p)
    gamma = _get_vol_param_coeffs(garch_params, "gamma", o)
    beta = _get_vol_param_coeffs(garch_params, "beta", q)

    variance_next = omega

    if p > 0:  # arch term
        shock_lags = _get_volatility_shock_lags(p, shock_hist)
        variance_next += float(alpha @ (shock_lags**2))
    if o > 0:  # leverage term (if any)
        shock_lags = _get_volatility_shock_lags(o, shock_hist)
        indicator = (shock_lags < 0.0).astype(float)
        variance_next += float(gamma @ (indicator * (shock_lags**2)))
    if q > 0:  # GARCH terms
        variance_lags = _get_volatility_shock_lags(q, variance_hist)
        variance_next += float(beta @ variance_lags)
    return max(variance_next, 0.0)


def conditional_variance_next(form: UnivariateModel, state: UnivariateState) -> float:
    if form.vol_kind == "none":
        return 0.0
    if form.vol_kind == "garch":
        if state.vol_residual_lags is None or state.var_hist is None:
            raise ValueError("Missing volatility state histories for GARCH.")
        return garch_recursion(
            garch_params=form.vol_params or {},
            garch_order=form.vol_order,
            shock_hist=state.vol_residual_lags,
            variance_hist=state.var_hist,
        )
    raise ValueError(f"Unknown vol_kind: {form.vol_kind}")


def draw_invariant_shock(
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
) -> tuple[NDArray[np.floating], ProbVector]:
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
        return simulated_draws, prob

    n_draws = n_sims * horizon
    idx = weighted_bootstrapping_idx(invariants, prob, n_samples=n_draws, seed=seed)
    simulated_draws = invariants_vector[idx].reshape(n_sims, horizon, len(assets))
    return simulated_draws, prob


#
# def next_step_bootstrap(
#     invariants_df: DataFrame,
#     assets: list[str],
#     models: dict[str, ForecastModel],
#     prob_vector: ProbVector,
#     n_sims: int = 1,
#     seed: int | None = None,
# ) -> dict[str, NDArray[np.floating]]:
#     # checking if any nulls and dropping (can't have this for this type of forecasting)
#     if invariants_df.null_count().sum_horizontal().item() > 0:
#         invariants_no_nulls = invariants_df.drop_nulls()
#         rows_droped = invariants_df.height - invariants_no_nulls.height
#         prob_vector = compensate_prob(prob_vector, rows_droped)
#         print(f"Total of {rows_droped} rows were droped due to nulls.")
#     else:
#         invariants_no_nulls = invariants_df
#
#     invariance_draws = weighted_bootstrapping(
#         invariants_no_nulls, prob_vector, n_sims, seed
#     )
#     selected_assets_models = {
#         asset: model for asset, model in models.items() if asset in assets
#     }
#     next_step_res: dict[str, NDArray[np.floating]] = {}
#     for asset, model in selected_assets_models.items():
#         asset_shock = invariance_draws.select(asset).to_numpy().ravel()
#         mean = conditional_mean_next(model.model.mean_model, model.state0.mean)
#         vol = conditional_variance_next(model.model.volatility_model, model.state0.vol)
#         if vol == 0.0:
#             shock_next = asset_shock
#         else:
#             shock_next = np.sqrt(vol) * asset_shock
#
#         next_step_res[asset] = mean + shock_next
#
#     return next_step_res
#
#
# def next_step_historical(
#     invariants_df: DataFrame,
#     assets: list[str],
#     models: dict[str, ForecastModel],
#     prob_vector: ProbVector,
# ) -> tuple[dict[str, NDArray[np.floating]], ProbVector]:
#     if invariants_df.null_count().sum_horizontal().item() > 0:
#         invariants_no_nulls = invariants_df.drop_nulls()
#         rows_droped = invariants_df.height - invariants_no_nulls.height
#         prob_vector = compensate_prob(prob_vector, rows_droped)
#         print(f"Total of {rows_droped} rows were droped due to nulls.")
#     else:
#         invariants_no_nulls = invariants_df
#
#     selected_assets_models = {
#         asset: model for asset, model in models.items() if asset in assets
#     }
#     next_step_res: dict[str, NDArray[np.floating]] = {}
#     for asset, model in selected_assets_models.items():
#         asset_shock = invariants_no_nulls.select(asset).to_numpy().ravel()
#         mean = conditional_mean_next(model.model.mean_model, model.state0.mean)
#         vol = conditional_variance_next(model.model.volatility_model, model.state0.vol)
#         if vol == 0.0:
#             shock_next = asset_shock
#         else:
#             shock_next = np.sqrt(vol) * asset_shock
#
#         next_step_res[asset] = mean + shock_next
#
#     return next_step_res, prob_vector
#
#
# def next_step_copula_marginal(
#     invariants_df: DataFrame,
#     assets: list[str],
#     models: dict[str, ForecastModel],
#     prob_vector: ProbVector,
#     seed: int | None = None,
#     target_copula: Literal["t", "norm"] | None = None,
#     target_marginals: dict[str, Literal["t", "norm"]] | None = None,
# ) -> tuple[dict[str, NDArray[np.floating]], ProbVector]:
#     if invariants_df.null_count().sum_horizontal().item() > 0:
#         invariants_no_nulls = invariants_df.drop_nulls()
#         rows_droped = invariants_df.height - invariants_no_nulls.height
#         prob_vector = compensate_prob(prob_vector, rows_droped)
#         print(f"Total of {rows_droped} rows were droped due to nulls.")
#     else:
#         invariants_no_nulls = invariants_df
#
#     # perform cma
#     invariants_cma = CopulaMarginalModel.from_data_and_prob(
#         data=invariants_no_nulls, prob=prob_vector
#     )
#
#     invariants_no_nulls, prob_vector = invariants_cma.update_distribution(
#         target_marginals=target_marginals, target_copula=target_copula, seed=seed
#     )
#
#     selected_assets_models = {
#         asset: model for asset, model in models.items() if asset in assets
#     }
#     next_step_res: dict[str, NDArray[np.floating]] = {}
#     for asset, model in selected_assets_models.items():
#         asset_shock = invariants_no_nulls.select(asset).to_numpy().ravel()
#         mean = conditional_mean_next(model.model.mean_model, model.state0.mean)
#         vol = conditional_variance_next(model.model.volatility_model, model.state0.vol)
#         if vol == 0.0:
#             shock_next = asset_shock
#         else:
#             shock_next = np.sqrt(vol) * asset_shock
#
#         next_step_res[asset] = mean + shock_next
#
#     return next_step_res, prob_vector

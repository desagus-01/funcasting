from dataclasses import dataclass
from functools import reduce
from typing import Literal

import numpy as np
import polars as pl
from numpy._typing import NDArray
from polars import DataFrame

from maths.sampling import weighted_bootstrapping
from methods.model_selection_pipeline import (
    UnivariateRes,
    get_univariate_results,
)
from methods.preprocess_pipeline import run_univariate_preprocess
from models.types import ProbVector

MeanKind = Literal["none", "demean", "arma"]
VolKind = Literal["none", "garch"]

ModelType = Literal[
    "ARMA + GARCH", "ARMA", "GARCH", "Random Walk", "Demean", "Demean + GARCH"
]


_MODEL_TYPE_MAP: dict[tuple[MeanKind, VolKind], ModelType] = {
    ("none", "none"): "Random Walk",
    ("none", "garch"): "GARCH",
    ("demean", "none"): "Demean",
    ("demean", "garch"): "Demean + GARCH",
    ("arma", "none"): "ARMA",
    ("arma", "garch"): "ARMA + GARCH",
}


@dataclass(frozen=True, slots=True)
class MeanForm:
    order: tuple[int, int] | None
    kind: Literal["demean", "arma"]
    params: dict[str, float]


@dataclass(frozen=True, slots=True)
class VolForm:
    order: tuple[int, int, int]
    kind: Literal["garch"]
    params: dict[str, float]


@dataclass(frozen=True, slots=True)
class UnivariateModel:
    mean_model: MeanForm | None
    volatility_model: VolForm | None

    @property
    def mean_kind(self) -> MeanKind:
        return "none" if self.mean_model is None else self.mean_model.kind

    @property
    def vol_kind(self) -> VolKind:
        return "none" if self.volatility_model is None else self.volatility_model.kind

    @property
    def model_type(self) -> ModelType:
        return _MODEL_TYPE_MAP[(self.mean_kind, self.vol_kind)]


@dataclass(slots=True)
class MeanState:
    series: NDArray[np.floating]
    mean_residuals: NDArray[np.floating] | None = None


@dataclass(slots=True)
class VolState:
    volatility_residuals: NDArray[np.floating]
    conditional_volatility_sq: NDArray[np.floating]


@dataclass(slots=True)
class UnivariateState:
    mean: MeanState
    vol: VolState | None = None


@dataclass(frozen=True, slots=True)
class ForecastModel:
    form: UnivariateModel
    state0: UnivariateState

    # TODO: Break out the below, confusing atm
    @classmethod
    def from_res_and_series(
        cls,
        res: UnivariateRes,
        post_series_non_null: NDArray[np.floating],
        x_hist_len: int = 10,
    ):
        if post_series_non_null.size == 0:
            raise ValueError("post_series_non_null is empty")

        mean_form = None
        vol_form = None
        if res.mean_res is not None:
            mean_form = MeanForm(
                order=res.mean_res.model_order,
                kind=res.mean_res.kind,
                params=res.mean_res.params,
            )

        if res.volatility_res is not None:
            vol_form = VolForm(
                order=res.volatility_res.model_order,
                kind="garch",
                params=res.volatility_res.params,
            )

        model = UnivariateModel(mean_model=mean_form, volatility_model=vol_form)

        x_hist = post_series_non_null[
            -min(x_hist_len, post_series_non_null.size) :
        ].copy()

        eps_hist_mean = None
        if res.mean_res is not None and res.mean_res.kind == "arma":
            p, q = res.mean_res.model_order
            # ensure x_hist has at least p values
            if x_hist.size < p:
                x_hist = post_series_non_null[-p:].copy()
            eps = res.mean_res.residuals
            eps_hist_mean = eps[-q:].copy() if q > 0 else None

        mean_state = MeanState(series=x_hist, mean_residuals=eps_hist_mean)

        vol_state = None
        if res.volatility_res is not None:
            p_g, o_g, q_g = res.volatility_res.model_order
            m = max(p_g, o_g)
            sig2 = res.volatility_res.conditional_volatility**2
            vol_state = VolState(
                volatility_residuals=res.volatility_res.residuals[-m:].copy(),
                conditional_volatility_sq=sig2[-q_g:].copy(),
            )

        state0 = UnivariateState(mean=mean_state, vol=vol_state)

        return cls(form=model, state0=state0)


@dataclass(frozen=True, slots=True)
class MultivariateForecastInfo:
    models: dict[str, ForecastModel]
    invariants: DataFrame


def _build_innovations_df_from_models(
    post: DataFrame, model_map: dict[str, UnivariateRes], assets=None
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

    return reduce(lambda acc, p: acc.join(p, on="date", how="left"), patches, base)


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
            res=univariate_results[asset],
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
    arma_order: tuple[int, int], arma_params: dict[str, float], mean_state: MeanState
) -> float:
    mean = 0.0
    # AR part
    for i in range(1, arma_order[0] + 1):
        phi_i = _get_lag(arma_params, "ar", i)
        mean += phi_i * float(mean_state.series[-i])
    # MA part
    for j in range(1, arma_order[1] + 1):
        theta_j = _get_lag(arma_params, "ma", j)
        if mean_state.mean_residuals is not None:
            mean += theta_j * float(mean_state.mean_residuals[-j])
    return mean


def conditional_mean_next(mean_form: MeanForm | None, mean_state: MeanState) -> float:
    # no mean case (ie for RW and GARCH only)
    if mean_form is None:
        return 0.0

    if mean_form.kind == "demean":
        return float(mean_form.params["mean"])

    if mean_form.kind == "arma":
        return _arma_recursive_mean(
            arma_order=mean_form.order if mean_form.order is not None else (0, 0),
            arma_params=mean_form.params,
            mean_state=mean_state,
        )
    else:
        raise ValueError(
            f"Your conditional mean model {mean_form.kind} is not part of the possibilities dude"
        )


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
    pass

    p, o, q = garch_order
    omega = float(garch_params["omega"])
    alpha = _get_vol_param_coeffs(garch_params, "alpha", p)
    gamma = _get_vol_param_coeffs(garch_params, "gamma", p)
    beta = _get_vol_param_coeffs(garch_params, "beta", p)

    variance_next = omega

    if p > 0:  # arch term
        shock_lags = _get_volatility_shock_lags(p, shock_hist)
        variance_next += float(alpha @ (shock_hist**2))
    if o > 0:  # leverage term (if any)
        shock_lags = _get_volatility_shock_lags(o, shock_hist)
        indicator = (shock_lags < 0.0).astype(float)
        variance_next += float(gamma @ (indicator * (shock_hist**2)))
    if q > 0:  # GARCH terms
        variance_lags = _get_volatility_shock_lags(q, variance_hist)
        variance_next += float(beta @ variance_lags)
    return max(variance_next, 0.0)


def conditional_volatility_next(
    volatility_form: VolForm | None, volatility_state: VolState | None
) -> float:
    if volatility_form is None:
        return 0.0

    if volatility_form.kind == "garch" and volatility_state is not None:
        return garch_recursion(
            garch_params=volatility_form.params,
            garch_order=volatility_form.order,
            shock_hist=volatility_state.volatility_residuals,
            variance_hist=volatility_state.conditional_volatility_sq,
        )
    else:
        raise ValueError(
            f"Your conditional volatility model {volatility_form.kind} is not accepted here stranger"
        )


def next_step(
    invariants_df: DataFrame,
    assets: list[str],
    models: dict[str, ForecastModel],
    prob_vector: ProbVector,
    n_steps: int = 1,
    seed: int | None = 1,
) -> dict[str, float]:
    invariant_shock = weighted_bootstrapping(invariants_df, prob_vector, n_steps, seed)
    selected_assets_models = {
        asset: model for asset, model in models.items() if asset in assets
    }
    next_step_res: dict[str, float] = {}
    for asset, model in selected_assets_models.items():
        asset_shock = invariant_shock.select(asset).item()
        mean = conditional_mean_next(model.form.mean_model, model.state0.mean)
        vol = conditional_volatility_next(model.form.volatility_model, model.state0.vol)
        if vol == 0.0:
            shock_next = asset_shock
        else:
            shock_next = np.sqrt(vol) * asset_shock

        next_step_res[asset] = float(mean + shock_next)

    return next_step_res

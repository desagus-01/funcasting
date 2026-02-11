from dataclasses import dataclass
from functools import reduce
from typing import Literal

import numpy as np
import polars as pl
from numpy._typing import NDArray
from polars import DataFrame

from methods.model_selection_pipeline import (
    UnivariateRes,
    get_univariate_results,
)
from methods.preprocess_pipeline import run_univariate_preprocess

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
    x_hist: np.ndarray
    eps_hist: np.ndarray | None = None


@dataclass(slots=True)
class VolState:
    eps_hist: np.ndarray
    sig2_hist: np.ndarray


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
        x = np.asarray(post_series_non_null, dtype=float)
        if x.size == 0:
            raise ValueError("post_series_non_null is empty")

        mean_form = None
        if res.mean_res is not None:
            if res.mean_res.kind == "arma":
                mean_form = MeanForm(
                    order=res.mean_res.model_order,
                    kind="arma",
                    params=dict(res.mean_res.params),
                )
            elif res.mean_res.kind == "demean":
                mean_form = MeanForm(
                    order=None,
                    kind="demean",
                    params=dict(res.mean_res.params),
                )
            else:
                raise ValueError(f"Unknown mean kind: {res.mean_res.kind}")

        vol_form = None
        if res.volatility_res is not None:
            vol_form = VolForm(
                order=res.volatility_res.model_order,
                kind="garch",
                params=dict(res.volatility_res.params),
            )

        form = UnivariateModel(mean_model=mean_form, volatility_model=vol_form)

        x_hist = x[-min(x_hist_len, x.size) :].copy()

        eps_hist_mean = None
        if res.mean_res is not None and res.mean_res.kind == "arma":
            p, q = res.mean_res.model_order
            # ensure x_hist has at least p values
            if x_hist.size < p:
                x_hist = x[-p:].copy()
            eps = np.asarray(res.mean_res.residuals, dtype=float)
            eps_hist_mean = eps[-q:].copy() if q > 0 else None

        mean_state = MeanState(x_hist=x_hist, eps_hist=eps_hist_mean)

        vol_state = None
        if res.volatility_res is not None:
            p_g, o_g, q_g = res.volatility_res.model_order
            m = max(p_g, o_g)
            eps_v = np.asarray(res.volatility_res.residuals, dtype=float)
            sigma = np.asarray(res.volatility_res.conditional_volatility, dtype=float)
            sig2 = sigma * sigma

            if eps_v.size < m or sig2.size < q_g:
                raise ValueError(
                    f"Insufficient GARCH history: eps={eps_v.size} need>={m}, sig2={sig2.size} need>={q_g}"
                )

            vol_state = VolState(
                eps_hist=eps_v[-m:].copy(),
                sig2_hist=sig2[-q_g:].copy(),
            )

        state0 = UnivariateState(mean=mean_state, vol=vol_state)

        return cls(form=form, state0=state0)


@dataclass(frozen=True, slots=True)
class MultivariateForecastInfo:
    model: dict[str, ForecastModel]
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
        model=forecast_models,
        invariants=invariants,
    )

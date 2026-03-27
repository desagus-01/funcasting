from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from numpy._typing import NDArray
from polars import DataFrame

from time_series.models.fitted_types import UnivariateRes
from time_series.models.model_types import UnivariateModel


@dataclass(slots=True)
class UnivariateState:
    series_hist: NDArray[np.floating]
    ma_residual_lags: NDArray[np.floating] | None = None
    vol_residual_lags: NDArray[np.floating] | None = None
    var_hist: NDArray[np.floating] | None = None

    @classmethod
    def from_fitting_results_and_model(
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

    @property
    def state_as_dict(self) -> Mapping[str, NDArray[np.floating]]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SimulationForecast:
    model: UnivariateModel
    state0: UnivariateState

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

        if fitting_results.mean_res is None and fitting_results.volatility_res is None:
            diff = np.diff(post_series_non_null)
            scale = float(np.std(diff, ddof=1)) if diff.size > 0 else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0

            model = UnivariateModel(
                mean_kind=model.mean_kind,
                mean_order=model.mean_order,
                mean_params=model.mean_params,
                vol_kind=model.vol_kind,
                vol_order=model.vol_order,
                vol_params=model.vol_params,
                innovation_scale=scale,
            )

        return cls(
            model=model,
            state0=UnivariateState.from_fitting_results_and_model(
                fitting_results=fitting_results,
                univariate_model=model,
                post_series_non_null=post_series_non_null,
                x_hist_len=x_hist_len,
            ),
        )


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

    forecast_models: dict[str, SimulationForecast] = {}
    for asset in assets_:
        post_series_non_null = (
            post_process_df.select(asset).drop_nulls().to_numpy().ravel()
        )
        forecast_models[asset] = SimulationForecast.from_res_and_series(
            fitting_results=assets_univariate_result[asset],
            post_series_non_null=post_series_non_null,
        )

    return forecast_models

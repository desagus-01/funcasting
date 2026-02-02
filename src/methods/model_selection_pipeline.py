from typing import Mapping

import numpy as np
from numpy._typing import NDArray
from polars import DataFrame
from typing_extensions import Literal

from maths.time_series.iid_tests import arch_test, ljung_box_test
from maths.time_series.models import AutoARMARes, DemeanRes, auto_arma, by_criteria

MeanModelRes = AutoARMARes | DemeanRes


def asset_needs_volatility_model(
    residual: NDArray[np.floating],
    degrees_of_freedom: int,
    ljung_box_lags: list[int],
    arch_lags: list[int],
    min_ljung_box_rejections: int = 2,
    min_arch_rejections: int = 1,
) -> bool:
    residual_sq = residual**2
    ljung_box_rejected = ljung_box_test(
        residual_sq, lags=ljung_box_lags, degrees_of_freedom=degrees_of_freedom
    ).rejected
    arch_rejected = arch_test(
        residual, lags_to_test=arch_lags, degrees_of_freedom=degrees_of_freedom
    ).rejected

    return (len(ljung_box_rejected) >= min_ljung_box_rejections) or (
        len(arch_rejected) >= min_arch_rejections
    )


# TODO: Don't love this Mapping thing -> should change?
def needs_volatility_modelling(
    mean_model_res: Mapping[str, MeanModelRes],
    ljung_box_lags: list[int] = [10, 20],
    arch_lags: list[int] = [5, 10, 15],
    min_ljung_box_rejections: int = 2,
    min_arch_rejections: int = 1,
) -> list[str]:
    needs: list[str] = []

    for asset, res in mean_model_res.items():
        if asset_needs_volatility_model(
            residual=res.residuals,
            degrees_of_freedom=res.degrees_of_freedom,
            ljung_box_lags=ljung_box_lags,
            arch_lags=arch_lags,
            min_ljung_box_rejections=min_ljung_box_rejections,
            min_arch_rejections=min_arch_rejections,
        ):
            needs.append(asset)

    return needs


def needs_arma_modelling(data: DataFrame, assets_to_test: list[str]) -> list[str]:
    """
    Identifies assets with significant autocorrelation in series using the Ljung–Box test.

    Note: Test to be run on series at this stage, not residuals.

    Returns a list of asset names that fail the test and require mean modeling.
    """
    needs_mean_modelling = []
    for asset in assets_to_test:
        lj = ljung_box_test(data=data, asset=asset)
        if len(lj.rejected) != 0:
            needs_mean_modelling.append(asset)
    return needs_mean_modelling


# TODO: Among all ARMA models whose residuals pass Ljung–Box, choose the one with the smallest (p+q).
def run_best_arma(
    asset_array: NDArray[np.floating],
    search_n_models: int = 5,
    information_criteria: Literal["bic", "aic"] = "bic",
) -> AutoARMARes:
    """
    Selects the best ARMA model by information criterion (AIC/BIC) among those
    whose residuals pass the Ljung–Box test for no autocorrelation.

    Falls back to the best information criterion model if none pass.
    """

    candidate_models_res = auto_arma(
        asset_array=asset_array,
        max_ar_order=3,
        max_ma_order=3,
        information_criteria=information_criteria,
        top_n_models=search_n_models,
    )

    candidates_by_information_criteria = sorted(candidate_models_res, key=by_criteria)

    for model in candidates_by_information_criteria:
        ar_order, ma_order = model.model_order
        lj_res = ljung_box_test(model.residuals, degrees_of_freedom=ar_order + ma_order)
        if len(lj_res.rejected) == 0:  # if no lag has been rejected stop
            return model
    print(
        "No model's residual has passed the ljung box test, please review your model, returning model with best information criteria for now."
    )

    return min(candidate_models_res, key=by_criteria)


def mean_modelling_pipeline(
    data: DataFrame, assets: list[str]
) -> dict[str, MeanModelRes]:
    """
    Return a mean-modelling result for every asset:
    - AutoARMARes if Ljung–Box suggests mean dependence
    - DemeanRes otherwise
    """
    assets_needing_arma = needs_arma_modelling(data=data, assets_to_test=assets)
    asset_mean_model_res: dict[str, MeanModelRes] = {}
    for asset in assets:
        array = data.select(asset).to_numpy().ravel()
        if asset in assets_needing_arma:
            asset_mean_model_res[asset] = run_best_arma(array)
        else:
            mean = array.mean()
            asset_mean_model_res[asset] = DemeanRes(
                degrees_of_freedom=0,
                mean_=mean,
                residuals=array - mean,
            )
    return asset_mean_model_res

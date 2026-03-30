import logging
from typing import Mapping

import numpy as np
from numpy._typing import NDArray
from polars import DataFrame
from typing_extensions import Literal

from time_series.models.fitted_types import (
    AutoARMARes,
    DemeanRes,
    MeanModelRes,
    UnivariateRes,
)
from time_series.models.mean import (
    auto_arma,
)
from time_series.models.volatility import (
    AutoGARCHRes,
    auto_garch,
)
from time_series.tests.iid import arch_test, ljung_box_test
from time_series.tests.multiple import multiple_tests_rejected

logger = logging.getLogger(__name__)


def by_criteria(res: AutoARMARes | AutoGARCHRes) -> float:
    return res.criteria_res


def _describe_mean_model(res: MeanModelRes | None) -> str:
    if res is None:
        return "random_walk"

    if isinstance(res, DemeanRes):
        return "demean"

    if isinstance(res, AutoARMARes):
        return f"ARMA{res.model_order}"

    return type(res).__name__


def _describe_vol_model(res: AutoGARCHRes | None) -> str:
    if res is None:
        return "none"

    return f"GARCH{res.model_order}"


def describe_univariate_result(res: UnivariateRes) -> str:
    mean_label = _describe_mean_model(res.mean_res)
    vol_label = _describe_vol_model(res.volatility_res)
    return f"mean={mean_label}, vol={vol_label}"


def asset_needs_volatility_model(
    residual: NDArray[np.floating],
    degrees_of_freedom: int,
    ljung_box_lags: list[int],
    arch_lags: list[int],
    min_ljung_box_rejections: int = 2,
    min_arch_rejections: int = 1,
) -> bool:
    """
    Decide whether a volatility (GARCH) model is required for an asset.

    The decision is based on Ljung–Box tests run on squared residuals
    (a proxy for conditional heteroskedasticity) and ARCH tests on
    residuals. If the number of rejected tests exceeds the provided
    thresholds a volatility model is recommended.

    Parameters
    ----------
    residual : NDArray[np.floating]
        Residual series from the mean model (or the series if demeaned).
    degrees_of_freedom : int
        Degrees of freedom to account for in tests (usually equal to number
        of estimated parameters in mean model).
    ljung_box_lags : list[int]
        Lag values to test in the Ljung–Box test applied to squared residuals.
    arch_lags : list[int]
        Lag values to test in the ARCH test applied to residuals.
    min_ljung_box_rejections : int, optional
        Minimum number of rejected Ljung–Box tests required to signal volatility
        dependence (default: 2).
    min_arch_rejections : int, optional
        Minimum number of rejected ARCH tests required to signal volatility
        dependence (default: 1).

    Returns
    -------
    bool
        True when volatility modelling (e.g. GARCH) is recommended.
    """
    residual_sq = residual**2
    ljung_box_pvals = ljung_box_test(  # This is effectively the McLeod- Li test
        residual_sq, lags=ljung_box_lags, degrees_of_freedom=degrees_of_freedom
    ).p_vals
    arch_vals = arch_test(
        residual, lags_to_test=arch_lags, degrees_of_freedom=degrees_of_freedom
    ).p_vals

    ljung_box_rejected = multiple_tests_rejected(ljung_box_pvals)
    arch_rejected = multiple_tests_rejected(arch_vals)
    return (sum(ljung_box_rejected) >= min_ljung_box_rejections) or (
        sum(arch_rejected) >= min_arch_rejections
    )


def asset_needs_mean_modelling(
    data: NDArray[np.floating],
    degrees_of_freedom: int,
    ljung_box_lags: list[int] = [10, 15, 20],
    min_ljung_box_rejections: int = 1,
) -> bool:
    """
    Determine whether an asset series requires mean modelling (ARMA).

    The function runs Ljung–Box tests at multiple lag values and counts the
    number of rejections. If the number of rejections meets or exceeds the
    provided threshold the asset is flagged for mean modelling.

    Parameters
    ----------
    data : NDArray[np.floating]
        Series values (not residuals) to test for autocorrelation.
    degrees_of_freedom : int
        Degrees of freedom to account for in the Ljung–Box statistic.
    ljung_box_lags : list[int], optional
        Lags to test (default: [10, 15, 20]).
    min_ljung_box_rejections : int, optional
        Minimum number of rejected tests across lags to flag the series
        (default: 1).

    Returns
    -------
    bool
        True when the series should be modelled for mean dependence.
    """
    ljung_box = ljung_box_test(
        data=data, lags=ljung_box_lags, degrees_of_freedom=degrees_of_freedom
    )
    ljung_box_rejected = multiple_tests_rejected(ljung_box.p_vals)
    return sum(ljung_box_rejected) >= min_ljung_box_rejections


# TODO: Don't love this Mapping thing -> should change?
def needs_volatility_modelling(
    mean_model_res: Mapping[str, MeanModelRes],
    ljung_box_lags: list[int] = [10, 20],
    arch_lags: list[int] = [5, 10, 15],
    min_ljung_box_rejections: int = 2,
    min_arch_rejections: int = 1,
) -> list[str]:
    """
    Determine which assets require volatility modelling based on residual tests.

    Parameters
    ----------
    mean_model_res : Mapping[str, MeanModelRes]
        Mapping from asset name to mean-model result object (contains residuals).
    ljung_box_lags : list[int], optional
        Lags to use in Ljung–Box tests (default: [10, 20]).
    arch_lags : list[int], optional
        Lags to use in ARCH tests (default: [5, 10, 15]).
    min_ljung_box_rejections : int, optional
        Threshold for number of rejected Ljung–Box tests to flag volatility.
    min_arch_rejections : int, optional
        Threshold for number of rejected ARCH tests to flag volatility.

    Returns
    -------
    list[str]
        Asset names that should receive volatility (GARCH) modelling.
    """
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


def needs_mean_modelling(
    data: DataFrame,
    assets_to_test: list[str],
    degrees_of_freedom: int,
    ljung_box_lags: list[int] = [10, 15, 20],
    min_ljung_box_rejections: int = 1,
) -> list[str]:
    """
    Identifies assets with significant autocorrelation in series using the Ljung–Box test.

    Note: Test to be run on series at this stage, not residuals.

    Parameters
    ----------
    data : DataFrame
        Polars DataFrame containing asset series (columns named by asset).
    assets_to_test : list[str]
        List of asset column names to test.
    degrees_of_freedom : int
        Degrees of freedom to account for in the Ljung–Box statistic.
    ljung_box_lags : list[int], optional
        Set of lags to test (default: [10, 15, 20]).
    min_ljung_box_rejections : int, optional
        Minimum number of rejected tests across lags to flag asset (default: 1).

    Returns
    -------
    list[str]
        Names of assets that fail the Ljung–Box battery and should be modelled
        for mean dependence.
    """
    needs_mean_modelling = []
    for asset in assets_to_test:
        array = (
            data.select(asset).drop_nulls().to_numpy().ravel()
        )  # remove null if asset has any
        if asset_needs_mean_modelling(
            array,
            ljung_box_lags=ljung_box_lags,
            degrees_of_freedom=degrees_of_freedom,
            min_ljung_box_rejections=min_ljung_box_rejections,
        ):
            needs_mean_modelling.append(asset)
    return needs_mean_modelling


def _demean_fallback(asset_array: NDArray[np.floating]) -> DemeanRes:
    """
    Create a simple demean model as a fallback when ARMA estimation is unavailable.

    The function estimates the mean and residuals; it returns a DemeanRes
    containing parameters and residual statistics used downstream.

    Parameters
    ----------
    asset_array : NDArray[np.floating]
        Series values for which to compute the demean fallback.

    Returns
    -------
    DemeanRes
        Result object encapsulating the demean fit and residual diagnostics.
    """
    mean = float(np.mean(asset_array))
    residuals = asset_array - mean
    scale = float(np.std(residuals, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    return DemeanRes(
        model_order=None,
        degrees_of_freedom=0,
        params={"mean": mean},
        residuals=residuals,
        residual_scale=scale,
    )


def get_appropriate_mean_model(
    asset_array: NDArray[np.floating],
    asset_name: str,
    search_n_models: int = 5,
    information_criteria: Literal["bic", "aic"] = "bic",
) -> MeanModelRes:
    """
    Select the best ARMA mean model satisfying residual diagnostics.

    Attempts to estimate a small set of ARMA models using the project's
    auto-ARMA routine and selects the top candidate by the provided
    information criterion whose residuals do not exhibit remaining
    autocorrelation. If no admissible ARMA model exists or none passes
    diagnostics, the function falls back to a simple demean model or the
    best information-criterion model.

    Parameters
    ----------
    asset_array : NDArray[np.floating]
        Series values for model selection.
    asset_name : str
        Asset identifier used for logging warnings.
    search_n_models : int, optional
        Number of top ARMA candidates to consider (default: 5).
    information_criteria : {"bic", "aic"}, optional
        Criterion used to rank candidate models (default: 'bic').

    Returns
    -------
    MeanModelRes
        Selected mean model result (AutoARMARes or DemeanRes).
    """
    try:
        candidate_models_res = auto_arma(
            asset_array=asset_array,
            max_ar_order=3,
            max_ma_order=3,
            information_criteria=information_criteria,
            top_n_models=search_n_models,
        )
    except ValueError:
        logger.warning(
            "Asset=%s no admissible ARMA models found; falling back to demean",
            asset_name,
        )
        return _demean_fallback(asset_array)

    candidates_by_information_criteria = sorted(candidate_models_res, key=by_criteria)

    for model in candidates_by_information_criteria:
        if not asset_needs_mean_modelling(
            model.residuals, degrees_of_freedom=model.degrees_of_freedom
        ):
            return model

    logger.warning(
        "Asset=%s no ARMA candidate passed residual diagnostics; falling back to best %s model",
        asset_name,
        information_criteria.upper(),
    )
    return min(candidate_models_res, key=by_criteria)


def run_best_garch(
    asset_array: NDArray[np.floating],
    asset_name: str,
) -> AutoGARCHRes:
    """
    Select a GARCH model from candidate fits that passes residual diagnostics.

    The function fits a small grid of GARCH models and returns the first
    candidate (ordered by information criterion) whose invariants do not
    indicate remaining ARCH-type dependence. If none pass, the best model
    by information criterion is returned as a fallback.

    Parameters
    ----------
    asset_array : NDArray[np.floating]
        Series values (residuals) to fit candidate GARCH models on.
    asset_name : str
        Asset identifier used for logging.

    Returns
    -------
    AutoGARCHRes
        Selected GARCH model fit result.
    """
    candidate_models_res = auto_garch(
        asset_array=asset_array,
        max_p_order=2,
        max_o_order=1,
        max_q_order=2,
    )

    candidates_by_information_criteria = sorted(candidate_models_res, key=by_criteria)

    for model in candidates_by_information_criteria:
        if not asset_needs_volatility_model(
            residual=model.invariants,
            degrees_of_freedom=model.degrees_of_freedom,
            ljung_box_lags=[10, 20],
            arch_lags=[5, 10, 15],
        ):
            return model
    logger.warning(
        "Asset=%s no GARCH candidate passed residual diagnostics; falling back to best information-criterion model",
        asset_name,
    )

    return min(candidate_models_res, key=by_criteria)


def mean_modelling_pipeline(
    data: DataFrame, assets: list[str]
) -> dict[str, MeanModelRes]:
    """
    Return a mean-modelling result for every asset.

    For each asset in ``assets`` this function selects either an
    AutoARMARes (if mean dependence is detected) or a simple DemeanRes.

    Parameters
    ----------
    data : DataFrame
        Polars DataFrame containing asset columns (and possibly a 'date' col).
    assets : list[str]
        Assets to process in order.

    Returns
    -------
    dict[str, MeanModelRes]
        Mapping from asset name to the selected mean model result object.
    """
    assets_needing_arma = needs_mean_modelling(
        data=data, assets_to_test=assets, degrees_of_freedom=0
    )

    asset_mean_model_res: dict[str, MeanModelRes] = {}
    for asset in assets:
        array = data.select(asset).drop_nulls().to_numpy().ravel()

        if asset in assets_needing_arma:
            res = get_appropriate_mean_model(array, asset_name=asset)
            asset_mean_model_res[asset] = res
            logger.info(
                "Selected mean model for %s: %s", asset, _describe_mean_model(res)
            )
        else:
            mean = array.mean().item()
            residuals = array - mean
            res = DemeanRes(
                model_order=None,
                degrees_of_freedom=0,
                params={"mean": mean},
                residuals=residuals,
                residual_scale=float(np.std(residuals, ddof=1)),
            )
            asset_mean_model_res[asset] = res
            logger.info(
                "Selected mean model for %s: %s", asset, _describe_mean_model(res)
            )

    return asset_mean_model_res


def volatility_modelling_pipeline(
    mean_model_res: Mapping[str, MeanModelRes],
):
    """
    Fit volatility (GARCH) models for assets that require them.

    Parameters
    ----------
    mean_model_res : Mapping[str, MeanModelRes]
        Mapping of asset name to mean-model results which provide residuals
        used to decide if volatility modelling is required.

    Returns
    -------
    dict[str, AutoGARCHRes]
        Mapping from asset name to selected GARCH model result for assets
        that required volatility modelling.
    """
    assets_needing_garch = needs_volatility_modelling(mean_model_res)
    logger.info("Assets needing volatility modelling: %s", assets_needing_garch)

    asset_vol_model_res = {}
    for asset in assets_needing_garch:
        res = run_best_garch(mean_model_res[asset].residuals, asset)
        asset_vol_model_res[asset] = res
        logger.info(
            "Selected volatility model for %s: %s", asset, _describe_vol_model(res)
        )

    return asset_vol_model_res


def get_univariate_results(
    data: DataFrame, assets_to_model: list[str]
) -> dict[str, UnivariateRes]:
    """
    Run mean and volatility modelling pipelines and aggregate results per asset.

    Parameters
    ----------
    data : DataFrame
        Polars DataFrame containing asset series (columns named by asset).
    assets_to_model : list[str]
        List of assets for which parametric models should be estimated for
        either mean or volatility. Assets not listed will still be included
        in the final mapping with ``None`` for missing components.

    Returns
    -------
    dict[str, UnivariateRes]
        Mapping of every asset (columns in ``data`` except 'date') to a
        UnivariateRes object containing mean and volatility fit results (or None).
    """
    mean_modelling = mean_modelling_pipeline(data=data, assets=assets_to_model)
    volatility_modelling = volatility_modelling_pipeline(mean_model_res=mean_modelling)

    all_assets = [c for c in data.columns if c != "date"]
    asset_model = {}

    for asset in all_assets:
        asset_model[asset] = UnivariateRes(
            mean_res=mean_modelling.get(asset),
            volatility_res=volatility_modelling.get(asset),
        )
        logger.info(
            "Final univariate result for %s: %s",
            asset,
            describe_univariate_result(asset_model[asset]),
        )

    return asset_model

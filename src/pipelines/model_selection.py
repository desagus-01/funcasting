import logging
from typing import Mapping

import numpy as np
from numpy._typing import NDArray
from polars import DataFrame
from typing_extensions import Literal

from time_series.models.fitted_types import (
    AutoARMARes,
    AutoGARCHRes,
    DemeanRes,
    MeanModelRes,
    UnivariateRes,
)
from time_series.models.mean import (
    auto_arma,
)
from time_series.models.model_quality import (
    QualityConfig,
    SelectionAudit,
    score_audit,
)
from time_series.models.volatility import (
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
    """
    ljung_box = ljung_box_test(
        data=data, lags=ljung_box_lags, degrees_of_freedom=degrees_of_freedom
    )
    ljung_box_rejected = multiple_tests_rejected(ljung_box.p_vals)
    return sum(ljung_box_rejected) >= min_ljung_box_rejections


def needs_volatility_modelling(
    mean_model_res: Mapping[str, MeanModelRes],
    ljung_box_lags: list[int] = [10, 20],
    arch_lags: list[int] = [5, 10, 15],
    min_ljung_box_rejections: int = 2,
    min_arch_rejections: int = 1,
) -> list[str]:
    """
    Determine which assets require volatility modelling based on residual tests.
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


def _select_first_candidate_with_valid_residuals(
    candidates_by_information_criteria: list[AutoARMARes],
) -> AutoARMARes | None:
    for model in candidates_by_information_criteria:
        if not asset_needs_mean_modelling(
            model.residuals, degrees_of_freedom=model.degrees_of_freedom
        ):
            return model
    return None


def get_appropriate_mean_model(
    asset_array: NDArray[np.floating],
    asset_name: str,
    max_ar_order: int = 2,
    max_ma_order: int = 2,
    search_n_models: int = 5,
    information_criteria: Literal["bic", "aic"] = "bic",
    audit: SelectionAudit | None = None,
) -> MeanModelRes:
    """
    Select the best ARMA mean model satisfying residual diagnostics.
    """
    try:
        candidate_models_res = auto_arma(
            asset_array=asset_array,
            max_ar_order=max_ar_order,
            max_ma_order=max_ma_order,
            information_criteria=information_criteria,
            top_n_models=search_n_models,
        )
    except ValueError:
        logger.warning(
            "Asset=%s no admissible ARMA models found; falling back to demean",
            asset_name,
        )
        if audit is not None:
            audit.add_event(
                "MEAN_FALLBACK_DEMEAN", f"Asset={asset_name} no admissible ARMA models"
            )
        return _demean_fallback(asset_array)

    candidates_by_information_criteria = sorted(candidate_models_res, key=by_criteria)
    model = _select_first_candidate_with_valid_residuals(
        candidates_by_information_criteria
    )
    if model is not None:
        return model

    logger.warning(
        "Asset=%s no ARMA candidate passed residual diagnostics; falling back to best %s model",
        asset_name,
        information_criteria.upper(),
    )
    if audit is not None:
        audit.add_event(
            "MEAN_FALLBACK_BEST_IC_NO_DIAG_PASS",
            f"Asset={asset_name} no ARMA candidate passed residual diagnostics; fallback to best {information_criteria.upper()}",
        )
    return min(candidate_models_res, key=by_criteria)


def _select_model_with_valid_arch_tests(
    candidates_by_information_criteria: list[AutoGARCHRes],
) -> AutoGARCHRes | None:
    for model in candidates_by_information_criteria:
        invariants = np.asarray(model.invariants, dtype=float)
        if not np.all(np.isfinite(invariants)):
            continue

        if not asset_needs_volatility_model(
            residual=invariants,
            degrees_of_freedom=model.degrees_of_freedom,
            ljung_box_lags=[10, 20],
            arch_lags=[5, 10, 15],
        ):
            return model
    return None


def run_best_garch(
    asset_array: NDArray[np.floating],
    asset_name: str,
    audit: SelectionAudit | None = None,
) -> AutoGARCHRes | None:
    """
    Select a GARCH model from candidate fits that passes residual diagnostics.
    """
    try:
        candidate_models_res = auto_garch(
            asset_array=asset_array,
            max_p_order=2,
            max_o_order=1,
            max_q_order=2,
        )
    except ValueError as exc:
        logger.warning(
            "Asset=%s no usable GARCH candidates were fitted; keeping volatility model as none (%s)",
            asset_name,
            exc,
        )
        return None

    candidates_by_information_criteria = sorted(candidate_models_res, key=by_criteria)
    model = _select_model_with_valid_arch_tests(candidates_by_information_criteria)
    if model is not None:
        return model

    logger.warning(
        "Asset=%s no GARCH candidate passed residual diagnostics; falling back to best information-criterion model",
        asset_name,
    )
    if audit is not None:
        audit.add_event(
            "VOL_FALLBACK_BEST_IC_NO_DIAG_PASS",
            f"Asset={asset_name} no GARCH candidate passed residual diagnostics; fallback to best information-criterion",
        )

    return min(candidate_models_res, key=by_criteria)


def mean_modelling_pipeline(
    data: DataFrame, assets: list[str]
) -> tuple[dict[str, MeanModelRes], dict[str, SelectionAudit]]:
    """
    Return a mean-modelling result for every asset, plus a selection audit per asset.
    """
    assets_needing_arma = needs_mean_modelling(
        data=data, assets_to_test=assets, degrees_of_freedom=0
    )

    asset_mean_model_res: dict[str, MeanModelRes] = {}
    mean_audits: dict[str, SelectionAudit] = {}
    for asset in assets:
        array = data.select(asset).drop_nulls().to_numpy().ravel()

        audit = SelectionAudit(events=[], notes=[])
        if asset in assets_needing_arma:
            res = get_appropriate_mean_model(array, asset_name=asset, audit=audit)
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

        mean_audits[asset] = audit

    return asset_mean_model_res, mean_audits


def volatility_modelling_pipeline(
    mean_model_res: Mapping[str, MeanModelRes],
) -> tuple[dict[str, AutoGARCHRes], dict[str, SelectionAudit]]:
    """
    Fit volatility (GARCH) models for assets that require them and return audits.
    """
    assets_needing_garch = needs_volatility_modelling(mean_model_res)
    logger.info("Assets needing volatility modelling: %s", assets_needing_garch)

    asset_vol_model_res: dict[str, AutoGARCHRes] = {}
    vol_audits: dict[str, SelectionAudit] = {}

    for asset in assets_needing_garch:
        audit = SelectionAudit(events=[], notes=[])
        res = run_best_garch(mean_model_res[asset].residuals, asset, audit=audit)
        if res is not None:
            asset_vol_model_res[asset] = res
        vol_audits[asset] = audit
        logger.info(
            "Selected volatility model for %s: %s", asset, _describe_vol_model(res)
        )

    return asset_vol_model_res, vol_audits


def get_univariate_results(
    data: DataFrame, assets_to_model: list[str]
) -> dict[str, UnivariateRes]:
    """
    Run mean and volatility modelling pipelines and aggregate results per asset, including quality.
    """
    mean_modelling, mean_audits = mean_modelling_pipeline(
        data=data, assets=assets_to_model
    )
    volatility_modelling, vol_audits = volatility_modelling_pipeline(
        mean_model_res=mean_modelling
    )

    all_assets = [c for c in data.columns if c != "date"]
    asset_model: dict[str, UnivariateRes] = {}

    for asset in all_assets:
        # combine audits
        combined_audit = SelectionAudit(events=[], notes=[])
        if asset in mean_audits:
            combined_audit.events.extend(mean_audits[asset].events)
            combined_audit.notes.extend(mean_audits[asset].notes)
        if asset in vol_audits:
            combined_audit.events.extend(vol_audits[asset].events)
            combined_audit.notes.extend(vol_audits[asset].notes)

        quality = score_audit(combined_audit, QualityConfig())
        print(f"{asset} model is {quality}")

        asset_model[asset] = UnivariateRes(
            mean_res=mean_modelling.get(asset),
            volatility_res=volatility_modelling.get(asset),
            quality=quality,
        )
        logger.info(
            "Final univariate result for %s: %s",
            asset,
            describe_univariate_result(asset_model[asset]),
        )

    return asset_model

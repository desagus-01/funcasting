import logging
from typing import Literal

import polars as pl
from polars.dataframe.frame import DataFrame

from models.types import ProbVector
from time_series.preprocessing.apply import (
    apply_deseason,
    apply_detrend,
    overwrite_with_transforms,
)
from time_series.preprocessing.decisions import (
    deseason_decision_rule,
    detrend_decision_rule,
)
from time_series.preprocessing.types import (
    AppliedTransform,
    PipelineAssetBatchRes,
    TransformDecision,
    UnivariatePreprocess,
)
from time_series.preprocessing.white_noise import test_increments_idd
from time_series.selection.seasonality import (
    seasonality_diagnostic,
)
from time_series.selection.trend import trend_diagnostic
from utils.helpers import (
    get_assets_names,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def deseason_pipeline(
    data: DataFrame,
    assets: list[str] | None = None,
    include_diagnostics: bool = False,
) -> PipelineAssetBatchRes:
    """Diagnose, decide, and apply deterministic seasonal adjustment."""
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    diagnostics = seasonality_diagnostic(
        data=data,
        assets=assets,
    )

    decision = deseason_decision_rule(diagnostics)

    updated, inverse_specs = apply_deseason(
        data=data,
        decision=decision,
    )

    return PipelineAssetBatchRes(
        type="seasonality",
        decision=decision,
        inverse_spec=inverse_specs,
        updated_data=updated,
        all_tests=diagnostics if include_diagnostics else None,
    )


def detrend_pipeline(
    data: DataFrame,
    assets: list[str] | None = None,
    order_max: int = 3,
    threshold_order: int = 2,
    include_diagnostics: bool = False,
    *,
    trend_type: Literal["deterministic", "stochastic", "both"] = "both",
) -> PipelineAssetBatchRes:
    """Diagnose, decide, and apply detrending per asset."""
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    diagnostics = trend_diagnostic(
        data=data,
        assets=assets,
        order_max=order_max,
        threshold_order=threshold_order,
        trend_type=trend_type,
    )

    per_asset_decision = detrend_decision_rule(
        detrend_res=diagnostics,
        assets=assets,
    )

    updated, inverse_specs = apply_detrend(
        data=data,
        decision=per_asset_decision,
    )

    return PipelineAssetBatchRes(
        type="trend",
        decision=per_asset_decision,
        inverse_spec=inverse_specs,
        updated_data=updated,
        all_tests=diagnostics if include_diagnostics else None,
    )


# TODO: Review dropping nulls blankly - prob is a better way
def run_univariate_preprocess(
    data: pl.DataFrame,
    prob: ProbVector,
    assets: list[str] | None = None,
) -> UnivariatePreprocess:
    """
    Pipeline:
      1) Screen assets by increments white-noise
      2) Detrend selected assets
      3) Deseason selected assets
    Returns:
      UnivariatePreprocess
    """

    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    logger.info(
        "Starting univariate preprocess: rows=%d assets=%s",
        data.height,
        assets,
    )

    assets_need_preprocess = test_increments_idd(
        data=data, original_prob=prob, assets=assets
    )
    applied_transforms: dict[str, list[AppliedTransform]] = {
        asset: [] for asset in assets_need_preprocess
    }

    # Trend
    detrend = detrend_pipeline(
        data=data.select(["date", *assets]),
        assets=assets_need_preprocess,
        include_diagnostics=False,
    )
    for asset, decision in detrend.decision.items():
        applied_transforms[asset].append(
            AppliedTransform(
                asset=asset,
                decision=decision,
                inverse_spec=detrend.inverse_spec[asset]
                if detrend.inverse_spec
                else None,
            )
        )

    after_detrend = overwrite_with_transforms(
        base=data, patch=detrend.updated_data, assets=assets, suffix="_detrend"
    )

    # Seasonality
    deseason = deseason_pipeline(
        data=after_detrend.select(["date", *assets]),
        assets=assets_need_preprocess,
        include_diagnostics=False,
    )
    for asset, seasons in deseason.decision.items():
        if seasons:
            applied_transforms[asset].append(
                AppliedTransform(
                    asset=asset,
                    decision=TransformDecision(kind="seasonality", order=None),
                    inverse_spec=deseason.inverse_spec[asset]
                    if deseason.inverse_spec
                    else None,
                )
            )

    final = overwrite_with_transforms(
        base=after_detrend,
        patch=deseason.updated_data,
        assets=assets,
        suffix="_deseason",
    )

    logger.info("Finished univariate preprocess: results=%s", applied_transforms)

    return UnivariatePreprocess(
        post_data=final,
        inverse_specs=applied_transforms,
        needs_further_modelling=assets_need_preprocess,
    )

from dataclasses import dataclass
from typing import Callable, Literal

from polars import Series
from polars.dataframe.frame import DataFrame

from globals import LAGS
from maths.distributions import uniform_probs
from maths.helpers import add_detrend_column, add_differenced_columns
from maths.time_series.diagnostics.seasonality import (
    SeasonalityPeriodTest,
    seasonality_diagnostic,
)
from maths.time_series.diagnostics.trends import TrendTest, trend_diagnostic
from maths.time_series.iid_tests import (
    TestResultByAsset,
    copula_lag_independence_test,
    ellipsoid_lag_test,
    univariate_kolmogrov_smirnov_test,
)
from maths.time_series.operations import deterministic_deseasoning
from methods.cma import CopulaMarginalModel
from models.types import ProbVector
from utils.helpers import (
    get_assets_names,
)


@dataclass(frozen=True)
class PipelineOutcome:
    type: Literal["trend", "seasonality"]
    decision: dict[str, tuple[str, int]] | dict[str, list[tuple[str, float]]]
    updated_data: DataFrame
    all_tests: (
        dict[str, dict[str, TrendTest]] | dict[str, list[SeasonalityPeriodTest]] | None
    )


DiagnosticFun = Callable[..., dict]
DecisionFun = Callable[..., dict]
ApplyFun = Callable[..., DataFrame]


def _run_pipeline(
    *,
    data: DataFrame,
    assets: list[str] | None,
    type_label: Literal["trend", "seasonality"],
    diagnostic_fn: DiagnosticFun,
    decision_rule: DecisionFun,
    apply_fn: ApplyFun,
    include_diagnostics: bool,
    **diag_kwargs,
) -> PipelineOutcome:
    # Resolve assets once; keep consistent with your helpers
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    # 1) diagnose
    diagnostic = diagnostic_fn(data=data, assets=assets, **diag_kwargs)

    # 2) decide
    # (Trend decision often needs assets order; seasonality doesn’t—this signature stays flexible)
    decision = decision_rule(diagnostic, assets=assets)

    # 3) apply — let your apply_* do the batching/fusing
    updated = apply_fn(data=data, decision=decision)

    return PipelineOutcome(
        type=type_label,
        decision=decision,
        updated_data=updated,
        all_tests=(diagnostic if include_diagnostics else None),
    )


def run_all_iid_tests(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    lags: int = LAGS["testing"],
) -> dict[str, TestResultByAsset]:
    ellipsoid_test = ellipsoid_lag_test(data=data, prob=prob, lags=lags, assets=assets)
    copula_marginal_model = CopulaMarginalModel.from_data_and_prob(data=data, prob=prob)

    copula_lag_test_res = copula_lag_independence_test(
        copula=copula_marginal_model.copula,
        prob=copula_marginal_model.prob,
        lags=lags,
        assets=assets,
    )

    ks_test = univariate_kolmogrov_smirnov_test(data=data, assets=assets)

    return {
        "ellipsoid": ellipsoid_test,
        "copula": copula_lag_test_res,
        "ks": ks_test,
    }


def check_white_noise(
    data: DataFrame,
    prob: ProbVector | None = None,
    assets: list[str] | None = None,
    lags: int = LAGS["testing"],
) -> dict[str, bool]:
    """
    Runs 3 tests:
    1.Copula independence test on lags
    2.Kolmogrov Smirnov Test
    3.Ellipsoid test on lags

    Per asset, if any fails (in any lag) then returns as False, else True

    """
    if prob is None:
        prob = uniform_probs(data.height)

    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    wn_tests = run_all_iid_tests(data=data, prob=prob, assets=assets, lags=lags)
    return {
        a: not any(wn_tests[t][a].rejected for t in ("ellipsoid", "copula", "ks"))
        for a in assets
    }


def detrend_decision_rule(
    detrend_res: dict[str, dict[str, TrendTest]], assets: list[str]
) -> dict[str, tuple[str, int]]:
    trend_trans = {}
    for asset in assets:
        deterministic_res = detrend_res["deterministic"][
            asset
        ].transformation_order_needed
        stochastic_res = detrend_res["stochastic"][asset].transformation_order_needed

        candidates = []

        if deterministic_res is not None:
            candidates.append(("polynomial", deterministic_res))

        if stochastic_res:
            candidates.append(("difference", stochastic_res))

        if not candidates:
            continue

        transformation, order = min(
            candidates, key=lambda x: (x[1], x[0] != "polynomial")
        )

        trend_trans[asset] = (transformation, order)
    return trend_trans


def apply_detrend(data: DataFrame, decision: dict[str, tuple[str, int]]) -> DataFrame:
    if not decision:
        return data

    # will group assets by transform
    by_group: dict[tuple[str, int], list[str]] = {}
    for asset, (transform, order) in decision.items():
        by_group.setdefault((transform, order), []).append(asset)

    for (transform, order), assets in by_group.items():
        if transform == "difference":
            data = add_differenced_columns(
                data=data, assets=assets, difference=order, keep_all=True
            )
        elif transform == "polynomial":
            data = add_detrend_column(
                data=data, assets=assets, polynomial_orders=[order]
            )
        else:
            raise ValueError(f"Unknown transform '{transform}' for assets {assets}")

    # renames to original cols
    rename_map = {}
    for asset, (transform, order) in decision.items():
        col = (
            f"{asset}_diff_{order}"
            if transform == "difference"
            else f"{asset}_detrended_p{order}"
        )
        if col not in data.columns:
            raise ValueError(
                f"Expected transformed column '{col}' not found. "
                f"Available columns: {data.columns}"
            )
        rename_map[col] = asset

    return data.drop(list(decision.keys())).rename(rename_map)


def deseason_decision_rule(
    seasonality_diagnostic: dict[str, list[SeasonalityPeriodTest]],
) -> dict[str, list[tuple[str, float]]]:
    return {
        asset: [
            (period.seasonal_period, period.seasonal_frequency_radian)
            for period in res
            if period.evidence_of_seasonality
        ]
        for asset, res in seasonality_diagnostic.items()
    }


def deseason_apply(
    data: DataFrame, decision: dict[str, list[tuple[str, float]]]
) -> DataFrame:
    if not any(decision.values()):
        return data
    deseasoned_assets = {
        asset: deterministic_deseasoning(
            data,
            asset=asset,
            frequency_radians=[rad for _, rad in seasons],
        )[asset]
        for asset, seasons in decision.items()
        if seasons
    }
    return data.with_columns(
        [Series(name, values) for name, values in deseasoned_assets.items()]
    )


def deseason_pipeline(
    data: DataFrame,
    assets: list[str] | None = None,
    include_diagnostics: bool = False,
) -> PipelineOutcome:
    return _run_pipeline(
        data=data,
        assets=assets,
        type_label="seasonality",
        diagnostic_fn=lambda **kw: seasonality_diagnostic(
            data=kw["data"],
            assets=kw["assets"],
        ),
        decision_rule=lambda tests, assets=None: deseason_decision_rule(tests),
        # adapt to your current signature: deseason_apply(data=..., deseason_rule=...)
        apply_fn=lambda *, data, decision: deseason_apply(data=data, decision=decision),
        include_diagnostics=include_diagnostics,
    )


def detrend_pipeline(
    data: DataFrame,
    assets: list[str] | None = None,
    order_max: int = 3,
    threshold_order: int = 2,
    include_diagnostics: bool = False,
    *,
    trend_type: Literal["deterministic", "stochastic", "both"] = "both",
) -> PipelineOutcome:
    return _run_pipeline(
        data=data,
        assets=assets,
        type_label="trend",
        diagnostic_fn=lambda **kw: trend_diagnostic(
            data=kw["data"],
            assets=kw["assets"],
            order_max=order_max,
            threshold_order=threshold_order,
            trend_type=trend_type,
        ),
        decision_rule=lambda tests, assets=None: detrend_decision_rule(
            detrend_res=tests, assets=(assets or [])
        ),
        # adapt to your current signature: apply_detrend(data=..., detrend_needed=...)
        apply_fn=lambda *, data, decision: apply_detrend(data=data, decision=decision),
        include_diagnostics=include_diagnostics,
    )

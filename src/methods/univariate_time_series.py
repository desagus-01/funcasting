from dataclasses import dataclass
from typing import Literal

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
    type: str
    decision: dict[str, tuple[str, int]] | dict[str, list[tuple[str, float]]]
    updated_data: DataFrame
    all_tests: dict[str, dict[str, TrendTest]] | dict[str, list[SeasonalityPeriodTest]]


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
):
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
    results = {}
    for asset in assets:
        failed = any(
            wn_tests[test][asset].rejected for test in ("ellipsoid", "copula", "ks")
        )
        results[asset] = not failed

    return results


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


def apply_detrend(
    data: DataFrame, detrend_needed: dict[str, tuple[str, int]]
) -> DataFrame:
    """
    Applies the chosen detrend transform per asset, then renames the resulting
    transformed column back to the original asset name (so downstream code
    always refers to the same column name).
    """
    for asset, (transform, order) in detrend_needed.items():
        if transform == "difference":
            transformed_col = f"{asset}_diff_{order}"
            data = add_differenced_columns(
                data=data, assets=[asset], difference=order, keep_all=True
            )
        elif transform == "polynomial":
            transformed_col = f"{asset}_detrended_p{order}"
            data = add_detrend_column(
                data=data, assets=[asset], polynomial_orders=[order]
            )
        else:
            raise ValueError(f"Unknown transform '{transform}' for asset '{asset}'")

        if transformed_col not in data.columns:
            raise ValueError(
                f"Expected transformed column '{transformed_col}' not found. "
                f"Available columns: {data.columns}"
            )

        data = data.drop(asset).rename({transformed_col: asset})

    return data


def detrend_pipeline(
    data: DataFrame,
    assets: list[str] | None = None,
    order_max: int = 3,
    threshold_order: int = 2,
    include_diagnostics: bool = False,
    *,
    trend_type: Literal["deterministic", "stochastic", "both"] = "both",
) -> DataFrame | PipelineOutcome:
    """
    Runs Stationarity tests on both deterministic (polynomial) and stochastic (differences) series to check if a transformation is needed to make series stationary.

    Decision rule is to go with lowest order one.
    """
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    assets_trend_res = trend_diagnostic(
        data, assets, order_max, threshold_order, trend_type=trend_type
    )

    detrends_needed = detrend_decision_rule(detrend_res=assets_trend_res, assets=assets)

    updated_df = apply_detrend(data=data, detrend_needed=detrends_needed)

    if include_diagnostics:
        return PipelineOutcome(
            type="trend",
            decision=detrends_needed,
            updated_data=updated_df,
            all_tests=assets_trend_res,
        )
    return updated_df


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


# TODO: Finish the deseason apply func below (check what did with detrend)
def deseason_apply(
    data: DataFrame, deseason_rule: dict[str, list[tuple[str, float]]]
) -> DataFrame:
    deseasoned_assets = {
        asset: deterministic_deseasoning(
            data,
            asset=asset,
            frequency_radians=[rad for _, rad in seasons],
        )[asset]
        for asset, seasons in deseason_rule.items()
        if seasons
    }
    return data.with_columns(
        [Series(name, values) for name, values in deseasoned_assets.items()]
    )


def deseason_pipeline(
    data: DataFrame, assets: list[str] | None = None, include_diagnostics: bool = False
) -> PipelineOutcome | DataFrame:
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    seasonality_res = seasonality_diagnostic(data=data, assets=assets)

    decision_rule = deseason_decision_rule(seasonality_res)

    updated_df = deseason_apply(data, decision_rule)

    if include_diagnostics:
        return PipelineOutcome(
            type="seasonality",
            decision=decision_rule,
            updated_data=updated_df,
            all_tests=seasonality_res,
        )
    return updated_df

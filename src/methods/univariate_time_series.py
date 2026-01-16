from typing import Literal

from polars.dataframe.frame import DataFrame

from globals import LAGS
from maths.distributions import uniform_probs
from maths.helpers import add_detrend_column, add_differenced_columns
from maths.time_series.diagnostics.trends import TrendTest, trend_diagnostic
from maths.time_series.iid_tests import (
    TestResultByAsset,
    copula_lag_independence_test,
    ellipsoid_lag_test,
    univariate_kolmogrov_smirnov_test,
)
from methods.cma import CopulaMarginalModel
from models.types import ProbVector
from utils.helpers import (
    get_assets_names,
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

        if deterministic_res is not None and deterministic_res > 0:
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
    *,
    trend_type: Literal["deterministic", "stochastic", "both"] = "both",
) -> DataFrame:
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

    return apply_detrend(data=data, detrend_needed=detrends_needed)

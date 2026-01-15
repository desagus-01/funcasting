# TODO: Initial pipeline
# 1. Test for white noise (done)
# 2. If false test/detrend
# 3. Test again for wh
# 4. If false test/deseason
# 5. test again for wh

from polars.dataframe.frame import DataFrame

from globals import LAGS
from maths.distributions import uniform_probs
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

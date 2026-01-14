# TODO: Initial pipeline
# 1. Test for white noise
# 2. If false test/detrend
# 3. Test again for wh
# 4. If false test/deseason
# 5. test again for wh
from itertools import chain

from polars.dataframe.frame import DataFrame

from globals import LAGS
from maths.distributions import uniform_probs
from maths.time_series.iid_tests import (
    copula_lag_independence_test,
    ellipsoid_lag_test,
    univariate_kolmogrov_smirnov_test,
)
from methods.cma import CopulaMarginalModel
from models.types import ProbVector


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
    """
    if prob is None:
        prob = uniform_probs(data.height)

    ellipsoid_test = ellipsoid_lag_test(data=data, prob=prob, lags=lags, assets=assets)
    copula_marginal_model = CopulaMarginalModel.from_data_and_prob(data=data, prob=prob)

    copula_lag_test_res = copula_lag_independence_test(
        copula=copula_marginal_model.copula,
        prob=copula_marginal_model.prob,
        lags=lags,
        assets=assets,
    )

    ks_test = univariate_kolmogrov_smirnov_test(data=data, assets=assets)
    ks_rejected = set(ks_test["rejected"])

    results = {}
    cols = data.select(assets).columns
    for asset in cols:
        results[asset] = not any(
            failed
            for failed in chain(
                (ellipsoid_test[asset]["rejected_lags"],),
                (copula_lag_test_res[asset]["rejected_lags"],),
                ((asset in ks_rejected),),
            )
        )

    return results

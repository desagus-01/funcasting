import logging

from polars.dataframe.frame import DataFrame

from globals import LAGS
from models.types import ProbVector
from scenarios.copula_marginal import CopulaMarginalModel
from time_series.tests.iid import (
    TestResultByAsset,
    copula_lag_independence_test,
    ellipsoid_lag_test,
    univariate_kolmogrov_smirnov_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _run_iid_simple(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    lags: int = LAGS["simple"],
) -> dict[str, TestResultByAsset]:
    """Run the cheaper IID screens."""
    ellipsoid_test = ellipsoid_lag_test(
        data=data,
        prob=prob,
        lags=lags,
        assets=assets,
    )
    ks_test = univariate_kolmogrov_smirnov_test(
        data=data,
        assets=assets,
    )
    return {
        "ellipsoid": ellipsoid_test,
        "ks": ks_test,
    }


def _run_iid_complex(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    lags: int = LAGS["complex"],
) -> dict[str, TestResultByAsset]:
    """Run the expensive copula IID screen."""
    copula_marginal_model = CopulaMarginalModel.from_data_and_prob(
        data=data.select(assets),
        prob=prob,
    )

    copula_lag_test_res = copula_lag_independence_test(
        copula=copula_marginal_model.copula,
        prob=copula_marginal_model.prob,
        lags=lags,
        assets=assets,
    )

    return {
        "copula": copula_lag_test_res,
    }


def check_white_noise(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    lags_simple: int = LAGS["simple"],
    lags_complex: int = LAGS["complex"],
) -> dict[str, bool]:
    """Return whether each asset passes the white-noise screen."""

    simple_tests = _run_iid_simple(
        data=data,
        prob=prob,
        assets=assets,
        lags=lags_simple,
    )

    simple_pass = {
        asset: not any(
            simple_tests[test_name][asset].rejected for test_name in ("ellipsoid", "ks")
        )
        for asset in assets
    }

    assets_for_copula = [asset for asset, passed in simple_pass.items() if passed]

    logger.info(
        "White-noise simple screen: passed_simple=%s, sent_to_complex=%s",
        assets_for_copula,
        assets_for_copula,
    )

    if not assets_for_copula:
        logger.info("White-noise final screen: passed_all=[]")
        return simple_pass

    complex_tests = _run_iid_complex(
        data=data,
        prob=prob,
        assets=assets_for_copula,
        lags=lags_complex,
    )

    copula_pass = {
        asset: not complex_tests["copula"][asset].rejected
        for asset in assets_for_copula
    }

    final_pass = simple_pass.copy()
    for asset in assets_for_copula:
        final_pass[asset] = final_pass[asset] and copula_pass[asset]

    logger.info(
        "White-noise final screen: passed_all=%s",
        [asset for asset, passed in final_pass.items() if passed],
    )

    return final_pass

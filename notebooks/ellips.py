from typing import TypedDict

import numpy as np
import scipy.stats as st
from numpy.typing import NDArray
from polars import DataFrame

from globals import DEFAULT_ROUNDING, SIGN_LVL
from models.types import ProbVector
from utils.distributions import uniform_probs
from utils.helpers import build_lag_df, compensate_prob, hyp_test_conc
from utils.template import get_template

info_all = get_template()
increms = info_all.increms_df.drop("date")
probs = uniform_probs(increms.height)


class MeanCovRes(TypedDict):
    assets: list[str]
    means: NDArray[np.floating]
    cov: NDArray[np.floating]


class AutoCorrRes(TypedDict):
    stat: float
    p_val: float
    sign_lvl: float
    null: str
    reject_null: bool
    desc: str


def sample_meancov(data: DataFrame, prob: ProbVector) -> MeanCovRes:
    assets = data.columns
    data_np = data.to_numpy()

    weighted_mean = prob @ data_np
    weighted_cov = ((data_np - weighted_mean).T * prob) @ (data_np - weighted_mean)

    return {"assets": assets, "means": weighted_mean, "cov": weighted_cov}


def autocorrelation_test(pair_df: DataFrame, prob: ProbVector) -> AutoCorrRes:
    mc = sample_meancov(pair_df, prob)
    cov = mc["cov"]

    var_t = cov[0, 0]
    var_lag = cov[1, 1]
    cov_t_lag = cov[0, 1]

    corr = float(cov_t_lag / np.sqrt(var_t * var_lag))
    Z = abs(corr) * np.sqrt(pair_df.height)
    p_val = 2 * (1 - st.norm.cdf(Z))
    hyp_conc = hyp_test_conc(float(p_val), null_hyp="Independence")
    return {
        "stat": round(float(corr), DEFAULT_ROUNDING),
        "p_val": round(float(p_val), DEFAULT_ROUNDING),
        "sign_lvl": SIGN_LVL,
        "null": "Independence",
        "reject_null": hyp_conc["reject_null"],
        "desc": hyp_conc["desc"],
    }


def get_autocorrelations(
    df_lagged: DataFrame,
    asset: str,
    lags: int,
    prob: ProbVector,
) -> dict[str, AutoCorrRes]:
    """
    Compute weighted autocorrelations for (asset, asset_lag_k) for k=1..lags.

    """

    corrs: dict[str, AutoCorrRes] = {}

    for lag in range(1, lags + 1):
        col_lag = f"{asset}_lag_{lag}"

        pair_df = df_lagged.select([asset, col_lag]).drop_nulls()
        lag_prob = compensate_prob(
            prob=prob, n_remove=df_lagged.height - pair_df.height
        )

        corrs[f"lag_{lag}"] = autocorrelation_test(pair_df=pair_df, prob=lag_prob)
    return corrs


def ellipsoid_test(
    data: DataFrame,
    lags: int,
    prob: ProbVector,
    assets: list[str] | None = None,
):
    if assets is None:
        sel_assets = [col for col in data.columns if col != "date"]
    else:
        sel_assets = data.select(assets).columns

    results: dict[str, dict[str, AutoCorrRes]] = {}

    # get auto corrs
    for asset in sel_assets:
        df_lagged = build_lag_df(data, asset, lags)
        results[asset] = get_autocorrelations(df_lagged, asset, lags, prob)

    return results


a = ellipsoid_test(increms, 2, probs, assets=["AAPL"])

print(a)

from typing import Callable, TypedDict

import numpy as np
import polars as pl
import scipy.stats as st
from numpy.typing import NDArray

from globals import DEFAULT_ROUNDING, ITERS, SIGN_LVL
from models.types import ProbVector
from utils.helpers import build_lag_df, compensate_prob, hyp_test_conc


class HypTestRes(TypedDict):
    stat: float
    p_val: float
    sign_lvl: float
    null: str
    reject_null: bool
    desc: str


class PerAssetLagResult(TypedDict):
    results: dict[str, HypTestRes]
    rejected_lags: list[str]


LagTestResult = dict[str, PerAssetLagResult]
StatFunc = Callable[[np.ndarray, ProbVector, np.random.Generator | None], float]
PairTest = Callable[[pl.DataFrame, ProbVector], HypTestRes]


def format_hyp_test_result(
    stat: float, p_val: float, null: str = "Independence"
) -> HypTestRes:
    p_val = float(p_val)
    stat = float(stat)
    hyp_conc = hyp_test_conc(p_val, null_hyp=null)
    return {
        "stat": round(stat, DEFAULT_ROUNDING),
        "p_val": round(p_val, DEFAULT_ROUNDING),
        "sign_lvl": SIGN_LVL,
        "null": null,
        "reject_null": hyp_conc["reject_null"],
        "desc": hyp_conc["desc"],
    }


def _select_assets(df: pl.DataFrame, assets: list[str] | None) -> list[str]:
    if assets is None:
        return [c for c in df.columns if c != "date"]
    return df.select(assets).columns


def run_lagged_tests(
    data: pl.DataFrame,
    prob: ProbVector,
    lags: int,
    assets: list[str] | None,
    test_fn: PairTest,
) -> LagTestResult:
    sel_assets = _select_assets(data, assets)
    results: LagTestResult = {}

    for asset in sel_assets:
        df_lagged = build_lag_df(data, asset, lags)

        per_lag: dict[str, HypTestRes] = {}
        for lag in range(1, lags + 1):
            col_lag = f"{asset}_lag_{lag}"

            pair_df = df_lagged.select([asset, col_lag]).drop_nulls()
            lag_prob = compensate_prob(
                prob=prob,
                n_remove=df_lagged.height - pair_df.height,
            )

            per_lag[f"lag_{lag}"] = test_fn(pair_df, lag_prob)

        rejected_lags = [k for k, res in per_lag.items() if res["reject_null"]]

        results[asset] = {
            "results": per_lag,
            "rejected_lags": rejected_lags,
        }

    return results


class MeanCovRes(TypedDict):
    assets: list[str]
    means: NDArray[np.floating]
    cov: NDArray[np.floating]


def sample_meancov(data: pl.DataFrame, prob: ProbVector) -> MeanCovRes:
    assets = data.columns
    data_np = data.to_numpy()

    weighted_mean = prob @ data_np
    weighted_cov = ((data_np - weighted_mean).T * prob) @ (data_np - weighted_mean)

    return {"assets": assets, "means": weighted_mean, "cov": weighted_cov}


def autocorrelation_pair_test(pair_df: pl.DataFrame, prob: ProbVector) -> HypTestRes:
    """
    Pairwise autocorrelation test on a 2-column DataFrame:
    column 0 = X_t, column 1 = X_{t-k}.
    """
    mc = sample_meancov(pair_df, prob)
    cov = mc["cov"]

    var_t = cov[0, 0]
    var_lag = cov[1, 1]
    cov_t_lag = cov[0, 1]

    corr = float(cov_t_lag / np.sqrt(var_t * var_lag))
    Z = abs(corr) * np.sqrt(pair_df.height)
    p_val = float(2 * (1 - st.norm.cdf(Z)))

    return format_hyp_test_result(stat=corr, p_val=p_val, null="Independence")


def ellipsoid_test(
    data: pl.DataFrame,
    lags: int,
    prob: ProbVector,
    assets: list[str] | None = None,
) -> LagTestResult:
    """
    Ellipsoidal (Gaussian) autocorrelation test per asset and lag.

    Returns:
        LagTestResult:
            {
                "asset_name": {
                    "per_lag": {
                        "lag_1": HypTestRes,
                        "lag_2": HypTestRes,
                        ...
                    },
                    "rejected_lags": ["lag_1", ...],
                },
                ...
            }
    """
    return run_lagged_tests(
        data=data,
        prob=prob,
        lags=lags,
        assets=assets,
        test_fn=autocorrelation_pair_test,
    )


def _copula_eval(pobs: np.ndarray, p: ProbVector, points: np.ndarray) -> np.ndarray:
    """Evaluate empirical copula at given points."""
    less_eq = pobs[:, None, :] <= points[None, :, :]
    inside = np.all(less_eq, axis=2)
    return p @ inside


def _sw_stat(pobs: np.ndarray, p: ProbVector, u_points: np.ndarray) -> float:
    """Compute the SW statistic for given uniform points."""
    est = _copula_eval(pobs, p, u_points)
    indep = u_points.prod(axis=1)
    return round(12.0 * np.abs(est - indep).mean(), DEFAULT_ROUNDING)


def sw_mc(
    pobs: np.ndarray,
    p: ProbVector,
    rng: np.random.Generator | None = None,
    mc_iters: int = ITERS["MC"],
) -> float:
    """Single Monte Carlo estimate of the SW statistic."""
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(0.0, 1.0, size=(mc_iters, pobs.shape[1]))
    return _sw_stat(pobs, p, u)

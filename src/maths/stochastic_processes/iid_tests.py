from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import numpy as np
import polars as pl
import scipy.stats as st
from numpy.typing import NDArray

from globals import ITERS, LAGS
from maths.stochastic_processes.base import HypTestRes, format_hyp_test_result
from models.types import ProbVector
from utils.helpers import build_lag_df, compensate_prob, select_assets, split_df_in_half

StatFunc = Callable[[np.ndarray, ProbVector, np.random.Generator | None], float]


class PerAssetLagResult(TypedDict):
    results: dict[str, HypTestRes]
    rejected_lags: list[str]


LagTestResult = dict[str, PerAssetLagResult]
PairTest = Callable[..., HypTestRes]


@dataclass(frozen=True, slots=True)
class MeanCovRes:
    assets: list[str]
    means: NDArray[np.floating]
    cov: NDArray[np.floating]


def run_lagged_tests(
    data: pl.DataFrame,
    prob: ProbVector,
    assets: list[str] | None,
    test_fn: PairTest,
    lags: int = LAGS["testing"],
    **test_kwargs: Any,
) -> LagTestResult:
    sel_assets = select_assets(data, assets)
    results: LagTestResult = {}
    for asset in sel_assets:
        df_lagged = build_lag_df(data=data, asset=asset, lags=lags)
        per_lag: dict[str, HypTestRes] = {}

        for lag in range(1, lags + 1):
            col_lag = f"{asset}_lag_{lag}"
            # need to drop nulls created from lags and compensate probs
            pair_df = df_lagged.select([asset, col_lag]).drop_nulls()
            lag_prob = compensate_prob(
                prob=prob,
                n_remove=df_lagged.height - pair_df.height,
            )

            res = test_fn(
                pair_df,
                lag_prob,
                assets=[asset, col_lag],  # must pass original and lag
                **test_kwargs,
            )
            per_lag[f"lag_{lag}"] = res

        rejected_lags = [k for k, res in per_lag.items() if res.reject_null]

        results[asset] = {
            "results": per_lag,
            "rejected_lags": rejected_lags,
        }

    return results


def _sample_meancov(
    data: pl.DataFrame,
    prob: ProbVector,
    assets: list[str],
) -> MeanCovRes:
    data_np = data.select(assets).to_numpy()

    weighted_mean = prob @ data_np
    weighted_cov = ((data_np - weighted_mean).T * prob) @ (data_np - weighted_mean)

    return MeanCovRes(
        assets=assets,
        means=weighted_mean,
        cov=weighted_cov,
    )


# TODO: Check on the correctness of p-val, currently assumes normal although we weight...
def autocorrelation_pair_test(
    pair_df: pl.DataFrame, prob: ProbVector, assets: list[str]
) -> HypTestRes:
    """
    Pairwise autocorrelation test on a 2-column DataFrame:
    column 0 = X_t, column 1 = X_{t-k}.
    """
    cov = _sample_meancov(pair_df, prob, assets).cov
    var_t = cov[0, 0]
    var_lag = cov[1, 1]
    cov_t_lag = cov[0, 1]

    corr = float(cov_t_lag / np.sqrt(var_t * var_lag))
    Z = abs(corr) * np.sqrt(pair_df.height)
    p_val = float(2 * (1 - st.norm.cdf(Z)))

    return format_hyp_test_result(stat=corr, p_val=p_val, null="Independence")


def ellipsoid_lag_test(
    data: pl.DataFrame,
    prob: ProbVector,
    lags: int = LAGS["testing"],
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
    return 12.0 * np.abs(est - indep).mean()


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


# TODO: Look at speeding this up with vectorization + multithreading
# INFO: Heavily inspired by the hyppo package in python
def independence_permutation_test(
    pair_df: pl.DataFrame,
    prob: ProbVector,
    assets: tuple[str, str],
    stat_fun: StatFunc = sw_mc,
    iter: int = ITERS["PERM_TEST"],
    rng: np.random.Generator | None = None,
) -> HypTestRes:
    if rng is None:
        rng = np.random.default_rng()

    assets_np = pair_df.select(assets).to_numpy()
    perm_asset = assets_np[:, 0].copy()
    stat = stat_fun(assets_np, prob, rng)

    null_dist = np.empty(iter, dtype=float)

    for i in range(iter):
        new_order = rng.permutation(assets_np.shape[0])
        new_p_asset = perm_asset[new_order]

        temp_df = pair_df.select(assets).with_columns(
            pl.lit(new_p_asset).alias(assets[0])
        )

        null_dist[i] = stat_fun(temp_df.to_numpy(), prob, rng)

    p_val = (1.0 + (null_dist >= stat).sum()) / (iter + 1.0)

    return format_hyp_test_result(stat=stat, p_val=p_val, null="Independence")


def copula_lag_independence_test(
    copula: pl.DataFrame,
    prob: ProbVector,
    lags: int = LAGS["testing"],
    assets: list[str] | None = None,
) -> LagTestResult:
    return run_lagged_tests(
        data=copula,
        prob=prob,
        lags=lags,
        assets=assets,
        test_fn=independence_permutation_test,
    )


def kolmogrov_smirnov_2stest(
    dist_1: NDArray[np.floating],
    dist_2: NDArray[np.floating],
) -> HypTestRes:
    res = st.kstest(
        dist_1,
        dist_2,
        alternative="two-sided",
    )

    return format_hyp_test_result(
        stat=res.statistic, p_val=res.pvalue, null="Identically Distributed"
    )


def univariate_kolmogrov_smirnov_test(
    data: pl.DataFrame, assets: list[str] | None = None
) -> dict[str, dict[str, HypTestRes] | list[str]]:
    sel_assets = select_assets(data, assets)

    ks_res: dict[str, HypTestRes] = {}
    for asset in sel_assets:
        data_asset = data.select(asset)
        split = split_df_in_half(data_asset)
        split_1, split_2 = (
            split.first_half.to_numpy().ravel(),
            split.second_half.to_numpy().ravel(),
        )
        ks_res[asset] = kolmogrov_smirnov_2stest(split_1, split_2)

    rejected = [k for k, res in ks_res.items() if res.reject_null]

    return {"results": ks_res, "rejected": rejected}

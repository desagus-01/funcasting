from typing import Callable, TypedDict

import numpy as np
import polars as pl

from globals import DEFAULT_ROUNDING, ITERS, SIGN_LVL
from models.types import ProbVector
from utils.helpers import build_lag_df, compensate_prob, hyp_test_conc

StatFunc = Callable[[np.ndarray, ProbVector, np.random.Generator | None], float]


class PermTestRes(TypedDict):
    stat: float
    p_val: float
    sign_lvl: float
    null: str
    reject_null: bool
    desc: str


class PerAssetLagResult(TypedDict):
    per_lag: dict[str, PermTestRes]
    rejected_lags: list[str]


LagIndTestResult = dict[str, PerAssetLagResult]


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


# INFO: BELOW IS HEAVILY INSPIRED BY THE HYPPO PACKAGE
# TODO: MUST MAKE THE BELOW MORE EFFICENT/FASTER
def independence_permutation_test(
    pobs: pl.DataFrame,
    p: ProbVector,
    stat_fun: StatFunc,
    assets: tuple[str, str],
    iter: int = ITERS["PERM_TEST"],
    rng: np.random.Generator | None = None,
) -> PermTestRes:
    if rng is None:
        rng = np.random.default_rng()

    assets_np = pobs.select(assets).to_numpy()
    perm_asset = assets_np[:, 0].copy()

    stat = stat_fun(assets_np, p, rng)

    null_dist = np.empty(iter, dtype=float)

    for i in range(iter):
        new_order = rng.permutation(assets_np.shape[0])
        new_p_asset = perm_asset[new_order]

        temp_df = pobs.select(assets).with_columns(pl.lit(new_p_asset).alias(assets[0]))

        null_dist[i] = stat_fun(temp_df.to_numpy(), p, rng)

    p_val = (1.0 + (null_dist >= stat).sum()) / (iter + 1.0)

    hyp_conc = hyp_test_conc(p_val, null_hyp="Independence")
    return {
        "stat": round(float(stat), DEFAULT_ROUNDING),
        "p_val": round(float(p_val), DEFAULT_ROUNDING),
        "sign_lvl": SIGN_LVL,
        "null": "Independence",
        "reject_null": hyp_conc["reject_null"],
        "desc": hyp_conc["desc"],
    }


def lag_independence_test(
    pobs: pl.DataFrame,
    prob: ProbVector,
    lags: int,
    assets: list[str] | None = None,
) -> LagIndTestResult:
    """
    Runs SW independence permutation test for chosen n lags for assets in the dataframe.

    If no assets are chosen, runs for every asset.
    """
    if assets is None:
        sel_assets = [col for col in pobs.columns if col != "date"]
    else:
        sel_assets = pobs.select(assets).columns

    sw_lag_res: LagIndTestResult = {}
    n_orig = pobs.height

    for asset in sel_assets:
        df_lagged = build_lag_df(pobs, asset, lags)

        per_lag: dict[str, PermTestRes] = {}

        for lag in range(1, lags + 1):
            col_lag = f"{asset}_lag_{lag}"
            # need to drop nulls created from lags
            pair_df = df_lagged.select([asset, col_lag]).drop_nulls()
            n_remove = n_orig - pair_df.height

            lag_prob = compensate_prob(prob=prob, n_remove=n_remove)

            res = independence_permutation_test(
                pair_df,
                lag_prob,
                sw_mc,
                (asset, col_lag),
            )
            per_lag[f"lag_{lag}"] = res
            print(f"""
            Lag: {lag}

            pair_df: {pair_df}

            to remove: {n_remove}
            """)

        rejected_lags = [
            lag_label for lag_label, res in per_lag.items() if res["reject_null"]
        ]

        sw_lag_res[asset] = {
            "per_lag": per_lag,
            "rejected_lags": rejected_lags,
        }

    return sw_lag_res

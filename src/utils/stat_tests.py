from typing import Callable, TypedDict

import numpy as np
import polars as pl

from globals import DEFAULT_ROUNDING, ITERS
from models.types import ProbVector
from utils.helpers import hyp_test_conc

StatFunc = Callable[[np.ndarray, ProbVector, np.random.Generator | None], float]


class PermTestRes(TypedDict):
    stat: float
    p_val: float
    h_test: str


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
def ind_perm_test(
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

    return {
        "stat": round(float(stat), DEFAULT_ROUNDING),
        "p_val": round(float(p_val), DEFAULT_ROUNDING),
        "h_test": hyp_test_conc(p_val, null_hyp="Independence"),
    }

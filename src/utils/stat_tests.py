from typing import Callable, TypedDict

import numpy as np
import polars as pl

from globals import DEFAULT_ROUNDING
from models.types import ProbVector

StatFunc = Callable[[np.ndarray, ProbVector | None], float]


class PermTestRes(TypedDict):
    stat: float
    p_val: float


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
    iters: int = 50_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Single Monte Carlo estimate of the SW statistic."""
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(0.0, 1.0, size=(iters, pobs.shape[1]))
    return _sw_stat(pobs, p, u)


def perm_test(
    pobs: pl.DataFrame,
    p: ProbVector,
    stat_fun: StatFunc,
    assets: tuple[str, str],
    iter: int,
    mc_iters: int = 50_000,
    rng: np.random.Generator | None = None,
) -> PermTestRes:
    if rng is None:
        rng = np.random.default_rng()

    assets_np = pobs.select(assets).to_numpy()
    n, d = assets_np.shape

    # Precompute uniforms ONCE
    u = rng.uniform(0.0, 1.0, size=(mc_iters, d))

    # Observed statistic
    stat = _sw_stat(assets_np, p, u)

    null_dist = np.empty(iter, dtype=float)

    # Prepare arrays for in-loop reuse
    perm_col = assets_np[:, 0].copy()
    fixed_col = assets_np[:, 1].copy()
    permuted = np.empty_like(assets_np)

    for i in range(iter):
        print(f"{i}")
        rng.shuffle(perm_col)
        permuted[:, 0] = perm_col
        permuted[:, 1] = fixed_col

        null_dist[i] = _sw_stat(permuted, p, u)

    p_val = (1.0 + (null_dist >= stat).sum()) / (iter + 1.0)

    return {
        "stat": round(float(stat), DEFAULT_ROUNDING),
        "p_val": round(float(p_val), DEFAULT_ROUNDING),
    }

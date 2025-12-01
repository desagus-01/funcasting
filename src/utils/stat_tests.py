from typing import Any, TypedDict

import numpy as np
from numpy._typing import NDArray

from models.types import ProbVector


class SWRes(TypedDict):
    iter_res: NDArray[np.floating[Any]]
    iter_avg: NDArray[np.floating[Any]]


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


def _sw_mc_single(pobs: np.ndarray, p: ProbVector, iters: int = 10_000) -> float:
    """Single Monte Carlo estimate of the SW statistic."""
    rng = np.random.default_rng()
    u = rng.uniform(0.0, 1.0, size=(iters, pobs.shape[1]))
    return _sw_stat(pobs, p, u)


def sw_mc_summary(pobs: np.ndarray, p: ProbVector, iters: int = 50) -> SWRes:
    """Run multiple SW Monte Carlo estimates and return results + average."""
    vals = np.empty(iters)
    for i in range(iters):
        vals[i] = _sw_mc_single(pobs, p)
    return {"iter_res": np.round(vals, 3), "iter_avg": np.round(vals.mean(), 3)}

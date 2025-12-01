import numpy as np

from models.types import ProbVector


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


def sw_mc(pobs: np.ndarray, p: ProbVector, iters: int = 50_000) -> float:
    """Single Monte Carlo estimate of the SW statistic."""
    rng = np.random.default_rng()
    u = rng.uniform(0.0, 1.0, size=(iters, pobs.shape[1]))
    return _sw_stat(pobs, p, u)

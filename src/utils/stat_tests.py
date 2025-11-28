import numpy as np

from models.types import ProbVector


def _eval_cop(pobs: np.ndarray, p: ProbVector, points: np.ndarray) -> np.ndarray:
    less_eq_coord = pobs[:, None, :] <= points[None, :, :]

    # For each (obs, point), check if obs is inside the lower orthant of that point
    inside_lower_orthant = np.all(less_eq_coord, axis=2)

    return p @ inside_lower_orthant


def _sw_eq(pobs: np.ndarray, p: ProbVector, uniform_draws: np.ndarray) -> float:
    est_cop = _eval_cop(pobs, p, uniform_draws)
    ind_cop = uniform_draws.prod(axis=1)
    res = np.abs(est_cop - ind_cop)

    return 12.0 * res.mean()


def sw_mc_vec(pobs: np.ndarray, p: ProbVector, iter: int = 10_000) -> float:
    rng = np.random.default_rng()

    # Draw all uniform points at once: (iter, d)
    uni_draws = rng.uniform(0.0, 1.0, size=(iter, pobs.shape[1]))
    return _sw_eq(pobs, p, uni_draws)


def sw_mc_u(pobs: np.ndarray, p: ProbVector, iter: int = 50) -> dict[np.ndarray, float]:
    sc_u_vec = np.empty(iter)
    for i in range(iter):
        sc_u_vec[i] = sw_mc_vec(pobs, p)

    return {"mean_res": sc_u_vec, "mean": sc_u_vec.mean()}

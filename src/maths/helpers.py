import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from data_types.vectors import ProbVector


def kernel_smoothing(
    data_array: NDArray[np.floating],
    half_life: float,
    kernel_type: float,
    reference: float | None,
    time_based: bool = False,
) -> NDArray[np.float64]:
    """
    General function for kernel smoothing, allows for exponential, gaussian among other through the kernel_type parameter.
    """
    if kernel_type < 0:
        raise ValueError("Kernel type must be positive integer")

    if time_based:
        data_n = len(data_array)
        data_array = np.arange(data_n)
        dist_to_ref = data_n - 1 - data_array  # uses last data point as ref
    else:
        dist_to_ref = np.abs(reference - data_array)

    bandwidth: float = half_life / (np.log(2.0) ** (1.0 / kernel_type))
    return np.exp(-((dist_to_ref / bandwidth) ** kernel_type))


def kl_div(prior: ProbVector, posterior: ProbVector) -> float:
    """
    Calculates the KL divergence between two probability vectors.
    """
    return np.sum(posterior * np.log(posterior / prior))


def simple_entropy_pooling(
    prior: ProbVector,
    Aeq: NDArray[np.floating],
    beq: NDArray[np.floating],
    **solver_kwargs,
) -> ProbVector:
    posterior = cp.Variable(prior.shape[0])

    constraints = [
        cp.sum(posterior) == 1,
        Aeq @ posterior == beq,
    ]

    obj = cp.Minimize(cp.sum(cp.kl_div(posterior, prior)))

    prob = cp.Problem(obj, constraints)
    prob.solve(solver="SCS", **solver_kwargs)

    return prob, prob.status, posterior.value

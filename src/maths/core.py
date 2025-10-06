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
    elif not time_based and reference is not None:
        dist_to_ref = np.abs(reference - data_array)

    bandwidth: float = half_life / (np.log(2.0) ** (1.0 / kernel_type))
    return np.exp(-((dist_to_ref / bandwidth) ** kernel_type))


def simple_entropy_pooling(
    prior: ProbVector,
    scenario_matrix: NDArray[np.floating],
    view_targets: NDArray[np.floating],
    *,
    solver: str = "SCS",
    **solver_kwargs,
):
    posterior = cp.Variable(prior.shape[0])

    constraints = [
        scenario_matrix @ posterior == view_targets,
    ]

    obj = cp.Minimize(cp.sum(cp.kl_div(posterior, prior)))

    prob = cp.Problem(obj, constraints)
    prob.solve(solver, **solver_kwargs)

    return posterior.value

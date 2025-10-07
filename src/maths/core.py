import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint as CvxConstraint
from numpy.typing import NDArray

from data_types.vectors import ConstraintSigns, ConstraintType, ProbVector, View


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


def build_constraints(
    views: View,
    posterior: cp.Variable,
) -> list[CvxConstraint]:
    match (views.const_type, views.sign_type):
        case (ConstraintType.equality, ConstraintSigns.equal):
            constraint = views.data @ posterior == views.views_targets

        case (ConstraintType.inequality, ConstraintSigns.equal_greater):
            constraint = views.data @ posterior >= views.views_targets

        case (ConstraintType.inequality, ConstraintSigns.equal_less):
            constraint = views.data @ posterior <= views.views_targets

        case _:
            raise ValueError(
                f"Invalid combination: const_type={views.const_type}, sign_type={views.sign_type}"
            )

    return [constraint]


def simple_entropy_pooling(
    prior: ProbVector,
    views: View,
    solver: str = "SCS",
    **solver_kwargs: str,
) -> NDArray[np.floating]:
    posterior = cp.Variable(prior.shape[0], nonneg=True)  # ensures probs > 0
    constraints = build_constraints(views=views, posterior=posterior)
    obj = cp.Minimize(cp.sum(cp.kl_div(posterior, prior)))
    prob = cp.Problem(obj, constraints)
    _ = prob.solve(solver, **solver_kwargs)

    if posterior.value is None:
        raise RuntimeError("Optimization failed or returned no solution!")

    return np.asarray(posterior.value, dtype=float)

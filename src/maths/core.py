import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint as CvxConstraint
from numpy.typing import NDArray

from data_types.vectors import ConstraintSigns, ConstraintType, ProbVector, View


# TODO: Change to it works on multi arrays
def kernel_smoothing(
    data_array: NDArray[np.floating],
    half_life: float,
    kernel_type: int,
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


def ens(prob_vector: ProbVector) -> int:
    """
    Effective number of scenarios (ENS) as measures by the exponential of the entropy
    """
    max = prob_vector.shape[0]

    ens = np.exp(-(np.sum(prob_vector * np.log(prob_vector))))

    if (ens < 1) or (ens > max):
        raise RuntimeError(
            "ENS is larger than total number of scenarios or smaller than 1."
        )

    return int(ens) + 1  # ensure rounding is ok


# TODO: Finish writing this
def effective_rank(views_target):
    pass


def assign_constraint_equation(views: View, posterior: cp.Variable) -> CvxConstraint:
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
    return constraint


def build_constraints(
    views: list[View],
    posterior: cp.Variable,
) -> list[CvxConstraint]:
    base: list[CvxConstraint] = [cp.sum(posterior) == 1]  # ensures we get probabilities
    constraints: list[CvxConstraint] = []
    for view in views:
        constraints.append(assign_constraint_equation(view, posterior))

    return constraints + base


# TODO: Finish writing this
def get_ep_diags(
    views: list[View], constraints: list[CvxConstraint], posterior_probs: ProbVector
) -> list[dict]:
    info: list[dict] = []
    for view, constraint in zip(views, constraints):
        dual_raw = constraint.dual_value

        if view.const_type == ConstraintType.equality:
            slack = view.data @ posterior_probs - view.views_targets
            active = abs(slack) <= 1e-5
            sensitivity = dual_raw

        info.append(
            {
                "type": view.const_type,
                "sign": view.const_type,
                "active": active,
                "sensitivity": sensitivity,
            }
        )

    return info


def simple_entropy_pooling(
    prior: ProbVector,
    views: list[View],
    solver: str = "SCS",
    include_diags: bool = False,
    **solver_kwargs: str,
) -> ProbVector:
    # ) -> NDArray[np.floating]:
    posterior = cp.Variable(prior.shape[0])
    constraints = build_constraints(views=views, posterior=posterior)
    obj = cp.Minimize(cp.sum(cp.kl_div(posterior, prior)))
    prob = cp.Problem(obj, constraints)
    _ = prob.solve(solver, **solver_kwargs)

    if posterior.value is None:
        raise RuntimeError("Optimization failed or returned no solution!")

    if include_diags:
        print(get_ep_diags(views, constraints, posterior.value))

    return np.asarray(posterior.value, dtype=float)

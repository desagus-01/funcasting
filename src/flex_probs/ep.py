import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint as CvxConstraint
from pydantic import validate_call

from globals import model_cfg
from helpers import select_operator, weighted_moments
from models.prob import ProbVector
from models.views import View


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


def _assign_constraint_equation(views: View, posterior: cp.Variable, prior: ProbVector):
    """
    Assigns appropriate linear constraint equation based on view type
    """
    operator_used = select_operator(views)

    match views.type:
        case "quantile":
            constraint = operator_used(views.data @ posterior, views.views_target)

        case "sorting":
            constraint = operator_used(
                views.data[0] @ posterior, views.data[1] @ posterior
            )

        case "std":
            mu_ref = views.data @ prior if views.mean_ref is None else views.mean_ref
            constraint = operator_used(
                views.data**2 @ posterior, views.views_target**2 + mu_ref**2
            )

        case "mean":
            constraint = operator_used(views.data @ posterior, views.views_target)

        case "corr":
            mu_ref, std_ref = weighted_moments(
                views.data, prior
            )  # Need to anchor both mean and std on prior

            constraint = operator_used(
                (views.data[0] * views.data[1]) @ posterior,
                views.views_target * std_ref[0] * std_ref[1] + mu_ref[0] * mu_ref[1],
            )

        case _:
            raise ValueError(f"Unsupported constraint type: {views.type}")

    return constraint


def _build_constraints(
    views: list[View],
    posterior: cp.Variable,
    prior: ProbVector,
) -> list[CvxConstraint]:
    """
    Compiles constraint equations to list of constraints used in EP.
    """
    base: list[CvxConstraint] = [cp.sum(posterior) == 1]  # ensures we get probabilities
    constraints: list[CvxConstraint] = []
    for view in views:
        constraints.append(_assign_constraint_equation(view, posterior, prior))

    return constraints + base


# TODO: Fix info dict
def get_constraints_diags(
    views: list[View], constraints: list[CvxConstraint], posterior_probs: ProbVector
) -> list[dict[str, int | bool | str]]:
    """
    Gives some diagnostics regarding the constraints added to the entropy pooling problem:

    active: True means that the constraint is relevant to shaping the posterior probability.

    sensitivity: The langrange multiplier, tell us the magnitude of the constraint has on the optimization (ie how much it is relevant).

    """
    info: list[dict] = []

    for view, constraint in zip(views, constraints):
        sensitivity = (
            constraint.dual_value
            if view.sign_type != "equal_greater"
            else -constraint.dual_value
        )
        if view.type == "sorting":
            active = bool(
                view.data[0] @ posterior_probs >= view.data[1] @ posterior_probs
            )
        elif view.type == "corr":
            active = bool(
                (
                    view.data[0] * view.data[1] @ posterior_probs - view.views_target
                    <= 1e-5
                )
            )

        else:
            active = bool(abs(view.data @ posterior_probs - view.views_target <= 1e-5))

        info.append(
            {
                "risk_driver": view.risk_driver,
                "sign": view.sign_type,
                "constraint_value": view.views_target,
                "active": active,
                "sensitivity": sensitivity,
            }
        )

    return info


def entropy_pooling(
    prior: ProbVector,
    views: list[View],
    solver: str = "SCS",
    include_diags: bool = False,
    **solver_kwargs: str,
) -> ProbVector:
    """
    Applied Entropy Pooling (EP) to current probability vector (prior)
    based on list of views.

    Outputs a new ProbVector.
    """
    posterior = cp.Variable(prior.shape[0])
    constraints = _build_constraints(views=views, posterior=posterior, prior=prior)
    obj = cp.Minimize(cp.sum(cp.kl_div(posterior, prior)))
    prob = cp.Problem(obj, constraints)
    _ = prob.solve(solver, **solver_kwargs)

    if posterior.value is None:
        raise RuntimeError("Optimization failed or returned no solution!")

    if include_diags:
        print(get_constraints_diags(views, constraints, posterior.value))

    return np.asarray(posterior.value, dtype=float)


@validate_call(config=model_cfg, validate_return=True)
def entropy_pooling_probs(
    prior: ProbVector,
    views: list[View],
    confidence: float = 1.0,
    include_diags: bool = False,
) -> ProbVector:
    """
    Implements entropy pooling optimization using KL divergence as the objective function, then adds confidence value linearly to the resulting posterior.
    """
    entropy_pooling_res = entropy_pooling(prior, views, include_diags=include_diags)

    return confidence * entropy_pooling_res + (1 - confidence) * prior

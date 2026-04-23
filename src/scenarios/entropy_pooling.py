import logging

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint as CvxConstraint
from numpy._typing import NDArray
from pydantic import validate_call

from globals import model_cfg
from scenarios.types import ProbVector, View
from utils.helpers import select_operator, weighted_moments

logger = logging.getLogger(__name__)


def ens(prob_vector: ProbVector) -> int:
    """
    Effective number of scenarios (ENS) computed from a probability vector.

    ENS is defined as the exponential of the Shannon entropy of the
    discrete distribution represented by ``prob_vector``. The function
    validates that the result is in a sensible range and returns an integer
    approximation.

    Parameters
    ----------
    prob_vector : ProbVector
        Array-like probability vector summing to 1.

    Returns
    -------
    int
        Effective number of scenarios (rounded up).

    Raises
    ------
    RuntimeError
        If the computed ENS is outside the range [1, n_scenarios].
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
    Map a View into a CVXPY linear constraint to be used in entropy pooling.

    The function inspects the view type and constructs the appropriate
    linear equality/inequality constraint for the optimization problem.

    Parameters
    ----------
    views : View
        View object describing the constraint (type, data, sign and target).
    posterior : cp.Variable
        CVXPY variable representing posterior probabilities.
    prior : ProbVector
        Prior probability vector used for reference values (e.g. means/stds).

    Returns
    -------
    cvxpy.Constraint
        A cvxpy constraint object implementing the view.
    """
    operator_used = select_operator(views)

    match views.type:
        case "quantile":
            if views.views_target is None:
                raise ValueError("Your views_target cannot be None")

            constraint = operator_used(views.data @ posterior, views.views_target)

        case "sorting":
            constraint = operator_used(
                views.data[0] @ posterior, views.data[1] @ posterior
            )

        case "std":
            if views.views_target is None:
                raise ValueError("Your views_target cannot be None")
            mu_ref = views.data @ prior if views.mean_ref is None else views.mean_ref
            constraint = operator_used(
                views.data**2 @ posterior, views.views_target**2 + mu_ref**2
            )

        case "mean":
            if views.views_target is None:
                raise ValueError("Your views_target cannot be None")
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
    Compile a list of CVXPY constraints for the entropy pooling problem.

    The function always includes the simplex constraints (sum(posterior) == 1
    and posterior >= 0) and appends linear constraints derived from each
    View object via :func:`_assign_constraint_equation`.

    Parameters
    ----------
    views : list[View]
        List of views to incorporate.
    posterior : cp.Variable
        Posterior probability variable for the optimization.
    prior : ProbVector
        Prior probability vector used for reference values in some views.

    Returns
    -------
    list[cvxpy.Constraint]
        Constraints to pass into the cvxpy Problem.
    """
    base: list[CvxConstraint] = [cp.sum(posterior) == 1, posterior >= 0]
    constraints: list[CvxConstraint] = []
    for view in views:
        constraints.append(_assign_constraint_equation(view, posterior, prior))

    return constraints + base


# TODO: Fix info dict
def get_constraints_diags(
    views: list[View], constraints: list[CvxConstraint], posterior_probs: ProbVector
) -> list[dict[str, int | bool | str]]:
    """
    Produce diagnostic information for constraints after solving EP.

    The diagnostics list includes for each view whether the constraint is
    active (binding) and the dual multiplier (sensitivity) indicating its
    influence on the posterior. The logic for 'active' checks varies by view type.

    Parameters
    ----------
    views : list[View]
        Views that were included in the entropy pooling problem.
    constraints : list[cvxpy.Constraint]
        Corresponding cvxpy constraints as returned by :func:`_build_constraints`.
    posterior_probs : ProbVector
        Posterior probability vector obtained from the optimization.

    Returns
    -------
    list[dict]
        Diagnostics dictionaries containing keys like 'risk_driver', 'sign',
        'constraint_value', 'active', and 'sensitivity'.
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


# TODO: Consider whether this is the best way (maybe instead of clipping give a v small value)
def clip_normalise_probs(prob: NDArray[np.floating]) -> ProbVector:
    """
    Clip tiny negative solver tolerances and renormalize a probability vector.

    Parameters
    ----------
    prob : NDArray[np.floating]
        Raw posterior returned by the solver, may contain small negative values
        due to numerical tolerances.

    Returns
    -------
    ProbVector
        Clipped and re-normalized posterior probabilities.

    Raises
    ------
    RuntimeError
        If clipping results in a zero-sum vector (posterior collapsed).
    """
    prob[prob < 0] = 0.0

    s = prob.sum()
    if s <= 0:
        raise RuntimeError("posterior collapsed after clipping (unexpected).")

    prob /= s
    return np.asarray(prob, dtype=float)


def entropy_pooling(
    prior: ProbVector,
    views: list[View],
    solver: str = "SCS",
    include_diags: bool = False,
    **solver_kwargs: str,
) -> ProbVector:
    """
    Apply Entropy Pooling (EP) to update a prior probability vector given views.

    The function sets up a convex optimisation minimizing KL divergence
    KL(posterior || prior) subject to linear constraints derived from
    ``views``. It returns a posterior probability vector if the problem
    solves to optimality.

    Parameters
    ----------
    prior : ProbVector
        Prior probability vector to be updated.
    views : list[View]
        List of views encoding linear constraints on expectations, quantiles,
        correlations or orderings.
    solver : str, optional
        CVXPY solver name to use (default: 'SCS').
    include_diags : bool, optional
        If True prints constraint diagnostics after solving.
    **solver_kwargs : dict
        Additional solver keyword arguments forwarded to cvxpy.

    Returns
    -------
    ProbVector
        Posterior probability vector satisfying the provided views.

    Raises
    ------
    RuntimeError
        If the CVXPY problem fails to solve to optimality or returns an
        infeasible/invalid posterior.
    """
    posterior = cp.Variable(prior.shape[0])
    constraints = _build_constraints(views=views, posterior=posterior, prior=prior)
    obj = cp.Minimize(cp.sum(cp.kl_div(posterior, prior)))
    prob = cp.Problem(obj, constraints)
    _ = prob.solve(solver=solver, **solver_kwargs)
    posterior_res = posterior.value

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"EP did not solve optimally. status={prob.status}")
    if posterior_res is None:
        raise RuntimeError("Optimization failed or returned no solution!")
    if posterior_res.min() < -1e-6:
        raise RuntimeError(
            f"Materially negative posterior probability: min={posterior_res.min()}"
        )
    elif posterior_res.min() < 0:
        posterior_res = clip_normalise_probs(posterior_res)
    if include_diags:
        logger.debug(
            "EP constraint diagnostics: %s",
            get_constraints_diags(views, constraints, posterior_res),
        )

    return posterior_res


@validate_call(config=model_cfg, validate_return=True)
def entropy_pooling_probs(
    prior: ProbVector,
    views: list[View],
    confidence: float = 1.0,
    include_diags: bool = False,
) -> ProbVector:
    """
    Run entropy pooling and blend the posterior with the prior by confidence.

    This wrapper computes the posterior using :func:`entropy_pooling` and
    then returns a convex combination ``confidence * posterior + (1-confidence) * prior``.

    Parameters
    ----------
    prior : ProbVector
        Prior probability vector.
    views : list[View]
        Views to impose through entropy pooling.
    confidence : float, optional
        Weight to give to the posterior in the convex combination (default: 1.0).
    include_diags : bool, optional
        If True print constraint diagnostics.

    Returns
    -------
    ProbVector
        Final probability vector after blending.
    """
    entropy_pooling_res = entropy_pooling(prior, views, include_diags=include_diags)

    return confidence * entropy_pooling_res + (1 - confidence) * prior

from typing import cast

import cvxpy as cp
import numpy as np
from cvxpy import Constraint, Expression
from numpy.typing import NDArray
from portfolio.policy.constraints import (
    DEFAULT_CONSTRAINTS,
    PortfolioConstraint,
)
from portfolio.policy.moments import HorizonMoments


def _structural_constraints(
    trades: Expression,
    weights: Expression,
    previous_weights: NDArray[np.floating] | Expression,
) -> list[Constraint]:
    """Self-financing and horizon-linking constraints."""
    return cast(
        list[Constraint],
        [
            cp.sum(trades) == 0,
            weights == previous_weights + trades,
        ],
    )


def mpo_mean_cov(
    horizon_moments: HorizonMoments,
    horizons: int,
    n_assets: int,
    risk_aversion: float,
    current_weights: NDArray[np.floating],
    transaction_cost: float = 0.005,
    constraints: list[PortfolioConstraint] = DEFAULT_CONSTRAINTS,
) -> dict:
    optimizer_trades = cp.Variable((horizons, n_assets))
    post_trade_weights = cp.Variable((horizons, n_assets))

    previous_weights: NDArray[np.floating] | Expression = current_weights
    objective_terms = []
    cvxpy_constraints: list[Constraint] = []

    for h in range(horizons):
        trades_h = optimizer_trades[h, :]
        weights_h = post_trade_weights[h, :]

        cvxpy_constraints += _structural_constraints(
            trades_h, weights_h, previous_weights
        )
        for c in constraints:
            cvxpy_constraints += c.compile_to_cvxpy(weights_h, trades_h)

        mean = horizon_moments.mean[h]
        cov = horizon_moments.covariances[h]
        cov = 0.5 * (cov + cov.T)

        objective_terms.append(mean @ weights_h)
        objective_terms.append(
            -risk_aversion * cp.quad_form(weights_h, cov, assume_PSD=True)
        )
        objective_terms.append(-transaction_cost * cp.norm1(trades_h))

        previous_weights = weights_h

    problem = cp.Problem(
        cp.Maximize(cp.sum(objective_terms)), constraints=cvxpy_constraints
    )
    problem.solve()

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"Optimization failed: {problem.status}")

    def clean(w: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        w = np.asarray(w, dtype=float).copy()
        w[np.abs(w) < tol] = 0.0
        w = np.maximum(w, 0.0)
        return w / w.sum()

    def readable(w: np.ndarray) -> dict:
        return dict(zip(horizon_moments.assets, np.round(w, 6)))

    planned_trades = np.asarray(optimizer_trades.value, dtype=float)
    planned_weights = np.asarray(post_trade_weights.value, dtype=float)
    target_weights = clean(planned_weights[0])
    first_trade = target_weights - current_weights

    return {
        "status": problem.status,
        "objective_value": problem.value,
        "target_weights": target_weights,
        "target_weights_by_asset": readable(target_weights),
        "first_trade_weights": first_trade,
        "first_trade_by_asset": readable(first_trade),
        "planned_trades": planned_trades,
        "planned_weights": planned_weights,
    }

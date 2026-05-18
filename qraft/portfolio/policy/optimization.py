# Multi period optimization inspired by cvxportfolio
# needs obj
# needs constraints
# needs cost f
import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from portfolio.policy.moments import HorizonMoments


# to simplify keep constraints AND objectives constant for now and just in cvxpy format
# TODO: take a look at cash & cash returns...
def mpo_mean_cov(
    horizon_moments: HorizonMoments,
    horizons: int,
    n_assets: int,
    risk_aversion: float,
    current_weights: NDArray[np.floating],
    transaction_cost: float = 0.2,
):
    optimizer_trades = cp.Variable((horizons, n_assets))
    post_trade_weights = cp.Variable((horizons, n_assets))

    previous_weights = current_weights

    constraints = []
    objective_terms = []

    for horizon in range(horizons):
        optimized_trades_at_horizon = optimizer_trades[horizon, :]
        post_trade_weights_at_horizon = post_trade_weights[horizon, :]

        # constraints
        constraints += [cp.sum(optimized_trades_at_horizon) == 0]  # self financing
        constraints += [
            post_trade_weights_at_horizon
            == previous_weights + optimized_trades_at_horizon
        ]  # horizon linking
        constraints += [post_trade_weights_at_horizon >= 0]  # long only
        constraints += [
            cp.sum(post_trade_weights_at_horizon) == 1
        ]  # explicit budget constraint

        # mean variance
        mean = horizon_moments.mean[horizon]
        cov = horizon_moments.covariances[horizon]
        print(cov)
        cov = 0.5 * (cov + cov.T)

        objective_terms.append(mean @ post_trade_weights_at_horizon)
        objective_terms.append(
            -risk_aversion
            * cp.quad_form(post_trade_weights_at_horizon, cov, assume_PSD=True)
        )

        # key MPO-enabling term
        objective_terms.append(
            -transaction_cost * cp.norm1(optimized_trades_at_horizon)
        )

        previous_weights = post_trade_weights_at_horizon

    problem = cp.Problem(cp.Maximize(cp.sum(objective_terms)), constraints=constraints)
    print(problem)
    problem.solve()

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"Optimization failed: {problem.status}")

    first_trade = optimizer_trades.value[0]
    target_weights = current_weights + first_trade

    return {
        "status": problem.status,
        "objective_value": problem.value,
        "first_trade_weights": first_trade,
        "target_weights": target_weights,
        "planned_trades": np.asarray(optimizer_trades.value, dtype=float),
        "planned_weights": np.asarray(post_trade_weights.value, dtype=float),
    }

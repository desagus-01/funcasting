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
    current_weights: NDArray[np.floating],
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

        # mean variance
        # expected return(mean)
        mean_at_horizon = horizon_moments.mean
        objective_terms.append(mean_at_horizon @ post_trade_weights_at_horizon)
        # risk
        cov_at_horizon = horizon_moments.covariances
        print(cov_at_horizon)
        print(post_trade_weights_at_horizon)
        objective_terms.append(
            -1
            * cp.quad_form(
                post_trade_weights_at_horizon, cov_at_horizon, assume_PSD=True
            )
        )

        # problem
        problem = cp.Problem(
            cp.Maximize(cp.sum(objective_terms)), constraints=constraints
        )

        first_trade = np.asarray(optimizer_trades[0])
        target_weights = current_weights + first_trade

        return {
            "status": problem.status,
            "objective_value": problem.value,
            "first_trade_weights": first_trade,
            "target_weights": target_weights,
            "planned_trades": np.asarray(optimizer_trades.value, dtype=float),
            "planned_weights": np.asarray(post_trade_weights.value, dtype=float),
        }

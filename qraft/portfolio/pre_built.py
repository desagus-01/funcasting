import numpy as np
from numpy.typing import NDArray
from portfolio.policy.constraints import PortfolioConstraint
from portfolio.policy.moments import HorizonMoments
from portfolio.policy.objectives.specs import (
    CovarianceRisk,
    CVaRRisk,
    ExpectedReturn,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)
from portfolio.policy.optimization import MultiPeriodOptimizer


def classic_mpo(
    horizons: int,
    n_assets: int,
    risk_aversion: float,
    transaction_cost: float,
    moments: HorizonMoments,
    current_weights: NDArray[np.floating],
    constraints: list[PortfolioConstraint] | None = None,
    **solver_options,
):
    objective = ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(risk_aversion, CovarianceRisk()),
            WeightedTerm(
                transaction_cost,
                TransactionCost(cost=1.0, market_impact=0.0, exponent=1.0),
            ),
        )
    )
    return MultiPeriodOptimizer(
        objective=objective,
        horizons=horizons,
        n_assets=n_assets,
        constraints=constraints,
        n_scenarios=moments.scenario_returns.shape[0],
    ).solve(moments, current_weights, inputs=None, **solver_options)


def cvar_mpo(
    horizons: int,
    n_assets: int,
    cvar_aversion: float,
    transaction_cost: float,
    moments: HorizonMoments,
    current_weights: NDArray[np.floating],
    alpha: float = 0.05,
    constraints: list[PortfolioConstraint] | None = None,
    **solver_options,
):
    """
    Mean-CVaR multi-period optimisation.

    Maximises   E[return] − cvar_aversion · CVaR_α(loss) − transaction_cost · ||z||_1

    Parameters
    ----------
    cvar_aversion:
        Coefficient on the CVaR penalty. Plays the same role as
        ``risk_aversion`` in ``classic_mpo`` but penalises tail loss rather
        than variance.
    alpha:
        Tail probability for CVaR (default 0.05 → 5% CVaR).
    n_scenarios:
        Number of Monte-Carlo paths in ``moments.scenario_returns``.
    """
    objective = ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(cvar_aversion, CVaRRisk(alpha=alpha)),
            WeightedTerm(
                transaction_cost,
                TransactionCost(cost=1.0, market_impact=0.0, exponent=1.0),
            ),
        )
    )
    return MultiPeriodOptimizer(
        objective=objective,
        horizons=horizons,
        n_assets=n_assets,
        constraints=constraints,
        n_scenarios=moments.scenario_returns.shape[0],
    ).solve(moments, current_weights, inputs=None, **solver_options)

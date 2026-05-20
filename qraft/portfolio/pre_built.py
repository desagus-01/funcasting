from portfolio.policy.objectives.specs import (
    CovarianceRisk,
    ExpectedReturn,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)
from portfolio.policy.optimization import MultiPeriodOptimizer


def classic_mpo(
    horizons,
    n_assets,
    risk_aversion,
    transaction_cost,
    moments,
    current_weights,
    constraints=None,
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
    ).solve(moments, current_weights, inputs=None, **solver_options)

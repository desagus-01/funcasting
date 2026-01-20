from models.scenarios import ScenarioProb
from utils.template import get_template
from utils.visuals import plot_hist_compare, scatter_compare

# %% Generate data

data = get_template().asset_info.risk_drivers

# %% Base scenario distribution (uniform prior)

scenarios = ScenarioProb.default_inst(scenarios=data)

# %% Case 1 - CMA "distribution stress test"
# Idea:
# - Make AAPL + MSFT marginals fat-tailed ("t") to reflect jump/crash risk
# - Use a t-copula to add tail dependence across names
#
# Probabilities remain the same; only scenarios are updated.

scenarios_cma = scenarios.apply_cma(
    target_marginals={"AAPL": "t", "MSFT": "t"},
    target_copula="t",
)

# %% Visualise change in distributions


for col in ["AAPL", "MSFT"]:
    plot_hist_compare(
        scenarios.scenarios,
        scenarios_cma.scenarios,
        col,
    )

scatter_compare(
    scenarios.scenarios, scenarios_cma.scenarios, "MSFT", "AAPL"
)  # choose only two assets to visualise (3d takes too long and can't really visualise in 4d, yet...)


# %% Case 2 - just t copula no assumption on marginals
scenarios_just_copula = scenarios.apply_cma(
    target_copula="t",
)

# %% Visualise change in distributions


for col in ["AAPL", "MSFT"]:
    plot_hist_compare(
        scenarios.scenarios,
        scenarios_just_copula.scenarios,
        col,
    )

scatter_compare(
    scenarios.scenarios, scenarios_just_copula.scenarios, "MSFT", "AAPL"
)  # choose only two assets to visualise (3d takes too long and can't really visualise in 4d, yet...)

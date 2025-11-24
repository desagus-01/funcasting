from models.scenarios import ScenarioProb
from template import test_template

info = test_template()

scenario_ex = ScenarioProb("x", scenarios=info.increms_df)

scenario_ex.with_cma(
    target_marginals={"AAPL": "t", "MSFT": "norm"}, target_copula="norm"
)

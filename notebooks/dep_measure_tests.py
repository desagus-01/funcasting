from models.scenarios import CopulaMarginalModel, ScenarioProb
from models.views_builder import ViewBuilder
from utils.template import get_template

info_all = get_template()

scenarios = ScenarioProb.default_inst(info_all.increms_df)

views = (
    ViewBuilder(scenarios.scenarios)
    .mean({"AAPL": 0.001, "GOOG": -0.03}, ["equal", "equal"])
    .build()
)

scenarios = scenarios.add_views(views).apply_views()

cma = CopulaMarginalModel.from_scenario_dist(scenarios.scenarios, scenarios.prob)

print(scenarios)

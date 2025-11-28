from models.scenarios import ScenarioProb
from models.views_builder import ViewBuilder
from utils.template import get_template

info_all = get_template()

scenarios = ScenarioProb.default_inst(info_all.increms_df)

views = (
    ViewBuilder(scenarios.scenarios)
    .mean({"AAPL": 0.001, "GOOG": -0.03}, ["equal", "equal"])
    .build()
)

scenarios_2 = scenarios.add_views(views).apply_views()

print(scenarios.schweizer_wolff(("AAPL", "GOOG")))

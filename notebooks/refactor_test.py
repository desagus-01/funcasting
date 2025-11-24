from models.scenarios import ScenarioProb
from template import test_template

info = test_template()

scenario_ex = ScenarioProb(scenarios=info.increms_df)

views = scenario_ex.build_views().mean(target_means={"AAPL": 0.01}, sign_type=["equal"])

print(type(views.views))

# scenario_ex.add_views(views.views)

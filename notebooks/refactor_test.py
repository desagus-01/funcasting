from models.scenarios import ScenarioProb
from models.views_builder import ViewBuilder
from utils.template import test_template

info = test_template().increms_df


x = ScenarioProb.from_scenarios(info)

views = (
    ViewBuilder(x.scenarios)
    .mean({"AAPL": 0.01}, sign_type=["equal"])
    .std({"MSFT": 0.1}, sign_type=["equal_less"])
    .build()
)

z = x.add_views(views).apply_views()

final = z.apply_cma(target_copula="norm")

print(x)
print(final)

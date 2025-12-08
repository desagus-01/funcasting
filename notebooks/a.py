from pprint import pprint as print

from methods.cma import CopulaMarginalModel
from models.scenarios import ScenarioProb
from utils.stat_tests import lag_independence_test
from utils.template import get_template

info_all = get_template()
increms = info_all.increms_df

test = ScenarioProb.default_inst(increms)

cma = CopulaMarginalModel.from_scenario_dist(test.scenarios, test.prob, test.dates)


print(lag_independence_test(cma.copula, cma.prob, 5, assets=["AAPL", "GOOG"]))

# %%
import time
from pprint import pprint as print

from maths.stat_tests import copula_lag_independence_test
from methods.cma import CopulaMarginalModel
from models.scenarios import ScenarioProb
from utils.template import get_template

# %%
# Load data
info_all = get_template()
increms = info_all.increms_df

# %%
test = ScenarioProb.default_inst(increms)
cma = CopulaMarginalModel.from_scenario_dist(test.scenarios, test.prob, test.dates)
assets = None
lag = 2
cma
# %%
# ---- Test copula_lag_independence_test ----
start = time.perf_counter()
result_copula = copula_lag_independence_test(cma.copula, cma.prob, lag, assets=assets)
elapsed_copula = time.perf_counter() - start

print(f"Execution time: {elapsed_copula:.6f} seconds\n")

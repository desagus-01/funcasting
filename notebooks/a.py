import time
from pprint import pprint as print

from maths.stat_tests import copula_lag_independence_test
from methods.cma import CopulaMarginalModel
from models.scenarios import ScenarioProb
from utils.stat_tests import lag_independence_test
from utils.template import get_template

# Load data
info_all = get_template()
increms = info_all.increms_df

test = ScenarioProb.default_inst(increms)

cma = CopulaMarginalModel.from_scenario_dist(test.scenarios, test.prob, test.dates)

assets = None
lag = 5

# ---- Test copula_lag_independence_test ----
start = time.perf_counter()
result_copula = copula_lag_independence_test(cma.copula, cma.prob, lag, assets=assets)
elapsed_copula = time.perf_counter() - start

print("copula_lag_independence_test result:")
print(result_copula)
print(f"Execution time: {elapsed_copula:.6f} seconds\n")


# ---- Test lag_independence_test ----
start = time.perf_counter()
result_basic = lag_independence_test(cma.copula, cma.prob, lag, assets=assets)
elapsed_basic = time.perf_counter() - start

print("lag_independence_test result:")
print(result_basic)
print(f"Execution time: {elapsed_basic:.6f} seconds\n")

# ---- Summary ----
print("SUMMARY:")
print(
    {
        "copula_lag_independence_test_sec": elapsed_copula,
        "lag_independence_test_sec": elapsed_basic,
        "speed_ratio (basic/copula)": elapsed_basic / elapsed_copula
        if elapsed_copula != 0
        else None,
    }
)

# %%

from pprint import pprint as print

from maths.distributions import uniform_probs
from maths.stat_tests import (
    copula_lag_independence_test,
    ellipsoid_lag_test,
    univariate_kolmogrov_smirnov_test,
)
from methods.cma import CopulaMarginalModel
from models.scenarios import ScenarioProb
from utils.template import get_template

# %%
# Load data
info_all = get_template()
increms = info_all.increms_df

PROB = uniform_probs(increms.height)

# KS
ks = univariate_kolmogrov_smirnov_test(increms)
# Elliptical
el = ellipsoid_lag_test(increms, PROB)
# Copula
sp = ScenarioProb.default_inst(increms)
cma = CopulaMarginalModel.from_scenario_dist(
    scenarios=sp.scenarios, prob=sp.prob, dates=sp.dates
)
cp = copula_lag_independence_test(cma.copula, prob=cma.prob)

print(f"""
KS:
{ks}

sp:
{sp}

cp:
{cp}
""")

# %% imports
from pprint import pprint as print

from maths.distributions import uniform_probs
from maths.econometrics.iid_tests import (
    copula_lag_independence_test,
    ellipsoid_lag_test,
    univariate_kolmogrov_smirnov_test,
)
from maths.econometrics.stationarity_tests import stationarity_tests
from maths.helpers import add_detrend_column
from methods.cma import CopulaMarginalModel
from models.scenarios import ScenarioProb
from utils.template import get_template

# %%
info_all = get_template()
risk_drivers = info_all.asset_info.risk_drivers

risk_drivers = add_detrend_column(
    data=risk_drivers,
)

risk_drivers
# %%

assets = risk_drivers.drop("date").columns
tests = ["level", "trend"]


stationarity_tests(risk_drivers, "MSFT_detrended", lags=10)

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

# %% imports
from pprint import pprint as print

from maths.distributions import uniform_probs
from maths.helpers import add_differenced_columns
from maths.stochastic_processes import (
    HypTestRes,
    copula_lag_independence_test,
    ellipsoid_lag_test,
    univariate_kolmogrov_smirnov_test,
)
from methods.cma import CopulaMarginalModel
from models.scenarios import ScenarioProb
from utils.template import get_template


# %% small helpers
def show_results(title: str, results: dict[str, HypTestRes]) -> None:
    print(f"\n=== {title} ===")
    for name, res in results.items():
        print(f"\n[{name}]")
        print(res)


# %% load once
info_all = get_template()


# =========================
# IID-ish checks on increms
# =========================
increms = info_all.asset_info.risk_drivers
increms_d = add_differenced_columns(increms, assets=["AAPL"])
increms = increms_d.select(["date", "AAPL_diff_1"])

prob = uniform_probs(increms.height)

# Build scenario + copula once (copula test reuses it)
sp = ScenarioProb.default_inst(increms)
cma = CopulaMarginalModel.from_scenario_dist(
    scenarios=sp.scenarios, prob=sp.prob, dates=sp.dates
)

iid_res = {
    "KS split (identically distributed)": univariate_kolmogrov_smirnov_test(increms),
    "Ellipsoid lag autocorr": ellipsoid_lag_test(increms, prob),
    "Copula lag independence": copula_lag_independence_test(cma.copula, prob=cma.prob),
}

show_results("IID diagnostics", iid_res)

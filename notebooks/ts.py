# %% imports
from pprint import pprint as print

from maths.stochastic_processes import (
    HypTestRes,
)
from maths.stochastic_processes.phenomenon_tests import (
    test_deterministic_trend,
)
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
# Stationarity quick checks
# =========================
# risk_drivers = add_detrend_column(data=info_all.asset_info.risk_drivers)
#
# # Choose which series to test
# series_to_test = [
#     "MSFT_detrended_p0",
#     "MSFT_detrended_p1",
#     "MSFT_detrended_p2",
#     "MSFT_detrended_p3",
# ]
#
# stationarity_res: dict[str, HypTestRes] = {}
# for col in series_to_test:
#     stationarity_res[col] = stationarity_tests(risk_drivers, col, lags=10, eq_type="c")
#
stationarity_res = test_deterministic_trend(
    data=info_all.asset_info.risk_drivers, assets=["MSFT", "AAPL", "GOOG"]
)


print(stationarity_res)

# show_results("Stationarity tests", stationarity_res)


# =========================
# IID-ish checks on increms
# =========================
# increms = info_all.increms_df
# prob = uniform_probs(increms.height)
#
# # Build scenario + copula once (copula test reuses it)
# sp = ScenarioProb.default_inst(increms)
# cma = CopulaMarginalModel.from_scenario_dist(
#     scenarios=sp.scenarios, prob=sp.prob, dates=sp.dates
# )
#
# iid_res: dict[str, HypTestRes] = {
#     "KS split (identically distributed)": univariate_kolmogrov_smirnov_test(increms),
#     "Ellipsoid lag autocorr": ellipsoid_lag_test(increms, prob),
#     "Copula lag independence": copula_lag_independence_test(cma.copula, prob=cma.prob),
# }
#
# show_results("IID diagnostics", iid_res)

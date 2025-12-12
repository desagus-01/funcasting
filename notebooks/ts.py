# %% imports
from globals import MACKIN_TAU_CUTOFFS
from maths.time_series import adf_max_lag, build_adf_equation, ols
from utils.template import get_template

# %%
# Load data
info_all = get_template()
risk_drivers = info_all.asset_info.risk_drivers

risk_drivers
# %%


# %% ADF

trenddict: dict[str, int | None] = {
    "n": None,
    "c": 0,
    "ct": 1,
    "ctt": 2,
}

# Step 1 - find n lags

max_lags = adf_max_lag(risk_drivers.height, trenddict["c"])

adf_eq = build_adf_equation(risk_drivers, "AAPL", max_lags)


# %% Steps 4 and beyond

# TODO: Look at AIC/BIC methods for autolag for now assume = max_lag

used_lag = max_lags


res = ols(adf_eq.dep_vars, adf_eq.ind_var)

MACKIN_TAU_CUTOFFS.keys()


# %%
# def mackinnonp(teststat, regression="c", N=1, lags=None):
#     min_cutoff = MACKIN_TAU_CUTOFFS[f"min_{regression}"]
#     max_cutoff = MACKIN_TAU_CUTOFFS[f"max_{regression}"]
#     starstat = MACKIN_TAU_CUTOFFS[f"star_{regression}"]
#
#     if teststat > max_cutoff[N - 1]:
#         return 1.0
#     elif teststat < min_cutoff[N - 1]:
#         return 0.0
#     if teststat <= starstat[N - 1]:
#         tau_coef = eval("tau_" + regression + "_smallp[" + str(N - 1) + "]")
#     #        teststat = np.log(np.abs(teststat))
#     # above is only for z stats
#     else:
#         tau_coef = eval("tau_" + regression + "_largep[" + str(N - 1) + "]")
#     return norm.cdf(polyval(tau_coef[::-1], teststat))

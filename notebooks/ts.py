# %% imports

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

res[2].shape

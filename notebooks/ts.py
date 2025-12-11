# %% imports
import numpy as np

from maths.time_series import adf_max_lag, build_adf_equation
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

# Step 4 Fit OLS and retrieve necessary info
res = np.linalg.lstsq(adf_eq.dep_vars, adf_eq.ind_var)
ols_res = res[0]
sum_of_squared_residuals = res[1]
if sum_of_squared_residuals.size == 0:
    ind_var_estimate = adf_eq.dep_vars @ ols_res
    residuals = adf_eq.ind_var - ind_var_estimate
    sum_of_squared_residuals = residuals @ residuals

n_obs, k = adf_eq.dep_vars.shape

cov_scaler = sum_of_squared_residuals / (n_obs - k)

cov_inv = np.linalg.inv(adf_eq.dep_vars.T @ adf_eq.dep_vars)
scaled_cov_inv = cov_scaler * cov_inv
standard_errors = np.sqrt(np.diag(scaled_cov_inv))

standard_errors

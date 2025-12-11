# %% imports
import polars as pl
from scipy.linalg import lstsq

from maths.time_series import adf_max_lag
from utils.helpers import build_diff_df, build_lag_df
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

# Step 2 - build diff df and lag diffs
diff_df = build_diff_df(risk_drivers, "AAPL")

lag_diff_df = build_lag_df(diff_df, "AAPL_diff_1", lags=max_lags)

lag_1 = build_lag_df(risk_drivers, "AAPL", 1)

# Step 3 Clean to regressors and ind var
regressors = lag_diff_df.with_columns(
    AAPL_diff_1=pl.Series(lag_1["AAPL_lag_1"])
).drop_nulls()  # drop nulls to make sure size is same

ind_var = regressors.select(pl.col("AAPL_diff_1"))
regressors = regressors.drop("AAPL_diff_1")


# %% Steps 4 and beyond

# TODO: Look at AIC/BIC methods for autolag for now assume = max_lag

used_lag = max_lags

# Step 4 Fit OLS and retrieve necessary info
ols_res, residuals, rank, a = lstsq(regressors.to_numpy(), ind_var.to_numpy())

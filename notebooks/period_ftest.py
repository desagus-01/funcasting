# %% imports + data
import numpy as np

import maths.stochastic_processes.seasonality as seas
from maths.helpers import add_detrend_columns_max, add_differenced_columns
from utils.template import get_template

aapl_rd = get_template().asset_info.risk_drivers.select("AAPL")

aapl_det = add_detrend_columns_max(aapl_rd, ["AAPL"], max_polynomial_order=2)

aapl_final = add_differenced_columns(aapl_det, ["AAPL"]).drop_nulls()

x = aapl_final.select("AAPL_diff_1").to_numpy().flatten()


# %%

# ---- Build synthetic series ----
# choose length that aligns Fourier bins with 5, 21, 63-day periods
n_days = 630  # LCM(5, 21, 63) = 315; take 2*315 for more samples
t = np.arange(n_days, dtype=float)

# Seasonal components
weekly = 8.0 * np.sin(2 * np.pi * t / 5.0 + 0.3)
monthly = 6.0 * np.sin(2 * np.pi * t / 21.0 - 1.1)
quarterly = 4.0 * np.sin(2 * np.pi * t / 63.0 + 2.0)
rng = np.random.default_rng(12345)
# Mildly autocorrelated noise
e = rng.normal(0.0, 2.0, size=n_days)
for i in range(1, n_days):
    e[i] += 0.4 * e[i - 1]

y = weekly + monthly + quarterly + e
y = y - y.mean()  # remove DC

# ---- Run tests ----
res_weekly = seas.periodogram_seasonality_test(y, "weekly")
res_monthly = seas.periodogram_seasonality_test(y, "monthly")
res_quarterly = seas.periodogram_seasonality_test(y, "quarterly")

res_quarterly

seas.plot_periodogram(y, max_period=130)

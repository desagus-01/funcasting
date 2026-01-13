# %% imports + data
import maths.stochastic_processes.seasonality as seas
from maths.helpers import add_detrend_columns_max, add_differenced_columns
from utils.template import get_template

aapl_rd = get_template().asset_info.risk_drivers.select("AAPL")

aapl_det = add_detrend_columns_max(aapl_rd, ["AAPL"], max_polynomial_order=2)

aapl_final = add_differenced_columns(aapl_det, ["AAPL"]).drop_nulls()

x = aapl_final.select("AAPL_diff_1").to_numpy().flatten()

# %%
seas.periodogram_seasonality_test(data=x, seasonal_period="weekly")

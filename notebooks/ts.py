# %% imports
from maths.helpers import add_detrend_column
from maths.stat_tests import stationarity_tests
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

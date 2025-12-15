# %% imports

from statsmodels.tsa.stattools import adfuller

from maths.stat_tests import augmented_dickey_fuller_test
from utils.template import get_template

# %%
# Load data
info_all = get_template()
risk_drivers = info_all.asset_info.risk_drivers

risk_drivers


# %% Statsmodels
adfuller(risk_drivers.select("GOOG").to_numpy(), regression="c")


# %% ADF

augmented_dickey_fuller_test(risk_drivers, "GOOG", 4, "c")

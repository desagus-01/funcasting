# %% imports

from maths.time_series import augmented_dickey_fuller, kpss
from utils.template import get_template

# %%
info_all = get_template()
risk_drivers = info_all.asset_info.risk_drivers

aapl = risk_drivers.select("AAPL")

aapl
# %%
x = kpss(risk_drivers, "AAPL", "level")

augmented_dickey_fuller()

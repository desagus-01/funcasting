# %% imports

from maths.time_series import augmented_dickey_fuller
from utils.template import get_template

# %%
# Load data
info_all = get_template()
risk_drivers = info_all.asset_info.risk_drivers

risk_drivers
# %%


# %% ADF

augmented_dickey_fuller(risk_drivers, "MSFT")

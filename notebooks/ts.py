# %% imports
from maths.stat_tests import kpss_test
from utils.template import get_template

# %%
info_all = get_template()
risk_drivers = info_all.asset_info.risk_drivers

aapl = risk_drivers.select("AAPL")

aapl
# %%

x = kpss_test(risk_drivers, "AAPL", "level")

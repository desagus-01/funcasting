# %% imports
from pprint import pprint as print

from maths.stochastic_processes.a import trend_diagnostic
from utils.template import get_template

# %% load once
info_all = get_template()


risk_drivers = info_all.asset_info.risk_drivers


x = trend_diagnostic(
    data=risk_drivers,
    assets=["MSFT"],
    order_max=3,
    threshold_order=3,
    trend_type="both",
)
print(x)

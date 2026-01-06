# %% imports
from pprint import pprint as print

from maths.stochastic_processes import (
    HypTestRes,
)
from maths.stochastic_processes.a import trend_diagnostic
from utils.template import get_template


# %% small helpers
def show_results(title: str, results: dict[str, HypTestRes]) -> None:
    print(f"\n=== {title} ===")
    for name, res in results.items():
        print(f"\n[{name}]")
        print(res)


# %% load once
info_all = get_template()


risk_drivers = info_all.asset_info.risk_drivers


x = trend_diagnostic(
    data=risk_drivers,
    assets=["AAPL", "MSFT"],
    order_max=3,
    threshold_order=1,
    trend_type="both",
)
print(x)

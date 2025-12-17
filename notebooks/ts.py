# %% imports
import polars as pl

from maths.stat_tests import kpss_test
from maths.time_series import deterministic_detrend
from utils.template import get_template

# %%
info_all = get_template()
risk_drivers = info_all.asset_info.risk_drivers

aapl = risk_drivers.select(["date", "AAPL"])


# %%


det_aapl = deterministic_detrend(aapl["AAPL"].to_numpy(), polynomial_order=1)

aapl = aapl.with_columns(
    pl.lit(det_aapl.ravel()).alias("aapl_detrended").cast(pl.Float64)
)

# %%

assets = ["AAPL", "aapl_detrended"]
tests = ["level", "trend"]

for asset in assets:
    for test in tests:
        print(f"Result for {asset}-{test}")
        res = kpss_test(aapl, asset, test)
        print(res)

# %% imports

from utils.helpers import build_diff_df, build_lag_df
from utils.template import get_template

# %%
# Load data
info_all = get_template()
risk_drivers = info_all.asset_info.risk_drivers
rd_np = risk_drivers.to_numpy()
# %%


# %%

diff_tbl = build_diff_df(risk_drivers, "AAPL")
build_lag_df(diff_tbl, "AAPL_diff_1")

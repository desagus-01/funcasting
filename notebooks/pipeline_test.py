# %%

from methods.univariate_time_series import is_trend
from utils.template import get_template

data = get_template().asset_info.risk_drivers
# wn_res = check_white_noise(data=data)

# %%

is_trend(data)

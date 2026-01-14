# %%

from methods.univariate_time_series import check_white_noise
from utils.template import get_template

data = get_template().asset_info.increments
wn_res = check_white_noise(data=data, assets=["MSFT"])

# %%
wn_res

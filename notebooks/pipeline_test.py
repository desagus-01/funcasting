# %%

from maths.time_series import stationarity_tests
from methods.univariate_time_series import detrend_pipeline
from utils.helpers import get_assets_names
from utils.template import get_template

data = get_template().asset_info.increments
# wn_res = check_white_noise(data=data)

# %%
assets = get_assets_names(data)

stationarity_tests(data=data, asset="AAPL")

# %%
detrend_pipeline(data=data)

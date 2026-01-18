# %%

from maths.time_series import stationarity_tests
from methods.univariate_time_series import (
    deseason_pipeline,
    detrend_pipeline,
)
from utils.helpers import get_assets_names
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# wn_res = check_white_noise(data=data)

# %%
assets = get_assets_names(data)
stationarity_tests(data=data, asset="fake")
# %%


data_2 = detrend_pipeline(data=data, include_diagnostics=True).updated_data.drop_nulls()

data_2

# %%
deseason_pipeline(data=data_2, include_diagnostics=True)

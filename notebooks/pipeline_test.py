# %%

from maths.time_series import stationarity_tests
from maths.time_series.diagnostics.seasonality import seasonality_diagnostic
from methods.univariate_time_series import (
    deseason_apply,
    deseason_decision_rule,
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

# %%

x = seasonality_diagnostic(data=data_2)

dseas = deseason_decision_rule(x)

deseason_apply(data_2, dseas)

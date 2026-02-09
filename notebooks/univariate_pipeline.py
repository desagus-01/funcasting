from pprint import pprint as p

from methods.forecasting_pipeline import multivariate_forecasting_info
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)


# %%
x = multivariate_forecasting_info(data)
p(x)
# %%

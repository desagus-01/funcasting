from pprint import pprint as p

from methods.forecasting_pipeline import info_for_forecasting
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)


# %%
x = info_for_forecasting(data)
p(x)
# %%
x[1]
for asset, model in x[1].items:
    print(asset, model.model_type)

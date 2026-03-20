# %%

from methods.preprocess_pipeline import deseason_pipeline, detrend_pipeline
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# %%

det_data = detrend_pipeline(data, ["fake"])

deseason_pipeline(data=det_data.updated_data, assets=["fake"])

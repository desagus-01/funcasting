# %%

from methods.preprocess_pipeline import (
    run_univariate_preprocess,
)
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

run_univariate_preprocess(data=data)

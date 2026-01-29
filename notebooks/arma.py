# %%

from maths.helpers import add_differenced_columns
from maths.time_series.models import burg_ar
from methods.preprocess_pipeline import (
    run_univariate_preprocess,
)
from utils.template import get_template, synthetic_series
from utils.visuals import plot_acf_simple

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# %%
data_2 = run_univariate_preprocess(data=data)

# %%
data_to_model = data_2.post_data
aapl = data_to_model.select("AAPL").to_numpy().ravel()
aapl_resid = (
    add_differenced_columns(data_to_model.select("AAPL"), ["AAPL"])
    .drop_nulls()
    .select("AAPL_diff_1")
    .to_numpy()
    .ravel()
)

aapl_resid

# %%

plot_acf_simple(
    add_differenced_columns(data_to_model, ["AAPL"]).drop_nulls(), "AAPL_diff_1", 15
)
# %%
burg_ar(aapl, 3, True)


# %%

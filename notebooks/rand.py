from maths.helpers import autocorrelation, autocovariance
from utils.template import get_template, synthetic_series
from utils.visuals import plot_acf_simple

# %%

data = get_template().asset_info.increments
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

series = data.select("AAPL").to_numpy().ravel()
# %%

autocovariance(series, lag_length=5, use_fft=True)

# %%
acf = autocorrelation(series, confint_alpha=0.1)
acf
# %%
plot_acf_simple(data, "AAPL", lag_length=15)

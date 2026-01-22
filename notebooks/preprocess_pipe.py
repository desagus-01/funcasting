# %%

from methods.preprocess_pipeline import (
    run_univariate_preprocess,
)
from utils.template import get_template, synthetic_series
from utils.visuals import plot_acf_simple

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

data_2 = run_univariate_preprocess(data=data)


# %%
assets = data_2.post_data.columns

for asset in assets:
    print(f"For {asset}:")
    print("Preprocess:")
    plot_acf_simple(data, asset)
    print("Post-Process:")
    plot_acf_simple(data_2.post_data, asset)

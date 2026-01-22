# %%

from maths.helpers import add_differenced_columns
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
data_2
# %%
increms_og = get_template().asset_info.increments.drop("date")


# %%
assets = increms_og.columns
increms_post = add_differenced_columns(data_2.post_data, assets=assets)
increms_post
# %%

for asset in assets:
    print(f"For {asset}:")
    print("Preprocess:")
    plot_acf_simple(increms_og, asset)
    print("Post-Process:")
    plot_acf_simple(increms_post.drop_nulls(), f"{asset}_diff_1")

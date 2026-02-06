from methods.forecasting_pipeline import fit_best_univariate_model
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)


# %%
x = fit_best_univariate_model(data)


# %%

# %%

from methods.model_selection_pipeline import assets_need_mean_modelling, run_best_arma
from methods.preprocess_pipeline import run_univariate_preprocess
from utils.template import get_template, synthetic_series
from utils.visuals import plot_residual_acf_pacf

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)
data_2 = run_univariate_preprocess(data=data)


# %%
x = assets_need_mean_modelling(data_2.post_data, data_2.needs_further_modelling)

models = {}
for asset in x:
    y = data_2.post_data.select(asset).to_numpy().ravel()
    models[asset] = run_best_arma(y)

models

# %%
plot_residual_acf_pacf(asset="fake", residuals=models["fake"].residuals)

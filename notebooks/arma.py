# %%

from maths.time_series.models import auto_arma, run_best_arma
from methods.model_selection_pipeline import assets_need_mean_modelling
from methods.preprocess_pipeline import run_univariate_preprocess
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)


# %%
array = data.select("fake").to_numpy().ravel()
x = auto_arma(array)

run_best_arma(array)

# %%


data_2 = run_univariate_preprocess(data=data)

x = assets_need_mean_modelling(data_2.post_data, data_2.needs_further_modelling)

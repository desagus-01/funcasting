from polars import DataFrame

from methods.model_selection_pipeline import (
    mean_modelling_pipeline,
    volatility_modelling_pipeline,
)
from methods.preprocess_pipeline import run_univariate_preprocess
from utils.helpers import timeit
from utils.template import get_template, synthetic_series

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)
data_2 = run_univariate_preprocess(data=data)

u_res = mean_modelling_pipeline(data_2.post_data, assets=data_2.needs_further_modelling)
u_res
# %%
volatility_modelling_pipeline(u_res)


@timeit
def fit_best_univariate_model(data: DataFrame, assets: list[str] | None = None):
    post_process = run_univariate_preprocess(data=data, assets=assets)
    mean_modelling = mean_modelling_pipeline(
        data=post_process.post_data, assets=post_process.needs_further_modelling
    )
    volatility_modelling = volatility_modelling_pipeline(mean_model_res=mean_modelling)
    return post_process, mean_modelling, volatility_modelling

# %%
from arch import arch_model

from methods.model_selection_pipeline import (
    mean_modelling_pipeline,
    needs_volatility_modelling,
)
from methods.preprocess_pipeline import run_univariate_preprocess
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)
data_2 = run_univariate_preprocess(data=data)

# %%

u_res = mean_modelling_pipeline(data_2.post_data, assets=data_2.needs_further_modelling)

u_res
# %%
assets_vol = needs_volatility_modelling(u_res)
assets_vol

# %%
x = u_res["AAPL"].residuals
model = arch_model(x, mean="zero", vol="constant", dist="normal", rescale=True)
res = model.fit()
print(res.summary())
res.plot()

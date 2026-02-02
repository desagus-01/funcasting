# %%

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

# %%

a = data_2.post_data
# %%


# %%
for asset in data_2.needs_further_modelling:
    if asset in u_res.keys():
        # print(f"{asset} which has done mean modelling")
        raw = u_res[asset].residuals
        df = sum(u_res[asset].model_order)
    else:
        # print(f"{asset} no mean modelling, therefore constant")
        raw = data_2.post_data.select(asset).to_numpy().ravel()
        raw = raw - raw.mean()  # constant mean residuals
        df = 0
    a = needs_volatility_modelling(
        raw,
        asset,
        ljung_box_lags=[10, 20],
        arch_lags=[5, 10, 15],
        degrees_of_freedom=df,
    )
    lj_res_rej = a[0].rejected
    arch_rej = a[1].rejected
    if (len(lj_res_rej) >= 1) or (len(arch_rej) >= 2):
        print("model")

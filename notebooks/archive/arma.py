# %%
from statsmodels.stats.diagnostic import het_arch

from maths.time_series.iid_tests import ljung_box_test
from methods.model_selection_pipeline import mean_modelling_pipeline
from methods.preprocess_pipeline import run_univariate_preprocess
from utils.template import get_template, synthetic_series
from utils.visuals import plot_residual_acf_pacf

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

for asset in data_2.needs_further_modelling:
    if asset in u_res.keys():
        print(f"{asset} which has done mean modelling")
        raw = u_res[asset].residuals
        df = sum(u_res[asset].model_order)
    else:
        print(f"{asset} no mean modelling, therefore constant")
        raw = data_2.post_data.select(asset).to_numpy().ravel()
        raw = raw - raw.mean()  # constant mean residuals
        df = 0
    array = raw**2
    a = ljung_box_test(array)
    print("Ljung-box on e^2")
    print(a)
    print("Engle test")
    print(het_arch(raw, nlags=10, ddof=df))
    plot_residual_acf_pacf(asset=f"{asset}", residuals=raw)

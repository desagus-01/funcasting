import numpy as np

from maths.distributions import uniform_probs
from methods.forecasting_pipeline import (
    conditional_mean_next,
    conditional_variance_next,
    draw_invariant_shock,
    multivariate_forecasting_info,
)
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# %%
x = multivariate_forecasting_info(data)
inv_df = x.invariants
prob_ex = uniform_probs(inv_df.height)
models = x.models

# %% example for appl
n_paths = 100

print(models["AAPL"].form)
#
# # %%
# # create arrays for both mean and vol params
# mean_params = models["AAPL"].form.mean_model.params
# for param_name, value in mean_params.items():
#     mean_params[param_name] = np.full(n_paths, value)
#
# vol_params = models["AAPL"].form.volatility_model.params
# for param_name, value in vol_params.items():
#     vol_params[param_name] = np.full(n_paths, value)
#
# print(mean_params, vol_params)
#
# # %%
# # do the same now for states
# mean_states = models["AAPL"].state0.mean
#
# np.full(n_paths, mean_states.series)
#
# %%
test = conditional_mean_next(
    mean_form=models["AAPL"].form.mean_model, mean_state=models["AAPL"].state0.mean
)
test_vol = conditional_variance_next(
    models["AAPL"].form.volatility_model, models["AAPL"].state0.vol
)

mean_next_vec, vol_next_vec = np.full(n_paths, test), np.full(n_paths, test_vol)

# %%
shocks, prob = draw_invariant_shock(
    invariants_df=inv_df,
    assets=["AAPL"],
    prob_vector=prob_ex,
    horizon=100,
    n_sims=100,
    seed=1,
)

shocks[0]

# %%

mean_next_vec + np.sqrt(vol_next_vec) * shocks[:, 0, 0]

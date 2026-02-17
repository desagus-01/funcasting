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


asset = "AAPL"
model = models[asset]
n_paths = 1000
h = 100

# draw shocks for horizon=3 (you can set horizon=3 directly)
shocks, prob = draw_invariant_shock(
    invariants_df=inv_df,
    assets=[asset],
    prob_vector=prob_ex,
    horizon=h,
    n_sims=n_paths,
    seed=1,
)

# ---- make local mutable copies of the state (so we can update it) ----
# x_hist = model.state0.mean.series.copy()  # last x values (post-processed series)
# eps_hist = None
# if model.form.mean_model is not None and model.form.mean_model.kind == "arma":
#     eps_hist = (
#         None
#         if model.state0.mean.mean_residuals is None
#         else model.state0.mean.mean_residuals.copy()
#     )
#
# vol_resid_hist = None
# sig2_hist = None
# if model.state0.vol is not None:
#     vol_resid_hist = model.state0.vol.volatility_residuals.copy()
#     sig2_hist = model.state0.vol.conditional_volatility_sq.copy()
#
# # store path results
# x_paths = np.zeros((n_paths, h), dtype=float)
#

# %%

mean_state = model.state0.mean
vol_state = model.state0.vol
x_paths = np.zeros((n_paths, h), dtype=float)
# ---- iterate steps ----
for t in range(h):
    # build a "current state" object for this step
    mu = conditional_mean_next(model.form.mean_model, mean_state)
    sig2 = conditional_variance_next(model.form.volatility_model, vol_state)

    eps_t = shocks[:, t, 0]  # (n_paths,)

    # next-step x for each path
    if sig2 == 0.0:
        x_next = mu + eps_t
    else:
        x_next = mu + np.sqrt(sig2) * eps_t

    x_paths[:, t] = x_next

    mean_state.add_to_series(x_paths)

# %%
#     if (
#         model.form.mean_model is not None
#         and model.form.mean_model.kind == "arma"
#         and model.form.mean_model.order is not None
#     ):
#         p, q = model.form.mean_model.order
#         keep = max(p, 1)
#         x_hist = x_hist[-keep:]
#         # update mean residual lags (MA part uses eps_hist)
#         if q > 0:
#             eps_val = x_next.mean() - mu  # simplest proxy residual
#             if eps_hist is None:
#                 eps_hist = np.array([eps_val], dtype=float)
#             else:
#                 eps_hist = np.append(eps_hist, eps_val)[-q:]
#     else:
#         x_hist = x_hist[-10:]  # keep last 10 like your builder
#
#     # update GARCH buffers (vol recursion uses shock_hist and variance_hist)
#     if vol_resid_hist is not None and sig2_hist is not None:
#         # for your recursion you used "volatility_residuals" as shocks
#         # simplest: treat eps as the shock innovation
#         shock_val = eps_t.mean()  # simplest proxy across paths
#         vol_resid_hist = np.append(vol_resid_hist, shock_val)
#         # keep last max(p,o)
#         p_g, o_g, q_g = model.form.volatility_model.order
#         m = max(p_g, o_g)
#         vol_resid_hist = vol_resid_hist[-m:]
#
#         sig2_hist = np.append(sig2_hist, sig2)
#         sig2_hist = sig2_hist[-q_g:]
#
# # x_paths.shape

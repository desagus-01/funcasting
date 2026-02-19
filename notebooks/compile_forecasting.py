from maths.distributions import uniform_probs
from methods.forecasting_pipeline import (
    draw_invariant_shock,
    multivariate_forecasting_info,
)
from methods.simulation_forecasting import (
    garch_simulation_paths,
    mean_simulation_paths,
    simulate_asset_paths,
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

# %% for AAPL only for now
models = x.models
n_paths = 1000
asset = "GOOG"
aapl_model = models[asset]
h = 2000


params = aapl_model.model.compile_params()
invariant_shock = draw_invariant_shock(
    inv_df, assets=["AAPL"], prob_vector=prob_ex, horizon=h, n_sims=n_paths, seed=None
)
innov_aapl = invariant_shock[:, :, 0]  # (n_paths, h)

sigma2_paths, eps_paths = garch_simulation_paths(
    params=params,
    garch_order=aapl_model.model.vol_order,
    innovations_for_asset=innov_aapl,
    var_start=aapl_model.state0.var_hist,
    eps_start=aapl_model.state0.vol_residual_lags,
)


mean_simulation_paths(
    params=params,
    mean_kind=aapl_model.model.mean_kind,
    mean_order=aapl_model.model.mean_order,
    eps_paths=eps_paths,
    state_series_hist=aapl_model.state0.series_hist,
    state_ma_resid_lags=aapl_model.state0.ma_residual_lags,
)

# %%
simulate_asset_paths(forecast_model=aapl_model, innovations=innov_aapl)

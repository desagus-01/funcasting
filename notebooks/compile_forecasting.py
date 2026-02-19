import numpy as np

from maths.distributions import uniform_probs
from methods.forecasting_pipeline import (
    draw_invariant_shock,
)
from methods.simulation_forecasting import (
    simulate_asset_paths,
)
from utils.template import get_template, synthetic_series
from utils.visuals import plot_simulation_results

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
n_paths = 10000
asset = "fake"
aapl_model = models[asset]
h = 200

models
# %%
params = aapl_model.model.compile_params()
invariant_shock = draw_invariant_shock(
    inv_df,
    assets=["AAPL", "GOOG"],
    prob_vector=prob_ex,
    horizon=h,
    n_sims=n_paths,
    seed=None,
    method="cma",
    target_copula="t",
)
innov_aapl = invariant_shock[:, :, 0]  # (n_paths, h)

y_paths = simulate_asset_paths(forecast_model=aapl_model, innovations=innov_aapl)

logP_paths = 0.0 + np.cumsum(y_paths, axis=1)
P_paths = np.exp(logP_paths)
plot_simulation_results(
    P_paths,
    integrate=False,
    title="Simulated price paths (starting at 1.0)",
)

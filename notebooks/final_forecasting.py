import numpy as np

from maths.distributions import uniform_probs
from methods.forecasting_pipeline import run_n_steps_forecast
from utils.template import get_template, synthetic_series
from utils.visuals import plot_simulation_results

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# %%
prob_ex = uniform_probs(data.height)
forecasts = run_n_steps_forecast(
    data=data,
    prob=prob_ex,
    horizon=100,
    n_sims=10000,
    seed=1,
    assets=["AAPL", "GOOG", "MSFT", "fake"],
    method="cma",
    target_copula="t",
)

# %%

forecasts
# %%

for asset, res in forecasts.items():
    x0 = data.select(asset).to_numpy()[-1, 0]
    logP_paths = x0 + np.cumsum(res, axis=1)
    paths = np.exp(logP_paths)
    plot_simulation_results(paths, title=f"{asset}")


# %%
assets = ["AAPL", "GOOG", "MSFT", "fake"]
s = 99  # zero-based index for step 100

J = np.column_stack([forecasts[a][:, s] for a in assets])

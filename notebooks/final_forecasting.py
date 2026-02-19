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
    n_sims=1000,
    seed=1,
    assets=["AAPL", "GOOG", "MSFT", "fake"],
    method="cma",
    target_copula="t",
)

# %%

for asset, res in forecasts.items():
    logP_paths = 0.0 + np.cumsum(res, axis=1)
    paths = np.exp(logP_paths)

    plot_simulation_results(paths, title=f"{asset}")

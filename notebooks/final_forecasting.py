from maths.distributions import uniform_probs
from methods.forecasting_pipeline import run_n_steps_forecast
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# %%
prob_ex = uniform_probs(data.height)
run_n_steps_forecast(
    data=data,
    prob=prob_ex,
    horizon=100,
    n_sims=1000,
    seed=1,
    assets=["AAPL", "GOOG", "MSFT", "fake"],
    method="bootstrap",
)

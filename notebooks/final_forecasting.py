import polars as pl

from pipelines.forecasting import run_n_steps_forecast
from probability.distributions import state_smooth_probs
from utils.tiingo import import_tickers_and_factors, plot_ticker_lines
from utils.visuals import plot_simulation_results

# %%
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv", "./data/tiingo_factors.csv"
)
cols_to_keep = [
    col
    for col in data.columns
    if col == "date" or (data[col].null_count() == 0 and data[col].min() >= 1)
]
data = (
    data.select(cols_to_keep)
    # .filter(pl.col("date") >= datetime(2021, 1, 1))
)
assets = data.columns[20:30] + factors_cols
# %%
data = data.select("date", *assets)
data_long = data.unpivot(
    index="date",
    on=assets,
    variable_name="ticker",
    value_name="adj_close",
).with_columns(pl.col("date").cast(pl.Date))
plot_ticker_lines(data_long)

# %%
horizon = 30
train_data = data.slice(0, data.height - horizon)
test_data = data.slice(data.height - horizon, horizon)
train_data

# %%
prob_ex = state_smooth_probs(train_data.height, half_life=60, time_based=True)
forecasts = run_n_steps_forecast(
    data=train_data,
    prob=prob_ex,
    horizon=horizon,
    n_sims=5000,
    seed=2,
    assets=assets,
    factors=factors_cols,
    method="bootstrap",
    # target_copula="t",
    back_to_price=False,
)

# %%
forecasts
# %%
# tradable asset forecasts only (factors excluded)
for asset, forecast in forecasts.tradable_paths.items():
    plot_simulation_results(forecast, title=f"{asset}")

# %%
# factor forecasts only
for factor, forecast in forecasts.factor_paths.items():
    plot_simulation_results(forecast, title=f"{factor} (factor)")

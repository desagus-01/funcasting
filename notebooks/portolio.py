from datetime import datetime

import polars as pl

from pipelines.forecasting import run_n_steps_forecast
from portfolio.value import build_equal_weight_portfolio_from_df
from probability.distributions import state_smooth_probs
from utils.helpers import wide_to_long

# %%
data = pl.read_csv("./data/tiingo_sample.csv")
cols_to_keep = [
    col
    for col in data.columns
    if col == "date" or (data[col].null_count() == 0 and data[col].min() >= 1)
]
data = (
    data.with_columns(pl.col("date").str.to_datetime("%Y-%m-%d"))
    .filter(pl.col("date") >= datetime(2021, 1, 1))
    .select(cols_to_keep)
)
assets = ["TWO", "BGS", "REG", "GCO", "DLTR"]

# assets = data.columns[1:10]
data = data.select("date", *assets)
long = wide_to_long(data, assets=assets)
d2 = long.with_columns(adj_close=pl.col("adj_close").exp())
port = build_equal_weight_portfolio_from_df(d2, initial_value=10000)
port[0].filter(pl.col("ticker") == "TWO").select("shares").get_columns()

# %%
horizon = 30
prob_ex = state_smooth_probs(data.height, half_life=20, time_based=True)
forecasts = run_n_steps_forecast(
    data=data,
    prob=prob_ex,
    horizon=horizon,
    n_sims=5000,
    seed=2,
    assets=assets,
    method="cma",
    target_copula="t",
    back_to_price=True,
)

# %%
first_step_all_sims = forecasts["TWO"][:, 0]
first_sim_full_horizon = forecasts["TWO"][0, :]


# %%
first_sim_full_horizon
first_step_all_sims

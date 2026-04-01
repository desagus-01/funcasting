import polars as pl

from pipelines.forecasting import run_n_steps_forecast
from portfolio.value import build_equal_weight_portfolio_from_df, portfolio_forecast
from probability.distributions import state_smooth_probs
from utils.helpers import wide_to_long
from utils.tiingo import plot_ticker_lines
from utils.visuals import plot_simulation_results

# %%
data = pl.read_csv("./data/tiingo_sample.csv")
cols_to_keep = [
    col
    for col in data.columns
    if col == "date" or (data[col].null_count() == 0 and data[col].min() >= 2)
]
data = (
    data.with_columns(pl.col("date").str.to_datetime("%Y-%m-%d"))
    # .filter(pl.col("date") >= datetime(2021, 1, 1))
    .select(cols_to_keep)
)
assets = data.columns[1:20]
# assets = ["CAR", "BAC", "KEYS", "TM", "TTC"]
data = data.select("date", *assets)
# %%
long = wide_to_long(data, assets=assets)
d2 = long.with_columns(adj_close=pl.col("adj_close").exp())
plot_ticker_lines(d2)
port = build_equal_weight_portfolio_from_df(d2, initial_value=10000)

# %%
d2
# %%
horizon = 30
prob_ex = state_smooth_probs(data.height, half_life=60, time_based=True)
# prob_ex = uniform_probs(data.height)
forecasts = run_n_steps_forecast(
    data=data,
    prob=prob_ex,
    horizon=horizon,
    n_sims=10000,
    seed=2,
    assets=assets,
    method="bootstrap",
    # target_copula="t",
    # target_marginals={"KEYS": "t", "TTC": "t", "NXRT": "t", "DHIL": "t"},
    back_to_price=True,
)

# %%
for asset, forecast in forecasts.items():
    plot_simulation_results(forecast, title=f"{asset}")

# %%
port_forecast = portfolio_forecast(forecasts, port.shares_mapping, pnl_type="relative")
port_forecast.plot(plot_cumulative=True)

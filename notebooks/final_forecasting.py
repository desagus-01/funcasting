import cProfile
import pstats

import polars as pl

from maths.distributions import uniform_probs
from methods.forecasting_pipeline import run_n_steps_forecast
from utils.tiingo import plot_ticker_lines
from utils.visuals import plot_simulation_results

# %%
data = pl.read_csv("./data/tiingo_sample.csv")
# test IPDN, MBOT, FWONA

# data = data.select(
#     *[col for col in data.columns if data.select(col).null_count().item() == 0]
# )
# assets = data.columns[1:10]
assets = ["FWONA"]
data = data.select("date", *assets)

data

# %%
data_long = (
    data.select("date", *assets)
    .unpivot(
        index="date",
        on=assets,
        variable_name="ticker",
        value_name="adj_close",
    )
    .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
)

plot_ticker_lines(data_long)

# %%
prob_ex = uniform_probs(data.height)

prof = cProfile.Profile()
prof.enable()
forecasts = run_n_steps_forecast(
    data=data,
    prob=prob_ex,
    horizon=30,
    n_sims=5000,
    seed=2,
    assets=assets,
    method="bootstrap",
    back_to_price=False,
    # target_copula="t",
)
prof.disable()
stats = pstats.Stats(prof)
stats.sort_stats("cumulative")  # or "tottime"
stats.print_stats(30)  # top 30 entries

for asset, forecast in forecasts[1].items():
    plot_simulation_results(forecast, title=f"{asset}")
# %%

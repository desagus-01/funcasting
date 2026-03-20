import numpy as np
import polars as pl

from maths.distributions import uniform_probs
from methods.forecasting_pipeline import run_n_steps_forecast
from methods.preprocess_pipeline import run_univariate_preprocess
from utils.tiingo import plot_ticker_lines
from utils.visuals import plot_simulation_results

# %%
data = pl.read_csv("./data/tiingo_sample.csv")
# assets = ["SMBC", "RDN", "BANC", "FCN"]

assets = ["SMBC", "RDN", "BANC", "FCN"]
data = data.select("date", *assets)

# %%
data
data.select(pl.col("BANC").exp())

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
forecasts = run_n_steps_forecast(
    data=data,
    prob=prob_ex,
    horizon=100,
    n_sims=10000,
    seed=1,
    assets=assets,
    method="cma",
    target_copula="t",
)


# %%
asset = "BANC"

post_process = run_univariate_preprocess(data=data, assets=assets)

orig = data.select(asset).to_numpy().ravel()
post = post_process.post_data.select(asset).to_numpy().ravel()

c = orig.mean() - post.mean()

res = forecasts[asset]
logP_paths = res + c
paths = np.exp(logP_paths)

plot_simulation_results(paths, title=f"{asset}")
print("c:", c)
print("res sample:", res[0, :5])
print("logP sample:", logP_paths[0, :5])
print("price sample:", paths[0, :5])

# %%
for asset, res in forecasts.items():
    x0 = data.select(asset).to_numpy()[-1, 0]
    logP_paths = x0 + np.cumsum(res, axis=1)
    paths = np.exp(logP_paths)
    plot_simulation_results(paths, title=f"{asset}")


# %%
# assets = [column for column in data.columns if column != "date"]
# s = 99  # zero-based index for step 100
#
# J = np.column_stack([forecasts[a][:, s] for a in assets])

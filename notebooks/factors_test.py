from typing import get_args

import polars as pl

from pipelines.forecasting import run_n_steps_forecast
from portfolio.factors import (
    portfolio_factor_attribution,
)
from portfolio.value import (
    build_equal_weight_portfolio_from_df,
    equal_weight_target_weights,
    portfolio_forecast,
)
from probability.distributions import state_smooth_probs
from time_series.feature_selection import Criterion
from utils.helpers import wide_to_long
from utils.tiingo import import_tickers_and_factors

# %%
# ── Data loading (unchanged) ─────────────────────────────────────────
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv", "./data/tiingo_factors.csv"
)
cols_to_keep = [
    col
    for col in data.columns
    if col == "date" or (data[col].null_count() == 0 and data[col].min() >= 1)
]
data = data.select(cols_to_keep)
assets = data.columns[20:30] + factors_cols
data = data.select("date", *assets)

# %%
# ── Portfolio construction (unchanged) ───────────────────────────────
tradable_assets = [a for a in assets if a not in factors_cols]
long = wide_to_long(data, assets=assets)
d2 = long.with_columns(pl.col("adj_close").exp().alias("adj_close")).filter(
    pl.col("ticker").is_in(tradable_assets)
)
port = build_equal_weight_portfolio_from_df(d2, initial_value=10000)

# %%
# ── Forecasting (unchanged) ───────────────────────────────────────────
horizon = 30
prob_ex = state_smooth_probs(data.height, half_life=60, time_based=True)
forecasts = run_n_steps_forecast(
    data=data,
    prob=prob_ex,
    horizon=horizon,
    n_sims=10000,
    seed=2,
    assets=assets,
    factors=factors_cols,
    method="bootstrap",
    back_to_price=True,
)


# %%
# ── Portfolio forecast (unchanged) ───────────────────────────────────
tradable = forecasts.tradable_paths
target_weights = equal_weight_target_weights(list(tradable.keys()))
port_forecast = portfolio_forecast(
    tradable,
    forecasts.path_probs,
    port.shares_mapping,
    port.t0_prices,
    pnl_type="relative",
    weight_mode="static",
    target_weights=target_weights,
)

port_forecast.plot(end_horizon=30)


# %%


for criteria in get_args(Criterion):
    print(f"Criteria {criteria}")
    f_a = portfolio_factor_attribution(
        port_forecast,
        forecasts.factor_paths,
        data,
        30,
        auto_select_factors=True,
        criterion=criteria,
    )
    print(f_a)

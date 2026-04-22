import numpy as np
import polars as pl

from pipelines.forecasting import run_n_steps_forecast
from portfolio.attribution.performance import portfolio_factor_attribution
from portfolio.attribution.risk import (
    PortfolioRiskAttribution,
)
from portfolio.risk import LossDistribution, var, var_scenario_index
from portfolio.value import (
    build_equal_weight_portfolio_from_df,
    equal_weight_target_weights,
    portfolio_forecast,
)
from probability.distributions import state_smooth_probs
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
assets = list(data.columns[15:20]) + list(factors_cols)
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
prob_ex = state_smooth_probs(data.height, half_life=120, time_based=True)
forecasts = run_n_steps_forecast(
    data=data,
    prob=prob_ex,
    horizon=horizon,
    n_sims=30000,
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

port_forecast.plot(end_horizon=30, plot_cumulative=True)
# %%
f_a = portfolio_factor_attribution(
    port_forecast,
    forecasts.factor_paths,
    data,
    30,
    auto_select_factors=True,
    criterion="bic",
)
# %%
loss_dist = LossDistribution.from_portfolio_forecast(port_forecast)
risk_attribution = PortfolioRiskAttribution.from_performance_attribution(f_a)
factor_exposures = risk_attribution.exposures
factor_keys = [
    k for k, v in factor_exposures.items() if isinstance(v, (float, np.floating))
]
var_vals = var(
    distribution=loss_dist.loss_values, prob=loss_dist.probs, method="empirical"
)
last_var = var_vals[-1]

# %%
joint_risk = risk_attribution.joint_distribution

factors_at_var = (
    joint_risk.filter(pl.col("loss") == last_var)
    .sort("loss", descending=True)
    .drop("loss")
)
exposures = risk_attribution.exposures
factors_at_var.select(
    [(pl.col(col) * exposures.get(col, 0)).alias(col) for col in factors_at_var.columns]
)

# %%
idx = var_scenario_index(loss_dist.loss_values[:, 0], prob=loss_dist.probs)

# %%

alpha = 0.05  # or 0.01 for 99% VaR
q = 1 - alpha

joint_risk = risk_attribution.joint_distribution.with_columns(
    prob=pl.Series(risk_attribution.probs)
)

var_row = (
    joint_risk.sort("loss")
    .with_columns(pl.col("prob").cum_sum().alias("cum_prob"))
    .filter(pl.col("cum_prob") >= q)
    .head(1)
)

factor_cols = [c for c in joint_risk.columns if c not in ("loss", "prob", "cum_prob")]
exposures = risk_attribution.exposures

contribs = {
    c: float(var_row.select(c).item()) * float(exposures[c]) for c in factor_cols
}

sum_contribs = sum(contribs.values())
loss_at_var_row = float(var_row.select("loss").item())

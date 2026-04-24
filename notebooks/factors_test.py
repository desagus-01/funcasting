# %%
import logging

import numpy as np
import polars as pl

from pipelines.forecasting import run_n_steps_forecast
from policy import LogConfig
from portfolio import (
    build_equal_weight_portfolio_from_df,
    equal_weight_target_weights,
    portfolio_forecast,
)
from portfolio.attribution.performance import portfolio_factor_attribution
from portfolio.attribution.risk import (
    PortfolioRiskAttribution,
    cvar_contribution,
)
from portfolio.risk import LossDistribution, cvar, var
from probability.distributions import state_smooth_probs
from scenarios.panel import ScenarioPanel
from utils.helpers import wide_to_long
from utils.log import setup_logging
from utils.tiingo import import_tickers_and_factors

setup_logging(LogConfig(level=logging.INFO))

# %%
# ── Data loading ─────────────────────────────────────────────────────

data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)

cols_to_keep = [
    col
    for col in data.columns
    if col == "date" or (data[col].null_count() == 0 and data[col].min() >= 1)
]

data = data.select(cols_to_keep)

assets = list(data.columns[15:20]) + list(factors_cols)
data = data.select("date", *assets)

tradable_assets = [a for a in assets if a not in factors_cols]

# %%
# ── Build historical ScenarioPanel ───────────────────────────────────
#
# This is the realised/historical weighted panel.
# It is not a forecast object.
# It gives the forecasting pipeline clean input data + probabilities.

horizon = 30
n_sims = 30_000
analysis_horizon = 30

prob_ex = state_smooth_probs(
    data.height,
    half_life=120,
    time_based=True,
)

historical_panel = ScenarioPanel.from_frame(
    data,
    prob=prob_ex,
)

historical_panel

# %%
# ── Portfolio construction ───────────────────────────────────────────
#
# This still uses realised data.
# Keep it separate from forecast paths.

long = wide_to_long(
    historical_panel.to_frame(),
    assets=assets,
)

d2 = long.with_columns(pl.col("adj_close").exp().alias("adj_close")).filter(
    pl.col("ticker").is_in(tradable_assets)
)

port = build_equal_weight_portfolio_from_df(
    d2,
    initial_value=10_000,
)

# %%
# ── Forecasting ──────────────────────────────────────────────────────
#
# run_n_steps_forecast remains the orchestration function.
# ScenarioPanel does not own the forecasting pipeline.

forecasts = run_n_steps_forecast(
    data=historical_panel.to_frame(),
    prob=historical_panel.prob,
    horizon=horizon,
    n_sims=n_sims,
    seed=2,
    assets=assets,
    factors=factors_cols,
    method="cma",
    target_copula="norm",
    back_to_price=True,
)

forecasts

# %%
# ── Horizon slices as ScenarioPanel ──────────────────────────────────
#
# These are now the clean bridge between Monte Carlo paths and
# downstream scenario/risk modelling.
#
# rows    = simulated paths
# columns = assets / factors
# prob    = path probabilities

forecast_panel_h = forecasts.at_horizon(
    analysis_horizon,
    subset="all",
)

tradable_panel_h = forecasts.at_horizon(
    analysis_horizon,
    subset="tradable",
)

factor_panel_h = forecasts.at_horizon(
    analysis_horizon,
    subset="factors",
)

forecast_panel_h.values.head()

# %%
# ── Portfolio forecast ───────────────────────────────────────────────
#
# Portfolio forecast still needs the full path object, not one horizon.
# So we continue using forecasts.tradable_paths here.

tradable_paths = forecasts.tradable_paths

target_weights = equal_weight_target_weights(list(tradable_paths.keys()))

port_forecast = portfolio_forecast(
    asset_forecasts=tradable_paths,
    path_probs=forecasts.path_probs,
    initial_asset_shares=port.shares_mapping,
    initial_prices=port.t0_prices,
    pnl_type="relative",
    weight_mode="static",
    target_weights=target_weights,
)

port_forecast.plot(
    end_horizon=analysis_horizon,
    plot_cumulative=True,
)
# %%
# ── Factor attribution at selected horizon ───────────────────────────
#
# This still uses the full factor paths because attribution computes
# cumulative performance up to the selected horizon.

perf_attr = portfolio_factor_attribution(
    portfolio_forecast=port_forecast,
    factors_forecast=forecasts.factor_paths,
    original_data=historical_panel.to_frame(),
    horizon=analysis_horizon,
    auto_select_factors=True,
    criterion="bic",
)

perf_attr.joint_distribution.head()

# %%
# ── Portfolio loss distribution ──────────────────────────────────────
#
# Existing API:
# LossDistribution.from_portfolio_forecast(port_forecast)
#
# This returns the full horizon matrix of cumulative losses.
# For single-horizon VaR/CVaR, slice explicitly.

loss_dist = LossDistribution.from_portfolio_forecast(port_forecast)

loss_at_horizon = loss_dist.at_horizon(analysis_horizon).values.to_numpy()
path_probs = loss_dist.probs

var_at_horizon = var(
    distribution=loss_at_horizon,
    prob=path_probs,
    method="empirical",
)

cvar_at_horizon = cvar(
    distribution=loss_at_horizon,
    prob=path_probs,
    method="empirical",
)

var_at_horizon, cvar_at_horizon
# %%
# ── Risk attribution ─────────────────────────────────────────────────

risk_attr = PortfolioRiskAttribution.from_performance_attribution(perf_attr)

factor_exposures = risk_attr.exposures

factor_keys = [
    k for k, v in factor_exposures.items() if isinstance(v, (float, np.floating))
]

factor_keys

# %%
# ── CVaR contribution ────────────────────────────────────────────────

cvar_contrib = cvar_contribution(
    joint_distribution_factors=risk_attr.joint_distribution,
    factors_exposures=risk_attr.exposures,
    prob=risk_attr.probs,
)

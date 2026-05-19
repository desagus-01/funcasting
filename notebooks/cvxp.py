# %%
import logging

import numpy as np
from pipelines.forecasting import AssetUniverse, run_n_steps_forecast
from policy import LogConfig
from portfolio.policy.constraints import FullyInvested, LongOnly, MaxWeight, MinWeight
from portfolio.policy.moments import HorizonMoments
from portfolio.policy.optimization import MeanCovMPO, mpo_mean_cov
from probability.distributions import state_smooth_probs
from scenarios.panel import ScenarioPanel
from utils.log import setup_logging
from utils.tiingo import import_tickers_and_factors

setup_logging(LogConfig(level=logging.WARNING))

# %%
# ── Data loading ─────────────────────────────────────────────────────
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)

cols_to_keep = [
    col
    for col in data.columns
    if col == "date"
    or (
        data[col].null_count() == 0
        and data[col].dtype.is_numeric()
        and float(data[col].min()) >= 1  # type: ignore[arg-type]
    )
]

data = data.select(cols_to_keep)

tradable_assets = list(data.columns[10:20])
factors_cols = list(factors_cols)
universe = AssetUniverse(assets=tradable_assets, factors=factors_cols)
data = data.select("date", *universe.all_tickers)

# %%
# ── Build historical ScenarioPanel ───────────────────────────────────
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
# ── Forecasting ──────────────────────────────────────────────────────
forecasts = run_n_steps_forecast(
    data=historical_panel.to_frame(),
    prob=historical_panel.prob,
    horizon=horizon,
    n_sims=n_sims,
    seed=2,
    universe=universe,
    method="bootstrap",
    # target_copula="t",
    back_to_price=True,
)


# %%
h = 10
forecast_moms = HorizonMoments.from_forecast_paths(forecasts, horizons=10)
assets = forecast_moms.assets


x = mpo_mean_cov(
    forecast_moms,
    horizons=10,
    n_assets=len(assets),
    risk_aversion=0.8,
    current_weights=np.full(len(assets), 1 / len(assets)),
    transaction_cost=0.005,
    constraints=[
        LongOnly(),
        FullyInvested(),
        MaxWeight(0.3),
        MinWeight(limit=0.02),
    ],
)

x["target_weights_by_asset"]
# %%

optimizer = MeanCovMPO(
    horizons=10,
    n_assets=len(assets),
    constraints=[
        LongOnly(),
        FullyInvested(),
        MaxWeight(0.3),
        MinWeight(limit=0.02),
    ],
)

# %%
res = optimizer.solve(
    horizon_moments=forecast_moms,
    risk_aversion=0.5,
    current_weights=np.full(len(assets), 1 / len(assets)),
)

res["target_weights_by_asset"]

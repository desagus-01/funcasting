# %%
import logging

import cvxportfolio as cvx
import pandas as pd
from pipelines.forecasting import AssetUniverse, run_n_steps_forecast
from policy import LogConfig
from portfolio.policy.moments import HorizonMoments
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
    if col == "date" or (data[col].null_count() == 0 and data[col].min() >= 1)
]

data = data.select(cols_to_keep)

tradable_assets = list(data.columns[15:40])
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

H = 30

# Get a stable asset ordering once.
forecast_moms_1 = HorizonMoments.from_forecast_paths(forecasts, step=1)
assets = forecast_moms_1.assets

objectives = []
constraints_by_step = []

for step in range(1, H + 1):
    # step=1: next-period forecast
    # step=30: 30th step-ahead forecast
    moms = HorizonMoments.from_forecast_paths(forecasts, step=step)

    r_hat_step = pd.Series(moms.mean, index=moms.assets).reindex(assets)

    Sigma_step = pd.DataFrame(
        moms.covariances,
        index=moms.assets,
        columns=moms.assets,
    ).reindex(index=assets, columns=assets)

    er_step = cvx.ReturnsForecast(r_hat=r_hat_step)
    cov_step = cvx.FullCovariance(Sigma_step)

    objectives.append(er_step - 0.5 * cov_step)

    constraints_by_step.append(
        [
            cvx.LongOnly(),
            cvx.LeverageLimit(1),
        ]
    )


# %%


strategy = cvx.MultiPeriodOptimization(
    objective=objectives,
    constraints=constraints_by_step,
    include_cash_return=False,
)  # h must be a pd.Series of dollar holdings with cash as the last element
# t must be provided explicitly when market_data=None
initial_value = 100_000.0
h = pd.Series(0.0, index=assets + ["cash"])
h["cash"] = initial_value  # start fully in cash

t = pd.Timestamp(data["date"].max().strftime("%Y-%m-%d"))

u, t_out, shares = strategy.execute(h=h, market_data=None, t=t)
print("Trade vector (dollars):")
print(u)
print("\nExecution timestamp:", t_out)

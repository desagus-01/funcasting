import numpy as np
import polars as pl

from pipelines.forecasting import run_n_steps_forecast
from portfolio.value import (
    build_equal_weight_portfolio_from_df,
    equal_weight_target_weights,
    portfolio_forecast,
)
from probability.distributions import state_smooth_probs
from time_series.estimation import (
    EquationTypes,
    OLSEquation,
    add_deterministics_to_eq,
    weighted_ols,
)
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

port_forecast.plot()

# %%
# Get last known factor prices from historical data (t0)
factor_t0 = {
    col: float(np.exp(data.select(col).drop_nulls()[-1, 0])) for col in factors_cols
}

port_total_return = np.prod(1.0 + port_forecast.pnl, axis=1) - 1.0

data_reg = {"port": port_total_return}
for factor, forecast in forecasts.factor_paths.items():
    t0_price = factor_t0[factor]
    # Now: t0 → step 30, matching portfolio window exactly
    data_reg[factor] = (forecast[:, -1] / t0_price) - 1.0

regression_df = pl.DataFrame(data_reg)

regression_df

# %%


def _build_factor_reg_eq(
    data: pl.DataFrame,
    eq_type: EquationTypes,
    port_col_name: str = "port",
) -> OLSEquation:
    dependent_var = data.select(port_col_name).to_numpy()
    independent_vars = data.drop(port_col_name).to_numpy()
    if eq_type != "nc":
        independent_vars = add_deterministics_to_eq(
            independent_vars=independent_vars, eq_type=eq_type
        )
    return OLSEquation(ind_var=independent_vars, dep_vars=dependent_var)


factor_eq = _build_factor_reg_eq(regression_df, eq_type="c")

results = weighted_ols(
    dependent_var=factor_eq.dep_vars,
    independent_vars=factor_eq.ind_var,
    prob=forecasts.path_probs,  # scenario weights from simulation
)
results

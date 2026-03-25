import numpy as np
import polars as pl

from maths.distributions import state_smooth_probs
from methods.forecasting_pipeline import run_n_steps_forecast
from utils.visuals import plot_simulation_results

# %%
data = pl.read_csv("./data/tiingo_sample.csv")

data = data.select(
    *[
        col
        for col in data.columns
        if col == "date"
        or (
            data.select(col).null_count().item() == 0
            and data.select(col).min().item() >= 1
        )
    ]
)

assets = data.columns[1:90]
data = data.select("date", *assets)

# %%
horizon = 30

train_data = data.slice(0, data.height - horizon)
test_data = data.slice(data.height - horizon, horizon)

prob_ex = state_smooth_probs(train_data.height, half_life=20, time_based=True)

forecasts = run_n_steps_forecast(
    data=train_data,
    prob=prob_ex,
    horizon=horizon,
    n_sims=5000,
    seed=2,
    assets=assets,
    method="cma",
    back_to_price=False,
    target_copula="t",
)
price_forecasts = forecasts[1]


def crps_ensemble(
    fcst: np.ndarray,
    obs: np.ndarray,
    method: str = "fair",
    axis: int = 0,
) -> np.ndarray:
    if method not in {"fair", "ecdf"}:
        raise ValueError("method must be 'fair' or 'ecdf'")

    if axis != 0:
        fcst = np.moveaxis(fcst, axis, 0)

    fcst = np.asarray(fcst, dtype=float)
    obs = np.asarray(obs, dtype=float)

    m = fcst.shape[0]
    if m < 2 and method == "fair":
        raise ValueError("Need at least 2 ensemble members for method='fair'")

    fcst_obs_term = np.mean(np.abs(fcst - obs), axis=0)

    fcst_sorted = np.sort(fcst, axis=0)
    i = np.arange(m).reshape((m,) + (1,) * (fcst.ndim - 1))
    coeffs = 2 * i - m + 1
    spread_numerator = 2.0 * np.sum(fcst_sorted * coeffs, axis=0)

    if method == "ecdf":
        spread_term = spread_numerator / (2.0 * m * m)
    else:
        spread_term = spread_numerator / (2.0 * m * (m - 1))

    return fcst_obs_term - spread_term


realized_prices = {asset: test_data.get_column(asset).to_numpy() for asset in assets}

crps_by_asset = {}
crps_by_asset_horizon = {}

for asset in assets:
    fcst = price_forecasts[asset]  # (5000, 30)
    obs = realized_prices[asset]  # (30,)

    crps_h = crps_ensemble(fcst, obs, method="fair")
    crps_by_asset_horizon[asset] = crps_h
    crps_by_asset[asset] = crps_h.mean()


# %%
crps_by_asset

b = ["TWO", "BGS", "REG", "GCO", "DLTR"]
for asset in ["TWO", "BGS", "REG", "GCO", "DLTR"]:
    x = price_forecasts[asset]
    print(asset)
    print("shape:", x.shape)
    print("min:", np.nanmin(x))
    print("max:", np.nanmax(x))
    print("mean:", np.nanmean(x))
    print("any inf:", np.isinf(x).any())
    print("any nan:", np.isnan(x).any())
    print()

# %%
for asset, forecast in forecasts[1].items():
    if asset in b:
        plot_simulation_results(forecast, title=f"{asset}")

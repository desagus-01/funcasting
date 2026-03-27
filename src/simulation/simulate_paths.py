from __future__ import annotations

import numpy as np
from numpy._typing import NDArray

from simulation.engines.mean import mean_simulation_paths
from simulation.engines.utils import as_sims_by_horizon
from simulation.engines.volatility import garch_simulation_paths
from simulation.state import SimulationForecast


def simulate_asset_paths(
    forecast_model: SimulationForecast,
    innovations: NDArray[np.floating],
) -> NDArray[np.floating]:
    model = forecast_model.model
    params = model.compile_params()
    state0 = forecast_model.state0

    innovations, n_sims, horizon = as_sims_by_horizon(innovations)

    if model.vol_kind == "none":
        eps_paths = model.innovation_scale * innovations
    elif model.vol_kind == "garch":
        _, eps_paths = garch_simulation_paths(
            params=params,
            garch_order=model.vol_order,
            eps_start=state0.vol_residual_lags,
            var_start=state0.var_hist,
            innovations=innovations,
        )
    else:
        raise ValueError(f"Unknown vol_kind: {model.vol_kind}")

    y_paths = mean_simulation_paths(
        params=params,
        mean_kind=model.mean_kind,
        mean_order=model.mean_order,
        state_series_hist=state0.series_hist,
        state_ma_resid_lags=state0.ma_residual_lags,
        eps_paths=eps_paths,
    )

    return y_paths

# 1. Create object for port + factors
# 2. Make sure data is correct
# 3. Apply weighted OLS - keep simple for now


from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame


@dataclass(frozen=True, slots=True)
class PortfolioFactors:
    portfolio_pnl_forecast: NDArray[np.floating]
    factors_forecast: dict[str, NDArray[np.floating]]


def _get_t0_factor_values(
    original_data: DataFrame, factors_names: list[str], is_log_price: bool = True
) -> dict[str, float]:
    if is_log_price:
        return {
            col: float(np.exp(original_data.select(col).drop_nulls()[-1, 0]))
            for col in factors_names
        }
    else:
        return {
            col: float(original_data.select(col).drop_nulls()[-1, 0])
            for col in factors_names
        }


def _factors_n_horizon_performance(
    factors_forecast: dict[str, NDArray[np.floating]],
    original_data: DataFrame,
    factors_names: list[str],
    end_horizon: int,
    is_log_price: bool = True,
) -> dict[str, NDArray]:
    if end_horizon <= 0:
        raise ValueError("end_horizon must be a positive integer")
    factors_t0 = _get_t0_factor_values(
        original_data=original_data,
        factors_names=factors_names,
        is_log_price=is_log_price,
    )
    factors_forecast_w_t0 = {}
    for factor, forecast in factors_forecast.items():
        idx = end_horizon - 1
        t0_price = factors_t0[factor]
        factors_forecast_w_t0[factor] = (forecast[:, idx] / t0_price) - 1.0
    return factors_forecast_w_t0

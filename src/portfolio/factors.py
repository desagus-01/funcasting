# 1. Create object for port + factors
# 2. Make sure data is correct
# 3. Apply weighted OLS - keep simple for now


from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame

from scenarios.types import ProbVector
from time_series.estimation import (
    EquationTypes,
    OLSEquation,
    OLSResults,
    add_deterministics_to_eq,
    weighted_ols,
)


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


def portfolio_factor_df(
    factors_cum_forecast: dict[str, NDArray],
    portfolio_cum_forecast: NDArray[np.floating],
    port_col_name: str = "port",
) -> DataFrame:
    port_factors = {port_col_name: portfolio_cum_forecast}
    for factor, forecast in factors_cum_forecast.items():
        port_factors[factor] = forecast
    return DataFrame(port_factors)


def _build_factor_ols_equation(
    portfolio_factor_df: DataFrame,
    eq_type: EquationTypes = "c",
    port_col_name: str = "port",
) -> OLSEquation:
    dependent_var = portfolio_factor_df.select(port_col_name).to_numpy()
    independent_vars = portfolio_factor_df.drop(port_col_name).to_numpy()
    if eq_type != "nc":
        independent_vars = add_deterministics_to_eq(
            independent_vars=independent_vars, eq_type=eq_type
        )
    return OLSEquation(ind_var=independent_vars, dep_vars=dependent_var)


def _run_factor_ols(
    dependent_var: NDArray[np.floating],
    independent_vars: NDArray[np.floating],
    prob: ProbVector | None,
) -> OLSResults:
    return weighted_ols(
        dependent_var=dependent_var, independent_vars=independent_vars, prob=prob
    )


def factor_ols_regression(
    factors_cum_forecast: dict[str, NDArray],
    portfolio_cum_forecast: NDArray[np.floating],
    prob: ProbVector | None = None,
):
    pass
    port_factor_df = portfolio_factor_df(
        factors_cum_forecast=factors_cum_forecast,
        portfolio_cum_forecast=portfolio_cum_forecast,
    )
    factor_ols_eq = _build_factor_ols_equation(port_factor_df)
    return _run_factor_ols(
        dependent_var=factor_ols_eq.dep_vars,
        independent_vars=factor_ols_eq.ind_var,
        prob=prob,
    )

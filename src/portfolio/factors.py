from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame

from portfolio.value import PortfolioForecast
from scenarios.types import ProbVector
from time_series.estimation import (
    EquationTypes,
    OLSEquation,
    OLSResults,
    add_deterministics_to_eq,
    weighted_ols,
)


@dataclass(frozen=True, slots=True)
class HorizonFactorAttribution:
    horizon: int
    factor_names: list[str]
    portfolio_performance_forecast: NDArray[np.floating]
    factor_performance_forecast: dict[str, NDArray[np.floating]]
    exposures: NDArray[np.floating]
    shift_term: float
    residuals: NDArray[np.floating]
    path_probs: ProbVector | None
    r2: float | None


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


def _build_factor_ols_equation(
    factors_cum_forecast: dict[str, NDArray],
    portfolio_cum_forecast: NDArray[np.floating],
    eq_type: EquationTypes = "c",
) -> OLSEquation:
    independent_vars = np.column_stack(list(factors_cum_forecast.values()))
    dependent_var = portfolio_cum_forecast.reshape(-1, 1)
    if eq_type != "nc":
        independent_vars = add_deterministics_to_eq(
            independent_vars=independent_vars, eq_type=eq_type
        )
    return OLSEquation(ind_var=independent_vars, dep_vars=dependent_var)


def factor_ols_regression(
    factors_cum_forecast: dict[str, NDArray[np.floating]],
    portfolio_cum_forecast: NDArray[np.floating],
    prob: ProbVector | None = None,
    eq_type: EquationTypes = "c",
) -> OLSResults:
    ols_eq = _build_factor_ols_equation(
        factors_cum_forecast=factors_cum_forecast,
        portfolio_cum_forecast=portfolio_cum_forecast,
        eq_type=eq_type,
    )
    return weighted_ols(
        dependent_var=ols_eq.dep_vars,
        independent_vars=ols_eq.ind_var,
        prob=prob,
    )


def _extract_ols_components(
    ols_results: OLSResults,
    n_factors: int,
    eq_type: EquationTypes,
) -> tuple[float, NDArray[np.floating]]:
    """Split OLS coefficients into (shift_term, exposures)."""
    n_deterministics = {"nc": 0, "c": 1, "ct": 2, "ctt": 3}[
        eq_type
    ]  # ie which column idx
    coeffs = ols_results.res.flatten()

    if n_deterministics == 0:
        shift_term = 0.0
    else:
        shift_term = float(coeffs[0])

    exposures = coeffs[n_deterministics : n_deterministics + n_factors]
    return shift_term, exposures


def portfolio_factor_attribution(
    portfolio_forecast: PortfolioForecast,
    factors_forecast: dict[str, NDArray[np.floating]],
    original_data: DataFrame,
    horizon: int,
    factor_names: list[str] | None = None,
    eq_type: EquationTypes = "c",
    is_log_price: bool = True,
) -> HorizonFactorAttribution:
    if factor_names is None:
        factor_names = list(factors_forecast.keys())

    factors_cum = _factors_n_horizon_performance(
        factors_forecast=factors_forecast,
        original_data=original_data,
        factors_names=factor_names,
        end_horizon=horizon,
        is_log_price=is_log_price,
    )

    portfolio_cum = portfolio_forecast.cumulative_pnl(at_horizon=horizon)

    ols_results = factor_ols_regression(
        factors_cum_forecast=factors_cum,
        portfolio_cum_forecast=portfolio_cum,
        prob=portfolio_forecast.path_probs,
        eq_type=eq_type,
    )

    shift_term, exposures = _extract_ols_components(
        ols_results=ols_results,
        n_factors=len(factor_names),
        eq_type=eq_type,
    )

    return HorizonFactorAttribution(
        horizon=horizon,
        factor_names=factor_names,
        portfolio_performance_forecast=portfolio_cum,
        factor_performance_forecast=factors_cum,
        exposures=exposures,
        shift_term=shift_term,
        residuals=ols_results.residuals.flatten(),
        path_probs=portfolio_forecast.path_probs,
        r2=ols_results.r_squared,
    )

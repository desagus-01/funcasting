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
from time_series.feature_selection import (
    Criterion,
    ForwardRegressionResult,
    forward_regression,
)


@dataclass(frozen=True, slots=True)
class FactorOLSResult:
    ols: OLSResults
    selected_factors: list[str]
    selection_result: ForwardRegressionResult | None


@dataclass(frozen=True, slots=True)
class PortfolioFactorAttribution:
    horizon: int
    portfolio_performance_forecast: NDArray[np.floating]
    factor_performance_forecast: dict[str, NDArray[np.floating]]
    exposures: NDArray[np.floating]
    shift_term: float
    residuals: NDArray[np.floating]
    path_probs: ProbVector
    r2: float

    @property
    def factor_names(self) -> list[str]:
        return list(self.factor_performance_forecast.keys())

    @property
    def full_exposures(self) -> dict[str, float]:
        exposures_dict: dict[str, float] = {
            name: float(self.exposures[i]) for i, name in enumerate(self.factor_names)
        }
        exposures_dict["z0"] = 1.0
        return exposures_dict

    @property
    def joint_distribution(self) -> DataFrame:
        return DataFrame(self.factor_performance_forecast).with_columns(
            z0=self.residuals + self.shift_term
        )


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


def _deterministic_names(eq_type: EquationTypes) -> list[str]:
    if eq_type == "nc":
        return []
    names = ["const"]
    if eq_type in ("ct", "ctt"):
        names.append("trend")
    if eq_type == "ctt":
        names.append("trend_sq")
    return names


def factor_ols_regression(
    factors_cum_forecast: dict[str, NDArray[np.floating]],
    portfolio_cum_forecast: NDArray[np.floating],
    factor_names: list[str],
    auto_select_factors: bool = False,
    criterion: Criterion | None = None,
    prob: ProbVector | None = None,
    eq_type: EquationTypes = "c",
) -> FactorOLSResult:
    if (auto_select_factors) and criterion is None:
        raise ValueError(
            "You must select a criterion if you wish for auto factor selection."
        )

    ols_eq = _build_factor_ols_equation(
        factors_cum_forecast=factors_cum_forecast,
        portfolio_cum_forecast=portfolio_cum_forecast,
        eq_type=eq_type,
    )

    det_names = _deterministic_names(eq_type)
    full_names = det_names + factor_names

    if auto_select_factors and criterion is not None:
        fwd_result = forward_regression(
            dependent_var=ols_eq.dep_vars,
            independent_vars=ols_eq.ind_var,
            feature_names=full_names,
            criterion=criterion,
            prob=prob,
        )
        selected = [n for n in fwd_result.selected_features if n not in det_names]
        return FactorOLSResult(
            ols=fwd_result.final_model,
            selected_factors=selected,
            selection_result=fwd_result,
        )
    # ie runs a normal ols on ALL features
    ols_result = weighted_ols(
        dependent_var=ols_eq.dep_vars,
        independent_vars=ols_eq.ind_var,
        feature_names=full_names,
        prob=prob,
    )
    return FactorOLSResult(
        ols=ols_result,
        selected_factors=factor_names,
        selection_result=None,
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
    auto_select_factors: bool = False,
    criterion: Criterion | None = None,
) -> PortfolioFactorAttribution:
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

    factor_result = factor_ols_regression(
        factors_cum_forecast=factors_cum,
        portfolio_cum_forecast=portfolio_cum,
        factor_names=factor_names,
        auto_select_factors=auto_select_factors,
        criterion=criterion,
        prob=portfolio_forecast.path_probs,
        eq_type=eq_type,
    )

    selected = factor_result.selected_factors
    ols = factor_result.ols

    shift_term, exposures = _extract_ols_components(
        ols_results=ols,
        n_factors=len(selected),
        eq_type=eq_type,
    )

    return PortfolioFactorAttribution(
        horizon=horizon,
        portfolio_performance_forecast=portfolio_cum,
        factor_performance_forecast={
            k: v for k, v in factors_cum.items() if k in selected
        },
        exposures=exposures,
        shift_term=shift_term,
        residuals=ols.residuals.flatten(),
        path_probs=portfolio_forecast.path_probs,
        r2=ols.r_squared,
    )

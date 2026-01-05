import re
from dataclasses import dataclass
from typing import Literal, NamedTuple

from polars.dataframe.frame import DataFrame

from maths.helpers import add_detrend_column
from maths.stochastic_processes.estimation import EquationTypes
from maths.stochastic_processes.stationarity_tests import (
    StationarityInference,
    stationarity_tests,
)


class TrendRes(NamedTuple):
    """Result for one asset and one polynomial order."""

    asset: str
    order_type: Literal["polynomial", "difference"]
    order: int
    stationarity_inf: str
    full_test_res: StationarityInference


@dataclass(frozen=True)
class TrendTest:
    """Summary of deterministic-trend evidence for one asset."""

    trend_type: Literal["deterministic", "stochastic"]
    trend_stationary: bool
    lowest_order_stationary: int | None
    results: list[TrendRes]


def is_stationary_result(res: TrendRes) -> bool:
    """True if stationarity inference is 'stationary'."""
    return res.stationarity_inf == "stationary"


def lowest_polynomial_trend(results: list[TrendRes]) -> int | None:
    """Return the lowest polynomial order that yields stationarity, else None."""
    orders = [r.order for r in results if is_stationary_result(r)]
    return min(orders) if orders else None


def _detrended_col(asset: str, p: int) -> str:
    """Column name for detrended series."""
    return f"{asset}_detrended_p{p}"


def _parse_poly_order(col: str) -> int:
    """Parse polynomial order from a detrended column name."""
    m = re.search(r"_p(\d+)$", col)
    if not m:
        raise ValueError(f"Can't parse polynomial order from column: {col}")
    return int(m.group(1))


def _stationarity_for_col(
    df: DataFrame,
    col: str,
    *,
    lags: int,
    eq_type: EquationTypes,
) -> StationarityInference:
    """Run stationarity tests for one column."""
    if col not in df.columns:
        raise ValueError(
            f"Expected detrended column '{col}' not found. "
            "Check add_detrend_column naming."
        )
    return stationarity_tests(data=df, asset=col, lags=lags, eq_type=eq_type)


def _asset_dtrend_results(
    df: DataFrame,
    asset: str,
    polynomial_orders: list[int],
    *,
    lags: int,
    eq_type: EquationTypes,
) -> list[TrendRes]:
    """Compute DTrendRes list for one asset across polynomial orders."""
    res: list[TrendRes] = []
    for p in polynomial_orders:
        col = _detrended_col(asset, p)
        inf = _stationarity_for_col(df, col, lags=lags, eq_type=eq_type)
        res.append(
            TrendRes(
                asset=asset,
                order_type="polynomial",
                order=_parse_poly_order(col),
                stationarity_inf=inf.label,
                full_test_res=inf,
            )
        )
    return res


def test_deterministic_trend(
    data: DataFrame,
    assets: list[str],
    polynomial_orders: list[int] | None = None,
    eq_type: EquationTypes = "c",
    lags: int = 10,
    max_order_accept: int = 1,
) -> dict[str, TrendTest]:
    """
    Detrend with polynomial orders and test residual stationarity.

    Returns per-asset decisions; accepts only if best order <= max_order_accept.
    """
    if polynomial_orders is None:
        polynomial_orders = [0, 1, 2, 3]

    detrended_df = add_detrend_column(
        data=data, assets=assets, polynomial_orders=polynomial_orders
    )

    d_trend_res: dict[str, TrendTest] = {}
    for asset in assets:
        results = _asset_dtrend_results(
            detrended_df, asset, polynomial_orders, lags=lags, eq_type=eq_type
        )
        lowest = lowest_polynomial_trend(results)
        evidence = (lowest is not None) and (lowest <= max_order_accept)
        d_trend_res[asset] = TrendTest(
            trend_type="deterministic",
            trend_stationary=evidence,
            lowest_order_stationary=lowest,
            results=results,
        )

    return d_trend_res

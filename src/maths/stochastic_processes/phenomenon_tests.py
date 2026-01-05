import re
from dataclasses import dataclass
from typing import NamedTuple

from polars.dataframe.frame import DataFrame

from maths.helpers import add_detrend_column
from maths.stochastic_processes.estimation import EquationTypes
from maths.stochastic_processes.stationarity_tests import (
    StationarityInference,
    stationarity_tests,
)


class DTrendRes(NamedTuple):
    """Result for one asset and one polynomial order."""

    asset: str
    polynomial_order: int
    stationarity_inf: str
    full_test_res: StationarityInference


@dataclass(frozen=True)
class DTrendTest:
    """Summary of deterministic-trend evidence for one asset."""

    evidence_of_deterministic_trend: bool
    lowest_polynomial_stationary: int | None
    results: list[DTrendRes]


def is_stationary_result(res: DTrendRes) -> bool:
    """True if stationarity inference is 'stationary'."""
    return res.stationarity_inf == "stationary"


def lowest_polynomial_trend(results: list[DTrendRes]) -> int | None:
    """Return the lowest polynomial order that yields stationarity, else None."""
    orders = [r.polynomial_order for r in results if is_stationary_result(r)]
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
) -> list[DTrendRes]:
    """Compute DTrendRes list for one asset across polynomial orders."""
    res: list[DTrendRes] = []
    for p in polynomial_orders:
        col = _detrended_col(asset, p)
        inf = _stationarity_for_col(df, col, lags=lags, eq_type=eq_type)
        res.append(
            DTrendRes(
                asset=asset,
                polynomial_order=_parse_poly_order(col),
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
) -> dict[str, DTrendTest]:
    """
    Detrend with polynomial orders and test residual stationarity.

    Returns per-asset decisions; accepts only if best order <= max_order_accept.
    """
    if polynomial_orders is None:
        polynomial_orders = [0, 1, 2, 3]

    detrended_df = add_detrend_column(
        data=data, assets=assets, polynomial_orders=polynomial_orders
    )

    d_trend_res: dict[str, DTrendTest] = {}
    for asset in assets:
        results = _asset_dtrend_results(
            detrended_df, asset, polynomial_orders, lags=lags, eq_type=eq_type
        )
        lowest = lowest_polynomial_trend(results)
        evidence = (lowest is not None) and (lowest <= max_order_accept)
        d_trend_res[asset] = DTrendTest(evidence, lowest, results)

    return d_trend_res

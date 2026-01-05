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
    asset: str
    polynomial_order: int
    stationarity_inf: str
    full_test_res: StationarityInference


@dataclass(frozen=True)
class DTrendTest:
    evidence_of_deterministic_trend: bool
    lowest_polynomial_stationary: int | None
    results: list[DTrendRes]


def is_stationary_result(res: DTrendRes) -> bool:
    return res.stationarity_inf == "stationary"


def lowest_polynomial_trend(results: list[DTrendRes]) -> int | None:
    orders = [r.polynomial_order for r in results if is_stationary_result(r)]
    return min(orders) if orders else None


def test_deterministic_trend(
    data: DataFrame,
    assets: list[str],
    polynomial_orders: list[int] | None = None,
    eq_type: EquationTypes = "c",
) -> DTrendTest:
    if polynomial_orders is None:
        polynomial_orders = [0, 1, 2, 3]

    detrended_df = add_detrend_column(
        data=data, assets=assets, polynomial_orders=polynomial_orders
    )

    detrended_cols = [c for c in detrended_df.columns if "detrended" in c]

    deterministic_stationary_res: list[DTrendRes] = []
    for col in detrended_cols:
        stationarity_res = stationarity_tests(
            data=detrended_df, asset=col, lags=10, eq_type=eq_type
        )

        m = re.search(r"(\d+)$", col)
        if not m:
            raise ValueError(f"Can't parse polynomial order from column: {col}")
        polynomial_order = int(m.group(1))

        deterministic_stationary_res.append(
            DTrendRes(
                asset=col,
                polynomial_order=polynomial_order,
                stationarity_inf=stationarity_res.label,
                full_test_res=stationarity_res,
            )
        )

    lowest = lowest_polynomial_trend(deterministic_stationary_res)
    evidence = lowest is not None

    return DTrendTest(
        evidence_of_deterministic_trend=evidence,
        lowest_polynomial_stationary=lowest,
        results=deterministic_stationary_res,
    )

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple

from polars.dataframe.frame import DataFrame

from maths.helpers import add_detrend_columns_max, add_differenced_columns
from maths.stochastic_processes.estimation import EquationTypes
from maths.stochastic_processes.stationarity_tests import (
    StationarityInference,
    stationarity_tests,
)

transform_types = Literal["polynomial", "difference"]
trend_types = Literal["deterministic", "stochastic"]


class TrendRes(NamedTuple):
    """Result for one asset and one polynomial order."""

    asset: str
    transform_type: transform_types
    order: int
    stationarity_inf: str
    full_test_res: StationarityInference


@dataclass(frozen=True)
class TrendTest:
    """Summary of deterministic-trend evidence for one asset."""

    trend_type: trend_types
    trend_stationary: bool
    lowest_order_threshold: int
    lowest_order_stationary: int | None
    results: list[TrendRes]


def lowest_stationary_order(results: list[TrendRes]) -> int | None:
    """
    Returns the smallest rested order labeled 'stationary', else None
    """
    for res in sorted(results, key=lambda x: x.order):
        if res.stationarity_inf == "stationary":
            return res.order
    return None


ColBuilder = Callable[[str, int], str]
AddColumnsFn = Callable[[DataFrame, list[str], int], DataFrame]


@dataclass(frozen=True)
class TrendTransformSpec:
    trend_type: trend_types
    transform_type: transform_types
    add_columns: AddColumnsFn
    col_builder: ColBuilder

    @classmethod
    def deterministic(cls) -> "TrendTransformSpec":
        return cls(
            trend_type="deterministic",
            transform_type="polynomial",
            add_columns=add_detrend_columns_max,
            col_builder=lambda a, p: f"{a}_detrended_p{p}",
        )

    @classmethod
    def stochastic(cls) -> "TrendTransformSpec":
        return cls(
            trend_type="stochastic",
            transform_type="difference",
            add_columns=add_differenced_columns,
            col_builder=lambda a, d: f"{a}_diff_{d}",
        )


@dataclass(frozen=True)
class OrderSpec:
    orders: list[int]
    max_accept: int

    @classmethod
    def polynomial(cls, *, max_order: int, max_accept: int) -> "OrderSpec":
        return cls(orders=list(range(0, max_order + 1)), max_accept=max_accept)

    @classmethod
    def difference(cls, *, max_order: int, max_accept: int) -> "OrderSpec":
        return cls(orders=list(range(1, max_order + 1)), max_accept=max_accept)


@dataclass(frozen=True)
class StationarityRunner:
    lags: int
    eq_type: EquationTypes

    def __call__(self, df: DataFrame, col: str) -> StationarityInference:
        return stationarity_tests(
            data=df.select(col).drop_nulls(),
            asset=col,
            lags=self.lags,
            eq_type=self.eq_type,
        )

    @classmethod
    def trend(cls) -> "StationarityRunner":
        return cls(lags=10, eq_type="c")


def run_stationary(
    data: DataFrame,
    asset: str,
    orders: list[int],
    *,
    transform_type: transform_types,
    col_builder: ColBuilder,
    run_stationary: StationarityRunner,
):
    res: list[TrendRes] = []

    for order in orders:
        column = col_builder(asset, order)

        if column not in data.columns:
            raise ValueError(f"Column {column} not found in data {data.columns}")

        stationary_res = run_stationary(data, column)

        res.append(
            TrendRes(
                asset=asset,
                transform_type=transform_type,
                order=order,
                stationarity_inf=stationary_res.label,
                full_test_res=stationary_res,
            )
        )

    return res


def _run_trend_diagnostic(
    *,
    data: DataFrame,
    assets: list[str],
    transform: TrendTransformSpec,
    order_spec: OrderSpec,
    runner: StationarityRunner,
):
    transformed_df = transform.add_columns(data, assets, max(order_spec.orders))

    asset_trend_res: dict[str, TrendTest] = {}

    for asset in assets:
        results = run_stationary(
            transformed_df,
            asset,
            order_spec.orders,
            transform_type=transform.transform_type,
            col_builder=transform.col_builder,
            run_stationary=runner,
        )

        lowest_order = lowest_stationary_order(results)

        is_stationary = (
            lowest_order is not None and lowest_order <= order_spec.max_accept
        )

        asset_trend_res[asset] = TrendTest(
            trend_type=transform.trend_type,
            trend_stationary=is_stationary,
            lowest_order_threshold=order_spec.max_accept,
            lowest_order_stationary=lowest_order,
            results=results,
        )
    return asset_trend_res


def trend_diagnostic(
    data: DataFrame,
    assets: list[str],
    order_max: int,
    threshold_order: int,
    *,
    trend_type: Literal["deterministic", "stochastic", "both"],
):
    runner = StationarityRunner.trend()

    if trend_type == "deterministic":
        return _run_trend_diagnostic(
            data=data,
            assets=assets,
            transform=TrendTransformSpec.deterministic(),
            order_spec=OrderSpec.polynomial(
                max_order=order_max, max_accept=threshold_order
            ),
            runner=runner,
        )

    if trend_type == "stochastic":
        return _run_trend_diagnostic(
            data=data,
            assets=assets,
            transform=TrendTransformSpec.stochastic(),
            order_spec=OrderSpec.difference(
                max_order=order_max, max_accept=threshold_order
            ),
            runner=runner,
        )

    # both
    return {
        "deterministic": _run_trend_diagnostic(
            data=data,
            assets=assets,
            transform=TrendTransformSpec.deterministic(),
            order_spec=OrderSpec.polynomial(
                max_order=order_max, max_accept=threshold_order
            ),
            runner=runner,
        ),
        "stochastic": _run_trend_diagnostic(
            data=data,
            assets=assets,
            transform=TrendTransformSpec.stochastic(),
            order_spec=OrderSpec.difference(
                max_order=order_max, max_accept=threshold_order
            ),
            runner=runner,
        ),
    }

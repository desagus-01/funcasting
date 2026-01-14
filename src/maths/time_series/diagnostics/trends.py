from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from polars.dataframe.frame import DataFrame

from maths.helpers import add_detrend_columns_max, add_differenced_columns
from maths.time_series.estimation import EquationTypes
from maths.time_series.stationarity_tests import (
    StationarityInference,
    stationarity_tests,
)

transform_types = Literal["polynomial", "difference"]
trend_types = Literal["deterministic", "stochastic"]


@dataclass(frozen=True)
class TrendRes:
    asset: str
    transform_type: transform_types
    order: int
    stationarity_inf: str
    full_test_res: StationarityInference = field(repr=False)


@dataclass(frozen=True)
class TrendTest:
    trend_type: trend_types
    transformation_needed: bool
    lowest_order_threshold: int
    transformation_order_needed: int | None
    results: list[TrendRes]

    @property
    def description(self) -> str:
        if self.transformation_order_needed is None:
            return (
                f"No stationary transform found up to order {self.lowest_order_threshold} "
                f"for {self.trend_type}."
            )
        if self.transformation_needed:
            return (
                f"Transformation needed: {self.trend_type} "
                f"({self.results[0].transform_type}) of order "
                f"{self.transformation_order_needed}."
            )
        return (
            f"Stationarity achieved at order {self.transformation_order_needed}, "
            f"but exceeds acceptance threshold (tested up to {self.lowest_order_threshold})."
        )


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


def lowest_stationary_order(results: list[TrendRes]) -> int | None:
    for res in sorted(results, key=lambda x: x.order):
        if res.stationarity_inf == "stationary":
            return res.order
    return None


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

        transformation_order_needed = lowest_stationary_order(results)
        transformation_needed = (
            transformation_order_needed is not None
            and transformation_order_needed <= order_spec.max_accept
        )

        asset_trend_res[asset] = TrendTest(
            trend_type=transform.trend_type,
            transformation_needed=transformation_needed,
            lowest_order_threshold=order_spec.max_accept,
            transformation_order_needed=transformation_order_needed,
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
) -> dict[str, dict[str, TrendTest]]:
    runner = StationarityRunner.trend()

    out: dict[str, dict[str, TrendTest]] = {"deterministic": {}, "stochastic": {}}

    if trend_type in ("deterministic", "both"):
        out["deterministic"] = _run_trend_diagnostic(
            data=data,
            assets=assets,
            transform=TrendTransformSpec.deterministic(),
            order_spec=OrderSpec.polynomial(
                max_order=order_max, max_accept=threshold_order
            ),
            runner=runner,
        )

    if trend_type in ("stochastic", "both"):
        out["stochastic"] = _run_trend_diagnostic(
            data=data,
            assets=assets,
            transform=TrendTransformSpec.stochastic(),
            order_spec=OrderSpec.difference(
                max_order=order_max, max_accept=threshold_order
            ),
            runner=runner,
        )
    return out

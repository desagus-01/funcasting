from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from polars.dataframe.frame import DataFrame

from maths.helpers import add_detrend_columns_max, add_differenced_columns
from maths.time_series.stationarity_tests import (
    StationarityInference,
    stationarity_tests,
)

TransformTypes = Literal["polynomial", "difference"]
TrendTypes = Literal["deterministic", "stochastic"]

ColBuilder = Callable[[str, int], str]
AddColumnsFn = Callable[[DataFrame, list[str], int], DataFrame]


@dataclass(frozen=True)
class TrendRes:
    asset: str
    transform_type: TransformTypes
    order: int
    stationarity_inf: str
    full_test_res: StationarityInference = field(repr=False)


@dataclass(frozen=True)
class TrendTest:
    trend_type: TrendTypes
    lowest_order_threshold: int
    transformation_order_needed: int | None
    results: list[TrendRes]

    @property
    def transformation_needed(self) -> bool:
        return (
            self.transformation_order_needed is not None
            and self.transformation_order_needed <= self.lowest_order_threshold
        )

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


@dataclass(frozen=True)
class StationarityRunner:
    lags: int

    def __call__(self, df: DataFrame, col: str) -> StationarityInference:
        return stationarity_tests(
            data=df.select(col).drop_nulls(),
            asset=col,
            lags=self.lags,
        )

    @classmethod
    def trend(cls) -> "StationarityRunner":
        return cls(lags=10)


TRANSFORMS: dict[TrendTypes, dict] = {
    "deterministic": {
        "transform_type": "polynomial",
        "add_columns": add_detrend_columns_max,
        "col_builder": lambda a, p: f"{a}_detrended_p{p}",
        "order_range": lambda max_order: range(0, max_order + 1),
    },
    "stochastic": {
        "transform_type": "difference",
        "add_columns": add_differenced_columns,
        "col_builder": lambda a, d: f"{a}_diff_{d}",
        "order_range": lambda max_order: range(1, max_order + 1),
    },
}


def _lowest_stationary_order(results: list[TrendRes]) -> int | None:
    for res in sorted(results, key=lambda x: x.order):
        if res.stationarity_inf == "stationary":
            return res.order
    return None


def run_stationary(
    data: DataFrame,
    asset: str,
    orders: list[int],
    *,
    transform_type: TransformTypes,
    col_builder: ColBuilder,
    runner: StationarityRunner,
):
    res: list[TrendRes] = []

    for order in orders:
        column = col_builder(asset, order)

        if column not in data.columns:
            raise ValueError(f"Column {column} not found in data {data.columns}")

        stationary_res = runner(data, column)

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
    trend_type: TrendTypes,
    order_max: int,
    threshold_order: int,
    runner: StationarityRunner,
) -> dict[str, TrendTest]:
    spec = TRANSFORMS[trend_type]
    orders: list[int] = spec["order_range"](order_max)

    transformed_df = spec["add_columns"](data, assets, max(orders))
    col_builder: ColBuilder = spec["col_builder"]
    transform_type: TransformTypes = spec["transform_type"]

    asset_trend_res: dict[str, TrendTest] = {}

    for asset in assets:
        results: list[TrendRes] = []
        for order in orders:
            col = col_builder(asset, order)
            if col not in transformed_df.columns:
                raise ValueError(
                    f"Column {col} not found in data {transformed_df.columns}"
                )

            inf = runner(transformed_df, col)
            results.append(
                TrendRes(
                    asset=asset,
                    transform_type=transform_type,
                    order=order,
                    stationarity_inf=inf.label,
                    full_test_res=inf,
                )
            )

        stationary_order = _lowest_stationary_order(results)
        asset_trend_res[asset] = TrendTest(
            trend_type=trend_type,
            lowest_order_threshold=threshold_order,
            transformation_order_needed=stationary_order,
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
            trend_type="deterministic",
            order_max=order_max,
            threshold_order=threshold_order,
            runner=runner,
        )

    if trend_type in ("stochastic", "both"):
        out["stochastic"] = _run_trend_diagnostic(
            data=data,
            assets=assets,
            trend_type="stochastic",
            order_max=order_max,
            threshold_order=threshold_order,
            runner=runner,
        )
    return out

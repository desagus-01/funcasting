from typing import Tuple

import polars as pl

from utils.helpers import wide_to_long


def get_latest_prices(data_long: pl.DataFrame) -> pl.DataFrame:
    return data_long.group_by("ticker").agg(
        pl.col("adj_close").last().alias("adj_close")
    )


def equal_weight_shares_from_prices(
    latest_price: pl.DataFrame, initial_value: float, n_assets: int | None = None
) -> pl.DataFrame:
    if n_assets is None:
        n_assets = latest_price.height

    assigned = initial_value / n_assets

    eq_portfolio = (
        latest_price.with_columns(
            # nominal shares (floor to integer shares)
            (pl.lit(assigned) / pl.col("adj_close"))
            .floor()
            .cast(pl.Int64)
            .alias("shares")
        )
        .with_columns((pl.col("shares") * pl.col("adj_close")).alias("value_allocated"))
        .with_columns(
            (pl.col("value_allocated") / pl.lit(initial_value)).alias(
                "pct_of_portfolio"
            ),
            (pl.lit(1 / n_assets)).alias("target_pct"),
        )
        .select(
            [
                "ticker",
                "adj_close",
                "shares",
                "value_allocated",
                "pct_of_portfolio",
                "target_pct",
            ]
        )
    )

    return eq_portfolio


def portfolio_value(positions: pl.DataFrame) -> Tuple[float, pl.DataFrame]:
    if "value_allocated" not in positions.columns:
        positions = positions.with_columns(
            (pl.col("adj_close") * pl.col("shares")).alias("value_allocated")
        )

    total = float(positions.select(pl.col("value_allocated").sum()).to_series()[0])

    # avoid division by zero
    if total == 0:
        positions = positions.with_columns(pl.lit(0.0).alias("pct_of_portfolio"))
    else:
        positions = positions.with_columns(
            (pl.col("value_allocated") / pl.lit(total)).alias("pct_of_portfolio")
        )

    return total, positions


def build_equal_weight_portfolio_from_df(
    data: pl.DataFrame, initial_value: float, assets: list[str] | None = None
) -> Tuple[pl.DataFrame, float]:
    cols = set(data.columns)

    # detect long format
    if {"ticker", "adj_close"}.issubset(cols):
        data_long = (
            data.with_columns(pl.col("date").cast(pl.Date)) if "date" in cols else data
        )
        if assets is None:
            assets = list(
                data_long.select(pl.col("ticker")).unique().to_series().to_list()
            )
    else:
        # treat as wide
        if assets is None:
            assets = [c for c in data.columns if c != "date"]
        data_long = wide_to_long(data, assets)

    latest_price = get_latest_prices(data_long)
    eq_portfolio = equal_weight_shares_from_prices(
        latest_price, initial_value, n_assets=len(assets)
    )
    total, positions = portfolio_value(eq_portfolio)

    return positions, total

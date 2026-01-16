import polars as pl
import polars.selectors as cs

from maths.time_series.operations import deterministic_detrend


def add_detrend_column(
    data: pl.DataFrame,
    assets: list[str] | None = None,
    polynomial_orders: list[int] = [0, 1, 2, 3],
    axis: int = 0,
) -> pl.DataFrame:
    if assets is None:
        assets = data.select(
            cs.numeric()
        ).columns  # only get numeric columns (ie no dates)

    asset_arrays = data.select(assets).to_numpy()

    new_cols: list[pl.Series] = []
    for p in polynomial_orders:
        detrended = deterministic_detrend(asset_arrays, polynomial_order=p, axis=axis)

        for i, asset in enumerate(assets):
            new_cols.append(
                pl.Series(
                    name=f"{asset}_detrended_p{p}",
                    values=detrended[:, i],
                ).cast(pl.Float64)
            )

    return data.with_columns(new_cols)


def add_detrend_columns_max(
    data: pl.DataFrame,
    assets: list[str],
    max_polynomial_order: int,
) -> pl.DataFrame:
    polynomial_orders = list(range(0, max_polynomial_order + 1))
    return add_detrend_column(
        data=data, assets=assets, polynomial_orders=polynomial_orders
    )


def add_differenced_columns(
    data: pl.DataFrame,
    assets: list[str],
    difference: int = 1,
    keep_all: bool = True,
) -> pl.DataFrame:
    if difference < 1:
        raise ValueError("difference must be >= 1")

    diffs = range(1, difference + 1) if keep_all else [difference]

    return data.with_columns(
        [pl.col(assets).diff(d).name.suffix(f"_diff_{d}") for d in diffs]
    )

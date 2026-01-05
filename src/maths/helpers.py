import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray


def deterministic_detrend(
    data: NDArray[np.floating], polynomial_order: int = 1, axis: int = 0
) -> NDArray[np.floating]:
    """
    Fits a deterministic polynomial trend and then subtracts it from the data
    """
    if data.ndim == 2 and int(axis) == 1:
        data = data.T
    elif data.ndim > 2:
        raise NotImplementedError("data.ndim > 2 is not implemented until it is needed")

    if polynomial_order == 0:  # Special case, just de-mean
        resid = data - data.mean(axis=0)
    else:
        trends = np.vander(np.arange(float(data.shape[0])), N=polynomial_order + 1)
        beta = np.linalg.pinv(trends).dot(data)
        resid = data - np.dot(trends, beta)

    if data.ndim == 2 and int(axis) == 1:
        resid = resid.T

    return resid


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

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

    if polynomial_order == 0:
        # Special case demean
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
    polynomial_order: int = 1,
    axis: int = 0,
) -> pl.DataFrame:
    if assets is None:
        assets = data.select(
            cs.numeric()
        ).columns  # only get numeric columns (ie no dates)

    detrended_series = deterministic_detrend(
        data=data.select(assets).to_numpy(),
        polynomial_order=polynomial_order,
        axis=axis,
    )

    new_cols = [
        pl.Series(name=f"{asset}_detrended", values=detrended_series[:, i]).cast(
            pl.Float64
        )
        for i, asset in enumerate(assets)
    ]

    return data.with_columns(new_cols)

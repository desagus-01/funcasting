import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from polars._typing import RankMethod


def pseudo_observations(
    data: pl.DataFrame, method: RankMethod = "average"
) -> pl.DataFrame | NDArray[np.floating]:
    return data.select(cs.numeric().rank(method)) / (
        len(data) + 1
    )  # we normalise so that it is between 0 and 1


def emp_cdf(data: pl.DataFrame, method: RankMethod = "average") -> pl.DataFrame:
    return data.select(cs.numeric().rank(method, descending=True))

import numpy as np
import polars as pl
from numpy.typing import NDArray

from globals import sign_operations
from models.prob import ProbVector
from models.views import CorrInfo, View


def weighted_moments(
    data: NDArray[np.floating], weights: ProbVector
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    avg = np.average(data, axis=1, weights=weights)
    var = np.average((data.T - avg) ** 2, axis=0, weights=weights)
    return avg, np.sqrt(var)


def select_operator(views: View):
    return sign_operations[views.sign_type]


def get_corr_info(data: pl.DataFrame) -> list[CorrInfo]:
    corr_df = (
        data.corr()
        .with_columns(index=pl.lit(pl.Series(data.columns)))
        .unpivot(index="index")
        .filter(pl.col("index") != pl.col("variable"))
        .with_columns(
            pair=pl.when(pl.col("index") < pl.col("variable"))
            .then(pl.concat_str([pl.col("index"), pl.col("variable")], separator="-"))
            .otherwise(
                pl.concat_str([pl.col("variable"), pl.col("index")], separator="-")
            )
        )
        .unique(subset=["pair"])
        .drop("pair")
    )

    return [
        CorrInfo(asset_pair=(corrs["index"], corrs["variable"]), corr=corrs["value"])
        for corrs in corr_df.rows(named=True)
    ]

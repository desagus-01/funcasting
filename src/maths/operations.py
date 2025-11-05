import numpy as np
import polars as pl
import polars.selectors as cs
from polars._typing import RankMethod
from scipy.stats import ecdf

from helpers import NDArray, ProbVector


def prior_cdf(
    data: pl.DataFrame,
    prob_vector: ProbVector,
    method: RankMethod = "average",
) -> pl.DataFrame:
    """
    Returns the quantiles, probs, and cumulative probs for a given data
    """

    if len(data.columns) == 1:  # Run as if marginal dist
        return (
            data.with_columns(prob=prob_vector)
            .sort(data.columns[0])
            .with_columns(cum_prob=pl.col("prob").cum_sum())
        )
    # TODO: Fix this for joint, currently does not work
    else:
        return data.with_columns(prob=prob_vector).select(
            cs.date(), cs.numeric().rank(method)
        )


def emp_cdf(data: NDArray[np.floating]):
    return ecdf(data).cdf


def indicator_quantile_marginal(
    data: pl.DataFrame, target_quantile: float
) -> pl.DataFrame:
    threshold = np.quantile(data, target_quantile)

    return data.with_columns(quant_ind=(cs.numeric() <= threshold).cast(pl.Int8))


# def pseudo_observations(
#     data: pl.DataFrame | NDArray[np.floating], method: str = "average"
# ) -> pl.DataFrame | NDArray[np.floating]:
#     return rank_data(data, method) / (len(data) + 1)

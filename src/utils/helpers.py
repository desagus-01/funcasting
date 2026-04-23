import logging
import time
from functools import wraps
from typing import NamedTuple

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from pydantic import validate_call

from globals import model_cfg, sign_operations
from scenarios.types import CorrInfo, ProbVector, View

logger = logging.getLogger(__name__)


class SplitDF(NamedTuple):
    first_half: pl.DataFrame
    second_half: pl.DataFrame


def split_df_in_half(data: pl.DataFrame) -> SplitDF:
    height = data.height

    if height % 2 != 0:
        height -= 1
        data = data.slice(0, height)

    mid = height // 2
    first_half = data.slice(0, mid)
    second_half = data.slice(mid, mid)

    return SplitDF(first_half, second_half)


def get_assets_names(df: pl.DataFrame, assets: list[str] | None = None) -> list[str]:
    """
    Retrieve asset column names from a DataFrame.

    If ``assets`` is provided the function validates and returns that list
    (raising if columns are missing by virtue of Polars selection semantics).
    If ``assets`` is ``None`` the function returns all columns except 'date'.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing asset columns (and possibly a 'date' column).
    assets : list[str] | None, optional
        Explicit list of asset column names to validate and return.

    Returns
    -------
    list[str]
        Asset column names.
    """
    if assets is None:
        return [c for c in df.columns if c != "date"]
    return df.select(assets).columns


def weighted_moments(
    data: NDArray[np.floating], weights: ProbVector
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    avg = np.average(data, axis=1, weights=weights)
    var = np.average((data.T - avg) ** 2, axis=0, weights=weights)
    return avg, np.sqrt(var)


def select_operator(views: View):
    return sign_operations[views.sign_type]


def get_corr_info(data: pl.DataFrame) -> list[CorrInfo]:
    """
    Compute pairwise correlations and return a list of CorrInfo named tuples.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame whose numeric columns are used to compute pairwise
        Pearson correlations.

    Returns
    -------
    list[CorrInfo]
        A list where each item contains a tuple of asset names and their
        correlation value. Each unordered pair appears only once.
    """
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


def indicator_quantile_marginal(
    data: pl.DataFrame, target_quantile: float
) -> pl.DataFrame:
    threshold = np.quantile(data, target_quantile)

    return data.with_columns(quant_ind=(cs.numeric() <= threshold).cast(pl.Int8))


def compute_cdf_and_pobs(
    data: pl.DataFrame,
    marginal_name: str,
    prob: ProbVector,
    compute_pobs: bool = True,
) -> pl.DataFrame:
    """
    Compute empirical CDF and pseudo-observations for a single marginal.

    The function expects no missing values in ``data`` and will raise if any
    are present. It returns a DataFrame containing the sorted marginal, the
    cumulative probability (cdf) and, optionally, the pseudo-observations
    (pobs) aligned to the original row order.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the marginal column.
    marginal_name : str
        Column name of the marginal to process.
    prob : ProbVector
        Probability weights associated with each row; must sum to one.
    compute_pobs : bool, optional
        Whether to compute pseudo-observations aligned to original order
        (default: True).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ['index', marginal_name, 'prob', 'cdf'] and
        optionally 'pobs' when ``compute_pobs`` is True.

    Raises
    ------
    ValueError
        If any nulls are present in the input ``data``.
    """
    if data.null_count().sum_horizontal().item() > 0:
        raise ValueError(
            f"You have a total of {data.null_count().sum_horizontal().item()} Nulls in your data, please fix this."
        )
    df = (
        data.select(pl.col(marginal_name))
        .with_row_index()
        .with_columns(prob=prob)
        .sort(marginal_name)
        .with_columns(
            cdf=pl.cum_sum("prob") * data.height / (data.height + 1),
        )
    )

    if compute_pobs:
        df = df.with_columns(
            pobs=pl.col("cdf").gather(pl.col("index").arg_sort()),
        )

    return df


def build_diff_df(data: pl.DataFrame, asset: str, diffs: int = 1) -> pl.DataFrame:
    if diffs <= 0:
        raise ValueError("Diffs need to be bigger than 0 dummy")
    return data.select(
        asset,
        *[
            pl.col(asset).diff(i).alias(f"{asset}_diff_{i}")
            for i in range(1, diffs + 1)
        ],
    )


@validate_call(config=model_cfg, validate_return=True)
def compensate_prob(prob: ProbVector, n_remove: int) -> ProbVector:
    """
    Remove the top ``n_remove`` probabilities and re-normalize the remainder.

    This helper is used when rows are dropped (e.g. due to nulls) to
    compensate the prior probability vector by evenly distributing the
    removed mass across the remaining entries.

    Parameters
    ----------
    prob : ProbVector
        Original probability vector.
    n_remove : int
        Number of top entries to remove from the probability vector.

    Returns
    -------
    ProbVector
        Adjusted probability vector of length ``len(prob) - n_remove`` that
        sums to one.
    """
    removed_probs = prob[0:n_remove]
    diff_fac = removed_probs.sum() / (len(prob) - len(removed_probs))

    return prob[n_remove:] + diff_fac


def drop_nulls_and_compensate_prob(
    data: pl.DataFrame, prob_vector: ProbVector
) -> tuple[pl.DataFrame, ProbVector]:
    nulls = data.null_count().sum_horizontal().item()
    if nulls > 0:
        no_nulls = data.drop_nulls()
        rows_dropped = data.height - no_nulls.height
        prob_vector = compensate_prob(prob_vector, rows_dropped)
        return no_nulls, prob_vector
    return data, prob_vector


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s took %.4fs", func.__name__, elapsed)
        return result

    return wrapper


def wide_to_long(data: pl.DataFrame, assets: list[str]) -> pl.DataFrame:
    return data.unpivot(
        index="date",
        on=list(assets),
        variable_name="ticker",
        value_name="adj_close",
    ).with_columns(pl.col("date").cast(pl.Date))

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray

from globals import SIGN_LVL, sign_operations
from models.types import CorrInfo, ProbVector, View


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


def lag_df(data: pl.DataFrame, asset: str, lags: int) -> pl.DataFrame:
    return data.select(
        "date",
        asset,
        *[pl.col(asset).shift(i).alias(f"{asset}_lag_{i}") for i in range(1, lags + 1)],
    ).drop_nulls()


def hyp_test_conc(p_val: float, null_hyp: str) -> str:
    if p_val >= SIGN_LVL:
        return (
            f"Fail to reject null hypothesis of {null_hyp} at {SIGN_LVL} significance."
        )
    else:
        return f"Reject null hypothesis of {null_hyp} at {SIGN_LVL} significance."

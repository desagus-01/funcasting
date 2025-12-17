from typing import NamedTuple, TypedDict

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from pydantic import validate_call

from globals import LAGS, SIGN_LVL, model_cfg, sign_operations
from models.types import CorrInfo, ProbVector, View


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


def select_assets(df: pl.DataFrame, assets: list[str] | None) -> list[str]:
    """
    Retrieves and makes sures that assets exist in df, if None, chooses all assets ex date
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


def build_lag_df(
    data: pl.DataFrame, asset: str, lags: int = LAGS["testing"]
) -> pl.DataFrame:
    return data.select(
        asset,
        *[pl.col(asset).shift(i).alias(f"{asset}_lag_{i}") for i in range(1, lags + 1)],
    )


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
    Removes obs top-down from probability vectors and equally adds to others to remain a valid prob vector.
    """
    removed_probs = prob[0:n_remove]
    diff_fac = removed_probs.sum() / (len(prob) - len(removed_probs))

    return prob[n_remove:] + diff_fac


class HypTestRes(TypedDict):
    reject_null: bool
    desc: str


def hyp_test_conc(p_val: float, null_hyp: str) -> HypTestRes:
    if p_val >= SIGN_LVL:
        return {
            "reject_null": False,
            "desc": (
                f"Fail to reject null hypothesis of {null_hyp} at {SIGN_LVL} significance."
            ),
        }
    else:
        return {
            "reject_null": True,
            "desc": f"Reject null hypothesis of {null_hyp} at {SIGN_LVL} significance.",
        }

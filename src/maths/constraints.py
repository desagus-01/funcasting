import numpy as np
import polars as pl
from polars import DataFrame

from data_types.vectors import (
    ConstraintSignLike,
    CorrInfo,
    View,
)
from maths.operations import indicator_quantile_marginal


def view_on_quantile(data: DataFrame, quant: float, quant_prob: float) -> list[View]:
    if quant_prob > quant:
        raise ValueError(
            f"Your target prob of {quant_prob}, must be smaller or equal to your current quant of {quant}!"
        )

    quant_ind = indicator_quantile_marginal(data, quant)

    return [
        View(
            type="quantile",
            risk_driver=data.columns[0],
            data=quant_ind.select(pl.col.quant_ind).to_numpy().T,
            views_target=np.array(quant_prob),
            sign_type="equal_less",
        )
    ]


def view_on_corr(
    data: DataFrame,
    corr_targets: list[CorrInfo],
    sign_type: list[ConstraintSignLike],
) -> list[View]:
    return [
        View(
            type="corr",
            risk_driver=corr_info.asset_pair,
            data=data[list(corr_info.asset_pair)].to_numpy().T,
            views_target=np.array(corr_info.corr),
            sign_type=sign_type[i],
        )
        for i, corr_info in enumerate(corr_targets)
    ]


def view_on_mean(
    data: DataFrame,
    target_means: dict[str, float],
    sign_type: list[ConstraintSignLike],
) -> list[View]:
    """
    Builds the constraints based on each asset's targeted mean.
    """
    # Checks we have equal amounts of data, constraints, and targets
    if not (len(target_means.keys()) == len(sign_type)):
        raise ValueError("All inputs must have the same length as the number of views.")

    return [
        View(
            type="mean",
            risk_driver=key,
            data=data[key].to_numpy().T,
            views_target=np.array(target_means[key]),
            sign_type=sign_type[i],
        )
        for i, key in enumerate(target_means.keys())
    ]


def view_on_std(
    data: DataFrame,
    target_std: dict[str, float],
    sign_type: list[ConstraintSignLike],
) -> list[View]:
    # Checks we have equal amounts of data, constraints, and targets
    if not (len(target_std.keys()) == len(sign_type)):
        raise ValueError("All inputs must have the same length as the number of views.")

    return [
        View(
            type="std",
            risk_driver=key,
            data=data[key].to_numpy().T,
            views_target=np.array(target_std[key]),
            sign_type=sign_type[i],
        )
        for i, key in enumerate(target_std.keys())
    ]


def view_on_ranking(
    data: DataFrame,
    asset_ranking: list[str],
) -> list[View]:
    """
    Build view based on which assets should have a higher expected return (ie A >= B >= C ...).
    """

    assets_to_iterate = range(len(asset_ranking) - 1)

    comparisons = [(asset_ranking[i], asset_ranking[i + 1]) for i in assets_to_iterate]

    return [
        View(
            type="sorting",
            risk_driver=comparisons[asset],
            data=data[comparisons[asset]].to_numpy().T,
            views_target=None,
            sign_type="equal_greater",
        )
        for asset in assets_to_iterate
    ]


def view_on_marginal(
    data: DataFrame, current_marginal: str, target_marginal: str
) -> list[View]:
    rd_np = data.select(target_marginal).to_numpy()

    quant_view = view_on_quantile(data.select(target_marginal), 0.25, 0.25)
    mean_view = view_on_mean(
        data, {current_marginal: rd_np.mean()}, sign_type=["equal"]
    )
    std_view = view_on_std(data, {current_marginal: rd_np.std()}, sign_type=["equal"])

    return [quant_view[0], mean_view[0], std_view[0]]

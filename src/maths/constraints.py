import numpy as np
from polars import DataFrame

from data_types.vectors import (
    ConstraintSignLike,
    ConstraintTypeLike,
    View,
)


def view_on_mean(
    data: DataFrame,
    target_means: dict[str, float],
    const_type: list[ConstraintTypeLike],
    sign_type: list[ConstraintSignLike],
) -> list[View]:
    """
    Builds the constraints based on each asset's targeted mean.
    """
    # Checks we have equal amounts of data, constraints, and targets
    if not (len(target_means.keys()) == len(const_type) == len(sign_type)):
        raise ValueError("All inputs must have the same length as the number of views.")

    return [
        View(
            type="mean",
            risk_driver=key,
            data=data[key].to_numpy().T,
            views_target=np.array(target_means[key]),
            const_type=const_type[i],
            sign_type=sign_type[i],
        )
        for i, key in enumerate(target_means.keys())
    ]


def view_on_std(
    data: DataFrame,
    target_std: dict[str, float],
    const_type: list[ConstraintTypeLike],
    sign_type: list[ConstraintSignLike],
) -> list[View]:
    # Checks we have equal amounts of data, constraints, and targets
    if not (len(target_std.keys()) == len(const_type) == len(sign_type)):
        raise ValueError("All inputs must have the same length as the number of views.")

    return [
        View(
            type="std",
            risk_driver=key,
            data=data[key].to_numpy().T,
            views_target=np.array(target_std[key]),
            const_type=const_type[i],
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
            const_type="inequality",
            sign_type="equal_greater",
        )
        for asset in assets_to_iterate
    ]

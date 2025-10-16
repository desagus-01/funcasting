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


def view_on_exp_return_ranking(
    data: DataFrame,
    asset_ranking: list[str],
) -> list[View]:
    """
    Build view based on which assets should have a higher expected return (ie A >= B >= C ...).
    """

    # TODO:Add check to make sure assets in asset_ranking match with those of data

    exp_values = (
        data.select(asset_ranking).mean().to_numpy().flatten()
    )  # re-ordering to match with asset_ranking

    diffs = -np.diff(exp_values)
    return [
        View(
            type="sorting",
            risk_driver=asset,
            data=data[asset].to_numpy().T,
            views_target=np.array(diff),
            const_type="inequality",
            sign_type="equal_greater",
        )
        for asset, diff in zip(asset_ranking[:-1], diffs)
    ]


#
# # TODO: Properly define this
# def view_on_mean_market(
#     data: NDArray[np.floating],
#     target_mean: NDArray[np.floating],
#     const_type: ConstraintTypeLike,
#     sign_type: ConstraintSignLike,
# ) -> View:
#     """
#     Build the constraint based on overall 'market' targeted mean.
#     """
#     # Checks we have equal amounts of data, constraints, and targets
#     if not (len(target_mean) == len(const_type) == len(sign_type) == data.shape[1]):
#         raise ValueError(
#             "All inputs must have the same length as the number of assets."
#         )
#
#     return View(
#         data=data.T,
#         views_targets=np.array(target_mean),
#         const_type=const_type,
#         sign_type=sign_type,
#     )

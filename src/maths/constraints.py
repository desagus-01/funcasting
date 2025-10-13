import numpy as np
from numpy.typing import NDArray

from data_types.vectors import (
    ConstraintSignLike,
    ConstraintTypeLike,
    View,
)


def view_on_mean(
    data: NDArray[np.floating],
    target_mean_vec: NDArray[np.floating],
    const_type: list[ConstraintTypeLike],
    sign_type: list[ConstraintSignLike],
) -> list[View]:
    """
    Builds the constraints based on each asset's targeted mean.
    """
    # Checks we have equal amounts of data, constraints, and targets
    if not (len(target_mean_vec) == len(const_type) == len(sign_type) == data.shape[1]):
        raise ValueError(
            "All inputs must have the same length as the number of assets."
        )

    return [
        View(
            data=data.T[i],
            views_targets=np.array(target_mean_vec[i]),
            const_type=const_type[i],
            sign_type=sign_type[i],
        )
        for i in range(data.T.shape[0])
    ]


def view_on_mean_market(
    data: NDArray[np.floating],
    target_mean: NDArray[np.floating],
    const_type: ConstraintTypeLike,
    sign_type: ConstraintSignLike,
) -> View:
    """
    Build the constraint based on overall 'market' targeted mean.
    """
    # Checks we have equal amounts of data, constraints, and targets
    if not (len(target_mean) == len(const_type) == len(sign_type) == data.shape[1]):
        raise ValueError(
            "All inputs must have the same length as the number of assets."
        )

    return View(
        data=data.T,
        views_targets=np.array(target_mean),
        const_type=const_type,
        sign_type=sign_type,
    )

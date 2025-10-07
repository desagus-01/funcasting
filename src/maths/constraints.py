import numpy as np
from numpy.typing import NDArray

from data_types.vectors import (
    ConstraintSignLike,
    ConstraintTypeLike,
    View,
)


def view_on_mean(
    data: NDArray[np.floating],
    target_mean: NDArray[np.floating],
    const_type: ConstraintTypeLike,
    sign_type: ConstraintSignLike,
) -> View:
    """
    Builds view for mean targets
    """

    if target_mean.shape[0] != data.ndim:
        raise ValueError(
            f"target_mean length {target_mean.shape[0]} must equal data columns {data.ndim}"
        )

    if data.ndim == 1:
        data = data[:, None]

    target_mean = np.asarray(target_mean, dtype=float).reshape(-1)

    return View(
        data=data.T,
        views_targets=target_mean,
        const_type=const_type,
        sign_type=sign_type,
    )

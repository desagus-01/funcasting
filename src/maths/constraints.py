import numpy as np
from numpy.typing import NDArray


def view_on_mean(
    data: NDArray[np.floating],
    target_mean: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Build equality constraints for mean targets
    """

    if target_mean.shape[0] != data.ndim:
        raise ValueError(
            f"target_mean length {target_mean.shape[0]} must equal data columns {data.ndim}"
        )

    if data.ndim == 1:
        data = data[:, None]

    target_mean = np.asarray(target_mean, dtype=float).reshape(-1)

    return data.T, target_mean

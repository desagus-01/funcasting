# TODO: Write Risk distribution code


from typing import Literal

import numpy as np
from numpy._typing import NDArray
from numpy.lib.array_utils import normalize_axis_index


def get_loss_distribution(
    pnl_distribution: NDArray[np.floating],
) -> NDArray[np.floating]:
    return -pnl_distribution


def VAR(
    distribution: NDArray[np.floating],
    method: Literal["empirical", "quantile"] = "quantile",
    alpha: float = 0.05,
    axis: int = 0,
    *,
    distribution_type: Literal["pnl", "loss"] = "loss",
) -> NDArray[np.floating]:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1 (exclusive)")
    if distribution.ndim == 0:
        raise ValueError("distribution must have at least 1 dimension")

    axis = normalize_axis_index(axis, distribution.ndim)
    quantile = alpha if distribution_type == "pnl" else 1 - alpha
    n_samples = distribution.shape[axis]

    if method == "empirical":
        index = int(np.ceil(quantile * n_samples)) - 1
        index = min(max(index, 0), n_samples - 1)
        partitioned = np.partition(distribution, index, axis=axis)
        quantile_val = np.take(partitioned, index, axis=axis)
    elif method == "quantile":
        quantile_val = np.quantile(distribution, quantile, axis=axis)
    else:
        raise ValueError(
            f"Method {method} is not valid, please choose either empirical or quantile"
        )

    return quantile_val if distribution_type == "loss" else -quantile_val

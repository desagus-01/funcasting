import numpy as np
from numpy.typing import NDArray

from data_types.vectors import View


def view_on_mean(
    data: NDArray[np.floating],
    target_mean: NDArray[np.floating],
    eq_ineq: str,
    ineq_sign: str | None,
) -> View:
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

    return View(
        data=data.T, views_targets=target_mean, const_type=eq_ineq, sign_type=ineq_sign
    )

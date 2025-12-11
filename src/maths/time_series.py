import numpy as np
from numpy.typing import NDArray

# INFO: Most of the code/idea below is taken from statsmodels but modified for this use case


def adf_max_lag(n_obs: int, n_reg: int | None) -> int:
    """
    Calculates max lag for augmented dickey fuller test.

    from Greene referencing Schwert 1989
    """
    if n_reg is None:
        return 0
    else:
        max_lag = np.ceil(12.0 * np.power(n_obs / 100.0, 1 / 4.0))
        return int(min(n_obs // 2 - n_reg - 1, max_lag))


def deterministic_detrend(
    data: NDArray[np.floating], polynomial_order: int = 1, axis: int = 0
) -> NDArray[np.floating]:
    """
    Fits a deterministic polynomial trend and then subtracts it from the data
    """
    if data.ndim == 2 and int(axis) == 1:
        data = data.T
    elif data.ndim > 2:
        raise NotImplementedError("data.ndim > 2 is not implemented until it is needed")

    if polynomial_order == 0:
        # Special case demean
        resid = data - data.mean(axis=0)
    else:
        trends = np.vander(np.arange(float(data.shape[0])), N=polynomial_order + 1)
        beta = np.linalg.pinv(trends).dot(data)
        resid = data - np.dot(trends, beta)

    if data.ndim == 2 and int(axis) == 1:
        resid = resid.T

    return resid

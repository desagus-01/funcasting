import numpy as np
from numpy.typing import NDArray


def kernel_smoothing(
    data_array: NDArray[np.floating],
    half_life: float,
    kernel_type: float,
    reference: float | None,
    time_based: bool = False,
) -> NDArray[np.float64]:
    """
    General function for kernel smoothing, allows for exponential, gaussian among other through the kernel_type parameter.
    """
    if time_based:
        data_n = len(data_array)
        data_array = np.arange(data_n)
        dist_to_ref = data_n - 1 - data_array
    else:
        dist_to_ref = np.abs(reference - data_array)

    bandwidth: float = half_life / (np.log(2.0) ** (1.0 / kernel_type))
    return np.exp(-((dist_to_ref / bandwidth) ** kernel_type))


def exponential_time_decay(
    data_array: NDArray[np.floating], half_life: int
) -> NDArray[np.float64]:
    """
    Applies Exponential Decay for a given length
    """
    return (kernel_smoothing(data_array, half_life, 1, None, time_based=True),)

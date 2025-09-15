import numpy as np
from numpy.typing import NDArray


def kernel_smoothing(
    data_array: NDArray[np.floating],
    reference: float,
    half_life: float,
    kernel_type: float,
) -> NDArray[np.float64]:
    """
    General function for kernel smoothing, allows for exponential, gaussian among other through the kernel_type parameter.
    """
    bandwidth: float = half_life / (np.log(2.0) ** (1.0 / kernel_type))
    dist_to_ref = np.abs(data_array - reference)
    return np.exp(-((dist_to_ref / bandwidth) ** kernel_type))


def exponential_decay(length: int, half_life: int) -> NDArray[np.float64]:
    """
    Applies Exponential Decay for a given length
    """
    return kernel_smoothing(length, half_life, kernel_type=1)

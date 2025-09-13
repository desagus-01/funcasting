import numpy as np
from numpy.typing import NDArray


def kernel_smoothing(
    length: int, half_life: float, kernel_type: float
) -> NDArray[np.float64]:
    """
    General function for kernel smoothing, allowes for exponential, gaussian among other through the kernel_type parameter.
    """
    n_array = np.arange(length)
    bandwidth: float = half_life / (np.log(2.0) ** (1.0 / kernel_type))
    dist_to_ref = (length - 1) - n_array  # assumes ref is latest

    return np.exp(-((dist_to_ref / bandwidth) ** kernel_type))


def exponential_decay(length: int, half_life: int) -> NDArray[np.float64]:
    """
    Applies Exponential Decay for a given length
    """
    return kernel_smoothing(length, half_life, kernel_type=1)

import numpy as np
from numpy.typing import NDArray


def exponential_decay(length: int, half_life: int) -> NDArray[np.float64]:
    """
    Applies Exponential Decay for a given length
    """
    n_array = np.arange(length)
    decay_rate = float(np.log(2) / half_life)
    return np.exp(-decay_rate * (length - 1 - n_array))

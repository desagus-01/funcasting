from typing import Sized

import numpy as np
from numpy.typing import NDArray


def exp_decay_probs(vector: Sized, half_life: int) -> NDArray[np.float64]:
    """
    Returns probability vector with exponential decay.

    This allows us to bake in recency bias to our otherwise uniform prior.
    """
    n = len(vector)
    n_array = np.arange(n)
    decay_rate = float(np.log(2) / half_life)
    latest_date = n - 1

    weights: NDArray[np.float64] = np.exp(-decay_rate * (latest_date - n_array))

    return weights / np.sum(weights)  # standardise to ensure probs


def time_crisp_window(vector: Sized, window: int) -> NDArray[np.float64]:
    """
    Returns a probability vector based on the window chosen.
    """
    n = len(vector)
    p = np.zeros(n, dtype=np.float64)
    p[-window:] = 1.0 / window
    return p

from typing import Sized

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame


def exp_decay_probs(vector: Sized | DataFrame, half_life: int) -> NDArray[np.float64]:
    """
    Return a length-n vector of recency-weighted probabilities with exponential decay.

    This allows us to bake in recency bias to our otherwise uniform prior.
    """
    n = len(vector)

    n_array = np.arange(n)

    decay_rate = float(np.log(2) / half_life)

    latest_date = n - 1

    weights: NDArray[np.float64] = np.exp(-decay_rate * (latest_date - n_array))

    return weights / np.sum(weights)  # standardise to ensure probs

import numpy as np
from numpy.typing import ArrayLike, NDArray


def exp_decay_probs(
    data: ArrayLike, half_life: float, reference_time: None | int = None
) -> np.ndarray:
    """
    Returns vector of exponential decayed probabilities based on length of the data and the decay factor.

    This allows us to bake in recency bias to our otherwise uniform prior.
    """

    time_index = np.arange(data)

    decay_rate = float(np.log(2) / half_life)

    reference_time = reference_time if reference_time else 0

    weights: NDArray[np.float64] = np.exp(-decay_rate * (reference_time - time_index))

    return weights / np.sum(weights)  # standardise to ensure probs

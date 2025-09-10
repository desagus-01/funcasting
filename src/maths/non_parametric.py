import numpy as np
from numpy.typing import NDArray


def exp_decay_probs(
    vector: NDArray[np.float64], half_life: float
) -> NDArray[np.float64]:
    """
    Returns vector of exponential decayed probabilities based on length of the vector and the decay factor.

    This allows us to bake in recency bias to our otherwise uniform prior.
    """

    decay_rate = float(np.log(2) / half_life)

    latest_date = len(vector) - 1

    weights: NDArray[np.float64] = np.exp(-decay_rate * (latest_date - vector))

    return weights / np.sum(weights)  # standardise to ensure probs


ex = np.random.rand(1, 50)

test = exp_decay_probs(ex, 50)

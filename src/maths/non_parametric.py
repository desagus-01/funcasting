from typing import Sized

import numpy as np
from numpy.typing import NDArray
from pydantic import validate_call

from data_types.vectors import ProbVector, model_cfg


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
    p = np.zeros(len(vector), dtype=np.float64)
    p[-window:] = 1.0 / window
    return p


@validate_call(config=model_cfg, validate_return=True)
def state_crisp_conditioning(
    vector: Sized, condition_vector: NDArray[np.bool_]
) -> ProbVector:
    """
    Returns a probability vector based on state condition passed.
    """

    if len(vector) != len(condition_vector):
        raise ValueError("Input and condition vector must have the same length.")

    p = np.zeros(len(condition_vector), dtype=np.float64)
    selected_indices = np.where(condition_vector)[0]

    p[selected_indices] = 1.0 / len(selected_indices)
    return p

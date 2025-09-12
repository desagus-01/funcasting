import numpy as np
from numpy.typing import NDArray
from pydantic import validate_call

from data_types.vectors import ProbVector, model_cfg


@validate_call(config=model_cfg, validate_return=True)
def exp_decay_probs(length: int, half_life: int) -> ProbVector:
    """
    Returns probability length with exponential decay.

    This allows us to bake in recency bias to our otherwise uniform prior.
    """
    n_array = np.arange(length)
    decay_rate = float(np.log(2) / half_life)
    latest_date = length - 1

    p: NDArray[np.float64] = np.exp(-decay_rate * (latest_date - n_array))

    return p / np.sum(p)


@validate_call(config=model_cfg, validate_return=True)
def time_crisp_window(length: int, window: int) -> ProbVector:
    """
    Returns a probability length based on the window chosen.
    """
    p = np.zeros(length, dtype=np.float64)
    p[-window:] = 1.0 / window
    return p


@validate_call(config=model_cfg, validate_return=True)
def state_crisp_conditioning(
    length: int, condition_vector: NDArray[np.bool_]
) -> ProbVector:
    """
    Returns a probability length based on state condition passed.
    """

    if length != len(condition_vector):
        raise ValueError("Input and condition length must have the same length.")

    p = np.zeros(len(condition_vector), dtype=np.float64)
    selected_indices = np.where(condition_vector)[0]

    p[selected_indices] = 1.0 / len(selected_indices)
    return p


def smooth_state_conditioning(
    length: int, half_life: int, condition_vector: NDArray[np.bool_]
) -> ProbVector:
    n_array = np.arange(length)
    decay_rate = float(np.log(2) / half_life)
    latest_date = length - 1

    p = np.zeros(length, dtype=np.float64)
    selected_indices = np.where(condition_vector)[0]
    time_diff = latest_date - n_array

    p[condition_vector] = np.exp(-decay_rate * time_diff[condition_vector])

    return p / np.sum(p)

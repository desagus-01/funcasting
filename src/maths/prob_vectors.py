import numpy as np
from numpy.typing import NDArray
from pydantic import validate_call

from data_types.vectors import ProbVector, model_cfg

from .helpers import exponential_decay


@validate_call(config=model_cfg, validate_return=True)
def exp_decay_probs(length: int, half_life: int) -> ProbVector:
    """
    Returns probability length with exponential decay by standardising results.

    This allows us to bake in recency bias to our otherwise uniform prior.
    """
    p = exponential_decay(length, half_life)

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


@validate_call(config=model_cfg, validate_return=True)
def smooth_state_conditioning(
    length: int, half_life: int, condition_vector: NDArray[np.bool_]
) -> ProbVector:
    """
    Applies exponential decay based on state conditions.
    """
    p = np.zeros(length, dtype=np.float64)
    full_decay = exponential_decay(length, half_life)
    p[condition_vector] = full_decay[condition_vector]

    return p / np.sum(p)

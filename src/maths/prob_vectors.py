import numpy as np
from numpy.typing import NDArray
from pydantic import validate_call

from data_types.vectors import ProbVector, model_cfg

from .helpers import exponential_time_decay, kernel_smoothing


@validate_call(config=model_cfg, validate_return=True)
def exp_decay_probs(data_array: NDArray[np.floating], half_life: int) -> ProbVector:
    """
    Returns probability length with exponential decay by standardising results.
    """
    p = exponential_time_decay(data_array, half_life)

    return p / np.sum(p)


@validate_call(config=model_cfg, validate_return=True)
def state_crisp_probs(length: int, condition_vector: NDArray[np.bool_]) -> ProbVector:
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
def state_smooth_probs(
    data: NDArray[np.floating],
    reference: float,
    half_life: int,
    kernel_type: int,
) -> ProbVector:
    """
    Applies kernel based smoothing based on reference target.
    """
    full_decay = kernel_smoothing(data, reference, half_life, kernel_type)
    return full_decay / np.sum(full_decay)

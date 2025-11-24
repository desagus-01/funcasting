import numpy as np
from numpy.typing import NDArray
from pydantic import validate_call

from globals import model_cfg
from models.prob import ProbVector
from models.views import View

from .core import kernel_smoothing, simple_entropy_pooling


@validate_call(config=model_cfg, validate_return=True)
def uniform_probs(len: int) -> ProbVector:
    """
    Returns a uniform probability vector
    """
    return np.ones(len) / len


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
    data_array: NDArray[np.floating],
    half_life: float,
    kernel_type: int,
    reference: float | None,
    time_based: bool = False,
) -> ProbVector:
    """
    Applies kernel based smoothing based on reference target and normalises to probability.
    """
    full_decay = kernel_smoothing(
        data_array=data_array,
        half_life=half_life,
        kernel_type=kernel_type,
        reference=reference,
        time_based=time_based,
    )
    return full_decay / np.sum(full_decay)


@validate_call(config=model_cfg, validate_return=True)
def entropy_pooling_probs(
    prior: ProbVector,
    views: list[View],
    confidence: float = 1.0,
    include_diags: bool = False,
) -> ProbVector:
    """
    Implements entropy pooling optimization using KL divergence as the objective function, then adds confidence value linearly to the resulting posterior.
    """
    entropy_pooling_res = simple_entropy_pooling(
        prior, views, include_diags=include_diags
    )

    return confidence * entropy_pooling_res + (1 - confidence) * prior

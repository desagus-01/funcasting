import numpy as np
from numpy.typing import NDArray
from pydantic import validate_call

from globals import model_cfg
from models.types import ProbVector


def kernel_smoothing(
    data_array: NDArray[np.floating],
    half_life: float,
    kernel_type: int,
    reference: float | None,
    time_based: bool = False,
) -> NDArray[np.float64]:
    """
    General function for kernel smoothing, allows for exponential, gaussian among other through the kernel_type parameter.
    """
    if kernel_type < 0:
        raise ValueError("Kernel type must be positive integer")

    if time_based:
        data_n = len(data_array)
        data = np.arange(data_n)
        dist_to_ref = data_n - 1 - data  # uses last data point as ref
    elif not time_based and reference is not None:
        dist_to_ref = np.abs(reference - data_array)
    else:
        raise ValueError("You must choose correct parameters")

    bandwidth: float = half_life / (np.log(2.0) ** (1.0 / kernel_type))
    return np.exp(-((dist_to_ref / bandwidth) ** kernel_type))


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

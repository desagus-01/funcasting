import numpy as np
from numpy._typing import NDArray

from maths.helpers import get_akicc


def _compute_reflection_coefficient(
    forward_error: NDArray[np.floating],
    backward_error: NDArray[np.floating],
    current_order: int,
    signal_length: int,
    prediction_error_scaling: float,
    total_denominator: float,
) -> tuple[float, float]:
    numerator = np.sum(
        [
            forward_error[j] * backward_error[j - 1]
            for j in range(current_order + 1, signal_length)
        ]
    )

    denominator = (
        prediction_error_scaling * total_denominator
        - forward_error[current_order] ** 2
        - backward_error[signal_length - 1] ** 2
    )

    return -2.0 * numerator / denominator, denominator


def _update_ar_coefficients(
    current_ar_coefficients: NDArray[np.floating],
    reflection_coefficient: float,
    current_order: int,
) -> NDArray[np.floating]:
    updated_coefficients = current_ar_coefficients.copy()
    updated_coefficients[current_order] = reflection_coefficient
    if current_order == 0:
        return updated_coefficients

    previous_coefficients = current_ar_coefficients[:current_order].copy()

    for j in range((current_order + 1) // 2):
        updated_coefficients[j] = (
            previous_coefficients[j]
            + reflection_coefficient * previous_coefficients[current_order - j - 1]
        )
        if j != current_order - j - 1:
            updated_coefficients[current_order - j - 1] = (
                previous_coefficients[current_order - j - 1]
                + reflection_coefficient * previous_coefficients[j]
            )

    return updated_coefficients


def _update_prediction_errors(
    forward_error: NDArray[np.floating],
    backward_error: NDArray[np.floating],
    reflection_coefficient: float,
    current_order: int,
    signal_length: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    updated_forward_error = forward_error.copy()
    updated_backward_error = backward_error.copy()

    for j in range(signal_length - 1, current_order, -1):
        temp_forward = updated_forward_error[j]
        updated_forward_error[j] = (
            temp_forward + reflection_coefficient * updated_backward_error[j - 1]
        )
        updated_backward_error[j] = (
            updated_backward_error[j - 1] + reflection_coefficient * temp_forward
        )

    return updated_forward_error, updated_backward_error


def autoregressive_burg(
    data: NDArray[np.floating], order: int, auto_order: bool = False
) -> NDArray[np.floating]:
    """
    Estimate AR coefficients using the Burg method.

    Parameters:
    -----------
    data : array_like
        1D real or complex-valued time series data.
    order : int
        Desired AR model order (p).
    aut_order: bool
        Automatically chooses best lag based on the AKICc criteria.
    Returns:
    --------
    ar_coeffs : ndarray
        Estimated AR coefficients (a_1, ..., a_p). Real if input is real.
    """

    if order <= 0 or order >= len(data):
        raise ValueError("Order must be > 0 and < length of data.")

    signal_length = len(data)
    forward_error = data.copy()
    backward_error = data.copy()
    ar_coefficients = np.zeros(order)

    total_signal_power = np.sum(data**2) / signal_length
    total_denominator = float(2.0 * signal_length * total_signal_power)
    prediction_error_scaling = 1.0

    criteria_score = []
    for current_order in range(order):
        reflection_coefficient, total_denominator = _compute_reflection_coefficient(
            forward_error,
            backward_error,
            current_order,
            signal_length,
            prediction_error_scaling,
            total_denominator,
        )

        temp = 1.0 - reflection_coefficient**2
        total_signal_power *= temp
        prediction_error_scaling *= temp

        criteria_score.append(
            get_akicc(signal_length, float(total_signal_power), current_order + 1)
        )
        if auto_order and len(criteria_score) > 1:
            if criteria_score[-1] > criteria_score[-2]:
                return ar_coefficients[:current_order]

        ar_coefficients = _update_ar_coefficients(
            ar_coefficients, reflection_coefficient, current_order
        )

        forward_error, backward_error = _update_prediction_errors(
            forward_error,
            backward_error,
            reflection_coefficient,
            current_order,
            signal_length,
        )

    return ar_coefficients


def _long_autoregressive_model(moving_average_order: int) -> int:
    """
    Rule of thumb for long AR model order to use to calc MA model
    """
    return max(2 * moving_average_order, 20)


def moving_average(data: NDArray[np.floating], order: int):
    ar_long_order = _long_autoregressive_model(order)
    ar_coefficients = autoregressive_burg(data=data, order=ar_long_order)
    # add unity?
    ar_coefficients = np.insert(ar_coefficients, 0, 1)

    return autoregressive_burg(data=ar_coefficients, order=order)

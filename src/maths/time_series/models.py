import warnings
from typing import NamedTuple

import numpy as np
from numpy._typing import NDArray
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
from typing_extensions import Literal

from maths.helpers import get_akicc


class AutoARMARes(NamedTuple):
    model_order: tuple[int, int]
    degrees_of_freedom: int
    criteria: Literal["aic", "bic"]
    criteria_res: float
    ar_params: NDArray[np.floating]
    ma_params: NDArray[np.floating]
    p_values: NDArray[np.floating]
    residuals: NDArray[np.floating]


class DemeanRes(NamedTuple):
    degrees_of_freedom: int
    mean_: float
    residuals: NDArray[np.floating]


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


def _arma_top_candidates(
    asset_array: NDArray[np.floating],
    max_ar_order: int = 3,
    max_ma_order: int = 3,
    information_criteria: Literal["bic", "aic"] = "bic",
    top_n_models: int = 3,
) -> list[
    tuple[
        int,
        int,
    ]
]:
    """
    Identify the top ARMA (p, q) orders with the lowest information criterion.

    Uses a fast, approximate estimation method to evaluate candidate ARMA
    models and returns the p and q orders of the best-performing models.
    """
    # Get top n model orders with lowest information criteria
    best_order = arma_order_select_ic(
        y=asset_array,
        max_ar=max_ar_order,
        max_ma=max_ma_order,
        ic=information_criteria,
        trend="n",
        model_kw={"enforce_stationarity": False, "enforce_invertibility": False},
        fit_kw={"method": "hannan_rissanen", "low_memory": True},
    )["bic"]

    top = best_order.stack().nsmallest(top_n_models)
    return [(int(ar_order), int(ma_order)) for (ar_order, ma_order) in top.index]


def by_criteria(res: AutoARMARes) -> float:
    """
    Extract the information criterion value from an AutoARMARes object.
    """
    return res.criteria_res


# TODO: Filter out p-vals?
def auto_arma(
    asset_array: NDArray[np.floating],
    max_ar_order: int,
    max_ma_order: int,
    top_n_models: int,
    information_criteria: Literal["bic", "aic"],
) -> list[AutoARMARes]:
    """
    Fit ARMA models for the top candidate orders and collect their results.

    Candidate model orders are selected using an information criterion,
    then re-fitted using a more accurate estimation method.
    """
    candidates_for_arma = _arma_top_candidates(
        asset_array,
        max_ar_order,
        max_ma_order,
        information_criteria,
        top_n_models,
    )

    arma_res = []

    for ar_order, ma_order in candidates_for_arma:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)

                res = ARIMA(asset_array, order=(ar_order, 0, ma_order), trend="n").fit(
                    method="statespace"
                )

        except ConvergenceWarning:
            # Skip models that fail to converge
            print(
                f"Model ({ar_order},{ma_order}) failed to converge. Will be dropped from candidates list"
            )
            continue

        arma_res.append(
            AutoARMARes(
                model_order=(ar_order, ma_order),
                degrees_of_freedom=ar_order + ma_order,
                criteria=information_criteria,
                criteria_res=float(getattr(res, information_criteria)),
                ar_params=res.arparams,  # type: ignore[attr-defined]
                ma_params=res.maparams,  # type: ignore[attr-defined]
                p_values=res.pvalues,  # type: ignore[attr-defined]
                residuals=res.resid,  # type: ignore[attr-defined]
            )
        )
    if not arma_res:
        raise ValueError("No ARMA models were fitted, likely due to failed convergence")

    return arma_res

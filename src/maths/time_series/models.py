import logging
import warnings
from itertools import product
from re import fullmatch
from typing import NamedTuple

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from numpy._typing import NDArray
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
from typing_extensions import Literal

from maths.helpers import get_akicc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AutoGARCHRes(NamedTuple):
    model_order: tuple[int, int, int]
    degrees_of_freedom: int
    criteria: Literal["aic", "bic"]
    criteria_res: float
    params: dict[str, float]
    p_values: NDArray[np.floating]
    residuals: NDArray[np.floating]
    conditional_volatility: NDArray[np.floating]
    invariants: NDArray[np.floating]
    kind: Literal["garch"] = "garch"


class AutoARMARes(NamedTuple):
    model_order: tuple[int, int]
    degrees_of_freedom: int
    criteria: Literal["aic", "bic"]
    criteria_res: float
    params: dict[str, float]
    p_values: NDArray[np.floating]
    residuals: NDArray[np.floating]
    residual_scale: float
    kind: Literal["arma"] = "arma"


class DemeanRes(NamedTuple):
    model_order: None
    degrees_of_freedom: int
    params: dict[str, float]
    residuals: NDArray[np.floating]
    residual_scale: float
    kind: Literal["demean"] = "demean"


def _build_arma_parameters(
    parameter_names: list[str],
    ar_estimates: NDArray[np.floating],
    ma_estimates: NDArray[np.floating],
) -> dict[str, float]:
    params = {}
    for param in parameter_names:
        m = fullmatch(r"(ar|ma)\.L(\d+)", param)
        if m:
            kind, lag = m.group(1), int(m.group(2))
            if lag < 1:
                raise ValueError(f"Invalid lag in {param}")
            if kind == "ar":
                params[param] = ar_estimates[lag - 1]
            if kind == "ma":
                params[param] = ma_estimates[lag - 1]

    return params


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


def by_criteria(res: AutoARMARes | AutoGARCHRes) -> float:
    """
    Extract the information criterion value from an AutoARMARes object.
    """
    return res.criteria_res


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

                model = ARIMA(
                    asset_array,
                    order=(ar_order, 0, ma_order),
                    seasonal_order=[0, 0, 0, 0],
                    trend="n",
                )  # no integration order needed as we have done that in pre-processing
                res = model.fit(method="statespace")
        except ConvergenceWarning:
            logger.info(
                "Model (%s, %s) failed to converge. Will be dropped from candidates list",
                ar_order,
                ma_order,
            )
            continue

        arma_res.append(
            AutoARMARes(
                model_order=(ar_order, ma_order),
                params=_build_arma_parameters(
                    model.param_names,
                    res.arparams,  # type: ignore[attr-defined]
                    res.maparams,  # type: ignore[attr-defined]
                ),
                degrees_of_freedom=ar_order + ma_order,
                criteria=information_criteria,
                criteria_res=float(getattr(res, information_criteria)),
                p_values=res.pvalues,  # type: ignore[attr-defined]
                residuals=res.resid,  # type: ignore[attr-defined]
                residual_scale=float(np.std(res.resid, ddof=1)),  # type: ignore[attr-defined]
            )
        )
    if not arma_res:
        raise ValueError("No ARMA models were fitted, likely due to failed convergence")

    return arma_res


Dist = Literal["t", "normal"]


def _garch_base_model(
    asset_array: NDArray[np.floating],
    innovation_distribution: Dist = "t",
    p_order: int = 1,
    o_order: int = 0,
    q_order: int = 1,
) -> ARCHModelResult:
    base_model = arch_model(
        y=asset_array,
        mean="zero",
        p=p_order,
        o=o_order,
        q=q_order,
        dist=innovation_distribution,
        rescale=False,
    ).fit(disp=False)
    return base_model


def auto_garch(
    asset_array: NDArray[np.floating],
    max_p_order: int = 2,
    max_o_order: int = 1,
    max_q_order: int = 2,
) -> list[AutoGARCHRes]:
    base_model = _garch_base_model(asset_array=asset_array)
    dists: tuple[Dist, Dist] = ("t", "normal")
    garch_candidates = []
    for p, q, o, distribution in product(
        range(1, max_p_order + 1),
        range(1, max_q_order + 1),
        range(0, max_o_order + 1),
        dists,
    ):
        key = (p, o, q)
        if (key == (1, 0, 1)) and (distribution == "t"):
            continue

        proposed_model = arch_model(
            asset_array, mean="zero", p=p, o=o, q=q, dist=distribution, rescale=False
        ).fit(disp="off")
        if proposed_model.convergence_flag != 0:
            print(
                "NO CONVERGE",
                key,
                distribution,
                "flag",
                proposed_model.convergence_flag,
            )
            continue

        if proposed_model.bic < base_model.bic:
            garch_candidates.append(
                AutoGARCHRes(
                    model_order=key,
                    degrees_of_freedom=len(proposed_model.params),
                    criteria="bic",
                    criteria_res=proposed_model.bic,
                    params=proposed_model.params.to_dict(),  # type: ignore[attr-defined]
                    p_values=proposed_model.pvalues,  # type: ignore[attr-defined]
                    invariants=proposed_model.std_resid,  # type: ignore[attr-defined]
                    residuals=proposed_model.resid,  # type: ignore[attr-defined]
                    conditional_volatility=proposed_model.conditional_volatility,  # type: ignore[attr-defined]
                )
            )

    garch_candidates.append(
        AutoGARCHRes(
            model_order=(1, 0, 1),
            degrees_of_freedom=len(base_model.params),
            criteria="bic",
            criteria_res=base_model.bic,
            params=base_model.params.to_dict(),  # type: ignore[attr-defined]
            p_values=base_model.pvalues,  # type: ignore[attr-defined]
            residuals=base_model.resid,  # type: ignore[attr-defined]
            invariants=base_model.std_resid,  # type: ignore[attr-defined]
            conditional_volatility=base_model.conditional_volatility,  # type: ignore[attr-defined]
        )
    )

    return garch_candidates

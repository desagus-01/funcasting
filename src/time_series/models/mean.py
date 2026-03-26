import logging
import warnings
from re import fullmatch
from typing import NamedTuple

import numpy as np
from numpy._typing import NDArray
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
from typing_extensions import Literal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


def _arma_filter(
    params: dict[str, float], max_abs_ar1: float = 0.98, max_sum_ar: float = 0.98
) -> bool:
    ar_vals = [float(v) for k, v in params.items() if k.startswith("ar.L")]
    if len(ar_vals) == 1 and abs(ar_vals[0]) >= max_abs_ar1:
        return False
    if len(ar_vals) >= 2 and sum(ar_vals) >= max_sum_ar:
        return False
    return True


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

        params = _build_arma_parameters(model.param_names, res.arparams, res.maparams)  # type: ignore[attr-defined]

        if not _arma_filter(params):
            logger.info(
                "Model (%s, %s) ar vals are close to 1, will drop it",
                ar_order,
                ma_order,
            )

            continue

        arma_res.append(
            AutoARMARes(
                model_order=(ar_order, ma_order),
                params=params,
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

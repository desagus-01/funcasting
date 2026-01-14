from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

EquationTypes = Literal["nc", "c", "ct", "ctt"]


class LeastSquaresRes(NamedTuple):
    results: NDArray[np.floating]
    residuals: NDArray[np.floating]
    sum_of_squared_residuals: float


class OLSEquation(NamedTuple):
    ind_var: NDArray[np.floating]
    dep_vars: NDArray[np.floating]


@dataclass
class OLSResults:
    res: NDArray[np.floating]
    std_errors: NDArray[np.floating]
    t_stats: NDArray[np.floating]
    residuals: NDArray[np.floating]
    aic: float
    bic: float

    def __post_init__(self) -> None:
        shape = self.res.shape
        if self.std_errors.shape != shape or self.t_stats.shape != shape:
            raise ValueError(f"""
            OLSResults values must all have the same shape:
            res : {shape}
            std_errors: {self.std_errors.shape}
            t_stats: {self.t_stats}
            """)


class AICBIC(NamedTuple):
    aic: float
    bic: float


def add_deterministics_to_eq(
    independent_vars: NDArray[np.floating], eq_type: EquationTypes
):
    n_obs = independent_vars.shape[0]

    time_index = np.arange(1, n_obs + 1, dtype=float).reshape(-1, 1)
    cols = [np.ones((n_obs, 1), dtype=float)]  # add constants "c" as default

    if eq_type == "nc":
        return independent_vars
    if eq_type in ("ct", "ctt"):
        cols.append(time_index)
    if eq_type == "ctt":
        cols.append(time_index**2)

    deterministics = np.hstack(cols)
    return np.hstack([deterministics, independent_vars])


def ols_log_likelihood(sum_of_squared_residuals: float, n_obs: int) -> float:
    return (
        -0.5
        * n_obs
        * (np.log(2 * np.pi) + 1.0 + np.log(sum_of_squared_residuals / n_obs))
    )


def get_aic(log_likelihood: float, n_parameters: int) -> float:
    return -2 * log_likelihood + 2 * n_parameters


def get_bic(log_likelihood: float, n_obs: int, n_parameters: int) -> float:
    return -2 * log_likelihood + np.log(n_obs) * n_parameters


def get_aic_bic(
    sum_of_squared_residuals: float, n_obs: int, n_parameters: int
) -> AICBIC:
    llf = ols_log_likelihood(
        sum_of_squared_residuals=sum_of_squared_residuals, n_obs=n_obs
    )
    aic = get_aic(log_likelihood=llf, n_parameters=n_parameters)
    bic = get_bic(log_likelihood=llf, n_obs=n_obs, n_parameters=n_parameters)
    return AICBIC(aic=aic, bic=bic)


def least_squares_fit(
    dependent_var: NDArray[np.floating], independent_vars: NDArray[np.floating]
) -> LeastSquaresRes:
    res = np.linalg.lstsq(a=independent_vars, b=dependent_var, rcond=None)
    ols_res = res[0]
    sum_of_squared_residuals = res[1]
    dependent_est = independent_vars @ ols_res
    residuals = dependent_var - dependent_est

    if sum_of_squared_residuals.size == 0:
        sum_of_squared_residuals = float(residuals.T @ residuals)
    else:
        sum_of_squared_residuals = float(res[1].item())

    return LeastSquaresRes(
        results=ols_res,
        residuals=residuals,
        sum_of_squared_residuals=sum_of_squared_residuals,
    )


def ols_standard_errors(
    independent_vars: NDArray[np.floating], sum_of_squared_residuals: float, n_obs: int
) -> NDArray[np.floating]:
    k = independent_vars.shape[1]
    cov_scaler = sum_of_squared_residuals / (n_obs - k)

    cov_inv = np.linalg.pinv(independent_vars.T @ independent_vars)
    scaled_cov_inv = cov_scaler * cov_inv
    return np.sqrt(np.diag(scaled_cov_inv)).reshape(-1, 1)


def ols_classic(
    dependent_var: NDArray[np.floating], independent_vars: NDArray[np.floating]
) -> OLSResults:
    n_obs = dependent_var.shape[0]

    ols_res, residuals, sum_of_squared_residuals = least_squares_fit(
        dependent_var=dependent_var, independent_vars=independent_vars
    )
    standard_errors = ols_standard_errors(
        independent_vars=independent_vars,
        sum_of_squared_residuals=sum_of_squared_residuals,
        n_obs=n_obs,
    )

    fit_evals = get_aic_bic(
        sum_of_squared_residuals=sum_of_squared_residuals,
        n_obs=n_obs,
        n_parameters=independent_vars.shape[1],
    )

    return OLSResults(
        res=ols_res,
        std_errors=standard_errors,
        t_stats=ols_res / standard_errors,
        residuals=residuals,
        aic=fit_evals.aic,
        bic=fit_evals.bic,
    )

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

EquationTypes = Literal["nc", "c", "ct", "ctt"]


class OLSEquation(NamedTuple):
    ind_var: NDArray[np.floating]
    dep_vars: NDArray[np.floating]


@dataclass
class OLSResults:
    res: NDArray[np.floating]
    std_errors: NDArray[np.floating]
    t_stats: NDArray[np.floating]
    residuals: NDArray[np.floating]

    def __post_init__(self) -> None:
        shape = self.res.shape
        if self.std_errors.shape != shape or self.t_stats.shape != shape:
            raise ValueError(f"""
            OLSResults values must all have the same shape:
            res : {shape}
            std_errors: {self.std_errors.shape}
            t_stats: {self.t_stats}
            """)


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


def ols_classic(
    dependent_var: NDArray[np.floating], independent_vars: NDArray[np.floating]
) -> OLSResults:
    res = np.linalg.lstsq(a=independent_vars, b=dependent_var, rcond=None)
    ols_res = res[0]
    sum_of_squared_residuals = res[1]
    dependent_est = independent_vars @ ols_res
    residuals = dependent_var - dependent_est

    if sum_of_squared_residuals.size == 0:
        sum_of_squared_residuals = float(residuals.T @ residuals)
    else:
        sum_of_squared_residuals = float(res[1].item())

    n_obs = dependent_var.shape[0]
    k = independent_vars.shape[1]
    cov_scaler = sum_of_squared_residuals / (n_obs - k)

    cov_inv = np.linalg.inv(independent_vars.T @ independent_vars)
    scaled_cov_inv = cov_scaler * cov_inv
    standard_errors = np.sqrt(np.diag(scaled_cov_inv)).reshape(-1, 1)
    t_stats = ols_res / standard_errors

    return OLSResults(
        res=ols_res, std_errors=standard_errors, t_stats=t_stats, residuals=residuals
    )

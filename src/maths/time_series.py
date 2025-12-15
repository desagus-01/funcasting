from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
from numpy import polyval
from numpy.typing import NDArray
from scipy.stats import norm

from globals import DF_EQ_TYPE, MACKIN_TAU_CUTOFFS, MACKIN_TAU_PVALS

# INFO: Most of the code/idea below is taken from statsmodels but modified for this use case


class ADFEquation(NamedTuple):
    ind_var: NDArray[np.floating]
    dep_vars: NDArray[np.floating]


EquationTypes = Literal["nc", "c", "ct", "ctt"]


@dataclass
class OLSResults:
    res: NDArray[np.floating]
    std_errors: NDArray[np.floating]
    t_stats: NDArray[np.floating]

    def __post_init__(self) -> None:
        shape = self.res.shape
        if self.std_errors.shape != shape or self.t_stats.shape != shape:
            raise ValueError(f"""
            OLSResults values must all have the same shape:
            res : {shape}
            std_errors: {self.std_errors.shape}
            t_stats: {self.t_stats}
            """)


def _adf_max_lag(n_obs: int, n_reg: int | None) -> int:
    """
    Calculates max lag for augmented dickey fuller test.

    from Greene referencing Schwert 1989
    """
    if n_reg is None:
        return 0
    else:
        max_lag = np.ceil(12.0 * np.power(n_obs / 100.0, 1 / 4.0))
        return int(min(n_obs // 2 - n_reg - 1, max_lag))


def deterministic_detrend(
    data: NDArray[np.floating], polynomial_order: int = 1, axis: int = 0
) -> NDArray[np.floating]:
    """
    Fits a deterministic polynomial trend and then subtracts it from the data
    """
    if data.ndim == 2 and int(axis) == 1:
        data = data.T
    elif data.ndim > 2:
        raise NotImplementedError("data.ndim > 2 is not implemented until it is needed")

    if polynomial_order == 0:
        # Special case demean
        resid = data - data.mean(axis=0)
    else:
        trends = np.vander(np.arange(float(data.shape[0])), N=polynomial_order + 1)
        beta = np.linalg.pinv(trends).dot(data)
        resid = data - np.dot(trends, beta)

    if data.ndim == 2 and int(axis) == 1:
        resid = resid.T

    return resid


def _add_deterministics_to_eq(
    independent_vars: NDArray[np.floating], eq_type: EquationTypes
):
    n_obs = independent_vars.shape[0]

    time_index = np.arange(1, n_obs + 1, dtype=float).reshape(-1, 1)
    cols = [np.ones((n_obs, 1), dtype=float)]

    if eq_type in ("nc"):
        return independent_vars
    if eq_type in ("ct", "ctt"):
        cols.append(time_index)
    if eq_type == "ctt":
        cols.append(time_index**2)

    deterministics = np.hstack(cols)
    return np.hstack([deterministics, independent_vars])


def _build_adf_equation(
    data: pl.DataFrame,
    asset: str,
    lags: int,
    eq_type: EquationTypes,
) -> ADFEquation:
    df = (
        data.select(asset)
        .with_columns(
            pl.col(asset).diff().alias(f"{asset}_diff_1"),
            pl.col(asset).shift(1).alias(f"{asset}_lag_1"),
            *[
                pl.col(asset).diff().shift(i).alias(f"{asset}_diff_1_lag_{i}")
                for i in range(1, lags + 1)
            ],
        )
        .drop_nulls()
    )

    x_col_order = [f"{asset}_lag_1"] + [
        f"{asset}_diff_1_lag_{i}" for i in range(1, lags + 1)
    ]

    y = df.select(f"{asset}_diff_1").to_numpy()
    x = df.select(x_col_order).to_numpy()
    x_with_determs = _add_deterministics_to_eq(independent_vars=x, eq_type=eq_type)
    return ADFEquation(ind_var=x_with_determs, dep_vars=y)


def ols(
    dependent_var: NDArray[np.floating], independent_vars: NDArray[np.floating]
) -> OLSResults:
    res = np.linalg.lstsq(a=independent_vars, b=dependent_var, rcond=None)
    ols_res = res[0]
    sum_of_squared_residuals = res[1]
    if sum_of_squared_residuals.size == 0:
        dependent_est = independent_vars @ ols_res
        residuals = dependent_var - dependent_est
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

    return OLSResults(res=ols_res, std_errors=standard_errors, t_stats=t_stats)


def p_val_approx(
    test_stat: float,
    regression: EquationTypes = "nc",
    n_integrated: int = 1,
) -> float:
    min_cutoff = MACKIN_TAU_CUTOFFS[f"min_{regression}"]
    max_cutoff = MACKIN_TAU_CUTOFFS[f"max_{regression}"]
    starstat = MACKIN_TAU_CUTOFFS[f"star_{regression}"]
    p_val_ind = n_integrated - 1

    if test_stat > max_cutoff[p_val_ind]:
        return 1.0
    elif test_stat < min_cutoff[p_val_ind]:
        return 0.0
    if test_stat <= starstat[p_val_ind]:
        tau_coef = MACKIN_TAU_PVALS[f"tau_{regression}_smallp"][p_val_ind]
    else:
        tau_coef = MACKIN_TAU_PVALS[f"tau_{regression}_largep"][p_val_ind]

    return float(norm.cdf(polyval(tau_coef[::-1], test_stat)))


class ADFResults(NamedTuple):
    test_stat: float
    std_error: float
    p_val: float


def augmented_dickey_fuller(
    data: pl.DataFrame, asset: str, eq_type: EquationTypes = "nc"
) -> ADFResults:
    added_regressors = DF_EQ_TYPE[eq_type]
    print(added_regressors)
    max_lags = _adf_max_lag(
        data.height,
        added_regressors,
    )
    adf_eq = _build_adf_equation(data=data, asset=asset, lags=max_lags, eq_type=eq_type)

    ols_res = ols(dependent_var=adf_eq.dep_vars, independent_vars=adf_eq.ind_var)
    adf_stat = float(ols_res.t_stats[added_regressors].item())

    approx_p_val = p_val_approx(
        test_stat=adf_stat,
        regression=eq_type,
        n_integrated=1,
    )

    return ADFResults(
        test_stat=adf_stat,
        std_error=float(ols_res.std_errors[added_regressors].item()),
        p_val=approx_p_val,
    )

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
from numpy import polyval
from numpy.typing import NDArray
from scipy.stats import norm

from globals import (
    EQ_TYPE_ADDED_DETS,
    KPSS_CRIT_VALUES,
    KPSS_P_VALS,
    MACKIN_TAU_CUTOFFS,
    MACKIN_TAU_PVALS,
)

# INFO: Most of the code/idea below is taken from statsmodels but modified for this use case


class OLSEquation(NamedTuple):
    ind_var: NDArray[np.floating]
    dep_vars: NDArray[np.floating]


class StationaryTestsRes(NamedTuple):
    test_stat: float
    p_val: float
    std_error: float | None


EquationTypes = Literal["nc", "c", "ct", "ctt"]


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


def _add_deterministics_to_eq(
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


def _build_adf_equation(
    data: pl.DataFrame,
    asset: str,
    lags: int,
    eq_type: EquationTypes,
) -> OLSEquation:
    diff_lag_exprs = (
        [
            pl.col(asset).diff().shift(i).alias(f"{asset}_diff_1_lag_{i}")
            for i in range(1, lags + 1)
        ]
        if lags > 0
        else []
    )

    df = (
        data.select(asset)
        .with_columns(
            pl.col(asset).diff().alias(f"{asset}_diff_1"),
            pl.col(asset).shift(1).alias(f"{asset}_lag_1"),
            *diff_lag_exprs,
        )
        .drop_nulls()
    )

    x_col_order = [f"{asset}_lag_1"] + (
        [f"{asset}_diff_1_lag_{i}" for i in range(1, lags + 1)] if lags > 0 else []
    )

    y = df.select(f"{asset}_diff_1").to_numpy()
    x = df.select(x_col_order).to_numpy()
    if eq_type != "nc":
        x = _add_deterministics_to_eq(independent_vars=x, eq_type=eq_type)

    return OLSEquation(ind_var=x, dep_vars=y)


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


def adf_p_val_approx(
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


def augmented_dickey_fuller(
    data: pl.DataFrame,
    asset: str,
    lags: int | None,
    eq_type: EquationTypes = "nc",
) -> StationaryTestsRes:
    added_regressors = EQ_TYPE_ADDED_DETS[eq_type]
    # TODO: Change below to autolag like statsmodels
    if lags is None:
        max_lags = _adf_max_lag(
            data.height,
            added_regressors,
        )
    else:
        max_lags = lags
    adf_eq = _build_adf_equation(data=data, asset=asset, lags=max_lags, eq_type=eq_type)

    ols_res = ols_classic(
        dependent_var=adf_eq.dep_vars, independent_vars=adf_eq.ind_var
    )
    adf_stat = float(ols_res.t_stats[added_regressors].item())

    approx_p_val = adf_p_val_approx(
        test_stat=adf_stat,
        regression=eq_type,
        n_integrated=1,
    )

    return StationaryTestsRes(
        test_stat=adf_stat,
        std_error=float(ols_res.std_errors[added_regressors].item()),
        p_val=approx_p_val,
    )


def build_kpss_equation(
    data: pl.DataFrame, asset: str, eq_type: EquationTypes = "ct"
) -> OLSEquation:
    dependent_var = data.select(asset).to_numpy()

    empty_arr = np.empty(shape=dependent_var.shape)
    np.zeros_like(dependent_var)
    determs = _add_deterministics_to_eq(independent_vars=empty_arr, eq_type=eq_type)[
        :, :-1
    ]  # we remove original

    return OLSEquation(ind_var=determs, dep_vars=dependent_var)


# INFO: Below function is straight from statsmodels with slight modification
def _kpss_choose_lag(residuals: NDArray[np.floating], n_obs: int) -> int:
    """
    Computes the number of lags for covariance matrix estimation in KPSS test
    using method of Hobijn et al (1998). See also Andrews (1991), Newey & West
    (1994), and Schwert (1989). Assumes Bartlett / Newey-West kernel.
    """
    truncation_window = int(np.power(n_obs, 2.0 / 9.0))
    radius = np.sum(residuals**2) / n_obs
    s1 = 0
    for i in range(1, truncation_window + 1):
        resids_prod = np.dot(residuals[i:], residuals[: n_obs - i])
        resids_prod /= n_obs / 2.0  # normalizing
        radius += resids_prod
        s1 += i * resids_prod
    s_hat = s1 / radius
    pwr = 1.0 / 3.0
    gamma_hat = 1.1447 * np.power(s_hat * s_hat, pwr)
    auto_lag = int(gamma_hat * np.power(n_obs, pwr))
    return min(auto_lag, n_obs - 1)


# INFO: So is the below
def _sigma_est_kpss(residuals: NDArray[np.floating], n_obs: int, lags: int) -> float:
    """
    Computes equation 10, p. 164 of Kwiatkowski et al. (1992). This is the
    consistent estimator for the variance.
    """
    s_hat = np.sum(residuals**2)
    for i in range(1, lags + 1):
        resids_prod = np.dot(residuals[i:], residuals[: n_obs - i])
        s_hat += 2 * resids_prod * (1.0 - (i / (lags + 1.0)))
    return float(s_hat / n_obs)


def _kpss_p_val_approx(t_stat: float, eq_type: EquationTypes) -> float:
    crit_values = KPSS_CRIT_VALUES[eq_type]
    return float(np.interp(t_stat, crit_values, KPSS_P_VALS))


def kpss(
    data: pl.DataFrame, asset: str, null_type_stationarity: Literal["trend", "level"]
) -> StationaryTestsRes:
    eq_type: EquationTypes = "ct" if null_type_stationarity == "trend" else "c"

    n_obs = data.height
    if eq_type == "ct":  # run OLS with constant and trend
        x = build_kpss_equation(data, asset, eq_type)
        residuals = ols_classic(
            dependent_var=x.dep_vars, independent_vars=x.ind_var
        ).residuals
    else:  # just demean
        dependent_var = data.select(asset).to_numpy()
        residuals = dependent_var - dependent_var.mean()

    lags = _kpss_choose_lag(residuals=residuals.ravel(), n_obs=n_obs)

    kpss_numerator = np.sum(residuals.cumsum() ** 2) / (n_obs**2)
    kpss_denominator = _sigma_est_kpss(
        residuals=residuals.ravel(), n_obs=n_obs, lags=lags
    )

    kpss_stat = kpss_numerator / kpss_denominator
    p_val = _kpss_p_val_approx(t_stat=kpss_stat, eq_type=eq_type)

    return StationaryTestsRes(test_stat=kpss_stat, p_val=p_val, std_error=None)

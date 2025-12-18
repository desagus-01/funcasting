from dataclasses import dataclass
from typing import Callable, Literal, NamedTuple

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
from maths.stochastic_processes.base import (
    HypTestRes,
    format_hyp_test_result,
)
from maths.stochastic_processes.estimation import (
    EquationTypes,
    OLSEquation,
    add_deterministics_to_eq,
    ols_classic,
)

InferenceLabel = Literal[
    "stationary",
    "trend-stationary",
    "unit-root/non-stationary",
    "inconclusive/conflict",
]


class StationaryTestsRes(NamedTuple):
    test_stat: float
    p_val: float
    std_error: float | None


@dataclass(frozen=True, slots=True)
class StationarityInference:
    label: InferenceLabel
    sign_lvl: float
    details: str
    adf: HypTestRes
    kpss_level: HypTestRes
    kpss_trend: HypTestRes


@dataclass(frozen=True)
class Rule:
    label: InferenceLabel
    when: Callable[[bool, bool, bool], bool]
    details: str


RULES: tuple[Rule, ...] = (
    Rule(
        "stationary",
        lambda A, L, T: A and (not L),
        "ADF rejects unit root; KPSS(level) fails to reject level stationarity.",
    ),
    Rule(
        "unit-root/non-stationary",
        lambda A, L, T: (not A) and L,
        "ADF fails to reject unit root; KPSS(level) rejects level stationarity.",
    ),
    Rule(
        "trend-stationary",
        lambda A, L, T: A and L and (not T),
        "ADF rejects unit root; KPSS(level) rejects; KPSS(trend) fails to reject trend stationarity.",
    ),
)


def infer_stationarity(
    adf: HypTestRes,
    kpss_level: HypTestRes,
    kpss_trend: HypTestRes,
) -> StationarityInference:
    A = adf.reject_null
    L = kpss_level.reject_null
    T = kpss_trend.reject_null

    for r in RULES:
        if r.when(A, L, T):
            return StationarityInference(
                label=r.label,
                sign_lvl=adf.sign_lvl,
                details=r.details,
                adf=adf,
                kpss_level=kpss_level,
                kpss_trend=kpss_trend,
            )

    return StationarityInference(
        label="inconclusive/conflict",
        sign_lvl=adf.sign_lvl,
        details="No clean ADF/KPSS agreement pattern at this significance level.",
        adf=adf,
        kpss_level=kpss_level,
        kpss_trend=kpss_trend,
    )


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
        x = add_deterministics_to_eq(independent_vars=x, eq_type=eq_type)

    return OLSEquation(ind_var=x, dep_vars=y)


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
    determs = add_deterministics_to_eq(independent_vars=empty_arr, eq_type=eq_type)[
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


def augmented_dickey_fuller_test(
    data: pl.DataFrame, asset: str, lags: int | None, eq_type: EquationTypes = "nc"
) -> HypTestRes:
    adf_res = augmented_dickey_fuller(
        data=data, asset=asset, lags=lags, eq_type=eq_type
    )

    return format_hyp_test_result(
        stat=adf_res.test_stat,
        p_val=adf_res.p_val,
        null=f"Unit Root/Non-Stationarity at {lags} lags",
    )


def kpss_test(
    data: pl.DataFrame, asset: str, null_type_stationarity: Literal["trend", "level"]
) -> HypTestRes:
    kpss_res = kpss(
        data=data, asset=asset, null_type_stationarity=null_type_stationarity
    )

    return format_hyp_test_result(
        stat=kpss_res.test_stat,
        p_val=kpss_res.p_val,
        null=f"{null_type_stationarity} stationarity",
    )


def stationarity_tests(
    data: pl.DataFrame,
    asset: str,
    lags: int | None,
    eq_type: EquationTypes = "nc",
) -> StationarityInference:
    adf_res = augmented_dickey_fuller_test(
        data=data, asset=asset, lags=lags, eq_type=eq_type
    )
    kpss_res_lvl = kpss_test(data=data, asset=asset, null_type_stationarity="level")

    kpss_res_trend = kpss_test(data=data, asset=asset, null_type_stationarity="trend")

    return infer_stationarity(
        adf=adf_res, kpss_level=kpss_res_lvl, kpss_trend=kpss_res_trend
    )

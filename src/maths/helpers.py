from dataclasses import dataclass

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy._typing import NDArray
from scipy import fft, stats
from statsmodels.compat.python import lzip


@dataclass(frozen=True)
class AutoCorrelation:
    lower: float
    value: float
    upper: float
    p_value: float


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

    if polynomial_order == 0:  # Special case, just de-mean
        resid = data - data.mean(axis=0)
    else:
        trends = np.vander(np.arange(float(data.shape[0])), N=polynomial_order + 1)
        beta = np.linalg.pinv(trends).dot(data)
        resid = data - np.dot(trends, beta)

    if data.ndim == 2 and int(axis) == 1:
        resid = resid.T

    return resid


def add_detrend_column(
    data: pl.DataFrame,
    assets: list[str] | None = None,
    polynomial_orders: list[int] = [0, 1, 2, 3],
    axis: int = 0,
) -> pl.DataFrame:
    if assets is None:
        assets = data.select(
            cs.numeric()
        ).columns  # only get numeric columns (ie no dates)

    asset_arrays = data.select(assets).to_numpy()

    new_cols: list[pl.Series] = []
    for p in polynomial_orders:
        detrended = deterministic_detrend(asset_arrays, polynomial_order=p, axis=axis)

        for i, asset in enumerate(assets):
            new_cols.append(
                pl.Series(
                    name=f"{asset}_detrended_p{p}",
                    values=detrended[:, i],
                ).cast(pl.Float64)
            )

    return data.with_columns(new_cols)


def add_detrend_columns_max(
    data: pl.DataFrame,
    assets: list[str],
    max_polynomial_order: int,
) -> pl.DataFrame:
    polynomial_orders = list(range(0, max_polynomial_order + 1))
    return add_detrend_column(
        data=data, assets=assets, polynomial_orders=polynomial_orders
    )


def add_differenced_columns(
    data: pl.DataFrame,
    assets: list[str],
    difference: int = 1,
    keep_all: bool = True,
) -> pl.DataFrame:
    if difference < 1:
        raise ValueError("difference must be >= 1")

    diffs = range(1, difference + 1) if keep_all else [difference]

    return data.with_columns(
        [pl.col(assets).diff(d).name.suffix(f"_diff_{d}") for d in diffs]
    )


# INFO: inspired by statsmodels (especially FFT part)
def autocovariance(
    data: NDArray[np.floating], lag_length: int = 10, use_fft: bool = True
) -> dict[str, float]:
    if np.isnan(data).any():
        raise ValueError("Your array contains Nans, please fix")
    # basic 1-d check
    if data.ndim != 1:
        raise ValueError(f"Data must be 1D, currently {data.ndim}")
    # demean by default
    demeaned_array: NDArray[np.floating] = data - data.mean()
    n = demeaned_array.shape[0]

    if not use_fft:
        auto_covariance = np.empty(lag_length + 1)
        auto_covariance[0] = demeaned_array @ demeaned_array  # lag 0
        for lag in range(lag_length):
            auto_covariance[lag + 1] = (
                demeaned_array[lag + 1 :] @ demeaned_array[: -(lag + 1)]
            )
    else:
        n_fft = fft.next_fast_len(target=2 * n - 1, real=True)
        fourier_transform = np.fft.fft(demeaned_array, n=n_fft)
        auto_covariance = np.fft.ifft(
            fourier_transform * np.conjugate(fourier_transform)
        ).real
        auto_covariance = auto_covariance[: lag_length + 1].copy()

    auto_covariance /= n - np.arange(lag_length + 1)  # adjustment

    return {f"lag_{i}": float(cov) for i, cov in enumerate(auto_covariance)}


def _autocorr_confint(autocorr_vals: NDArray[np.floating], n: int, alpha: float = 0.05):
    varacf = np.ones_like(autocorr_vals) / n
    varacf[0] = 0
    varacf[1] = 1.0 / n
    varacf[2:] *= 1 + 2 * np.cumsum(autocorr_vals[1:-1] ** 2)
    interval = stats.norm.ppf(1 - alpha / 2.0) * np.sqrt(varacf)
    confint = np.array(lzip(autocorr_vals - interval, autocorr_vals + interval))
    return confint


def _ljung_box_stat(acf: NDArray[np.floating], n: int) -> tuple[float, float]:
    """
    acf: array of autocorrelations for lags 1..m (NO lag 0)
    n: sample size of the original series
    """
    m = len(acf)
    if m == 0:
        return 0.0, 1.0

    k = np.arange(1, m + 1)
    stat_seq = n * (n + 2) * np.cumsum((acf**2) / (n - k))
    stat = stat_seq[-1]
    p_val = stats.chi2.sf(stat, m)
    return float(stat), float(p_val)


def autocorrelation(
    data: NDArray[np.floating],
    lag_length: int = 10,
    use_fft: bool = True,
    confint_alpha: float = 0.05,
) -> dict[str, AutoCorrelation]:
    if np.isnan(data).any():
        raise ValueError("Your array contains Nans, please fix")
    if confint_alpha <= 0:
        raise ValueError("Your chosen alpha must be between 0 and 1")
    autocovariances = autocovariance(data=data, lag_length=lag_length, use_fft=use_fft)
    acov_vals = np.asarray(list(autocovariances.values()))
    autocorr_vals = acov_vals[: lag_length + 1] / acov_vals[0]
    n = len(data)
    confint = _autocorr_confint(autocorr_vals=autocorr_vals, n=n, alpha=confint_alpha)
    return {
        f"lag_{lag}": AutoCorrelation(
            lower=float(confint[lag, 0]),
            value=float(autocorr_vals[lag]),
            upper=float(confint[lag, 1]),
            p_value=_ljung_box_stat(autocorr_vals[1 : lag + 1], n=n)[1],
        )
        for lag in range(lag_length + 1)
    }


def get_aic(log_likelihood: float, n_parameters: int) -> float:
    return -2 * log_likelihood + 2 * n_parameters


def get_bic(log_likelihood: float, n_obs: int, n_parameters: int) -> float:
    return -2 * log_likelihood + np.log(n_obs) * n_parameters


def get_akicc(sample_size: int, rho: float, n_parameters: int) -> float:
    """Approximate Corrected Kullback Information Criterion (AKICc), Spectrum-compatible."""
    if n_parameters >= sample_size - 2:
        # denominator (N-k-2) would be <= 0
        return np.inf
    return (
        np.log(rho)
        + n_parameters / (sample_size * (sample_size - n_parameters))
        + (3.0 - (n_parameters + 2.0) / sample_size)
        * ((n_parameters + 1.0) / (sample_size - n_parameters - 2.0))
    )

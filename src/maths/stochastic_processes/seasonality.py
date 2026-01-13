from dataclasses import dataclass
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from numpy._typing._array_like import NDArray
from scipy.stats import f as f_dist
from typing_extensions import Literal

from maths.stochastic_processes.base import format_hyp_test_result

SEASONAL_PERIODS = Literal["weekly", "monthly", "quarterly", "semi-annual", "annual"]

trading_days_seasonal_periods_dict = {
    "weekly": 5,
    "monthly": 21,
    "semi_annual": 126,
    "annual": 252,
    "quarterly": 63,
}


class SeasonalBin(NamedTuple):
    harmonic: int
    power: float
    idx: int
    frequency: float
    period: float


@dataclass(frozen=True)
class Periodogram:
    power: NDArray[np.floating]
    freq_cycles: NDArray[np.floating]
    freq_radians: NDArray[np.floating]
    sample_count: int


class FStatRes(NamedTuple):
    stat: float
    numerator_degree_of_freedom: int
    denominator_degree_of_freedom: int


def plot_periodogram(freq, spectrum, max_period=260):
    mask = freq > 0
    periods = 1.0 / freq[mask]
    spec = spectrum[mask]

    fig, ax = plt.subplots()
    ax.plot(periods, spec)
    ax.set_xlim(1, max_period)
    ax.set_xlabel("Trading Days per cycle")
    ax.set_ylabel("Power")
    ax.set_title("Periodogram ")
    return ax


def _make_len_multiple_of_seasonal_period(
    data: NDArray[np.floating], seasonal_period: int
) -> NDArray[np.floating]:
    n = data.shape[0]
    extra = n % seasonal_period
    if extra == 0:
        return data
    return data[extra:]


def periodogram(data: NDArray[np.floating]) -> Periodogram:
    sample_count = data.size

    total_sum = data.sum()
    total_energy = float(np.dot(data, data))

    if total_energy == 0.0:
        power = np.zeros(1 + sample_count // 2, dtype=float)
    else:
        spectrum = np.fft.rfft(data)
        power = np.empty(spectrum.shape, dtype=float)

        # DC component
        power[0] = (total_sum * total_sum) / total_energy

        # Interior frequencies
        if sample_count % 2 == 0:  # even length: has Nyquist bin
            if power.size > 2:
                power[1:-1] = 2.0 * (np.abs(spectrum[1:-1]) ** 2) / total_energy
            # Nyquist frequency (no doubling)
            power[-1] = (np.abs(spectrum[-1]) ** 2) / total_energy
        else:  # odd length: no Nyquist bin
            if power.size > 1:
                power[1:] = 2.0 * (np.abs(spectrum[1:]) ** 2) / total_energy

    # Frequency grids
    k = np.arange(power.size, dtype=float)
    freq_cycles = k / sample_count
    freq_radians = 2.0 * np.pi * freq_cycles

    return Periodogram(
        power=power,
        freq_cycles=freq_cycles,
        freq_radians=freq_radians,
        sample_count=sample_count,
    )


def get_seasonal_bins(
    power: NDArray[np.floating],
    frequency: NDArray[np.floating],
    n_samples: int,
    seasonal_period: int,
) -> list[SeasonalBin]:
    if n_samples % seasonal_period != 0:
        raise ValueError(
            f"n_samples={n_samples} must be a multiple of cycle_period={seasonal_period} "
        )

    n_harmonics = (seasonal_period - 1) // 2
    cycles_in_sample = n_samples // seasonal_period

    out: list[SeasonalBin] = []
    for k in range(1, n_harmonics + 1):
        idx = k * cycles_in_sample
        pwr = float(power[idx])
        f = float(frequency[idx])
        out.append(
            SeasonalBin(harmonic=k, power=pwr, idx=idx, frequency=f, period=1.0 / f)
        )

    return out


def get_period_fstat(periodogram: Periodogram, seasonal_period: int) -> FStatRes:
    seasonal_bins = get_seasonal_bins(
        power=periodogram.power,
        frequency=periodogram.freq_cycles,
        n_samples=periodogram.sample_count,
        seasonal_period=seasonal_period,
    )
    seasonal_period_power = np.asarray([bin.power for bin in seasonal_bins]).sum()

    n_harmonics = (seasonal_period - 1) // 2
    degrees_of_freedom = 2 * n_harmonics

    # Needs Nyquist harmonic if seasonal period is even
    if seasonal_period % 2 == 0:
        degrees_of_freedom += 1
        seasonal_period_power += periodogram.power[-1]

    remaining_degrees_of_freedom = (
        periodogram.sample_count - degrees_of_freedom - 1
    )  # 1 here accounts for intercept

    f_stat_numerator = remaining_degrees_of_freedom * seasonal_period_power
    f_stat_denominator = (
        periodogram.sample_count - seasonal_period_power - periodogram.power[0]
    ) * degrees_of_freedom

    return FStatRes(
        stat=f_stat_numerator / f_stat_denominator,
        numerator_degree_of_freedom=degrees_of_freedom,
        denominator_degree_of_freedom=remaining_degrees_of_freedom,
    )
    # return f_stat_numerator / f_stat_denominator


def get_periodogram_p_val(
    periodogram: Periodogram, seasonal_period: int
) -> tuple[float, float]:
    t_stat = get_period_fstat(periodogram=periodogram, seasonal_period=seasonal_period)
    return t_stat.stat, f_dist.sf(
        t_stat.stat,
        t_stat.numerator_degree_of_freedom,
        t_stat.denominator_degree_of_freedom,
    )


def periodogram_seasonality_test(
    data: NDArray[np.floating], seasonal_period: SEASONAL_PERIODS
):
    seasonal_period_n = trading_days_seasonal_periods_dict[seasonal_period]
    data = _make_len_multiple_of_seasonal_period(
        data=data, seasonal_period=seasonal_period_n
    )
    period = periodogram(data=data)
    stat, p_val = get_periodogram_p_val(
        periodogram=period, seasonal_period=seasonal_period_n
    )

    return format_hyp_test_result(
        p_val=p_val, stat=stat, null=f"No seasonality (period={seasonal_period})"
    )

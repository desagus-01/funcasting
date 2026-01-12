# %% imports + data
from dataclasses import dataclass
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from numpy._typing._array_like import NDArray

from maths.helpers import add_detrend_columns_max, add_differenced_columns
from utils.template import get_template

aapl_rd = get_template().asset_info.risk_drivers.select("AAPL")

aapl_det = add_detrend_columns_max(aapl_rd, ["AAPL"], max_polynomial_order=2)

aapl_final = add_differenced_columns(aapl_det, ["AAPL"]).drop_nulls()

aapl_final


# %% plot func


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


# %% periodogram


# trying to replicate shrink
def make_len_multiple_of_period(
    data: NDArray[np.floating], period: int
) -> NDArray[np.floating]:
    n = data.shape[0]
    extra = n % period
    if extra == 0:
        return data
    return data[extra:]


x = make_len_multiple_of_period(
    aapl_final.select("AAPL_diff_1").to_numpy().flatten(), 5
)

x.shape


# %% periodogram_custom
@dataclass(frozen=True)
class JDPeriodogram:
    power: NDArray[np.floating]
    freq_cycles: NDArray[np.floating]
    freq_radians: NDArray[np.floating]
    sample_count: int


def jdplus_periodogram_fft(signal: NDArray[np.floating]) -> JDPeriodogram:
    """
    FFT-based implementation equivalent to jdplus Periodogram.of(y)
    for arrays with no NaNs.
    """
    sample_count = signal.size

    total_sum = signal.sum()
    total_energy = float(np.dot(signal, signal))

    if total_energy == 0.0:
        power = np.zeros(1 + sample_count // 2, dtype=float)
    else:
        spectrum = np.fft.rfft(signal)
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

    return JDPeriodogram(
        power=power,
        freq_cycles=freq_cycles,
        freq_radians=freq_radians,
        sample_count=sample_count,
    )


periodogram = jdplus_periodogram_fft(x)

plot_periodogram(freq=periodogram.freq_cycles, spectrum=periodogram.power)

# %% Harmonic selection


class HarmonicPeriod(NamedTuple):
    harmonic: int
    idx: int
    frequency: float
    period: float


def harmonic_bins(
    frequency: NDArray[np.floating], n_samples: int, cycle_period: int
) -> list[HarmonicPeriod]:
    if n_samples % cycle_period != 0:
        raise ValueError(
            f"n_samples={n_samples} must be a multiple of cycle_period={cycle_period} "
            "to match JD buildF bin alignment."
        )

    n_harmonics = (cycle_period - 1) // 2
    cycles_in_sample = n_samples // cycle_period  # JD's m

    out: list[HarmonicPeriod] = []
    for k in range(1, n_harmonics + 1):
        idx = k * cycles_in_sample
        f = float(frequency[idx])
        out.append(HarmonicPeriod(harmonic=k, idx=idx, frequency=f, period=1.0 / f))

    return out


harmonic_bins(periodogram.freq_cycles, x.size, 5)

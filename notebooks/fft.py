# %% imports
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from maths.helpers import add_detrend_columns_max, add_differenced_columns
from utils.template import get_template

# %% load once
rd = get_template().asset_info.risk_drivers

x = add_detrend_columns_max(rd, ["AAPL"], max_polynomial_order=2)
a = x.select("AAPL_detrended_p2").to_numpy()

# %%

aapl = (
    add_differenced_columns(rd, ["AAPL"]).select("AAPL_diff_1").drop_nulls().to_numpy()
)
aapl_r = aapl.ravel()


# %%

f, pxx = signal.welch(
    aapl_r,
    fs=252.0,  # trading days
    detrend=False,
    # window="boxcar",
    scaling="spectrum",
)

mask = f > 0
f2 = f[mask]
pxx2 = pxx[mask]

# Peak picking with some guardrails (tune these)
peak_idx, props = signal.find_peaks(
    pxx2,
    prominence=np.percentile(pxx2, 95)
    * 0.05,  # example: 5% of the 95th percentile level
)

prom = props["prominences"]


def plot_peak_prominence_table(f, peak_idx, prom, fs=252.0, top_n=20):
    order = np.argsort(prom)[::-1][: min(top_n, len(prom))]
    pk = peak_idx[order]
    pr = prom[order]

    periods_days = fs / f[pk]

    fig, ax = plt.subplots()
    ax.barh(np.arange(len(pk)), pr)
    ax.set_yticks(np.arange(len(pk)))
    ax.set_yticklabels([f"{d:.1f}d" for d in periods_days])
    ax.invert_yaxis()
    ax.set_xlabel("Prominence")
    ax.set_title("Most Prominent Spectral Peaks (by period)")
    return ax


plot_peak_prominence_table(f2, peak_idx, prom, fs=252.0, top_n=20)
plt.show()


# %%


def plot_periodogram(ts, detrend=False, ax=None):
    frequencies, spectrum = signal.welch(
        ts,
        fs=126.0,  # trading days
        detrend=detrend,
        # window="boxcar",
        scaling="spectrum",
    )

    if ax is None:
        _, ax = plt.subplots()

    ax.step(frequencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")

    return ax


plot_periodogram(aapl_r)


# %%


def plot_welch_peaks_fast(
    x,
    fs=252.0,
    nperseg=256,
    detrend="constant",
    window="bartlett",  # same as paper
    top_k=15,
    prominence_q=0.95,
    label_periods=True,
    ax=None,
):
    x = np.asarray(x).reshape(-1).astype(float, copy=False)

    f, pxx = signal.welch(
        x,
        fs=fs,
        detrend=detrend,
        window=window,
        nperseg=min(nperseg, len(x)),
        scaling="spectrum",
    )

    # remove DC for log-x and cleaner peak picking
    mask = f > 0
    f = f[mask]
    pxx = pxx[mask]

    # One cheap threshold: only consider peaks with prominence above a high quantile
    prom_thresh = np.quantile(pxx, prominence_q)

    peak_idx, props = signal.find_peaks(
        pxx,
        prominence=prom_thresh,
        distance=2,  # tiny guardrail, keeps it fast and avoids adjacent-bin "duplicates"
    )

    if peak_idx.size == 0:
        if ax is None:
            _, ax = plt.subplots()
        ax.step(f, pxx, where="mid", color="purple", linewidth=1)
        ax.set_xscale("log")
        ax.set_title("Welch PSD (no peaks above threshold)")
        ax.set_ylabel("Variance")
        return ax

    prom = props["prominences"]

    # Keep only top_k peaks by prominence (O(n) argpartition, fast)
    k = min(top_k, peak_idx.size)
    top = np.argpartition(prom, -k)[-k:]
    peak_idx = peak_idx[top]
    prom = prom[top]

    # Sort those top peaks by frequency for nicer plotting
    order = np.argsort(f[peak_idx])
    peak_idx = peak_idx[order]
    prom = prom[order]

    if ax is None:
        _, ax = plt.subplots()

    ax.step(f, pxx, where="mid", color="purple", linewidth=1)
    ax.set_xscale("log")
    ax.set_ylabel("Variance")
    ax.set_title("Welch PSD with Top Peaks (fast)")

    # Peak markers
    ax.plot(f[peak_idx], pxx[peak_idx], "o", markersize=5)

    # Vectorized prominence stems (fast)
    y_top = pxx[peak_idx]
    y_bot = y_top - prom
    ax.vlines(f[peak_idx], y_bot, y_top, linewidth=1)

    # Optional labels (this is the slow part if many)
    if label_periods:
        periods_days = fs / f[peak_idx]
        for fx, yt, pdays in zip(f[peak_idx], y_top, periods_days):
            ax.annotate(
                f"{pdays:.1f}d",
                (fx, yt),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    return ax


plot_welch_peaks_fast(aapl_r)

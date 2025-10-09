import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from polars import DataFrame

from data_types.vectors import ProbVector


def choose_hist_bin(data: NDArray[np.floating] | ProbVector) -> int:
    """
    Uses square root rule to get a good number for histogram bin
    """
    return int(np.sqrt(len(data)))


def plt_prob_eval(prob_vector: ProbVector) -> None:
    p_cum = np.cumsum(prob_vector)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(prob_vector)
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 4))
    axs[0].set_ylabel("% Assigned")

    axs[1].plot(p_cum)
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("% Cumulative")

    axs[2].hist(prob_vector, bins=choose_hist_bin(prob_vector))
    axs[2].xaxis.set_major_formatter(mtick.PercentFormatter(1.0, 4))

    plt.tight_layout()
    plt.show()


# made the below w/ AI -- cba
def plt_returns_dens(data: DataFrame) -> None:
    df = data.to_pandas()
    x = df["date"].to_numpy()
    y = df["return"].to_numpy()

    # Symmetric y-limits with a bit of robustness to outliers
    ylim = np.nanpercentile(np.abs(y), 99)
    ylim = float(ylim if np.isfinite(ylim) and ylim > 0 else np.nanmax(np.abs(y)))
    ylim *= 6

    # Figure layout: wide main panel + slim histogram on the right
    fig = plt.figure(figsize=(12, 2.4), dpi=150)
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[20, 3], wspace=0.05, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax)

    # --- main scatter ---
    ax.scatter(x, y, s=8, c="black", alpha=0.6, linewidths=0)

    ax.set_ylim(-ylim, ylim)
    ax.set_ylabel("return", labelpad=6)

    # Dates on x-axis: compact formatting
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # --- side histogram (horizontal) ---
    bins = choose_hist_bin(y)
    ax_hist.hist(
        y,
        bins=bins,
        orientation="horizontal",
        color="gray",
        alpha=0.7,
        edgecolor="none",
    )
    ax_hist.set_xlabel("")  # not needed
    ax_hist.set_xlim(left=0)
    ax_hist.tick_params(axis="y", labelleft=False)  # keep only left plot's y labels

    # Clean up histogram spines/ticks
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["left"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.xaxis.set_ticks([])

    # Tight layout with a bit of breathing room for the title
    plt.tight_layout()
    plt.show()

import itertools

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from numpy.typing import NDArray
from polars import DataFrame

from data_types.vectors import ProbVector


def choose_hist_bin(data: NDArray[np.floating] | ProbVector) -> int:
    """
    Uses square root rule to get a good number for histogram bin
    """
    return int(np.sqrt(len(data)))


def plt_prob_eval(
    prob_vector: ProbVector, data: DataFrame, constraints: list[float] | None = None
) -> None:
    df = data.to_pandas()
    x = df["date"].drop_duplicates().to_numpy()  # get shared x-axis

    p_cum = np.cumsum(prob_vector)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)  # <-- share x-axis

    axs[1].plot(x, prob_vector)  # <-- now x is date
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 3))
    axs[1].set_ylabel("% Assigned")

    axs[2].plot(x, p_cum)
    axs[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
    axs[2].set_ylabel("% Cumulative")

    # Plot returns density on axs[2]
    plt_returns_dens(data, axs[0], constraints)

    axs[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))

    plt.tight_layout()
    plt.show()


# made the below w/ AI -- cba
def plt_returns_dens(
    data: DataFrame, ax: plt.Axes, constraints: list[float] | None
) -> None:
    df = data.to_pandas()
    x = df["date"].to_numpy()
    y = df["return"].to_numpy()

    ylim = np.nanpercentile(np.abs(y), 99)
    ylim = float(ylim if np.isfinite(ylim) and ylim > 0 else np.nanmax(np.abs(y)))
    ylim *= 3

    ax.scatter(x, y, s=8, c="black", alpha=0.6, linewidths=0)
    ax.set_ylim(-ylim, ylim)
    ax.set_ylabel("Asset Returns", labelpad=6)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    if constraints is not None:
        color_cycle = itertools.cycle(["red", "blue", "green", "orange", "purple"])
        for value, color in zip(constraints, color_cycle):
            ax.axhline(y=value, color=color, linestyle="--", linewidth=1.5, alpha=0.8)

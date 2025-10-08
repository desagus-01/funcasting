import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from numpy.typing import NDArray

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

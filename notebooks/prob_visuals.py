import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from data_types.vectors import ProbVector
from get_data import get_example_assets
from maths.prob_vectors import (
    exp_decay_probs,
    state_crisp_conditioning,
    state_smooth_conditioning,
)


def plot_post_prob(prob_vector: ProbVector) -> None:
    p_cum = np.cumsum(prob_vector)

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle("Posterior Probability")
    axs[0].plot(prob_vector)
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 5))
    axs[0].set_ylabel("% Assigned")

    axs[1].plot(p_cum)
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("% Cumulative")

    plt.show()


# getting data
tickers = ["AAPPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments
increms_n = increms_df.height
u = increms_df["AAPL"].to_numpy().mean()

# smoothing methods
half_life = 50
ex_state_conds = np.random.choice([True, False], size=increms_n)
exp_dec_probs = exp_decay_probs(increms_df["AAPL"].to_numpy(), half_life)
state_crisp_probs = state_crisp_conditioning(increms_n, ex_state_conds)
state_smooth_probs = state_smooth_conditioning(
    increms_df["AAPL"].to_numpy(), u, half_life, 1
)


plot_post_prob(exp_dec_probs)

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.animation import FuncAnimation

# --- your imports and helpers ---
from get_data import get_example_assets
from maths.constraints import view_on_mean
from maths.prob_vectors import entropy_pooling_probs, uniform_probs


# (keeping your choose_hist_bin)
def choose_hist_bin(data) -> int:
    return int(np.sqrt(len(data)))


# --- (optional) if you already have a plt_prob_eval elsewhere, no problem.
# We'll do an animation-specific plotting pipeline below. ---

# --------- DATA SETUP (same as yours) ----------
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")
increms_np = increms_df.to_numpy()
increms_n = increms_df.height
u = increms_np.mean(axis=0) - 0.019
half_life = 3

prior = uniform_probs(increms_n)
mean_ineq = view_on_mean(
    increms_np, u, ["inequality"] * (u.shape[0]), ["equal_less"] * (u.shape[0])
)


# --------- PROBABILITY FACTORY ----------
def probs_at_conf(conf: float):
    """
    Return probability vector for a given confidence level in [0, 1].
    Uses entropy_pooling_probs(prior, mean_ineq, conf).
    """
    # include_diags=False for speed during animation (set True if you need them)
    pv = entropy_pooling_probs(prior, mean_ineq, conf, include_diags=False)
    # ensure np.array
    return np.asarray(pv, dtype=float)


# --------- ANIMATION ----------
# Precompute bins for the histogram so bars remain stable across frames
p0 = probs_at_conf(0.0)
n_bins = max(1, choose_hist_bin(p0))  # guard against 0
# Lock bin edges to [0,1] range to avoid jittering bars
bins = np.linspace(0.0, 1.0, n_bins + 1)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
bin_widths = np.diff(bins)

fig, axs = plt.subplots(3, 1, figsize=(10, 12))
ax_prob, ax_cum, ax_hist = axs

# Lines we’ll update
x_idx = np.arange(len(p0))
(line_prob,) = ax_prob.plot(x_idx, p0, lw=1.8)
ax_prob.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 4))
ax_prob.set_ylabel("% Assigned")

pc0 = np.cumsum(p0)
(line_cum,) = ax_cum.plot(x_idx, pc0, lw=1.8)
ax_cum.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
ax_cum.set_xlabel("Time Steps")
ax_cum.set_ylabel("% Cumulative")

# Histogram drawn as fixed bars we’ll just update heights on
counts, _ = np.histogram(p0, bins=bins)
# Normalize counts to % of mass in each bin
counts = counts / counts.sum() if counts.sum() > 0 else counts
bars = ax_hist.bar(
    bin_centers, counts, width=bin_widths, align="center", edgecolor="black"
)
ax_hist.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, 4))
ax_hist.set_xlabel("Probability Value")
ax_hist.set_ylabel("% of Mass (by bin)")

# Titles + a small text showing current confidence
fig.suptitle("Entropy Pooling Confidence Sweep", fontsize=16)
conf_text = fig.text(0.92, 0.96, "conf = 0.00", ha="right", va="center", fontsize=12)

# Y-limits stay stable to reduce flicker. You can tweak if you prefer autoscale.
ax_prob.set_ylim(0, max(0.01, p0.max()) * 1.15)
ax_cum.set_ylim(0, 1.0)
ax_hist.set_ylim(0, max(0.05, max(counts) * 1.25))

plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # leave space for conf_text

# Frames: smooth sweep 0 → 1 and back (optional). If you only want 0→1, remove the second linspace.
conf_forward = np.linspace(0.0, 1.0, 61)
conf_cycle = conf_forward + 0.05  # just 0→1


def update(frame_conf):
    pv = probs_at_conf(frame_conf)

    # Update lines
    line_prob.set_data(x_idx, pv)
    pc = np.cumsum(pv)
    line_cum.set_data(x_idx, pc)

    # Keep axes sized nicely (remove if you prefer fixed)
    ax_prob.set_ylim(0, max(0.01, pv.max()) * 1.15)
    # ax_cum stays [0,1]

    # Update histogram bars using fixed bins
    counts, _ = np.histogram(pv, bins=bins)
    counts = counts / counts.sum() if counts.sum() > 0 else counts
    for rect, h in zip(bars, counts):
        rect.set_height(h)

    # dynamic y-limit for hist (optional)
    ax_hist.set_ylim(0, max(0.05, counts.max() * 1.25 if counts.size else 0.05))

    # Update confidence label
    conf_text.set_text(f"conf = {frame_conf:0.2f}")

    return (line_prob, line_cum, *bars, conf_text)


anim = FuncAnimation(
    fig,
    update,
    frames=conf_cycle,
    interval=60,  # ms per frame (≈16 fps → 60ms ≈ ~16.7fps)
    blit=False,  # blit=False is simplest for multi-axes updates
)

plt.show()

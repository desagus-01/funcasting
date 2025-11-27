import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from models.scenarios import ScenarioProb
from models.views_builder import ViewBuilder
from utils.template import get_template

info_all = get_template()

scenarios = ScenarioProb.default_inst(info_all.increms_df)

u_vec = scenarios.scenarios.drop("date").to_numpy().mean(axis=0)


views = (
    ViewBuilder(scenarios.scenarios)
    .mean({"AAPL": 0.001, "GOOG": -0.03}, ["equal", "equal"])
    .build()
)

scenarios_wviews = scenarios.add_views(views).apply_views()

# Simple monte carlo ex for AAPL

flex_probs = scenarios_wviews.prob

latest_value = (
    info_all.raw_data.filter(pl.col("date") == pl.max("date"))
    .select("AAPL")
    .to_numpy()
    .item()
)


sampler = np.random.default_rng()


aapl_vals = scenarios_wviews.scenarios.select("AAPL").to_numpy()

n_steps = 1000
n_paths = 500

# 1) Sample all shocks at once: shape (n_paths, n_steps)
samples = sampler.choice(
    aapl_vals,
    size=(n_paths, n_steps),
    p=flex_probs,
)
increments = np.cumsum(samples, axis=1)  # shape (n_paths, n_steps)
paths = latest_value + increments

paths_T = paths.T

t = np.arange(1, n_steps + 1)

plt.figure(figsize=(8, 5))
for i in range(n_paths):
    plt.plot(t, paths[i], linewidth=0.8, alpha=0.4)

plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Monte Carlo AAPL Paths")
plt.grid(True)
plt.tight_layout()
plt.show()

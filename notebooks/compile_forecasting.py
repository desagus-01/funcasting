import matplotlib.pyplot as plt
import numpy as np

from maths.distributions import uniform_probs
from methods.forecasting_pipeline import (
    draw_invariant_shock,
    multivariate_forecasting_info,
)
from methods.fos_2 import (
    simulate_asset_paths,
)
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# %%
x = multivariate_forecasting_info(data)
inv_df = x.invariants
prob_ex = uniform_probs(inv_df.height)

# %% for AAPL only for now
models = x.models
n_paths = 10000
asset = "AAPL"
aapl_model = models[asset]
h = 200


params = aapl_model.model.compile_params()
invariant_shock = draw_invariant_shock(
    inv_df,
    assets=["AAPL", "GOOG"],
    prob_vector=prob_ex,
    horizon=h,
    n_sims=n_paths,
    seed=1,
    method="cma",
    target_copula="t",
)
innov_aapl = invariant_shock[:, :, 0]  # (n_paths, h)


def plot_simulation_results(
    y_paths: np.ndarray,
    *,
    max_paths: int = 30,
    quantiles=(0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
    integrate: bool = False,
    logP0: float = 0.0,
    title: str | None = None,
):
    """
    Plot simulation paths and a fan chart.

    Parameters
    ----------
    y_paths : array, shape (n_paths, horizon)
        Output from simulate_asset_paths.
    max_paths : int
        How many sample paths to draw as lines.
    quantiles : tuple
        Quantiles for the fan chart (must include 0.5 for the median).
    integrate : bool
        If True, plots cumulative sum along time (useful for log-price diffs).
    logP0 : float
        Starting log price if integrate=True (e.g., np.log(last_price)).
    title : str | None
        Figure title.
    """
    y = np.asarray(y_paths, dtype=float)
    if y.ndim != 2:
        raise ValueError("y_paths must be a 2D array (n_paths, horizon).")

    n_paths, horizon = y.shape
    x = np.arange(1, horizon + 1)

    if integrate:
        y_plot = logP0 + np.cumsum(y, axis=1)
        y_label = "Cumulative log-price (logP0 + cumsum)"
    else:
        y_plot = y
        y_label = "Simulated values"

    # Select a subset of paths to plot
    rng = np.random.default_rng(0)
    k = min(max_paths, n_paths)
    idx = (
        rng.choice(n_paths, size=k, replace=False)
        if n_paths > k
        else np.arange(n_paths)
    )

    # Compute quantiles across paths for each time step
    qs = np.array(quantiles, dtype=float)
    qvals = np.quantile(y_plot, qs, axis=0)  # shape (len(qs), horizon)

    # Identify median index (must exist)
    if not np.any(np.isclose(qs, 0.50)):
        raise ValueError("quantiles must include 0.50 for the median.")
    med_i = int(np.where(np.isclose(qs, 0.50))[0][0])

    plt.figure(figsize=(12, 6))

    # Fan chart: fill between symmetric quantiles around median if provided
    # We'll pair (0.25,0.75), (0.05,0.95), (0.01,0.99) when available.
    pairs = [(0.25, 0.75), (0.05, 0.95), (0.01, 0.99)]
    for lo, hi in pairs:
        if np.any(np.isclose(qs, lo)) and np.any(np.isclose(qs, hi)):
            lo_i = int(np.where(np.isclose(qs, lo))[0][0])
            hi_i = int(np.where(np.isclose(qs, hi))[0][0])
            plt.fill_between(x, qvals[lo_i], qvals[hi_i], alpha=0.20, linewidth=0)

    # Median line
    plt.plot(x, qvals[med_i], linewidth=2, label="Median")

    # Sample paths
    for j in idx:
        plt.plot(x, y_plot[j], alpha=0.35, linewidth=1)

    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.title(
        title or ("Simulated paths (integrated)" if integrate else "Simulated paths")
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


y_paths = simulate_asset_paths(forecast_model=aapl_model, innovations=innov_aapl)
logP_paths = 0.0 + np.cumsum(y_paths, axis=1)
P_paths = np.exp(logP_paths)
plot_simulation_results(
    P_paths,
    integrate=False,
    title="Simulated price paths (starting at 1.0)",
)

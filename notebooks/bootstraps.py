import numpy as np
import polars as pl

from maths.distributions import uniform_probs
from maths.sampling import weighted_bootstrapping
from methods.cma import CopulaMarginalModel
from methods.forecasting_pipeline import (
    multivariate_forecasting_info,
    next_step_bootstrap,
    next_step_historical,
)
from utils.helpers import compensate_prob
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# %%
x = multivariate_forecasting_info(data)
inv_df = x.invariants
prob_ex = uniform_probs(inv_df.height)
# %%

models = x.models

ns = next_step_bootstrap(
    invariants_df=inv_df,
    assets=[asset for asset in models.keys()],
    models=models,
    prob_vector=prob_ex,
    n_sims=inv_df.height - 200,
)

nsh = next_step_historical(
    invariants_df=inv_df,
    assets=[asset for asset in models.keys()],
    models=models,
    prob_vector=prob_ex,
)
# %%
# next_step_copula_marginal(
#     invariants_df=inv_df,
#     assets=[asset for asset in models.keys()],
#     models=models,
#     prob_vector=prob_ex,
#     target_copula="t",
#     seed=1,
#     target_marginals={"AAPL": "norm", "fake": "t"},
# )
# %%
inv_nn = x.invariants.drop_nulls()
prob_2 = compensate_prob(prob_ex, 1)
cma = CopulaMarginalModel.from_data_and_prob(inv_nn, prob_2)
cma_2 = cma.update_copula(seed=1, target_copula="t")

# %%
copula, prob = cma_2.copula, cma_2.prob
one_path = weighted_bootstrapping(copula, prob, n_samples=100, seed=1)
# %%
assets = list(one_path.columns)
A = len(assets)

H = 20
MC = 100
seed = 1

X = np.empty((H, MC, A), dtype=float)

# variance state per scenario
h = np.zeros((MC, A), dtype=float)

# params (default to 0)
omega = np.zeros(A, dtype=float)
alpha = np.zeros(A, dtype=float)
beta = np.zeros(A, dtype=float)
gamma = np.zeros(A, dtype=float)
mu = np.zeros(A, dtype=float)

# ---- init per-asset params + initial variance ----
for i, asset in enumerate(assets):
    model = models.get(asset)

    # mean params (mean_model can be None)
    mean_model = None if model is None else model.form.mean_model
    mp = (
        {}
        if (mean_model is None or getattr(mean_model, "params", None) is None)
        else mean_model.params
    )
    mu[i] = float(mp.get("mean", 0.0))  # constant mean fallback

    # vol params (volatility_model can be None)
    vol_model = None if model is None else model.form.volatility_model
    vp = (
        {}
        if (vol_model is None or getattr(vol_model, "params", None) is None)
        else vol_model.params
    )
    omega[i] = float(vp.get("omega", 0.0))
    alpha[i] = float(vp.get("alpha[1]", 0.0))
    beta[i] = float(vp.get("beta[1]", 0.0))
    gamma[i] = float(vp.get("gamma[1]", 0.0)) if "gamma[1]" in vp else 0.0

    # initial variance from state0.vol (VolState is not subscriptable)
    vol_state = None if model is None else model.state0.vol

    # Try common scalar/array attributes first; otherwise fall back to float(vol_state) if possible.
    v0 = None
    if vol_state is None:
        v0 = 0.0
    elif isinstance(vol_state, (int, float, np.floating)):
        v0 = float(vol_state)
    else:
        # common attribute names that might exist on VolState
        for name in ("var", "variance", "h", "sigma2", "cond_var", "v"):
            val = getattr(vol_state, name, None)
            if val is None:
                continue
            if isinstance(val, (list, tuple, np.ndarray)):
                v0 = float(val[-1]) if len(val) else 0.0
            elif isinstance(val, (int, float, np.floating)):
                v0 = float(val)
            break

        if v0 is None:
            # last resort: try casting vol_state directly
            try:
                v0 = float(vol_state)
            except Exception:
                v0 = 0.0

    h[:, i] = v0  # same initial variance for all scenarios

# ---- Monte Carlo recursion (time loop) ----
rng = np.random.default_rng(seed)

for k in range(H):
    eps_df: pl.DataFrame = weighted_bootstrapping(
        copula, prob, n_samples=MC, seed=int(rng.integers(0, 1_000_000_000))
    )
    E = eps_df.select(assets).to_numpy()  # (MC, A) innovations (already eps)

    sigma = np.sqrt(np.maximum(h, 0.0))  # (MC, A)

    X[k] = mu[None, :] + sigma * E  # (MC, A)

    Ineg = (E < 0.0).astype(float)  # (MC, A) for GJR term
    h = (
        omega[None, :]
        + alpha[None, :] * (E**2)
        + gamma[None, :] * (E**2) * Ineg
        + beta[None, :] * h
    )

paths_by_asset = {a: X[:, :, i] for i, a in enumerate(assets)}

import numpy as np
import polars as pl

from maths.distributions import uniform_probs
from maths.sampling import weighted_bootstrapping
from methods.forecasting_pipeline import (
    multivariate_forecasting_info,
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

# %%
# Forecast is made from a draw of innovation through bootstrap + functional form
# AAPL Ex


aapl = x.model["AAPL"]

p = aapl.form.volatility_model.params
s = aapl.state0.vol

eps_t = s.volatility_residuals[-1]  # ε_t
sig2_t = s.conditional_volatility_sq[-1]  # σ_t^2

I_neg = 1.0 if eps_t < 0 else 0.0

sig2_next = (
    p["omega"]
    + p["alpha[1]"] * (eps_t**2)
    + p["gamma[1]"] * I_neg * (eps_t**2)
    + p["beta[1]"] * sig2_t
)

sig_next = np.sqrt(sig2_next)

mu = aapl.form.mean_model.params["mean"]

# z_next must be a standardized innovation draw (mean 0, var 1)
inv_boot = weighted_bootstrapping(data=inv_df, prob_vector=prob_ex, n_samples=1)
z_next = inv_boot.select("AAPL").item()

eps_next = sig_next * z_next
r_next = mu + eps_next
r_next
# %%

inv_df.select(pl.var("AAPL"))

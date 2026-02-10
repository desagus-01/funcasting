from pprint import pprint as p

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
p(x)
# %%
inv_df = x.invariants
prob_ex = uniform_probs(inv_df.height)
inv_df
# %%
weighted_bootstrapping(data=inv_df, prob_vector=prob_ex, n_samples=1)
# %%

uni = x.models["fake"]

mean_res = uni.mean_model

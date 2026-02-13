from maths.distributions import uniform_probs
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
no_nulls_inv = inv_df.drop_nulls()
new_prob = compensate_prob(prob_ex, inv_df.height - no_nulls_inv.height)
inv_cma = CopulaMarginalModel.from_data_and_prob(no_nulls_inv, new_prob)
# %%
inv_cma.update_distribution(target_marginals={"AAPL": "t"}, target_copula="t")

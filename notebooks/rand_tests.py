import cvxpy as cp
import numpy as np

from get_data import get_example_assets
from helpers import get_corr_info, weighted_moments
from maths.constraints import view_on_corr, view_on_mean, view_on_ranking, view_on_std
from maths.core import (
    assign_constraint_equation,
    build_constraints,
    simple_entropy_pooling,
)
from maths.prob_vectors import uniform_probs
from maths.visuals import plt_prob_eval

# set-up
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")
increms_np = increms_df.to_numpy()
increms_n = increms_df.height
u = increms_np.mean(axis=0) - 0.019
half_life = 3
data_long = assets.increments.unpivot(
    on=tickers, value_name="return", variable_name="ticker", index="date"
)

prior = uniform_probs(increms_n)
posterior = cp.Variable(prior.shape[0])

corr_views = get_corr_info(increms_df)

for view in corr_views:
    view.corr = 1


views = view_on_corr(increms_df, corr_views, ["equal"] * 3)


probs = simple_entropy_pooling(prior, views, include_diags=True)


plt_prob_eval(probs, data_long)

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

x = get_corr_info(increms_df)
#
print(x)

# views = view_on_corr(increms_df, x, ["equal"] * 3)
# mu_ref, std_ref = weighted_moments(views[0].data, prior)
#
# lhs = views[0].data[0] * views[0].data[1] @ prior
#
# rhs = views[0].views_target * std_ref[0] * std_ref[1] + mu_ref[0] * mu_ref[1]
#
#
# probs = simple_entropy_pooling(prior, views)
#
#
# plt_prob_eval(probs, data_long)

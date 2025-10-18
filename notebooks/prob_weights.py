import cvxpy as cp
from polars.meta import build

from get_data import get_example_assets
from maths.constraints import view_on_mean, view_on_ranking
from maths.core import (
    assign_constraint_equation,
    build_constraints,
    simple_entropy_pooling,
)
from maths.prob_vectors import entropy_pooling_probs, state_smooth_probs, uniform_probs
from maths.visuals import plt_prob_eval

# set-up
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")
increms_np = increms_df.to_numpy()
increms_n = increms_df.height
u = increms_np.mean(axis=0) - 0.01
half_life = 3

data_long = assets.increments.unpivot(
    on=tickers, value_name="return", variable_name="ticker", index="date"
)


prior = uniform_probs(increms_n)


prior_2 = state_smooth_probs(
    data_array=increms_np[:, 0],
    half_life=half_life,
    kernel_type=2,
    reference=0.015,
)

posterior = cp.Variable(prior.shape[0])

# mean_view = view_on_mean()
rankings_view = view_on_ranking(increms_df, ["MSFT", "GOOG", "AAPL"])
mean_ineq = view_on_mean(increms_df, u, ["inequality"], ["equal_less"])

all_views = rankings_view + mean_ineq
print(rankings_view)

test_eq = simple_entropy_pooling(prior, rankings_view + mean_ineq)
# test_ineq = entropy_pooling_probs(prior_2, rankings_view, 0.5, include_diags=True)
# test_ineq_2 = entropy_pooling_probs(prior_2, mean_ineq, 0, include_diags=True)

plt_prob_eval(test_eq, data_long)
# plt_prob_eval(test_ineq, data_long, mean_targets)

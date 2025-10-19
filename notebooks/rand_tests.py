import cvxpy as cp

from get_data import get_example_assets
from maths.constraints import view_on_ranking, view_on_std
from maths.core import (
    assign_constraint_equation,
    build_constraints,
    simple_entropy_pooling,
)
from maths.prob_vectors import uniform_probs

# set-up
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")
increms_np = increms_df.to_numpy()
increms_n = increms_df.height
u = increms_np.mean(axis=0) - 0.019
half_life = 3

prior = uniform_probs(increms_n)
posterior = cp.Variable(prior.shape[0])


x = view_on_std(increms_df, {"AAPL": 0.20}, ["equality"], ["equal"])

y = build_constraints(x, posterior, prior)

z = simple_entropy_pooling(prior, x)

import cvxpy as cp

from get_data import get_example_assets
from maths.constraints import view_on_ranking, view_on_std
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


sigma = increms_np.std(axis=0)

prior = uniform_probs(increms_n)
posterior = cp.Variable(prior.shape[0])


x = view_on_std(increms_df, {"AAPL": 0.03}, ["equality"], ["equal"])

y = build_constraints(x, posterior, prior)

z = simple_entropy_pooling(prior, x)


plt_prob_eval(z, data_long)

import cvxpy as cp
import numpy as np
from scipy.stats import ecdf

from get_data import get_example_assets
from helpers import get_corr_info, weighted_moments
from maths.constraints import view_on_corr, view_on_mean, view_on_ranking, view_on_std
from maths.core import (
    assign_constraint_equation,
    build_constraints,
    simple_entropy_pooling,
)
from maths.operations import prior_cdf
from maths.prob_vectors import uniform_probs
from maths.visuals import plt_prob_eval

# set-up
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")
increms_np = increms_df["AAPL"].to_numpy()
increms_n = increms_df.height
u = increms_np.mean(axis=0) - 0.019
half_life = 3
data_long = assets.increments.unpivot(
    on=tickers, value_name="return", variable_name="ticker", index="date"
)

prior = uniform_probs(increms_n)

print(increms_np[:-1].shape)

print(ecdf(increms_np[:-1]).cdf)

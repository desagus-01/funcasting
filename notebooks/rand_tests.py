import numpy as np
import polars as pl

from maths.constraints import view_on_marginal, view_on_quantile, view_on_std
from maths.core import simple_entropy_pooling
from maths.prob_vectors import uniform_probs
from maths.visuals import plt_prob_eval
from template import test_template

info = test_template()


aapl_df = info["increms_df"].select(pl.col.AAPL)
msft_np = info["increms_df"].select(pl.col.MSFT).to_numpy().flatten()


test = view_on_quantile(aapl_df, 0.25, 0.02)

prior = uniform_probs(aapl_df.height)

example_rd = np.random.normal(
    aapl_df.to_numpy().mean() - 0.02, aapl_df.to_numpy().std() * 1.3, aapl_df.height
)


test_df = aapl_df.with_columns(rd=example_rd)


marg_view = view_on_marginal(test_df, "AAPL", "rd")

std_view = view_on_std(test_df, {"AAPL": aapl_df.to_numpy().std() * 1.5}, ["equal"])

test_eq = simple_entropy_pooling(prior, std_view)

plt_prob_eval(test_eq, info["increms_df_long"])

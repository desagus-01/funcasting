import polars as pl

from maths.constraints import view_on_quantile
from maths.prob_vectors import uniform_probs
from template import test_template

info = test_template()


aapl_df = info["increms_df"].select(pl.col.AAPL)
msft_np = info["increms_df"].select(pl.col.MSFT).to_numpy().flatten()


test = view_on_quantile(aapl_df, 0.25, 0.02)

prior = uniform_probs(aapl_df.height)


a = test.data[1] @ prior
print(a)

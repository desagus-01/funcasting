import polars as pl

from cma.distributions import update_cma_marginal
from cma.operations import cma_combination, cma_separation
from flex_probs.prob_vectors import uniform_probs
from template import test_template

info = test_template()

aapl_df = info.increms_df.select(pl.col.AAPL)

prob = uniform_probs(aapl_df.height)

multi_df = info.increms_df

seps = cma_separation(multi_df, prob)

seps_upd = update_cma_marginal(seps, "MSFT", prob_dist="norm")

print(multi_df, cma_combination(seps_upd))

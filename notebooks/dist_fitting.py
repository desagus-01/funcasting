import polars as pl

from cma._utils import cma_combination, cma_separation
from cma.distributions import update_cma_copula
from flex_probs.prob_vectors import uniform_probs
from template import test_template

info = test_template()

aapl_df = info.increms_df.select(pl.col.AAPL)

prob = uniform_probs(aapl_df.height)

multi_df = info.increms_df

seps = cma_separation(multi_df, prob)

cop_np = seps.copula.to_numpy()

updated_cma = update_cma_copula(seps, "t")

x = cma_combination(updated_cma)

import polars as pl
from copulas.multivariate import GaussianMultivariate

from cma.operations import cma_separation
from flex_probs.prob_vectors import uniform_probs
from template import test_template

info = test_template()

aapl_df = info.increms_df.select(pl.col.AAPL)

prob = uniform_probs(aapl_df.height)

multi_df = info.increms_df

seps = cma_separation(multi_df, prob)

cop_np = seps.copula.to_numpy()

g_cop = GaussianMultivariate()

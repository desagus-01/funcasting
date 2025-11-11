import polars as pl

from cma.operations import cma_separation
from flex_probs.prob_vectors import uniform_probs
from template import test_template

info = test_template()

aapl_df = info["increms_df"].select(pl.col.AAPL)

prob = uniform_probs(aapl_df.height)

norm_const = aapl_df.height / (aapl_df.height + 1)

test = (
    aapl_df.with_row_index()
    .with_columns(prob=prob)
    .sort(pl.col.AAPL)
    .with_columns(
        cdf=pl.cum_sum("prob") * norm_const,
    )
    .with_columns(
        pobs=pl.col("cdf").gather(pl.col("index").arg_sort()),
    )
)


multi_df = info["increms_df"]

test = cma_separation(multi_df, prob)

print(test)

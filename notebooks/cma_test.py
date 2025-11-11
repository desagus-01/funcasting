import polars as pl

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

cdf_cols = {}
copula_cols = {}

for col in multi_df.iter_columns():
    name = col.name
    df = pl.DataFrame({name: col})
    n = df.height
    norm_const = n / (n + 1)
    prob = uniform_probs(n)

    temp = (
        df.with_row_index()
        .with_columns(prob=prob)
        .sort(name)
        .with_columns(
            cdf=pl.cum_sum("prob") * norm_const,
        )
        .with_columns(
            pobs=pl.col("cdf").gather(pl.col("index").arg_sort()),
        )
    )

    # you said you’ll just pluck pobs, so do that
    cdf_cols[name] = temp["cdf"]
    copula_cols[name] = temp["pobs"]

cdf = pl.DataFrame(cdf_cols)
copula = pl.DataFrame(copula_cols)

print(copula)

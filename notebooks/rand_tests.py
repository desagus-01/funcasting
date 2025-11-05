import numpy as np
import polars as pl
from scipy.stats import ecdf

from maths.operations import emp_cdf, prior_cdf
from template import test_template

info = test_template()


aapl_df = info["increms_df"].select(pl.col.AAPL)
msft_np = info["increms_df"].select(pl.col.MSFT).to_numpy().flatten()


prior_cdf = prior_cdf(aapl_df, info["uniform_prior"])
emp_cdf = emp_cdf(msft_np)


cdfs_df = prior_cdf.with_columns(target_quantiles=emp_cdf.quantiles).with_columns(
    prior_less_target=pl.col.AAPL <= pl.col.target_quantiles
)

print(cdfs_df)

import polars as pl
from scipy.stats import t

from template import test_template

info = test_template()

aapl_df = info.increms_df.select(pl.col.AAPL).to_numpy()

t_params = t.fit(aapl_df)

dist = t(*t_params)

# pdf_vals = dist.pdf(aapl_df)
print(dist.pdf(aapl_df))
cdf_vals = dist.cdf(aapl_df)
print(cdf_vals)

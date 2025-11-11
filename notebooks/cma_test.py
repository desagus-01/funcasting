import numpy as np
import polars as pl
from scipy.interpolate import interp1d

from cma.operations import cma_separation
from flex_probs.prob_vectors import uniform_probs
from template import test_template

info = test_template()

aapl_df = info.increms_df.select(pl.col.AAPL)

prob = uniform_probs(aapl_df.height)

multi_df = info.increms_df

test = cma_separation(multi_df, prob)

col_name = "AAPL"


f = interp1d(
    x=test.copula.select(col_name).to_numpy().ravel(),
    y=test.cdfs.select(col_name).to_numpy().ravel(),
    fill_value="extrapolate",
)

print(
    np.interp(
        x=test.copula.select(col_name).to_numpy().ravel(),
        xp=test.cdfs.select(col_name).to_numpy().ravel(),
        fp=test.marginals.select(col_name).to_numpy().ravel(),
    )
)

import numpy as np
import polars as pl

from maths.helpers import add_detrend_column

data = pl.read_csv("./data/tiingo_sample.csv")

assets = ["SMBC", "RDN", "BANC", "FCN"]
data = data.select("date", *assets)


# %%
ex, betas = add_detrend_column(data, ["BANC"], polynomial_orders=[0, 1, 2, 3])

coeffs = betas[2]["BANC"]  # shape should now be (3,)
x = data.height + 1
y = np.polyval(coeffs, x)

print(coeffs)
print(coeffs.shape)
print(y)

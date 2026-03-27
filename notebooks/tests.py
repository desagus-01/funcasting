import polars as pl

from time_series.selection.trend import trend_diagnostic

data = pl.read_csv("./data/tiingo_sample.csv")

assets = ["SMBC", "RDN"]
data = data.select("date", *assets)


# %%

ex = trend_diagnostic(
    data, assets=["SMBC", "RDN"], order_max=3, threshold_order=0, trend_type="both"
)

# %%

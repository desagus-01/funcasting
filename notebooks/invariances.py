import polars as pl

from utils.template import get_template

info_all = get_template()
ts_aapl = info_all.increms_df.select(["date", "AAPL"])

ts_2 = ts_aapl.select("date", "AAPL", lag_aapl=pl.col("AAPL").shift(1))


print(ts_2)

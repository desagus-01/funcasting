# %
import polars as pl

from utils.tiingo import (
    clean_and_save_sample,
    get_sampled_ticker_prices,
    get_tiingo_tickers,
)

tickers = get_tiingo_tickers()
pl.DataFrame(tickers)

# %%
ex = get_sampled_ticker_prices(n_tickers=100)

# %%

clean_and_save_sample(ex, "./data/tiingo_sample.csv")

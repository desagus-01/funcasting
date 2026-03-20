# %%
from utils.tiingo import clean_and_save_sample, get_sampled_ticker_prices

ex = get_sampled_ticker_prices(n_tickers=2)


# %%

clean_and_save_sample(ex, "./data/tiingo_sample.csv")

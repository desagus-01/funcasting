import numpy as np

from get_data import get_example_assets
from maths.prob_vectors import exp_decay_probs, exponential_time_decay

tickers = ["AAPPL", "MSFT", "GOOG"]

assets = get_example_assets(tickers)

increms_df = assets.increments["AAPL"]

exp_dec_probs = exp_decay_probs(increms_df, 50)


print(np.arange(increms_df))

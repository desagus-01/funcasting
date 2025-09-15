import numpy as np

from get_data import get_example_assets

tickers = ["AAPPL", "MSFT", "GOOG"]

assets = get_example_assets(tickers)

increms_df = assets.increments.height

# exp_dec_probs = exp_decay_probs(increms_df, 50)

ex_state_conds = np.random.choice([True, False], size=increms_df)

# print(np.arange(increms_df)

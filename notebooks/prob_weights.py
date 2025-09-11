import numpy as np

from get_data import get_example_assets
from maths.non_parametric import exp_decay_probs, time_crisp_window

tickers = ["AAPPL", "MSFT", "GOOG"]

assets = get_example_assets(tickers)

increms_df = assets.increments

exp_dec_probs = exp_decay_probs(increms_df, 50)

time_probs = time_crisp_window(increms_df, len(increms_df))

ex_state_condition = np.zeros(len(increms_df), dtype=np.int16)

print(time_probs)

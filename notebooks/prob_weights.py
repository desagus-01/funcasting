import numpy as np

from get_data import get_example_assets
from maths.prob_vectors import (
    exp_decay_probs,
    state_crisp_conditioning,
    time_crisp_window,
)

tickers = ["AAPPL", "MSFT", "GOOG"]

assets = get_example_assets(tickers)

increms_df = assets.increments.height

exp_dec_probs = exp_decay_probs(increms_df, 50)

ex_state_conds = np.random.choice([True, False], size=increms_df)

ex = state_crisp_conditioning(increms_df, ex_state_conds)

print(type(ex))

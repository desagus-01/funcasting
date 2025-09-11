import numpy as np

from get_data import get_example_assets
from maths.non_parametric import exp_decay_probs

tickers = ["AAPPL", "MSFT", "GOOG"]

assets = get_example_assets(tickers)

increms_df = assets.increments

scenario_probs = exp_decay_probs(increms_df, 50)

print(scenario_probs.ndim)

import numpy as np

from get_data import get_example_assets
from maths.constraints import view_on_mean
from maths.helpers import simple_entropy_pooling
from maths.prob_vectors import state_smooth_probs, uniform_probs

tickers = ["AAPPL", "MSFT", "GOOG"]

assets = get_example_assets(tickers)

increms_df = assets.increments["AAPL"].to_numpy()


# getting data
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments
increms_n = increms_df.height
u = increms_df["AAPL"].to_numpy().mean() + 2.0

# smoothing methods
half_life = 3

prior = uniform_probs(increms_n)

posterior = state_smooth_probs(
    data_array=increms_df["AAPL"].to_numpy(),
    half_life=half_life,
    kernel_type=2,
    reference=u,
)


mean_const = view_on_mean(increms_df["AAPL"].to_numpy(), np.array([0.015]))

print(simple_entropy_pooling(prior, mean_const[0], mean_const[1]))

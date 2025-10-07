import numpy as np

from get_data import get_example_assets
from maths.constraints import view_on_mean
from maths.prob_vectors import entropy_pooling_probs, state_smooth_probs, uniform_probs
from maths.visuals import plot_post_prob

tickers = ["AAPPL", "MSFT", "GOOG"]

assets = get_example_assets(tickers)

increms_df = assets.increments["AAPL"].to_numpy()


# getting data
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments
increms_n = increms_df.height
u = increms_df["AAPL"].to_numpy().mean()

# smoothing methods
half_life = 3

prior = uniform_probs(increms_n)


prior_2 = state_smooth_probs(
    data_array=increms_df["AAPL"].to_numpy(),
    half_life=half_life,
    kernel_type=2,
    reference=u,
)


mean_const = view_on_mean(increms_df["AAPL"].to_numpy(), np.array([u]), "ineq", "gtr")


# test = entropy_pooling_probs(prior_2, mean_const)

test = entropy_pooling_probs(prior_2, mean_const)

print(test)

# test = entropy_pooling_probs(prior_2, mean_const[0], mean_const[1])

# plot_post_prob(prior)

plot_post_prob(test)

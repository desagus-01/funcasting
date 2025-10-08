from get_data import get_example_assets
from maths.constraints import view_on_mean
from maths.prob_vectors import uniform_probs

# set-up
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")
increms_np = increms_df.to_numpy()
increms_n = increms_df.height
u = increms_np.mean(axis=0) + 0.03
half_life = 3


prior = uniform_probs(increms_n)

#
# prior_2 = state_smooth_probs(
#     data_array=increms_np,
#     half_life=half_life,
#     kernel_type=2,
#     reference=u,
# )
#

mean_ineq = view_on_mean(
    increms_np,
    u,
    ["inequality"] * u.shape[0],
    ["equal_less"] * u.shape[0],
)


mean_eq = view_on_mean(
    increms_np,
    u,
    ["inequality"] * u.shape[0],
    ["equal_less"] * u.shape[0],
)

print(mean_eq)


# test_ineq = entropy_pooling_probs(prior, mean_ineq)
# test_eq = entropy_pooling_probs(prior_2, mean_eq)
#
#
# print(test_ineq)
# print(test_eq)
# print(test_ineq == test_eq)
#
# plot_post_prob(test_ineq)
# plot_post_prob(test_eq)

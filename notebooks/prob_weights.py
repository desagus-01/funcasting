from get_data import get_example_assets
from maths.constraints import view_on_mean
from maths.core import simple_entropy_pooling
from maths.prob_vectors import state_smooth_probs, uniform_probs
from maths.visuals import plt_prob_eval

# set-up
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")
increms_np = increms_df.to_numpy()
increms_n = increms_df.height
u = increms_np.mean(axis=0) + 0.015
half_life = 3


prior = uniform_probs(increms_n)


prior_2 = state_smooth_probs(
    data_array=increms_np[:, 0],
    half_life=half_life,
    kernel_type=2,
    reference=0.015,
)


mean_ineq = view_on_mean(
    increms_np,
    u,
    ["inequality"] * (u.shape[0] - 1) + ["equality"],
    ["equal_less"] * (u.shape[0] - 1) + ["equal"],
)


mean_eq = view_on_mean(
    increms_np,
    u,
    ["equality"] * u.shape[0],
    ["equal"] * u.shape[0],
)


test_eq = simple_entropy_pooling(prior, mean_eq, include_diags=True)
print("next")
test_eq_2 = simple_entropy_pooling(prior_2, mean_eq, include_diags=True)

# test_ineq = entropy_pooling_probs(prior, mean_ineq)
# test_eq = entropy_pooling_probs(prior, mean_eq)
#
#
print(test_eq_2 == test_eq)

# print(ens(prior))
# print(ens(test_ineq))
# print(ens(test_eq))

print(u, increms_np.T @ test_eq)

#
# plt_prob_eval(prior)
plt_prob_eval(test_eq_2)
plt_prob_eval(test_eq)

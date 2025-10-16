from get_data import get_example_assets
from maths.constraints import view_on_exp_return_ranking, view_on_mean
from maths.prob_vectors import entropy_pooling_probs, state_smooth_probs, uniform_probs
from maths.visuals import plt_prob_eval

# set-up
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")
increms_np = increms_df.to_numpy()
increms_n = increms_df.height
u = increms_np.mean(axis=0) - 0.019
half_life = 3

print(f"current means are: {u} for {tickers}")
data_long = assets.increments.unpivot(
    on=tickers, value_name="return", variable_name="ticker", index="date"
)

# plt_returns_dens(data_long)

prior = uniform_probs(increms_n)


prior_2 = state_smooth_probs(
    data_array=increms_np[:, 0],
    half_life=half_life,
    kernel_type=2,
    reference=0.015,
)

targs = {"AAPL": u[0]}

mean_ineq = view_on_mean(increms_df, targs, ["inequality"], ["equal_less"])


rankings_view = view_on_exp_return_ranking(increms_df, ["GOOG", "AAPL", "MSFT"])
mean_targets = [float(view.views_target) for view in mean_ineq]

# test_eq = simple_entropy_pooling(prior, mean_eq, include_diags=True)
test_ineq = entropy_pooling_probs(prior_2, rankings_view, 0.5, include_diags=True)
# test_ineq_2 = entropy_pooling_probs(prior_2, mean_ineq, 0, include_diags=True)

plt_prob_eval(test_ineq, data_long)
# plt_prob_eval(test_ineq, data_long, mean_targets)

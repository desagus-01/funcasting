from get_data import get_example_assets
from maths.constraints import view_on_mean
from maths.core import simple_entropy_pooling
from maths.prob_vectors import state_smooth_probs, uniform_probs
from maths.visuals import plt_prob_eval, plt_returns_dens

# set-up
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")
increms_np = increms_df.to_numpy()
increms_n = increms_df.height
u = increms_np.mean(axis=0) - 0.019
half_life = 3


data_long = assets.increments.unpivot(
    on=tickers, value_name="return", variable_name="ticker", index="date"
)

plt_returns_dens(data_long)

prior = uniform_probs(increms_n)


prior_2 = state_smooth_probs(
    data_array=increms_np[:, 0],
    half_life=half_life,
    kernel_type=2,
    reference=0.015,
)


mean_ineq = view_on_mean(
    increms_np, u, ["inequality"] * (u.shape[0]), ["equal_less"] * (u.shape[0])
)


mean_eq = view_on_mean(
    increms_np,
    u,
    ["equality"] * u.shape[0],
    ["equal"] * u.shape[0],
)


# test_eq = simple_entropy_pooling(prior, mean_eq, include_diags=True)
test_ineq = simple_entropy_pooling(prior, mean_ineq, include_diags=True)


plt_prob_eval(prior)
plt_prob_eval(test_ineq)

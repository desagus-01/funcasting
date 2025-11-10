from get_data import get_example_assets
from maths.prob_vectors import state_smooth_probs, uniform_probs

# set-up
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")
increms_np = increms_df.to_numpy()
increms_n = increms_df.height
u = increms_np.mean(axis=0) - 0.01
half_life = 3


data_long = assets.increments.unpivot(
    on=tickers, value_name="return", variable_name="ticker", index="date"
)


prior = uniform_probs(increms_n)


prior_2 = state_smooth_probs(
    data_array=increms_np[:, 0],
    half_life=half_life,
    kernel_type=2,
    reference=0.015,
)


# marginal_view =
#
# test_eq = simple_entropy_pooling(prior, quant_view, include_diags=True)
#
# plt_prob_eval(test_eq, data_long)

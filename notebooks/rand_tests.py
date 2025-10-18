from get_data import get_example_assets
from maths.constraints import view_on_ranking

# set-up
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")
increms_np = increms_df.to_numpy()
increms_n = increms_df.height
u = increms_np.mean(axis=0) - 0.019
half_life = 3

x = view_on_ranking(increms_df, ["GOOG", "AAPL", "MSFT"])
print(x)

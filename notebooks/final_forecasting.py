import polars as pl

from maths.distributions import uniform_probs
from methods.forecasting_pipeline import run_n_steps_forecast
from utils.tiingo import plot_ticker_lines
from utils.visuals import plot_simulation_results

# %%
data = pl.read_csv("./data/tiingo_sample.csv")
assets = ["SMBC", "RDN", "BANC", "FCN"]
data = data.select("date", *assets)

# %%
data_long = (
    data.select("date", *assets)
    .unpivot(
        index="date",
        on=assets,
        variable_name="ticker",
        value_name="adj_close",
    )
    .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
)

plot_ticker_lines(data_long)

# %%
prob_ex = uniform_probs(data.height)
forecasts = run_n_steps_forecast(
    data=data,
    prob=prob_ex,
    horizon=30,
    n_sims=5000,
    seed=1,
    assets=assets,
    method="bootstrap",
    # target_copula="t",
)

# %%
forecasts[1]
# %%
for asset, forecast in forecasts[1].items():
    plot_simulation_results(forecast, title=f"{asset}")
# %%
# specs = forecasts[1].inverse_specs
# paths = forecasts[0]
#
# if specs is not None:
#     restored_paths = {}
#
#     for asset, transforms in specs.items():
#         current = np.asarray(paths[asset], dtype=float)
#
#         if len(transforms) > 1:
#             ordered_transforms = sorted(
#                 transforms,
#                 key=lambda t: (
#                     0 if isinstance(t.inverse_spec, SeasonalInverseSpec) else 1
#                 ),
#             )
#         else:
#             ordered_transforms = transforms
#
#         for transform in ordered_transforms:
#             inverse_spec = transform.inverse_spec
#
#             if isinstance(inverse_spec, SeasonalInverseSpec):
#                 current = inverse_spec.inverse_for_forecasts(
#                     current,
#                     data.height + 1,
#                 )
#
#             elif isinstance(inverse_spec, PolynomialInverseSpec):
#                 current = inverse_spec.inverse_for_forecasts(
#                     current,
#                     data.height + 1,
#                 )
#
#             elif isinstance(inverse_spec, DifferenceInverseSpec):
#                 current = inverse_spec.inverse_for_forecasts(current)
#
#         restored_paths[asset] = current
#         plot_simulation_results(current, title=f"{asset}")
#
# print(restored_paths)

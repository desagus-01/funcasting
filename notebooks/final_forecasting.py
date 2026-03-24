import numpy as np
import polars as pl

from maths.distributions import uniform_probs
from methods.forecasting_pipeline import run_n_steps_forecast
from methods.preprocess_pipeline import DifferenceInverseSpec, PolynomialInverseSpec
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
assets = ["BANC", "SMBC"]
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
specs = forecasts[1].inverse_specs

paths = forecasts[0]
# %%
if specs is not None:
    for asset, transforms in specs.items():
        for transform in transforms:
            inverse_spec = transform.inverse_spec
            if isinstance(inverse_spec, PolynomialInverseSpec):
                trend_poly = inverse_spec.inverse_for_forecasts(
                    paths[asset], data.height + 1
                )

                plot_simulation_results(np.exp(trend_poly), title=f"{asset}")

            if isinstance(inverse_spec, DifferenceInverseSpec):
                trend_diff = inverse_spec.inverse_for_forecasts(paths[asset])

                plot_simulation_results(np.exp(trend_diff), title=f"{asset}")

        # decision = transform.decision
        # inverse_spec = transform.inverse_spec
        #
        # if isinstance(inverse_spec, PolynomialInverseSpec):
        #     trend_banc = inverse_spec.inverse_for_forecasts(banc_paths, data.height + 1)

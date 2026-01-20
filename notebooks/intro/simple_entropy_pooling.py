import polars as pl

from models.scenarios import ScenarioProb
from models.views_builder import ViewBuilder
from utils.template import get_template, synthetic_series
from utils.visuals import plt_prob_shift

# %% Generate data

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# %% Create Scenario Prob Object

scenarios = ScenarioProb.default_inst(
    scenarios=data
)  # default instance assumes uniform distribution as prior

# %% Create multiple views for entropy pooling (AAPL + MSFT)

# Compute both tickers' mean/std in one go
stats = data.select(
    pl.col("AAPL").mean().alias("AAPL_mean"),
    pl.col("AAPL").std().alias("AAPL_std"),
    pl.col("MSFT").mean().alias("MSFT_mean"),
    pl.col("MSFT").std().alias("MSFT_std"),
).row(0)

aapl_mean, aapl_std, msft_mean, msft_std = stats

print(f"""
AAPL mean: {aapl_mean}
AAPL std:  {aapl_std}

MSFT mean: {msft_mean}
MSFT std:  {msft_std}
""")

# Example: views that AAPL & MSFT will perform lower than historical average AND be more volatile
personal_views = (
    ViewBuilder(data=data)
    .mean(
        target_means={
            "AAPL": aapl_mean - 0.01,
            "MSFT": msft_mean - 0.02,
        },
        sign_type=["equal_less", "equal_less"],
    )
    .std(
        target_std={
            "AAPL": aapl_std + 0.003,
            "MSFT": msft_std + 0.001,
        },
        sign_type=["equal_greater", "equal_greater"],
    )
    .build()
)

personal_views

# %% Add views to model and apply them

scenarios_updated = scenarios.add_views(personal_views).apply_views()
scenarios_updated

plt_prob_shift(
    prior_prob=scenarios.prob,
    post_prob=scenarios_updated.prob,
    dates=scenarios_updated.dates,
    scenarios=scenarios_updated.scenarios,
    why_assets=["AAPL", "MSFT"],
)

import polars as pl

from models.scenarios import ScenarioProb
from models.views_builder import ViewBuilder
from utils.template import get_template, synthetic_series

# %% Generate data

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# %% Create Scenario Prob Object

scenarios = ScenarioProb.default_inst(
    scenarios=data
)  # default instance assumes uniform distribution as prior


# %% Create multiple views for entropy pooling

aapl_stats = data.select(
    pl.col("AAPL").mean().alias("mean"),
    pl.col("AAPL").std().alias("std"),
).row(0)

aapl_mean, aapl_std = aapl_stats


# Example, view that AAPL will perform lower than its historical average AND be more volatile
personal_views = (
    ViewBuilder(data=data)
    .mean(target_means={"AAPL": aapl_mean - 0.01}, sign_type=["equal_less"])
    .std(
        target_std={"AAPL": aapl_std + 0.01},
        sign_type=["equal_greater"],
        # mean_ref=aapl_mean - 0.01,
    )
    # .build()
)

# %% Add views to model and apply them
views = personal_views.views

means = {}  # risk_driver -> views_target (mean)
pending_std = {}  # risk_driver -> list of std views waiting for a mean

for v in views:
    rd = v.risk_driver

    if v.type == "mean":
        means[rd] = v.views_target
        # attach mean to any std views we saw earlier for this driver
        for sv in pending_std.pop(rd, []):
            sv.mean_ref = v.views_target

    elif v.type == "std":
        if rd in means:
            v.mean_ref = means[rd]
        else:
            pending_std.setdefault(rd, []).append(v)

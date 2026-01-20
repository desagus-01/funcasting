import polars as pl

from models.scenarios import ScenarioProb
from models.types import CorrInfo
from models.views_builder import ViewBuilder
from utils.template import get_template, synthetic_series
from utils.visuals import plt_prob_shift

# %% Investor thesis:
# AAPL faces softer demand + margin pressure, and rates drift higher.
# Also: tightens co-movement between MSFT & GOOG."

# %% Generate data

data = get_template().asset_info.risk_drivers  # columns: MSFT, AAPL, GOOG, fake
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# Rates: a synthetic driver for demonstration
rates = synthetic_series(data.height)
data = data.with_columns(Rates=rates)

# %% Create Scenario Prob Object

scenarios = ScenarioProb.default_inst(scenarios=data)  # uniform prior by default

# %% Compute some historical stats to anchor targets

stats = data.select(
    pl.col("AAPL").mean().alias("AAPL_mean"),
    pl.col("AAPL").std().alias("AAPL_std"),
    pl.col("MSFT").mean().alias("MSFT_mean"),
    pl.col("MSFT").std().alias("MSFT_std"),
    pl.col("GOOG").mean().alias("GOOG_mean"),
    pl.col("GOOG").std().alias("GOOG_std"),
    pl.col("Rates").mean().alias("Rates_mean"),
    pl.col("Rates").std().alias("Rates_std"),
).row(0)

aapl_mean, aapl_std, msft_mean, msft_std, goog_mean, goog_std, rates_mean, rates_std = (
    stats
)

# Historical correlations (for sanity checks / target anchoring)
corr_msft_goog = data.select(pl.corr("MSFT", "GOOG")).item()
corr_aapl_msft = data.select(pl.corr("AAPL", "MSFT")).item()
corr_rates_msft = data.select(pl.corr("Rates", "MSFT")).item()

print(f"""
--- Historical moments ---
AAPL  mean: {aapl_mean:.6f} | std: {aapl_std:.6f}
MSFT  mean: {msft_mean:.6f} | std: {msft_std:.6f}
GOOG  mean: {goog_mean:.6f} | std: {goog_std:.6f}
Rates mean: {rates_mean:.6f} | std: {rates_std:.6f}

--- Historical correlations ---
corr(MSFT, GOOG):  {corr_msft_goog:.4f}
corr(AAPL, MSFT):  {corr_aapl_msft:.4f}
corr(Rates, MSFT): {corr_rates_msft:.4f}
""")

# %% Build a complex set of views
#
# Views included:
# 1) Means: AAPL down, GOOG slightly up, Rates up (hawkish drift)
# 2) Stds: GOOG more volatile (bigger dispersion), AAPL slightly less volatile
# 3) Corr: MSFT-GOOG corr increases, AAPL-MSFT corr decreases, Rates-MSFT becomes more negative
# 4) Ranking: GOOG >= MSFT >= AAPL (expected returns ordering)

# Mean targets (small tilts, plausible)
target_means = {
    "AAPL": aapl_mean - 0.015,  # softer demand / margin pressure
    "GOOG": goog_mean + 0.008,  # AI + ads resilience
    "Rates": rates_mean + 0.010,  # rates drift higher
}

# Std targets
target_std = {
    "GOOG": goog_std + 0.004,  # more dispersion / event risk
    "AAPL": aapl_std - 0.002,  # slightly calmer than usual
}

# Correlation targets (nudges relative to history)
corr_targets = [
    CorrInfo(asset_pair=("MSFT", "GOOG"), corr=min(corr_msft_goog + 0.12, 0.95)),
    CorrInfo(asset_pair=("AAPL", "MSFT"), corr=max(corr_aapl_msft - 0.10, -0.95)),
    CorrInfo(asset_pair=("Rates", "MSFT"), corr=max(corr_rates_msft - 0.15, -0.95)),
]

personal_views = (
    ViewBuilder(data=data)
    .mean(
        target_means=target_means,
        sign_type=[
            "equal_less",
            "equal_greater",
            "equal_greater",
        ],  # AAPL down, GOOG up, Rates up
    )
    .std(
        target_std=target_std,
        sign_type=[
            "equal_greater",
            "equal_less",
        ],  # GOOG more volatile, AAPL less volatile
    )
    .corr(
        corr_targets=corr_targets,
        sign_type=[
            "equal_greater",
            "equal_less",
            "equal_less",
        ],  #  decouple AAPL, rates drag MSFT more
    )
    .ranking(
        asset_ranking=["GOOG", "MSFT", "AAPL"]  # GOOG >= MSFT >= AAPL
    )
    .build()
)

personal_views

# %% Apply views via entropy pooling and visualise changes

scenarios_updated = scenarios.add_views(personal_views).apply_views()


plt_prob_shift(
    prior_prob=scenarios.prob,
    post_prob=scenarios_updated.prob,
    dates=scenarios_updated.dates,
)

from dataclasses import dataclass

import polars as pl
from polars import DataFrame

from portfolio.attribution.performance import PortfolioPerformanceAttribution
from scenarios.types import ProbVector


@dataclass(frozen=True, slots=True)
class PortfolioRiskAttribution:
    horizon: int
    exposures: dict[str, float]
    joint_distribution: DataFrame
    probs: ProbVector

    @classmethod
    def from_performance_attribution(
        cls, performance_attribution: PortfolioPerformanceAttribution
    ):
        joint = performance_attribution.joint_distribution.with_columns(
            loss=-pl.col("portfolio_performance")
        ).drop("portfolio_performance")

        return PortfolioRiskAttribution(
            horizon=performance_attribution.horizon,
            exposures={
                factor: -exposure
                for factor, exposure in performance_attribution.full_exposures.items()
            },
            joint_distribution=joint,
            probs=performance_attribution.path_probs,
        )

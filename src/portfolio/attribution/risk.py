from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
from numpy._typing import NDArray
from polars import DataFrame

from portfolio.attribution.performance import PortfolioPerformanceAttribution
from scenarios.types import ProbVector
from time_series.dimensionality_reduction import minimum_torsion_matrix
from time_series.estimation import weighted_covariance
from utils.visuals import plot_effective_bets


class RiskContributions(NamedTuple):
    risk_measure: str
    value: float
    contributions: dict[str, float]


class EffectiveBets(NamedTuple):
    factor_contributions: dict[str, float]
    effective_bets: float

    def plot(self) -> None:
        return plot_effective_bets(self)


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


def _min_torso_factor_exposures(
    min_torso_matrix: NDArray[np.floating],
    factor_exposures: dict[str, float],
) -> NDArray[np.floating]:
    inv_min_torso = np.linalg.inv(min_torso_matrix)
    exposures = np.array(
        [v for v in factor_exposures.values() if isinstance(v, (float, np.floating))],
        dtype=float,
    )
    return inv_min_torso.T @ exposures


def effective_bets(
    factor_joint_distribution: NDArray[np.floating],
    factor_exposures: dict[str, float],
    prob: ProbVector,
    method: Literal["approximate", "exact"] = "approximate",
    max_iter: int | None = None,
) -> EffectiveBets:
    """Compute effective bets and factor risk contributions."""
    min_torso_matrix = minimum_torsion_matrix(
        factor_joint_distribution, prob, method, max_iter
    )

    # Preserve the ordering of factors when extracting exposures
    factor_keys = [
        k for k, v in factor_exposures.items() if isinstance(v, (float, np.floating))
    ]

    # b in the R code
    exposures = np.array([factor_exposures[k] for k in factor_keys], dtype=float)
    covariance = weighted_covariance(data=factor_joint_distribution, prob=prob)
    min_torso_exposures = _min_torso_factor_exposures(
        min_torso_matrix, factor_exposures
    )

    transformed_covariance_exposure = min_torso_matrix @ covariance @ exposures
    portfolio_variance = float(exposures @ covariance @ exposures)

    factor_risk_contribution = (
        min_torso_exposures * transformed_covariance_exposure / portfolio_variance
    )

    enb = float(
        np.exp(
            -np.sum(
                factor_risk_contribution
                * np.log(
                    1.0
                    + (factor_risk_contribution - 1.0)
                    * (factor_risk_contribution > 1e-5)
                )
            )
        )
    )

    factor_contributions = {
        k: float(v) for k, v in zip(factor_keys, factor_risk_contribution)
    }

    return EffectiveBets(factor_contributions=factor_contributions, effective_bets=enb)


def var_contribution(
    joint_distribution_factors: DataFrame,
    factors_exposures: dict[str, float],
    prob: ProbVector,
    alpha: float = 0.05,
) -> RiskContributions:
    q = 1 - alpha

    joint_risk = joint_distribution_factors.with_columns(prob=pl.Series(prob))

    var_row = (
        joint_risk.sort("loss")
        .with_columns(pl.col("prob").cum_sum().alias("cum_prob"))
        .filter(pl.col("cum_prob") >= q)
        .head(1)
    )
    factor_cols = [
        c for c in joint_risk.columns if c not in ("loss", "prob", "cum_prob")
    ]

    contributions = {
        c: float(var_row.select(c).item()) * float(factors_exposures[c])
        for c in factor_cols
    }

    return RiskContributions(
        risk_measure="var",
        value=float(var_row.select("loss").item()),
        contributions=contributions,
    )

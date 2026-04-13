from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy._typing import NDArray
from polars import DataFrame

from portfolio.attribution.performance import PortfolioPerformanceAttribution
from scenarios.types import ProbVector
from time_series.dimensionality_reduction import minimum_torsion_matrix
from time_series.estimation import weighted_variance


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


def _min_torso_updated_joint(
    min_torso_matrix: NDArray[np.floating], joint_distribution: NDArray[np.floating]
) -> NDArray[np.floating]:
    return joint_distribution @ min_torso_matrix.T


def variance_share_contributions(
    factor_joint_distribution: NDArray[np.floating],
    loss_values: NDArray[np.floating],
    factor_exposures: dict[str, float],
    prob: ProbVector,
    method: Literal["approximate", "exact"] = "approximate",
    max_iter: int | None = None,
):
    min_torso_matrix = minimum_torsion_matrix(
        factor_joint_distribution, prob, method, max_iter
    )
    min_torso_exposures = _min_torso_factor_exposures(
        min_torso_matrix, factor_exposures
    )
    updated_joint = _min_torso_updated_joint(
        min_torso_matrix, factor_joint_distribution
    )

    variance_joint = weighted_variance(data=updated_joint, prob=prob)
    contribution = (min_torso_exposures**2) * variance_joint

    variance_loss = weighted_variance(data=loss_values, prob=prob)

    return contribution / variance_loss

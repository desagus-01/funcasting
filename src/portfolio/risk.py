from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy._typing import NDArray
from numpy.lib.array_utils import normalize_axis_index
from polars import DataFrame

from portfolio.simulation import PortfolioForecast
from scenarios.panel import ScenarioPanel
from scenarios.types import ProbVector


@dataclass(frozen=True, slots=True)
class LossDistribution:
    panel: ScenarioPanel
    asset_weights: dict[str, NDArray[np.floating]] | None = None

    @classmethod
    def from_portfolio_forecast(
        cls,
        portfolio_forecast: PortfolioForecast,
    ) -> "LossDistribution":
        loss_values = -portfolio_forecast.cumulative_performance

        columns = {f"h{h}": loss_values[:, h] for h in range(loss_values.shape[1])}

        return cls(
            panel=ScenarioPanel(
                values=DataFrame(columns),
                dates=None,
                prob=portfolio_forecast.path_probs,
            ),
            asset_weights=portfolio_forecast.asset_weights,
        )

    @property
    def loss_values(self) -> NDArray[np.floating]:
        return self.panel.values.to_numpy()

    @property
    def probs(self) -> ProbVector:
        return self.panel.prob

    def at_horizon(self, horizon: int) -> ScenarioPanel:
        col = f"h{horizon}"

        if col not in self.panel.values.columns:
            raise ValueError(f"horizon={horizon} is out of range for LossDistribution")

        return ScenarioPanel(
            values=self.panel.values.select(col).rename({col: "loss"}),
            dates=None,
            prob=self.panel.prob,
        )


def _loss_value_from_pnl(
    performance: NDArray[np.floating],
) -> NDArray[np.floating]:
    return -performance


def var(
    distribution: NDArray[np.floating],
    prob: ProbVector | None,
    method: Literal["empirical", "quantile"] = "quantile",
    alpha: float = 0.05,
    axis: int = 0,
    *,
    distribution_type: Literal["pnl", "loss"] = "loss",
) -> NDArray[np.floating]:
    distribution = np.asarray(distribution, dtype=float)

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1 (exclusive)")
    if distribution.ndim == 0:
        raise ValueError("distribution must have at least 1 dimension")
    if distribution_type not in {"pnl", "loss"}:
        raise ValueError("distribution_type must be either 'pnl' or 'loss'")
    if method not in {"empirical", "quantile"}:
        raise ValueError("method must be either 'empirical' or 'quantile'")

    axis = normalize_axis_index(axis, distribution.ndim)
    q = alpha if distribution_type == "pnl" else 1 - alpha

    if method == "empirical" and (prob is not None):
        quantile_val = np.quantile(
            distribution, q, axis=axis, method="inverted_cdf", weights=prob
        )
    elif (method == "quantile") and (prob is None):
        quantile_val = np.quantile(distribution, q, axis=axis)
    else:
        raise ValueError(
            "Must choose either empirical or quantile methods with corresponding prob vector."
        )

    return quantile_val if distribution_type == "loss" else -quantile_val


def cvar(
    distribution: NDArray[np.floating],
    prob: ProbVector,
    method: Literal["empirical", "quantile"] = "quantile",
    alpha: float = 0.05,
    axis: int = 0,
    *,
    distribution_type: Literal["pnl", "loss"] = "loss",
) -> NDArray[np.floating]:
    var_cutoff = var(
        distribution=distribution,
        prob=prob,
        method=method,
        alpha=alpha,
        distribution_type=distribution_type,
    )
    expanded_cutoff = np.expand_dims(var_cutoff, axis=axis)

    if distribution_type == "pnl":
        tail_mask = distribution <= expanded_cutoff
        tail_values = np.where(tail_mask, distribution, np.nan)
        cvar = -np.nanmean(tail_values, axis=axis)
    else:
        tail_mask = distribution >= expanded_cutoff
        tail_values = np.where(tail_mask, distribution, np.nan)
        cvar = np.nanmean(tail_values, axis=axis)

    return cvar

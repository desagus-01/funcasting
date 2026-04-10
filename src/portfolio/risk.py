from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy._typing import NDArray
from numpy.lib.array_utils import normalize_axis_index

from portfolio.value import PortfolioForecast
from scenarios.types import ProbVector


@dataclass(frozen=True, slots=True)
class LossDistribution:
    loss_values: NDArray[np.floating]
    probs: ProbVector
    asset_weights: dict[str, NDArray[np.floating]]

    @classmethod
    def from_portfolio_forecast(cls, portfolio_forecast: PortfolioForecast):
        return cls(
            loss_values=_loss_value_from_pnl(portfolio_forecast.pnl),
            probs=portfolio_forecast.path_probs,
            asset_weights=portfolio_forecast.asset_weights,
        )


def _loss_value_from_pnl(
    portfolio_pnl: NDArray[np.floating],
) -> NDArray[np.floating]:
    return -portfolio_pnl


def VAR(
    distribution: NDArray[np.floating],
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

    if method == "empirical":
        quantile_val = np.quantile(distribution, q, axis=axis, method="inverted_cdf")
    else:
        quantile_val = np.quantile(distribution, q, axis=axis)

    return quantile_val if distribution_type == "loss" else -quantile_val


def CVAR(
    distribution: NDArray[np.floating],
    method: Literal["empirical", "quantile"] = "quantile",
    alpha: float = 0.05,
    axis: int = 0,
    *,
    distribution_type: Literal["pnl", "loss"] = "loss",
) -> NDArray[np.floating]:
    var_cutoff = VAR(
        distribution=distribution,
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

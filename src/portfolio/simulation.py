from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from portfolio.construction import (
    WEIGHT_MODE,
    _initial_portfolio_values_from_shares,
    _portfolio_values_static_weights,
    _validate_target_weights,
    portfolio_weights_forecast_buy_and_hold,
    portfolio_weights_forecast_static,
)
from scenarios.types import ProbVector
from utils.visuals import plot_simulation_results

PnL_OPTIONS = Literal["relative", "absolute", "log"]


@dataclass(frozen=True, slots=True)
class PortfolioForecast:
    values: NDArray[np.floating]
    incremental_pnl: NDArray[np.floating]
    pnl_type: PnL_OPTIONS
    weight_mode: WEIGHT_MODE
    asset_weights: dict[str, NDArray[np.floating]]
    path_probs: ProbVector

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                f"values must be 2-D (n_sims, n_periods); got ndim={self.values.ndim}"
            )
        if self.incremental_pnl.ndim != 2:
            raise ValueError(
                f"pnl must be 2-D (n_sims, n_periods or n_periods-1); got ndim={self.incremental_pnl.ndim}"
            )

        n_sims, n_periods = self.values.shape

        if self.incremental_pnl.shape[0] != n_sims:
            raise ValueError(
                f"Mismatch in number of simulations: values has {n_sims} but pnl has {self.incremental_pnl.shape[0]}"
            )

        if self.incremental_pnl.shape[1] not in (n_periods, n_periods - 1):
            raise ValueError(
                f"pnl must have either the same number of periods as values ({n_periods}) or one fewer; got {self.incremental_pnl.shape[1]}"
            )

        for name, arr in self.asset_weights.items():
            w = np.asarray(arr, dtype=float)
            if w.ndim != 2:
                raise ValueError(
                    f"Weight array for asset '{name}' must be 2-D (n_sims, n_periods); got ndim={w.ndim}"
                )
            if w.shape != (n_sims, n_periods):
                raise ValueError(
                    f"Weight array shape mismatch for asset '{name}': expected ({n_sims}, {n_periods}), got {w.shape}"
                )

        try:
            if len(self.path_probs) != n_sims:
                raise ValueError(
                    f"path_probs length ({len(self.path_probs)}) must equal number of simulations ({n_sims})"
                )
        except TypeError:
            # path_probs might be a callable or other unsized object; ignore in that case
            pass

    @property
    def cumulative_performance(self) -> NDArray[np.floating]:
        return cumulative_pnl_forecast(self.incremental_pnl, self.pnl_type)

    def performance_at_horizon(
        self,
        end_horizon: int | None = None,
        at_horizon: int | None = None,
    ) -> NDArray[np.floating]:
        return cumulative_pnl_forecast(
            self.incremental_pnl,
            self.pnl_type,
            end_horizon=end_horizon,
            at_horizon=at_horizon,
        )

    def plot(
        self,
        end_horizon: int,
        value_type: Literal["value", "pnl"] = "pnl",
        plot_cumulative: bool = False,
    ) -> None:
        if value_type not in {"value", "pnl"}:
            raise ValueError(f"Unknown value_type: {value_type}")

        if not plot_cumulative:
            data = self.values if value_type == "value" else self.incremental_pnl
            plot_simulation_results(
                data,
                title=f"Portfolio {value_type} ({self.weight_mode})",
            )
            return

        if value_type == "value":
            initial = self.values[:, :1].astype(float, copy=True)
            initial[np.abs(initial) < 1e-12] = np.nan
            cumulative_changes = self.values / initial - 1.0
        else:
            cumulative_changes = cumulative_pnl_forecast(
                self.incremental_pnl, self.pnl_type, end_horizon=end_horizon
            )

        plot_simulation_results(
            cumulative_changes,
            title=f"Portfolio Cumulative {value_type} ({self.weight_mode})",
        )


def cumulative_pnl_forecast(
    pnl: NDArray[np.floating],
    pnl_type: PnL_OPTIONS,
    end_horizon: int | None = None,
    at_horizon: int | None = None,
) -> NDArray[np.floating]:
    if end_horizon is not None and at_horizon is not None:
        raise ValueError("end_horizon and at_horizon are mutually exclusive")
    if end_horizon is not None and end_horizon <= 0:
        raise ValueError("end_horizon must be a positive integer")
    if at_horizon is not None and at_horizon <= 0:
        raise ValueError("at_horizon must be a positive integer")

    if pnl_type not in {"relative", "absolute", "log"}:
        raise ValueError(f"Unknown pnl_type: {pnl_type}")

    if pnl.ndim != 2:
        raise ValueError("pnl must have shape (n_sims, n_periods)")

    # When at_horizon is set we only need to accumulate up to that point.
    effective_end = at_horizon if at_horizon is not None else end_horizon
    slice_ = slice(None, effective_end)
    p = pnl[:, slice_]
    zeros = np.zeros((p.shape[0], 1), dtype=float)

    if pnl_type == "relative":
        running = np.cumprod(1.0 + p, axis=1) - 1.0
    elif pnl_type == "absolute":
        running = np.cumsum(p, axis=1)
    else:  # log
        running = np.exp(np.cumsum(p, axis=1)) - 1.0

    full = np.concatenate([zeros, running], axis=1)  # (n_sims, H + 1)

    if at_horizon is not None:
        return full[:, at_horizon]  # (n_sims,)

    return full


def _validate_asset_order(
    asset_forecasts: dict[str, NDArray[np.floating]],
    asset_order: list[str],
) -> None:
    if not asset_order:
        raise ValueError("asset_order cannot be empty")

    missing = [a for a in asset_order if a not in asset_forecasts]
    extra = [a for a in asset_forecasts if a not in asset_order]

    if missing:
        raise KeyError(f"Missing asset forecasts for: {missing}")
    if extra:
        raise KeyError(f"Found extra asset forecasts not in asset_order: {extra}")


def _build_price_stack(
    asset_forecasts: dict[str, NDArray[np.floating]],
    asset_order: list[str],
) -> NDArray[np.floating]:
    _validate_asset_order(asset_forecasts, asset_order)

    arrays: list[NDArray[np.floating]] = []
    base_shape: tuple[int, int] | None = None

    for asset in asset_order:
        arr = np.asarray(asset_forecasts[asset], dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"Asset {asset} must have shape (n_sims, n_periods), got {arr.shape}"
            )

        if base_shape is None:
            base_shape = arr.shape
        elif arr.shape != base_shape:
            raise ValueError(
                f"Shape mismatch for asset {asset}: {arr.shape} != {base_shape}"
            )

        arrays.append(arr)

    return np.stack(arrays, axis=0)  # (n_assets, n_sims, n_periods)


def _shares_vector(
    initial_asset_shares: dict[str, int],
    asset_order: list[str],
) -> NDArray[np.floating]:
    missing = [a for a in asset_order if a not in initial_asset_shares]
    if missing:
        raise KeyError(f"No shares specified for assets: {missing}")

    shares = np.array([initial_asset_shares[a] for a in asset_order], dtype=float)
    if np.any(shares < 0):
        raise ValueError("initial_asset_shares must be non-negative")

    return shares


def _build_asset_values_array(
    asset_forecasts: dict[str, NDArray[np.floating]],
    initial_asset_shares: dict[str, int],
    asset_order: list[str],
) -> NDArray[np.floating]:
    prices = _build_price_stack(asset_forecasts, asset_order)
    shares = _shares_vector(initial_asset_shares, asset_order)
    return prices * shares[:, None, None]  # (n_assets, n_sims, n_periods)


def _prepend_t0_prices(
    prices: NDArray[np.floating],
    asset_order: list[str],
    initial_prices: dict[str, float],
) -> NDArray[np.floating]:
    """Prepend last known prices (t0) to a price stack (n_assets, n_sims, n_periods).

    Returns a new writeable array of shape (n_assets, n_sims, n_periods + 1) where
    column 0 is the known t0 price (identical across all simulations) and columns
    1..n_periods are the simulated forecast prices.
    """
    missing = [a for a in asset_order if a not in initial_prices]
    if missing:
        raise KeyError(f"Missing t0 prices for assets: {missing}")

    t0 = np.array([initial_prices[a] for a in asset_order], dtype=float)
    n_assets, n_sims, _ = prices.shape
    t0_col = np.tile(t0[:, None, None], (1, n_sims, 1))
    return np.concatenate([t0_col, prices], axis=2)


def portfolio_value_forecast(
    asset_forecasts: dict[str, NDArray[np.floating]],
    initial_asset_shares: dict[str, int],
    asset_order: list[str] | None = None,
) -> NDArray[np.floating]:
    if asset_order is None:
        asset_order = list(initial_asset_shares.keys())

    asset_values = _build_asset_values_array(
        asset_forecasts=asset_forecasts,
        initial_asset_shares=initial_asset_shares,
        asset_order=asset_order,
    )
    return asset_values.sum(axis=0)


def portfolio_pnl_forecast_from_values(
    forecast_portfolio_values: NDArray[np.floating],
    mode: PnL_OPTIONS = "relative",
    safe_eps: float = 1e-12,
) -> NDArray[np.floating]:
    if mode not in {"relative", "absolute", "log"}:
        raise ValueError(f"Unknown mode: {mode}")

    values = np.asarray(forecast_portfolio_values, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            "forecast_portfolio_values must have shape (n_sims, n_periods)"
        )

    prev = values[:, :-1]
    curr = values[:, 1:]

    if mode == "absolute":
        return curr - prev

    if mode == "relative":
        denom = prev.astype(float, copy=True)
        denom[np.abs(denom) < safe_eps] = np.nan
        return (curr - prev) / denom

    invalid = (prev <= safe_eps) | (curr <= safe_eps)
    denom = prev.astype(float, copy=True)
    denom[invalid] = np.nan
    ratio = curr / denom
    return np.log(ratio)


def portfolio_forecast(
    asset_forecasts: dict[str, NDArray[np.floating]],
    path_probs: ProbVector,
    initial_asset_shares: dict[str, int],
    initial_prices: dict[str, float],
    asset_order: list[str] | None = None,
    pnl_type: PnL_OPTIONS = "relative",
    weight_mode: WEIGHT_MODE = "buy_and_hold",
    target_weights: dict[str, float] | None = None,
    safe_eps: float = 1e-12,
) -> PortfolioForecast:
    """
    Build a full PortfolioForecast.

    Parameters
    ----------
    initial_prices
        Last known prices for each asset (t0). Used to anchor the price stack
        so that PnL and weights include the t0 -> t1 step.
    weight_mode
        - "buy_and_hold": fixed shares, drifting weights
        - "static": fixed target weights, implicitly rebalanced each step
    target_weights
        Required when weight_mode="static". Should map asset -> weight and sum to 1.
    """
    if asset_order is None:
        asset_order = list(initial_asset_shares.keys())

    prices = _build_price_stack(asset_forecasts, asset_order)
    prices = _prepend_t0_prices(prices, asset_order, initial_prices)
    _, n_sims, n_periods = prices.shape
    if weight_mode == "buy_and_hold":
        asset_values = (
            prices * _shares_vector(initial_asset_shares, asset_order)[:, None, None]
        )
        values = asset_values.sum(axis=0)
        weights = portfolio_weights_forecast_buy_and_hold(
            asset_values=asset_values,
            portfolio_values=values,
            asset_order=asset_order,
            safe_eps=safe_eps,
        )

    elif weight_mode == "static":
        if target_weights is None:
            raise ValueError(
                "target_weights must be provided when weight_mode='static'"
            )

        _validate_target_weights(target_weights, asset_order)
        initial_portfolio_values = _initial_portfolio_values_from_shares(
            prices=prices,
            initial_asset_shares=initial_asset_shares,
            asset_order=asset_order,
        )

        values = _portfolio_values_static_weights(
            prices=prices,
            target_weights=target_weights,
            asset_order=asset_order,
            initial_portfolio_values=initial_portfolio_values,
            safe_eps=safe_eps,
        )
        weights = portfolio_weights_forecast_static(
            target_weights=target_weights,
            n_sims=n_sims,
            n_periods=n_periods,
            asset_order=asset_order,
        )

    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    pnl = portfolio_pnl_forecast_from_values(
        forecast_portfolio_values=values,
        mode=pnl_type,
        safe_eps=safe_eps,
    )

    return PortfolioForecast(
        values=values,
        incremental_pnl=pnl,
        pnl_type=pnl_type,
        weight_mode=weight_mode,
        asset_weights=weights,
        path_probs=path_probs,
    )

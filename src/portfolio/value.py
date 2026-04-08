from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from scenarios.types import ProbVector
from utils.helpers import wide_to_long
from utils.visuals import plot_simulation_results

PnL_OPTIONS = Literal["relative", "absolute", "log"]
WEIGHT_MODE = Literal["buy_and_hold", "static"]


@dataclass(frozen=True, slots=True)
class PortfolioInfoT0:
    all_info: pl.DataFrame
    estimated_t0: float
    shares_mapping: dict[str, int]
    t0_prices: dict[str, float]


@dataclass(frozen=True, slots=True)
class PortfolioForecast:
    values: NDArray[np.floating]
    pnl: NDArray[np.floating]
    pnl_type: PnL_OPTIONS
    weight_mode: WEIGHT_MODE
    asset_weights: dict[str, NDArray[np.floating]]
    path_probs: ProbVector

    def cumulative_pnl(
        self,
        end_horizon: int | None = None,
        at_horizon: int | None = None,
    ) -> NDArray[np.floating]:
        return cumulative_pnl_forecast(
            self.pnl,
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
            data = self.values if value_type == "value" else self.pnl
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
                self.pnl, self.pnl_type, end_horizon=end_horizon
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


def equal_weight_target_weights(asset_order: list[str]) -> dict[str, float]:
    if not asset_order:
        raise ValueError("asset_order cannot be empty")

    w = 1.0 / len(asset_order)
    return {asset: w for asset in asset_order}


def get_latest_prices(data_long: pl.DataFrame) -> pl.DataFrame:
    required = {"ticker", "adj_close"}
    missing = required - set(data_long.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if "date" in data_long.columns:
        data_long = data_long.with_columns(pl.col("date").cast(pl.Date)).sort(
            ["ticker", "date"]
        )
    else:
        dupes = (
            data_long.group_by("ticker")
            .len()
            .filter(pl.col("len") > 1)
            .select("ticker")
            .to_series()
            .to_list()
        )
        if dupes:
            raise ValueError(
                "Multiple rows per ticker found but no 'date' column is present. "
                f"Ambiguous latest price for tickers: {dupes}"
            )

    latest_price = data_long.group_by("ticker").agg(
        pl.col("adj_close").last().alias("adj_close")
    )

    if latest_price.height == 0:
        raise ValueError("No prices available after preprocessing")

    return latest_price


def equal_weight_shares_from_prices(
    latest_price: pl.DataFrame,
    initial_value: float,
    n_assets: int | None = None,
) -> pl.DataFrame:
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")

    if n_assets is None:
        n_assets = latest_price.height

    if n_assets <= 0:
        raise ValueError("n_assets must be positive")

    if latest_price.height == 0:
        raise ValueError("latest_price cannot be empty")

    bad_prices = latest_price.filter(pl.col("adj_close") <= 0)
    if bad_prices.height > 0:
        raise ValueError("All adj_close values must be positive")

    assigned = initial_value / n_assets

    return (
        latest_price.with_columns(
            (pl.lit(assigned) / pl.col("adj_close"))
            .floor()
            .cast(pl.Int64)
            .alias("shares")
        )
        .with_columns((pl.col("shares") * pl.col("adj_close")).alias("value_allocated"))
        .with_columns(
            (pl.col("value_allocated") / pl.lit(initial_value)).alias(
                "pct_of_portfolio"
            ),
            pl.lit(1.0 / n_assets).alias("target_pct"),
        )
        .select(
            [
                "ticker",
                "adj_close",
                "shares",
                "value_allocated",
                "pct_of_portfolio",
                "target_pct",
            ]
        )
    )


def portfolio_value(positions: pl.DataFrame) -> tuple[float, pl.DataFrame]:
    required = {"adj_close", "shares"}
    if "value_allocated" not in positions.columns:
        missing = required - set(positions.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")
        positions = positions.with_columns(
            (pl.col("adj_close") * pl.col("shares")).alias("value_allocated")
        )

    total = float(positions["value_allocated"].sum())

    if total == 0:
        positions = positions.with_columns(pl.lit(0.0).alias("pct_of_portfolio"))
    else:
        positions = positions.with_columns(
            (pl.col("value_allocated") / pl.lit(total)).alias("pct_of_portfolio")
        )

    return total, positions


def _get_ticker_shares(
    portfolio_initial_info: pl.DataFrame,
    ticker_col: str = "ticker",
    shares_col: str = "shares",
) -> dict[str, int]:
    ticker_shares = portfolio_initial_info.select([ticker_col, shares_col])
    return dict(
        zip(
            ticker_shares[ticker_col].to_list(),
            ticker_shares[shares_col].to_list(),
        )
    )


def _get_target_weights(
    portfolio_initial_info: pl.DataFrame,
    ticker_col: str = "ticker",
    weight_col: str = "pct_of_portfolio",
) -> dict[str, float]:
    ticker_weights = portfolio_initial_info.select([ticker_col, weight_col])
    return dict(
        zip(
            ticker_weights[ticker_col].to_list(),
            ticker_weights[weight_col].to_list(),
        )
    )


def build_equal_weight_portfolio_from_df(
    data: pl.DataFrame,
    initial_value: float,
    assets: list[str] | None = None,
) -> PortfolioInfoT0:
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")

    cols = set(data.columns)

    if {"ticker", "adj_close"}.issubset(cols):
        data_long = (
            data.with_columns(pl.col("date").cast(pl.Date)) if "date" in cols else data
        )

        if assets is None:
            assets = data_long.select("ticker").unique().to_series().to_list()
        else:
            data_long = data_long.filter(pl.col("ticker").is_in(assets))
    else:
        if assets is None:
            assets = [c for c in data.columns if c != "date"]
        data_long = wide_to_long(data, assets)

    if not assets:
        raise ValueError("No assets provided or inferred")

    latest_price = get_latest_prices(data_long)

    available_assets = set(latest_price["ticker"].to_list())
    missing_assets = [a for a in assets if a not in available_assets]
    if missing_assets:
        raise KeyError(f"Missing latest prices for assets: {missing_assets}")

    latest_price = latest_price.filter(pl.col("ticker").is_in(assets))

    eq_portfolio = equal_weight_shares_from_prices(
        latest_price=latest_price,
        initial_value=initial_value,
        n_assets=len(assets),
    )
    total, positions = portfolio_value(eq_portfolio)

    return PortfolioInfoT0(
        all_info=positions,
        estimated_t0=total,
        shares_mapping=_get_ticker_shares(positions),
        t0_prices=dict(
            zip(latest_price["ticker"].to_list(), latest_price["adj_close"].to_list())
        ),
    )


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


def portfolio_weights_forecast_buy_and_hold(
    asset_values: NDArray[np.floating],
    portfolio_values: NDArray[np.floating],
    asset_order: list[str],
    safe_eps: float = 1e-12,
) -> dict[str, NDArray[np.floating]]:
    denom = portfolio_values.astype(float, copy=True)
    denom[np.abs(denom) < safe_eps] = np.nan

    weights_arr = asset_values / denom[None, :, :]
    return {asset: weights_arr[i] for i, asset in enumerate(asset_order)}


def portfolio_weights_forecast_static(
    target_weights: dict[str, float],
    n_sims: int,
    n_periods: int,
    asset_order: list[str],
) -> dict[str, NDArray[np.floating]]:
    return {
        asset: np.full((n_sims, n_periods), float(target_weights[asset]), dtype=float)
        for asset in asset_order
    }


def _validate_target_weights(
    target_weights: dict[str, float],
    asset_order: list[str],
) -> None:
    missing = [a for a in asset_order if a not in target_weights]
    extra = [a for a in target_weights if a not in asset_order]

    if missing:
        raise KeyError(f"Missing target weights for assets: {missing}")
    if extra:
        raise KeyError(f"Extra target weights for unknown assets: {extra}")

    w = np.array([target_weights[a] for a in asset_order], dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError("target_weights must be finite")
    if np.any(w < 0):
        raise ValueError("target_weights must be non-negative")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"target_weights must sum to 1. Got {w.sum():.8f}")


def _initial_portfolio_values_from_shares(
    prices: NDArray[np.floating],
    initial_asset_shares: dict[str, int],
    asset_order: list[str],
) -> NDArray[np.floating]:
    """Compute per-simulation t0 portfolio value from the pre-built price stack.

    Parameters
    ----------
    prices
        Price stack of shape (n_assets, n_sims, n_periods) where column 0 is t0.
    """
    shares = _shares_vector(initial_asset_shares, asset_order)
    # t0 prices are at column 0; multiply by shares and sum over assets
    return (prices[:, :, 0] * shares[:, None]).sum(axis=0)  # (n_sims,)


def _portfolio_values_static_weights(
    prices: NDArray[np.floating],
    target_weights: dict[str, float],
    asset_order: list[str],
    initial_portfolio_values: NDArray[np.floating],
    safe_eps: float = 1e-12,
) -> NDArray[np.floating]:
    """Compute portfolio values under static (rebalanced) weights.

    Parameters
    ----------
    prices
        Pre-built price stack of shape (n_assets, n_sims, n_periods) where
        column 0 is the known t0 price.
    """
    _, n_sims, n_periods = prices.shape

    initial_portfolio_values = np.asarray(initial_portfolio_values, dtype=float)
    if initial_portfolio_values.shape != (n_sims,):
        raise ValueError(
            f"initial_portfolio_values must have shape ({n_sims},), "
            f"got {initial_portfolio_values.shape}"
        )

    _validate_target_weights(target_weights, asset_order)
    w = np.array([target_weights[a] for a in asset_order], dtype=float)

    prev = prices[:, :, :-1]
    curr = prices[:, :, 1:]

    denom = prev.astype(float, copy=True)
    denom[np.abs(denom) < safe_eps] = np.nan
    asset_returns = (curr - prev) / denom  # (n_assets, n_sims, n_periods - 1)

    portfolio_returns = np.einsum("a,ast->st", w, asset_returns)

    values = np.empty((n_sims, n_periods), dtype=float)
    values[:, 0] = initial_portfolio_values
    values[:, 1:] = initial_portfolio_values[:, None] * np.cumprod(
        1.0 + portfolio_returns,
        axis=1,
    )
    return values


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
        pnl=pnl,
        pnl_type=pnl_type,
        weight_mode=weight_mode,
        asset_weights=weights,
        path_probs=path_probs,
    )

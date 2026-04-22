from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from utils.helpers import wide_to_long

WEIGHT_MODE = Literal["buy_and_hold", "static"]


@dataclass(frozen=True, slots=True)
class PortfolioInfoT0:
    all_info: pl.DataFrame
    estimated_t0: float
    shares_mapping: dict[str, int]
    t0_prices: dict[str, float]


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


def equal_weight_target_weights(asset_order: list[str]) -> dict[str, float]:
    if not asset_order:
        raise ValueError("asset_order cannot be empty")

    w = 1.0 / len(asset_order)
    return {asset: w for asset in asset_order}


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

    # Inline extraction of shares and prices
    shares_mapping = dict(
        zip(
            positions["ticker"].to_list(),
            positions["shares"].to_list(),
        )
    )

    t0_prices = dict(
        zip(
            latest_price["ticker"].to_list(),
            latest_price["adj_close"].to_list(),
        )
    )

    return PortfolioInfoT0(
        all_info=positions,
        estimated_t0=total,
        shares_mapping=shares_mapping,
        t0_prices=t0_prices,
    )


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
    missing = [a for a in asset_order if a not in initial_asset_shares]
    if missing:
        raise KeyError(f"No shares specified for assets: {missing}")

    shares = np.array([initial_asset_shares[a] for a in asset_order], dtype=float)
    if np.any(shares < 0):
        raise ValueError("initial_asset_shares must be non-negative")

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

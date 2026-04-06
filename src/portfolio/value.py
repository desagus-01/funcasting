from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy._typing import NDArray

from utils.helpers import wide_to_long
from utils.visuals import plot_simulation_results

PnL_OPTIONS = Literal["relative", "absolute", "log"]


@dataclass(frozen=True, slots=True)
class PortfolioInfot0:
    all_info: pl.DataFrame
    estimated_t0: float
    shares_mapping: dict[str, int]


@dataclass(frozen=True, slots=True)
class PortfolioForecast:
    values: NDArray[np.floating]
    pnl: NDArray[np.floating]
    pnl_type: PnL_OPTIONS
    asset_weights: dict[str, NDArray[np.floating]]

    def plot(
        self, value_type: Literal["value", "pnl"] = "pnl", plot_cumulative: bool = False
    ) -> None:
        if not plot_cumulative:
            data = self.values if value_type == "value" else self.pnl
            plot_simulation_results(data, title=f"Portfolio {value_type}")
            return

        if value_type == "value":
            initial = self.values[:, :1].astype(float)
            initial[np.abs(initial) < 1e-12] = np.nan
            cumulative_changes = self.values / initial - 1.0
        else:
            if self.pnl_type == "relative":
                cumulative_changes = np.concatenate(
                    [
                        np.zeros((self.pnl.shape[0], 1)),
                        np.cumprod(1.0 + self.pnl, axis=1) - 1.0,
                    ],
                    axis=1,
                )
            elif self.pnl_type == "absolute":
                cumulative_changes = np.concatenate(
                    [np.zeros((self.pnl.shape[0], 1)), np.cumsum(self.pnl, axis=1)],
                    axis=1,
                )
            else:  # log
                cumulative_changes = np.concatenate(
                    [
                        np.zeros((self.pnl.shape[0], 1)),
                        np.exp(np.cumsum(self.pnl, axis=1)) - 1.0,
                    ],
                    axis=1,
                )

        plot_simulation_results(
            cumulative_changes, title=f"Portfolio Cumulative {value_type}"
        )


def get_latest_prices(data_long: pl.DataFrame) -> pl.DataFrame:
    if "date" in data_long.columns:
        data_long = data_long.sort(["ticker", "date"])
    return data_long.group_by("ticker").agg(
        pl.col("adj_close").last().alias("adj_close")
    )


def equal_weight_shares_from_prices(
    latest_price: pl.DataFrame, initial_value: float, n_assets: int | None = None
) -> pl.DataFrame:
    if n_assets is None:
        n_assets = latest_price.height

    assigned = initial_value / n_assets

    eq_portfolio = (
        latest_price.with_columns(
            # nominal shares (floor to integer shares)
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
            (pl.lit(1 / n_assets)).alias("target_pct"),
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

    return eq_portfolio


def portfolio_value(positions: pl.DataFrame) -> tuple[float, pl.DataFrame]:
    if "value_allocated" not in positions.columns:
        positions = positions.with_columns(
            (pl.col("adj_close") * pl.col("shares")).alias("value_allocated")
        )

    total = float(positions.select(pl.col("value_allocated").sum()).to_series()[0])

    # avoid division by zero
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


def build_equal_weight_portfolio_from_df(
    data: pl.DataFrame, initial_value: float, assets: list[str] | None = None
) -> PortfolioInfot0:
    cols = set(data.columns)

    if {"ticker", "adj_close"}.issubset(cols):
        data_long = (
            data.with_columns(pl.col("date").cast(pl.Date)) if "date" in cols else data
        )
        if assets is None:
            assets = list(
                data_long.select(pl.col("ticker")).unique().to_series().to_list()
            )
        else:
            data_long = data_long.filter(pl.col("ticker").is_in(assets))
    else:
        if assets is None:
            assets = [c for c in data.columns if c != "date"]
        data_long = wide_to_long(data, assets)

    latest_price = get_latest_prices(data_long)
    eq_portfolio = equal_weight_shares_from_prices(
        latest_price, initial_value, n_assets=len(assets)
    )
    total, positions = portfolio_value(eq_portfolio)

    return PortfolioInfot0(
        all_info=positions,
        estimated_t0=total,
        shares_mapping=_get_ticker_shares(positions),
    )


def _calc_asset_value_forecast(
    asset_forecasts: dict[str, NDArray[np.floating]],
    initial_asset_shares: dict[str, int],
) -> dict[str, NDArray[np.floating]]:
    return {
        asset: mc_forecast * initial_asset_shares[asset]
        for asset, mc_forecast in asset_forecasts.items()
    }


def _build_asset_value_arrays(
    asset_forecasts: dict[str, NDArray[np.floating]],
    initial_asset_shares: dict[str, int],
    asset_order: list[str],
) -> dict[str, NDArray[np.floating]]:
    """Validate inputs and return per-asset value arrays (price * shares).

    Returns
    -------
    dict[str, NDArray[np.floating]]
        Mapping from ticker -> value array of shape (n_sims, n_periods).
    """
    missing = [a for a in asset_order if a not in asset_forecasts]
    extra = [a for a in asset_forecasts.keys() if a not in asset_order]
    if missing:
        raise KeyError(f"Missing asset forecasts for: {missing}")
    if extra:
        raise KeyError(
            f"Found forecasts for assets not present in initial_asset_shares/order: {extra}"
        )

    result: dict[str, NDArray[np.floating]] = {}
    base_shape = None
    for asset in asset_order:
        arr = asset_forecasts[asset]
        if base_shape is None:
            base_shape = arr.shape
        elif arr.shape != base_shape:
            raise ValueError(
                f"Shape mismatch for asset {asset}: {arr.shape} != {base_shape}"
            )
        shares = initial_asset_shares.get(asset)
        if shares is None:
            raise KeyError(f"No shares specified for asset {asset}")
        result[asset] = arr * shares
    return result


def portfolio_value_forecast(
    asset_forecasts: dict[str, NDArray[np.floating]],
    initial_asset_shares: dict[str, int],
    asset_order: list[str] | None = None,
) -> NDArray[np.floating]:
    """Compute forecasted portfolio value paths.

    Parameters
    ----------
    asset_forecasts : dict[str, NDArray[np.floating]]
        Mapping from asset ticker -> simulated price paths with shape
        (n_sims, n_periods).
    initial_asset_shares : dict[str, int]
        Mapping from asset ticker -> integer (or fractional) shares held at t0.
    asset_order : list[str] | None
        Deterministic order to use when stacking arrays. If None, the order of
        keys in ``initial_asset_shares`` is used.

    Notes
    -----
    This function validates that forecasts cover the requested assets and
    that all arrays share the same shape. It returns an array of shape
    (n_sims, n_periods).
    """
    if asset_order is None:
        asset_order = list(initial_asset_shares.keys())

    asset_values = _build_asset_value_arrays(
        asset_forecasts, initial_asset_shares, asset_order
    )
    return np.sum(np.stack(list(asset_values.values()), axis=0), axis=0)


def portfolio_pnl_forecast_from_values(
    forecast_portfolio_values: NDArray[np.floating],
    mode: PnL_OPTIONS = "relative",
    safe_eps: float = 1e-12,
) -> NDArray[np.floating]:
    """Compute PnL/returns from forecasted portfolio values.

    Parameters
    ----------
    forecast_portfolio_values : NDArray[np.floating]
        Array of portfolio values with shape (n_sims, n_periods).
    type : {"relative", "absolute", "log"}
        - "relative": (v_t - v_{t-1}) / v_{t-1} (default)
        - "absolute": v_t - v_{t-1}
        - "log": log(v_t / v_{t-1})
    safe_eps : float
        Small threshold used to treat denominators close to zero. When a
        previous value's absolute magnitude is below safe_eps the result will
        be set to np.nan for modes that divide by the previous value.

    Returns
    -------
    NDArray[np.floating]
        Array shaped (n_sims, n_periods - 1) containing the requested
        PnL/returns. Values that would be infinite due to division by zero are
        set to np.nan.
    """
    if mode not in {"relative", "absolute", "log"}:
        raise ValueError(f"Unknown mode: {mode}")

    prev = forecast_portfolio_values[:, :-1]
    curr = forecast_portfolio_values[:, 1:]

    if mode == "absolute":
        return curr - prev

    # For relative and log modes we must guard against zero/near-zero prev
    small = np.abs(prev) < safe_eps

    if mode == "relative":
        denom = prev.copy().astype(float)
        denom[small] = np.nan
        return (curr - prev) / denom

    denom = prev.copy().astype(float)
    denom[small] = np.nan
    return np.log(curr / denom)


def portfolio_weights_forecast(
    asset_values: dict[str, NDArray[np.floating]],
    portfolio_values: NDArray[np.floating],
    safe_eps: float = 1e-12,
) -> dict[str, NDArray[np.floating]]:
    denom = portfolio_values.copy().astype(float)
    denom[np.abs(denom) < safe_eps] = np.nan

    return {asset: arr / denom for asset, arr in asset_values.items()}


def portfolio_forecast(
    asset_forecasts: dict[str, NDArray[np.floating]],
    initial_asset_shares: dict[str, int],
    asset_order: list[str] | None = None,
    pnl_type: PnL_OPTIONS = "relative",
    safe_eps: float = 1e-12,
) -> PortfolioForecast:
    """Build a full :class:`PortfolioForecast`"""
    if asset_order is None:
        asset_order = list(initial_asset_shares.keys())

    asset_values = _build_asset_value_arrays(
        asset_forecasts, initial_asset_shares, asset_order
    )
    stacked = np.stack(list(asset_values.values()), axis=0)
    values = stacked.sum(axis=0)
    pnl = portfolio_pnl_forecast_from_values(values, mode=pnl_type, safe_eps=safe_eps)
    weights = portfolio_weights_forecast(
        asset_values=asset_values, portfolio_values=values
    )

    return PortfolioForecast(
        values=values, pnl=pnl, pnl_type=pnl_type, asset_weights=weights
    )

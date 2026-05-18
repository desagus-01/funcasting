from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pipelines.forecasting import AssetSubset, ForecastPaths
from portfolio.forecast import PnL_OPTIONS, pnl_from_values
from time_series.estimation import (
    weighted_correlation,
    weighted_covariance,
    weighted_mean,
)


@dataclass(frozen=True, slots=True)
class HorizonMoments:
    assets: list[str]
    correlations: NDArray[np.floating]
    covariances: NDArray[np.floating]
    mean: NDArray[np.floating]

    @property
    def n_horizons(self) -> int:
        return int(self.mean.shape[0])

    @property
    def n_assets(self) -> int:
        return int(self.mean.shape[1])

    @property
    def mean_frame(self) -> pl.DataFrame:
        """Wide frame: one row per horizon, one column per asset."""
        rows = [
            {
                "horizon": h + 1,
                **{a: float(self.mean[h, i]) for i, a in enumerate(self.assets)},
            }
            for h in range(self.n_horizons)
        ]
        return pl.DataFrame(rows)

    def _matrix_frame(self, matrix: NDArray[np.floating], horizon: int) -> pl.DataFrame:
        """Return slice [horizon] of an (H, N, N) array as a square DataFrame."""
        m = matrix[horizon]
        data: dict[str, list] = {"asset": self.assets}
        for i, asset in enumerate(self.assets):
            data[asset] = m[:, i].tolist()
        return pl.DataFrame(data)

    def correlation_frame(self, horizon: int = 0) -> pl.DataFrame:
        return self._matrix_frame(self.correlations, horizon)

    def covariance_frame(self, horizon: int = 0) -> pl.DataFrame:
        return self._matrix_frame(self.covariances, horizon)

    @classmethod
    def from_forecast_paths(
        cls,
        forecast_paths: ForecastPaths,
        horizons: int | None = None,
        subset: AssetSubset = "tradable",
        pnl_type: PnL_OPTIONS = "relative",
    ) -> "HorizonMoments":
        """
        Build stacked multi-horizon moments from simulated price paths.

        Moments at each horizon h use the **incremental** (period-over-period)
        return from t_{h-1} to t_h, not the cumulative return from t_0.

        Parameters
        ----------
        forecast_paths:
            Simulated price paths and associated metadata.
        horizons:
            Number of forecast horizons to compute. Defaults to all available
            horizons in ``forecast_paths``.
        subset:
            Which assets to include (``"tradable"``, ``"factors"``, ``"all"``).
        pnl_type:
            Return type — ``"relative"`` (default), ``"absolute"``, or ``"log"``.
        """
        paths = forecast_paths._paths_for(subset)
        if not paths:
            raise ValueError(f"No paths available for subset={subset!r}")

        max_horizons = forecast_paths.n_horizons
        if horizons is None:
            horizons = max_horizons
        if horizons > max_horizons:
            raise ValueError(
                f"horizons={horizons} exceeds the number of available forecast "
                f"steps ({max_horizons})."
            )

        assets = list(paths.keys())
        n_assets = len(assets)
        prob = forecast_paths.path_probs
        initial_prices = forecast_paths.initial_prices
        n_paths = forecast_paths.n_paths

        # inc_returns[path, horizon, asset] — incremental return at each step
        inc_returns = np.empty((n_paths, horizons, n_assets), dtype=float)

        for col_idx, (asset, price_paths) in enumerate(paths.items()):
            t0 = initial_prices[asset]
            t0_col = np.full((n_paths, 1), t0, dtype=float)
            # values shape: (n_paths, n_forecast_horizons + 1)  [t0 prepended]
            values = np.concatenate([t0_col, price_paths], axis=1)
            # inc shape: (n_paths, n_forecast_horizons) — return from t_{h-1} to t_h
            inc = pnl_from_values(values, mode=pnl_type)
            inc_returns[:, :, col_idx] = inc[:, :horizons]

        # Stack moments across horizons
        means = np.empty((horizons, n_assets), dtype=float)
        covs = np.empty((horizons, n_assets, n_assets), dtype=float)
        corrs = np.empty((horizons, n_assets, n_assets), dtype=float)

        for h in range(horizons):
            pnl_matrix = inc_returns[:, h, :]  # (n_paths, n_assets)
            means[h] = weighted_mean(data=pnl_matrix, prob=prob)
            covs[h] = weighted_covariance(data=pnl_matrix, prob=prob)
            corrs[h] = weighted_correlation(data=pnl_matrix, prob=prob)

        return cls(
            assets=assets,
            correlations=corrs,
            covariances=covs,
            mean=means,
        )

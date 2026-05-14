from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pipelines.forecasting import AssetSubset, ForecastPaths
from polars import DataFrame
from portfolio.forecast import PnL_OPTIONS, cumulative_pnl, pnl_from_values
from scenarios.panel import ScenarioPanel
from scenarios.types import ProbVector
from time_series.estimation import (
    weighted_correlation,
    weighted_covariance,
    weighted_mean,
)


def asset_pnl_from_paths(
    price_paths: NDArray[np.floating],
    initial_price: float,
    path_probs: ProbVector,
    pnl_type: PnL_OPTIONS = "relative",
    safe_eps: float = 1e-12,
) -> ScenarioPanel:
    if price_paths.ndim != 2:
        raise ValueError(
            f"price_paths must be 2-D (n_paths, n_horizons); got {price_paths.shape}"
        )

    n_paths = price_paths.shape[0]
    t0_col = np.full((n_paths, 1), initial_price, dtype=float)
    values = np.concatenate([t0_col, price_paths], axis=1)

    inc = pnl_from_values(values, mode=pnl_type, safe_eps=safe_eps)
    cum = cumulative_pnl(inc, pnl_type)
    n_periods = cum.shape[1]

    return ScenarioPanel(
        values=DataFrame({f"h{h}": cum[:, h] for h in range(n_periods)}),
        dates=None,
        prob=path_probs,
    )


@dataclass
class HorizonMoments:
    assets: list[str]
    correlations: NDArray[np.floating]
    covariances: NDArray[np.floating]
    mean: NDArray[np.floating]

    @property
    def mean_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {asset: [float(val)] for asset, val in zip(self.assets, self.mean)}
        )

    def _matrix_frame(self, matrix: NDArray[np.floating]) -> pl.DataFrame:
        """Return a square matrix as a DataFrame: an 'asset' index column
        plus one column per asset.
        """
        data = {"asset": self.assets}
        for i, asset in enumerate(self.assets):
            data[asset] = matrix[:, i].tolist()
        return pl.DataFrame(data)

    @property
    def correlation_frame(self) -> pl.DataFrame:
        return self._matrix_frame(self.correlations)

    @property
    def covariance_frame(self) -> pl.DataFrame:
        return self._matrix_frame(self.covariances)

    @classmethod
    def from_forecast_paths(
        cls,
        forecast_paths: ForecastPaths,
        step: int,
        subset: AssetSubset = "tradable",
        pnl_type: PnL_OPTIONS = "relative",
    ) -> "HorizonMoments":
        initial_prices = forecast_paths.initial_prices
        paths = forecast_paths._paths_for(subset)
        assets = list(paths.keys())

        pnl_at_step: dict[str, NDArray[np.floating]] = {}
        for asset, price_paths in paths.items():
            t0 = initial_prices[asset]
            panel = asset_pnl_from_paths(
                price_paths=price_paths,
                initial_price=t0,
                path_probs=forecast_paths.path_probs,
                pnl_type=pnl_type,
            )
            pnl_at_step[asset] = panel.values[f"h{step - 1}"].to_numpy()

        prob = forecast_paths.path_probs
        pnl_matrix = np.column_stack(list(pnl_at_step.values()))

        return cls(
            assets=assets,
            correlations=weighted_correlation(data=pnl_matrix, prob=prob),
            covariances=weighted_covariance(data=pnl_matrix, prob=prob),
            mean=weighted_mean(data=pnl_matrix, prob=prob),
        )

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pipelines.forecasting import AssetSubset, ForecastPaths
from time_series.estimation import (
    weighted_correlation,
    weighted_covariance,
    weighted_mean,
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
        cls, forecast_paths: ForecastPaths, step: int, subset: AssetSubset = "tradable"
    ):
        horizon_panel = forecast_paths.at_step(step=step, subset=subset)
        assets = horizon_panel.values.columns
        horizon_vals_np = horizon_panel.values.to_numpy()
        return HorizonMoments(
            assets=assets,
            correlations=weighted_correlation(
                data=horizon_vals_np, prob=horizon_panel.prob
            ),
            covariances=weighted_covariance(
                data=horizon_vals_np, prob=horizon_panel.prob
            ),
            mean=weighted_mean(data=horizon_vals_np, prob=horizon_panel.prob),
        )

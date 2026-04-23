from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import reduce
from typing import Literal, Mapping

import numpy as np
import polars as pl
from numpy._typing import NDArray
from polars import DataFrame

from pipelines.model_selection import (
    get_univariate_results,
)
from pipelines.preprocess import (
    run_univariate_preprocess,
)
from scenarios.copula_marginal import CopulaMarginalModel
from scenarios.panel import AssetPanel
from scenarios.resampling import weighted_bootstrapping_idx
from scenarios.types import ProbVector
from simulation.simulate_paths import (
    simulate_asset_paths,
)
from simulation.state import SimulationForecast
from time_series.models.fitted_types import UnivariateRes
from time_series.transforms.inverses import apply_inverse_transforms

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AssetUniverse:
    """Classifies tickers as tradable assets or non-tradable factors.

    Both assets and factors are forecast together (preserving cross-correlations),
    but only assets participate in portfolio construction (weights, PnL, value).
    """

    assets: list[str]
    factors: list[str]

    def __post_init__(self) -> None:
        overlap = set(self.assets) & set(self.factors)
        if overlap:
            raise ValueError(
                f"Tickers appear in both assets and factors: {sorted(overlap)}"
            )
        if not self.assets:
            raise ValueError("Must have at least one tradable asset")

    @property
    def all_tickers(self) -> list[str]:
        """All tickers in forecast order (assets first, then factors)."""
        return self.assets + self.factors

    def is_factor(self, ticker: str) -> bool:
        return ticker in set(self.factors)

    def is_asset(self, ticker: str) -> bool:
        return ticker in set(self.assets)


@dataclass(frozen=True, slots=True)
class InnovationPaths:
    values: NDArray[np.floating]
    path_probs: ProbVector


@dataclass(frozen=True, slots=True)
class ForecastPaths:
    asset_paths: dict[str, NDArray[np.floating]]
    path_probs: ProbVector
    universe: AssetUniverse | None = None

    @property
    def tradable_paths(self) -> dict[str, NDArray[np.floating]]:
        """Only paths for tradable assets (excludes factors)."""
        if self.universe is None:
            return self.asset_paths
        assets_set = set(self.universe.assets)
        return {k: v for k, v in self.asset_paths.items() if k in assets_set}

    @property
    def factor_paths(self) -> dict[str, NDArray[np.floating]]:
        """Only paths for factors (non-tradable)."""
        if self.universe is None:
            return {}
        factors_set = set(self.universe.factors)
        return {k: v for k, v in self.asset_paths.items() if k in factors_set}


def _build_innovations_df_from_models(
    post: DataFrame,
    model_map: dict[str, UnivariateRes],
    assets: list[str] | None = None,
) -> DataFrame:
    """
    Build a dataframe of invariant (innovation) series for given assets.

    This function extracts the invariant (innovation) series from fitted
    univariate models for each asset and aligns them with the dates present
    in the provided `post` dataframe. Missing values are preserved using a
    left-join on the date index.

    Parameters
    ----------
    post : DataFrame
        Preprocessed dataframe containing per-asset post series and a ``date`` column.
    model_map : dict[str, UnivariateRes]
        Mapping from asset name to fitted univariate results; used to compute invariants.
    assets : list[str] | None, optional
        Explicit list of assets to include. If ``None``, all columns except
        'date' in ``post`` are used.

    Returns
    -------
    DataFrame
        Polars DataFrame with the 'date' column and one column per asset
        containing the invariant series (aligned by date).
    """
    if assets is None:
        assets = [c for c in post.columns if c != "date"]

    logger.info("Building innovations dataframe for assets=%s", assets)

    base = post.select("date")
    patches: list[pl.DataFrame] = []

    for asset in assets:
        model = model_map[asset]
        used_dates = post.filter(pl.col(asset).is_not_null()).select("date")

        non_null_values = post.select(asset).drop_nulls().to_numpy().ravel()
        invariant = model.invariant(non_null_values)

        patch = used_dates.with_columns(pl.Series(asset, invariant))
        patches.append(patch)

    innovations_full = reduce(
        lambda acc, p: acc.join(p, on="date", how="left"), patches, base
    )

    logger.info(
        "Built innovations dataframe with shape=%s",
        innovations_full.shape,
    )
    return innovations_full


def draw_innovations(
    invariants_df: DataFrame,
    assets: list[str],
    prob_vector: ProbVector,
    horizon: int,
    n_sims: int,
    seed: int | None,
    method: Literal["bootstrap", "historical", "cma"] = "bootstrap",
    *,
    target_copula: Literal["t", "norm"] | None = None,
    copula_fit_method: Literal["ml", "irho", "itau"] | None = None,
    target_marginals: dict[str, Literal["t", "norm"]] | None = None,
) -> InnovationPaths:
    """
    Draw innovation (invariant) scenarios for simulation.

    This function prepares invariant scenarios from the provided
    ``invariants_df`` and draws simulated innovations according to the
    specified method. Supported methods are bootstrap resampling,
    returning the historical set without resampling, or applying a
    copula-marginal adjustment (CMA) update.

    Parameters
    ----------
    invariants_df : DataFrame
        Polars DataFrame with one column per asset (and optionally a 'date'
        column) containing invariant observations.
    assets : list[str]
        Ordered list of assets corresponding to the columns to draw from.
    prob_vector : ProbVector
        Prior probability weights corresponding to the rows of ``invariants_df``.
    horizon : int
        Number of steps to forecast (must be >= 1).
    n_sims : int
        Number of simulation paths to draw.
    seed : int | None
        Random seed for reproducibility.
    method : {"bootstrap", "historical", "cma"}, optional
        Resampling method. "historical" returns the historical draws
        without resampling (only supported when horizon==1), "bootstrap"
        draws with weighted bootstrapping, and "cma" applies an entropy
        pool / copula-marginal adjustment prior to resampling.

    Other Parameters
    ----------------
    target_copula : {"t", "norm"} or None, optional
        Target copula family for CMA updates (only valid when ``method`` is "cma").
    copula_fit_method : {"ml", "irho", "itau"} or None, optional
        Method to fit the copula when applying CMA. If not provided,
        defaults to "itau".
    target_marginals : dict[str, {"t", "norm"}] or None, optional
        Per-asset desired marginal families for CMA updates.

    Returns
    -------
    NDArray[np.floating]
        Array of simulated draws reshaped as ``(n_sims, horizon, n_assets)``
        for bootstrap/CMA methods. For the historical method returns the
        raw historical invariants in shape ``(n_scenarios, 1, n_assets)``.

    Raises
    ------
    ValueError
        If ``horizon < 1``, or if CMA-specific options are provided when
        ``method`` is not "cma".
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    logger.info(
        "Drawing innovations with method=%s, horizon=%d, n_sims=%d, assets=%s",
        method,
        horizon,
        n_sims,
        assets,
    )

    invariants_df = invariants_df.select(assets)
    panel = AssetPanel.from_frame(invariants_df, prob_vector).drop_nulls()

    logger.info(
        "Prepared invariant scenarios after null handling: n_scenarios=%d",
        panel.n_rows,
    )

    if (target_copula is not None or target_marginals is not None) and method != "cma":
        raise ValueError(
            "You can only have target_marginal and/or target_copula when method is cma!"
        )

    if method == "cma":
        if copula_fit_method is None:
            copula_fit_method = "itau"
            logger.info(
                "No copula_fit_method provided; defaulting to '%s'", copula_fit_method
            )

        logger.info(
            "Applying CMA update with target_copula=%s target_marginals=%s",
            target_copula,
            target_marginals,
        )

        panel = CopulaMarginalModel.from_panel(panel).update_distribution(
            seed=seed,
            target_marginals=target_marginals,
            target_copula=target_copula,
            copula_fit_method=copula_fit_method,
        )

        logger.info("CMA update complete: n_scenarios=%d", panel.n_rows)

    invariants_vector = panel.values.to_numpy()
    prob = panel.prob

    if method == "historical":
        logger.info("Returning historical innovations without resampling")
        simulated_draws = invariants_vector[:, None, :]
        return InnovationPaths(values=simulated_draws, path_probs=prob)

    n_draws = n_sims * horizon
    logger.info("Bootstrapping %d innovation draws", n_draws)

    idx = weighted_bootstrapping_idx(
        panel.values,
        prob,
        n_samples=n_draws,
        seed=seed,
    )
    simulated_draws = invariants_vector[idx].reshape(n_sims, horizon, len(assets))

    logger.info("Innovation draw complete with output shape=%s", simulated_draws.shape)
    return InnovationPaths(
        values=simulated_draws,
        path_probs=np.full(n_sims, 1.0 / n_sims),  # ie uniform for MC
    )


def get_assets_models(
    data: DataFrame,
    assets_univariate_result: Mapping[str, UnivariateRes],
    assets: list[str] | None = None,
) -> dict[str, SimulationForecast]:
    """
    Construct simulation forecast objects for each asset.

    For each asset this function builds a ``SimulationForecast`` using the
    fitted univariate results and the non-null post-processed series. The
    returned mapping can be passed to the simulation engine to generate
    asset paths.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing the post-processed series (including a 'date'
        column).
    assets_univariate_result : Mapping[str, UnivariateRes]
        Mapping of asset name to fitted univariate results used to build the
        simulation model/state for each asset.
    assets : list[str] | None, optional
        List of assets to prepare. If ``None``, all columns except 'date' are used.

    Returns
    -------
    dict[str, SimulationForecast]
        Mapping from asset name to a ``SimulationForecast`` containing the
        univariate model and initial state needed for simulation.
    """
    assets_ = assets if assets is not None else [c for c in data.columns if c != "date"]

    forecast_models: dict[str, SimulationForecast] = {}
    for asset in assets_:
        post_series_non_null = data.select(asset).drop_nulls().to_numpy().ravel()
        forecast_models[asset] = SimulationForecast.from_res_and_series(
            fitting_results=assets_univariate_result[asset],
            post_series_non_null=post_series_non_null,
        )

    return forecast_models


def run_n_steps_forecast(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    horizon: int = 100,
    n_sims: int = 1000,
    seed: int | None = None,
    method: Literal["bootstrap", "historical", "cma"] = "bootstrap",
    *,
    factors: list[str] | None = None,
    back_to_price: bool = True,
    target_copula: Literal["t", "norm"] | None = None,
    copula_fit_method: Literal["ml", "irho", "itau"] | None = None,
    target_marginals: dict[str, Literal["t", "norm"]] | None = None,
) -> ForecastPaths:
    """
    Run a full n-step forecasting pipeline for a set of assets.

    The pipeline performs the following high-level steps:
      1. Run univariate preprocessing (detrending / deseasonalising) to
         prepare invariant series and inverse transform specifications.
      2. Select and fit univariate mean/volatility models where required.
      3. Build per-asset simulation models/state objects.
      4. Draw innovations according to the chosen method (bootstrap/historical/CMA).
      5. Simulate asset paths and apply inverse transforms to obtain results
         on the original scale if requested.

    Parameters
    ----------
    data : DataFrame
        Raw input data containing a 'date' column and one column per asset.
    prob : ProbVector
        Prior probability vector to use when preparing invariants and
        performing any probability-weighted resampling.
    assets : list[str]
        List of assets to include in the forecast (order determines asset
        ordering in the returned arrays).
    horizon : int, optional
        Number of forecasting steps to simulate (default: 100).
    n_sims : int, optional
        Number of Monte Carlo simulation paths to generate (default: 1000).
    seed : int | None, optional
        RNG seed for reproducibility.
    method : {"bootstrap", "historical", "cma"}, optional
        Method used to draw innovations. See :func:`draw_innovations` for
        details and CMA-specific options.

    Other Parameters
    ----------------
    factors : list[str] | None, optional
        Tickers that are non-tradable factors. They are forecast jointly with
        assets (preserving cross-correlations) but are excluded from portfolio
        construction. Access factor-only paths via ``ForecastPaths.factor_paths``
        and tradable-only paths via ``ForecastPaths.tradable_paths``.
    back_to_price : bool, optional
        Whether to convert simulated returns back to price level using the
        stored inverse transforms (default: True).
    target_copula, copula_fit_method, target_marginals
        CMA-specific options forwarded to :func:`draw_innovations` when
        ``method == 'cma'``.

    Returns
    -------
    - ``transformed`` is the result of applying inverse transforms
      to obtain outputs on the original data scale.

    Raises
    ------
    ValueError
        If invalid method/horizon/asset combinations are provided (e.g.
        historical method with horizon>1, CMA with a single asset, etc.).
    """
    logger.info(
        "Starting n-step forecast: assets=%s horizon=%d n_sims=%d method=%s seed=%s",
        assets,
        horizon,
        n_sims,
        method,
        seed,
    )

    if (horizon > 1) and (method == "historical"):
        raise ValueError(
            "Historical method for innovations can only be used for one step forecasts."
        )
    if (len(assets) <= 1) and (method == "cma"):
        raise ValueError(
            "Must have more than one asset in order to use the copula method."
        )

    post_process = run_univariate_preprocess(data=data, prob=prob, assets=assets)
    logger.info(
        "Preprocessing complete: post_data_shape=%s assets_to_model=%s",
        post_process.post_data.shape,
        post_process.needs_further_modelling,
    )

    univariate_results = get_univariate_results(
        data=post_process.post_data,
        assets_to_model=post_process.needs_further_modelling,
    )
    logger.info(
        "Univariate model selection complete for %d assets",
        len(post_process.needs_further_modelling),
    )

    assets_models = get_assets_models(
        data=post_process.post_data,
        assets_univariate_result=univariate_results,
        assets=assets,
    )

    invariants_df = _build_innovations_df_from_models(
        post=post_process.post_data,
        model_map=univariate_results,
        assets=assets,
    )

    innovations = draw_innovations(
        invariants_df=invariants_df,
        assets=assets,
        prob_vector=prob,
        horizon=horizon,
        n_sims=n_sims,
        seed=seed,
        method=method,
        target_copula=target_copula,
        target_marginals=target_marginals,
        copula_fit_method=copula_fit_method,
    )

    logger.info("Simulating asset paths for %d assets", len(assets))

    assets_forecasts = {}
    for i, asset in enumerate(assets):
        assets_forecasts[asset] = simulate_asset_paths(
            forecast_model=assets_models[asset],
            innovations=innovations.values[:, :, i],
        )

    logger.info("Forecast complete - will apply inverse transforms now")
    transformed = apply_inverse_transforms(
        asset_data_dict=assets_forecasts,
        n_original=data.height,
        inverse_specs=post_process.inverse_specs,
        back_to_price=True if back_to_price else False,
    )

    factors_ = factors if factors is not None else []
    tradable = [a for a in assets if a not in set(factors_)]
    universe = AssetUniverse(assets=tradable, factors=factors_)

    return ForecastPaths(
        asset_paths=transformed,
        path_probs=innovations.path_probs,
        universe=universe,
    )

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame

from pipelines.fitted_universe import FittedUniverse
from scenarios.copula_marginal import CopulaMarginalModel
from scenarios.panel import AssetPanel
from scenarios.resampling import weighted_bootstrapping_idx
from scenarios.types import ProbVector
from time_series.transforms.inverses import apply_inverse_transforms

logger = logging.getLogger(__name__)

Method = Literal["bootstrap", "historical", "cma"]


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


def _validate_method_options(method: Method, horizon: int, assets: list[str]) -> None:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if horizon > 1 and method == "historical":
        raise ValueError(
            "Historical method for innovations can only be used for one-step forecasts."
        )
    if len(assets) <= 1 and method == "cma":
        raise ValueError(
            "Must have more than one asset in order to use the copula method."
        )


def draw_innovations(
    invariants: AssetPanel,
    horizon: int,
    n_sims: int,
    seed: int | None,
    method: Method = "bootstrap",
    *,
    target_copula: Literal["t", "norm"] | None = None,
    copula_fit_method: Literal["ml", "irho", "itau"] | None = None,
    target_marginals: dict[str, Literal["t", "norm"]] | None = None,
) -> InnovationPaths:
    """Draw innovation (invariant) scenarios for simulation.

    Parameters
    ----------
    invariants : AssetPanel
        Panel of invariant observations — one column per asset, in the
        desired asset order. Nulls are dropped (with prob compensation) here.
    horizon : int
        Number of steps to forecast (must be >= 1).
    n_sims : int
        Number of simulation paths to draw.
    seed : int | None
        Random seed for reproducibility.
    method : {"bootstrap", "historical", "cma"}, optional
        ``historical`` returns historical draws without resampling (horizon
        must be 1); ``bootstrap`` draws with weighted bootstrapping;
        ``cma`` applies a copula-marginal adjustment prior to resampling.

    Other Parameters
    ----------------
    target_copula, copula_fit_method, target_marginals
        CMA-only options (ignored for bootstrap/historical). A ``ValueError``
        is raised if they are supplied with a non-CMA method.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    assets = invariants.asset_names

    logger.info(
        "Drawing innovations with method=%s, horizon=%d, n_sims=%d, assets=%s",
        method,
        horizon,
        n_sims,
        assets,
    )

    if (target_copula is not None or target_marginals is not None) and method != "cma":
        raise ValueError(
            "You can only have target_marginal and/or target_copula when method is cma!"
        )

    panel = invariants.drop_nulls()

    logger.info(
        "Prepared invariant scenarios after null handling: n_scenarios=%d",
        panel.n_rows,
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
        return InnovationPaths(
            values=invariants_vector[:, None, :],
            path_probs=prob,
        )

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
        path_probs=np.full(n_sims, 1.0 / n_sims),  # uniform for MC
    )


def run_n_steps_forecast(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    horizon: int = 100,
    n_sims: int = 1000,
    seed: int | None = None,
    method: Method = "bootstrap",
    *,
    factors: list[str] | None = None,
    back_to_price: bool = True,
    target_copula: Literal["t", "norm"] | None = None,
    copula_fit_method: Literal["ml", "irho", "itau"] | None = None,
    target_marginals: dict[str, Literal["t", "norm"]] | None = None,
) -> ForecastPaths:
    """Run a full n-step forecasting pipeline for a set of assets.

    Pipeline stages:
      1. ``FittedUniverse.fit`` — preprocess, model selection, build invariants.
      2. ``draw_innovations`` — draw innovation paths for the chosen method.
      3. ``FittedUniverse.simulate`` — per-asset path simulation.
      4. ``apply_inverse_transforms`` — lift back to the original scale.

    Parameters
    ----------
    data : DataFrame
        Raw input with a ``date`` column and one numeric column per asset.
    prob : ProbVector
        Prior probability vector over historical rows.
    assets : list[str]
        Assets to forecast (order is preserved through the pipeline).
    horizon, n_sims, seed, method
        See :func:`draw_innovations`.

    Other Parameters
    ----------------
    factors
        Non-tradable factors forecast jointly with assets but excluded from
        portfolio construction. Available as ``ForecastPaths.factor_paths``.
    back_to_price
        If True, inverse transforms include the final ``exp(...)`` step to
        return to price scale.
    target_copula, copula_fit_method, target_marginals
        CMA-only options; forwarded to :func:`draw_innovations`.
    """
    logger.info(
        "Starting n-step forecast: assets=%s horizon=%d n_sims=%d method=%s seed=%s",
        assets,
        horizon,
        n_sims,
        method,
        seed,
    )

    _validate_method_options(method, horizon, assets)

    universe_fit = FittedUniverse.fit(data=data, prob=prob, assets=assets)

    innovations = draw_innovations(
        invariants=universe_fit.invariants,
        horizon=horizon,
        n_sims=n_sims,
        seed=seed,
        method=method,
        target_copula=target_copula,
        target_marginals=target_marginals,
        copula_fit_method=copula_fit_method,
    )

    simulated = universe_fit.simulate(innovations.values)

    logger.info("Forecast complete - applying inverse transforms")
    transformed = apply_inverse_transforms(
        asset_data_dict=simulated,
        n_original=data.height,
        inverse_specs=universe_fit.inverse_specs,
        back_to_price=back_to_price,
    )

    factors_ = factors or []
    tradable = [a for a in assets if a not in set(factors_)]
    return ForecastPaths(
        asset_paths=transformed,
        path_probs=innovations.path_probs,
        universe=AssetUniverse(assets=tradable, factors=factors_),
    )

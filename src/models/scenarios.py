from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl
from polars import DataFrame

from methods.cma import CopulaMarginalModel
from methods.ep import entropy_pooling_probs
from models.types import ProbVector, View
from utils.distributions import uniform_probs
from utils.stat_tests import SWRes, sw_mc_summary


@dataclass(frozen=True)
class ScenarioDistribution:
    """
    Core representation of a discrete scenario distribution.

    All complex processes (EP, CMA) take a ScenarioDistribution in
    and return a new ScenarioDistribution out.
    """

    scenarios: DataFrame
    prob: ProbVector
    dates: DataFrame | None = None

    def __post_init__(self):
        # Extract date column if present
        if "date" in self.scenarios.columns:
            date_df = self.scenarios.select(pl.col("date"))
            scenarios_exdate = self.scenarios.drop("date")
            object.__setattr__(self, "scenarios", scenarios_exdate)
            object.__setattr__(self, "dates", date_df)

        # Always check prob length
        if self.scenarios.height != self.prob.shape[0]:
            raise ValueError("prob vector length does not match number of scenarios")

        # Only check dates if they exist
        if self.dates is not None and self.scenarios.height != self.dates.height:
            raise ValueError("dates do not have the same number of rows as scenarios")

    @classmethod
    def default_instance(
        cls, scenarios: DataFrame, prob: ProbVector | None = None
    ) -> ScenarioDistribution:
        """
        Defaults prob vector to uniform if None
        """
        if prob is None:
            prob = uniform_probs(scenarios.height)
        return cls(scenarios=scenarios, prob=prob)


@dataclass
class ScenarioProb:
    """
    Orchestrator for ScenarioDistribution

    Owns:
    1. Original ScenarioDistribution
    2. Updated Scenario Distribution
    3. List of Views

    Delegates CMA to CopulaMarginalModel and entropy pooling to entry_pooling_probs
    """

    _base_dist: ScenarioDistribution
    _dist: ScenarioDistribution
    views: list[View] = field(default_factory=list)

    @classmethod
    def default_inst(
        cls, scenarios: DataFrame, prob: ProbVector | None = None
    ) -> ScenarioProb:
        dist = ScenarioDistribution.default_instance(scenarios=scenarios, prob=prob)
        return cls(_dist=dist, _base_dist=dist)

    @property
    def assets(self) -> list[str]:
        return self.scenarios.columns

    @property
    def scenarios(self) -> DataFrame:
        return self._dist.scenarios

    @property
    def prob(self) -> ProbVector:
        return self._dist.prob

    @property
    def dates(self) -> DataFrame | None:
        if self._dist.dates is not None:
            return self._dist.dates
        else:
            pass

    def add_views(self, new_views: list[View]) -> ScenarioProb:
        """
        Updates ScenarioProb with additional views
        """

        return ScenarioProb(
            _base_dist=self._base_dist,
            _dist=self._dist,
            views=[*self.views, *new_views],
        )

    def clear_views(self) -> ScenarioProb:
        """
        Removes all Views.
        """
        return ScenarioProb(
            _base_dist=self._base_dist,
            _dist=self._dist,
            views=[],
        )

    def apply_views(
        self, confidence: float = 1.0, include_diags: bool = False
    ) -> ScenarioProb:
        """
        Applies Entropy Pooling to current object using views.
        Returns a new ScenarioProb with updated probabilities.
        """

        new_prob = entropy_pooling_probs(
            prior=self.prob,
            views=self.views,
            confidence=confidence,
            include_diags=include_diags,
        )

        new_dist = ScenarioDistribution(
            scenarios=self.scenarios, prob=new_prob, dates=self.dates
        )

        return ScenarioProb(
            _base_dist=self._base_dist, _dist=new_dist, views=self.views
        )

    def apply_cma(
        self,
        target_marginals: dict[str, Literal["t", "norm"]] | None = None,
        target_copula: Literal["t", "norm"] | None = None,
    ) -> ScenarioProb:
        """
        Applies CMA to current scenario distribution using current probs and scenarios.
        Returns a new ScenarioProb with updated scenarios.
        """
        if target_copula is None and target_marginals is None:
            raise ValueError("You must choose a target marginal or copula")

        scenarios, prob = CopulaMarginalModel.from_scenario_dist(
            self.scenarios, self.prob
        ).update_distribution(
            target_marginals=target_marginals, target_copula=target_copula
        )

        new_dist = ScenarioDistribution(
            scenarios=scenarios, prob=prob, dates=self.dates
        )

        return ScenarioProb(
            _base_dist=self._base_dist,
            _dist=new_dist,
            views=self.views,
        )

    def schweizer_wolff(
        self, assets: tuple[str, str], iter: int = 50, original_dist: bool = True
    ) -> SWRes:
        """
        Compute the Schweizer–Wolff dependence measure between two assets using
        Monte Carlo integration of their copula.

        The Schweizer–Wolff measure is a non-parametric dependence metric that
        quantifies how far the copula of two variables deviates from
        what would be expected under independence. It is defined as:

            SW = 12 ∫₀¹ ∫₀¹ | C(u₁, u₂) − C_indep(u₁, u₂) | du₁ du₂,

        where:
            - C(u₁, u₂) is the empirical copula of the two assets,
            - C_indep(u₁, u₂) = u₁ * u₂ is the independence copula.

        Because this double integral has no
        closed form for empirical copulas, it is approximated via Monte Carlo
        sampling over the unit square.

        Parameters
        ----------
        assets : tuple[str, str]
            The pair of assets for which to compute the dependence measure.
            Both assets must exist in the underlying scenario set.

        iter : int, default=50
            Number of Monte Carlo replications. Increasing this improves stability
            of the estimate but increases computation time.

        original_dist : bool, default=True
            If True, use the original scenario distribution prior to any views and/or cma.
            If False, use the current distribution after applying views or CMA
            adjustments.

        Returns
        -------
        SWRes
            A dictionary containing:
            - ``iter_res``: The Schweizer–Wolff estimate from each Monte Carlo run.
            - ``iter_avg``: The averaged dependence estimate across runs.
        """

        if any(asset not in self.assets for asset in assets):
            raise ValueError(
                f"Your two chosen assets {assets} must be in your scenarios; these are {self.assets}"
            )

        if original_dist:
            dist = self._base_dist
        else:
            dist = self._dist

        cma = CopulaMarginalModel.from_scenario_dist(dist.scenarios, dist.prob)

        cop_assets = cma.copula.select(assets).to_numpy()
        return sw_mc_summary(cop_assets, cma.prob, iter)

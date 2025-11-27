from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl
from polars import DataFrame

from methods.cma import CopulaMarginalModel
from methods.ep import entropy_pooling_probs
from models.types import ProbVector, View
from utils.distributions import uniform_probs


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
            scenarios_exdate = self.scenarios.drop(pl.col("date"))
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
        if not prob:
            prob = uniform_probs(scenarios.height)
        return cls(scenarios=scenarios, prob=prob)


@dataclass
class ScenarioProb:
    """
    Orchestrator for ScenarioDistribution

    Owns:
    1. Current ScenarioDistribution
    2. List of Views

    Delegates CMA to CopulaMarginalModel and entropy pooling to EntropyPooling
    """

    _dist: ScenarioDistribution
    views: list[View] = field(default_factory=list)

    @classmethod
    def default_inst(
        cls, scenarios: DataFrame, prob: ProbVector | None = None
    ) -> ScenarioProb:
        dist = ScenarioDistribution.default_instance(scenarios=scenarios, prob=prob)
        return cls(_dist=dist)

    @property
    def scenarios(self) -> DataFrame:
        return self._dist.scenarios

    @property
    def prob(self) -> ProbVector:
        return self._dist.prob

    @property
    def dates(self) -> ProbVector:
        return self._dist.dates

    def add_views(self, new_views: list[View]) -> ScenarioProb:
        """
        Updates ScenarioProb with additional views
        """

        return ScenarioProb(
            _dist=self._dist,
            views=[*self.views, *new_views],
        )

    def clear_views(self) -> ScenarioProb:
        """
        Removes all Views.
        """
        return ScenarioProb(
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

        return ScenarioProb(_dist=new_dist, views=self.views)

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
            _dist=new_dist,
            views=self.views,
        )

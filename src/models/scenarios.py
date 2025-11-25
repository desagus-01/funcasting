from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from polars import DataFrame

from flex_probs.prob_vectors import entropy_pooling_probs, uniform_probs
from models.cma import CopulaMarginalModel
from models.prob import ProbVector
from models.views import View


@dataclass(frozen=True)
class ScenarioDistribution:
    """
    Core representation of a discrete scenario distribution.

    All complex processes (EP, CMA) take a ScenarioDistribution in
    and return a new ScenarioDistribution out.
    """

    scenarios: DataFrame
    prob: ProbVector

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
    def from_scenarios(
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

    def with_views(self, new_views: list[View]) -> ScenarioProb:
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
        self, *, confidence: float = 1.0, include_diags: bool = False
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

        new_dist = ScenarioDistribution(scenarios=self._dist.scenarios, prob=new_prob)

        return ScenarioProb(_dist=new_dist, views=self.views)

    def apply_cma(
        self,
        *,
        target_marginals: dict[str, Literal["t", "norm"]] | None = None,
        target_copula: Literal["t", "norm"] | None = None,
    ) -> ScenarioProb:
        """
        Applies CMA to current scenario distribution using current probs and scenarios.
        Returns a new ScenarioProb with updated scenarios.
        """
        new_dist = CopulaMarginalModel.from_scenario_dist(
            self._dist
        ).update_distribution(
            self._dist, target_marginals=target_marginals, target_copula=target_copula
        )

        return ScenarioProb(
            _dist=new_dist,
            views=self.views,
        )

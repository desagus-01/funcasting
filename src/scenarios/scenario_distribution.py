from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
from polars import DataFrame

from probability.distributions import uniform_probs
from scenarios.types import ProbVector, View


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
        """
        Create a default ScenarioProb with the provided scenarios and an optional prior.

        Parameters
        ----------
        scenarios : DataFrame
            DataFrame of scenarios (rows are scenarios, columns are assets).
        prob : ProbVector | None, optional
            Optional probability vector. If ``None`` a uniform prior is used.

        Returns
        -------
        ScenarioProb
            Initialized ScenarioProb where both _base_dist and _dist point to
            the same default ScenarioDistribution instance.
        """
        dist = ScenarioDistribution.default_instance(scenarios=scenarios, prob=prob)
        return cls(_dist=dist, _base_dist=dist)

    @property
    def assets(self) -> list[str]:
        """
        Return the asset column names present in the current scenario distribution.

        Returns
        -------
        list[str]
            Asset column names.
        """
        return self.scenarios.columns

    @property
    def scenarios(self) -> DataFrame:
        """
        Return the current scenarios DataFrame maintained by this orchestrator.

        Returns
        -------
        DataFrame
            Current scenario matrix (rows=scenarios, cols=assets).
        """
        return self._dist.scenarios

    @property
    def prob(self) -> ProbVector:
        """
        Return the probability vector associated with the current scenarios.

        Returns
        -------
        ProbVector
            Posterior probability vector for the scenarios.
        """
        return self._dist.prob

    @property
    def dates(self) -> DataFrame | None:
        """
        Return the dates DataFrame if present in the current distribution.

        Returns
        -------
        DataFrame | None
            Dates corresponding to the scenario rows or ``None`` if absent.
        """
        if self._dist.dates is not None:
            return self._dist.dates
        else:
            pass

    def add_views(self, new_views: list[View]) -> ScenarioProb:
        """
        Add view objects to the current ScenarioProb instance.

        This method appends the provided list of views to the internal
        ``views`` list and returns ``self`` to allow method chaining.

        Parameters
        ----------
        new_views : list[View]
            Views to append to the orchestrator.

        Returns
        -------
        ScenarioProb
            The same instance with views updated.
        """
        # Append new views to the existing list
        self.views.extend(new_views)
        return self

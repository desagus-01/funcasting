from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
from polars import DataFrame

from globals import ITERS
from methods.cma import CopulaMarginalModel
from methods.ep import entropy_pooling_probs
from models.types import ProbVector, View
from utils.distributions import uniform_probs
from utils.stat_tests import PermTestRes, ind_perm_test, sw_mc


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
            self.scenarios, self.prob, self.dates
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
        self,
        assets: tuple[str, str],
        h_test: bool,
        mc_iter: int = ITERS["MC"],
        tests_iter: int = ITERS["PERM_TEST"],
        original_dist: bool = True,
        rng: np.random.Generator | None = None,
    ) -> float | PermTestRes:
        """
        Compute the Schweizer–Wolff dependence measure between two assets, or
        optionally perform a permutation-based hypothesis test of independence.

        The Schweizer–Wolff (SW) dependence measure quantifies how far the
        empirical copula of two variables deviates from the independence copula.
        Formally,

            σ = 12 ∫ |C(u₁, u₂) − u₁ u₂| du,

        where C is the empirical copula and u₁u₂ is the independence copula.

        A value σ = 0 corresponds to perfect independence. Larger values indicate
        stronger dependence, including nonlinear and non-monotonic forms.

        **Two Modes**
        -------------

        (1) Point Estimate Mode (h_test = False)
            Returns a Monte Carlo estimate σ̂ of the Schweizer–Wolff statistic using
            ``mc_iter`` integration points.

        (2) Hypothesis Testing Mode (h_test = True)
            Performs a non-parametric permutation test of independence by
            repeatedly permuting one pseudo-observation margin and recomputing
            σ̂.

            **Hypotheses:**

            H0 (Null):
                The two assets are independent;
                the joint copula equals the independence copula C(u₁, u₂) = u₁u₂.
                Under H0, σ̂ from the original data should be statistically similar
                to σ̂ computed on permuted data.

            H1 (Alternative):
                The two assets exhibit dependence;
                their copula differs from u₁u₂.
                Under H1, the observed σ̂ is expected to be larger (greater
                absolute deviation from independence) than the permuted values.

            The p-value is the fraction of permuted statistics ≥ the observed σ̂.

        Parameters
        ----------
        assets : tuple[str, str]
            Names of the two assets being evaluated.
        h_test : bool
            Whether to return only σ̂ (False) or perform the full permutation test (True).
        mc_iter : int, default=50_000
            Number of Monte Carlo integration points.
        tests_iter : int, default=10
            Number of permutations used in the hypothesis test.
        original_dist : bool, default=True
            Whether to compute dependence based on the original scenario distribution
            or a post-view/CMA-adjusted one.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        float
            If h_test=False, the Monte Carlo estimate σ̂.
        PermTestRes
            If h_test=True, a dictionary with:
            - ``stat`` : the observed σ̂ statistic,
            - ``p_val`` : the permutation-test p-value.
        """

        if any(asset not in self.assets for asset in assets):
            raise ValueError(
                f"Your two chosen assets {assets} must be in your scenarios; these are {self.assets}"
            )
        if rng is None:
            rng = np.random.default_rng()

        if original_dist:
            dist = self._base_dist
        else:
            dist = self._dist

        cma = CopulaMarginalModel.from_scenario_dist(
            dist.scenarios, dist.prob, dates=self.dates
        )

        cop_assets = cma.copula.select(assets).to_numpy()
        if not h_test:
            return sw_mc(cop_assets, cma.prob, rng=rng, mc_iters=mc_iter)
        else:
            return ind_perm_test(
                pobs=cma.copula,
                p=cma.prob,
                stat_fun=sw_mc,
                assets=assets,
                iter=tests_iter,
                rng=rng,
            )

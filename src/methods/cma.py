from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from numpy import interp
from polars import DataFrame

from maths.distributions import uniform_probs
from maths.sampling import sample_copula, sample_marginal
from models.types import ProbVector
from utils.helpers import compute_cdf_and_pobs


@dataclass
class CopulaMarginalModel:
    marginals: DataFrame
    cdfs: DataFrame
    copula: DataFrame
    prob: ProbVector
    dates: DataFrame | None

    @classmethod
    def from_data_and_prob(
        cls, data: DataFrame, prob: ProbVector | None
    ) -> CopulaMarginalModel:
        if prob is None:
            prob = uniform_probs(data.height)

        if "date" in data:
            dates = data.select("date")
            data = data.drop("date")
        else:
            dates = None

        cdf_cols = {}
        copula_cols = {}
        sorted_marginals = {}

        for col in data.iter_columns():
            asset = col.name
            temp = compute_cdf_and_pobs(data, asset, prob)

            cdf_cols[asset] = temp["cdf"]
            copula_cols[asset] = temp["pobs"]
            sorted_marginals[asset] = temp[asset]

        return CopulaMarginalModel(
            marginals=DataFrame(sorted_marginals),
            cdfs=DataFrame(cdf_cols),
            copula=DataFrame(copula_cols),
            prob=prob,
            dates=dates,
        )

    @classmethod
    def from_scenario_dist(
        cls, scenarios: DataFrame, prob: ProbVector, dates: DataFrame | None
    ) -> CopulaMarginalModel:
        """
        Build CMA model from scenarios and prob distributions
        """
        cdf_cols = {}
        copula_cols = {}
        sorted_marginals = {}

        for col in scenarios.iter_columns():
            asset = col.name
            temp = compute_cdf_and_pobs(scenarios, asset, prob)

            cdf_cols[asset] = temp["cdf"]
            copula_cols[asset] = temp["pobs"]
            sorted_marginals[asset] = temp[asset]

        return CopulaMarginalModel(
            marginals=DataFrame(sorted_marginals),
            cdfs=DataFrame(cdf_cols),
            copula=DataFrame(copula_cols),
            prob=prob,
            dates=dates,
        )

    def to_scenario_dist(self) -> tuple[DataFrame, ProbVector]:
        """
        Converts back to ScenarioDistribution by interpolating marginals.
        """
        interp_res = {}
        for asset in self.marginals.columns:
            interp_res[asset] = interp(
                x=self.copula.select(asset).to_numpy().ravel(),
                xp=self.cdfs.select(asset).to_numpy().ravel(),
                fp=self.marginals.select(asset).to_numpy().ravel(),
            )

        return DataFrame(interp_res), self.prob

    def update_marginals(self, target_dists: dict[str, Literal["t", "norm"]]) -> Self:
        for marginal, target_dist in target_dists.items():
            new_sample = sample_marginal(
                self.marginals, marginals=marginal, kind=target_dist
            )

            cdf = compute_cdf_and_pobs(
                new_sample, marginal, self.prob, compute_pobs=False
            )

            self.marginals = self.marginals.with_columns(
                new_sample[marginal].alias(marginal)
            )
            self.cdfs = self.cdfs.with_columns(cdf["cdf"].alias(marginal))

        return self

    def update_copula(self, target_copula: Literal["t", "norm"] = "t") -> Self:
        self.copula = sample_copula(self.copula, parametric_copula=target_copula)
        return self

    def update_distribution(
        self,
        target_marginals: dict[str, Literal["t", "norm"]] | None = None,
        target_copula: Literal["t", "norm"] | None = None,
    ) -> tuple[DataFrame, ProbVector]:
        cma: CopulaMarginalModel | None = None

        if target_marginals is not None:
            cma = self.update_marginals(target_marginals)

        if target_copula is not None:
            cma = self.update_copula(target_copula)

        if cma is None:
            raise ValueError("You must choose a target marginal or target copula!")

        return cma.to_scenario_dist()

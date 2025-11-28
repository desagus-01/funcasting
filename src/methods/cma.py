from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from numpy import interp
from numpy.typing import NDArray
from polars import DataFrame

from models.types import ProbVector
from utils.distributions import sample_copula, sample_marginal
from utils.helpers import compute_cdf_and_pobs
from utils.stat_tests import sw_mc_u


@dataclass
class CopulaMarginalModel:
    marginals: DataFrame
    cdfs: DataFrame
    copula: DataFrame
    prob: ProbVector

    @classmethod
    def from_scenario_dist(
        cls, scenarios: DataFrame, prob: ProbVector
    ) -> CopulaMarginalModel:
        """
        Build CMA model from a ScenarioDistribution and convert back.
        """
        cdf_cols = {}
        copula_cols = {}
        sorted_marginals = {}

        for col in scenarios.iter_columns():
            name = col.name
            temp = compute_cdf_and_pobs(scenarios, name, prob)

            cdf_cols[name] = temp["cdf"]
            copula_cols[name] = temp["pobs"]
            sorted_marginals[name] = temp[name]

        return CopulaMarginalModel(
            marginals=DataFrame(sorted_marginals),
            cdfs=DataFrame(cdf_cols),
            copula=DataFrame(copula_cols),
            prob=prob,
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
        if target_marginals:
            cma = self.update_marginals(target_marginals)
        if target_copula:
            cma = self.update_copula(target_copula)

        return cma.to_scenario_dist()

    def sw_dependence(self, iter: int = 50) -> dict[NDArray, NDArray]:
        return sw_mc_u(self.copula.to_numpy(), self.prob, iter)

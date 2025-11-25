from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

import polars as pl
from numpy import interp
from polars import DataFrame

from flex_probs.prob_vectors import uniform_probs
from models.prob import ProbVector
from models.scenarios import ScenarioDistribution
from stats.distributions import sample_copula, sample_marginal


@dataclass
class CopulaMarginalModel:
    marginals: DataFrame
    cdfs: DataFrame
    copula: DataFrame
    prob: ProbVector

    @classmethod
    def from_scenario_dist(cls, dist: ScenarioDistribution) -> CopulaMarginalModel:
        """
        Build CMA model from a ScenarioDistribution and convert back.
        """
        cdf_cols = {}
        copula_cols = {}
        sorted_marginals = {}

        for col in dist.scenarios.iter_columns():
            name = col.name
            temp = compute_cdf_and_pobs(dist.scenarios, name, dist.prob)

            cdf_cols[name] = temp["cdf"]
            copula_cols[name] = temp["pobs"]
            sorted_marginals[name] = temp[name]

        return CopulaMarginalModel(
            marginals=DataFrame(sorted_marginals),
            cdfs=DataFrame(cdf_cols),
            copula=DataFrame(copula_cols),
            prob=dist.prob,
        )

    def to_scenario_dist(self) -> ScenarioDistribution:
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

        return ScenarioDistribution(
            scenarios=DataFrame(interp_res),
            prob=self.prob,
        )

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
        dist: ScenarioDistribution,
        *,
        target_marginals: dict[str, Literal["t", "norm"]] | None = None,
        target_copula: Literal["t", "norm"] | None = None,
    ) -> ScenarioDistribution:
        cma = CopulaMarginalModel.from_scenario_dist(dist)

        if target_marginals:
            cma = cma.update_marginals(target_marginals)
        if target_copula:
            cma = cma.update_copula(target_copula)

        return cma.to_scenario_dist()


def compute_cdf_and_pobs(
    data: DataFrame,
    marginal_name: str,
    prob: ProbVector | None = None,  # uniform if none
    compute_pobs: bool = True,
) -> DataFrame:
    if not prob:
        prob = uniform_probs(data.height)

    df = (
        data.select(pl.col(marginal_name))
        .with_row_index()
        .with_columns(prob=prob)
        .sort(marginal_name)
        .with_columns(
            cdf=pl.cum_sum("prob") * data.height / (data.height + 1),
        )
    )

    if compute_pobs:
        df = df.with_columns(
            pobs=pl.col("cdf").gather(pl.col("index").arg_sort()),
        )

    return df

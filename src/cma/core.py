from dataclasses import dataclass
from typing import Literal, Self

from numpy import interp
from polars import DataFrame

from data_types.scenarios import ProbVector, ScenarioProb
from helpers import compute_cdf_and_pobs

from .distributions import sample_copula, sample_marginal


@dataclass
class CMASeparation:
    marginals: DataFrame
    cdfs: DataFrame
    copula: DataFrame
    prob: ProbVector

    def combination(self) -> ScenarioProb:
        interp_res = {}
        for asset in self.marginals.columns:
            interp_res[asset] = interp(
                x=self.copula.select(asset).to_numpy().ravel(),
                xp=self.cdfs.select(asset).to_numpy().ravel(),
                fp=self.marginals.select(asset).to_numpy().ravel(),
            )

        return ScenarioProb(
            type="cma_parametric",
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

    def update_cma_copula(self, target_copula: Literal["t", "norm"] = "t") -> Self:
        self.copula = sample_copula(self.copula, parametric_copula=target_copula)
        return self


def cma_separation(scenario_prob: ScenarioProb) -> CMASeparation:
    cdf_cols = {}
    copula_cols = {}
    sorted_marginals = {}

    for col in scenario_prob.scenarios.iter_columns():
        name = col.name
        temp = compute_cdf_and_pobs(scenario_prob.scenarios, name, scenario_prob.prob)

        cdf_cols[name] = temp["cdf"]
        copula_cols[name] = temp["pobs"]
        sorted_marginals[name] = temp[name]

    return CMASeparation(
        marginals=DataFrame(sorted_marginals),
        cdfs=DataFrame(cdf_cols),
        copula=DataFrame(copula_cols),
        prob=scenario_prob.prob,
    )

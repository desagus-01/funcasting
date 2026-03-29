from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

import polars as pl
from numpy import interp
from polars import DataFrame

from scenarios.types import ProbVector
from probability.distributions import uniform_probs
from probability.sampling import marginal_quantile_mapping, sample_copula
from utils.helpers import compute_cdf_and_pobs


# TODO: Need to find a way to allow nulls at the start
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
            grades = self.copula.select(marginal).to_numpy().ravel()
            sample_values = self.marginals.select(marginal).to_numpy().ravel()

            new_scenarios = marginal_quantile_mapping(
                marginal=sample_values, grades=grades, kind=target_dist
            )

            rebuilt = compute_cdf_and_pobs(
                pl.DataFrame({marginal: new_scenarios.ravel()}),
                marginal,
                self.prob,
                compute_pobs=False,
            )

            self.marginals = self.marginals.with_columns(
                rebuilt[marginal].alias(marginal)
            )
            self.cdfs = self.cdfs.with_columns(rebuilt["cdf"].alias(marginal))

        return self

    def update_copula(
        self,
        seed: int | None = None,
        fit_method: Literal["ml", "irho", "itau"] = "itau",
        target_copula: Literal["t", "norm"] = "t",
    ) -> Self:
        self.copula = sample_copula(
            self.copula,
            seed=seed,
            parametric_copula=target_copula,
            fit_method=fit_method,
        )
        return self

    def update_distribution(
        self,
        seed: int | None = None,
        target_marginals: dict[str, Literal["t", "norm"]] | None = None,
        *,
        target_copula: Literal["t", "norm"] | None = None,
        copula_fit_method: Literal["ml", "irho", "itau"] | None = None,
    ) -> tuple[DataFrame, ProbVector]:
        if target_copula is None and target_marginals is None:
            raise ValueError("Choose a target marginal or target copula!")

        if (target_copula is not None) and (copula_fit_method is not None):
            self.update_copula(
                seed=seed, target_copula=target_copula, fit_method=copula_fit_method
            )

        if target_marginals is not None:
            self.update_marginals(target_marginals)

        return self.to_scenario_dist()

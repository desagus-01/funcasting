from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Self

import polars as pl
from numpy import interp
from polars import DataFrame

from flex_probs.prob_vectors import entropy_pooling_probs, uniform_probs
from models.prob import ProbVector
from models.views import View, ViewBuilder
from stats.distributions import sample_copula, sample_marginal


@dataclass
class ScenarioProb:
    scenarios: DataFrame
    views: list[View] = field(default_factory=list)
    prob: ProbVector | None = None  # defaults to uniform if None

    def __post_init__(self) -> None:
        if self.prob is None:
            self.prob = uniform_probs(self.scenarios.height)

    def build_views(self) -> ViewBuilder:
        return ViewBuilder(self.scenarios, self.views)

    #
    # def add_views(self: list[View]) -> Self:
    #     self.views.extend(new_views)
    #     return self
    #
    def apply_views(
        self, *, confidence: float = 1.0, include_diags: bool = False
    ) -> ScenarioProb:
        if not self.views or not self.prob:
            raise ValueError("Must first have views to apply them")

        ep_res = entropy_pooling_probs(
            prior=self.prob,
            views=self.views,
            confidence=confidence,
            include_diags=include_diags,
        )

        return ScenarioProb(scenarios=self.scenarios, views=self.views, prob=ep_res)

    def to_copula_marginal(self) -> CopulaMarginalModel:
        cdf_cols = {}
        copula_cols = {}
        sorted_marginals = {}

        for col in self.scenarios.iter_columns():
            name = col.name
            temp = compute_cdf_and_pobs(self.scenarios, name, self.prob)

            cdf_cols[name] = temp["cdf"]
            copula_cols[name] = temp["pobs"]
            sorted_marginals[name] = temp[name]

        return CopulaMarginalModel(
            marginals=DataFrame(sorted_marginals),
            cdfs=DataFrame(cdf_cols),
            copula=DataFrame(copula_cols),
            prob=self.prob,
            views=self.views,
        )

    def with_cma(
        self,
        target_marginals: dict[str, Literal["t", "norm"]] | None = None,
        target_copula: Literal["t", "norm"] | None = None,
    ) -> ScenarioProb:
        cma = self.to_copula_marginal()

        if target_marginals:
            cma = cma.update_marginals(target_marginals)
        if target_copula:
            cma = cma.update_copula(target_copula)

        return cma.to_scenario_prob()


@dataclass
class CopulaMarginalModel:
    marginals: DataFrame
    cdfs: DataFrame
    copula: DataFrame
    prob: ProbVector | None  # uniform if None
    views: list[View]

    def to_scenario_prob(self) -> ScenarioProb:
        interp_res = {}
        for asset in self.marginals.columns:
            interp_res[asset] = interp(
                x=self.copula.select(asset).to_numpy().ravel(),
                xp=self.cdfs.select(asset).to_numpy().ravel(),
                fp=self.marginals.select(asset).to_numpy().ravel(),
            )

        return ScenarioProb(
            scenarios=DataFrame(interp_res),
            views=self.views,
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

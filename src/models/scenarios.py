from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Self

import numpy as np
import polars as pl
from numpy import interp
from numpy.typing import NDArray
from polars import DataFrame
from pydantic import AfterValidator, ConfigDict

from stats.distributions import sample_copula, sample_marginal


def _as_prob_vector(a: NDArray[np.float64]) -> NDArray[np.float64]:
    if a.ndim != 1:
        raise ValueError("Array must be 1D.")
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        raise ValueError("Array must not contain NaN or infinite values.")
    if np.any(a < 0):
        raise ValueError("All probabilities must be non-negative.")
    if not np.isclose(a.sum(dtype=np.float64), 1.0, rtol=0, atol=1e-5):
        raise ValueError(
            f"Probabilities must sum to 1. Currently this is {a.sum(dtype=np.float64)}"
        )
    return a


ProbVector = Annotated[NDArray[np.float64], AfterValidator(_as_prob_vector)]


@dataclass
class ScenarioProb:
    type: str
    scenarios: DataFrame
    prob: ProbVector

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
    prob: ProbVector

    def to_scenario_prob(self) -> ScenarioProb:
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

    def update_copula(self, target_copula: Literal["t", "norm"] = "t") -> Self:
        self.copula = sample_copula(self.copula, parametric_copula=target_copula)
        return self


def compute_cdf_and_pobs(
    data: DataFrame,
    marginal_name: str,
    prob: ProbVector,
    compute_pobs: bool = True,
) -> DataFrame:
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


model_cfg = ConfigDict(arbitrary_types_allowed=True)

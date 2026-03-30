from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

import polars as pl
from numpy import interp
from polars import DataFrame

from probability.distributions import uniform_probs
from probability.sampling import marginal_quantile_mapping, sample_copula
from scenarios.types import ProbVector
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
        """
        Construct a CopulaMarginalModel from raw data and a prior probability vector.

        This factory computes per-asset empirical cumulative distribution
        functions (CDFs) and probability integral transforms (pseudo-observations)
        required to build a copula/marginal representation of the joint
        distribution. If ``prob`` is ``None``, a uniform prior is used.

        Parameters
        ----------
        data : DataFrame
            Polars DataFrame with one column per asset. May include a 'date'
            column which will be separated and stored in the resulting model.
        prob : ProbVector | None
            Probability weights for the rows of ``data``. If omitted, rows
            are treated as equally likely.

        Returns
        -------
        CopulaMarginalModel
            Instance containing sorted marginals, their CDFs, pseudo-observations
            (copula matrix) and the probability vector.
        """
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
        Build a CopulaMarginalModel from an existing scenario distribution.

        This method assumes ``scenarios`` contains per-asset scenario values
        (rows are scenarios) and uses the provided probability vector to
        compute empirical CDFs and pseudo-observations.

        Parameters
        ----------
        scenarios : DataFrame
            DataFrame with one column per asset representing scenario values.
        prob : ProbVector
            Probability weights associated with the rows of ``scenarios``.
        dates : DataFrame | None
            Optional dates column corresponding to scenarios.

        Returns
        -------
        CopulaMarginalModel
            Copula/marginal model constructed from the scenario distribution.
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
        Convert the copula/marginal representation back to a scenario distribution.

        The conversion interpolates each marginal by mapping copula pseudo-
        observations through the empirical marginal CDFs to recover scenario
        values corresponding to the current copula.

        Returns
        -------
        tuple[DataFrame, ProbVector]
            A tuple ``(scenarios, prob)`` where ``scenarios`` is a DataFrame of
            reconstructed per-asset scenario values (rows correspond to
            copula rows) and ``prob`` is the associated probability vector.
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
        """
        Replace selected marginals with target families via quantile mapping.

        For each specified marginal the function treats the current copula
        pseudo-observations as grades and maps the empirical marginal quantiles
        to the target distribution (e.g. Student-t or Normal) using
        quantile-matching sampling.

        Parameters
        ----------
        target_dists : dict[str, {"t", "norm"}]
            Mapping of asset name -> target marginal distribution family.

        Returns
        -------
        CopulaMarginalModel
            The modified model with updated marginals and their CDFs.
        """
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
        """
        Re-fit or sample a copula given the current pseudo-observations.

        This function uses the project's copula sampling utility to either
        fit a parametric copula (Student-t or Normal) to the pseudo-
        observations or to resample a copula preserving dependence structure.

        Parameters
        ----------
        seed : int | None, optional
            RNG seed for reproducibility.
        fit_method : {"ml", "irho", "itau"}, optional
            Method to estimate copula parameters.
        target_copula : {"t", "norm"}, optional
            Parametric copula family to fit or sample.

        Returns
        -------
        CopulaMarginalModel
            Self with the updated copula matrix (pseudo-observations).
        """
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
        """
        Apply CMA updates to marginals and/or the copula and return scenarios.

        Parameters
        ----------
        seed : int | None, optional
            RNG seed forwarded to copula resampling if requested.
        target_marginals : dict[str, {"t", "norm"}] | None, optional
            Per-asset target marginal families for quantile mapping.
        target_copula : {"t", "norm"} | None, optional
            Target parametric copula family. If provided, the copula will be
            refit/resampled using ``copula_fit_method``.
        copula_fit_method : {"ml", "irho", "itau"} | None, optional
            Copula fitting method; if omitted defaults may be used by
            underlying utilities.

        Returns
        -------
        tuple[DataFrame, ProbVector]
            Reconstructed scenario DataFrame and the associated probability
            vector after applying updates.

        Raises
        ------
        ValueError
            If neither ``target_copula`` nor ``target_marginals`` is provided.
        """
        if target_copula is None and target_marginals is None:
            raise ValueError("Choose a target marginal or target copula!")

        if (target_copula is not None) and (copula_fit_method is not None):
            self.update_copula(
                seed=seed, target_copula=target_copula, fit_method=copula_fit_method
            )

        if target_marginals is not None:
            self.update_marginals(target_marginals)

        return self.to_scenario_dist()

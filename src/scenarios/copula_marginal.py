from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Self

import polars as pl
from numpy import interp
from polars import DataFrame

from probability.sampling import marginal_quantile_mapping, sample_copula
from scenarios.panel import AssetPanel
from scenarios.types import ProbVector
from utils.helpers import compute_cdf_and_pobs


# TODO: Need to find a way to allow nulls at the start
@dataclass(frozen=True)
class CopulaMarginalModel:
    marginals: DataFrame
    cdfs: DataFrame
    copula: DataFrame
    prob: ProbVector
    dates: pl.Series | None

    @classmethod
    def from_panel(cls, panel: AssetPanel) -> CopulaMarginalModel:
        """Construct a CopulaMarginalModel from an :class:`AssetPanel`.

        The panel must already have its date column separated (handled by
        ``AssetPanel.from_frame``) and nulls dropped before calling this.
        """
        cdf_cols: dict[str, pl.Series] = {}
        copula_cols: dict[str, pl.Series] = {}
        sorted_marginals: dict[str, pl.Series] = {}

        for col in panel.values.iter_columns():
            asset = col.name
            temp = compute_cdf_and_pobs(panel.values, asset, panel.prob)

            cdf_cols[asset] = temp["cdf"]
            copula_cols[asset] = temp["pobs"]
            sorted_marginals[asset] = temp[asset]

        return cls(
            marginals=DataFrame(sorted_marginals),
            cdfs=DataFrame(cdf_cols),
            copula=DataFrame(copula_cols),
            prob=panel.prob,
            dates=panel.dates,
        )

    @classmethod
    def from_data_and_prob(
        cls, data: DataFrame, prob: ProbVector | None = None
    ) -> CopulaMarginalModel:
        """Construct from a raw DataFrame and optional prior.

        Delegates to :meth:`from_panel` after normalising via
        :class:`AssetPanel`.
        """
        return cls.from_panel(AssetPanel.from_frame(data, prob))

    def to_panel(self) -> AssetPanel:
        """Convert back to an :class:`AssetPanel` by interpolating marginals.

        Each marginal is reconstructed by mapping copula pseudo-observations
        through the empirical CDF.
        """
        interp_res = {}
        for asset in self.marginals.columns:
            interp_res[asset] = interp(
                x=self.copula.select(asset).to_numpy().ravel(),
                xp=self.cdfs.select(asset).to_numpy().ravel(),
                fp=self.marginals.select(asset).to_numpy().ravel(),
            )
        return AssetPanel(
            values=DataFrame(interp_res),
            dates=self.dates,
            prob=self.prob,
        )

    def to_scenario_dist(self) -> tuple[DataFrame, ProbVector]:
        """Backward-compatible wrapper around :meth:`to_panel`."""
        panel = self.to_panel()
        return panel.values, panel.prob

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
            New model instance with updated marginals and their CDFs.
        """
        new_marginals = self.marginals
        new_cdfs = self.cdfs

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

            new_marginals = new_marginals.with_columns(
                rebuilt[marginal].alias(marginal)
            )
            new_cdfs = new_cdfs.with_columns(rebuilt["cdf"].alias(marginal))

        return replace(self, marginals=new_marginals, cdfs=new_cdfs)

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
            New model instance with the updated copula matrix (pseudo-observations).
        """
        new_copula = sample_copula(
            self.copula,
            seed=seed,
            parametric_copula=target_copula,
            fit_method=fit_method,
        )
        return replace(self, copula=new_copula)

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

        model = self

        if (target_copula is not None) and (copula_fit_method is not None):
            model = model.update_copula(
                seed=seed, target_copula=target_copula, fit_method=copula_fit_method
            )

        if target_marginals is not None:
            model = model.update_marginals(target_marginals)

        return model.to_scenario_dist()

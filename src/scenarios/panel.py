from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from probability.distributions import uniform_probs
from scenarios.types import ProbVector
from utils.helpers import compensate_prob


def drop_nulls_and_compensate_prob(
    data: pl.DataFrame, prob_vector: ProbVector
) -> tuple[pl.DataFrame, ProbVector]:
    panel = AssetPanel.from_frame(data, prob_vector).drop_nulls()
    return panel.values, panel.prob


@dataclass(frozen=True)
class AssetPanel:
    values: pl.DataFrame  # columns = assets only
    dates: pl.Series | None  # 'date' series or None
    prob: ProbVector

    def __post_init__(self) -> None:
        if "date" in self.values.columns:
            raise ValueError(
                "AssetPanel.values must not contain a 'date' column; "
                "pass it through from_frame() instead."
            )
        if self.values.height != self.prob.shape[0]:
            raise ValueError(
                f"prob length {self.prob.shape[0]} != rows {self.values.height}"
            )
        if self.dates is not None and len(self.dates) != self.values.height:
            raise ValueError(
                f"dates length {len(self.dates)} != rows {self.values.height}"
            )

    @classmethod
    def from_frame(
        cls,
        df: pl.DataFrame,
        prob: ProbVector | None = None,
    ) -> AssetPanel:
        if "date" in df.columns:
            dates: pl.Series | None = df.get_column("date")
            values = df.drop("date")
        else:
            dates = None
            values = df

        if prob is None:
            prob = uniform_probs(values.height)

        return cls(values=values, dates=dates, prob=prob)

    def without_dates(self) -> AssetPanel:
        """Return a copy with dates stripped."""
        return AssetPanel(values=self.values, dates=None, prob=self.prob)

    def drop_nulls(self) -> AssetPanel:
        """Drop leading null rows and compensate the probability vector.

        Nulls are assumed to be at the *start* of the frame (the typical
        pattern after differencing).  The removed probability mass is
        redistributed evenly across the remaining rows.
        """

        total_nulls = self.values.null_count().sum_horizontal().item()
        if total_nulls == 0:
            return self

        clean = self.values.drop_nulls()
        rows_dropped = self.values.height - clean.height
        new_prob = compensate_prob(self.prob, rows_dropped)

        new_dates: pl.Series | None = None
        if self.dates is not None:
            new_dates = self.dates.slice(rows_dropped)

        return AssetPanel(values=clean, dates=new_dates, prob=new_prob)

    @property
    def asset_names(self) -> list[str]:
        return self.values.columns

    @property
    def n_rows(self) -> int:
        return self.values.height

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from probability.distributions import uniform_probs
from scenarios.types import ProbVector, validate_prob_vector


def redistribute_prob_mass(
    prob: ProbVector,
    dropped_idx: NDArray[np.int_],
) -> ProbVector:
    """Drop entries at ``dropped_idx`` and spread their mass across the rest."""
    if dropped_idx.size == 0:
        return prob.copy()

    kept = np.delete(prob, dropped_idx)
    if kept.size == 0:
        raise ValueError("Cannot redistribute: all entries would be removed")

    mass_removed = float(prob[dropped_idx].sum())
    return kept + mass_removed / kept.size


def compensate_prob(prob: ProbVector, n_remove: int) -> ProbVector:
    """Drop the first ``n_remove`` entries and redistribute their mass.

    Convenience wrapper around :func:`redistribute_prob_mass` for the common
    "leading rows were trimmed" case (e.g. after differencing or lag alignment).
    """
    if n_remove < 0:
        raise ValueError("n_remove must be non-negative")
    return redistribute_prob_mass(prob, np.arange(n_remove, dtype=np.int_))


@dataclass(frozen=True)
class AssetPanel:
    """Canonical (values × dates × prob) triple used throughout the pipeline.

    Invariants (enforced in ``__post_init__``):
      * ``values`` has no ``date`` column; dates live in ``dates``.
      * ``values.height == len(prob)`` and ``== len(dates)`` when dates present.
      * ``prob`` is a valid :data:`ProbVector`: 1-D, finite, non-negative,
        summing to 1. Stored as a read-only array.
      * ``dates.name == 'date'`` when present.
      * The panel is non-empty.
    """

    values: pl.DataFrame
    dates: pl.Series | None
    prob: ProbVector

    def __post_init__(self) -> None:
        if "date" in self.values.columns:
            raise ValueError(
                "AssetPanel.values must not contain a 'date' column; "
                "use AssetPanel.from_frame() to separate it."
            )
        if self.values.height == 0:
            raise ValueError("AssetPanel cannot be empty")

        if self.prob.shape[0] != self.values.height:
            raise ValueError(
                f"prob length {self.prob.shape[0]} != rows {self.values.height}"
            )
        validate_prob_vector(self.prob)
        self.prob.setflags(write=False)
        object.__setattr__(self, "prob", self.prob)

        if self.dates is not None:
            if len(self.dates) != self.values.height:
                raise ValueError(
                    f"dates length {len(self.dates)} != rows {self.values.height}"
                )
            if self.dates.name != "date":
                object.__setattr__(self, "dates", self.dates.rename("date"))

    @classmethod
    def from_frame(
        cls,
        df: pl.DataFrame,
        prob: ProbVector | None = None,
    ) -> AssetPanel:
        """Build a panel from a DataFrame, splitting out the ``date`` column.

        A uniform prior is assigned when ``prob`` is not provided.
        """
        if "date" in df.columns:
            dates: pl.Series | None = df.get_column("date")
            values = df.drop("date")
        else:
            dates = None
            values = df

        if prob is None:
            prob = uniform_probs(values.height)

        return cls(values=values, dates=dates, prob=prob)

    def drop_nulls(self) -> AssetPanel:
        """Drop every row that contains a null in any column.

        The probability mass of each dropped row is redistributed evenly
        across the remaining rows (unlike a simple prefix-trim, this is
        correct for nulls that appear anywhere in the panel).
        """
        null_mask = self.values.select(
            pl.any_horizontal(pl.all().is_null())
        ).to_series()

        if not bool(null_mask.any()):
            return self

        keep_mask = ~null_mask
        clean = self.values.filter(keep_mask)
        new_dates = self.dates.filter(keep_mask) if self.dates is not None else None

        dropped_idx = np.flatnonzero(null_mask.to_numpy())
        new_prob = redistribute_prob_mass(self.prob, dropped_idx)

        return AssetPanel(values=clean, dates=new_dates, prob=new_prob)

    def diff(self, lag: int = 1) -> AssetPanel:
        """Apply a ``lag``-step first-difference to every column.

        The leading ``lag`` rows are dropped (they become null) and their
        probability mass is redistributed across the remainder.
        """
        if lag < 1:
            raise ValueError("lag must be >= 1")

        diffed = self.values.with_columns(
            [pl.col(c).diff(lag).alias(c) for c in self.values.columns]
        ).slice(lag)
        new_dates = self.dates.slice(lag) if self.dates is not None else None
        new_prob = compensate_prob(self.prob, lag)
        return AssetPanel(values=diffed, dates=new_dates, prob=new_prob)

    def with_prob(self, prob: ProbVector) -> AssetPanel:
        """Return a new panel with a replaced probability vector."""
        return AssetPanel(values=self.values, dates=self.dates, prob=prob)

    def with_values(
        self,
        values: pl.DataFrame,
        prob: ProbVector | None = None,
    ) -> AssetPanel:
        """Return a new panel with replaced values.

        If the row count matches, ``dates`` and ``prob`` are preserved by
        default (``prob`` can still be overridden). If the row count changes
        a new ``prob`` must be provided and ``dates`` are dropped because
        they no longer align.
        """
        if values.height == self.values.height:
            return AssetPanel(
                values=values,
                dates=self.dates,
                prob=prob if prob is not None else self.prob,
            )
        if prob is None:
            raise ValueError("row count changed; a new prob vector must be supplied")
        return AssetPanel(values=values, dates=None, prob=prob)

    @property
    def asset_names(self) -> list[str]:
        return self.values.columns

    @property
    def n_rows(self) -> int:
        return self.values.height

    def __len__(self) -> int:
        return self.values.height

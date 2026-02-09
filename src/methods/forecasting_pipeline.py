from dataclasses import dataclass
from typing import Literal

import polars as pl
from polars import DataFrame

from maths.time_series.models import AutoGARCHRes
from methods.model_selection_pipeline import (
    MeanModelRes,
    mean_modelling_pipeline,
    volatility_modelling_pipeline,
)
from methods.preprocess_pipeline import run_univariate_preprocess


def _build_innovations_df(
    post: DataFrame,
    mean_map: dict,
    vol_map: dict,
    assets: list[str] | None = None,
) -> DataFrame:
    base = post.select("date")

    if assets is None:
        assets = [c for c in post.columns if c != "date"]

    out = base

    for asset in assets:
        if asset in vol_map:
            resid = vol_map[asset].invariants
            # align residuals to the dates actually used (non-null input series)
            used_dates = post.filter(pl.col(asset).is_not_null()).select("date")
            patch = used_dates.with_columns(pl.Series(asset, resid))

        elif asset in mean_map:
            resid = mean_map[asset].residuals
            used_dates = post.filter(pl.col(asset).is_not_null()).select("date")
            patch = used_dates.with_columns(pl.Series(asset, resid))

        else:
            # "raw" fallback: innovations = diff as random walk
            patch = post.select("date", pl.col(asset).diff().alias(asset))

        out = out.join(patch, on="date", how="left")

    return out


MeanKind = Literal["none", "demean", "arma"]
VolKind = Literal["none", "garch"]

ModelType = Literal[
    "ARMA + GARCH", "ARMA", "GARCH", "Random Walk", "Demean", "Demean + GARCH"
]


_MODEL_TYPE_MAP: dict[tuple[MeanKind, VolKind], ModelType] = {
    ("none", "none"): "Random Walk",
    ("none", "garch"): "GARCH",
    ("demean", "none"): "Demean",
    ("demean", "garch"): "Demean + GARCH",
    ("arma", "none"): "ARMA",
    ("arma", "garch"): "ARMA + GARCH",
}


@dataclass
class UnivariateModel:
    mean_model: MeanModelRes | None
    volatility_model: AutoGARCHRes | None

    @property
    def mean_kind(self) -> MeanKind:
        return "none" if self.mean_model is None else self.mean_model.kind  # type: ignore[return-value]

    @property
    def vol_kind(self) -> VolKind:
        return "none" if self.volatility_model is None else self.volatility_model.kind

    @property
    def model_type(self) -> ModelType:
        return _MODEL_TYPE_MAP[(self.mean_kind, self.vol_kind)]


def build_best_univariate_model(
    data: DataFrame, assets_to_model: list[str]
) -> dict[str, UnivariateModel]:
    mean_modelling = mean_modelling_pipeline(data=data, assets=assets_to_model)
    volatility_modelling = volatility_modelling_pipeline(mean_model_res=mean_modelling)
    asset_model = {}
    for asset in (
        data.columns
    ):  # we want for all assets (as some will be RW) so run in original columns
        if asset != "date":
            asset_model[asset] = UnivariateModel(
                mean_model=mean_modelling.get(asset),
                volatility_model=volatility_modelling.get(asset),
            )

    return asset_model


def info_for_forecasting(data: DataFrame, assets: list[str] | None = None):
    post_process = run_univariate_preprocess(data=data, assets=assets)
    chosen_model = build_best_univariate_model(
        data=post_process.post_data,
        assets_to_model=post_process.needs_further_modelling,
    )
    # innovs = build_innovations_df(
    #     post=post_process.post_data,
    #     mean_map=chosen_model,
    #     vol_map=volatility_modelling,
    # )
    #
    return post_process.post_data, chosen_model

from dataclasses import dataclass
from functools import reduce

import polars as pl
from polars import DataFrame

from methods.model_selection_pipeline import (
    UnivariateRes,
    build_best_univariate_model,
)
from methods.preprocess_pipeline import run_univariate_preprocess


@dataclass
class MultivariateForecastInfo:
    risk_drivers: DataFrame
    model_res: dict[str, UnivariateRes]
    invariants: DataFrame


def _build_innovations_df_from_models(
    post: DataFrame, model_map: dict[str, UnivariateRes], assets=None
) -> DataFrame:
    if assets is None:
        assets = [c for c in post.columns if c != "date"]

    base = post.select("date")
    patches: list[pl.DataFrame] = []

    for asset in assets:
        model = model_map[asset]
        used_dates = post.filter(pl.col(asset).is_not_null()).select("date")

        non_null_values = post.select(asset).drop_nulls().to_numpy().ravel()
        invariant = model.invariant(non_null_values)

        patch = used_dates.with_columns(pl.Series(asset, invariant))

        patches.append(patch)

    return reduce(lambda acc, p: acc.join(p, on="date", how="left"), patches, base)


def multivariate_forecasting_info(
    data: DataFrame, assets: list[str] | None = None
) -> MultivariateForecastInfo:
    post_process = run_univariate_preprocess(data=data, assets=assets)
    chosen_models = build_best_univariate_model(
        data=post_process.post_data,
        assets_to_model=post_process.needs_further_modelling,
    )
    invariants = _build_innovations_df_from_models(
        post=post_process.post_data,
        model_map=chosen_models,
    )

    return MultivariateForecastInfo(post_process.post_data, chosen_models, invariants)

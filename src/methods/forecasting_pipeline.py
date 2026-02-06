import polars as pl
from polars import DataFrame

from methods.model_selection_pipeline import (
    mean_modelling_pipeline,
    volatility_modelling_pipeline,
)
from methods.preprocess_pipeline import run_univariate_preprocess


# TODO: Finish writing the Forecast model below, to include conditional mean and conditional vol
class ForecastModel:
    def __init__(
        self,
    ):
        pass


def build_innovations_df(
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
            resid = vol_map[asset].residuals
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


def fit_best_univariate_model(data: DataFrame, assets: list[str] | None = None):
    post_process = run_univariate_preprocess(data=data, assets=assets)
    mean_modelling = mean_modelling_pipeline(
        data=post_process.post_data, assets=post_process.needs_further_modelling
    )
    volatility_modelling = volatility_modelling_pipeline(mean_model_res=mean_modelling)
    innovs = build_innovations_df(
        post=post_process.post_data,
        mean_map=mean_modelling,
        vol_map=volatility_modelling,
    )

    return post_process, mean_modelling, volatility_modelling, innovs

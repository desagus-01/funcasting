# %%
import polars as pl
import polars.selectors as cs

from methods.preprocess_pipeline import (
    check_white_noise,
    deseason_pipeline,
    detrend_pipeline,
)
from utils.helpers import get_assets_names
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)


# %%
def make_increments(data: pl.DataFrame, drop_nulls: bool = True) -> pl.DataFrame:
    increments = data.select(cs.numeric().diff())

    if drop_nulls:
        return increments.drop_nulls()
    return increments


def increments_not_white_noise(data: pl.DataFrame):
    increments_df = make_increments(data)
    first_wn_check = check_white_noise(data=increments_df)
    needs_preprocess = [
        asset for asset, is_white_noise in first_wn_check.items() if not is_white_noise
    ]

    return needs_preprocess


# TODO: Takes way too long at white noise, need to review copula test
def univariate_preprocess_pipeline(data: pl.DataFrame, assets: list[str] | None = None):
    pipeline_decisions = {}
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)
    level_df = data.select(assets)

    assets_need_preprocess = increments_not_white_noise(data=level_df)
    detrend = detrend_pipeline(
        data=level_df, assets=assets_need_preprocess, include_diagnostics=False
    )
    pipeline_decisions["trend"] = detrend.decision
    detrend_df = detrend.updated_data.drop_nulls()
    print(detrend_df)
    deseason = deseason_pipeline(
        data=detrend_df, assets=assets_need_preprocess, include_diagnostics=False
    )
    pipeline_decisions["deseason"] = deseason.decision

    return (
        deseason.updated_data,
        pipeline_decisions,
        increments_not_white_noise(deseason.updated_data),
    )


# %%
univariate_preprocess_pipeline(data=data)

# %%
data
# %%
# transformed_data = res.updated_data.drop_nulls()
#
# check_white_noise(data=make_increments(transformed_data))
#
# # %%
# deseason_pipeline(data=transformed_data, include_diagnostics=False)

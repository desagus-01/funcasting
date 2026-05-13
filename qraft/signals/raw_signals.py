# %%
import logging

import polars as pl
from polars import DataFrame, Series
from policy import LogConfig
from utils.log import setup_logging

setup_logging(LogConfig(level=logging.WARNING))


# Momentum
def ewma_signal(
    risk_drivers: DataFrame,
    dates: Series,
    half_life: str = "60d",
    drop_nulls: bool = True,
) -> DataFrame:
    ewm_df = risk_drivers.with_columns(
        pl.all().diff().ewm_mean_by(dates, half_life=half_life)
    )

    return ewm_df.drop_nulls() if drop_nulls else ewm_df

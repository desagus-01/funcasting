import datetime
import os
from dataclasses import dataclass

import databento as db
import numpy as np
import polars as pl
import polars.selectors as cs
from dotenv import load_dotenv
from polars import DataFrame
from pydantic import BaseModel, ConfigDict

from utils.distributions import uniform_probs


class Assets(BaseModel):
    """
    Base Class for all Assets, we include the risk driver which is deterministic, and increments which if test are passed can be considered invariants.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    raw_data: pl.DataFrame
    risk_drivers: pl.DataFrame
    increments: pl.DataFrame


def get_db_sample(tickers: list[str]) -> pl.DataFrame:
    if not os.path.exists("data/databento_ohlc.csv"):
        _ = load_dotenv()
        db_client = db.Historical(os.getenv("DATABENTO_API"))

        data = db_client.timeseries.get_range(
            dataset="XNAS.ITCH",
            start="2021-01-01",
            end="2025-01-01",
            symbols=tickers,
            stype_in="raw_symbol",
            schema="ohlcv-1d",
        )

        data.to_csv("data/databento_ohlc.csv")

    return pl.read_csv(
        "data/databento_ohlc.csv", schema_overrides={"ts_event": datetime.datetime}
    )


def simp_df(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.cast({"ts_event": datetime.date})
        .select(["ts_event", "symbol", "close"])
        .rename({"ts_event": "date"})
        .pivot("symbol", values="close")
    )


def get_rd_inc(raw_df: pl.DataFrame) -> Assets:
    risk_drivers = raw_df.select(pl.col.date, cs.numeric().log())

    increments = risk_drivers.select(pl.col.date, cs.numeric().diff()).drop_nulls()

    return Assets(raw_data=raw_df, risk_drivers=risk_drivers, increments=increments)


def get_example_assets(tickers: list[str]) -> Assets:
    raw_data = get_db_sample(tickers)
    return get_rd_inc(simp_df(raw_data))


@dataclass
class TestTemplateResult:
    tickers: list[str]
    raw_data: DataFrame
    increms_df_long: DataFrame
    increms_df: DataFrame
    increms_np: np.ndarray
    increms_n: int
    uniform_prior: np.ndarray


def get_template():
    tickers = ["AAPL", "MSFT", "GOOG"]
    assets = get_example_assets(tickers)
    increms_df = assets.increments
    increms_df_long = assets.increments.unpivot(
        on=tickers, value_name="return", variable_name="ticker", index="date"
    )
    increms_np = increms_df.to_numpy()
    increms_n = increms_df.height
    uniform_prior = uniform_probs(increms_n)

    return TestTemplateResult(
        tickers=tickers,
        raw_data=assets.raw_data,
        increms_df_long=increms_df_long,
        increms_df=increms_df,
        increms_np=increms_np,
        increms_n=increms_n,
        uniform_prior=uniform_prior,
    )

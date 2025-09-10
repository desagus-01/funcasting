import datetime
import os

import databento as db
import polars as pl
import polars.selectors as cs
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

# NOTE: Only for Equity for now.


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

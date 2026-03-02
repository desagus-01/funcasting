# %%
import logging
import os
from typing import Any, Literal

import polars as pl
import requests
from dotenv import load_dotenv
from polars import DataFrame
from requests import Session
from requests.exceptions import HTTPError

_ = load_dotenv()
session = requests.Session()

BASE_URL = "https://api.tiingo.com"

TIINGO_API = os.getenv("TIINGO_API")


# %%
def _request(
    session: Session,
    method: Literal["GET", "POST"],
    url: str,
    headers: dict[str, str] | None = None,
    **kwargs,
):
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)

    resp = session.request(method, url, headers=merged_headers, **kwargs)
    try:
        resp.raise_for_status()
    except HTTPError:
        logging.error("HTTP %s: %s", resp.status_code, resp.text)
        raise
    return resp


def _get_url(ticker: str) -> str:
    return f"{BASE_URL}/tiingo/daily/{ticker}/prices"


def get_single_price(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    columns: str | None = None,
    frequency: str = "daily",
) -> DataFrame:
    url = _get_url(ticker)

    params: dict[str, Any] = {
        "token": TIINGO_API,
        "format": "json",
        "resampleFreq": frequency,
    }
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    if columns:
        params["columns"] = columns

    data: list[dict[str, Any]] = _request(session, "GET", url, params=params).json()
    return DataFrame(data)


def get_ticker_prices(
    tickers: list[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    columns: str | None = None,
    frequency: str = "daily",
    return_clean: bool = True,
):
    dfs = []
    for t in tickers:
        out = get_single_price(
            t,
            start_date=start_date,
            end_date=end_date,
            columns=columns,
            frequency=frequency,
        ).with_columns(pl.lit(t).alias("ticker"))

        dfs.append(out)

    df = pl.concat(dfs, how="vertical_relaxed")

    df = df.with_columns(
        pl.col("date")
        .str.to_datetime(time_zone="UTC", strict=False)
        .dt.date()
        .alias("date")
    )

    if return_clean:
        df = df.select(
            [
                "date",
                "ticker",
                "close",
                pl.col("adjClose").alias("adj_close"),
                pl.col("adjVolume").alias("adj_volume"),
            ]
        )

    return df


# %%
# example
df = get_ticker_prices(start_date="2026-01-01", tickers=["AAPL", "MSFT"])

# %%
df.columns

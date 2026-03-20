# %%
import logging
import os
from datetime import date, timedelta
from typing import Any, Literal

import matplotlib.pyplot as plt
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


def plot_ticker_lines(
    df: pl.DataFrame,
    date_column: str = "date",
    value_column: str = "adj_close",
    ticker_column: str = "ticker",
):
    tickers = df.select(pl.col(ticker_column).unique()).to_series().to_list()
    figs_axes = []

    for ticker in tickers:
        ticker_df = df.filter(pl.col(ticker_column) == ticker).sort(date_column)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            ticker_df[date_column].to_list(),
            ticker_df[value_column].to_list(),
            linewidth=1.5,
        )

        ax.set_title(f"{ticker}")
        ax.set_ylabel(value_column)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        figs_axes.append((ticker, fig, ax))

    return figs_axes


# %%
df = get_ticker_prices(start_date="2010-01-01", tickers=["AAPL", "MSFT", "GOOG"])

# %%
df = df.with_columns(lg_adj_close=pl.col("adj_close").log())

plot_ticker_lines(df, value_column="lg_adj_close")

# %%
tiingo_tickers = pl.read_csv("./data/supported_tickers.csv", try_parse_dates=True)

# %%


def previous_business_day(d: date) -> date:
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


target_enddate = previous_business_day(date.today())

nas_uni = tiingo_tickers.filter(
    (pl.col("priceCurrency") == "USD")
    & (pl.col("exchange") == "NASDAQ")
    & (pl.col("assetType") == "Stock")
    & pl.col("startDate").is_not_null()
    & (pl.col("endDate") == pl.lit(target_enddate))
)

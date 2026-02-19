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

from maths.distributions import uniform_probs
from maths.time_series.diagnostics.seasonality import SEASONAL_MAP


class Assets(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    raw_data: pl.DataFrame
    risk_drivers: pl.DataFrame
    increments: pl.DataFrame


def get_db_sample(tickers: list[str] | None) -> pl.DataFrame:
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
    asset_info: Assets
    increms_df_long: DataFrame
    increms_df: DataFrame
    increms_np: np.ndarray
    increms_n: int
    uniform_prior: np.ndarray


def synthetic_series(n: int) -> np.ndarray:
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(20250112)

    comps = {
        "weekly": (0.0030, SEASONAL_MAP["weekly"], 0.35),
        "monthly": (0.0022, SEASONAL_MAP["monthly"], -1.10),
        "quarterly": (0.0018, SEASONAL_MAP["quarterly"], 2.00),
        "semi-annual": (0.0015, SEASONAL_MAP["semi-annual"], 0.75),
        "annual": (0.0012, SEASONAL_MAP["annual"], -2.20),
    }

    mu = np.zeros(n, dtype=float)
    for amp, period, phase in comps.values():
        mu += amp * np.sin(2.0 * np.pi * t / period + phase)

    df = 6.0
    z = rng.standard_t(df, size=n)
    z /= np.sqrt(df / (df - 2.0))  # normalize to unit variance (df>2)

    omega = 1e-6
    alpha = 0.08
    beta = 0.90
    sigma2 = np.empty(n, dtype=float)
    eps = np.empty(n, dtype=float)

    sigma2[0] = 1e-4
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    for i in range(1, n):
        sigma2[i] = omega + alpha * (eps[i - 1] ** 2) + beta * sigma2[i - 1]
        eps[i] = np.sqrt(sigma2[i]) * z[i]

    phi = 0.10
    r = np.empty(n, dtype=float)
    r[0] = mu[0] + eps[0]
    for i in range(1, n):
        r[i] = mu[i] + phi * (r[i - 1] - mu[i - 1]) + eps[i]

    jump_prob = 0.005
    jumps = (rng.random(n) < jump_prob) * rng.normal(0.0, 0.03, size=n)
    r += jumps

    return r


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
        asset_info=assets,
        increms_df_long=increms_df_long,
        increms_df=increms_df,
        increms_np=increms_np,
        increms_n=increms_n,
        uniform_prior=uniform_prior,
    )

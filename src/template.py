from dataclasses import dataclass

import numpy as np
from polars import DataFrame

from flex_probs.prob_vectors import uniform_probs
from get_data import get_example_assets


@dataclass
class TestTemplateResult:
    tickers: list[str]
    increms_df_long: DataFrame
    increms_df: DataFrame
    increms_np: np.ndarray
    increms_n: int
    uniform_prior: np.ndarray


def test_template():
    tickers = ["AAPL", "MSFT", "GOOG"]
    assets = get_example_assets(tickers)
    increms_df = assets.increments.drop("date")
    increms_df_long = assets.increments.unpivot(
        on=tickers, value_name="return", variable_name="ticker", index="date"
    )
    increms_np = increms_df.to_numpy()
    increms_n = increms_df.height
    uniform_prior = uniform_probs(increms_n)

    return TestTemplateResult(
        tickers=tickers,
        increms_df_long=increms_df_long,
        increms_df=increms_df,
        increms_np=increms_np,
        increms_n=increms_n,
        uniform_prior=uniform_prior,
    )

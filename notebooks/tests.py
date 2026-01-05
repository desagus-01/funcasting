# %% imports
from pprint import pprint as print

import polars as pl

from maths.helpers import add_differenced_columns
from maths.stochastic_processes import (
    HypTestRes,
)
from maths.stochastic_processes.stationarity_tests import stationarity_tests
from utils.template import get_template


# %% small helpers
def show_results(title: str, results: dict[str, HypTestRes]) -> None:
    print(f"\n=== {title} ===")
    for name, res in results.items():
        print(f"\n[{name}]")
        print(res)


# %% load once
info_all = get_template()


risk_drivers = info_all.asset_info.risk_drivers


def test_stochastic_trend(data: pl.DataFrame, assets: list[str], difference: int = 3):
    difference_df = add_differenced_columns(
        data=data, assets=assets, difference=difference
    )
    stationary_res = []
    for asset in assets:
        for diff in range(1, difference):
            selected_column = f"{asset}_diff_{diff}"
            stat_test = stationarity_tests(
                data=difference_df.select(selected_column).drop_nulls(),
                asset=selected_column,
            )
            if stat_test.label == "stationary":
                print(f"{asset} diff stopped at {diff}")
                stationary_res.append(stat_test)
                break
            else:
                continue
    return stationary_res


print(test_stochastic_trend(data=risk_drivers, assets=["AAPL"]))

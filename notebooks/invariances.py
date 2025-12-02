from typing import Any

from models.scenarios import ScenarioProb
from utils.helpers import lag_df
from utils.stat_tests import PermTestRes
from utils.template import get_template

info_all = get_template()
increms = info_all.increms_df

assets = [c for c in increms.columns if c != "date"]

# loosened the inner dict type a bit to allow the extra list entry
sw_lag_res: dict[str, dict[str, Any]] = {}

for asset in assets:
    df = lag_df(increms, asset, 2)
    scenario = ScenarioProb.default_inst(scenarios=df)

    cols = scenario.scenarios.columns
    lag_pairs = list(zip(cols, cols[1:]))

    # lag_1, lag_2, ...
    results: dict[str, PermTestRes] = {
        f"lag_{i + 1}": scenario.schweizer_wolff(assets=lag_t, h_test=True)
        for i, lag_t in enumerate(lag_pairs)
    }

    # extra summary: which lags rejected the null
    rejected_lags = [
        lag_label for lag_label, res in results.items() if res["reject_null"]
    ]

    # combine: per asset you now get all lag results + the summary list
    sw_lag_res[asset] = {
        **results,
        "rejected_lags": rejected_lags,
    }

print(sw_lag_res)

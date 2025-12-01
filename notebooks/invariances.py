from typing import Callable

import numpy as np
import polars as pl

from methods.cma import CopulaMarginalModel
from models.scenarios import ScenarioProb
from models.types import ProbVector
from utils.stat_tests import sw_mc
from utils.template import get_template

info_all = get_template()
scenarios = ScenarioProb.default_inst(info_all.increms_df)

cop_m = CopulaMarginalModel.from_scenario_dist(scenarios.scenarios, scenarios.prob)


StatFunc = Callable[[np.ndarray, ProbVector], float]


def perm_test(
    pobs: pl.DataFrame,
    p: ProbVector,
    stat_fun: StatFunc,
    assets: tuple[str, str],
    iter: int = 10,
) -> tuple[float, float]:
    # fix a seed
    rng = np.random.default_rng(42)  # seed for reproducibility
    # select assets from df and convert to numpy + select an asset to be the one to be permm
    assets_np = pobs.select(assets).to_numpy()
    perm_asset = assets_np[:, 0]

    stat = stat_fun(assets_np, p)

    null_dist = np.empty(iter, dtype=float)

    for i in range(iter):
        new_order = rng.permutation(assets_np.shape[0])
        new_p_asset = perm_asset[new_order]

        temp_df = pobs.select(assets).with_columns(pl.lit(new_p_asset).alias(assets[0]))

        null_dist[i] = stat_fun(temp_df.to_numpy(), p)

    p_val = (1.0 + (null_dist >= stat).sum()) / (iter + 1.0)

    return stat, p_val


a = perm_test(cop_m.copula, cop_m.prob, sw_mc, ("GOOG", "MSFT"))

print(a)

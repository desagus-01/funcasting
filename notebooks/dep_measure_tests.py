import polars as pl

from methods.cma import CopulaMarginalModel
from models.scenarios import ScenarioProb
from models.views_builder import ViewBuilder
from utils.stat_tests import sw_mc_u
from utils.template import get_template

info_all = get_template()

scenarios = ScenarioProb.default_inst(info_all.increms_df)

views = (
    ViewBuilder(scenarios.scenarios)
    .mean({"AAPL": 0.001, "GOOG": -0.03}, ["equal", "equal"])
    .build()
)

scenarios = scenarios.add_views(views).apply_views()

cma = CopulaMarginalModel.from_scenario_dist(scenarios.scenarios, scenarios.prob)

# ex_aapl = cma.copula.drop("GOOG").to_numpy()

ex_aapl = (
    cma.copula.select(pl.col("AAPL")).with_columns(aapl_2=pl.col("AAPL")).to_numpy()
)


print(sw_mc_u(ex_aapl, cma.prob))

import numpy as np

from models.scenarios import CopulaMarginalModel, ScenarioProb
from models.types import ProbVector
from models.views_builder import ViewBuilder
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

ex_aapl = cma.copula.drop("AAPL").to_numpy()


def eval_cop(pobs: np.ndarray, p: ProbVector, point: np.ndarray):
    less_eq_coord = pobs <= point
    inside_lower_orthant = np.all(less_eq_coord, axis=1)

    return p @ inside_lower_orthant


def sw_int(pobs: np.ndarray, p: ProbVector, point: np.ndarray):
    est_cop = eval_cop(pobs, p, point)
    ind_cop = float(np.prod(point))
    return abs(est_cop - ind_cop)


def sw_mc(pobs: np.ndarray, p: ProbVector, iter: int = 10_000):
    rng = np.random.default_rng()
    uni_draws = rng.uniform(0.0, 1.0, size=(iter, pobs.shape[1]))

    res = np.empty(iter, dtype=float)
    for i in range(iter):
        res[i] = sw_int(pobs, p, uni_draws[i])
    return 12 * res.mean()


test = np.empty(50)
for i in range(50):
    print(f"Iteration {i}")
    test[i] = sw_mc(ex_aapl, cma.prob)

print(test, test.mean())

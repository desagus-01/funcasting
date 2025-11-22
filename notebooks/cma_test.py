from cma.core import cma_separation
from data_types.scenarios import ScenarioProb
from flex_probs.prob_vectors import uniform_probs
from template import test_template

info = test_template()
prob = uniform_probs(info.increms_df.height)

scenario_ex = ScenarioProb("x", scenarios=info.increms_df, prob=prob)
cma = cma_separation(scenario_ex)

new_joint = cma.update_marginals({"AAPL": "norm"}).update_cma_copula("t").combination()

print(scenario_ex)

print(new_joint)

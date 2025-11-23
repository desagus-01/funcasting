from flex_probs.prob_vectors import uniform_probs
from models.scenarios import ScenarioProb
from template import test_template

info = test_template()
prob = uniform_probs(info.increms_df.height)

scenario_ex = ScenarioProb("x", scenarios=info.increms_df, prob=prob)

a = scenario_ex.to_copula_marginal()

scenario_ex.with_cma(target_marginals={"AAPL": "t"}, target_copula="norm")

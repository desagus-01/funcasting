import numpy as np

from models.scenarios import ScenarioProb
from utils.template import get_template

rng = np.random.default_rng(42)
info_all = get_template()
scenarios = ScenarioProb.default_inst(info_all.increms_df)

print(scenarios.schweizer_wolff(("AAPL", "MSFT"), h_test=False, rng=rng))

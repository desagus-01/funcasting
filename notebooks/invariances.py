import numpy as np

from models.scenarios import ScenarioProb
from utils.helpers import lag_df
from utils.template import get_template

rng = np.random.default_rng(42)
info_all = get_template()

increms_lag = lag_df(info_all.raw_data, "AAPL", 5)
scenarios = ScenarioProb.default_inst(increms_lag)

print(scenarios.schweizer_wolff(("AAPL", "AAPL_lag_5"), h_test=True, rng=rng))

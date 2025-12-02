import time

import numpy as np

from models.scenarios import ScenarioProb
from utils.helpers import lag_df
from utils.template import get_template


def add_units(value, unit):
    return f"{value:.6f} {unit}" if value < 1 else f"{value:.3f} {unit}s"


rng = np.random.default_rng(42)
info_all = get_template()

increms_lag = lag_df(info_all.raw_data, "AAPL", 1)
scenarios = ScenarioProb.default_inst(increms_lag)

# Warm-up
_ = scenarios.schweizer_wolff(("AAPL", "AAPL_lag_1"), h_test=True, rng=rng)

# Run N times
N = 10
times = []

for _ in range(N):
    t0 = time.perf_counter()
    _ = scenarios.schweizer_wolff(("AAPL", "AAPL_lag_1"), h_test=True, rng=rng)
    t1 = time.perf_counter()
    times.append(t1 - t0)

print("Average:", add_units(np.mean(times), "second"))
print("Std dev:", add_units(np.std(times), "second"))
print("Min:", add_units(np.min(times), "second"))
print("Max:", add_units(np.max(times), "second"))

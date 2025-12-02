from models.scenarios import ScenarioProb
from utils.helpers import lag_df
from utils.template import get_template

info_all = get_template()
increms = info_all.increms_df

lag_scenarios: dict[str, ScenarioProb] = {}
for asset in increms.columns:
    if asset == "date":
        continue
    df = lag_df(increms, asset, 5)
    lag_scenarios[asset] = ScenarioProb.default_inst(scenarios=df)


lag_tuples: dict[str, list[tuple[str, str]]] = {}

for asset in lag_scenarios.keys():
    tuples: list[tuple[str, str]] = []

    cols = lag_scenarios[asset].scenarios.columns
    for i in range(len(cols) - 1):
        tuples.append((cols[i], cols[i + 1]))

    lag_tuples[asset] = tuples


# lag_scenarios["MSFT"].schweizer_wolff(lag_tuples["MSFT"][0], h_test=True)

sw_lag_res: dict[str, list[object]] = {}

for asset, scenario in lag_scenarios.items():
    lags_to_run = lag_tuples[asset]
    results = []

    for lag_t in lags_to_run:
        print(f"Running {asset} {lag_t}")
        results.append(scenario.schweizer_wolff(assets=lag_t, h_test=True))

    sw_lag_res[asset] = results

print(sw_lag_res)

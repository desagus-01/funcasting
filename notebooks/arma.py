# %%
import time

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic

from maths.time_series.models import get_start_and_max_orders
from methods.model_selection_pipeline import assets_need_mean_modelling, auto_arma
from methods.preprocess_pipeline import run_univariate_preprocess
from utils.template import get_template, synthetic_series

# %%

data = get_template().asset_info.risk_drivers
series = synthetic_series(data.height)
data = data.with_columns(fake=series)

# %%
data_2 = run_univariate_preprocess(data=data)

x = assets_need_mean_modelling(data_2.post_data, data_2.needs_further_modelling)

start, end = get_start_and_max_orders(data_2.post_data.height)


# %%

start_time = time.time()
array = data_2.post_data.select("fake").to_numpy().ravel()
auto_arma(array)
print("--- %s seconds ---" % (time.time() - start_time))

# %%
start_time = time.time()
assets_model: dict[str, dict[tuple[int, int, int], float]] = {}
for asset in data_2.needs_further_modelling:
    criterias = {}
    array = data_2.post_data.select(asset).to_numpy().ravel()
    for p in range(start, end + 1):
        for q in range(start, end + 1):
            res = ARIMA(
                endog=array,
                order=(p, 0, q),
                trend="n",
                enforce_invertibility=False,
                enforce_stationarity=False,
            ).fit(method="hannan_rissanen", low_memory=True)
            pvals = res.pvalues
            if np.any(pvals[~np.isnan(pvals)] > 0.1):
                continue
            # residual = res.resid
            criteria = float(res.bic)
            criterias[(p, 0, q)] = criteria
    assets_model[f"{asset}"] = criterias

assets_model

for asset, res in assets_model.items():
    print(asset)
    best_model, best_criteria = min(res.items(), key=lambda x: x[1])
    print(best_model)
    print(best_criteria)

print("--- %s seconds ---" % (time.time() - start_time))
# %%

start_time = time.time()
K = 8
best = []
array = data_2.post_data.select("fake").to_numpy().ravel()
for p in range(start, end + 1):
    for q in range(start, end + 1):
        res = ARIMA(
            array,
            order=(p, 0, q),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(method="hannan_rissanen", low_memory=True)
        best.append(((p, 0, q), float(res.bic)))

best = sorted(best, key=lambda x: x[1])[:K]

final = {}
for order, _ in best:
    res = ARIMA(
        array,
        order=order,
        trend="n",
    ).fit(method="statespace", low_memory=True)
    pvals = res.pvalues
    if np.any(pvals[~np.isnan(pvals)] > 0.1):
        continue
    final[order] = float(res.bic)
print("--- %s seconds ---" % (time.time() - start_time))
final


# %%
start_time = time.time()
best_order = arma_order_select_ic(
    y=array,
    max_ar=3,
    max_ma=3,
    ic="bic",
    trend="n",
    model_kw={"enforce_stationarity": False, "enforce_invertibility": False},
    fit_kw={"method": "hannan_rissanen", "low_memory": True},
)["bic"]

top3 = best_order.stack().nsmallest(3)

top3_orders = [(int(p), 0, int(q)) for (p, q) in top3.index]

best = []
for order in top3_orders:
    res = ARIMA(array, order=order, trend="n").fit(method="statespace")
    best.append((order, float(res.bic)))
best = sorted(best, key=lambda x: x[1])[:1]

print(best)
print("--- %s seconds ---" % (time.time() - start_time))


# %%

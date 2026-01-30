from heapq import nsmallest

import numpy as np
from numpy._typing import NDArray
from polars import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
from typing_extensions import Literal

from maths.time_series.iid_tests import ljung_box_test


def assets_need_mean_modelling(data: DataFrame, assets_to_test: list[str]) -> list[str]:
    needs_mean_modelling = []
    for asset in assets_to_test:
        lj = ljung_box_test(data=data, asset=asset)
        if len(lj.rejected) != 0:
            needs_mean_modelling.append(asset)
    return needs_mean_modelling


def auto_arma(
    asset_array: NDArray[np.floating],
    max_ar_order: int = 3,
    max_ma_order: int = 3,
    information_criteria: Literal["bic", "aic"] = "bic",
    top_n_models: int = 3,
    output_best_n: int = 1,
):
    # Get top n model orders with lowest information criteria
    best_order = arma_order_select_ic(
        y=asset_array,
        max_ar=max_ar_order,
        max_ma=max_ma_order,
        ic=information_criteria,
        trend="n",
        model_kw={"enforce_stationarity": False, "enforce_invertibility": False},
        fit_kw={"method": "hannan_rissanen", "low_memory": True},
    )["bic"]

    top = best_order.stack().nsmallest(top_n_models)
    candidates_for_arma = [
        (int(ar_order), 0, int(ma_order)) for (ar_order, ma_order) in top.index
    ]

    # Run for those top n, get back best performing model
    best = []
    for candidate_order in candidates_for_arma:
        res = ARIMA(asset_array, order=candidate_order, trend="n").fit(
            method="statespace"
        )
        best.append((candidate_order, float(res.bic)))
    return nsmallest(output_best_n, best, key=lambda x: x[1])

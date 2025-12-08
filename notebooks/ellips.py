from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame

from models.types import ProbVector
from utils.distributions import uniform_probs
from utils.helpers import build_lag_df
from utils.template import get_template

info_all = get_template()
increms = info_all.increms_df


class MeanCovRes(TypedDict):
    assets: list[str]
    means: NDArray[np.floating]
    cov: NDArray[np.floating]


def sample_meancov(data: DataFrame, p: ProbVector | None = None) -> MeanCovRes:
    if p is None:
        p = uniform_probs(data.height)

    assets = data.columns
    data_np = data.to_numpy()

    weighted_mean = p @ data_np
    weighted_cov = ((data_np - weighted_mean).T * p) @ (data_np - weighted_mean)

    return {"assets": assets, "means": weighted_mean, "cov": weighted_cov}


aapl_lag = build_lag_df(increms, "AAPL", 5)

print(aapl_lag, sample_meancov(aapl_lag))

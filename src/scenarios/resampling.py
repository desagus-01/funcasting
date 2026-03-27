import numpy as np
from numpy import random
from numpy._typing import NDArray
from polars import DataFrame

from models.types import ProbVector


def weighted_bootstrapping_idx(
    data: DataFrame,
    prob_vector: ProbVector,
    n_samples: int,
    seed: int | None = None,
) -> NDArray[np.int64]:
    if data.height != len(prob_vector):
        raise ValueError(
            f"Data size {data.height} and probability vector size {len(prob_vector)} do not match."
        )
    rng = random.default_rng(seed)
    return rng.choice(data.height, size=n_samples, replace=True, p=prob_vector)


def weighted_bootstrapping(
    data: DataFrame,
    prob_vector: ProbVector,
    n_samples: int,
    seed: int | None = None,
) -> DataFrame:
    sample_row_n = weighted_bootstrapping_idx(
        data=data, prob_vector=prob_vector, n_samples=n_samples, seed=seed
    )
    return data[sample_row_n]

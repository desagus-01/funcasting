from typing import Literal

import numpy as np
import polars as pl
from copulae import NormalCopula, StudentCopula
from numpy import random
from numpy._typing import NDArray
from polars import DataFrame
from scipy.stats import norm, t

from models.types import ProbVector


def sample_marginal(
    data: pl.DataFrame, marginals: str, kind: Literal["t", "norm"] = "t"
) -> pl.DataFrame:
    values = data.select(marginals).to_numpy().flatten()

    if kind == "t":
        params = t.fit(values)
        frozen = t(*params)
    elif kind == "norm":
        params = norm.fit(values)
        frozen = norm(*params)
    else:
        raise ValueError(f"Unknown distribution kind: {kind}")

    return pl.DataFrame({marginals: frozen.rvs(size=data.height)})


def marginal_quantile_mapping(
    marginal: NDArray[np.floating],
    grades: NDArray[np.floating],
    kind: Literal["t", "norm"] = "t",
) -> NDArray[np.floating]:
    if kind == "t":
        params = t.fit(marginal)
        frozen = t(*params)
    elif kind == "norm":
        params = norm.fit(marginal)
        frozen = norm(*params)
    else:
        raise ValueError(f"Unknown distribution kind: {kind}")
    return frozen.ppf(grades)


def sample_copula(
    copula: pl.DataFrame,
    seed: int | None,
    parametric_copula: Literal["t", "norm"] = "t",
    fit_method: Literal["ml", "irho", "itau"] = "itau",
    to_pobs: bool = False,
) -> pl.DataFrame:
    values = copula.to_numpy()
    col_names = copula.columns
    if parametric_copula == "t":
        cop = StudentCopula(values.shape[1])
    elif parametric_copula == "norm":
        cop = NormalCopula(values.shape[1])
    else:
        raise ValueError("You must choose either t or norm")

    samples = cop.fit(values, method=fit_method, to_pobs=to_pobs).random(
        n=copula.height, seed=seed
    )
    return pl.DataFrame(samples, col_names)


def weighted_bootstrapping(
    data: DataFrame, prob_vector: ProbVector, n_samples: int, seed: int | None = None
) -> DataFrame:
    if data.height != len(prob_vector):
        raise ValueError(
            f"Data size {data.height} and probability vector size {len(prob_vector)} do not match."
        )
    rng = random.default_rng(seed)
    sample_row_n = rng.choice(data.height, size=n_samples, replace=True, p=prob_vector)

    return (
        data.with_row_index()
        .filter(pl.col("index").is_in(sample_row_n.tolist()))
        .drop("index")
    )

from typing import Literal

import polars as pl
from copulae import NormalCopula, StudentCopula
from scipy.stats import norm, t


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


def sample_copula(
    copula: pl.DataFrame, parametric_copula: Literal["t", "norm"] = "t"
) -> pl.DataFrame:
    values = copula.to_numpy()
    col_names = copula.columns

    if parametric_copula == "t":
        cop = StudentCopula(values.shape[1])
    elif parametric_copula == "norm":
        cop = NormalCopula(values.shape[1])

    _ = cop.fit(values, to_pobs=False)
    samples = cop.random(n=copula.height)
    return pl.DataFrame(samples, col_names)

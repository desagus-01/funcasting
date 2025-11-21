from typing import Literal

import polars as pl
from copulae import NormalCopula, StudentCopula
from scipy.stats import norm, t

from cma.operations import compute_cdf_and_pobs
from data_types.vectors import CMASeparation


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


def update_cma_marginal(
    cma_data: CMASeparation, marginal: str, prob_dist: Literal["t", "norm"] = "t"
):
    new_sample = sample_marginal(cma_data.marginals, marginals=marginal, kind=prob_dist)

    cdf = compute_cdf_and_pobs(
        new_sample, marginal, cma_data.posterior, compute_pobs=False
    )

    return CMASeparation(
        marginals=cma_data.marginals.with_columns(
            (new_sample[marginal]).alias(marginal)
        ),
        cdfs=cma_data.cdfs.with_columns((cdf["cdf"]).alias(marginal)),
        copula=cma_data.copula,
        posterior=cma_data.posterior,
    )


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


def update_cma_copula(
    cma_data: CMASeparation, parametric_copula: Literal["t", "norm"] = "t"
):
    return CMASeparation(
        marginals=cma_data.marginals,
        cdfs=cma_data.cdfs,
        copula=sample_copula(cma_data.copula, parametric_copula=parametric_copula),
        posterior=cma_data.posterior,
    )

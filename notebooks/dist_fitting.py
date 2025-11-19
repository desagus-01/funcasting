from typing import Literal

import polars as pl
from scipy.stats import norm, t

from cma.operations import _compute_cdf_and_pobs, cma_separation
from data_types.vectors import CMASeparation
from flex_probs.prob_vectors import uniform_probs
from template import test_template

info = test_template()

aapl_df = info.increms_df.select(pl.col.AAPL)

prob = uniform_probs(aapl_df.height)

multi_df = info.increms_df

seps = cma_separation(multi_df, prob)


def sample_dist(
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

    samples = frozen.rvs(size=data.height)
    return pl.DataFrame({marginals: samples})


test = sample_dist(info.increms_df, "AAPL", "t")

x = _compute_cdf_and_pobs(test, "AAPL", prob, compute_pobs=False)


def update_cma_marginal(
    cma_data: CMASeparation, marginal: str, prob_dist: Literal["t", "norm"] = "t"
):
    new_sample = sample_dist(cma_data.marginals, marginals=marginal, kind=prob_dist)

    cdf = _compute_cdf_and_pobs(
        new_sample, marginal, cma_data.posterior, compute_pobs=False
    )

    return CMASeparation(
        marginals=cma_data.marginals.with_columns(
            (new_sample["marginals"]).alias(marginal)
        ),
        cdfs=cma_data.cdfs.with_columns((cdf["pobs"]).alias(marginal)),
        copula=cma_data.copula,
        posterior=cma_data.posterior,
    )

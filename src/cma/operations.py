import polars as pl
from numpy import interp

from data_types.vectors import CMASeparation, ProbVector


def _compute_cdf_and_pobs(
    data: pl.DataFrame, marginal_name: str, prob: ProbVector
) -> pl.DataFrame:
    norm_const = data.height / (data.height + 1)

    return (
        data.select(pl.col(marginal_name))
        .with_row_index()
        .with_columns(prob=prob)
        .sort(marginal_name)
        .with_columns(
            cdf=pl.cum_sum("prob") * norm_const,
        )
        .with_columns(
            pobs=pl.col("cdf").gather(pl.col("index").arg_sort()),
        )
    )


def cma_separation(data: pl.DataFrame, prob: ProbVector) -> CMASeparation:
    cdf_cols = {}
    copula_cols = {}
    sorted_marginals = {}

    for col in data.iter_columns():
        name = col.name
        temp = _compute_cdf_and_pobs(data, name, prob)

        cdf_cols[name] = temp["cdf"]  #
        copula_cols[name] = temp["pobs"]
        sorted_marginals[name] = temp[name]

    return CMASeparation(
        marginals=pl.DataFrame(sorted_marginals),
        cdfs=pl.DataFrame(cdf_cols),
        copula=pl.DataFrame(copula_cols),
        posterior=prob,
    )


def cma_combination(cma_separation: CMASeparation) -> pl.DataFrame:
    interp_res = {}
    for asset in cma_separation.marginals.columns:
        interp_res[asset] = interp(
            x=cma_separation.copula.select(asset).to_numpy().ravel(),
            xp=cma_separation.cdfs.select(asset).to_numpy().ravel(),
            fp=cma_separation.marginals.select(asset).to_numpy().ravel(),
        )

    return pl.DataFrame(interp_res)

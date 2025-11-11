import polars as pl

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

        cdf_cols[f"{name}_cdf"] = temp["cdf"]
        copula_cols[f"{name}_pobs"] = temp["pobs"]
        sorted_marginals[f"{name}_sorted"] = temp[name]

    return CMASeparation(
        pl.DataFrame(sorted_marginals),
        pl.DataFrame(cdf_cols),
        pl.DataFrame(copula_cols),
    )

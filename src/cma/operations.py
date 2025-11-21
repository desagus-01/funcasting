import polars as pl
from numpy import interp

from data_types.vectors import CMASeparation, ProbVector, ScenarioProb


def compute_cdf_and_pobs(
    data: pl.DataFrame,
    marginal_name: str,
    prob: ProbVector,
    compute_pobs: bool = True,
) -> pl.DataFrame:
    df = (
        data.select(pl.col(marginal_name))
        .with_row_index()
        .with_columns(prob=prob)
        .sort(marginal_name)
        .with_columns(
            cdf=pl.cum_sum("prob") * data.height / (data.height + 1),
        )
    )

    if compute_pobs:
        df = df.with_columns(
            pobs=pl.col("cdf").gather(pl.col("index").arg_sort()),
        )

    return df


def cma_separation(data: pl.DataFrame, prob: ProbVector) -> CMASeparation:
    cdf_cols = {}
    copula_cols = {}
    sorted_marginals = {}

    for col in data.iter_columns():
        name = col.name
        temp = compute_cdf_and_pobs(data, name, prob)

        cdf_cols[name] = temp["cdf"]
        copula_cols[name] = temp["pobs"]
        sorted_marginals[name] = temp[name]

    return CMASeparation(
        marginals=pl.DataFrame(sorted_marginals),
        cdfs=pl.DataFrame(cdf_cols),
        copula=pl.DataFrame(copula_cols),
        posterior=prob,
    )


def cma_combination(cma_separation: CMASeparation) -> ScenarioProb:
    interp_res = {}
    for asset in cma_separation.marginals.columns:
        interp_res[asset] = interp(
            x=cma_separation.copula.select(asset).to_numpy().ravel(),
            xp=cma_separation.cdfs.select(asset).to_numpy().ravel(),
            fp=cma_separation.marginals.select(asset).to_numpy().ravel(),
        )

    return ScenarioProb(
        type="parametric",
        scenarios=pl.DataFrame(interp_res),
        prob=cma_separation.posterior,
    )

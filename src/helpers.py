import polars as pl

from data_types.vectors import CorrInfo, View
from globals import sign_operations


def select_operator(views: View):
    return sign_operations[views.sign_type]


def get_corr_info(data: pl.DataFrame) -> list[CorrInfo]:
    corr_df = (
        data.corr()
        .with_columns(index=pl.lit(pl.Series(data.columns)))
        .unpivot(index="index")
        .filter(pl.col("index") != pl.col("variable"))
        .with_columns(
            pair=pl.when(pl.col("index") < pl.col("variable"))
            .then(pl.concat_str([pl.col("index"), pl.col("variable")], separator="-"))
            .otherwise(
                pl.concat_str([pl.col("variable"), pl.col("index")], separator="-")
            )
        )
        .unique(subset=["pair"])
        .drop("pair")
    )

    return [
        CorrInfo(asset_pair=(corrs["index"], corrs["variable"]), corr=corrs["value"])
        for corrs in corr_df.rows(named=True)
    ]

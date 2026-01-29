import numpy as np
import polars as pl
from polars.dataframe.frame import DataFrame

from maths.time_series.estimation import OLSEquation, OLSResults, ols_classic


def build_harmonic_regression_equation(
    data: DataFrame,
    frequency_radians: list[float],
    asset: str,
) -> OLSEquation:
    dependent_variable = data.select(pl.col(asset)).to_numpy()

    # time index
    time_index_df = data.select(pl.col(asset)).with_row_index(name="t")
    cos_cols = [
        (pl.lit(w) * pl.col("t")).cos().alias(f"cos_w_{i}")
        for i, w in enumerate(frequency_radians)
        if not np.isclose(w, 0.0)
    ]
    sin_cols = [
        (pl.lit(w) * pl.col("t")).sin().alias(f"sin_w_{i}")
        for i, w in enumerate(frequency_radians)
        if not np.isclose(w, 0.0)
    ]

    independent_variables = time_index_df.select(cos_cols + sin_cols).to_numpy()

    return OLSEquation(ind_var=independent_variables, dep_vars=dependent_variable)


def run_harmonic_regression(
    data: DataFrame, asset: str, frequency_radians: list[float]
) -> OLSResults:
    harmonic_equation = build_harmonic_regression_equation(
        data=data, asset=asset, frequency_radians=frequency_radians
    )
    return ols_classic(
        dependent_var=harmonic_equation.dep_vars,
        independent_vars=harmonic_equation.ind_var,
    )


def deterministic_deseasoning(
    data: DataFrame, asset: str, frequency_radians: list[float]
) -> dict[str, list[float]]:
    harmonic_residuals = run_harmonic_regression(
        data=data, asset=asset, frequency_radians=frequency_radians
    ).residuals.ravel()

    return {asset: harmonic_residuals.tolist()}

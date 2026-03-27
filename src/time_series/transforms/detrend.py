from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy._typing import NDArray

TrendTypes = Literal["polynomial", "difference"]


@dataclass(frozen=True)
class TrendCandidate:
    asset: str
    transform_type: TrendTypes
    order: int
    column_name: str


@dataclass(frozen=True)
class CandidateBatch:
    data: pl.DataFrame
    candidates_by_asset: dict[str, list[TrendCandidate]]


def polynomial_detrend(
    data: NDArray[np.floating],
    polynomial_order: int = 1,
    axis: int = 0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if data.ndim > 2:
        raise NotImplementedError("data.ndim > 2 is not implemented.")

    transposed = False

    if data.ndim == 2 and axis == 1:
        data = data.T
        transposed = True

    if polynomial_order < 0:
        raise ValueError("polynomial_order must be >= 0")

    if polynomial_order == 0:
        beta = data.mean(axis=0)
        fitted_trend = np.broadcast_to(beta, data.shape)
    else:
        time = np.arange(data.shape[0], dtype=float)
        trends = np.vander(time, N=polynomial_order + 1)
        beta = np.linalg.pinv(trends) @ data
        fitted_trend = trends @ beta

    resid = data - fitted_trend

    if transposed:
        resid = resid.T

    return resid, beta


def add_detrend_column(
    original_data: pl.DataFrame,
    assets: list[str] | None = None,
    polynomial_orders: list[int] | None = None,
) -> tuple[pl.DataFrame, dict[int, dict[str, NDArray[np.floating]]]]:
    if assets is None:
        assets = original_data.select(cs.numeric()).columns

    if polynomial_orders is None:
        polynomial_orders = [0, 1, 2, 3]

    asset_arrays = original_data.select(assets).to_numpy()

    new_cols: list[pl.Series] = []
    betas_by_order: dict[int, dict[str, NDArray[np.floating]]] = {}

    for p in polynomial_orders:
        resid, beta = polynomial_detrend(
            asset_arrays,
            polynomial_order=p,
            axis=0,
        )

        if p == 0:
            betas_by_order[p] = {
                asset: np.array([beta[i]], dtype=float).reshape(-1)
                for i, asset in enumerate(assets)
            }
        else:
            betas_by_order[p] = {
                asset: np.asarray(beta[:, i], dtype=float).reshape(-1)
                for i, asset in enumerate(assets)
            }

        for i, asset in enumerate(assets):
            new_cols.append(
                pl.Series(
                    name=f"{asset}_detrended_p_{p}",
                    values=resid[:, i],
                ).cast(pl.Float64)
            )

    return original_data.with_columns(new_cols), betas_by_order


def add_detrend_columns_max(
    data: pl.DataFrame,
    assets: list[str],
    max_polynomial_order: int,
) -> pl.DataFrame:
    polynomial_orders = list(range(0, max_polynomial_order + 1))
    return add_detrend_column(
        original_data=data, assets=assets, polynomial_orders=polynomial_orders
    )[0]


def add_differenced_columns(
    data: pl.DataFrame,
    assets: list[str],
    difference: int = 1,
    keep_all: bool = True,
) -> pl.DataFrame:
    if difference < 1:
        raise ValueError("difference must be >= 1")

    diffs = range(1, difference + 1) if keep_all else [difference]

    return data.with_columns(
        [pl.col(assets).diff(d).name.suffix(f"_diff_{d}") for d in diffs]
    )


def build_polynomial_candidates(
    data: pl.DataFrame, assets: list[str], max_order: int = 3
) -> CandidateBatch:
    if max_order < 0:
        raise ValueError("max order must be >= 0, duh")

    polynomial_orders = list(range(0, max_order + 1))

    transformed_df, betas_by_order = add_detrend_column(
        original_data=data, assets=assets, polynomial_orders=polynomial_orders
    )

    candidates_by_asset = {}

    for asset in assets:
        candidates = []

        for order in polynomial_orders:
            column_name = f"{asset}_detrended_p_{order}"

            candidates.append(
                TrendCandidate(
                    asset=asset,
                    transform_type="polynomial",
                    order=order,
                    column_name=column_name,
                )
            )

        candidates_by_asset[asset] = candidates

    return CandidateBatch(data=transformed_df, candidates_by_asset=candidates_by_asset)


def build_difference_candidates(
    data: pl.DataFrame, assets: list[str], max_order: int = 3, drop_nulls: bool = True
) -> CandidateBatch:
    if max_order < 0:
        raise ValueError("max order must be >= 0, duh")

    transformed_df = add_differenced_columns(
        data=data, assets=assets, difference=max_order
    )

    candidates_by_asset = {}

    for asset in assets:
        candidates = []

        for order in range(1, max_order + 1):
            column_name = f"{asset}_diff_{order}"

            candidates.append(
                TrendCandidate(
                    asset=asset,
                    transform_type="difference",
                    order=order,
                    column_name=column_name,
                )
            )

        candidates_by_asset[asset] = candidates

    return CandidateBatch(
        data=transformed_df.drop_nulls() if drop_nulls else transformed_df,
        candidates_by_asset=candidates_by_asset,
    )

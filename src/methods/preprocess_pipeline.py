import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy._typing import NDArray
from polars.dataframe.frame import DataFrame

from globals import LAGS
from methods.cma import CopulaMarginalModel
from models.types import ProbVector
from time_series.diagnostics.seasonality import (
    SEASONAL_MAP,
    SeasonalityPeriodTest,
    seasonality_diagnostic,
)
from time_series.diagnostics.trend import TrendTest, trend_diagnostic
from time_series.tests.iid import (
    TestResultByAsset,
    copula_lag_independence_test,
    ellipsoid_lag_test,
    univariate_kolmogrov_smirnov_test,
)
from time_series.transforms.deseason import (
    HarmonicTerm,
    deterministic_seasonal_adjustment,
)
from time_series.transforms.detrend import add_detrend_column, add_differenced_columns
from utils.helpers import (
    compensate_prob,
    get_assets_names,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformDecision:
    kind: Literal["none", "polynomial", "difference", "seasonality"]
    order: int | None = None


@dataclass(frozen=True)
class PolynomialInverseSpec:
    order: int
    betas: NDArray[np.floating]

    def inverse_for_forecasts(
        self,
        data: NDArray[np.floating],
        start_x: int,
    ) -> NDArray[np.floating]:
        current = np.asarray(data, dtype=float)

        if current.ndim != 2:
            raise ValueError(
                f"PolynomialInverseSpec expects 2D forecasts of shape "
                f"(n_sims, horizon), got shape {current.shape}"
            )

        polynomials = np.asarray(self.betas, dtype=float).reshape(-1)
        horizon = current.shape[1]
        future_x = np.arange(start_x, start_x + horizon)
        trend = np.polyval(polynomials, future_x)

        return current + trend[None, :]


@dataclass(frozen=True)
class DifferenceInverseSpec:
    order: int
    initial_values: NDArray[np.floating]

    def inverse_for_forecasts(
        self,
        data: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        current = np.asarray(data, dtype=float)

        if current.ndim != 2:
            raise ValueError(
                f"DifferenceInverseSpec expects 2D forecasts of shape "
                f"(n_sims, horizon), got shape {current.shape}"
            )

        init = np.asarray(self.initial_values, dtype=float).reshape(-1)

        if init.shape[0] != self.order:
            raise ValueError(
                f"Expected {self.order} initial values for differencing order "
                f"{self.order}, got {init.shape[0]}"
            )

        for anchor in init[::-1]:
            current = np.cumsum(current, axis=1) + anchor

        return current


@dataclass(frozen=True)
class SeasonalInverseSpec:
    terms: list[HarmonicTerm]

    @staticmethod
    def evaluate_seasonal_terms(
        terms: list[HarmonicTerm],
        time: NDArray[np.int_],
    ) -> NDArray[np.floating]:
        time = np.asarray(time).reshape(-1)
        seasonal = np.zeros_like(time, dtype=float)

        for term in terms:
            if term.kind == "cos":
                if term.omega is None:
                    raise ValueError("Cos must have omega")
                seasonal += term.coefficient * np.cos(term.omega * time)
            elif term.kind == "sin":
                if term.omega is None:
                    raise ValueError("Sin must have omega")
                seasonal += term.coefficient * np.sin(term.omega * time)

        return seasonal

    def inverse_for_forecasts(
        self,
        data: NDArray[np.floating],
        n_train: int,
    ) -> NDArray[np.floating]:
        current = np.asarray(data, dtype=float)

        if current.ndim != 2:
            raise ValueError(
                f"SeasonalInverseSpec expects 2D forecasts of shape "
                f"(n_sims, horizon), got shape {current.shape}"
            )

        horizon = current.shape[1]
        future_t = np.arange(n_train, n_train + horizon, dtype=float)
        seasonal_future = self.evaluate_seasonal_terms(
            terms=self.terms,
            time=future_t,
        )

        return current + seasonal_future[None, :]


InverseSpec = PolynomialInverseSpec | DifferenceInverseSpec | SeasonalInverseSpec


@dataclass(frozen=True)
class AppliedTransform:
    asset: str
    decision: TransformDecision
    inverse_spec: InverseSpec | None = None


@dataclass(frozen=True)
class PipelineAssetBatchRes:
    type: Literal["trend", "seasonality"]
    decision: dict
    inverse_spec: dict[str, InverseSpec] | None
    updated_data: DataFrame
    all_tests: dict | None


@dataclass(frozen=True)
class UnivariatePreprocess:
    post_data: DataFrame
    inverse_specs: dict[str, list[AppliedTransform]]
    needs_further_modelling: list[str]


def _run_iid_simple(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    lags: int = LAGS["simple"],
) -> dict[str, TestResultByAsset]:
    """Run the cheaper IID screens."""
    ellipsoid_test = ellipsoid_lag_test(
        data=data,
        prob=prob,
        lags=lags,
        assets=assets,
    )
    ks_test = univariate_kolmogrov_smirnov_test(
        data=data,
        assets=assets,
    )
    return {
        "ellipsoid": ellipsoid_test,
        "ks": ks_test,
    }


def _run_iid_complex(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    lags: int = LAGS["complex"],
) -> dict[str, TestResultByAsset]:
    """Run the expensive copula IID screen."""
    copula_marginal_model = CopulaMarginalModel.from_data_and_prob(
        data=data.select(assets),
        prob=prob,
    )

    copula_lag_test_res = copula_lag_independence_test(
        copula=copula_marginal_model.copula,
        prob=copula_marginal_model.prob,
        lags=lags,
        assets=assets,
    )

    return {
        "copula": copula_lag_test_res,
    }


def check_white_noise(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    lags_simple: int = LAGS["simple"],
    lags_complex: int = LAGS["complex"],
) -> dict[str, bool]:
    """Return whether each asset passes the white-noise screen."""

    simple_tests = _run_iid_simple(
        data=data,
        prob=prob,
        assets=assets,
        lags=lags_simple,
    )

    simple_pass = {
        asset: not any(
            simple_tests[test_name][asset].rejected for test_name in ("ellipsoid", "ks")
        )
        for asset in assets
    }

    assets_for_copula = [asset for asset, passed in simple_pass.items() if passed]

    logger.info(
        "White-noise simple screen: passed_simple=%s, sent_to_complex=%s",
        assets_for_copula,
        assets_for_copula,
    )

    if not assets_for_copula:
        logger.info("White-noise final screen: passed_all=[]")
        return simple_pass

    complex_tests = _run_iid_complex(
        data=data,
        prob=prob,
        assets=assets_for_copula,
        lags=lags_complex,
    )

    copula_pass = {
        asset: not complex_tests["copula"][asset].rejected
        for asset in assets_for_copula
    }

    final_pass = simple_pass.copy()
    for asset in assets_for_copula:
        final_pass[asset] = final_pass[asset] and copula_pass[asset]

    logger.info(
        "White-noise final screen: passed_all=%s",
        [asset for asset, passed in final_pass.items() if passed],
    )

    return final_pass


def _detrend_decision_rule(
    detrend_res: dict[str, dict[str, TrendTest]],
    assets: list[str],
    tie_break: Literal["polynomial", "difference"] = "difference",
) -> dict[str, TransformDecision]:
    """Choose the lowest-order winning trend transform per asset.

    Strategy:
        Prefer the smallest order; tie-break
    """
    trend_trans = {}
    for asset in assets:
        deterministic_res = detrend_res["deterministic"][
            asset
        ].transformation_order_needed
        stochastic_res = detrend_res["stochastic"][asset].transformation_order_needed

        candidates = []

        if deterministic_res is not None:
            candidates.append(("polynomial", deterministic_res))

        if stochastic_res is not None:
            candidates.append(("difference", stochastic_res))

        if not candidates:
            continue

        transformation, order = min(candidates, key=lambda x: (x[1], x[0] != tie_break))

        trend_trans[asset] = TransformDecision(kind=transformation, order=order)

    return trend_trans


def _group_detrend_assets(
    decision: dict[str, TransformDecision],
) -> dict[tuple[str, int], list[str]]:
    """Group assets by (transform kind, order)."""
    by_group: dict[tuple[str, int], list[str]] = {}

    for asset, dec in decision.items():
        if dec.order is None:
            continue
        by_group.setdefault((dec.kind, dec.order), []).append(asset)

    return by_group


def _apply_grouped_detrend(
    data: DataFrame,
    by_group: dict[tuple[str, int], list[str]],
) -> tuple[DataFrame, dict[str, InverseSpec]]:
    """Apply each grouped transform to the data and retain inverse specs."""
    inverse_specs: dict[str, InverseSpec] = {}

    for (transform, order), assets in by_group.items():
        if transform == "difference":
            # store anchors from the series BEFORE differencing
            anchors_by_asset = {
                asset: data.select(asset).to_series().tail(order).to_numpy()
                for asset in assets
            }

            data = add_differenced_columns(
                data=data,
                assets=assets,
                difference=order,
                keep_all=True,
            )

            for asset in assets:
                inverse_specs[asset] = DifferenceInverseSpec(
                    order=order,
                    initial_values=np.asarray(anchors_by_asset[asset], dtype=float),
                )
        elif transform == "polynomial":
            data, betas_by_order = add_detrend_column(
                original_data=data,
                assets=assets,
                polynomial_orders=[order],
            )

            beta = betas_by_order[order]

            for asset in assets:
                inverse_specs[asset] = PolynomialInverseSpec(
                    order=order, betas=np.asarray(beta[asset], dtype=float)
                )

        else:
            raise ValueError(f"Unknown transform '{transform}' for assets {assets}")

    return data, inverse_specs


def _select_and_rename_detrended_columns(
    data: DataFrame,
    decision: dict[str, TransformDecision],
) -> DataFrame:
    """Keep only date + transformed asset columns, renamed back to asset names."""
    transformed_cols: list[str] = []
    rename_map: dict[str, str] = {}

    for asset, dec in decision.items():
        if dec.order is None:
            continue

        if dec.kind == "difference":
            col = f"{asset}_diff_{dec.order}"
        elif dec.kind == "polynomial":
            col = f"{asset}_detrended_p{dec.order}"
        else:
            raise ValueError(f"Unknown transform '{dec.kind}' for asset '{asset}'")

        if col not in data.columns:
            raise ValueError(
                f"Expected transformed column '{col}' not found. "
                f"Available columns: {data.columns}"
            )

        transformed_cols.append(col)
        rename_map[col] = asset

    return data.select(["date", *transformed_cols]).rename(rename_map)


def _apply_detrend(
    data: DataFrame,
    decision: dict[str, TransformDecision],
) -> tuple[DataFrame, dict[str, InverseSpec]]:
    """Apply detrending decisions and return date + transformed asset columns."""
    if not decision:
        return data.select(["date"]), {}

    by_group = _group_detrend_assets(decision)
    if not by_group:
        return data.select(["date"]), {}

    transformed, inverse_specs = _apply_grouped_detrend(data, by_group)
    selected = _select_and_rename_detrended_columns(transformed, decision)
    return selected, inverse_specs


def _expand_period_to_harmonics(period_label: str) -> list[tuple[str, float]]:
    """
    Given a period label ('weekly', 'monthly', ...), return a list of
    (label_for_harmonic, omega_radians) covering the full harmonic set.
    The label is augmented (e.g., 'monthly_h2'); _deseason_apply ignores labels anyway.
    """
    P = SEASONAL_MAP[period_label]
    base = 2 * np.pi / P

    out: list[tuple[str, float]] = []
    H = (P - 1) // 2
    for h in range(1, H + 1):
        out.append((f"{period_label}_h{h}", h * base))

    if P % 2 == 0:
        out.append((f"{period_label}_nyq", np.pi))  # Nyquist cosine

    return out


def _deseason_decision_rule(
    seasonality_diagnostic: dict[str, list[SeasonalityPeriodTest]],
) -> dict[str, list[tuple[str, float]]]:
    """
    Extract significant seasonal periods and expand each to its harmonic set.
    """
    decision: dict[str, list[tuple[str, float]]] = {}

    for asset, tests in seasonality_diagnostic.items():
        freqs: list[tuple[str, float]] = []
        for t in tests:
            if t.evidence_of_seasonality:
                freqs.extend(_expand_period_to_harmonics(t.seasonal_period))

        #  deduplicate angular frequencies if multiple periods collide
        if freqs:
            freqs_sorted = sorted(freqs, key=lambda kv: kv[1])
            dedup: list[tuple[str, float]] = []
            for lab, w in freqs_sorted:
                if not dedup or not np.isclose(w, dedup[-1][1], rtol=0, atol=1e-12):
                    dedup.append((lab, w))
            decision[asset] = dedup
        else:
            decision[asset] = []

    return decision


def _deseason_apply(
    data: pl.DataFrame,
    decision: dict[str, list[tuple[str, float]]],
) -> tuple[pl.DataFrame, dict[str, InverseSpec]]:
    assets = [asset for asset, seasons in decision.items() if seasons]
    if not assets:
        return data.select(["date"]), {}

    out = data.select(["date"])
    inverse_specs: dict[str, InverseSpec] = {}

    for asset in assets:
        seasons = decision[asset]
        omega = [rad for _, rad in seasons]

        seas_adj_res = deterministic_seasonal_adjustment(
            data=data,
            asset=asset,
            frequency_radians=omega,
        )

        out = out.join(seas_adj_res.residuals, on="date", how="left")
        inverse_specs[asset] = SeasonalInverseSpec(terms=seas_adj_res.terms)

    return out, inverse_specs


def deseason_pipeline(
    data: DataFrame,
    assets: list[str] | None = None,
    include_diagnostics: bool = False,
) -> PipelineAssetBatchRes:
    """Diagnose, decide, and apply deterministic seasonal adjustment."""
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    diagnostics = seasonality_diagnostic(
        data=data,
        assets=assets,
    )

    decision = _deseason_decision_rule(diagnostics)

    updated, inverse_specs = _deseason_apply(
        data=data,
        decision=decision,
    )

    return PipelineAssetBatchRes(
        type="seasonality",
        decision=decision,
        inverse_spec=inverse_specs,
        updated_data=updated,
        all_tests=diagnostics if include_diagnostics else None,
    )


def detrend_pipeline(
    data: DataFrame,
    assets: list[str] | None = None,
    order_max: int = 3,
    threshold_order: int = 2,
    include_diagnostics: bool = False,
    *,
    trend_type: Literal["deterministic", "stochastic", "both"] = "both",
) -> PipelineAssetBatchRes:
    """Diagnose, decide, and apply detrending per asset."""
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    diagnostics = trend_diagnostic(
        data=data,
        assets=assets,
        order_max=order_max,
        threshold_order=threshold_order,
        trend_type=trend_type,
    )

    per_asset_decision = _detrend_decision_rule(
        detrend_res=diagnostics,
        assets=assets,
    )

    updated, inverse_specs = _apply_detrend(
        data=data,
        decision=per_asset_decision,
    )

    return PipelineAssetBatchRes(
        type="trend",
        decision=per_asset_decision,
        inverse_spec=inverse_specs,
        updated_data=updated,
        all_tests=diagnostics if include_diagnostics else None,
    )


def _diff_assets(data: DataFrame, assets: list[str]) -> DataFrame:
    """First-difference selected asset columns; drop the leading null via slice."""
    df = data.select(list(assets)).with_columns(
        [pl.col(a).diff().alias(a) for a in assets]
    )
    return df.slice(1)  # removes the first null introduced by diff


def _find_nonwhite_noise_assets(
    increments_df: pl.DataFrame, prob: ProbVector, assets: list[str]
) -> list[str]:
    """Return assets whose increments fail white-noise tests."""
    wn = check_white_noise(data=increments_df.select(assets), assets=assets, prob=prob)
    return [a for a, ok in wn.items() if not ok]


def overwrite_with_transforms(
    base: pl.DataFrame,
    patch: pl.DataFrame,
    assets: list[str],
    suffix: str,
) -> pl.DataFrame:
    j = base.join(patch, on="date", how="left", suffix=suffix)

    exprs: list[pl.Expr] = []
    drop_cols: list[str] = []

    for a in assets:
        pa = f"{a}{suffix}"
        if pa in j.columns:
            exprs.append(pl.col(pa).alias(a))
            drop_cols.append(pa)
        else:
            exprs.append(pl.col(a))

    return j.with_columns(exprs).drop(drop_cols)


def test_increments_idd(
    data: pl.DataFrame, original_prob: ProbVector, assets: list[str]
) -> list[str]:
    increments_df = _diff_assets(data, assets)
    increments_prob = compensate_prob(original_prob, data.height - increments_df.height)
    assets_need_preprocess = _find_nonwhite_noise_assets(
        increments_df=increments_df, prob=increments_prob, assets=assets
    )
    return assets_need_preprocess


# TODO: Review dropping nulls blankly - prob is a better way
def run_univariate_preprocess(
    data: pl.DataFrame,
    prob: ProbVector,
    assets: list[str] | None = None,
) -> UnivariatePreprocess:
    """
    Pipeline:
      1) Screen assets by increments white-noise
      2) Detrend selected assets
      3) Deseason selected assets
    Returns:
      UnivariatePreprocess
    """

    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    logger.info(
        "Starting univariate preprocess: rows=%d assets=%s",
        data.height,
        assets,
    )

    assets_need_preprocess = test_increments_idd(
        data=data, original_prob=prob, assets=assets
    )
    applied_transforms: dict[str, list[AppliedTransform]] = {
        asset: [] for asset in assets_need_preprocess
    }

    # Trend
    detrend = detrend_pipeline(
        data=data.select(["date", *assets]),
        assets=assets_need_preprocess,
        include_diagnostics=False,
    )
    for asset, decision in detrend.decision.items():
        applied_transforms[asset].append(
            AppliedTransform(
                asset=asset,
                decision=decision,
                inverse_spec=detrend.inverse_spec[asset]
                if detrend.inverse_spec
                else None,
            )
        )

    after_detrend = overwrite_with_transforms(
        base=data, patch=detrend.updated_data, assets=assets, suffix="_detrend"
    )

    # Seasonality
    deseason = deseason_pipeline(
        data=after_detrend.select(["date", *assets]),
        assets=assets_need_preprocess,
        include_diagnostics=False,
    )
    for asset, seasons in deseason.decision.items():
        if seasons:
            applied_transforms[asset].append(
                AppliedTransform(
                    asset=asset,
                    decision=TransformDecision(kind="seasonality", order=None),
                    inverse_spec=deseason.inverse_spec[asset]
                    if deseason.inverse_spec
                    else None,
                )
            )

    final = overwrite_with_transforms(
        base=after_detrend,
        patch=deseason.updated_data,
        assets=assets,
        suffix="_deseason",
    )

    logger.info("Finished univariate preprocess: results=%s", applied_transforms)

    return UnivariatePreprocess(
        post_data=final,
        inverse_specs=applied_transforms,
        needs_further_modelling=assets_need_preprocess,
    )

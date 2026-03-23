import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import polars as pl
from polars.dataframe.frame import DataFrame

from globals import LAGS
from maths.distributions import uniform_probs
from maths.helpers import add_detrend_column, add_differenced_columns
from maths.time_series.diagnostics.seasonality import (
    SEASONAL_MAP,
    SeasonalityPeriodTest,
    seasonality_diagnostic,
)
from maths.time_series.diagnostics.trends import TrendTest, trend_diagnostic
from maths.time_series.iid_tests import (
    TestResultByAsset,
    copula_lag_independence_test,
    ellipsoid_lag_test,
    univariate_kolmogrov_smirnov_test,
)
from maths.time_series.operations import deterministic_seasonal_adjustment
from methods.cma import CopulaMarginalModel
from models.types import ProbVector
from utils.helpers import (
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
    coefficients: list[float]


@dataclass(frozen=True)
class DifferenceInverseSpec:
    order: int
    initial_values: list[float]


@dataclass(frozen=True)
class SeasonalInverseSpec:
    frequencies_radians: list[float]
    coefficients: list[float]


InverseSpec = PolynomialInverseSpec | DifferenceInverseSpec | SeasonalInverseSpec


@dataclass(frozen=True)
class AppliedTransform:
    asset: str
    decision: TransformDecision
    inverse_spec: InverseSpec | None = None


@dataclass(frozen=True)
class PipelineOutcome:
    type: Literal["trend", "seasonality"]
    decision: dict[str, AppliedTransform]
    updated_data: DataFrame
    all_tests: dict | None


@dataclass(frozen=True)
class UnivariatePreprocess:
    post_data: DataFrame
    pipeline_decisions: dict[str, dict[str, AppliedTransform]]
    needs_further_modelling: list[str]


DiagnosticFun = Callable[..., dict]
DecisionFun = Callable[..., dict]
ApplyFun = Callable[..., DataFrame]


def _run_pipeline(
    *,
    data: DataFrame,
    assets: list[str] | None,
    type_label: Literal["trend", "seasonality"],
    diagnostic_fn: DiagnosticFun,
    decision_rule: DecisionFun,
    apply_fn: ApplyFun,
    include_diagnostics: bool,
    **diag_kwargs,
) -> PipelineOutcome:
    """Orchestrate diagnose → decide → apply for a univariate pipeline.

    Args:
        data: Input DataFrame containing asset columns.
        assets: Subset of asset column names to process. If None, inferred.
        type_label: "trend" or "seasonality" (tags the outcome).
        diagnostic_fn: Callable that returns per-asset diagnostics dict.
        decision_rule: Callable that converts diagnostics -> decision dict.
        apply_fn: Callable that applies the decision to the DataFrame.
        include_diagnostics: If True, include diagnostics in the outcome.
        **diag_kwargs: Extra keyword args forwarded to `diagnostic_fn`.

    Returns:
        PipelineOutcome with applied data, decision, and optional diagnostics.
    """
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    diagnostic = diagnostic_fn(data=data, assets=assets, **diag_kwargs)
    decision = decision_rule(diagnostic, assets=assets)
    updated = apply_fn(data=data, decision=decision)

    return PipelineOutcome(
        type=type_label,
        decision=decision,
        updated_data=updated,
        all_tests=(diagnostic if include_diagnostics else None),
    )


def _run_all_iid_tests(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    lags: int = LAGS["testing"],
) -> dict[str, TestResultByAsset]:
    """Execute all IID tests and return raw per-test, per-asset results.

    Args:
        data: Input DataFrame with asset columns.
        prob: Probability vector aligned to rows.
        assets: Asset column names to test.
        lags: Max lag to evaluate.

    Returns:
        {
          "ellipsoid": {asset -> TestResultByAsset},
          "copula":    {asset -> TestResultByAsset},
          "ks":        {asset -> TestResultByAsset},
        }
    """
    ellipsoid_test = ellipsoid_lag_test(data=data, prob=prob, lags=lags, assets=assets)
    copula_marginal_model = CopulaMarginalModel.from_data_and_prob(data=data, prob=prob)

    copula_lag_test_res = copula_lag_independence_test(
        copula=copula_marginal_model.copula,
        prob=copula_marginal_model.prob,
        lags=lags,
        assets=assets,
    )

    ks_test = univariate_kolmogrov_smirnov_test(data=data, assets=assets)

    return {
        "ellipsoid": ellipsoid_test,
        "copula": copula_lag_test_res,
        "ks": ks_test,
    }


def check_white_noise(
    data: DataFrame,
    prob: ProbVector | None = None,
    assets: list[str] | None = None,
    lags: int = LAGS["testing"],
) -> dict[str, bool]:
    """Run three IID tests and summarize white-noise by asset.

    Tests:
        1) Copula-based lag independence
        2) Kolmogorov–Smirnov (two-sample, split)
        3) Ellipsoid (Gaussian) lag test

    Args:
        data: Input DataFrame with asset columns.
        prob: Optional probability vector aligned to rows. If None, uniform.
        assets: Asset columns to test; inferred if None.
        lags: Max lag to test for the lag-based tests.

    Returns:
        Mapping {asset -> True if all tests pass across all lags, else False}.
    """
    if prob is None:
        prob = uniform_probs(data.height)

    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    wn_tests = _run_all_iid_tests(data=data, prob=prob, assets=assets, lags=lags)
    return {
        a: not any(wn_tests[t][a].rejected for t in ("ellipsoid", "copula", "ks"))
        for a in assets
    }


def _detrend_decision_rule(
    detrend_res: dict[str, dict[str, TrendTest]], assets: list[str]
) -> dict[str, TransformDecision]:
    """Choose the lowest-order winning trend transform per asset.

    Strategy:
        Prefer the smallest order; tie-break in favor of "polynomial".
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

        transformation, order = min(
            candidates, key=lambda x: (x[1], x[0] != "polynomial")
        )

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
) -> DataFrame:
    """Apply each grouped transform to the data."""
    for (transform, order), assets in by_group.items():
        if transform == "difference":
            data = add_differenced_columns(
                data=data,
                assets=assets,
                difference=order,
                keep_all=True,
            )
        elif transform == "polynomial":
            data = add_detrend_column(
                data=data,
                assets=assets,
                polynomial_orders=[order],
            )
        else:
            raise ValueError(f"Unknown transform '{transform}' for assets {assets}")

    return data


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
) -> DataFrame:
    """Apply detrending decisions and return date + transformed asset columns."""
    if not decision:
        return data.select(["date"])

    by_group = _group_detrend_assets(decision)
    if not by_group:
        return data.select(["date"])

    transformed = _apply_grouped_detrend(data, by_group)
    return _select_and_rename_detrended_columns(transformed, decision)


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
    data: pl.DataFrame, decision: dict[str, list[tuple[str, float]]]
) -> pl.DataFrame:
    assets = [asset for asset, seasons in decision.items() if seasons]
    if not assets:
        return data.select(["date"])

    out = data.select(["date"])

    for asset in assets:
        seasons = decision[asset]
        omega = [rad for _, rad in seasons]

        adj = deterministic_seasonal_adjustment(
            data=data,
            asset=asset,
            frequency_radians=omega,
        )

        out = out.join(adj, on="date", how="left")

    return out


def deseason_pipeline(
    data: DataFrame,
    assets: list[str] | None = None,
    include_diagnostics: bool = False,
) -> PipelineOutcome:
    """Detect and remove seasonality for selected assets.

    Steps:
        diagnose -> select significant periods -> deseason (harmonic residuals).

    Args:
        data: Input DataFrame with asset columns.
        assets: Asset names; if None, inferred.
        include_diagnostics: If True, include per-asset period tests.

    Returns:
        PipelineOutcome tagged as "seasonality".
    """
    return _run_pipeline(
        data=data,
        assets=assets,
        type_label="seasonality",
        diagnostic_fn=lambda **kw: seasonality_diagnostic(
            data=kw["data"],
            assets=kw["assets"],
        ),
        decision_rule=lambda tests, assets=None: _deseason_decision_rule(tests),
        apply_fn=lambda *, data, decision: _deseason_apply(
            data=data, decision=decision
        ),
        include_diagnostics=include_diagnostics,
    )


def detrend_pipeline(
    data: DataFrame,
    assets: list[str] | None = None,
    order_max: int = 3,
    threshold_order: int = 2,
    include_diagnostics: bool = False,
    *,
    trend_type: Literal["deterministic", "stochastic", "both"] = "both",
) -> PipelineOutcome:
    """Select and apply the minimal trend transform per asset.

    Steps:
        diagnose (deterministic/stochastic) -> pick lowest order -> apply.

    Args:
        data: Input DataFrame with asset columns.
        assets: Asset names; if None, inferred.
        order_max: Maximum order to test for each trend family.
        threshold_order: Evidence threshold for accepting a transform.
        include_diagnostics: If True, include per-asset TrendTest results.
        trend_type: Which family to consider ("deterministic", "stochastic", "both").

    Returns:
        PipelineOutcome tagged as "trend".
    """
    return _run_pipeline(
        data=data,
        assets=assets,
        type_label="trend",
        diagnostic_fn=lambda **kw: trend_diagnostic(
            data=kw["data"],
            assets=kw["assets"],
            order_max=order_max,
            threshold_order=threshold_order,
            trend_type=trend_type,
        ),
        decision_rule=lambda tests, assets=None: _detrend_decision_rule(
            detrend_res=tests, assets=(assets or [])
        ),
        apply_fn=lambda *, data, decision: _apply_detrend(data=data, decision=decision),
        include_diagnostics=include_diagnostics,
    )


def _diff_assets(data: DataFrame, assets: list[str]) -> DataFrame:
    """First-difference selected asset columns; drop the leading null via slice."""
    df = data.select(list(assets)).with_columns(
        [pl.col(a).diff().alias(a) for a in assets]
    )
    return df.slice(1)  # removes the first null introduced by diff


def _find_nonwhite_noise_assets(
    increments_df: pl.DataFrame, assets: list[str]
) -> list[str]:
    """Return assets whose increments fail white-noise tests."""
    wn = check_white_noise(data=increments_df.select(assets))
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


# TODO: Make sure date is also returned
# TODO: Review dropping nulls blankly - prob is a better way
def run_univariate_preprocess(
    data: pl.DataFrame,
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

    increments_df = _diff_assets(data, assets)
    assets_need_preprocess = _find_nonwhite_noise_assets(increments_df, assets)

    if not assets_need_preprocess:
        logger.info("No preprocessing needed")
        return UnivariatePreprocess(data, {"trend": {}, "deseason": {}}, [])

    # Trend
    detrend = detrend_pipeline(
        data=data.select(["date", *assets]),
        assets=assets_need_preprocess,
        include_diagnostics=False,
    )
    logger.info("Detrend decisions: %s", detrend.decision)

    after_detrend = overwrite_with_transforms(
        base=data, patch=detrend.updated_data, assets=assets, suffix="_detrend"
    )

    # Seasonality
    deseason = deseason_pipeline(
        data=after_detrend.select(["date", *assets]),
        assets=assets_need_preprocess,
        include_diagnostics=False,
    )
    logger.info("Deseason decisions: %s", deseason.decision)

    final = overwrite_with_transforms(
        base=after_detrend,
        patch=deseason.updated_data,
        assets=assets,
        suffix="_deseason",
    )

    pipeline_decisions = {"trend": detrend.decision, "deseason": deseason.decision}

    logger.info(
        "Finished univariate preprocess: transformed_assets=%s",
        assets_need_preprocess,
    )

    return UnivariatePreprocess(final, pipeline_decisions, assets_need_preprocess)

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import polars as pl
from polars import Series
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


@dataclass(frozen=True)
class PipelineOutcome:
    """Result of a univariate preprocessing pipeline run."""

    type: Literal["trend", "seasonality"]
    decision: dict[str, tuple[str, int]] | dict[str, list[tuple[str, float]]]
    updated_data: DataFrame
    all_tests: (
        dict[str, dict[str, TrendTest]] | dict[str, list[SeasonalityPeriodTest]] | None
    )


@dataclass(frozen=True)
class UnivariatePreprocess:
    post_data: DataFrame
    pipeline_decisions: dict[  # TODO: Make datastructs here as too complicated
        str, dict[str, list[tuple[str, float]]] | dict[str, tuple[str, int]]
    ]
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
    # Resolve assets once; keep consistent with your helpers
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    # 1) diagnose
    diagnostic = diagnostic_fn(data=data, assets=assets, **diag_kwargs)

    # 2) decide
    # (Trend decision often needs assets order; seasonality doesn’t—this signature stays flexible)
    decision = decision_rule(diagnostic, assets=assets)

    # 3) apply — let your apply_* do the batching/fusing
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
) -> dict[str, tuple[str, int]]:
    """Choose the lowest-order winning trend transform per asset.

    Strategy:
        Prefer the smallest order; tie-break in favor of "polynomial".

    Args:
        detrend_res: {"deterministic" | "stochastic" -> {asset -> TrendTest}}.
        assets: Ordered list of asset names to consider.

    Returns:
        {asset -> ("polynomial" | "difference", order:int)} for decided assets.
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

        if stochastic_res:
            candidates.append(("difference", stochastic_res))

        if not candidates:
            continue

        transformation, order = min(
            candidates, key=lambda x: (x[1], x[0] != "polynomial")
        )

        trend_trans[asset] = (transformation, order)
    return trend_trans


def _apply_detrend(data: DataFrame, decision: dict[str, tuple[str, int]]) -> DataFrame:
    """Apply per-asset trend decisions in batched passes.

    Batches:
        - Group by (transform, order) to minimize frame materializations.
        - Compute all diffs of the same order together; same for polynomials.

    Args:
        data: Input DataFrame with asset columns.
        decision: {asset -> ("polynomial" | "difference", order:int)}.

    Returns:
        DataFrame where selected assets are replaced with transformed series.

    Raises:
        ValueError: If an expected transformed column is not found.
    """
    if not decision:
        return data

    # will group assets by transform
    by_group: dict[tuple[str, int], list[str]] = {}
    for asset, (transform, order) in decision.items():
        by_group.setdefault((transform, order), []).append(asset)

    for (transform, order), assets in by_group.items():
        if transform == "difference":
            data = add_differenced_columns(
                data=data, assets=assets, difference=order, keep_all=True
            )
        elif transform == "polynomial":
            data = add_detrend_column(
                data=data, assets=assets, polynomial_orders=[order]
            )
        else:
            raise ValueError(f"Unknown transform '{transform}' for assets {assets}")

    # renames to original cols
    rename_map = {}
    for asset, (transform, order) in decision.items():
        col = (
            f"{asset}_diff_{order}"
            if transform == "difference"
            else f"{asset}_detrended_p{order}"
        )
        if col not in data.columns:
            raise ValueError(
                f"Expected transformed column '{col}' not found. "
                f"Available columns: {data.columns}"
            )
        rename_map[col] = asset

    return data.drop(list(decision.keys())).rename(rename_map)


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
    Returns {asset -> [(label, omega_radians), ...]}.
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
    data: DataFrame, decision: dict[str, list[tuple[str, float]]]
) -> DataFrame:
    """Deterministically remove seasonality via harmonic regression.

    For each asset with detected periods, fit sin/cos harmonics at the chosen
    angular frequencies and replace the asset column with residuals.

    Args:
        data: Input DataFrame.
        decision: {asset -> [(period_label, omega_radians), ...]}.

    Returns:
        DataFrame with deseasoned asset columns; no-op if `decision` is empty.
    """
    if not any(decision.values()):
        return data
    deseasoned_assets = {
        asset: deterministic_seasonal_adjustment(
            data,
            asset=asset,
            frequency_radians=[rad for _, rad in seasons],
        )[asset]
        for asset, seasons in decision.items()
        if seasons
    }
    return data.with_columns(
        [Series(name, values) for name, values in deseasoned_assets.items()]
    )


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

    increments_df = _diff_assets(data, assets)

    assets_need_preprocess = _find_nonwhite_noise_assets(increments_df, assets)
    if not assets_need_preprocess:
        return UnivariatePreprocess(data, {"trend": {}, "deseason": {}}, [])

    # Trend
    detrend = detrend_pipeline(
        data=data.select(assets),
        assets=assets_need_preprocess,
        include_diagnostics=False,
    )
    detrended = detrend.updated_data.drop_nulls()

    # Deseason
    deseason = deseason_pipeline(
        data=detrended,
        assets=assets_need_preprocess,
        include_diagnostics=False,
    )
    transformed = deseason.updated_data

    pipeline_decisions = {"trend": detrend.decision, "deseason": deseason.decision}
    return UnivariatePreprocess(transformed, pipeline_decisions, assets_need_preprocess)

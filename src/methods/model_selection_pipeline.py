from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy._typing import NDArray
from polars import DataFrame
from statsmodels.stats.multitest import multipletests
from typing_extensions import Literal

from maths.time_series.iid_tests import arch_test, ljung_box_test
from maths.time_series.models import (
    AutoARMARes,
    AutoGARCHRes,
    DemeanRes,
    auto_arma,
    auto_garch,
    by_criteria,
)

MeanModelRes = AutoARMARes | DemeanRes


MeanKind = Literal["none", "demean", "arma"]
VolKind = Literal["none", "garch"]

ModelType = Literal[
    "ARMA + GARCH", "ARMA", "GARCH", "Random Walk", "Demean", "Demean + GARCH"
]


_MODEL_TYPE_MAP: dict[tuple[MeanKind, VolKind], ModelType] = {
    ("none", "none"): "Random Walk",
    ("none", "garch"): "GARCH",
    ("demean", "none"): "Demean",
    ("demean", "garch"): "Demean + GARCH",
    ("arma", "none"): "ARMA",
    ("arma", "garch"): "ARMA + GARCH",
}


@dataclass
class UnivariateModel:
    mean_model: MeanModelRes | None
    volatility_model: AutoGARCHRes | None

    @property
    def mean_kind(self) -> MeanKind:
        return "none" if self.mean_model is None else self.mean_model.kind

    @property
    def vol_kind(self) -> VolKind:
        return "none" if self.volatility_model is None else self.volatility_model.kind

    @property
    def model_type(self) -> ModelType:
        return _MODEL_TYPE_MAP[(self.mean_kind, self.vol_kind)]

    def invariant(self, non_null_values: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Returns the correct invariant to be used downstream for forecasting purposes.

        Rules are:
        - If volatility_model exists: invariants are already aligned to non-null fit sample.
        - If only mean_model exists: residuals are aligned to non-null fit sample.
        - Random Walk (no mean, no vol): innovations = [nan] + diff(non_null_values)
          so length matches non-null sample, like Polars diff does.
        """
        if self.volatility_model is not None:
            invariant = self.volatility_model.invariants
        elif self.mean_model is not None:
            invariant = self.mean_model.residuals
        else:
            invariant = np.concatenate(
                [np.array([np.nan], dtype=float), np.diff(non_null_values)]
            )
        if invariant.shape[0] != non_null_values.shape[0]:
            raise ValueError(
                f"Innovation length mismatch: innov={invariant.shape[0]} "
                f"vs non_null_values={non_null_values.shape[0]} "
                f"(model_type={self.model_type})"
            )
        return invariant.astype(float)


# TODO: Move to a more appropriate place
def multiple_tests_rejected(
    p_values: list[float], significance_level: float = 0.05
) -> list[bool]:
    """
    Adjusts p-values for multiple tests then compares to alpha to determine rejection
    """
    return multipletests(p_values, alpha=significance_level, method="holm-sidak")[
        0
    ].tolist()


def asset_needs_volatility_model(
    residual: NDArray[np.floating],
    degrees_of_freedom: int,
    ljung_box_lags: list[int],
    arch_lags: list[int],
    min_ljung_box_rejections: int = 2,
    min_arch_rejections: int = 1,
) -> bool:
    residual_sq = residual**2
    ljung_box_pvals = ljung_box_test(  # This is effectively the McLeod- Li test
        residual_sq, lags=ljung_box_lags, degrees_of_freedom=degrees_of_freedom
    ).p_vals
    arch_vals = arch_test(
        residual, lags_to_test=arch_lags, degrees_of_freedom=degrees_of_freedom
    ).p_vals

    ljung_box_rejected = multiple_tests_rejected(ljung_box_pvals)
    arch_rejected = multiple_tests_rejected(arch_vals)
    return (sum(ljung_box_rejected) >= min_ljung_box_rejections) or (
        sum(arch_rejected) >= min_arch_rejections
    )


def asset_needs_mean_modelling(
    data: NDArray[np.floating],
    degrees_of_freedom: int,
    ljung_box_lags: list[int] = [10, 15, 20],
    min_ljung_box_rejections: int = 1,
) -> bool:
    ljung_box = ljung_box_test(
        data=data, lags=ljung_box_lags, degrees_of_freedom=degrees_of_freedom
    )
    ljung_box_rejected = multiple_tests_rejected(ljung_box.p_vals)
    return sum(ljung_box_rejected) >= min_ljung_box_rejections


# TODO: Don't love this Mapping thing -> should change?
def needs_volatility_modelling(
    mean_model_res: Mapping[str, MeanModelRes],
    ljung_box_lags: list[int] = [10, 20],
    arch_lags: list[int] = [5, 10, 15],
    min_ljung_box_rejections: int = 2,
    min_arch_rejections: int = 1,
) -> list[str]:
    needs: list[str] = []

    for asset, res in mean_model_res.items():
        if asset_needs_volatility_model(
            residual=res.residuals,
            degrees_of_freedom=res.degrees_of_freedom,
            ljung_box_lags=ljung_box_lags,
            arch_lags=arch_lags,
            min_ljung_box_rejections=min_ljung_box_rejections,
            min_arch_rejections=min_arch_rejections,
        ):
            needs.append(asset)

    return needs


def needs_mean_modelling(
    data: DataFrame,
    assets_to_test: list[str],
    degrees_of_freedom: int,
    ljung_box_lags: list[int] = [10, 15, 20],
    min_ljung_box_rejections: int = 1,
) -> list[str]:
    """
    Identifies assets with significant autocorrelation in series using the Ljung–Box test.

    Note: Test to be run on series at this stage, not residuals.

    Returns a list of asset names that fail the test and require mean modeling.
    """
    needs_mean_modelling = []
    for asset in assets_to_test:
        array = (
            data.select(asset).drop_nulls().to_numpy().ravel()
        )  # remove null if asset has any
        if asset_needs_mean_modelling(
            array,
            ljung_box_lags=ljung_box_lags,
            degrees_of_freedom=degrees_of_freedom,
            min_ljung_box_rejections=min_ljung_box_rejections,
        ):
            needs_mean_modelling.append(asset)
    return needs_mean_modelling


def run_best_arma(
    asset_array: NDArray[np.floating],
    asset_name: str,
    search_n_models: int = 5,
    information_criteria: Literal["bic", "aic"] = "bic",
) -> AutoARMARes:
    """
    Selects the best ARMA model by information criterion (AIC/BIC) among those
    whose residuals pass the Ljung–Box test for no autocorrelation.

    Falls back to the best information criterion model if none pass.
    """

    candidate_models_res = auto_arma(
        asset_array=asset_array,
        max_ar_order=3,
        max_ma_order=3,
        information_criteria=information_criteria,
        top_n_models=search_n_models,
    )

    candidates_by_information_criteria = sorted(candidate_models_res, key=by_criteria)

    for model in candidates_by_information_criteria:
        if not asset_needs_mean_modelling(
            model.residuals, degrees_of_freedom=model.degrees_of_freedom
        ):
            return model
    print(
        f"Mean Modelling - For {asset_name} No model's residual has passed the ljung box test, please review your model, returning model with best information criteria for now."
    )

    return min(candidate_models_res, key=by_criteria)


def run_best_garch(
    asset_array: NDArray[np.floating],
    asset_name: str,
) -> AutoGARCHRes:
    candidate_models_res = auto_garch(
        asset_array=asset_array,
        max_p_order=2,
        max_o_order=1,
        max_q_order=2,
    )

    candidates_by_information_criteria = sorted(candidate_models_res, key=by_criteria)

    for model in candidates_by_information_criteria:
        if not asset_needs_volatility_model(
            residual=model.invariants,
            degrees_of_freedom=model.degrees_of_freedom,
            ljung_box_lags=[10, 20],
            arch_lags=[5, 10, 15],
        ):
            return model
    print(
        f"Volatility Modelling - for {asset_name} No model's residual has passed the ljung box test, please review your model, returning model with best information criteria for now."
    )

    return min(candidate_models_res, key=by_criteria)


def mean_modelling_pipeline(
    data: DataFrame, assets: list[str]
) -> dict[str, MeanModelRes]:
    """
    Return a mean-modelling result for every asset:
    - AutoARMARes if Ljung–Box suggests mean dependence
    - DemeanRes otherwise
    """
    assets_needing_arma = needs_mean_modelling(
        data=data, assets_to_test=assets, degrees_of_freedom=0
    )  # At this point DOF is 0
    asset_mean_model_res: dict[str, MeanModelRes] = {}
    for asset in assets:
        array = (
            data.select(asset).drop_nulls().to_numpy().ravel()
        )  # drop null if asset contains any
        if asset in assets_needing_arma:
            asset_mean_model_res[asset] = run_best_arma(array, asset_name=asset)
        else:
            mean = array.mean()
            asset_mean_model_res[asset] = DemeanRes(
                degrees_of_freedom=0,
                params={"mean": mean},
                residuals=array - mean,
            )
    return asset_mean_model_res


def volatility_modelling_pipeline(
    mean_model_res: Mapping[str, MeanModelRes],
):
    assets_needing_garch = needs_volatility_modelling(mean_model_res)
    asset_vol_model_res = {}
    for asset in assets_needing_garch:
        asset_vol_model_res[asset] = run_best_garch(
            mean_model_res[asset].residuals, asset
        )

    return asset_vol_model_res


def build_best_univariate_model(
    data: DataFrame, assets_to_model: list[str]
) -> dict[str, UnivariateModel]:
    mean_modelling = mean_modelling_pipeline(data=data, assets=assets_to_model)
    volatility_modelling = volatility_modelling_pipeline(mean_model_res=mean_modelling)
    all_assets = [c for c in data.columns if c != "date"]
    asset_model = {}
    for asset in all_assets:
        asset_model[asset] = UnivariateModel(
            mean_model=mean_modelling.get(asset),
            volatility_model=volatility_modelling.get(asset),
        )

    return asset_model

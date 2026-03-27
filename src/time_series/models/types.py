from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, NamedTuple

import numpy as np
from numpy._typing import NDArray

MeanKind = Literal["none", "demean", "arma"]
VolKind = Literal["none", "garch"]


class AutoGARCHRes(NamedTuple):
    model_order: tuple[int, int, int]
    degrees_of_freedom: int
    criteria: Literal["aic", "bic"]
    criteria_res: float
    params: dict[str, float]
    p_values: NDArray[np.floating]
    residuals: NDArray[np.floating]
    conditional_volatility: NDArray[np.floating]
    invariants: NDArray[np.floating]
    kind: Literal["garch"] = "garch"


GARCH_DISTRIBUTIONS = Literal["t", "normal"]


class AutoARMARes(NamedTuple):
    model_order: tuple[int, int]
    degrees_of_freedom: int
    criteria: Literal["aic", "bic"]
    criteria_res: float
    params: dict[str, float]
    p_values: NDArray[np.floating]
    residuals: NDArray[np.floating]
    residual_scale: float
    kind: Literal["arma"] = "arma"


class DemeanRes(NamedTuple):
    model_order: None
    degrees_of_freedom: int
    params: dict[str, float]
    residuals: NDArray[np.floating]
    residual_scale: float
    kind: Literal["demean"] = "demean"


MeanModelRes = AutoARMARes | DemeanRes


@dataclass(frozen=True, slots=True)
class CompiledParams:
    mu: float
    ar: NDArray[np.floating]
    ma: NDArray[np.floating]
    omega: float
    alpha: NDArray[np.floating]
    gamma: NDArray[np.floating]
    beta: NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class UnivariateModel:
    mean_kind: MeanKind
    mean_params: dict[str, float]
    vol_params: dict[str, float]
    innovation_scale: float = 1.0
    mean_order: tuple[int, int] = (0, 0)
    vol_kind: VolKind = "none"
    vol_order: tuple[int, int, int] = (0, 0, 0)

    @classmethod
    def from_fitting_results(
        cls,
        fitting_results: UnivariateRes,
    ):
        if fitting_results.mean_res is None:
            mean_kind: MeanKind = "none"
            mean_order = (0, 0)
            mean_params = {}
        else:
            mean_kind = fitting_results.mean_res.kind
            mean_order = fitting_results.mean_res.model_order
            if mean_order is None:
                mean_order = (0, 0)
            mean_params = fitting_results.mean_res.params

        if fitting_results.volatility_res is None:
            vol_kind: VolKind = "none"
            vol_order = (0, 0, 0)
            vol_params = {}
            if fitting_results.mean_res is not None:
                innovation_scale = float(fitting_results.mean_res.residual_scale)
            else:
                innovation_scale = 1.0
        else:
            vol_kind = "garch"
            vol_order = fitting_results.volatility_res.model_order
            vol_params = fitting_results.volatility_res.params
            innovation_scale = 1.0

        return cls(
            mean_kind=mean_kind,
            mean_order=mean_order,
            mean_params=mean_params,
            vol_kind=vol_kind,
            vol_order=vol_order,
            innovation_scale=innovation_scale,
            vol_params=vol_params,
        )

    @property
    def all_params(self) -> Mapping[str, float | NDArray[np.floating]]:
        return self.mean_params | self.vol_params

    def _param_get(
        self, params: dict[str, float], *keys: str, default: float = 0.0
    ) -> float:
        for k in keys:
            if k in params:
                return float(params[k])
        return float(default)

    def _get_lag(self, params: dict[str, float], base: str, lag: int) -> float:
        return self._param_get(params, f"{base}[{lag}]", f"{base}.L{lag}", default=0.0)

    def compile_arma_params(
        self,
    ) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
        # returns (mu, ar(p), ma(q))
        if self.mean_kind == "none":
            return 0.0, np.zeros(0), np.zeros(0)
        if self.mean_kind == "demean":
            mu = float((self.mean_params or {}).get("mean", 0.0))
            return mu, np.zeros(0), np.zeros(0)
        if self.mean_kind == "arma":
            p, q = self.mean_order
            mu = float(
                (self.mean_params or {}).get(
                    "const", (self.mean_params or {}).get("mu", 0.0)
                )
            )
            ar = np.array(
                [
                    self._get_lag(self.mean_params or {}, "ar", i)
                    for i in range(1, p + 1)
                ],
                dtype=float,
            )
            ma = np.array(
                [
                    self._get_lag(self.mean_params or {}, "ma", j)
                    for j in range(1, q + 1)
                ],
                dtype=float,
            )
            return mu, ar, ma
        raise ValueError(self.mean_kind)

    def compile_garch_params(
        self,
    ) -> tuple[float, NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        # returns (omega, alpha(p), gamma(o), beta(q))
        if self.vol_kind == "none":
            return 0.0, np.zeros(0), np.zeros(0), np.zeros(0)
        if self.vol_kind == "garch":
            p, o, q = self.vol_order
            vp = self.vol_params or {}
            omega = float(vp.get("omega", 0.0))
            alpha = np.array(
                [float(vp.get(f"alpha[{i}]", 0.0)) for i in range(1, p + 1)],
                dtype=float,
            )
            gamma = np.array(
                [float(vp.get(f"gamma[{i}]", 0.0)) for i in range(1, o + 1)],
                dtype=float,
            )
            beta = np.array(
                [float(vp.get(f"beta[{i}]", 0.0)) for i in range(1, q + 1)], dtype=float
            )
            return omega, alpha, gamma, beta
        raise ValueError(self.vol_kind)

    def compile_params(self) -> CompiledParams:
        return CompiledParams(
            *self.compile_arma_params(),
            *self.compile_garch_params(),
        )


@dataclass
class UnivariateRes:
    mean_res: MeanModelRes | None
    volatility_res: AutoGARCHRes | None

    def innovation_scale(self, non_null_values: NDArray[np.floating]) -> float:
        """
        Scale used to standardize / unstandardize innovations for non-GARCH assets.
        For GARCH assets, innovations are already standardized dynamically.
        """
        if self.volatility_res is not None:
            return 1.0

        if self.mean_res is not None:
            scale = float(self.mean_res.residual_scale)
            return 1.0 if (not np.isfinite(scale) or scale <= 0) else scale

        diff = np.diff(non_null_values)
        if diff.size == 0:
            return 1.0

        scale = float(np.std(diff, ddof=1))
        return 1.0 if (not np.isfinite(scale) or scale <= 0) else scale

    def invariant(self, non_null_values: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return standardized innovations for all asset types.

        - GARCH asset: standardized residuals from GARCH fit
        - mean-only asset: residuals / residual_scale
        - no-model asset: [nan] + diff(series) / diff_scale
        """
        if self.volatility_res is not None:
            invariant = np.asarray(self.volatility_res.invariants, dtype=float)

        elif self.mean_res is not None:
            scale = self.innovation_scale(non_null_values)
            invariant = np.asarray(self.mean_res.residuals, dtype=float) / scale

        else:
            diff = np.concatenate(
                [np.array([np.nan], dtype=float), np.diff(non_null_values)]
            )
            scale = self.innovation_scale(non_null_values)
            invariant = diff / scale

        if invariant.shape[0] != non_null_values.shape[0]:
            raise ValueError(
                f"Innovation length mismatch: innov={invariant.shape[0]} "
                f"vs non_null_values={non_null_values.shape[0]}"
            )
        return invariant.astype(float)

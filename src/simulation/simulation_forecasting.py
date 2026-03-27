from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from numpy._typing import NDArray

from time_series.models.types import CompiledParams, UnivariateModel, UnivariateRes


@dataclass(slots=True)
class UnivariateState:
    series_hist: NDArray[np.floating]
    ma_residual_lags: NDArray[np.floating] | None = None
    vol_residual_lags: NDArray[np.floating] | None = None
    var_hist: NDArray[np.floating] | None = None

    @classmethod
    def from_fitting_results_and_model(
        cls,
        fitting_results: UnivariateRes,
        univariate_model: UnivariateModel,
        post_series_non_null: NDArray[np.floating],
        x_hist_len: int = 10,
    ):
        x_hist = post_series_non_null[
            -min(x_hist_len, post_series_non_null.size) :
        ].copy()

        eps_mean_hist = None
        if (
            univariate_model.mean_kind == "arma"
            and fitting_results.mean_res is not None
        ):
            p, q = univariate_model.mean_order
            if x_hist.size < p:
                x_hist = post_series_non_null[-p:].copy()
            eps = fitting_results.mean_res.residuals
            eps_mean_hist = eps[-q:].copy() if q > 0 else None

        eps_vol_hist = None
        var_hist = None
        if (
            univariate_model.vol_kind == "garch"
            and fitting_results.volatility_res is not None
        ):
            p_g, o_g, q_g = univariate_model.vol_order
            m = max(p_g, o_g, 1)
            sig2 = fitting_results.volatility_res.conditional_volatility**2
            eps_vol_hist = (
                fitting_results.volatility_res.residuals[-m:].copy() if m > 0 else None
            )
            var_hist = sig2[-q_g:].copy() if q_g > 0 else None

        return cls(
            series_hist=x_hist,
            ma_residual_lags=eps_mean_hist,
            vol_residual_lags=eps_vol_hist,
            var_hist=var_hist,
        )

    @property
    def state_as_dict(self) -> Mapping[str, NDArray[np.floating]]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ForecastModel:
    model: UnivariateModel
    state0: UnivariateState

    @classmethod
    def from_res_and_series(
        cls,
        fitting_results: UnivariateRes,
        post_series_non_null: NDArray[np.floating],
        x_hist_len: int = 10,
    ):
        if post_series_non_null.size == 0:
            raise ValueError("post_series_non_null is empty")

        model = UnivariateModel.from_fitting_results(fitting_results=fitting_results)

        if fitting_results.mean_res is None and fitting_results.volatility_res is None:
            diff = np.diff(post_series_non_null)
            scale = float(np.std(diff, ddof=1)) if diff.size > 0 else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0

            model = UnivariateModel(
                mean_kind=model.mean_kind,
                mean_order=model.mean_order,
                mean_params=model.mean_params,
                vol_kind=model.vol_kind,
                vol_order=model.vol_order,
                vol_params=model.vol_params,
                innovation_scale=scale,
            )

        return cls(
            model=model,
            state0=UnivariateState.from_fitting_results_and_model(
                fitting_results=fitting_results,
                univariate_model=model,
                post_series_non_null=post_series_non_null,
                x_hist_len=x_hist_len,
            ),
        )


def as_sims_by_horizon(
    array: NDArray[np.floating],
) -> tuple[NDArray[np.floating], int, int]:
    """
    Normalize x to shape (n_sims, horizon), dtype float.
    Accepts 1D (horizon,) or 2D (n_sims, horizon).
    """
    x = np.asarray(array, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("Expected 1D or 2D array.")
    n_sims, horizon = x.shape
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return x, n_sims, horizon


def broadcast_last_k_lags(
    lags_1d: NDArray[np.floating] | None, n_sims: int, k: int, *, name: str
) -> NDArray[np.floating]:
    """
    Return shape (n_sims, k) containing the last k values from lags_1d,
    broadcasted across simulations.

    If k == 0 => returns empty (n_sims, 0).
    """
    if k <= 0:
        return np.empty((n_sims, 0), dtype=float)
    if lags_1d is None:
        raise ValueError(f"Missing {name} lags: need {k} values.")
    lags_1d = np.asarray(lags_1d, dtype=float).ravel()
    if lags_1d.size < k:
        raise ValueError(f"Insufficient {name} lags: need {k}, have {lags_1d.size}.")
    return np.broadcast_to(lags_1d[-k:], (n_sims, k)).copy()


def lag_matrix(
    ext: NDArray[np.floating], last_col: int, k: int
) -> NDArray[np.floating]:
    """
    Return a (n_sims, k) matrix of lags from a 2D 'extended' buffer.
    Column order: [t-1, t-2, ..., t-k]
    """
    if k <= 0:
        return ext[:, :0]
    return ext[:, last_col - np.arange(k)]


def take_vector(
    params: CompiledParams, attr: str, expected: int
) -> NDArray[np.floating]:
    x = np.asarray(getattr(params, attr), dtype=float).ravel()
    if x.size != expected:
        raise ValueError(f"{attr} size {x.size} != {expected}")
    return x


def take_scalar(params: CompiledParams, attr: str) -> float:
    return float(getattr(params, attr))


def garch_params(
    params: CompiledParams, order: tuple[int, int, int]
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    p, o, q = order
    omega = take_scalar(params, "omega")
    alpha = take_vector(params, "alpha", p)
    gamma = take_vector(params, "gamma", o)
    beta = take_vector(params, "beta", q)
    return omega, alpha, gamma, beta


def mean_params(
    params: CompiledParams, order: tuple[int, int]
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    p, q = order
    mu = take_scalar(params, "mu")
    ar = take_vector(params, "ar", p)
    ma = take_vector(params, "ma", q)
    return mu, ar, ma


@dataclass
class GarchSimulator:
    omega: float
    alpha: NDArray[np.floating]
    gamma: NDArray[np.floating]
    beta: NDArray[np.floating]
    p: int
    o: int
    q: int
    eps_ext: NDArray[np.floating]
    var_ext: NDArray[np.floating]
    eps_lag: int
    var_lag: int
    var_floor: float = 1e-12

    @classmethod
    def from_state(
        cls,
        *,
        params: CompiledParams,
        order: tuple[int, int, int],
        n_sims: int,
        horizon: int,
        eps_start: NDArray[np.floating] | None,
        var_start: NDArray[np.floating] | None,
    ) -> GarchSimulator:
        p, o, q = order
        omega, alpha, gamma, beta = garch_params(params, order)

        eps_lag = max(p, o)
        var_lag = q

        eps_ext = np.empty((n_sims, eps_lag + horizon), dtype=float)
        var_ext = np.empty((n_sims, var_lag + horizon), dtype=float)

        if eps_lag > 0:
            eps_ext[:, :eps_lag] = broadcast_last_k_lags(
                eps_start, n_sims, eps_lag, name="eps"
            )
        if var_lag > 0:
            var_ext[:, :var_lag] = broadcast_last_k_lags(
                var_start, n_sims, var_lag, name="variance"
            )

        return cls(
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            p=p,
            o=o,
            q=q,
            eps_ext=eps_ext,
            var_ext=var_ext,
            eps_lag=eps_lag,
            var_lag=var_lag,
        )

    def variance_step(self, t: int) -> NDArray[np.floating]:
        """
        Compute next variance vector v_next for time t.
        """
        n_sims = self.eps_ext.shape[0]
        v_next = np.full(n_sims, self.omega, dtype=float)

        eps_last = self.eps_lag + t - 1
        var_last = self.var_lag + t - 1

        if self.p > 0:
            eps_lags = lag_matrix(self.eps_ext, eps_last, self.p)
            v_next += (eps_lags * eps_lags) @ self.alpha

        if self.o > 0:
            eps_lags_o = lag_matrix(self.eps_ext, eps_last, self.o)
            ind = (eps_lags_o < 0.0).astype(float)
            v_next += (ind * (eps_lags_o * eps_lags_o)) @ self.gamma

        if self.q > 0:
            var_lags = lag_matrix(self.var_ext, var_last, self.q)
            v_next += var_lags @ self.beta

        return np.maximum(v_next, self.var_floor)

    def push(
        self, t: int, eps_next: NDArray[np.floating], var_next: NDArray[np.floating]
    ) -> None:
        """
        Store computed eps and variance into the extended buffers.
        """
        if self.eps_lag > 0:
            self.eps_ext[:, self.eps_lag + t] = eps_next
        if self.var_lag > 0:
            self.var_ext[:, self.var_lag + t] = var_next


def garch_simulation_paths(
    *,
    params: CompiledParams,
    garch_order: tuple[int, int, int],
    eps_start: NDArray[np.floating] | None,
    var_start: NDArray[np.floating] | None,
    innovations: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    innovations, n_sims, horizon = as_sims_by_horizon(innovations)

    sim = GarchSimulator.from_state(
        params=params,
        order=garch_order,
        n_sims=n_sims,
        horizon=horizon,
        eps_start=eps_start,
        var_start=var_start,
    )

    sigma2 = np.empty((n_sims, horizon), dtype=float)
    eps = np.empty((n_sims, horizon), dtype=float)

    for t in range(horizon):
        v_next = sim.variance_step(t)
        eps_next = np.sqrt(v_next) * innovations[:, t]

        sigma2[:, t] = v_next
        eps[:, t] = eps_next

        sim.push(t, eps_next, v_next)

    return sigma2, eps


@dataclass
class MeanSimulator:
    mu: float
    ar: NDArray[np.floating]
    ma: NDArray[np.floating]
    p: int
    q: int
    y_ext: NDArray[np.floating]
    e_ext: NDArray[np.floating]

    @classmethod
    def from_state(
        cls,
        *,
        params: CompiledParams,
        order: tuple[int, int],
        n_sims: int,
        horizon: int,
        series_hist: NDArray[np.floating],
        ma_resid_lags: NDArray[np.floating] | None,
    ) -> MeanSimulator:
        p, q = order
        mu, ar, ma = mean_params(params, order)

        if p > 0 and series_hist.size < p:
            raise ValueError(f"Need at least {p} series lags, have {series_hist.size}.")

        if q > 0:
            if ma_resid_lags is None:
                raise ValueError(f"Need {q} mean residual lags for MA part.")
            if ma_resid_lags.size < q:
                raise ValueError(
                    f"Need {q} mean residual lags, have {ma_resid_lags.size}."
                )
        else:
            ma_resid_lags = np.asarray([], dtype=float)

        y_ext = (
            np.empty((n_sims, p + horizon), dtype=float)
            if p > 0
            else np.empty((n_sims, 0), dtype=float)
        )
        e_ext = (
            np.empty((n_sims, q + horizon), dtype=float)
            if q > 0
            else np.empty((n_sims, 0), dtype=float)
        )

        if p > 0:
            y_ext[:, :p] = np.broadcast_to(series_hist[-p:], (n_sims, p))
        if q > 0:
            e_ext[:, :q] = np.broadcast_to(ma_resid_lags[-q:], (n_sims, q))

        return cls(mu=mu, ar=ar, ma=ma, p=p, q=q, y_ext=y_ext, e_ext=e_ext)

    def mean_step(self, time: int, eps: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute y_next for time t given eps[:, t].
        """
        ar_part = 0.0
        if self.p > 0:
            y_last = self.p + time - 1
            y_lags = lag_matrix(self.y_ext, y_last, self.p)
            ar_part = y_lags @ self.ar

        ma_part = 0.0
        if self.q > 0:
            e_last = self.q + time - 1
            e_lags = lag_matrix(self.e_ext, e_last, self.q)
            ma_part = e_lags @ self.ma

        return self.mu + ar_part + ma_part + eps[:, time]

    def push(
        self, time: int, y_next: NDArray[np.floating], eps_t: NDArray[np.floating]
    ) -> None:
        """
        Update buffers with the new output and eps (as MA residual).
        """
        if self.p > 0:
            self.y_ext[:, self.p + time] = y_next
        if self.q > 0:
            self.e_ext[:, self.q + time] = eps_t


def mean_simulation_paths(
    *,
    params: CompiledParams,
    mean_kind: str,
    mean_order: tuple[int, int],
    state_series_hist: NDArray[np.floating],
    state_ma_resid_lags: NDArray[np.floating] | None,
    eps_paths: NDArray[np.floating],
) -> NDArray[np.floating]:
    eps_paths, n_sims, horizon = as_sims_by_horizon(eps_paths)

    if mean_kind == "none":
        return eps_paths.copy()

    mu, _, _ = mean_params(params, mean_order)

    if mean_kind == "demean":
        return (mu + eps_paths).copy()

    if mean_kind != "arma":
        raise ValueError(f"Unknown mean_kind: {mean_kind}")

    simulation = MeanSimulator.from_state(
        params=params,
        order=mean_order,
        n_sims=n_sims,
        horizon=horizon,
        series_hist=state_series_hist,
        ma_resid_lags=state_ma_resid_lags,
    )

    simulation_results = np.empty((n_sims, horizon), dtype=float)
    for time in range(horizon):
        y_next = simulation.mean_step(time, eps_paths)
        simulation_results[:, time] = y_next
        simulation.push(time, y_next, eps_paths[:, time])

    return simulation_results


def simulate_asset_paths(
    forecast_model: ForecastModel,
    innovations: NDArray[np.floating],
) -> NDArray[np.floating]:
    model = forecast_model.model
    params = model.compile_params()
    state0 = forecast_model.state0

    innovations, n_sims, horizon = as_sims_by_horizon(innovations)

    if model.vol_kind == "none":
        eps_paths = model.innovation_scale * innovations
    elif model.vol_kind == "garch":
        _, eps_paths = garch_simulation_paths(
            params=params,
            garch_order=model.vol_order,
            eps_start=state0.vol_residual_lags,
            var_start=state0.var_hist,
            innovations=innovations,
        )
    else:
        raise ValueError(f"Unknown vol_kind: {model.vol_kind}")

    y_paths = mean_simulation_paths(
        params=params,
        mean_kind=model.mean_kind,
        mean_order=model.mean_order,
        state_series_hist=state0.series_hist,
        state_ma_resid_lags=state0.ma_residual_lags,
        eps_paths=eps_paths,
    )

    return y_paths

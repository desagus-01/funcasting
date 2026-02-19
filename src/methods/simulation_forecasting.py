import numpy as np
from numpy._typing import NDArray

from methods.forecasting_pipeline import CompiledParams, ForecastModel


def _ensure_2d(
    innovations: NDArray[np.floating],
) -> NDArray[np.floating]:
    innovations = np.asarray(innovations, dtype=float)
    if innovations.ndim == 2:
        return innovations
    if innovations.ndim == 1:
        return innovations.reshape(-1, 1)
    raise ValueError("innovations must be 1D or 2D")


def _as_broadcast_lags(
    lags_1d: NDArray[np.floating] | None, n_sims: int, needed: int, *, name: str
) -> NDArray[np.floating]:
    if needed <= 0:
        return np.empty((n_sims, 0), dtype=float)
    if lags_1d is None:
        raise ValueError(f"Missing {name} lags: need {needed} values.")
    lags_1d = np.asarray(lags_1d, dtype=float).ravel()
    if lags_1d.size < needed:
        raise ValueError(
            f"Insufficient {name} lags: need {needed}, have {lags_1d.size}."
        )
    return np.broadcast_to(lags_1d[-needed:], (n_sims, needed)).copy()


def _validate_horizon_and_get_shape(
    values: NDArray[np.floating],
) -> tuple[NDArray[np.floating], int, int]:
    innovations_for_asset = _ensure_2d(innovations=values)
    n_sims, horizon = innovations_for_asset.shape
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return innovations_for_asset, n_sims, horizon


def _extract_and_validate_garch_params(
    params: CompiledParams, garch_order: tuple[int, int, int]
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    p, o, q = garch_order

    omega = float(params.omega)
    alpha = np.asarray(params.alpha, dtype=float)
    gamma = np.asarray(params.gamma, dtype=float)
    beta = np.asarray(params.beta, dtype=float)

    if alpha.size != p:
        raise ValueError(f"alpha size {alpha.size} != p {p}")
    if gamma.size != o:
        raise ValueError(f"gamma size {gamma.size} != o {o}")
    if beta.size != q:
        raise ValueError(f"beta size {beta.size} != q {q}")

    return omega, alpha, gamma, beta


def _init_garch_lag_buffers(
    *,
    n_sims: int,
    horizon: int,
    p: int,
    o: int,
    q: int,
    eps_start: NDArray[np.floating] | None,
    var_start: NDArray[np.floating] | None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], int, int]:
    eps_lag_needed = max(p, o)
    var_lag_needed = q

    eps_ext = np.empty((n_sims, eps_lag_needed + horizon), dtype=float)
    var_ext = np.empty((n_sims, var_lag_needed + horizon), dtype=float)

    if eps_lag_needed > 0:
        eps_ext[:, :eps_lag_needed] = _as_broadcast_lags(
            eps_start, n_sims, eps_lag_needed, name="eps"
        )
    if var_lag_needed > 0:
        var_ext[:, :var_lag_needed] = _as_broadcast_lags(
            var_start, n_sims, var_lag_needed, name="variance"
        )

    return eps_ext, var_ext, eps_lag_needed, var_lag_needed


def _next_variance(
    *,
    h: int,
    omega: float,
    alpha: NDArray[np.floating],
    gamma: NDArray[np.floating],
    beta: NDArray[np.floating],
    p: int,
    o: int,
    q: int,
    eps_ext: NDArray[np.floating],
    var_ext: NDArray[np.floating],
    eps_lag_needed: int,
    var_lag_needed: int,
) -> NDArray[np.floating]:
    n_sims = eps_ext.shape[0]
    v_next = np.full(n_sims, omega, dtype=float)

    eps_last = eps_lag_needed + h - 1
    var_last = var_lag_needed + h - 1

    if p > 0:
        idx = eps_last - np.arange(p)
        eps_lags = eps_ext[:, idx]
        v_next += (eps_lags * eps_lags) @ alpha

    if o > 0:
        idx = eps_last - np.arange(o)
        eps_lags_o = eps_ext[:, idx]
        ind = (eps_lags_o < 0.0).astype(float)
        v_next += (ind * (eps_lags_o * eps_lags_o)) @ gamma

    if q > 0:
        idx = var_last - np.arange(q)
        var_lags = var_ext[:, idx]
        v_next += var_lags @ beta

    return np.maximum(v_next, 1e-12)


def _roll_forward_buffers(
    *,
    h: int,
    eps_next: NDArray[np.floating],
    v_next: NDArray[np.floating],
    eps_ext: NDArray[np.floating],
    var_ext: NDArray[np.floating],
    eps_lag_needed: int,
    var_lag_needed: int,
) -> None:
    if eps_lag_needed > 0:
        eps_ext[:, eps_lag_needed + h] = eps_next
    if var_lag_needed > 0:
        var_ext[:, var_lag_needed + h] = v_next


def garch_simulation_paths(
    params: CompiledParams,
    garch_order: tuple[int, int, int],
    eps_start: NDArray[np.floating] | None,
    var_start: NDArray[np.floating] | None,
    innovations_for_asset: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    innovations_for_asset, n_sims, horizon = _validate_horizon_and_get_shape(
        innovations_for_asset
    )
    omega, alpha, gamma, beta = _extract_and_validate_garch_params(params, garch_order)

    p, o, q = garch_order
    eps_ext, var_ext, eps_lag_needed, var_lag_needed = _init_garch_lag_buffers(
        n_sims=n_sims,
        horizon=horizon,
        p=p,
        o=o,
        q=q,
        eps_start=eps_start,
        var_start=var_start,
    )

    sigma2_paths = np.empty((n_sims, horizon), dtype=float)
    eps_paths = np.empty((n_sims, horizon), dtype=float)

    for h in range(horizon):
        v_next = _next_variance(
            h=h,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            p=p,
            o=o,
            q=q,
            eps_ext=eps_ext,
            var_ext=var_ext,
            eps_lag_needed=eps_lag_needed,
            var_lag_needed=var_lag_needed,
        )

        eps_next = np.sqrt(v_next) * innovations_for_asset[:, h]

        sigma2_paths[:, h] = v_next
        eps_paths[:, h] = eps_next

        _roll_forward_buffers(
            h=h,
            eps_next=eps_next,
            v_next=v_next,
            eps_ext=eps_ext,
            var_ext=var_ext,
            eps_lag_needed=eps_lag_needed,
            var_lag_needed=var_lag_needed,
        )

    return sigma2_paths, eps_paths


def _extract_and_validate_mean_params(
    params: CompiledParams, arma_order: tuple[int, int]
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    mu = float(params.mu)
    ar = np.asarray(params.ar, dtype=float)
    ma = np.asarray(params.ma, dtype=float)
    p, q = arma_order

    if ar.size != p:
        raise ValueError(f"ar size {ar.size} != p {p}")
    if ma.size != q:
        raise ValueError(f"ma size {ma.size} != q {q}")
    return mu, ar, ma


def _determine_mean_buffers(
    state_series_hist: NDArray[np.floating],
    state_ma_resid_lags: NDArray[np.floating] | None,
    p_order: int,
    q_order: int,
    horizon: int,
    n_sims: int,
) -> tuple[NDArray[np.floating] | None, NDArray[np.floating] | None]:
    if p_order > 0 and state_series_hist.size < p_order:
        raise ValueError(
            f"Need at least {p_order} series lags, have {state_series_hist.size}."
        )

    if p_order > 0:
        y_ext = np.empty((n_sims, p_order + horizon), dtype=float)
        y_ext[:, :p_order] = np.broadcast_to(
            state_series_hist[-p_order:], (n_sims, p_order)
        )
    else:
        y_ext = None

    if q_order > 0:
        if state_ma_resid_lags is None:
            raise ValueError(f"Need {q_order} mean residual lags for MA part.")
        if state_ma_resid_lags.size < q_order:
            raise ValueError(
                f"Need {q_order} mean residual lags, have {state_ma_resid_lags.size}."
            )
        e_ext = np.empty((n_sims, q_order + horizon), dtype=float)
        e_ext[:, :q_order] = np.broadcast_to(
            state_ma_resid_lags[-q_order:], (n_sims, q_order)
        )
    else:
        e_ext = None

    return y_ext, e_ext


def _mean_next(
    ar: NDArray[np.floating],
    ma: NDArray[np.floating],
    e_ext: NDArray[np.floating] | None,
    eps_paths: NDArray[np.floating],
    p: int,
    mu: float,
    q: int,
    h: int,
    y_ext: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    ar_part = 0.0
    if p > 0:
        assert y_ext is not None
        y_last = p + h - 1
        idx_y = y_last - np.arange(p)
        y_lags = y_ext[:, idx_y]
        ar_part = y_lags @ ar

    ma_part = 0.0
    if q > 0:
        assert e_ext is not None
        e_last = q + h - 1
        idx_e = e_last - np.arange(q)
        e_lags = e_ext[:, idx_e]
        ma_part = e_lags @ ma

    return mu + ar_part + ma_part + eps_paths[:, h]


def mean_simulation_paths(
    params: CompiledParams,
    mean_kind: str,
    mean_order: tuple[int, int],
    state_series_hist: NDArray[np.floating],
    state_ma_resid_lags: NDArray[np.floating] | None,
    eps_paths: NDArray[np.floating],
) -> NDArray[np.floating]:
    eps_paths, n_sims, horizon = _validate_horizon_and_get_shape(eps_paths)
    mu, ar, ma = _extract_and_validate_mean_params(params=params, arma_order=mean_order)
    p, q = mean_order

    if mean_kind == "none":
        return eps_paths.copy()
    if mean_kind == "demean":
        return (mu + eps_paths).copy()
    if mean_kind != "arma":
        raise ValueError(f"Unknown mean_kind: {mean_kind}")

    y_paths = np.empty((n_sims, horizon), dtype=float)

    y_ext, e_ext = _determine_mean_buffers(
        state_series_hist=state_series_hist,
        state_ma_resid_lags=state_ma_resid_lags,
        p_order=p,
        q_order=q,
        horizon=horizon,
        n_sims=n_sims,
    )

    for h in range(horizon):
        y_next = _mean_next(
            ar=ar,
            ma=ma,
            e_ext=e_ext,
            eps_paths=eps_paths,
            p=p,
            mu=mu,
            q=q,
            h=h,
            y_ext=y_ext,
        )

        y_paths[:, h] = y_next

        if p > 0:
            assert y_ext is not None
            y_ext[:, p + h] = y_next
        if q > 0:
            assert e_ext is not None
            e_ext[:, q + h] = eps_paths[:, h]

    return y_paths


def simulate_asset_paths(
    forecast_model: ForecastModel, innovations: NDArray[np.floating]
):
    model = forecast_model.model
    model_parameters = model.compile_params()
    state0 = forecast_model.state0
    innovations = _ensure_2d(innovations)
    n_sims, horizon = innovations.shape
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    if model.vol_kind == "none":
        eps_paths = innovations
    elif model.vol_kind == "garch":
        sigma2_paths, eps_paths = garch_simulation_paths(
            params=model_parameters,
            garch_order=model.vol_order,
            eps_start=state0.vol_residual_lags,
            var_start=state0.var_hist,
            innovations_for_asset=innovations,
        )
    else:
        raise ValueError(f"Unknown vol_kind: {model.vol_kind}")

    y_paths = mean_simulation_paths(
        params=model_parameters,
        mean_kind=model.mean_kind,
        mean_order=model.mean_order,
        state_series_hist=state0.series_hist,
        state_ma_resid_lags=state0.ma_residual_lags,
        eps_paths=eps_paths,
    )

    return y_paths

import numpy as np
from numpy._typing import NDArray

from methods.forecasting_pipeline import CompiledParams


def _ensure_innovations_2d(
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


def garch_simulation_paths(
    params: CompiledParams,
    garch_order: tuple[int, int, int],
    eps_start: NDArray[np.floating] | None,
    var_start: NDArray[np.floating] | None,
    innovations_for_asset: NDArray[np.floating],
):
    innovations_for_asset = _ensure_innovations_2d(innovations=innovations_for_asset)
    n_sims, horizon = innovations_for_asset.shape
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

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

    sigma2_paths = np.empty((n_sims, horizon), dtype=float)
    eps_paths = np.empty((n_sims, horizon), dtype=float)

    for h in range(horizon):
        v_next = omega * np.ones(n_sims, dtype=float)

        # last available eps index at this step
        eps_last = eps_lag_needed + h - 1
        var_last = var_lag_needed + h - 1

        if p > 0:
            idx = eps_last - np.arange(p)  # [t-1, t-2, ..., t-p]
            eps_lags = eps_ext[:, idx]  # (n_sims, p)
            v_next += (eps_lags * eps_lags) @ alpha

        if o > 0:
            idx = eps_last - np.arange(o)
            eps_lags_o = eps_ext[:, idx]  # (n_sims, o)
            ind = (eps_lags_o < 0.0).astype(float)
            v_next += (ind * (eps_lags_o * eps_lags_o)) @ gamma

        if q > 0:
            idx = var_last - np.arange(q)  # [t-1, ..., t-q]
            var_lags = var_ext[:, idx]  # (n_sims, q)
            v_next += var_lags @ beta

        eps_next = np.sqrt(v_next) * innovations_for_asset[:, h]

        sigma2_paths[:, h] = v_next
        eps_paths[:, h] = eps_next

        if eps_lag_needed > 0:
            eps_ext[:, eps_lag_needed + h] = eps_next
        if var_lag_needed > 0:
            var_ext[:, var_lag_needed + h] = v_next

    return sigma2_paths, eps_paths

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy._typing import NDArray

from simulation.engines.utils import as_sims_by_horizon, lag_matrix, mean_params
from time_series.models.model_types import CompiledParams


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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy._typing import NDArray

from simulation.engines.utils import (
    as_sims_by_horizon,
    broadcast_last_k_lags,
    garch_params,
    lag_matrix,
)
from time_series.models.model_types import CompiledParams


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

from typing import Sequence, cast

import cvxpy as cp
import numpy as np
from cvxpy import Constraint, Expression
from numpy.typing import NDArray
from portfolio.policy.constraints import (
    DEFAULT_CONSTRAINTS,
    PortfolioConstraint,
)
from portfolio.policy.moments import HorizonMoments


def _covariance_sqrt_factor(
    covariance: NDArray[np.floating],
) -> NDArray[np.floating]:
    covariance = 0.5 * (covariance + covariance.T)

    eigvals, eigvecs = np.linalg.eigh(covariance)
    return np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _structural_constraints(
    trades: Expression,
    weights: Expression,
    previous_weights: NDArray[np.floating] | Expression,
) -> list[Constraint]:
    """Self-financing and horizon-linking constraints."""
    return cast(
        list[Constraint],
        [
            cp.sum(trades) == 0,
            weights == previous_weights + trades,
        ],
    )


class MeanCovMPO:
    def __init__(
        self,
        horizons: int,
        n_assets: int,
        constraints: Sequence[PortfolioConstraint] | None = None,
    ) -> None:
        self.horizons = horizons
        self.n_assets = n_assets
        self.constraints = list(
            DEFAULT_CONSTRAINTS if constraints is None else constraints
        )
        self._build_variables_and_parameters()
        self._build_problem()

    def _build_problem(self) -> None:
        previous_weights: Expression = self.current_weights
        objective_terms: list[Expression] = []
        cvxpy_constraints: list[Constraint] = []

        for h in range(self.horizons):
            trades_h = self.trades[h, :]
            weights_h = self.weights[h, :]

            cvxpy_constraints += _structural_constraints(
                trades_h,
                weights_h,
                previous_weights,
            )

            for constraint in self.constraints:
                cvxpy_constraints += constraint.compile_to_cvxpy(
                    weights_h,
                    trades_h,
                )

            objective_terms += [
                self.mean[h, :] @ weights_h,
                -cp.sum_squares(self.risk_factors[h] @ weights_h),
                -self.transaction_cost * cp.norm1(trades_h),
            ]

            previous_weights = weights_h

        self.problem = cp.Problem(
            cp.Maximize(cp.sum(objective_terms)),
            constraints=cvxpy_constraints,
        )

    def _build_variables_and_parameters(self) -> None:
        self.trades = cp.Variable(
            (self.horizons, self.n_assets),
            name="trades",
        )
        self.weights = cp.Variable(
            (self.horizons, self.n_assets),
            name="weights",
        )

        self.current_weights = cp.Parameter(
            self.n_assets,
            name="current_weights",
        )
        self.mean = cp.Parameter(
            (self.horizons, self.n_assets),
            name="mean",
        )
        self.transaction_cost = cp.Parameter(
            nonneg=True,
            name="transaction_cost",
        )

        self.risk_factors = [
            cp.Parameter(
                (self.n_assets, self.n_assets),
                name=f"risk_factor_{h}",
            )
            for h in range(self.horizons)
        ]

    def _set_values(
        self,
        horizon_moments: HorizonMoments,
        risk_aversion: float,
        current_weights: NDArray[np.floating],
        transaction_cost: float,
    ) -> None:
        self.current_weights.value = current_weights
        self.mean.value = horizon_moments.mean
        self.transaction_cost.value = transaction_cost
        for horizon, covariance in enumerate(horizon_moments.covariances):
            if horizon >= self.horizons:
                break
            self.risk_factors[horizon].value = np.sqrt(
                risk_aversion
            ) * _covariance_sqrt_factor(covariance)

    @staticmethod
    def _clean(w: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        w = np.asarray(w, dtype=float).copy()
        w[np.abs(w) < tol] = 0.0
        w = np.maximum(w, 0.0)

        total = w.sum()

        if total <= tol:
            raise ValueError("Cannot normalise weights because their sum is zero.")

        return w / total

    @staticmethod
    def _readable(w: np.ndarray, assets: Sequence[str]) -> dict[str, float]:
        return dict(zip(assets, np.round(w, 6)))

    def _result(
        self,
        horizon_moments: HorizonMoments,
        current_weights: NDArray[np.floating],
    ) -> dict:
        planned_trades = np.asarray(self.trades.value, dtype=float)
        planned_weights = np.asarray(self.weights.value, dtype=float)

        target_weights = self._clean(planned_weights[0])
        first_trade = target_weights - current_weights

        return {
            "status": self.problem.status,
            "objective_value": self.problem.value,
            "target_weights": target_weights,
            "target_weights_by_asset": self._readable(
                target_weights,
                horizon_moments.assets,
            ),
            "first_trade_weights": first_trade,
            "first_trade_by_asset": self._readable(
                first_trade,
                horizon_moments.assets,
            ),
            "planned_trades": planned_trades,
            "planned_weights": planned_weights,
            "solver_stats": self.problem.solver_stats,
        }

    def solve(
        self,
        horizon_moments: HorizonMoments,
        risk_aversion: float,
        current_weights: NDArray[np.floating],
        transaction_cost: float = 0.005,
        solver: str | None = None,
        **solver_options,
    ) -> dict:
        self._set_values(
            horizon_moments=horizon_moments,
            risk_aversion=risk_aversion,
            current_weights=current_weights,
            transaction_cost=transaction_cost,
        )
        solver_kwargs = {"enforce_dpp": True, "warm_start": True, **solver_options}
        if solver is not None:
            solver_kwargs["solver"] = solver

        self.problem.solve(**solver_kwargs)

        if self.problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"Optimization failed: {self.problem.status}")

        return self._result(
            horizon_moments=horizon_moments,
            current_weights=current_weights,
        )

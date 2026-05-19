from typing import Any, Protocol

import cvxpy as cp


class MPOObjectiveHandler(Protocol):
    def allocate(self, spec, horizons: int, n_assets: int) -> dict[str, Any]:
        """
        Called ONCE at the start of a backtest.

        Create and return all CVXPY Parameter objects this term needs.
        Parameters are sized to the universe (n_assets MUST include cash).

        Example return value:
          {"r_hat": cp.Parameter(n_assets - 1)}
        """

    def compile(
        self,
        spec,
        params: dict[str, Any],
        weights_at_horizon: cp.Expression,
        trades_at_horizong: cp.Expression,
        horizon: int,
    ) -> cp.Expression:
        """
        Called ONCE after allocate()

        Build and return the CVXPY expression for this term.
        Must use cp.Parameter objects (not raw numpy) so the expression
        is DPP-compliant and can be reused across time steps.

        The expression should be written as something to MAXIMISE.
        """

    def update(
        self, spec, params: dict[str, cp.Parameter], inputs: dict[str, Any]
    ) -> None:
        pass

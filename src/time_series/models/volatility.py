import logging
from itertools import product

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from numpy._typing import NDArray

from time_series.models.types import GARCH_DISTRIBUTIONS, AutoGARCHRes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _garch_base_model(
    asset_array: NDArray[np.floating],
    innovation_distribution: GARCH_DISTRIBUTIONS = "t",
    p_order: int = 1,
    o_order: int = 0,
    q_order: int = 1,
) -> ARCHModelResult:
    base_model = arch_model(
        y=asset_array,
        mean="zero",
        p=p_order,
        o=o_order,
        q=q_order,
        dist=innovation_distribution,
        rescale=False,
    ).fit(disp=False)
    return base_model


def _garch_persistence_calc(params: dict[str, float]) -> float:
    alpha_sum = sum(v for k, v in params.items() if k.startswith("alpha"))
    beta_sum = sum(v for k, v in params.items() if k.startswith("beta"))
    gamma_sum = sum(v for k, v in params.items() if k.startswith("gamma"))
    return alpha_sum + beta_sum + 0.5 + gamma_sum


def _garch_boundaries_check(
    params: dict[str, float],
    tolerance_zero: float = 1e-10,
    tolerance_dups: float = 1e-6,
) -> bool:
    vals = {k: float(v) for k, v in params.items()}

    # zero higher order lags
    for k, v in vals.items():
        if (
            k.startswith("alpha") or k.startswith("beta") or k.startswith("gamma")
        ) and abs(v) < tolerance_zero:
            return True

    # duplicated betas mean weak identification
    betas = [v for k, v in vals.items() if k.startswith("beta")]
    if len(betas) >= 2:
        for i in range(len(betas)):
            for j in range(i + 1, len(betas)):
                if abs(betas[i] - betas[j]) < tolerance_dups:
                    return True

    return False


def _admissable_garch_model(
    params: dict[str, float], max_persistence: float = 0.98
) -> bool:
    persistence = _garch_persistence_calc(params)

    if persistence >= max_persistence:
        return False
    if _garch_boundaries_check(params):
        return False

    omega = params.get("omega", 0.0)
    if omega <= 0:
        return False

    return True


def auto_garch(
    asset_array: NDArray[np.floating],
    max_p_order: int = 2,
    max_o_order: int = 1,
    max_q_order: int = 2,
) -> list[AutoGARCHRes]:
    base_model = _garch_base_model(asset_array=asset_array)
    dists: tuple[GARCH_DISTRIBUTIONS, GARCH_DISTRIBUTIONS] = ("t", "normal")
    garch_candidates = []
    for p, q, o, distribution in product(
        range(1, max_p_order + 1),
        range(1, max_q_order + 1),
        range(0, max_o_order + 1),
        dists,
    ):
        key = (p, o, q)
        if (key == (1, 0, 1)) and (distribution == "t"):
            continue

        proposed_model = arch_model(
            asset_array, mean="zero", p=p, o=o, q=q, dist=distribution, rescale=False
        ).fit(disp="off")
        if proposed_model.convergence_flag != 0:
            print(
                "NO CONVERGE",
                key,
                distribution,
                "flag",
                proposed_model.convergence_flag,
            )
            continue

        if proposed_model.bic < base_model.bic:
            params = proposed_model.params.to_dict()  # type: ignore[attr-defined]
            if not _admissable_garch_model(params):
                continue
            garch_candidates.append(
                AutoGARCHRes(
                    model_order=key,
                    degrees_of_freedom=len(proposed_model.params),
                    criteria="bic",
                    criteria_res=proposed_model.bic,
                    params=proposed_model.params.to_dict(),  # type: ignore[attr-defined]
                    p_values=proposed_model.pvalues,  # type: ignore[attr-defined]
                    invariants=proposed_model.std_resid,  # type: ignore[attr-defined]
                    residuals=proposed_model.resid,  # type: ignore[attr-defined]
                    conditional_volatility=proposed_model.conditional_volatility,  # type: ignore[attr-defined]
                )
            )

    garch_candidates.append(
        AutoGARCHRes(
            model_order=(1, 0, 1),
            degrees_of_freedom=len(base_model.params),
            criteria="bic",
            criteria_res=base_model.bic,
            params=base_model.params.to_dict(),  # type: ignore[attr-defined]
            p_values=base_model.pvalues,  # type: ignore[attr-defined]
            residuals=base_model.resid,  # type: ignore[attr-defined]
            invariants=base_model.std_resid,  # type: ignore[attr-defined]
            conditional_volatility=base_model.conditional_volatility,  # type: ignore[attr-defined]
        )
    )

    return garch_candidates

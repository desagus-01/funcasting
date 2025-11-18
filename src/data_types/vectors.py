from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame
from pydantic import AfterValidator, BaseModel, ConfigDict


def _as_prob_vector(a: NDArray[np.float64]) -> NDArray[np.float64]:
    if a.ndim != 1:
        raise ValueError("Array must be 1D.")
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        raise ValueError("Array must not contain NaN or infinite values.")
    if np.any(a < 0):
        raise ValueError("All probabilities must be non-negative.")
    if not np.isclose(a.sum(dtype=np.float64), 1.0, rtol=0, atol=1e-5):
        raise ValueError(
            f"Probabilities must sum to 1. Currently this is {a.sum(dtype=np.float64)}"
        )
    return a


class ConstraintSigns(str, Enum):
    equal_greater = "equal_greater"
    equal_less = "equal_less"
    equal = "equal"


ConstraintSignLike: TypeAlias = (
    ConstraintSigns | Literal["equal_greater", "equal_less", "equal"]
)


class View(BaseModel):
    """
    Allows to create a view on a single scenario
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str
    risk_driver: tuple[str, str] | str
    data: NDArray[np.floating]
    views_target: NDArray[np.floating] | None
    sign_type: ConstraintSignLike
    mean_ref: NDArray[np.floating] | None = None


class CorrInfo(BaseModel):
    asset_pair: tuple[str, str]
    corr: float


ProbVector = Annotated[NDArray[np.float64], AfterValidator(_as_prob_vector)]


@dataclass
class CMASeparation:
    marginals: DataFrame
    cdfs: DataFrame
    copula: DataFrame
    posterior: ProbVector


model_cfg = ConfigDict(arbitrary_types_allowed=True)

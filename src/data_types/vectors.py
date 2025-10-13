from enum import Enum
from typing import Annotated, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator


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


class ConstraintType(str, Enum):
    equality = "equality"
    inequality = "inequality"


class ConstraintSigns(str, Enum):
    equal_greater = "equal_greater"
    equal_less = "equal_less"
    equal = "equal"


ConstraintTypeLike: TypeAlias = ConstraintType | Literal["equality", "inequality"]
ConstraintSignLike: TypeAlias = (
    ConstraintSigns | Literal["equal_greater", "equal_less", "equal"]
)


class View(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    risk_driver: str
    data: NDArray[np.floating]
    views_target: NDArray[np.floating]
    const_type: ConstraintTypeLike
    sign_type: ConstraintSignLike

    @model_validator(mode="after")
    def _check_constraint_logic(self) -> None:
        if (
            self.const_type == ConstraintType.equality
            and self.sign_type != ConstraintSigns.equal
        ):
            raise ValueError("For equality constraints, sign_type must be 'equal'.")
        if (
            self.const_type == ConstraintType.inequality
            and self.sign_type == ConstraintSigns.equal
        ):
            raise ValueError("For inequality constraints, sign_type cannot be 'equal'.")
        return self


ProbVector = Annotated[NDArray[np.float64], AfterValidator(_as_prob_vector)]
model_cfg = ConfigDict(arbitrary_types_allowed=True)

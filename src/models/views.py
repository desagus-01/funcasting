from enum import Enum
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class ConstraintSigns(str, Enum):
    equal_greater = "equal_greater"
    equal_less = "equal_less"
    equal = "equal"


ConstraintSignLike: TypeAlias = (
    ConstraintSigns | Literal["equal_greater", "equal_less", "equal"]
)


class CorrInfo(BaseModel):
    asset_pair: tuple[str, str]
    corr: float


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

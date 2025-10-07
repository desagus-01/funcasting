from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator, BaseModel, ConfigDict


def _as_prob_vector(a: NDArray[np.float64]) -> NDArray[np.float64]:
    if a.ndim != 1:
        raise ValueError("Array must be 1D.")
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        raise ValueError("Array must not contain NaN or infinite values.")
    if np.any(a < 0):
        raise ValueError("All probabilities must be non-negative.")
    if not np.isclose(a.sum(dtype=np.float64), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Probabilities must sum to 1.")
    return a


class View(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: NDArray[np.floating]
    views_targets: NDArray[np.floating]
    const_type: str
    sign_type: str | None


ProbVector = Annotated[NDArray[np.float64], AfterValidator(_as_prob_vector)]
model_cfg = ConfigDict(arbitrary_types_allowed=True)

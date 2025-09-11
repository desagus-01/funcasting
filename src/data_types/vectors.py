import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ProbabilityArray(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    probs: NDArray[np.float64]

    @field_validator("probs")
    @classmethod
    def check_is_numpy_and_1d(cls, v):
        if not isinstance(v, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if v.ndim != 1:
            raise ValueError("Array must be 1D.")
        return v

    @field_validator("probs")
    @classmethod
    def check_non_negative(cls, v):
        if np.any(v < 0):
            raise ValueError("All probabilities must be non-negative.")
        return v

    @model_validator(mode="after")
    def check_sum_to_one(self):
        if not np.isclose(self.probs.sum(), 1.0):
            raise ValueError(f"Probabilities must sum to 1")
        return self

import operator as _op

from pydantic import ConfigDict

from models.types import ConstraintSigns

sign_operations = {
    (ConstraintSigns.equal): _op.eq,
    "equal": _op.eq,
    (ConstraintSigns.equal_greater): _op.ge,
    "equal_greater": _op.ge,
    (ConstraintSigns.equal_less): _op.le,
    "equal_less": _op.le,
}

DEFAULT_ROUNDING = 4

ITERS = {"PERM_TEST": 50, "MC": 10_000}

SIGN_LVL = 0.05

model_cfg = ConfigDict(arbitrary_types_allowed=True)

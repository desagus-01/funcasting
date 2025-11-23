import operator as _op

from models.views import ConstraintSigns

sign_operations = {
    (ConstraintSigns.equal): _op.eq,
    "equal": _op.eq,
    (ConstraintSigns.equal_greater): _op.ge,
    "equal_greater": _op.ge,
    (ConstraintSigns.equal_less): _op.le,
    "equal_less": _op.le,
}

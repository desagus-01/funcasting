import operator as _op

from numpy import inf
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

LAGS = {"testing": 2, "strict": 10}

model_cfg = ConfigDict(arbitrary_types_allowed=True)


# INFO: Below tau related stuff is taken straight from statsmodels
# These are the new estimates from MacKinnon 2010
# the first axis is N -1
# the second axis is 1 %, 5 %, 10 %
# the last axis is the coefficients

MACKIN_TAU_CUTOFFS = {
    "star_nc": [-1.04, -1.53, -2.68, -3.09, -3.07, -3.77],
    "min_nc": [-19.04, -19.62, -21.21, -23.25, -21.63, -25.74],
    "max_nc": [inf, 1.51, 0.86, 0.88, 1.05, 1.24],
    "star_c": [-1.61, -2.62, -3.13, -3.47, -3.78, -3.93],
    "min_c": [-18.83, -18.86, -23.48, -28.07, -25.96, -23.27],
    "max_c": [2.74, 0.92, 0.55, 0.61, 0.79, 1],
    "star_ct": [-2.89, -3.19, -3.50, -3.65, -3.80, -4.36],
    "min_ct": [-16.18, -21.15, -25.37, -26.63, -26.53, -26.18],
    "max_ct": [0.7, 0.63, 0.71, 0.93, 1.19, 1.42],
    "star_ctt": [-3.21, -3.51, -3.81, -3.83, -4.12, -4.63],
    "min_ctt": [-17.17, -21.1, -24.33, -24.03, -24.33, -28.22],
    "max_ctt": [0.54, 0.79, 1.0],
}

import math
from dataclasses import dataclass
from typing import TypedDict

from globals import DEFAULT_ROUNDING, SIGN_LVL


@dataclass(frozen=True, slots=True)
class HypTestRes:
    stat: float
    p_val: float
    sign_lvl: float
    null: str
    reject_null: bool
    desc: str


class HypTestConclusion(TypedDict):
    reject_null: bool
    desc: str


def hyp_test_conclusion(p_val: float, null_hyp: str) -> HypTestConclusion:
    if p_val >= SIGN_LVL:
        return {
            "reject_null": False,
            "desc": (
                f"Fail to reject null hypothesis of {null_hyp} at {SIGN_LVL} significance."
            ),
        }
    else:
        return {
            "reject_null": True,
            "desc": f"Reject null hypothesis of {null_hyp} at {SIGN_LVL} significance.",
        }


def format_hyp_test_result(
    stat: float, p_val: float, null: str = "Independence"
) -> HypTestRes:
    try:
        stat = float(stat)
        p_val = float(p_val)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid stat or p_val: stat={stat}, p_val={p_val}")

    if math.isnan(stat) or math.isnan(p_val):
        raise ValueError(f"Invalid stat or p_val: stat={stat}, p_val={p_val}")

    hyp_conc = hyp_test_conclusion(p_val, null_hyp=null)

    return HypTestRes(
        stat=round(stat, DEFAULT_ROUNDING),
        p_val=round(p_val, DEFAULT_ROUNDING),
        sign_lvl=SIGN_LVL,
        null=null,
        reject_null=hyp_conc["reject_null"],
        desc=hyp_conc["desc"],
    )

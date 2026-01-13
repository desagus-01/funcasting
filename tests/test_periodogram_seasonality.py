import numpy as np
import pytest

import maths.stochastic_processes.seasonality as seas

SEASONAL_LABELS = ["weekly", "monthly", "quarterly", "semi-annual", "annual"]
SEASONAL_MAP = {
    "weekly": 5,
    "monthly": 21,
    "quarterly": 63,
    "semi-annual": 126,
    "annual": 252,
}


def _run_periodogram_test(y: np.ndarray, label: str):
    """
    Calls the user's function. Accepts either:
      - seas.periodogram_seasonality_test(...)
      - seas.periodogram_Seasonality_test(...)  # if someone used a different style
    Returns the result object/dict as-is.
    """
    fn = getattr(seas, "periodogram_seasonality_test", None) or getattr(
        seas, "periodogram_Seasonality_test", None
    )
    if fn is None:
        raise RuntimeError(
            "Could not find periodogram_seasonality_test in 'seas' module."
        )
    return fn(data=y, seasonal_period=label)


def _get_field(obj, name, default=None):
    """Helper to support either dict-like or attr-like results."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


@pytest.fixture(scope="module")
def synthetic_series() -> np.ndarray:
    """
    Construct a signal containing all requested seasonalities on the Fourier grid,
    plus AR(1)-flavored noise. Length LCM(5,21,63,126,252)=1260.
    """
    n = 1260
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(20250112)

    # amplitudes and phases (tuned so all are detectable and not collinear)
    comps = {
        "weekly": (8.0, SEASONAL_MAP["weekly"], 0.35),
        "monthly": (6.0, SEASONAL_MAP["monthly"], -1.10),
        "quarterly": (5.0, SEASONAL_MAP["quarterly"], 2.00),
        "semi-annual": (4.5, SEASONAL_MAP["semi-annual"], 0.75),
        "annual": (3.5, SEASONAL_MAP["annual"], -2.20),
    }

    x = np.zeros(n, dtype=float)
    for amp, period, phase in comps.values():
        x += amp * np.sin(2.0 * np.pi * t / period + phase)

    # AR(1) noise with moderate variance
    e = rng.normal(0.0, 2.0, size=n)
    for i in range(1, n):
        e[i] += 0.35 * e[i - 1]

    y = x + e
    y -= y.mean()  # remove DC
    return y


@pytest.mark.parametrize("label", SEASONAL_LABELS)
def test_periodogram_detects_each_seasonality(synthetic_series, label):
    res = _run_periodogram_test(synthetic_series, label)
    reject = _get_field(res, "reject_null")
    p_val = _get_field(res, "p_val", _get_field(res, "p_value"))
    stat = _get_field(res, "stat")

    assert reject is True, (
        f"{label}: expected reject_null=True but got False (p={p_val}, stat={stat})"
    )

    assert p_val is None or p_val < 1e-6, (
        f"{label}: p-value not sufficiently small (p={p_val})"
    )

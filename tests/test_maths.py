import numpy as np

from maths.non_parametric import exp_decay_probs


def test_exp_decay_probs_sum_to_one():
    ex = np.random.rand(1, 50)
    res = exp_decay_probs(ex, 50)

    assert np.isclose(res.sum(), 1.0)

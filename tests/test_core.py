import numpy as np
import polars as pl
import pytest

from methods.cma import CopulaMarginalModel
from methods.ep import entropy_pooling_probs
from models.types import _as_prob_vector
from models.views_builder import ViewBuilder
from utils.distributions import uniform_probs


def test_probvector_basic():
    p = _as_prob_vector(np.array([0.2, 0.3, 0.5]))
    assert p.shape == (3,)
    assert np.isclose(p.sum(), 1.0)

    with pytest.raises(ValueError):
        _as_prob_vector(np.array([-0.1, 1.1]))  # negative

    with pytest.raises(ValueError):
        _as_prob_vector(np.array([[0.5, 0.5]]))  # 2D


def test_uniform_probs():
    p = uniform_probs(5)
    assert p.shape == (5,)
    assert np.all(p >= 0)
    assert np.isclose(p.sum(), 1.0)


def test_entropy_pooling_changes_mean_simple_case():
    scenarios = np.array([0.0, 1.0, 2.0])
    prior = uniform_probs(3)
    df = pl.DataFrame({"X": [0.0, 1.0, 2.0]})
    vb = ViewBuilder(df)
    views = vb.mean(target_means={"X": 1.0}, sign_type=["equal"]).build()

    post = entropy_pooling_probs(prior, views)
    new_mean = np.average(scenarios, weights=post)
    assert np.isclose(new_mean, 1.8, atol=1e-2)


def test_cma_round_trip_shapes_and_probs():
    df = pl.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 1.0, 1.0]})
    p = uniform_probs(df.height)

    model = CopulaMarginalModel.from_scenario_dist(df, p)
    df2, p2 = model.to_scenario_dist()

    assert df2.shape == df.shape
    assert np.isclose(p2.sum(), 1.0)
    assert p2.shape == p.shape

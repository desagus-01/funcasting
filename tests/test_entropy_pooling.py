"""
Tests for scenarios.entropy_pooling.

Covers:
- compile_view: correct lhs/rhs construction and constraint type per view type
- compile_view: validation errors on missing views_target
- diagnose_view: active flag correct after solve
- get_constraints_diags: delegates to diagnose_view cleanly
- _build_constraints: returns right number of CompiledViews + simplex constraints
- entropy_pooling: end-to-end mean, std, sorting, quantile views
- ens / clip_normalise_probs utilities
- Precedence fix: diagnose_view handles array-valued constraints without raising
"""

import numpy as np
import pytest

from scenarios.entropy_pooling import (
    CompiledView,
    _build_constraints,
    clip_normalise_probs,
    compile_view,
    diagnose_view,
    ens,
    entropy_pooling,
    get_constraints_diags,
)
from scenarios.types import View

SCENARIOS_N = 20


@pytest.fixture
def uniform_prior() -> np.ndarray:
    p = np.ones(SCENARIOS_N) / SCENARIOS_N
    return p.astype(np.float64)


def _make_view(**kwargs) -> View:
    """Helper: build a View with sensible defaults."""
    defaults = dict(
        type="mean",
        risk_driver="asset_a",
        data=np.linspace(-1, 1, SCENARIOS_N),
        views_target=np.array([0.0]),
        sign_type="equal",
        mean_ref=None,
    )
    defaults.update(kwargs)
    return View(**defaults)


class TestCompileView:
    def test_mean_lhs_rhs_shapes(self, uniform_prior):
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        view = _make_view(
            type="mean",
            data=np.linspace(-1, 1, SCENARIOS_N),
            views_target=np.array([0.1]),
        )
        cv = compile_view(view, posterior, uniform_prior)

        assert isinstance(cv, CompiledView)
        assert cv.lhs is not None
        assert cv.constraint is not None

    def test_sorting_rhs_is_expression(self, uniform_prior):
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        data = np.vstack(
            [np.linspace(0, 1, SCENARIOS_N), np.linspace(1, 2, SCENARIOS_N)]
        )
        view = _make_view(
            type="sorting", data=data, views_target=None, sign_type="equal_greater"
        )
        cv = compile_view(view, posterior, uniform_prior)

        # rhs must be a cp.Expression for sorting (data[1] @ posterior)
        assert isinstance(cv.rhs, cp.Expression)

    def test_std_view_compiles(self, uniform_prior):
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        data = np.linspace(-1, 1, SCENARIOS_N)
        view = _make_view(type="std", data=data, views_target=np.array([0.3]))
        cv = compile_view(view, posterior, uniform_prior)
        assert cv.constraint is not None

    def test_quantile_view_compiles(self, uniform_prior):
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        indicator = (np.linspace(0, 1, SCENARIOS_N) <= 0.5).astype(float)
        view = _make_view(type="quantile", data=indicator, views_target=np.array([0.5]))
        cv = compile_view(view, posterior, uniform_prior)
        assert cv.constraint is not None

    def test_corr_view_compiles(self, uniform_prior):
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        xs = np.linspace(-1, 1, SCENARIOS_N)
        ys = xs + np.random.default_rng(0).normal(0, 0.1, SCENARIOS_N)
        data = np.vstack([xs, ys])
        view = _make_view(
            type="corr",
            data=data,
            views_target=np.array([0.9]),
            sign_type="equal",
            risk_driver=("asset_a", "asset_b"),
        )
        cv = compile_view(view, posterior, uniform_prior)
        assert cv.constraint is not None

    def test_unsupported_type_raises(self, uniform_prior):
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        view = _make_view(type="bogus")
        with pytest.raises(ValueError, match="Unsupported constraint type"):
            compile_view(view, posterior, uniform_prior)

    @pytest.mark.parametrize("view_type", ["mean", "std", "quantile"])
    def test_none_target_raises(self, uniform_prior, view_type):
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        view = _make_view(type=view_type, views_target=None)
        with pytest.raises(ValueError, match="views_target cannot be None"):
            compile_view(view, posterior, uniform_prior)


class TestBuildConstraints:
    def test_returns_one_compiled_view_per_input(self, uniform_prior):
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        views = [
            _make_view(
                type="mean",
                data=np.linspace(-1, 1, SCENARIOS_N),
                views_target=np.array([0.0]),
            ),
            _make_view(
                type="mean",
                data=np.linspace(0, 2, SCENARIOS_N),
                views_target=np.array([1.0]),
            ),
        ]
        compiled, all_constraints = _build_constraints(views, posterior, uniform_prior)

        assert len(compiled) == 2
        # all_constraints = 2 view + 2 simplex
        assert len(all_constraints) == 4

    def test_simplex_constraints_included(self, uniform_prior):
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        views = [_make_view()]
        _, all_constraints = _build_constraints(views, posterior, uniform_prior)

        # Total = 1 view + 2 simplex
        assert len(all_constraints) == 3


def _solve_mean_view(prior, target):
    """Run EP with a single mean equality view; return compiled view."""
    import cvxpy as cp

    from scenarios.entropy_pooling import _build_constraints

    posterior = cp.Variable(SCENARIOS_N)
    data = np.linspace(-1, 1, SCENARIOS_N)
    view = _make_view(type="mean", data=data, views_target=np.array([target]))
    compiled_views, all_constraints = _build_constraints([view], posterior, prior)
    obj = cp.Minimize(cp.sum(cp.kl_div(posterior, prior)))
    prob = cp.Problem(obj, all_constraints)
    prob.solve(solver="SCS")
    return compiled_views[0]


class TestDiagnoseView:
    def test_active_true_for_satisfied_equality(self, uniform_prior):
        # Mean of linspace(-1,1) under uniform = 0; target=0 → active
        cv = _solve_mean_view(uniform_prior, target=0.0)
        diag = diagnose_view(cv)
        assert diag["active"] is True

    def test_active_false_when_constraint_not_binding(self, uniform_prior):
        """
        Use an inequality (equal_less) with a target well above the prior mean.
        The constraint is slack → active should be False.
        """
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        data = np.linspace(-1, 1, SCENARIOS_N)
        # prior mean ≈ 0, target=0.9 → mean <= 0.9 is slack
        view = _make_view(
            type="mean", data=data, views_target=np.array([0.9]), sign_type="equal_less"
        )
        compiled_views, all_constraints = _build_constraints(
            [view], posterior, uniform_prior
        )
        obj = cp.Minimize(cp.sum(cp.kl_div(posterior, uniform_prior)))
        prob = cp.Problem(obj, all_constraints)
        prob.solve(solver="SCS")

        diag = diagnose_view(compiled_views[0])
        assert diag["active"] is False

    def test_sensitivity_flipped_for_equal_greater(self, uniform_prior):
        """equal_greater sensitivity = -dual_value (the sign flip)."""
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        data = np.linspace(-1, 1, SCENARIOS_N)
        view = _make_view(
            type="mean",
            data=data,
            views_target=np.array([-0.3]),
            sign_type="equal_greater",
        )
        compiled_views, all_constraints = _build_constraints(
            [view], posterior, uniform_prior
        )
        obj = cp.Minimize(cp.sum(cp.kl_div(posterior, uniform_prior)))
        prob = cp.Problem(obj, all_constraints)
        prob.solve(solver="SCS")

        cv = compiled_views[0]
        diag = diagnose_view(cv)
        raw_dual = float(np.asarray(cv.constraint.dual_value).flat[0])
        assert diag["sensitivity"] == pytest.approx(-raw_dual, abs=1e-10)

    def test_array_valued_constraint_no_raise(self, uniform_prior):
        """
        Regression: old code did abs(arr <= tol) → bool(bool_array) → ValueError
        for array-shaped residuals.  New code must not raise.
        """
        cv = _solve_mean_view(uniform_prior, target=0.0)
        # Should not raise regardless of residual shape
        diag = diagnose_view(cv)
        assert isinstance(diag["active"], bool)

    def test_get_constraints_diags_length(self, uniform_prior):
        import cvxpy as cp

        posterior = cp.Variable(SCENARIOS_N)
        views = [
            _make_view(
                type="mean",
                data=np.linspace(-1, 1, SCENARIOS_N),
                views_target=np.array([0.0]),
            ),
            _make_view(
                type="mean",
                data=np.linspace(0, 2, SCENARIOS_N),
                views_target=np.array([1.0]),
            ),
        ]
        compiled, all_constraints = _build_constraints(views, posterior, uniform_prior)
        obj = cp.Minimize(cp.sum(cp.kl_div(posterior, uniform_prior)))
        cp.Problem(obj, all_constraints).solve(solver="SCS")

        diags = get_constraints_diags(compiled)
        assert len(diags) == 2
        assert all("active" in d and "sensitivity" in d for d in diags)


class TestEntropyPooling:
    def test_mean_view_satisfied(self, uniform_prior):
        target = 0.3
        data = np.linspace(-1, 1, SCENARIOS_N)
        view = _make_view(type="mean", data=data, views_target=np.array([target]))
        posterior = entropy_pooling(uniform_prior, [view])

        assert posterior.sum() == pytest.approx(1.0, abs=1e-4)
        assert np.all(posterior >= 0)
        assert data @ posterior == pytest.approx(target, abs=1e-3)

    def test_sorting_view_satisfied(self, uniform_prior):
        # data[0] @ posterior >= data[1] @ posterior
        rng = np.random.default_rng(42)
        d0 = rng.standard_normal(SCENARIOS_N)
        d1 = d0 - 1.0  # d0 typically has higher mean
        data = np.vstack([d0, d1])
        view = _make_view(
            type="sorting", data=data, views_target=None, sign_type="equal_greater"
        )
        posterior = entropy_pooling(uniform_prior, [view])

        assert d0 @ posterior >= d1 @ posterior - 1e-4

    def test_posterior_sums_to_one(self, uniform_prior):
        view = _make_view(
            type="mean",
            data=np.linspace(-1, 1, SCENARIOS_N),
            views_target=np.array([0.0]),
        )
        posterior = entropy_pooling(uniform_prior, [view])
        assert posterior.sum() == pytest.approx(1.0, abs=1e-5)

    def test_infeasible_raises(self, uniform_prior):
        """Contradictory views should cause EP to fail."""
        data = np.linspace(-1, 1, SCENARIOS_N)
        views = [
            _make_view(type="mean", data=data, views_target=np.array([0.5])),
            _make_view(type="mean", data=data, views_target=np.array([-0.5])),
        ]
        with pytest.raises(RuntimeError):
            entropy_pooling(uniform_prior, views)

    def test_std_view_satisfied(self, uniform_prior):
        data = np.linspace(-1, 1, SCENARIOS_N)
        target_std = 0.4
        view = _make_view(type="std", data=data, views_target=np.array([target_std]))
        posterior = entropy_pooling(uniform_prior, [view])
        mu_post = data @ posterior
        var_post = (data**2) @ posterior - mu_post**2
        assert np.sqrt(var_post) == pytest.approx(target_std, abs=5e-3)


class TestEns:
    def test_uniform_ens_equals_n(self):
        # exp(log(N)) = N exactly; ceil(N) = N
        p = np.ones(SCENARIOS_N) / SCENARIOS_N
        assert ens(p) == SCENARIOS_N

    def test_skewed_ens_less_than_n(self):
        # Concentrating mass reduces ENS below N
        p = np.zeros(SCENARIOS_N)
        p[0] = 0.9
        p[1:] = 0.1 / (SCENARIOS_N - 1)
        assert ens(p) < SCENARIOS_N
        assert ens(p) >= 1

    def test_degenerate_raises_due_to_nan(self):
        p = np.zeros(SCENARIOS_N)
        p[0] = 1.0
        with pytest.raises((ValueError, RuntimeError)):
            ens(p)


class TestClipNormaliseProbs:
    def test_clips_small_negatives(self):
        p = np.array([0.5, -1e-9, 0.5])
        result = clip_normalise_probs(p)
        assert np.all(result >= 0)
        assert result.sum() == pytest.approx(1.0, abs=1e-10)

    def test_raises_on_collapsed_posterior(self):
        p = np.array([-1.0, -1.0, -1.0])
        with pytest.raises(RuntimeError, match="collapsed"):
            clip_normalise_probs(p)

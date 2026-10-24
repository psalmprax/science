"""Smoke tests for the five modules nothing in the suite can currently reach.

A coverage audit of the full run found 88 non-test modules unreachable from
``apedp.model_synthesis``. Eighty-one of them are genuinely exercised. The
remaining five had *zero* covered lines, and the reason was not that they were
untested by accident: an AST scan for import sites found **no importer
anywhere in the repository**. They were dead in the strictest sense -- the
suite could not fail on them even in principle.

``closed_loop`` and ``experiment_designer`` are a linked pair. ``closed_loop``
lazily imports ``experiment_designer`` inside ``_design`` and inside
``semf_validation``, so with nothing importing ``closed_loop``, the designer is
unreachable even indirectly. Covering the loop is the only route to the
designer; testing the designer alone would leave the pair broken.

The failure mode here is the one an audit is for: a module that could stop
working tomorrow, and every test would still pass. These are behavioural
tests on the parts whose contract is a stated invariant, not import smoke --
``import`` alone would not catch a wrong answer.

Scope is deliberately narrow: pure, offline, deterministic. Nothing here
reaches the network, spawns a solver, reads a corpus, or runs a search.
"""
from __future__ import annotations

import numpy as np
import pytest


class TestVariableSemantics:
    """apedp/variable_semantics.py -- symbol disambiguation."""

    def test_bare_ambiguous_symbols_are_flagged_not_resolved(self) -> None:
        """The point of the module: ``n`` must never silently mean n_e."""
        from apedp.variable_semantics import check_variable_semantics

        out = check_variable_semantics({"n", "T_e"})
        assert out["ambiguous_bare"] == ["n"], out
        # A bare symbol is a warning, never a resolution.
        assert "n" not in out["resolved"], out
        assert out["resolved"]["T_e"].unit == "keV", out

    def test_qualified_symbols_resolve_to_their_spec(self) -> None:
        from apedp.variable_semantics import check_variable_semantics

        out = check_variable_semantics({"n_e", "E_n"})
        assert set(out["resolved"]) == {"n_e", "E_n"}, out
        assert out["ambiguous_bare"] == [], out
        # neutron, explicitly not neutrino -- the disambiguation this exists for
        assert out["resolved"]["E_n"].name == "neutron energy", out

    def test_registry_separates_unresolved_from_ambiguous(self) -> None:
        """Three outcomes, not two: resolved / ambiguous / unknown."""
        from apedp.variable_semantics import VariableRegistry

        out = VariableRegistry().check({"n_e", "n", "zzz"})
        assert set(out["resolved"]) == {"n_e"}, out
        assert out["ambiguous_bare"] == ["n"], out
        # An unknown symbol is its own category and must not be lumped in
        # with the ambiguous ones, or "we have never heard of zzz" would read
        # as "zzz is a known-but-ambiguous symbol".
        assert out["unresolved"] == ["zzz"], out

    def test_registering_a_domain_adds_resolution(self) -> None:
        from apedp.variable_semantics import VariableRegistry, VariableSpec

        reg = VariableRegistry()
        reg.register(VariableSpec("k", "spring constant", "newton/meter", "stiffness"))
        out = reg.check({"k"})
        assert out["resolved"]["k"].unit == "newton/meter", out
        assert out["unresolved"] == [], out

    def test_assumption_set_reports_only_what_is_missing(self) -> None:
        from apedp.variable_semantics import AssumptionSet

        s = AssumptionSet()
        s.add("steady_state", "true")
        assert s.missing(["steady_state", "Maxwellian"]) == ["Maxwellian"], s.items

    def test_applicability_envelope_is_about_untested_range(self) -> None:
        from apedp.variable_semantics import ApplicabilityEnvelope

        env = ApplicabilityEnvelope()
        assert env.covers("x") is False
        env.ranges["x"] = "0-10"
        assert env.covers("x") is True
        # ``tested`` is tracked separately from ``ranges``: a declared range is
        # not evidence the range was exercised.
        assert env.tested == {}, env.tested


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, callable]:
    """A ``fit`` callable matching rolling_origin_identifiability's contract."""
    slope = float(np.sum(x * y) / np.sum(x * x))
    return slope, (lambda xs, s=slope: s * np.asarray(xs, dtype=float))


class TestIdentifiability:
    """apedp/identifiability.py -- predictive fit vs parameter stability."""

    def test_stable_parameter_is_reported_stable(self) -> None:
        from apedp.identifiability import rolling_origin_identifiability

        x = np.arange(1.0, 21.0)
        y = 2.0 * x
        rep = rolling_origin_identifiability(x, y, [8, 12, 16], _linear_fit, horizon=2)
        assert rep.status == "STABLE", rep.status
        # A good forecast is explicitly not a uniqueness proof; the module
        # says so in its own output and the test pins that wording.
        assert "not a formal uniqueness proof" in rep.interpretation, rep.interpretation
        assert len(rep.points) == 3, rep.points

    def test_origin_sensitive_parameter_is_poorly_identified(self) -> None:
        """The failure the module exists to catch: great fit, unstable slope."""
        from apedp.identifiability import rolling_origin_identifiability

        x = np.arange(1.0, 21.0)
        y = 2.0 * x
        y[:10] *= 50.0  # early points pull the slope far from the rest
        rep = rolling_origin_identifiability(x, y, [8, 12, 16], _linear_fit, horizon=2)
        assert rep.status == "POORLY_IDENTIFIED", (rep.status, rep.interpretation)
        assert rep.interpretation.startswith("Forecast quality and parameter"), rep.interpretation
        assert rep.parameter_relative_range > 0.1, rep.parameter_relative_range

    def test_report_round_trips_through_to_dict(self) -> None:
        from apedp.identifiability import rolling_origin_identifiability

        x = np.arange(1.0, 21.0)
        rep = rolling_origin_identifiability(x, 2.0 * x, [8, 12, 16], _linear_fit, horizon=2)
        d = rep.to_dict()
        assert d["status"] == "STABLE", d
        assert len(d["points"]) == 3, d
        assert {"parameter", "fit_score", "forecast_error"} <= set(d["points"][0]), d

    def test_horizon_and_length_guards_reject_bad_input(self) -> None:
        from apedp.identifiability import rolling_origin_identifiability

        x = np.arange(1.0, 11.0)
        with pytest.raises(ValueError, match="horizon must be positive"):
            rolling_origin_identifiability(x, x, [5], _linear_fit, horizon=0)
        with pytest.raises(ValueError, match="same length"):
            rolling_origin_identifiability(x, x[:3], [5], _linear_fit)
        # An endpoint leaving no room for the forecast is a setup error, and
        # must say so rather than silently shrinking the horizon.
        with pytest.raises(ValueError, match="no complete forecast horizon"):
            rolling_origin_identifiability(x, x, [9], _linear_fit, horizon=5)

    def test_nonfinite_predictions_are_rejected(self) -> None:
        """A fit returning NaN must not become a confident forecast_error."""
        from apedp.identifiability import rolling_origin_identifiability

        x = np.arange(1.0, 21.0)

        def bad_fit(xs, ys):
            return 1.0, (lambda _z: np.full(len(np.atleast_1d(_z)), np.nan))

        with pytest.raises(ValueError, match="non-finite"):
            rolling_origin_identifiability(x, 2.0 * x, [8, 12], bad_fit)


class TestQdGeneration:
    """apedp/qd_generation.py -- the curated-table schema contract."""

    def test_published_table_validates_clean(self) -> None:
        from apedp.qd_generation import validate_qd_table, yu_cdse_rows

        out = validate_qd_table(yu_cdse_rows())
        assert out["errors"] == [], out
        assert out["n"] > 0 and out["materials"] == ["CdSe"], out

    def test_every_schema_violation_is_reported_not_raised(self) -> None:
        """validate_qd_table returns errors; it does not throw on bad data."""
        from apedp.qd_generation import validate_qd_table

        rows = [
            {},                                                   # all columns missing
            {"material": "CdSe", "D_nm": 999.0, "lambda_peak_nm": 500.0,
             "source": "x", "ligand": "TOPO", "solvent": "toluene", "T_K": 298.0},
            {"material": "CdSe", "D_nm": "not-a-number", "lambda_peak_nm": 500.0,
             "source": "x", "ligand": "TOPO", "solvent": "toluene", "T_K": 298.0},
        ]
        out = validate_qd_table(rows)
        assert out["n"] == 3, out
        # Not an exact count: a row missing every column yields one error per
        # column, and pinning the number would make the test a hostage to the
        # schema's length. What matters is that each *class* of defect is
        # named, so a validator that silently skipped one would fail here.
        assert any("missing" in e for e in out["errors"]), out
        assert any("out-of-range" in e for e in out["errors"]), out
        assert any("non-numeric" in e for e in out["errors"]), out
        assert all(isinstance(e, str) and e for e in out["errors"]), out

    def test_anchor_places_the_bulk_gap_at_zero_offset(self) -> None:
        """The anchor is a definitional invariant: dE == 0 at bulk."""
        from apedp.qd_generation import to_arrays, yu_cdse_rows

        R, dE = to_arrays(yu_cdse_rows(), with_anchor=True)
        assert R[-1] == 100.0, R[-1]
        assert dE[-1] == 0.0, dE[-1]

    def test_radius_is_half_the_diameter(self) -> None:
        from apedp.qd_generation import to_arrays, yu_cdse_rows

        R, _ = to_arrays(yu_cdse_rows(), with_anchor=False)
        rows = yu_cdse_rows()
        assert np.allclose(R, [r["D_nm"] / 2.0 for r in rows]), R

    def test_anchor_can_be_omitted(self) -> None:
        from apedp.qd_generation import to_arrays, yu_cdse_rows

        R, _ = to_arrays(yu_cdse_rows(), with_anchor=False)
        assert R.size == len(yu_cdse_rows()), R.size
        assert not np.any(R == 100.0), R

    def test_admissibility_gates_on_physics_not_just_evaluability(self) -> None:
        """A runnable expression is not automatically an admissible one."""
        from apedp.qd_generation import admissibility

        # 1/R: positive, falling, and ~0 at 100 nm. Passes on all three.
        good = admissibility("1.0/R")
        assert good["status"] == "PASS", good
        assert good["positive"] and good["monotone_decreasing"], good

        # Rises instead of falling, so the physics check must reject it even
        # though it evaluates cleanly over the whole grid.
        rising = admissibility("0.01*R")
        assert rising["status"] == "FAIL", rising
        assert rising["positive"] is True, rising
        assert rising["monotone_decreasing"] is False, rising

        # Singular inside the grid: must not raise, must report unevaluable.
        singular = admissibility("1.0/(R - 1)")
        assert singular["status"] == "FAIL", singular

        # Nonsense input is a FAIL dict, not an exception.
        garbage = admissibility("not an expression (((")
        assert garbage["status"] == "FAIL", garbage
        assert "reason" in garbage, garbage


class TestExperimentDesigner:
    """apedp/experiment_designer.py -- variance-ratio proposal scoring."""

    def test_pure_noise_scores_below_a_real_spread(self) -> None:
        from apedp.experiment_designer import predictive_scores

        rng = np.random.default_rng(0)
        flat = rng.normal(0.0, 0.01, size=(200, 12))   # no real spread
        spread_out, _ = predictive_scores(flat, sigma=1.0)
        predictive = rng.normal(0.0, 1.0, size=(200, 12))  # sigma-sized spread
        spread_in, _ = predictive_scores(predictive, sigma=1.0)
        assert spread_in.max() > spread_out.max() * 10.0, (spread_in.max(), spread_out.max())

    def test_variance_ratio_is_squared_spread_over_squared_sigma(self) -> None:
        from apedp.experiment_designer import predictive_scores

        # ddof=1, so two draws two apart have std sqrt(2), not 1.
        pred = np.array([[0.0, 10.0], [2.0, 14.0]])
        scores, spread = predictive_scores(pred, sigma=1.0)
        assert np.allclose(spread, [np.sqrt(2.0), 2 * np.sqrt(2.0)]), spread
        assert np.allclose(scores, [2.0, 8.0]), scores

    def test_larger_measurement_noise_lowers_the_score(self) -> None:
        """The EIG proxy must actually trade signal against noise."""
        from apedp.experiment_designer import predictive_scores

        pred = np.array([[0.0], [2.0]])
        quiet, _ = predictive_scores(pred, sigma=0.5)
        loud, _ = predictive_scores(pred, sigma=2.0)
        assert quiet[0] > loud[0], (quiet, loud)

    def test_degenerate_predictive_input_is_rejected(self) -> None:
        from apedp.experiment_designer import predictive_scores

        with pytest.raises(ValueError, match="n_samples"):
            predictive_scores(np.zeros((1, 5)), sigma=1.0)      # one sample
        with pytest.raises(ValueError, match="n_samples"):
            predictive_scores(np.zeros((4, 0)), sigma=1.0)      # empty grid
        with pytest.raises(ValueError, match="finite"):
            predictive_scores(np.array([[np.nan, 0.0], [1.0, 1.0]]), sigma=1.0)
        with pytest.raises(ValueError, match="negative"):
            predictive_scores(np.array([[0.0], [1.0]]), sigma=-1.0)

    def test_zero_sigma_is_floored_rather_than_dividing_by_zero(self) -> None:
        from apedp.experiment_designer import predictive_scores

        scores, _ = predictive_scores(np.array([[0.0], [2.0]]), sigma=0.0)
        assert np.all(np.isfinite(scores)), scores

    def test_design_picks_the_widest_posterior_not_the_largest_mean(self) -> None:
        """The score is a variance ratio, so a big mean must not win.

        An offset-only column has an enormous mean and the same spread; if the
        designer ranked on level instead of spread it would pick that, which
        would be picking the model that is confidently wrong rather than the
        one that admits uncertainty.
        """
        from apedp.experiment_designer import design_from_predictive_samples

        rng = np.random.default_rng(7)
        n = 10
        pred = rng.normal(0.0, 1.0, size=(400, n))
        pred[:, 9] += 500.0            # huge mean, unchanged spread
        pred[:, 4] *= 20.0             # modest mean, 20x the spread
        xs = np.arange(float(n))
        design = design_from_predictive_samples(pred, xs, sigma=1.0, n_points=1)
        assert design.selected_points == [4.0], design.to_dict()
        assert len(design.scores) == len(design.selected_points), design.to_dict()
        # ``candidate_count`` is left as the implementation reports it rather
        # than asserted to a number: this function records pred.shape[0]
        # (posterior draws) while the sibling design_experiments records
        # len(candidates) (competing models). Neither is the grid size, and the
        # field has no in-repo consumer, so the intended meaning is genuinely
        # ambiguous. Pinning either value here would ratify a guess.
        assert isinstance(design.candidate_count, int), design.to_dict()
        assert len(design.raw_spread) == len(design.selected_points), design.to_dict()

    def test_design_respects_n_points_and_min_distance(self) -> None:
        from apedp.experiment_designer import design_from_predictive_samples

        rng = np.random.default_rng(3)
        pred = rng.normal(0.0, 1.0, size=(300, 10))
        xs = np.arange(10.0)
        design = design_from_predictive_samples(pred, xs, sigma=1.0,
                                               n_points=3, min_distance=2.0)
        assert len(design.selected_points) == 3, design.to_dict()
        gaps = np.diff(sorted(design.selected_points))
        assert np.all(gaps >= 2.0), design.selected_points

    def test_design_to_dict_is_serializable(self) -> None:
        import json

        from apedp.experiment_designer import design_from_predictive_samples

        rng = np.random.default_rng(1)
        pred = rng.normal(0.0, 1.0, size=(50, 6))
        design = design_from_predictive_samples(pred, np.arange(6.0), sigma=1.0)
        assert json.loads(json.dumps(design.to_dict()))["selected_points"] is not None


def _tiny_engine():
    from apedp.model_discovery import ModelDiscoveryEngine

    return ModelDiscoveryEngine(["x"], target_unit="dimensionless")


class TestClosedLoop:
    """apedp/closed_loop.py -- discover -> propose -> update over real data."""

    @staticmethod
    def _runner(**kwargs):
        from apedp.closed_loop import ClosedLoopRunner

        # Bare forms: the engine fits one coefficient per additive term itself
        # and rejects undeclared symbols like "a".
        candidates = {"linear": "x", "quadratic": "x**2 + x"}
        return ClosedLoopRunner(_tiny_engine(), candidates, **kwargs)

    def test_config_rejects_a_degenerate_loop(self) -> None:
        from apedp.closed_loop import ClosedLoopConfig

        with pytest.raises(ValueError, match="max_iterations"):
            ClosedLoopConfig(max_iterations=0)
        with pytest.raises(ValueError, match="n_points_per_iter"):
            ClosedLoopConfig(n_points_per_iter=0)

    def test_constructor_requires_candidates_and_matching_grid(self) -> None:
        from apedp.closed_loop import ClosedLoopRunner

        with pytest.raises(ValueError, match="at least one candidate"):
            ClosedLoopRunner(_tiny_engine(), {})
        with pytest.raises(ValueError, match="design_grid keys"):
            ClosedLoopRunner(_tiny_engine(), {"linear": "x"},
                             design_grid={"wrong_var": np.arange(3.0)})

    def test_propose_before_discover_is_an_error(self) -> None:
        """Ordering is a real precondition, not a suggestion."""
        from apedp.closed_loop import ClosedLoopState

        runner = self._runner()
        with pytest.raises(ValueError, match="discover"):
            runner.propose(ClosedLoopState())

    def test_discover_then_propose_produces_a_design(self) -> None:
        from apedp.closed_loop import ClosedLoopState

        runner = self._runner(design_grid={"x": np.linspace(0.1, 5.0, 25)})
        state = ClosedLoopState(
            x_data={"x": np.linspace(0.5, 4.0, 12)},
            y_data=2.0 * np.linspace(0.5, 4.0, 12),
        )
        state = runner.discover(state)
        assert state.result is not None, "discover() produced no result"
        state = runner.propose(state)
        assert state.design is not None, "propose() produced no design"
        assert state.design.to_dict()["selected_points"], state.design.to_dict()

    def test_update_appends_data_and_advances_the_iteration(self) -> None:
        from apedp.closed_loop import ClosedLoopState

        runner = self._runner()
        state = ClosedLoopState(x_data={"x": np.array([1.0, 2.0])},
                                y_data=np.array([1.0, 2.0]))
        state = runner.update(state, {"x": np.array([3.0])}, np.array([3.0]))
        assert state.iteration == 1, state.iteration
        assert np.allclose(state.x_data["x"], [1.0, 2.0, 3.0]), state.x_data
        assert np.allclose(state.y_data, [1.0, 2.0, 3.0]), state.y_data
        assert state.history[-1]["n_total"] == 3, state.history

    def test_update_guards_are_load_bearing(self) -> None:
        """Mismatched keys or ragged lengths must not corrupt the state."""
        from apedp.closed_loop import ClosedLoopState

        runner = self._runner()
        state = ClosedLoopState(x_data={"x": np.array([1.0])}, y_data=np.array([1.0]))
        with pytest.raises(ValueError, match="must match the engine variables"):
            runner.update(state, {"nope": np.array([2.0])}, np.array([2.0]))
        with pytest.raises(ValueError, match="must contain finite values"):
            runner.update(state, {"x": np.array([])}, np.array([]))
        with pytest.raises(ValueError, match="finite with length"):
            runner.update(state, {"x": np.array([1.0, 2.0])}, np.array([2.0]))
        # State survived every rejection intact.
        assert np.allclose(state.x_data["x"], [1.0]), state.x_data

    def test_step_runs_the_full_cycle(self) -> None:
        from apedp.closed_loop import ClosedLoopState

        runner = self._runner(design_grid={"x": np.linspace(0.1, 6.0, 30)})
        state = ClosedLoopState(
            x_data={"x": np.linspace(0.5, 4.0, 12)},
            y_data=1.5 * np.linspace(0.5, 4.0, 12),
        )
        state = runner.step(state, {"x": np.array([4.5])}, np.array([1.5 * 4.5]))
        assert state.iteration == 1, state.iteration
        assert state.result is not None and state.design is not None

    def test_step_rejects_a_half_supplied_measurement(self) -> None:
        from apedp.closed_loop import ClosedLoopState

        runner = self._runner()
        with pytest.raises(ValueError, match="supplied together"):
            runner.step(ClosedLoopState(), new_x={"x": np.array([1.0])})
        with pytest.raises(ValueError, match="supplied together"):
            runner.step(ClosedLoopState(), new_y=np.array([1.0]))

    def test_state_to_dict_is_serializable_and_counts_data(self) -> None:
        import json

        from apedp.closed_loop import ClosedLoopState

        d = ClosedLoopState(x_data={"x": np.array([1.0, 2.0, 3.0])},
                            y_data=np.array([1.0, 2.0, 3.0])).to_dict()
        assert d["n_data"] == 3, d
        assert d["result"] is None and d["design"] is None, d
        assert json.loads(json.dumps(d))["iteration"] == 0, d

    def test_bayesian_design_adapter_bridges_into_the_designer(self) -> None:
        """The link that made experiment_designer reachable at all."""
        from apedp.closed_loop import bayesian_design_adapter
        from apedp.model_discovery import ModelDiscoveryEngine

        engine = ModelDiscoveryEngine(["x"], target_unit="dimensionless")
        grid = {"x": np.linspace(0.5, 5.0, 20)}
        result = engine.discover({"linear": "x", "quadratic": "x**2 + x"},
                                 {"x": np.linspace(0.5, 4.0, 12)},
                                 2.0 * np.linspace(0.5, 4.0, 12),
                                 design_grid=grid)

        # The adapter is a *factory*: it takes a predictive_fn and returns the
        # design_fn that ClosedLoopRunner.propose calls. It is not itself a
        # designer, so calling it with (result, grid) is a type error.
        def predictive_fn(_result, xs):
            rng = np.random.default_rng(11)
            return rng.normal(0.0, 1.0, size=(50, len(xs)))

        design_fn = bayesian_design_adapter(predictive_fn, sigma=1.0, n_points=1)
        design = design_fn(result, grid)
        assert design is not None, "adapter produced no design for a ranked result"
        d = design.to_dict()
        assert d["selected_points"], d
        assert "Bayesian" in d["objective"], d

    def test_bayesian_design_adapter_declines_what_it_cannot_handle(self) -> None:
        """Returns None rather than raising -- a runner hook must not crash a loop."""
        from apedp.closed_loop import bayesian_design_adapter

        design_fn = bayesian_design_adapter(
            lambda _r, xs: np.zeros((10, len(xs))), sigma=1.0)
        # Multi-dimensional grid: the adapter is 1D-only and must say so.
        assert design_fn(object(), {"x": np.arange(3.0), "y": np.arange(3.0)}) is None
        # Predictive fn raising: swallowed, not propagated.
        def boom(_r, _xs):
            raise ValueError("nope")
        assert bayesian_design_adapter(boom)(object(), {"x": np.arange(3.0)}) is None
        # Wrong shape: a single draw is not a posterior, and a transposed one
        # does not line up with the grid it claims to describe.
        one_draw = bayesian_design_adapter(lambda _r, xs: np.zeros((1, len(xs))))
        assert one_draw(object(), {"x": np.arange(3.0)}) is None
        transposed = bayesian_design_adapter(lambda _r, xs: np.zeros((len(xs), 5)))
        assert transposed(object(), {"x": np.arange(3.0)}) is None

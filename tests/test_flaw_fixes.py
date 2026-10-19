"""Regression tests for ASSESSMENT.md flaw fixes (no code changes to production paths here)."""
from __future__ import annotations

import time

import numpy as np
import pytest
import sympy as sp

from apedp import safety
from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
from apedp.discovery import BayesianRefinementEngine, MultiVariableSRE
from apedp.metrology import AnalyticalUncertaintyPropagator
from apedp.repair import AgnosticClosedLoopAssuranceSystem
from apedp.safety import limit_with_timeout, safe_parse_expr


def test_safe_parse_accepts_valid():
    expr = safe_parse_expr("G*m1*m2/(r**2)")
    assert isinstance(expr, sp.Expr)


def test_safe_parse_rejects_injection():
    with pytest.raises(ValueError):
        safe_parse_expr("__import__('os').system('x')")
    with pytest.raises(ValueError):
        safe_parse_expr("x.__class__")
    with pytest.raises(ValueError):
        safe_parse_expr("eval('1')")
    with pytest.raises(ValueError):
        safe_parse_expr("x" * 3000)


def test_limit_with_timeout_simple():
    x = sp.Symbol("x")
    assert limit_with_timeout(1 / x, x, sp.oo, timeout=2.0) == 0


def test_limit_with_timeout_bounds_wall_clock(monkeypatch):
    """Regression: a hung symbolic limit must not delay the caller past the budget.

    The previous ThreadPoolExecutor-based implementation raised on
    ``fut.result(timeout=...)`` but then blocked in ``Executor.__exit__``'s
    ``shutdown(wait=True)``, which waits on the un-cancellable hung worker. A 1s
    timeout therefore returned only when ``sp.limit`` eventually finished.
    """
    x = sp.Symbol("x")
    calls = {"count": 0}

    def hanging_limit(*args, **kwargs):
        calls["count"] += 1
        time.sleep(5.0)
        return 0

    monkeypatch.setattr(safety.sp, "limit", hanging_limit)

    # 1. Timeout expires, the numeric fallback succeeds, and the caller returns
    #    on budget instead of waiting out the hung worker.
    start = time.perf_counter()
    value = limit_with_timeout(1 / x, x, 0, timeout=0.5)
    elapsed = time.perf_counter() - start
    assert calls["count"] == 1
    assert elapsed < 2.0, f"timeout did not bound wall clock ({elapsed:.2f}s for a 0.5s budget)"
    assert abs(float(value)) < 1e-3  # symmetric numeric probe of 1/x averages to ~0

    # 2. When the numeric probe also fails, LimitTimeout is still raised on budget.
    start = time.perf_counter()
    with pytest.raises(safety.LimitTimeout):
        limit_with_timeout(sp.sqrt(-1 - x**2), x, 0, timeout=0.5)
    assert time.perf_counter() - start < 2.0


def test_limit_with_timeout_propagates_worker_errors():
    """A genuine symbolic failure must surface, not be masked as a timeout."""
    x = sp.Symbol("x")

    def failing_limit(*args, **kwargs):
        raise ValueError("sympy rejected the limit")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(safety.sp, "limit", failing_limit)
        with pytest.raises(ValueError, match="sympy rejected the limit"):
            limit_with_timeout(1 / x, x, 0, timeout=5.0)


def test_repair_probe_guards_swallow_only_symbolic_failures(monkeypatch):
    """Regression: repair probe sites caught bare ``Exception``.

    Every symbolic probe in the agnostic repair path used
    ``except Exception``, so a genuine defect (any non-symbolic error) was
    absorbed and reported as "no anomaly detected" -- a silent false negative
    in singularity detection. The probes now name an explicit exception set, so
    symbolic/numeric failures are still tolerated while unexpected errors
    propagate to the caller.
    """
    from apedp import repair as repair_mod

    assert ValueError in repair_mod.PROBE_SWALLOWED_ERRORS
    assert ArithmeticError in repair_mod.PROBE_SWALLOWED_ERRORS
    assert AttributeError in repair_mod.PROBE_SWALLOWED_ERRORS
    assert KeyError in repair_mod.PROBE_SWALLOWED_ERRORS
    assert RecursionError in repair_mod.PROBE_SWALLOWED_ERRORS
    assert NotImplementedError in repair_mod.PROBE_SWALLOWED_ERRORS
    assert sp.PolynomialError in repair_mod.PROBE_SWALLOWED_ERRORS
    # A plain RuntimeError is deliberately NOT in the set (only its
    # RecursionError subclass is), so it must not be treated as a probe failure.
    assert RuntimeError not in repair_mod.PROBE_SWALLOWED_ERRORS

    system = AgnosticClosedLoopAssuranceSystem()

    # Realistic symbolic/numeric failures are absorbed and the repair proceeds.
    for exc_type in (ValueError, TypeError, ZeroDivisionError, AttributeError, KeyError):
        def raising_limit(*args, _exc=exc_type, **kwargs):
            raise _exc(f"symbolic failure: {_exc.__name__}")

        monkeypatch.setattr(repair_mod, "limit_with_timeout", raising_limit)
        result = system.autonomous_stress_test_and_repair("1/x")
        assert result["status"] in {"REPAIRED", "APPROVED"}, exc_type

    # Anything outside the documented set is an APEDP defect and must surface
    # rather than being silently downgraded to "no anomaly found".
    class UnexpectedDefect(RuntimeError):
        """Stand-in for an APEDP bug, not a symbolic failure."""

    def boom(*args, **kwargs):
        raise UnexpectedDefect("not a symbolic failure")

    monkeypatch.setattr(repair_mod, "limit_with_timeout", boom)
    with pytest.raises(UnexpectedDefect):
        system.autonomous_stress_test_and_repair("1/x")


def test_repair_probes_still_detect_real_singularities():
    """Narrowing the probe guards must not cost detection power."""
    system = AgnosticClosedLoopAssuranceSystem()

    # Division-by-zero at 0 is still found and repaired.
    zero = system.autonomous_stress_test_and_repair("1/x")
    assert zero["status"] == "REPAIRED"
    assert any("Division" in kind for _, kind in zero["anomalies"])

    # The relativistic sqrt(1 - (v/c)^2) pole at v = c is still found, which is
    # the path whose numeric confirmation is guarded by the narrowed handlers.
    rel = system.autonomous_stress_test_and_repair("m_phys / sqrt(1 - (v_phys/c_light)**2)")
    assert rel["status"] == "REPAIRED"
    assert any("Relativistic" in kind for _, kind in rel["anomalies"])

    # A benign expression stays approved.
    assert system.autonomous_stress_test_and_repair("1/(v_phys**2 + 1)")["status"] == "APPROVED"


def test_score_candidate_falls_back_instead_of_raising():
    """Regression: ``score_candidate`` guards used ``except Exception``.

    The guards hid genuine defects, and the intended fallbacks were unverified.
    Non-SymPy expressions and unscoreable data must produce a documented
    fallback result rather than an exception.
    """
    # Not a SymPy expression: free-symbol counting falls back (AttributeError),
    # and the candidate is scored on the supplied arrays rather than rejected.
    string_expr = BayesianRefinementEngine.score_candidate(
        expression="not an expression",
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([1.1, 1.9, 3.2]),
    )
    assert string_expr["num_params"] == 1
    assert string_expr["complexity"] >= 1

    # Empty data: no likelihood exists, so BIC/AIC are unscoreable (infinite)
    # and the candidate is refused as degenerate rather than crashing.
    empty = BayesianRefinementEngine.score_candidate(
        expression=sp.Symbol("x"),
        y_true=np.array([]),
        y_pred=np.array([]),
        num_params=1,
    )
    assert empty["log_likelihood"] == float("-inf")
    assert empty["bic"] == float("inf")
    assert empty["aic"] == float("inf")
    assert empty["degenerate"] is True

    # Shape mismatch: previously this raised from compute_r_squared before any
    # guard could run, so the documented -inf fallback was unreachable.
    mismatched = BayesianRefinementEngine.score_candidate(
        expression=sp.Symbol("x"),
        y_true=np.array([1.0, 2.0]),
        y_pred=np.array([1.0, 2.0, 3.0]),
    )
    assert np.isnan(mismatched["r_squared"])
    assert mismatched["log_likelihood"] == float("-inf")
    assert mismatched["bic"] == float("inf")
    assert mismatched["degenerate"] is True
    assert "shape mismatch" in mismatched["degenerate_reason"]

    # A broadcastable mismatch ((1,) against (n,)) is equally unscoreable; numpy
    # would otherwise compute a number from mismatched inputs without complaint.
    broadcast = BayesianRefinementEngine.score_candidate(
        expression=sp.Symbol("x"),
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([2.0]),
    )
    assert np.isnan(broadcast["r_squared"])
    assert broadcast["degenerate"] is True
    assert "shape mismatch" in broadcast["degenerate_reason"]

    # An unexpected error from the traversal helper is a defect, not a fallback.
    class UnexpectedDefect(RuntimeError):
        """Stand-in for an APEDP bug in the scoring path."""

    def boom(*args, **kwargs):
        raise UnexpectedDefect("not a traversal failure")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sp, "preorder_traversal", boom)
        with pytest.raises(UnexpectedDefect):
            BayesianRefinementEngine.score_candidate(
                expression=sp.Symbol("x"),
                y_true=np.array([1.0, 2.0, 3.0]),
                y_pred=np.array([1.0, 2.0, 3.0]),
            )


def test_invariant_probes_sample_the_model_not_its_symbol_names():
    """Regression: structural-invariant probes matched names and declared bounds.

    Two failure modes are covered. (1) Gating on a variable's *name* meant
    renaming a symbol silently disabled a physical check. (2) Evaluating only at
    the declared bounds meant a violation in the interior of the declared
    envelope was invisible. Both probes now evaluate the model across the
    envelope and consult no names at all.
    """
    from apedp.falsification import IndependentAdversarialFalsifier

    falsifier = IndependentAdversarialFalsifier()

    # (2) The declared bounds are [-1, 1]; the old probe evaluated only x = -1,
    # where x**2 - 0.5 = +0.5, and reported no violation.
    interior = falsifier.falsify(
        formula="x**2 - 0.5",
        variable_domains={"x": (-1.0, 1.0)},
        physical_invariants=["positivity"],
        critical_boundaries={},
    )
    assert "PositivityViolationProbe" in {c.probe_name for c in interior.counterexamples}

    # (1) Same physics, symbol renamed: the verdict must not change. "T_temp" is
    # the name the temperature probe used to require.
    for symbol in ("T_temp", "zzz_not_a_temperature"):
        renamed = falsifier.falsify(
            formula=f"24.942 * (1.0 + 0.0001 * {symbol})",
            variable_domains={symbol: (1.0, 300.0)},
            physical_invariants=["third_law_heat_capacity"],
            critical_boundaries={},
        )
        assert "ThirdLawHeatCapacityProbe" in {c.probe_name for c in renamed.counterexamples}, symbol


def test_unevaluable_invariant_is_reported_as_an_error_not_a_pass():
    """Regression: a crashed probe was silently equivalent to a satisfied invariant.

    ``sqrt(-500 * rho)`` has no real value anywhere in its declared envelope, so
    the positivity probe used to ``except Exception: pass`` and the invariant was
    still counted among ``invariants_checked`` -- an unevaluated check reported as
    a passing one. It must now be surfaced as a probe error with zero checks
    attributed, and never as a clean bill of health.
    """
    from apedp.falsification import IndependentAdversarialFalsifier

    report = IndependentAdversarialFalsifier().falsify(
        formula="sqrt(-500.0 * rho)",
        variable_domains={"rho": (1.0, 1000.0)},
        physical_invariants=["positivity", "strictly_positive"],
        critical_boundaries={},
    )
    metrics = report.detailed_metrics["invariants"]
    assert metrics["invariants_declared"] == 2
    assert metrics["invariants_checked"] == 0
    assert metrics["envelope_samples_unevaluated"] == metrics["envelope_samples"]
    assert any(e["error"] == "NoRealValues" for e in metrics["probe_errors"])
    # The model is still falsified, but by the numerical axis rather than by a
    # probe that never ran.
    assert not report.survived


# ---------------------------------------------------------------------------
# W6: a probe that could not decide is an error on every axis, not a pass
# ---------------------------------------------------------------------------


def _report_for(label: str, formula: str, **kwargs):
    from apedp.falsification import IndependentAdversarialFalsifier

    report = IndependentAdversarialFalsifier().falsify(formula=formula, **kwargs)
    metrics = report.detailed_metrics
    assert metrics["probe_errors"], f"{label}: no probe error recorded"
    assert metrics["unresolved_probe_count"] >= 1, f"{label}: no unresolved probe"
    assert all("axis" in e for e in metrics["probe_errors"]), "flat entries must carry their axis"
    assert report.verdict == "INCONCLUSIVE_PROBE_ERROR", (
        f"{label}: a model with an unresolved probe reported {report.verdict}"
    )
    assert report.survived is False
    return report


def test_mathematical_axis_surfaces_a_crashed_pole_search(monkeypatch):
    """A denominator-root search that raises is an unknown, not a clean axis.

    ``sp.real_roots`` is made to fail so the crash is deterministic; the model is
    otherwise clean over the declared domain (its roots are outside it), so before
    this fix the axis silently reported nothing and the model was cleared.
    """
    from apedp import falsification

    def explode(*args, **kwargs):
        raise ValueError("injected: root isolation failed")

    monkeypatch.setattr(falsification.sp, "real_roots", explode)
    report = _report_for(
        "mathematical",
        "1.0 / (x**2 - 2.0)",
        variable_domains={"x": (2.0, 10.0)},
        physical_invariants=[],
        critical_boundaries={},
    )
    errors = report.detailed_metrics["mathematical"]["probe_errors"]
    # The denominator appears twice in the collected set (top-level and negative
    # power), so both attempts are recorded; both are the same probe failing.
    assert {e["probe"] for e in errors} == {"DenominatorRootProbe"}
    assert all(e["error"] == "ValueError" for e in errors)
    assert all(e["unresolved"] is True for e in errors)


def test_boundary_axis_surfaces_a_timed_out_limit(monkeypatch):
    """A limit that times out is unknown: neither a divergence nor a clean pass."""
    from apedp.falsification import probes_boundary as boundary_impl
    from apedp.safety import LimitTimeout

    def timeout(*args, **kwargs):
        raise LimitTimeout("injected: limit budget exhausted")

    # The boundary probe lives in `apedp.falsification.probes_boundary`; patching
    # the facade (or `falsifier`) would rebind a name no probe reads.
    monkeypatch.setattr(boundary_impl, "limit_with_timeout", timeout)
    report = _report_for(
        "boundary",
        "log(x - 5.0)",
        variable_domains={"x": (5.001, 10.0)},
        physical_invariants=[],
        critical_boundaries={"x": [5.0]},
    )
    errors = report.detailed_metrics["boundary"]["probe_errors"]
    assert "AsymptoticDivergenceProbe" in [e["probe"] for e in errors]
    assert all(e["unresolved"] is True for e in errors)
    assert "boundary:AsymptoticDivergenceProbe" in report.refusal_reason


def test_numerical_axis_surfaces_a_model_it_cannot_condition():
    """A constant model has no conditioning to measure: the axis says so."""
    report = _report_for(
        "numerical",
        "5.0",
        variable_domains={"x": (1.0, 10.0)},
        physical_invariants=[],
        critical_boundaries={},
    )
    errors = report.detailed_metrics["numerical"]["probe_errors"]
    assert [e["probe"] for e in errors] == ["NumericalStabilityProbe"]
    assert errors[0]["error"] == "NoFreeVariables"


def test_empirical_axis_surfaces_a_skipped_multi_variable_comparison():
    """Skipping the empirical axis is not passing it."""
    report = _report_for(
        "empirical",
        "a * x + b",
        variable_domains={"x": (0.001, 100.0), "a": (0.001, 100.0), "b": (0.001, 100.0)},
        physical_invariants=[],
        critical_boundaries={},
        empirical_data={"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]},
    )
    errors = report.detailed_metrics["empirical"]["probe_errors"]
    assert [e["probe"] for e in errors] == ["EmpiricalResidualProbe"]
    assert errors[0]["error"] == "ModelNotUnivariate"


def test_empirical_axis_rejects_non_finite_metrics():
    report = _report_for(
        "empirical",
        "2.0 * x",
        variable_domains={"x": (0.001, 100.0)},
        physical_invariants=[],
        critical_boundaries={},
        empirical_data={"x": [1.0, 2.0, 3.0], "y": [2.0, float("nan"), 6.0]},
    )
    assert report.verdict == "INCONCLUSIVE_PROBE_ERROR"
    errors = report.detailed_metrics["empirical"]["probe_errors"]
    assert len(errors) == 1
    assert errors[0]["error"] == "NonFiniteMetrics"
    assert errors[0]["unresolved"] is True


def test_every_falsification_axis_initialises_the_probe_error_channel():
    """The channel exists on all five axes, so 'no errors' is distinguishable from 'not recorded'."""
    from apedp.falsification import IndependentAdversarialFalsifier

    report = IndependentAdversarialFalsifier().falsify(
        formula="2.0 * x + 1.0", variable_domains={"x": (1.0, 10.0)}, critical_boundaries={}
    )
    metrics = report.detailed_metrics
    assert report.verdict == "SURVIVED_PROVISIONAL"
    for axis in ("mathematical", "boundary", "numerical", "invariants", "empirical"):
        if axis in metrics:
            assert metrics[axis]["probe_errors"] == [], f"{axis} axis has no error channel"
    assert metrics["probe_errors"] == []
    assert metrics["probe_error_count"] == 0


def test_a_degraded_probe_with_a_measured_fallback_is_reported_but_not_blocking():
    """A symbolic derivative that will not evaluate is recorded; finite differences still measured it.

    The distinction is the point of the ``unresolved`` flag: it separates a
    degradation of a probe from a hole in the evidence.
    """
    from apedp.falsification import IndependentAdversarialFalsifier

    metrics: dict = {"probe_errors": []}
    IndependentAdversarialFalsifier._record_probe_error(
        metrics, "GradientExplosionProbe", ValueError("no symbolic form"), measured_alternative=True
    )
    IndependentAdversarialFalsifier._record_probe_error(
        metrics, "DenominatorRootProbe", ValueError("no root isolation")
    )
    assert [e["unresolved"] for e in metrics["probe_errors"]] == [False, True]
    assert metrics["probe_errors"][0]["detail"] == "no symbolic form"


def test_bayesian_real_likelihood_and_selection():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_good = np.array([1.1, 1.9, 3.1, 3.9])
    y_bad = np.array([0.0, 0.0, 0.0, 0.0])
    ll_good, _ = BayesianRefinementEngine.gaussian_log_likelihood(y_true, y_good)
    ll_bad, _ = BayesianRefinementEngine.gaussian_log_likelihood(y_true, y_bad)
    assert ll_good > ll_bad
    bic_good = BayesianRefinementEngine.compute_bic(ll_good, 2, 4)
    bic_bad = BayesianRefinementEngine.compute_bic(ll_bad, 2, 4)
    assert bic_good < bic_bad
    best = BayesianRefinementEngine.select_best_model([
        {"bic": bic_bad, "complexity": 3},
        {"bic": bic_good, "complexity": 5},
    ])
    assert best is not None and best["bic"] == bic_good
    assert BayesianRefinementEngine.select_best_model([{"bic": float("inf")}]) is None


def test_degeneracy_refusal():
    y_true = np.array([1.0, 2.0, 3.0])
    degen, reason = BayesianRefinementEngine.is_degenerate(
        y_true, np.array([1.0, 1.0, 1.0]), sp.Integer(1)
    )
    assert degen
    assert "constant prediction" in reason, reason
    ok, _ = BayesianRefinementEngine.is_degenerate(
        y_true, np.array([1.0, 2.0, 3.0]), sp.Symbol("x")
    )
    assert not ok
    assert np.isnan(BayesianRefinementEngine.compute_r_squared(y_true, np.array([np.nan] * 3)))


def test_repair_detects_zero_and_critical():
    system = AgnosticClosedLoopAssuranceSystem()
    r = system.autonomous_stress_test_and_repair("1/x")
    assert r["status"] == "REPAIRED"
    assert any("Division" in kind for _, kind in r["anomalies"])
    assert r["eps_symbols"], "expected regularizer symbols after repair"
    assert "distortion" in r and "validity_domain" in r
    # Caller-supplied critical value on a dimensionless token (beta)
    r2 = system.autonomous_stress_test_and_repair(
        "1/(beta - 0.5)", critical_values={"beta": [0.5]}
    )
    assert r2["status"] == "REPAIRED"
    # Injection rejected, not executed
    r3 = system.autonomous_stress_test_and_repair("__import__('os')")
    assert r3["status"] == "REJECTED"


def test_repair_strict_units():
    system = AgnosticClosedLoopAssuranceSystem()
    ambiguous = sp.Symbol("Zqxwobble")
    with pytest.raises(ValueError):
        system.infer_units_for_expression(ambiguous + 1, strict=True)
    ok = system.infer_units_for_expression(
        sp.Symbol("Q_dot") + 1, variable_units={"Q_dot": "watt"}, strict=True
    )
    assert ok is not None


def test_monte_carlo_matches_gum_linear():
    x = sp.Symbol("x")
    mc = AnalyticalUncertaintyPropagator.propagate_uncertainty_monte_carlo(
        2 * x, {"x": 10.0}, {"x": 0.5}, num_samples=20000, seed=0
    )
    assert abs(mc["mean"] - 20.0) < 0.1
    assert abs(mc["std"] - 1.0) < 0.1
    assert mc["non_finite_fraction"] == 0.0


def test_pipeline_ingest_skips_bad_rows(tmp_path):
    from apedp.pipeline import DiscoveryPipeline

    p = DiscoveryPipeline(output_dir=str(tmp_path / "out"))
    csv_path = tmp_path / "d.csv"
    rows = "\n".join([f"{i},{i**2}" for i in range(1, 10)] + ["bad,2"])
    csv_path.write_text("x,y\n" + rows + "\n")
    out = p.ingest_csv(str(csv_path), "y")
    assert len(out["data"]["y"]) == 9
    csv_bad = tmp_path / "allbad.csv"
    csv_bad.write_text("x,y\nbad,bad\nnope,nope\n")
    with pytest.raises(ValueError):
        p.ingest_csv(str(csv_bad), "y")


def test_epistemic_honesty():
    from apedp.epistemic import (
        AdversarialDeliberationTournament,
        EpistemicGraph,
        LifecycleManager,
        RequirementsTraceabilityMatrix,
    )

    m = RequirementsTraceabilityMatrix()
    m.register_requirement("R1", "Enforce dimensional consistency", "owner-team-x", "ADR-001-ontology", "impl.py-module", "test_x_suite", "evidence-001-run")
    # Short/placeholder fields must now fail completeness
    m.register_requirement("R2", "x", "y", "z", "w", "v", "u")
    complete, missing = m.verify_completeness()
    assert not complete and any("R2" in s for s in missing)

    g = EpistemicGraph()
    lc = LifecycleManager()
    t = AdversarialDeliberationTournament(g, lc)
    res = t.run_tournament("H", "1/x", "zero", True, "1/(x+eps)")
    assert res["mathematical_proof"] is False


def test_experiment_log_bounded(tmp_path):
    from apedp.persistence import ExperimentLog

    log = ExperimentLog(str(tmp_path / "e.jsonl"), max_entries=5)
    for i in range(10):
        log.log("t", {"i": i}, {"r": i})
    assert len(log.get_experiments()) == 5


def test_operator_set_trig_and_sqrt():
    engine = DimensionalAssuranceEngine()
    engine.register_variable("x", "dimensionless")
    platform = ClosedLoopAssuranceSystem(engine)
    sre = MultiVariableSRE(
        assurance_system=platform, independent_vars=["x"], allow_trig=True, allow_sqrt=True
    )
    seen = set()
    for _ in range(300):
        c = sre._generate_candidate(depth=0, max_depth=3)
        for node in sp.preorder_traversal(c):
            if isinstance(node, (sp.sin, sp.cos)):
                seen.add(type(node).__name__)
            if node == sp.sqrt(sp.Symbol("x")) or (
                isinstance(node, sp.Pow) and node.args[1] == sp.Rational(1, 2)
            ):
                seen.add("sqrt")
    assert "sqrt" in seen  # fractional powers / sqrt representable
    # Trig may or may not appear in 300 draws, so assert the gatekeeper instead:
    # every argument offered to sin/cos/exp/log must be dimensionally consistent
    # and dimensionless, and the generator must actually accept some (it declines
    # roughly 7% of attempts, so a handful of draws is never enough to prove it
    # works). Previously this was `assert inner is None or True`, which asserted
    # nothing at all.
    dimensionless = platform.engine.registry.dimensionless.dimensionality
    accepted = 0
    for _ in range(50):
        inner = sre._random_dimensionless_inner(0, 2)
        if inner is None:
            continue
        accepted += 1
        consistency = platform.engine.check_expression_consistency(inner)
        assert consistency.dimensionality == dimensionless, (
            f"transcendental argument {inner} is not dimensionless "
            f"(got {consistency.dimensionality})"
        )
    assert accepted > 0, "no dimensionless inner expression generated in 50 draws"

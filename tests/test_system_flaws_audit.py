"""Regression tests for the flaws catalogued in docs/SYSTEM_FLAWS_AUDIT.md.

Each test names the finding it covers and asserts the *behaviour* that was wrong,
not the shape of the fix, so a different implementation can satisfy it.
"""
from __future__ import annotations

import csv
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from fastapi.testclient import TestClient

import apedp.repair as repair_module
from apedp.api import create_app, rate_limit_per_minute
from apedp.core import (
    ClosedLoopAssuranceSystem,
    DimensionalAssuranceEngine,
    VectorizedDimensionalEngine,
    ureg,
)
from apedp.metrology import has_singularity_risk
from apedp.persistence import ExperimentLog, ModelStore
from apedp.pipeline import DiscoveryPipeline
from apedp.repair import AgnosticClosedLoopAssuranceSystem
from apedp.safety import quiet_float_warnings, safe_parse_expr

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Critical 1: the container never booted
# ---------------------------------------------------------------------------


def test_dockerfile_uses_the_uvicorn_factory_flag():
    """``apedp.api:create_app`` is a factory; uvicorn needs ``--factory``."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "--factory" in dockerfile
    assert re.search(r'CMD \[.*--factory.*"apedp\.api:create_app"|CMD \[.*"apedp\.api:create_app".*--factory.*', dockerfile)


def _dockerfile_instructions() -> list[str]:
    """Effective instructions only: comments describe the fix, they do not run."""
    lines = (REPO_ROOT / "Dockerfile").read_text().splitlines()
    return [line for line in lines if line.strip() and not line.strip().startswith("#")]


def test_dockerfile_does_not_hide_a_failing_build_step():
    instructions = _dockerfile_instructions()
    assert not any("|| true" in line for line in instructions)
    # The test suite is not part of the runtime image.
    assert not any(line.startswith("COPY tests/") for line in instructions)


def test_create_app_returns_a_servable_app():
    """The object uvicorn's factory flag is handed must be a routable app."""
    app = create_app()
    paths = set(app.openapi().get("paths", {}))
    assert "/health" in paths
    assert any(path.startswith("/discover") for path in paths)
    assert any(path.startswith("/repair") for path in paths)


# ---------------------------------------------------------------------------
# Critical 2: API trust boundary (CORS)
# ---------------------------------------------------------------------------


def _allow_origin_header(monkeypatch, env_value: str | None) -> tuple[str | None, str | None]:
    if env_value is None:
        monkeypatch.delenv("APEDP_CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("APEDP_CORS_ORIGINS", env_value)
    client = TestClient(create_app())
    response = client.get("/healthz", headers={"Origin": "https://app.example"})
    return (
        response.headers.get("access-control-allow-origin"),
        response.headers.get("access-control-allow-credentials"),
    )


def test_cors_is_same_origin_by_default(monkeypatch):
    origin, credentials = _allow_origin_header(monkeypatch, None)
    assert origin is None
    assert credentials is None


def test_cors_allows_an_operator_named_origin(monkeypatch):
    origin, credentials = _allow_origin_header(monkeypatch, "https://app.example, https://other.example")
    assert origin == "https://app.example"
    assert credentials == "true"


def test_cors_never_pairs_a_wildcard_with_credentials(monkeypatch):
    """Browsers reject that combination, so it cannot have been working."""
    origin, credentials = _allow_origin_header(monkeypatch, "*")
    assert origin == "*"
    assert credentials is None


# ---------------------------------------------------------------------------
# Critical 3: silent unit bypass
# ---------------------------------------------------------------------------


def _force_system(strict_units: bool = False) -> ClosedLoopAssuranceSystem:
    engine = DimensionalAssuranceEngine()
    engine.register_variable("m_phys", "kilogram")
    engine.register_variable("a_acc", "meter / second**2")
    system = ClosedLoopAssuranceSystem(engine, strict_units=strict_units)
    system.register_model_spec("F", sp.Symbol("m_phys") * sp.Symbol("a_acc"), "newton")
    return system


def test_bare_numbers_are_recorded_as_unit_assumptions():
    system = _force_system()
    assert system.evaluate_model("F", {"m_phys": 2.0, "a_acc": 3.0}) == pytest.approx(6.0)
    # The interpretation is no longer silent: it is on the record.
    assert set(system.unit_assumptions) == {"m_phys", "a_acc"}
    assert "kg" in system.unit_assumptions["m_phys"]


def test_strict_units_refuses_a_bare_number():
    system = _force_system(strict_units=True)
    with pytest.raises(TypeError, match="registered as"):
        system.evaluate_model("F", {"m_phys": 2.0, "a_acc": 3.0})
    assert system.unit_assumptions == {}


def test_strict_units_accepts_explicit_quantities():
    system = _force_system(strict_units=True)
    value = system.evaluate_model(
        "F",
        {"m_phys": ureg.Quantity(2.0, "kilogram"), "a_acc": ureg.Quantity(3.0, "meter / second**2")},
    )
    assert value == pytest.approx(6.0)


def test_dimensionless_variables_need_no_unit_declaration():
    """A dimensionless bare number is not an assumption, so nothing is recorded."""
    engine = DimensionalAssuranceEngine()
    engine.register_variable("x", "dimensionless")
    system = ClosedLoopAssuranceSystem(engine, strict_units=True)
    system.register_model_spec("y", sp.Symbol("x") ** 2, "dimensionless")
    assert system.evaluate_model("y", {"x": 3.0}) == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Critical 4: dimension-vector overflow
# ---------------------------------------------------------------------------


def test_dimension_vectors_use_a_dtype_that_cannot_wrap():
    assert VectorizedDimensionalEngine.DIM_MAP["meter"].dtype == np.int64
    assert all(vector.dtype == np.int64 for vector in VectorizedDimensionalEngine.DIM_MAP.values())


def test_large_exponents_do_not_wrap_in_the_dimension_arithmetic():
    """int8 wrapped silently: 100 * 2 became -56, so a wild exponent could pass."""
    meter = VectorizedDimensionalEngine.DIM_MAP["meter"]
    assert (200 * meter)[1] == 200
    assert (130 * meter)[1] == 130
    assert np.all(np.asarray(300 * meter, dtype=np.int64) == np.array([0, 300, 0, 0, 0, 0, 0]))


def test_overflow_would_have_broken_homogeneity_checks():
    """The wrapped vector no longer equals the true exponent, which is the point."""
    with np.errstate(over="ignore"):
        wrapped = np.int8(100) * np.int8(2)
    assert int(wrapped) == -56  # the old dtype's behaviour, for the record
    assert (np.int64(100) * np.int64(2)) == 200


# ---------------------------------------------------------------------------
# Major 5: a stagnated search was still persisted
# ---------------------------------------------------------------------------


def _write_csv(path: Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        for x in range(1, 11):
            writer.writerow([x, 2 * x + 1])


def test_stagnated_search_is_refused_instead_of_persisted(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "results"))
    stagnation = {
        "stagnated": True,
        "generations_without_improvement": 30,
        "final_normalized_score": 1.0,
    }
    monkeypatch.setattr(
        pipeline,
        "discover",
        lambda *args, **kwargs: {
            "best_expression": "x",
            "best_score": 1.0,
            "r_squared": 0.99,
            "bic": -10.0,
            "aic": -12.0,
            "degenerate": False,
            "degenerate_reason": "",
            "search_stagnated": True,
            "stagnation": stagnation,
        },
    )

    result = pipeline.run(str(csv_path), "y", variable_units={"x": "dimensionless"})
    assert result["status"] == "REFUSED"
    assert "stagnated" in result["reason"]
    assert result["search_stagnated"] is True
    # Nothing reached the store: a seed artefact is not a discovery result.
    assert pipeline.store.list_models() == []


def test_a_converged_search_still_persists(monkeypatch, tmp_path):
    """The refusal must key on stagnation, not on the mere presence of the field."""
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(
        pipeline,
        "discover",
        lambda *args, **kwargs: {
            "best_expression": "2*x + 1",
            "best_score": 1e-6,
            "r_squared": 0.999,
            "bic": -40.0,
            "aic": -42.0,
            "degenerate": False,
            "degenerate_reason": "",
            "search_stagnated": False,
            "stagnation": {"stagnated": False, "generations_without_improvement": 0},
        },
    )
    result = pipeline.run(str(csv_path), "y", variable_units={"x": "dimensionless"})
    assert result["status"] == "PERSISTED"
    assert pipeline.store.list_models()


# ---------------------------------------------------------------------------
# Major 8: safety false positives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "formula",
    ["importance * 2.0", "getattr_index + hasattr_flag", "deflection", "lambda_eff * x", "compile_time"],
)
def test_identifiers_containing_dangerous_words_are_accepted(formula):
    """Substring matching rejected real engineering names, e.g. ``importance``."""
    assert isinstance(safe_parse_expr(formula), sp.Expr)


@pytest.mark.parametrize(
    "formula",
    ["__import__('os')", "x.__class__", "eval('1')", 'open("/etc/passwd")', "import os", "lambda x: x", "a;b"],
)
def test_execution_forms_are_still_rejected(formula):
    with pytest.raises(ValueError):
        safe_parse_expr(formula)


# ---------------------------------------------------------------------------
# Major 9: persistence
# ---------------------------------------------------------------------------


def test_list_models_finds_models_saved_by_another_process(tmp_path):
    first = ModelStore(str(tmp_path))
    saved = first.save_model("m", sp.Symbol("x") ** 2, ["x"], 0.9, 2)

    reopened = ModelStore(str(tmp_path))
    assert reopened.list_models() == [saved]
    assert reopened.load_model(saved)["expression"] == "x**2"


def test_rapid_saves_do_not_collide(tmp_path):
    store = ModelStore(str(tmp_path))
    ids = [
        store.save_model("m", sp.Symbol("x") ** power, ["x"], 0.9, power)
        for power in range(2, 6)
    ]
    assert len(set(ids)) == len(ids)
    assert len(store.list_models()) == len(ids)


def test_experiment_log_timestamps_are_timezone_aware(tmp_path):
    log = ExperimentLog(str(tmp_path / "experiments.jsonl"))
    log.log("discovery", {"a": 1}, {"status": "ok"})
    timestamp = log._entries[0]["timestamp"]
    parsed = __import__("datetime").datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None


def test_a_crash_during_save_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    """``json.dump`` wrote in place, so an interrupted save destroyed the old model."""
    store = ModelStore(str(tmp_path))
    saved = store.save_model("m", sp.Symbol("x") ** 2, ["x"], 0.9, 2)
    target = tmp_path / f"{saved}.json"
    original = target.read_text()

    def exploding_replace(src, dst):
        raise OSError("crash between write and rename")

    monkeypatch.setattr(os, "replace", exploding_replace)
    with pytest.raises(OSError):
        ModelStore(str(tmp_path)).save_model("m2", sp.Symbol("y"), ["y"], 0.5, 1)

    # The earlier model is untouched, and no half-written record is visible as
    # a model: the transaction either happened or it did not.
    assert target.read_text() == original
    assert ModelStore(str(tmp_path)).list_models() == [saved]


def test_load_model_expr_returns_an_expression_not_a_record(tmp_path):
    """The audit item: a stored model is an expression, not a JSON dict."""
    store = ModelStore(str(tmp_path))
    saved = store.save_model("m", sp.sympify("x**2 + 1"), ["x"], 0.99, 3)
    loaded = store.load_model_expr(saved)
    assert isinstance(loaded, sp.Expr)
    assert loaded == sp.sympify("x**2 + 1")
    assert store.load_model_expr("no_such_model") is None


def test_a_corrupt_record_is_reported_not_raised(tmp_path):
    store = ModelStore(str(tmp_path))
    saved = store.save_model("m", sp.Symbol("x"), ["x"], 0.9, 1)
    path = tmp_path / f"{saved}.json"
    record = json.loads(path.read_text())
    record["expression"] = "not ) a ( formula"
    path.write_text(json.dumps(record))
    assert ModelStore(str(tmp_path)).load_model_expr(saved) is None


# ---------------------------------------------------------------------------
# Critical 2 (continued): the service was unauthenticated and unthrottled
# ---------------------------------------------------------------------------


def test_the_api_is_open_only_when_no_key_is_configured(monkeypatch):
    monkeypatch.delenv("APEDP_API_KEY", raising=False)
    assert TestClient(create_app()).get("/health").status_code == 200


def test_a_configured_key_is_required_on_every_non_exempt_route(monkeypatch):
    monkeypatch.setenv("APEDP_API_KEY", "s3cret")
    client = TestClient(create_app())
    route = "/metrology/calibrate/verify?sensor_id=s"
    assert client.get(route).status_code == 401
    assert client.get(route, headers={"X-API-Key": "wrong"}).status_code == 401
    # 405 is this route's own method restriction -- i.e. auth let the call through.
    assert client.get(route, headers={"X-API-Key": "s3cret"}).status_code != 401


def test_liveness_and_documentation_do_not_need_a_key(monkeypatch):
    """A health check that needs a credential cannot gate traffic."""
    monkeypatch.setenv("APEDP_API_KEY", "s3cret")
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_the_rate_limiter_answers_429_past_the_budget(monkeypatch):
    monkeypatch.delenv("APEDP_API_KEY", raising=False)
    monkeypatch.setenv("APEDP_RATE_LIMIT_PER_MINUTE", "3")
    client = TestClient(create_app())
    codes = [client.get("/health").status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429
    assert client.get("/health").headers.get("Retry-After")


def test_a_nonsense_rate_limit_value_is_ignored_rather_than_fatal(monkeypatch):
    from apedp.api import DEFAULT_RATE_LIMIT_PER_MINUTE

    monkeypatch.setenv("APEDP_RATE_LIMIT_PER_MINUTE", "lots")
    assert rate_limit_per_minute() == DEFAULT_RATE_LIMIT_PER_MINUTE
    assert TestClient(create_app()).get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Major 7 (continued): a crashed repair probe was a clean verdict
# ---------------------------------------------------------------------------


def _exploding_limit(*args, **kwargs):
    raise ValueError("probe exploded")


def test_an_unanswered_probe_cannot_read_as_approved(monkeypatch):
    system = AgnosticClosedLoopAssuranceSystem()
    with monkeypatch.context() as patched:
        patched.setattr(repair_module, "limit_with_timeout", _exploding_limit)
        result = system.autonomous_stress_test_and_repair("exp(x)")
    assert result["status"] == "UNDETERMINED"
    assert result["probe_error_count"] >= 1
    assert result["unresolved_probe_count"] >= 1
    assert "not established" in result["validity_domain"]
    # The control: with the probes working, the same law is cleanly approved.
    assert AgnosticClosedLoopAssuranceSystem().autonomous_stress_test_and_repair("exp(x)")["status"] == "APPROVED"


def test_a_crashed_probe_with_a_measured_fallback_is_recorded_not_blocking(monkeypatch):
    """Recording every crash is not the same as failing every crash."""
    system = AgnosticClosedLoopAssuranceSystem()
    with monkeypatch.context() as patched:
        patched.setattr(repair_module, "limit_with_timeout", _exploding_limit)
        result = system.autonomous_stress_test_and_repair("1/(x**2 + 1)")
    assert result["probe_error_count"] >= 1
    assert result["unresolved_probe_count"] == 0
    assert result["status"] == "APPROVED"


def test_pipeline_refuses_a_repair_it_could_not_establish(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    monkeypatch.setattr(repair_module, "limit_with_timeout", _exploding_limit)

    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(
        pipeline,
        "discover",
        lambda *args, **kwargs: {
            "best_expression": "2*x + 1",
            "best_score": 1e-6,
            "r_squared": 0.999,
            "bic": -40.0,
            "aic": -42.0,
            "degenerate": False,
            "degenerate_reason": "",
            "search_stagnated": False,
            "stagnation": {"stagnated": False, "generations_without_improvement": 0},
        },
    )
    result = pipeline.run(str(csv_path), "y", variable_units={"x": "dimensionless"})
    assert result["status"] == "REFUSED"
    assert "undetermined" in result["reason"]
    assert pipeline.store.list_models() == []


# ---------------------------------------------------------------------------
# Minor: first-order GUM risk classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("formula", "risky"),
    [
        ("x**2", False),
        ("2*x + 1", False),
        ("exp(x)", False),
        ("sin(x)", False),
        ("1/x", True),
        ("1/x**3", True),
        ("sqrt(x)", True),
        ("x**(1/3)", True),
        ("x**Rational(-1, 2)", True),
        ("tan(x)", True),
        ("log(x)", True),
        ("asin(x)", True),
        ("1/sqrt(1 - x**2)", True),
    ],
)
def test_gum_risk_covers_poles_and_branch_cuts(formula, risky):
    """Keying on exponents -1/-2 and on log/sqrt alone missed x**-3, cube roots
    and every trigonometric pole."""
    assert has_singularity_risk(sp.sympify(formula)) is risky


def test_the_metrology_route_warns_for_a_pole_it_now_recognises(monkeypatch):
    """The classifier has to reach the response, not just exist."""
    client = TestClient(create_app())
    response = client.post(
        "/metrology/propagate",
        json={
            "expression": "tan(x)",
            "param_values": {"x": 0.25},
            "param_uncertainties": {"x": 0.01},
            "method": "gum",
        },
    )
    assert response.status_code == 200
    assert response.json()["warning"]


def test_the_metrology_auto_method_selects_monte_carlo_for_poles(monkeypatch):
    client = TestClient(create_app())
    response = client.post(
        "/metrology/propagate",
        json={
            "expression": "tan(x)",
            "param_values": {"x": 0.25},
            "param_uncertainties": {"x": 0.01},
            "method": "auto",
        },
    )
    assert response.status_code == 200
    assert response.json()["method"] == "monte_carlo"


# ---------------------------------------------------------------------------
# Minor: exact rational exponents (deliberately unchanged, now pinned)
# ---------------------------------------------------------------------------


def test_sqrt_a_half_power_and_a_rational_exponent_agree():
    """Why ``float(exponent)`` is kept in the Pow branch of the engine.

    pint stores an exact sympy rational but compares it *unequal* to the
    equivalent float, so returning the exact rational would make ``sqrt(x)`` and
    ``x**0.5`` disagree about one dimension. Normalising every exponent to a
    float is what keeps the three spellings consistent; this pins that invariant
    so a well-meant "exactness" change cannot silently break homogeneity.
    """
    assert (ureg.meter ** sp.Rational(1, 2))._units != (ureg.meter ** 0.5)._units

    engine = DimensionalAssuranceEngine()
    engine.register_variable("x", "meter")
    dimensions = {
        engine.check_expression_consistency(safe_parse_expr(f)).dimensionality
        for f in ("sqrt(x)", "x**0.5", "x**Rational(1, 2)")
    }
    assert len(dimensions) == 1


# ---------------------------------------------------------------------------
# Minor: the bad-row ingest threshold was an unexplained literal
# ---------------------------------------------------------------------------


def test_the_bad_row_threshold_is_overridable(tmp_path):
    path = tmp_path / "dirty.csv"
    path.write_text("x,y\n1,2\n2,4\nbad,6\n5,bad\n6,12\n")
    strict = DiscoveryPipeline(output_dir=str(tmp_path / "strict"))
    with pytest.raises(ValueError, match="max_bad_row_fraction"):
        strict.ingest_csv(str(path), "y")
    lenient = DiscoveryPipeline(output_dir=str(tmp_path / "lenient"), max_bad_row_fraction=0.5)
    ingested = lenient.ingest_csv(str(path), "y")
    assert len(ingested["data"]["y"]) == 3


# ---------------------------------------------------------------------------
# Noise: the suppression is scoped to deliberate probes
# ---------------------------------------------------------------------------


def test_deliberate_singular_evaluations_are_silent_but_real_ones_are_not():
    """~125 warnings per run buried the real ones; silencing everything would
    have buried them differently."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with quiet_float_warnings():
            np.sqrt(np.array([-1.0]))
        np.sqrt(np.array([-1.0]))  # outside the probe: still reported
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1

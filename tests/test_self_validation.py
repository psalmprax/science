import json

import numpy as np
import pytest
import sympy as sp

import self_validation
from apedp.persistence import ModelStore, deserialize_model, serialize_model
from self_validation import SelfValidationReport, SelfValidator


def test_harness_records_errors_and_continues(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    validator = SelfValidator()
    executed = []

    def broken():
        raise RuntimeError("deliberate test failure")

    def healthy():
        executed.append(True)
        validator.report.add("healthy check", True)

    for name in dir(validator):
        if name.startswith("validate_"):
            monkeypatch.setattr(validator, name, healthy)
    monkeypatch.setattr(validator, "validate_bayesian_selection", broken)

    report = validator.run_all()
    summary = report.summary()
    assert executed
    assert summary["failed"] == 1
    assert "deliberate test failure" in summary["results"][0]["error"]
    saved = json.loads((tmp_path / "self_validation_report.json").read_text())
    assert saved == summary


def test_report_serializes_numpy_bool():
    report = SelfValidationReport()
    report.add("numpy bool check", np.bool_(True))
    summary = report.summary()
    assert summary["results"][0]["passed"] is True
    json.dumps(summary)


@pytest.mark.parametrize("failed, expected", [(False, 0), (True, 1)])
def test_cli_exit_status(monkeypatch, capsys, failed, expected):
    report = SelfValidationReport()
    report.add("check", not failed)
    monkeypatch.setattr(self_validation, "run_self_validation", lambda: report)
    assert self_validation.main() == expected
    assert "SUMMARY:" in capsys.readouterr().out


@pytest.mark.parametrize("expression", [sp.Integer(0), sp.Symbol("x")**2])
def test_persistence_latex_round_trip(tmp_path, expression):
    store = ModelStore(str(tmp_path))
    model_id = store.save_model("test", expression, ["x"], 1.0, 2)
    loaded = ModelStore(str(tmp_path)).load_model(model_id)
    assert loaded["sympy_expr"] == sp.latex(expression)
    restored = deserialize_model(serialize_model(expression, ["x"], {}))
    assert restored["sympy_expr"] == sp.latex(expression)


def test_report_records_which_checks_have_an_independent_oracle():
    report = SelfValidationReport()
    report.add("external check", True, oracle="closed form derived by hand")
    report.add("self-referential check", True)
    summary = report.summary()
    assert summary["checks_with_independent_oracle"] == 1
    assert summary["self_referential_checks"] == 1
    assert summary["results"][0]["oracle"] == "closed form derived by hand"
    json.dumps(summary)


def test_bayes_check_fails_when_evidence_is_a_summed_likelihood(monkeypatch):
    """The oracle must reject the pre-W5 behaviour: a summed likelihood called evidence.

    This is the mutation test for the check itself. If it passes with the wrong
    implementation injected, it is not measuring anything.
    """

    def fake_marginal_likelihood(log_likelihood_fn, prior_logpdf, **kwargs):
        point = kwargs.get("map_params")
        if point is None:
            bounds = np.asarray(kwargs["bounds"], dtype=float)
            point = bounds.mean(axis=1)
        total = float(np.exp(log_likelihood_fn(np.asarray(point, dtype=float))))
        return self_validation.ModelEvidence(float(np.log(total)), "summed-likelihood", 0)

    monkeypatch.setattr(
        self_validation.BayesianRefinementEngine,
        "marginal_likelihood",
        staticmethod(fake_marginal_likelihood),
    )
    validator = SelfValidator()
    validator.validate_bayesian_selection()
    results = {r["test"]: r for r in validator.report.results}
    evidence_check = results["Bayesian: quadrature evidence equals the conjugate closed form"]
    assert not evidence_check["passed"]
    assert "closed form" in evidence_check["details"]
    assert any(not r["passed"] for r in validator.report.results)


def test_anomaly_repair_check_fails_for_a_repair_that_does_not_regularize(monkeypatch):
    """A repair is judged at the singular point, not by having run.

    The injected "repair" returns a model that is still 1/Z_j, so it is
    singular exactly where the audit claims to have fixed it.
    """
    validator = SelfValidator()
    validator.repair.audit_and_repair_bayes_factor()
    z_j = sp.Symbol("Z_j")
    validator.platform.register_model_spec(
        "Useless_Repair", sp.Integer(1) / z_j, "dimensionless"
    )
    monkeypatch.setattr(
        validator.repair, "audit_and_repair_bayes_factor", lambda: "Useless_Repair"
    )
    validator.validate_anomaly_repair()
    results = {r["test"]: r for r in validator.report.results}
    finite_check = results["Anomaly: repaired Bayes factor Z_i/Z_j at Z_j -> 0 is finite at the singularity"]
    assert not finite_check["passed"]
    # The pole check on the published law still passes: the law does diverge there.
    assert results["Anomaly: unregularized Bayes factor diverges at the singular point"]["passed"]


def test_discovery_rejects_constant_prediction(monkeypatch):
    monkeypatch.setattr(
        self_validation.MultiVariableSRE,
        "run_discovery",
        lambda *args, **kwargs: (sp.Integer(7), 32.0, [{"best_score": 32.0}]),
    )
    validator = SelfValidator()
    validator.validate_evolutionary_search()
    fits = [r for r in validator.report.results if "held-out fit" in r["test"]]
    assert len(fits) == 2
    assert all(not r["passed"] for r in fits)

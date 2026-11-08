import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_dfthcsos_arm1_receipts.py"
spec = importlib.util.spec_from_file_location("audit_dfthcsos_arm1_receipts", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def receipt(*, score=1000.0, lam=2.0, q=10, t=50.0, sha="a"):
    return {
        "status": "PASS", "score": score, "lambda_eff": lam,
        "num_qubits": q, "toffoli_count": t,
        "submission_sha256": sha * 64,
        "resources": {"angles_within_cap": True},
    }


def test_candidate_that_passes_and_improves_is_only_eligible_after_cap_flag():
    baseline = receipt(score=1000, lam=2, q=10, t=50, sha="a")
    candidate = receipt(score=900, lam=2, q=10, t=45, sha="b")
    out = mod.audit(baseline, candidate)
    assert out["promotion"] == "PROMOTE_ELIGIBLE_PENDING_FULL_CONSTRAINT_AUDIT"
    assert out["cost_gain_baseline_over_candidate"] == pytest.approx(1000 / 900)


def test_non_improving_candidate_is_not_promoted():
    baseline = receipt(score=1000, lam=2, q=10, t=50, sha="a")
    candidate = receipt(score=1100, lam=2, q=10, t=55, sha="b")
    assert mod.audit(baseline, candidate)["promotion"] == "DO_NOT_PROMOTE"


def test_score_mismatch_fails_closed():
    baseline = receipt(score=1001, lam=2, q=10, t=50, sha="a")
    with pytest.raises(ValueError, match="score mismatch"):
        mod.audit(baseline, receipt(score=900, lam=2, q=10, t=45, sha="b"))


def test_explicit_qubit_cap_failure_fails_closed():
    baseline = receipt(score=1000, lam=2, q=10, t=50, sha="a")
    candidate = receipt(score=900, lam=2, q=10, t=45, sha="b")
    candidate["resources"]["angles_within_cap"] = False
    with pytest.raises(ValueError, match="angles_within_cap is false"):
        mod.audit(baseline, candidate)

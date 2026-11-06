"""Arm 3 is the interaction arm: it needs Arm 1 and Arm 2 to have passed alone.

Status: audit-only

The planner and auditor mirror the Arm 1/Arm 2 campaign scripts. Arm 3 asks one
question the isolated arms cannot: does the combination add up? Its own receipts are
audited against the shared baseline *and* both isolated arms, and the additivity
ratio is reported rather than assumed.
"""
from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(name: str, relative: str):
    spec = spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


plan = load("plan_dfthcsos_arm3", "scripts/plan_dfthcsos_arm3.py")
auditor = load("audit_dfthcsos_arm3_receipts", "scripts/audit_dfthcsos_arm3_receipts.py")

BASELINE_SCORE = 100_000.0


def receipt(
    score: float,
    digest: str,
    *,
    arm: str = auditor.ARM3,
    arm1_enabled: bool = True,
    angle_qubits: int = 119,
    status: str = "PASS",
    source: str = "c" * 64,
    npz: str = "b" * 64,
) -> dict:
    angles = [
        {
            "name": f"angle_k_{i}.kmb",
            "num_qubits": angle_qubits,
            "sha256": f"{i + 1:064x}",
            "lookup_semantics_pass": True,
            "phase_semantics_pass": True,
        }
        for i in range(10)
    ]
    # score == lambda_eff * Q * T, as the official verifier reports it.
    return {
        "official_receipt": {
            "status": status,
            "score": score,
            "lambda_eff": 10.0,
            "num_qubits": 100.0,
            "toffoli_count": score / 1000.0,
            "submission_sha256": digest,
        },
        "experiment_manifest": {
            "arm": arm,
            "arm1_selectswap_enabled": arm1_enabled,
            "other_transformations_enabled": [],
            "input_npz_unchanged": True,
            "submission_npz_sha256": npz,
            "source_members_sha256": {"source": source},
            "lookup_semantics_pass": True,
            "phase_semantics_pass": True,
            "angle_subcircuits": angles,
        },
    }


def arm1_receipt(score: float) -> dict:
    """Arm 1 receipts are raw verifier receipts, not Arm 2-style wrappers."""
    return {
        "status": "PASS",
        "score": score,
        "lambda_eff": 10.0,
        "num_qubits": 100.0,
        "toffoli_count": score / 1000.0,
        "submission_sha256": "e" * 64,
    }


def baseline() -> dict:
    return receipt(BASELINE_SCORE, "a" * 64, arm=auditor.ARM2, arm1_enabled=False)


def test_arm3_cannot_be_planned_without_both_isolated_receipts():
    with pytest.raises(ValueError, match="interaction arm"):
        plan.make_plan()
    with pytest.raises(ValueError, match="interaction arm"):
        plan.make_plan(baseline=baseline(), arm1_isolated=arm1_receipt(80_000.0))


def test_plan_refuses_an_isolated_arm_that_did_not_pass_or_did_not_help():
    failing = arm1_receipt(80_000.0)
    failing["status"] = "FAIL"
    with pytest.raises(ValueError, match="must be PASS"):
        plan.make_plan(
            baseline=baseline(), arm1_isolated=failing, arm2_isolated=receipt(90_000.0, "d" * 64)
        )
    with pytest.raises(ValueError, match="must strictly beat the baseline"):
        plan.make_plan(
            baseline=baseline(),
            arm1_isolated=arm1_receipt(80_000.0),
            arm2_isolated=receipt(BASELINE_SCORE, "d" * 64),
        )


def test_plan_declares_both_levers_and_claims_no_cost():
    rows = plan.make_plan(
        baseline=baseline(), arm1_isolated=arm1_receipt(80_000.0), arm2_isolated=receipt(90_000.0, "d" * 64)
    )
    assert [r["angle_batch_size_J"] for r in rows] == [4, 6]
    assert all(r["levers"] == {plan.ARM1: True, plan.ARM2: True} for r in rows)
    assert all(r["arm"] == plan.ARM3 for r in rows)
    assert all(r["official_cost"] is None for r in rows)
    assert all(r["status"] == "PLAN_ONLY_NEEDS_COMBINED_RECEIPT" for r in rows)
    assert all(r["additivity_is_hypothesis"] is True for r in rows)
    assert all(r["peak_angle_qubits_estimate"] is None for r in rows)
    # The composition is an estimate of the hypothesis, not a measured cost.
    assert rows[0]["arm1_isolated_gain"] == pytest.approx(1.25)
    assert rows[0]["arm2_isolated_gain"] == pytest.approx(BASELINE_SCORE / 90_000.0)
    assert rows[0]["expected_multiplicative_gain"] == pytest.approx(
        rows[0]["arm1_isolated_gain"] * rows[0]["arm2_isolated_gain"]
    )


def test_auditor_promotes_a_combined_improvement_that_beats_both_arms():
    report = auditor.audit(
        baseline(),
        receipt(70_000.0, "f" * 64),
        arm1_receipt(80_000.0),
        receipt(90_000.0, "d" * 64, arm=auditor.ARM2, arm1_enabled=False),
    )
    assert report["promotion"] == "PROMOTION_ELIGIBLE_PENDING_HUMAN_REVIEW"
    assert report["arm3_declares_both_levers"] is True
    assert report["candidate_beats_both_isolated_arms"] is True
    assert report["cost_gain_baseline_over_candidate"] == pytest.approx(100_000.0 / 70_000.0)
    assert report["interaction"] == "ADDITIVE_WITHIN_TOLERANCE"


def test_auditor_refuses_a_combination_that_does_not_beat_the_isolated_arms():
    report = auditor.audit(
        baseline(),
        receipt(80_000.0, "f" * 64),
        arm1_receipt(80_000.0),
        receipt(90_000.0, "d" * 64, arm=auditor.ARM2, arm1_enabled=False),
    )
    assert report["promotion"] == "DO_NOT_PROMOTE"
    assert report["candidate_beats_both_isolated_arms"] is False
    assert report["interaction"] == "SUB_ADDITIVE"


def test_auditor_requires_the_candidate_to_declare_both_levers():
    with pytest.raises(ValueError, match="arm1_selectswap_enabled=true"):
        auditor.audit(
            baseline(),
            receipt(70_000.0, "f" * 64, arm1_enabled=False),
            arm1_receipt(80_000.0),
            receipt(90_000.0, "d" * 64, arm=auditor.ARM2, arm1_enabled=False),
        )


def test_auditor_refuses_without_an_isolated_pass():
    with pytest.raises(ValueError, match="must be PASS"):
        auditor.audit(
            baseline(),
            receipt(70_000.0, "f" * 64),
            arm1_receipt(80_000.0),
            receipt(90_000.0, "d" * 64, arm=auditor.ARM2, arm1_enabled=False, status="FAIL"),
        )


def test_additivity_classes_are_measured_not_assumed():
    assert auditor.classify_interaction(1.20 * 1.3 * 1.1, 1.2, 1.3)["interaction"] == "SUPER_ADDITIVE"
    assert auditor.classify_interaction(1.2 * 1.3, 1.2, 1.3)["interaction"] == (
        "ADDITIVE_WITHIN_TOLERANCE"
    )
    assert auditor.classify_interaction(1.0, 1.2, 1.3)["interaction"] == "SUB_ADDITIVE"

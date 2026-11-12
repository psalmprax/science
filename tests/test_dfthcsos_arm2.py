from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def load(name, rel):
    spec = spec_from_file_location(name, ROOT / rel)
    assert spec and spec.loader
    mod = module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

plan = load("plan_dfthcsos_arm2", "scripts/plan_dfthcsos_arm2.py")
auditor = load("audit_dfthcsos_arm2_receipts", "scripts/audit_dfthcsos_arm2_receipts.py")

def test_plan_does_not_claim_unknown_width_or_cost():
    rows = plan.make_plan()
    assert [r["angle_batch_size_J"] for r in rows] == [4, 6]
    assert [r["block_size"] for r in rows] == [2, 4]
    assert all(r["peak_angle_qubits_estimate"] is None for r in rows)
    assert all(r["official_cost"] is None and r["status"] == "PLAN_ONLY_NEEDS_MEASURED_WIDTHS" for r in rows)
    assert [r["total_qubits_estimate"] for r in rows] == [682, 706]

def test_plan_uses_measured_delta_and_fixup_width_inputs():
    rows = plan.make_plan(buffered_delta_bits_per_entry=11, phase_fixup_qubits=0)
    assert rows[0]["peak_angle_qubits_estimate"] == 82
    assert rows[0]["feasibility_by_width_estimate"] is True
    assert rows[1]["peak_angle_qubits_estimate"] == 128
    assert rows[1]["feasibility_by_width_estimate"] is True

def test_rejects_width_over_cap():
    rows = plan.make_plan(buffered_delta_bits_per_entry=12, phase_fixup_qubits=1)
    assert rows[1]["peak_angle_qubits_estimate"] == 132
    assert rows[1]["status"] == "REJECTED_WIDTH_ESTIMATE"

def receipt(score, lam, q, t, digest, sourcehash, config, maxangle=119, status="PASS"):
    angles = []
    for i in range(10):
        angles.append({"name": f"angle_k_{i}.kmb", "num_qubits": maxangle,
                       "sha256": f"{i+1:064x}", "lookup_semantics_pass": True,
                       "phase_semantics_pass": True})
    return {"official_receipt": {"status": status, "score": score, "lambda_eff": lam,
              "num_qubits": q, "toffoli_count": t, "submission_sha256": digest},
            "experiment_manifest": {"arm":"arm2_reed_muller_angle_buffering_only",
              "arm1_selectswap_enabled":False,"other_transformations_enabled":[],
              "input_npz_unchanged":True,"submission_npz_sha256":sourcehash,
              "source_members_sha256":config,"lookup_semantics_pass":True,
              "phase_semantics_pass":True,"angle_subcircuits":angles}}

def test_auditor_promotes_only_verified_isolated_improvement():
    # baseline cost = 10 * 100 * 100 = 100000; candidate 90 * 100 * 100 = 90000
    b=receipt(100000, 10, 100, 100, "a"*64, "b"*64, {"source":"c"*64})
    c=receipt(90000, 9, 100, 100, "d"*64, "b"*64, {"source":"c"*64}, maxangle=120)
    report=auditor.audit(b,c)
    assert report["promotion"] == "PROMOTION_ELIGIBLE_PENDING_HUMAN_REVIEW"
    c["experiment_manifest"]["arm1_selectswap_enabled"] = True
    try:
        auditor.audit(b,c)
    except ValueError as e:
        assert "arm1_selectswap_enabled=false" in str(e)
    else:
        raise AssertionError("Arm 1 must be explicitly disabled")

def test_auditor_rejects_bad_score_identity():
    b=receipt(100001, 10, 100, 100, "a"*64, "b"*64, {"source":"c"*64})
    c=receipt(90000, 9, 100, 100, "d"*64, "b"*64, {"source":"c"*64})
    try:
        auditor.audit(b,c)
    except ValueError as e:
        assert "does not reconcile" in str(e)
    else:
        raise AssertionError("bad score identity should be rejected")

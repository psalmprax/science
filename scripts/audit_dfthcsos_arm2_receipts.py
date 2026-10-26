#!/usr/bin/env python3
"""Audit Arm 2 metadata plus official DFTHCSOS score receipts; does not verify circuits."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return obj


def finite(obj: dict[str, Any], key: str, label: str) -> float:
    x = obj.get(key)
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{label}.{key} must be a finite number")
    return float(x)


def sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return value.lower()


def check_receipt(w: dict[str, Any], label: str) -> dict[str, Any]:
    r = w.get("official_receipt")
    m = w.get("experiment_manifest")
    if not isinstance(r, dict) or not isinstance(m, dict):
        raise ValueError(f"{label} must contain official_receipt and experiment_manifest objects")
    status = r.get("status")
    if not isinstance(status, str):
        raise ValueError(f"{label}.official_receipt.status is required")
    score, lam, q, t = (finite(r, k, f"{label}.official_receipt") for k in
                        ("score", "lambda_eff", "num_qubits", "toffoli_count"))
    if min(score, lam, q, t) <= 0:
        raise ValueError(f"{label}: score, lambda_eff, qubits and Toffolis must be positive")
    digest = sha(r.get("submission_sha256") or r.get("circuit_sha256"), f"{label}.receipt hash")
    relative = abs(lam*q*t-score)/score
    if relative > 1e-9:
        raise ValueError(f"{label}: score does not reconcile with lambda_eff*Q*T (relative error {relative:.3g})")
    if m.get("arm") != "arm2_reed_muller_angle_buffering_only":
        raise ValueError(f"{label}: manifest arm must be arm2_reed_muller_angle_buffering_only")
    if m.get("arm1_selectswap_enabled") is not False:
        raise ValueError(f"{label}: explicitly set arm1_selectswap_enabled=false for isolation")
    if m.get("input_npz_unchanged") is not True:
        raise ValueError(f"{label}: input_npz_unchanged must be true")
    if m.get("lookup_semantics_pass") is not True or m.get("phase_semantics_pass") is not True:
        raise ValueError(f"{label}: lookup and phase semantics must both pass")
    input_hash = sha(m.get("submission_npz_sha256"), f"{label}.submission_npz_sha256")
    angle_rows = m.get("angle_subcircuits")
    if not isinstance(angle_rows, list) or not angle_rows:
        raise ValueError(f"{label}.angle_subcircuits must be a non-empty list")
    expected_names = {f"angle_k_{i}.kmb" for i in range(10)}
    seen: set[str] = set()
    max_qubits = 0
    for i, a in enumerate(angle_rows):
        if not isinstance(a, dict):
            raise ValueError(f"{label}.angle_subcircuits[{i}] must be an object")
        name = a.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"{label}: angle circuit names must be unique non-empty strings")
        seen.add(name)
        aq = a.get("num_qubits")
        if isinstance(aq, bool) or not isinstance(aq, int) or aq < 1:
            raise ValueError(f"{label}.{name}.num_qubits must be a positive integer")
        if a.get("lookup_semantics_pass") is not True or a.get("phase_semantics_pass") is not True:
            raise ValueError(f"{label}.{name}: angle lookup and phase semantics must pass")
        sha(a.get("sha256"), f"{label}.{name}.sha256")
        max_qubits = max(max_qubits, aq)
    if seen != expected_names:
        missing = sorted(expected_names - seen)
        extra = sorted(seen - expected_names)
        raise ValueError(f"{label}: expected all 10 ANGLE circuits angle_k_0.kmb..angle_k_9.kmb; missing={missing}, extra={extra}")
    source_hashes = m.get("source_members_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError(f"{label}.source_members_sha256 must be a non-empty source-member hash map")
    for member, digest_value in source_hashes.items():
        if not isinstance(member, str) or not member:
            raise ValueError(f"{label}: source-member names must be non-empty strings")
        sha(digest_value, f"{label}.source_members_sha256[{member}]")
    if max_qubits > 128:
        raise ValueError(f"{label}: max ANGLE width {max_qubits} exceeds 128-qubit cap")
    return {"status": status, "score": score, "lambda_eff": lam, "num_qubits": q,
            "toffoli_count": t, "recomputed_score": lam*q*t, "score_relative_error": relative,
            "official_hash": digest, "submission_npz_sha256": input_hash,
            "angle_circuit_count": len(angle_rows), "max_angle_qubits": max_qubits,
            "manifest": m}


def audit(bw: dict[str, Any], cw: dict[str, Any]) -> dict[str, Any]:
    b = check_receipt(bw, "baseline")
    c = check_receipt(cw, "candidate")
    if b["status"].upper() != "PASS":
        raise ValueError("baseline official receipt must be PASS")
    same_input = b["submission_npz_sha256"] == c["submission_npz_sha256"]
    same_config = b["manifest"].get("source_members_sha256") == c["manifest"].get("source_members_sha256")
    arm2_only = (c["manifest"].get("arm1_selectswap_enabled") is False and
                 c["manifest"].get("other_transformations_enabled") == [])
    eligible = (c["status"].upper() == "PASS" and c["score"] < b["score"] and
                same_input and same_config and arm2_only and c["max_angle_qubits"] <= 128)
    return {"benchmark": "dfthcsos", "arm": "arm2_reed_muller_angle_buffering_only",
            "baseline": b, "candidate": c, "same_input_npz_hash": same_input,
            "same_source_members_manifest": same_config, "arm2_isolation_declared": arm2_only,
            "cost_gain_baseline_over_candidate": b["score"]/c["score"],
            "relative_cost_reduction": (b["score"]-c["score"])/b["score"],
            "promotion": "PROMOTION_ELIGIBLE_PENDING_HUMAN_REVIEW" if eligible else "DO_NOT_PROMOTE",
            "warning": "This audits receipt arithmetic and manifest assertions only; it does not independently prove semantic equivalence or authenticate the official verifier binary."}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    report=audit(read(a.baseline), read(a.candidate))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output":str(a.output),"promotion":report["promotion"],
                      "gain":report["cost_gain_baseline_over_candidate"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

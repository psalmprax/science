#!/usr/bin/env python3
"""Audit saved official DFTHCSOS baseline/candidate receipts for Arm 1.

This validates receipt consistency and the promotion gate; it does not build circuits
or replace the official verifier.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"receipt must be a JSON object: {path}")
    return obj


def require_number(obj: dict[str, Any], key: str, label: str) -> float:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}.{key} must be a finite number")
    return float(value)


def normalize_receipt(obj: dict[str, Any], label: str) -> dict[str, Any]:
    status = obj.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"{label}.status is required")
    score = require_number(obj, "score", label)
    lam = require_number(obj, "lambda_eff", label)
    qubits = require_number(obj, "num_qubits", label)
    toffoli = require_number(obj, "toffoli_count", label)
    if score <= 0 or lam <= 0 or qubits <= 0 or toffoli <= 0:
        raise ValueError(f"{label}: score, lambda_eff, num_qubits and toffoli_count must be positive")
    digest = obj.get("submission_sha256") or obj.get("circuit_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
        raise ValueError(f"{label} needs a 64-hex-character submission_sha256 or circuit_sha256")
    resources = obj.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError(f"{label}.resources must be an object when supplied")
    cap_flag = resources.get("angles_within_cap")
    if cap_flag is False:
        raise ValueError(f"{label}: resources.angles_within_cap is false")
    # A full-precision verifier receipt should reproduce its own reported cost.
    recomputed = lam * qubits * toffoli
    relative_error = abs(recomputed - score) / score
    if relative_error > 1e-9:
        raise ValueError(
            f"{label} score mismatch: reported={score:.15g}, "
            f"lambda_eff*Q*T={recomputed:.15g}, relative_error={relative_error:.3g}. "
            "Use full-precision verifier receipt fields."
        )
    return {
        "status": status,
        "score": score,
        "lambda_eff": lam,
        "num_qubits": int(qubits) if qubits.is_integer() else qubits,
        "toffoli_count": toffoli,
        "recomputed_score": recomputed,
        "score_relative_error": relative_error,
        "sha256": digest.lower(),
        "angles_within_cap": cap_flag,
        "resources": resources,
        "metrics": obj.get("metrics", {}),
    }


def audit(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base = normalize_receipt(baseline, "baseline")
    cand = normalize_receipt(candidate, "candidate")
    cand_pass = cand["status"].upper() == "PASS"
    base_pass = base["status"].upper() == "PASS"
    if not base_pass:
        raise ValueError("baseline receipt is not PASS; cannot use it as the incumbent")
    improves = cand["score"] < base["score"]
    if cand["angles_within_cap"] is None:
        cap_note = "Verifier receipt does not expose resources.angles_within_cap; confirm every ANGLE circuit is <=128 qubits from the full verifier output."
    else:
        cap_note = "Explicit resources.angles_within_cap flag checked."
    return {
        "benchmark": "dfthcsos",
        "arm": "selectswap_only",
        "promotion": "PROMOTE_ELIGIBLE_PENDING_FULL_CONSTRAINT_AUDIT" if (cand_pass and improves and cand["angles_within_cap"] is True) else "DO_NOT_PROMOTE",
        "baseline": base,
        "candidate": cand,
        "cost_gain_baseline_over_candidate": base["score"] / cand["score"],
        "relative_cost_reduction": (base["score"] - cand["score"]) / base["score"],
        "candidate_official_pass": cand_pass,
        "candidate_strictly_improves": improves,
        "same_submission_hash": base["sha256"] == cand["sha256"],
        "cap_note": cap_note,
        "warning": "Receipt arithmetic is not an independent correctness proof. Confirm Arm 1 isolation, unchanged official verifier, all chemistry constraints, source/member hashes and <=128-qubit limit before promotion.",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = audit(load_json(a.baseline), load_json(a.candidate))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(a.output), "promotion": report["promotion"],
                      "gain": report["cost_gain_baseline_over_candidate"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

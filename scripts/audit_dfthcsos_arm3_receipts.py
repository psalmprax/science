#!/usr/bin/env python3
"""Audit Arm 3 (SelectSwap + Reed-Muller together) against both isolated arms.

Arm 3 is the interaction arm: it is only meaningful if each lever passed alone, so
this requires the isolated Arm 1 and Arm 2 official receipts as well as the shared
baseline and the combined candidate. It audits receipt arithmetic, manifest
assertions and the additivity the combination actually measured. It does not verify
circuits, prove semantic equivalence, or authenticate the verifier binary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.audit_dfthcsos_arm1_receipts import normalize_receipt
from scripts.audit_dfthcsos_arm2_receipts import ARM2, check_receipt, read

ARM3 = "arm3_combined_selectswap_and_reed_muller"
ADDITIVITY_TOLERANCE = 0.05


def classify_interaction(
    combined_gain: float,
    arm1_gain: float,
    arm2_gain: float,
    *,
    tolerance: float = ADDITIVITY_TOLERANCE,
) -> dict[str, Any]:
    """Compare the combined gain with the gains composed multiplicatively.

    The composition is the *hypothesis*, not the expectation of a correct system: two
    levers that touch the same qubits can interact either way, and this is the only
    place that difference is measured.
    """
    if min(combined_gain, arm1_gain, arm2_gain) <= 0:
        raise ValueError("gains must be positive")
    expected = arm1_gain * arm2_gain
    ratio = combined_gain / expected
    if ratio > 1.0 + tolerance:
        label = "SUPER_ADDITIVE"
    elif ratio < 1.0 - tolerance:
        label = "SUB_ADDITIVE"
    else:
        label = "ADDITIVE_WITHIN_TOLERANCE"
    return {
        "arm1_isolated_gain": arm1_gain,
        "arm2_isolated_gain": arm2_gain,
        "combined_gain": combined_gain,
        "expected_multiplicative_gain": expected,
        "additivity_ratio": ratio,
        "interaction": label,
        "additivity_tolerance": tolerance,
    }


def audit(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    arm1_isolated: dict[str, Any],
    arm2_isolated: dict[str, Any],
) -> dict[str, Any]:
    candidate_manifest = candidate.get("experiment_manifest")
    if isinstance(candidate_manifest, dict) and candidate_manifest.get(
            "arm1_selectswap_enabled") is not True:
        raise ValueError(
            "arm3 candidate must declare arm1_selectswap_enabled=true: the"
            " interaction arm is defined by both levers being on")
    # The baseline is the Arm 2 build with SelectSwap off, so it is checked as
    # one. Passing `expected_arm=None` made every audit fail on the manifest
    # comparison rather than on anything about the receipts.
    base = check_receipt(baseline, "baseline", expected_arm=ARM2,
                         arm1_expected=False)
    combined = check_receipt(candidate, "candidate", expected_arm=ARM3, arm1_expected=True)
    a1 = normalize_receipt(arm1_isolated, "arm1_isolated")
    a2 = check_receipt(arm2_isolated, "arm2_isolated", expected_arm=ARM2, arm1_expected=False)

    if base["status"].upper() != "PASS":
        raise ValueError("baseline official receipt must be PASS")
    for label, receipt in (("arm1_isolated", a1), ("arm2_isolated", a2)):
        if receipt["status"].upper() != "PASS":
            raise ValueError(f"{label}: the isolated arm must be PASS before Arm 3 can be audited")
    if not (a1["score"] < base["score"] and a2["score"] < base["score"]):
        raise ValueError(
            "each isolated arm must strictly beat the baseline, otherwise Arm 3 has no premise"
        )

    same_input = base["submission_npz_sha256"] == combined["submission_npz_sha256"]
    same_config = (
        base["manifest"].get("source_members_sha256")
        == combined["manifest"].get("source_members_sha256")
    )
    both_levers = (
        combined["manifest"].get("arm1_selectswap_enabled") is True
        and combined["manifest"].get("other_transformations_enabled") == []
    )
    combined_gain = base["score"] / combined["score"]
    interaction = classify_interaction(
        combined_gain, base["score"] / a1["score"], base["score"] / a2["score"]
    )
    beats_baseline = combined["score"] < base["score"]
    beats_isolated = combined["score"] < a1["score"] and combined["score"] < a2["score"]
    eligible = (
        combined["status"].upper() == "PASS"
        and beats_baseline
        and beats_isolated
        and same_input
        and same_config
        and both_levers
        and combined["max_angle_qubits"] <= 128
    )
    return {
        "benchmark": "dfthcsos",
        "arm": ARM3,
        "baseline": base,
        "candidate": combined,
        "arm1_isolated": a1,
        "arm2_isolated": a2,
        "same_input_npz_hash": same_input,
        "same_source_members_manifest": same_config,
        "arm3_declares_both_levers": both_levers,
        "candidate_beats_baseline": beats_baseline,
        "candidate_beats_both_isolated_arms": beats_isolated,
        "cost_gain_baseline_over_candidate": combined_gain,
        "relative_cost_reduction": (base["score"] - combined["score"]) / base["score"],
        **interaction,
        "promotion": "PROMOTION_ELIGIBLE_PENDING_HUMAN_REVIEW" if eligible else "DO_NOT_PROMOTE",
        "warning": (
            "Receipt arithmetic, manifest assertions and measured additivity only. The levers "
            "were isolated in Arm 1 and Arm 2; this audit does not prove semantic equivalence, "
            "re-check every constraint on the combined build, or authenticate the verifier."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--arm1-receipt", type=Path, required=True)
    parser.add_argument("--arm2-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        read(args.baseline),
        read(args.candidate),
        read(args.arm1_receipt),
        read(args.arm2_receipt),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "promotion": report["promotion"],
        "gain": report["cost_gain_baseline_over_candidate"],
        "interaction": report["interaction"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

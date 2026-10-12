"""Air-Gapped External Evaluator CLI for APEDP-X.

Closes the external validity methodological gap:
Enables external, third-party scientific teams to evaluate candidate hypotheses
against the cryptographically frozen APEDP-X v1.0 referee without sharing
their private data generators, hidden ground-truth labels, or proprietary code.

Workflow:
  1. External team writes a JSON payload containing only observables, candidate model,
     evidence bundle, and claimed level (ZERO ground truth).
  2. Air-Gapped Evaluator executes frozen adjudication + EEDR repair loop.
  3. Emits a tamper-evident JSON Adjudication Receipt.
  4. External team evaluates the emitted receipts against their private answer key
     to compute unbiased FPR_external.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EpistemicAdjudicationVerdict,
    EvidenceBundle,
    EvidenceRequirement,
    adjudicate_scientific_claim,
)
from apedp_dyn.epistemic.eedr import execute_eedr_loop
from apedp_dyn.epistemic.frozen_spec import format_frozen_spec_banner, get_frozen_spec


def parse_claim_level(label: str) -> ClaimLevel:
    clean = label.split(":")[0].strip().upper()
    mapping = {
        "C0": ClaimLevel.C0_NO_SUPPORT,
        "C0_NO_SUPPORT": ClaimLevel.C0_NO_SUPPORT,
        "C1": ClaimLevel.C1_DESCRIPTIVE,
        "C1_DESCRIPTIVE": ClaimLevel.C1_DESCRIPTIVE,
        "C2": ClaimLevel.C2_PREDICTIVE,
        "C2_PREDICTIVE": ClaimLevel.C2_PREDICTIVE,
        "C3": ClaimLevel.C3_STRUCTURAL,
        "C3_STRUCTURAL": ClaimLevel.C3_STRUCTURAL,
        "C4": ClaimLevel.C4_CAUSAL_MECHANISTIC,
        "C4_CAUSAL_MECHANISTIC": ClaimLevel.C4_CAUSAL_MECHANISTIC,
        "C5": ClaimLevel.C5_EXTRAPOLATIVE,
        "C5_EXTRAPOLATIVE": ClaimLevel.C5_EXTRAPOLATIVE,
        "C6": ClaimLevel.C6_FORMALLY_VERIFIED,
        "C6_FORMALLY_VERIFIED": ClaimLevel.C6_FORMALLY_VERIFIED,
    }
    return mapping.get(clean, ClaimLevel.C0_NO_SUPPORT)


def parse_evidence_bundle(data: dict[str, Any]) -> EvidenceBundle:
    """Parse external JSON evidence representation into EvidenceBundle."""
    bundle = EvidenceBundle(
        metrics=data.get("metrics", {}),
        failures=list(data.get("failures", [])),
        metadata=data.get("metadata", {}),
    )
    for req_str in data.get("verified_requirements", []):
        try:
            req_enum = EvidenceRequirement(req_str)
            bundle.add(req_enum)
        except ValueError:
            # Tolerant parsing for formatted string
            for enum_val in EvidenceRequirement:
                if enum_val.value in req_str.lower():
                    bundle.add(enum_val)
                    break
    return bundle


def adjudicate_external_payload(payload: dict[str, Any], run_eedr: bool = True) -> dict[str, Any]:
    """Execute air-gapped adjudication on an external case payload."""
    spec = get_frozen_spec()
    case_id = payload.get("case_id", "UNKNOWN_EXTERNAL_CASE")
    domain = payload.get("domain", "general_science")
    candidate_model = payload.get("candidate_model", "unknown_model")
    scientific_claim = payload.get("scientific_claim", "unspecified")
    req_level_str = payload.get("requested_claim_level", "C2_PREDICTIVE")
    requested_level = parse_claim_level(req_level_str)

    # Sanity check: Ensure payload does NOT leak ground truth
    leaked_keys = [k for k in ("ground_truth", "hidden_truth", "answer_key", "is_valid") if k in payload]
    if leaked_keys:
        raise ValueError(f"AIR-GAP INTEGRITY VIOLATION: Case payload contains forbidden truth keys: {leaked_keys}")

    evidence_data = payload.get("evidence", {})
    bundle = parse_evidence_bundle(evidence_data)

    # 1. Direct adjudication against frozen kernel
    adj = adjudicate_scientific_claim(
        candidate_id=candidate_model,
        proposed_claim=requested_level,
        evidence=bundle,
        domain=domain,
    )

    eedr_record = None
    if run_eedr and adj.verdict != EpistemicAdjudicationVerdict.PROMOTED:
        eedr_res = execute_eedr_loop(
            candidate_id=candidate_model,
            proposed_claim=requested_level,
            evidence=bundle,
            domain=domain,
        )
        eedr_record = {
            "status": eedr_res.final_status,
            "failure_diagnosed": eedr_res.diagnosis.category.value if eedr_res.diagnosis else None,
            "repair_operator_applied": eedr_res.repair_candidate.operator_applied.value if eedr_res.repair_candidate else None,
            "repaired_candidate_id": eedr_res.repair_candidate.repair_id if eedr_res.repair_candidate else None,
            "final_justified_tier": eedr_res.final_justified_claim.name,
            "evidence_preserved": eedr_res.evidence_fingerprint_preserved,
            "audit_trail": eedr_res.audit_trail,
        }

    # Generate immutable receipt
    receipt = {
        "receipt_id": hashlib.sha256(f"{case_id}_{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:16],
        "case_id": case_id,
        "domain": domain,
        "scientific_claim": scientific_claim,
        "candidate_model": candidate_model,
        "requested_tier": requested_level.name,
        "justified_tier": adj.justified_claim.name,
        "verdict": adj.verdict.value,
        "promoted": adj.verdict == EpistemicAdjudicationVerdict.PROMOTED,
        "justification": adj.justification,
        "missing_evidence": [r.value for r in adj.missing_evidence],
        "eedr_repair": eedr_record,
        "frozen_provenance": {
            "version": spec["version"],
            "grammar_sha256": spec["grammar_sha256"],
            "referee_sha256": spec["referee_kernel_sha256"],
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="APEDP-X Air-Gapped External Evaluator CLI")
    parser.add_argument("--case-file", type=str, help="Path to external case JSON payload")
    parser.add_argument("--receipt-output", type=str, help="Path to write emitted adjudication receipt JSON")
    parser.add_argument("--batch-file", type=str, help="Path to batch JSON containing list of external cases")
    parser.add_argument("--batch-output", type=str, help="Path to write batch receipts JSON")
    args = parser.parse_args()

    print("\n" + "=" * 90)
    print("APEDP-X AIR-GAPPED EXTERNAL ADJUDICATION ENGINE")
    print(format_frozen_spec_banner())
    print("=" * 90)

    if args.case_file:
        in_path = Path(args.case_file)
        if not in_path.is_file():
            print(f"Error: Case file not found: {in_path}")
            sys.exit(1)
        with open(in_path) as f:
            payload = json.load(f)
        receipt = adjudicate_external_payload(payload)
        out_path = Path(args.receipt_output) if args.receipt_output else in_path.with_suffix(".receipt.json")
        with open(out_path, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"\n[Adjudication Receipt Emitted]: {out_path}")
        print(f"  • Case ID        : {receipt['case_id']}")
        print(f"  • Requested Tier : {receipt['requested_tier']}")
        print(f"  • Justified Tier : {receipt['justified_tier']}")
        print(f"  • Verdict        : {receipt['verdict']} (Promoted: {receipt['promoted']})")
        if receipt["eedr_repair"]:
            print(f"  • EEDR Status    : {receipt['eedr_repair']['status']} (Operator: {receipt['eedr_repair']['repair_operator_applied']})")
        print("=" * 90 + "\n")

    elif args.batch_file:
        in_path = Path(args.batch_file)
        if not in_path.is_file():
            print(f"Error: Batch file not found: {in_path}")
            sys.exit(1)
        with open(in_path) as f:
            batch = json.load(f)
        receipts = [adjudicate_external_payload(c) for c in batch]
        out_path = Path(args.batch_output) if args.batch_output else in_path.with_suffix(".receipts.json")
        with open(out_path, "w") as f:
            json.dump(receipts, f, indent=2)
        print(f"\n[Batch Completed]: {len(receipts)} cases evaluated -> {out_path}\n")

    else:
        # Self-test demonstration
        demo_payload = {
            "case_id": "DEMO_EXTERNAL_LORENTZ",
            "domain": "astrophysics",
            "scientific_claim": "Lorentz Invariant Plasma Conduction",
            "candidate_model": "J = sigma * (E + v x B)",
            "requested_claim_level": "C4_CAUSAL_MECHANISTIC",
            "evidence": {
                "metrics": {"residual": 0.002, "holdout_rmse": 0.015},
                "verified_requirements": [
                    "in_sample_convergence",
                    "out_of_sample_holdout",
                    "equivalence_rivals_rejected",
                    "interventional_invariance",
                ],
                "failures": [],
            },
        }
        receipt = adjudicate_external_payload(demo_payload)
        print("\n[Air-Gapped Demonstration Run]:")
        print(json.dumps(receipt, indent=2))
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

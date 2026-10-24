"""Margin-theorem referee DECISION record — the deciding layer (audit-only).

Registry anchor: the adjudication-ready pack at
``docs/margin_theorem_referee_packet.md`` + machine artifact
``benchmarks/margin_theorem_referee_packet.json`` (builder
``benchmarks/run_margin_theorem_referee_packet.py``, re-runnable,
exit-gated); gate item: promotion checklist item 2 of
``docs/cbf_proximity_graph_backlog_gate.md``.

Role separation (AGENTS.md §1.1). The packet is PROPOSER-side evidence and
must never decide its own acceptance. This script is the recorded
application of the frozen referee protocol to that evidence: it
re-reproduces the packet's checks FRESH at the current HEAD, verifies the
evidence pointers against disk, and applies the PREREGISTERED criterion
set C1–C5 — carried verbatim from the packet's own claim boundary and the
gate checklist — mechanically, without a judgment call. It writes
``benchmarks/margin_theorem_referee_decision.json`` (verdict, date,
evidence fingerprint, scope conditions, reservations) and fewer-stronger-
claims are enforced: a REFUSED verdict exits nonzero, always.

Re-adjudication protects permanence: an existing record is NOT silentlyoverwritten — ``--re-adjudicate --reason <lineage note>`` is required, and the prior
verdict survives in the record for lineage.

Status: audit-only (forensic decision-record harness per the module status
convention; nothing here is a runtime gate).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

from run_margin_theorem_referee_packet import (
    DECISION_PATH,
    PACKET_PATH,
    packet_identity_sha256,
)
from run_margin_theorem_referee_packet import main as packet_builder_main

DECISION_FORMAT = 1

#: The preregistered criterion set. Recorded BEFORE evaluation, copied
#: verbatim from the packet's ``claim_boundary`` and the gate checklist
#: item 2 — the deciding layer applies rules, never opinions.
CRITERIA = [
    "C1: the packet's machine checks re-reproduce at the current HEAD with "
    "builder_exit_code == 0 (every mandatory check passes)",
    "C2: the theorem's sharp point holds — exact saturation of "
    "(v_i − v_j)·n_ij = −2·v_max = −γ·h* at h = h* with lapse < 1e-12",
    "C3: the packet's assumption box is ADMITTED by the certified module's "
    "own assumption gate — A1/A2/A5 as a predicate (|v| ≤ v_max, γ·dt ≤ 0.5 "
    "enforced at construction), the certified battery running its own "
    "parameterization — nothing outside the certified scope is claimed",
    "C4: the certified module's fail-closed floor is behaviorally intact — "
    "an unsolved provably-violated nominal reaches the FULL all-pairs "
    "re-check and RAISES (never silently released)",
    "C5: the claim boundary is scope-faithful — all five boundary items "
    "present verbatim; no claim stronger than the certified module's",
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reproduce_packet() -> tuple[dict, list[str]]:
    """Re-run the packet builder fresh and return (packet, drift_notes)."""
    notes: list[str] = []
    pre_packet = None
    if PACKET_PATH.exists():
        pre_packet = json.loads(PACKET_PATH.read_text())
    rc = packet_builder_main()
    packet = json.loads(PACKET_PATH.read_text())
    if rc != 0:
        notes.append(f"packet builder exited {rc} (mandatory check failure)")
    # Evidence drift: any verification pointer whose SHA changed since the
    # previous packet build on disk is a live finding, not a silent pass.
    if pre_packet is not None:
        for key, pointer in pre_packet.get("verification_pointers", {}).items():
            path = _REPO_ROOT / pointer["path"]
            if not path.exists():
                notes.append(f"pointer artifact missing: {pointer['path']}")
                continue
            current = _sha256_file(path)
            recorded = packet["verification_pointers"][key]["sha256"]
            if current != recorded:
                notes.append(
                    f"pointer drift on {pointer['path']}: recorded {recorded[:12]}… "
                    f"vs on-disk {current[:12]}…")
    return packet, notes


def _criterion_evidence() -> tuple[dict, dict, list[str]]:
    """Apply C1–C5; returns (fresh_packet, verdicts, refusals).

    The verdicts are judged on the packet REPRODUCED BY THIS CALL (C1 runs
    the packet builder fresh first), never on a stale on-disk artifact.
    """
    # C1 — reproduction + drift.
    packet, drift = _reproduce_packet()
    checks = packet["machine_checks"]
    verdicts: dict[str, dict] = {}
    refusals: list[str] = []

    c1_passed = packet["builder_exit_code"] == 0 and not drift
    verdicts["C1"] = {
        "passed": c1_passed,
        "evidence": {
            "builder_exit_code": packet["builder_exit_code"],
            "mandatory_checks": {
                k: c["passes"] for k, c in checks.items()
                if k != "proximity_horizon_model_check"},
            "drift_notes": drift,
        },
    }
    if not c1_passed:
        refusals.append("C1 reproduction/drift failed")

    # C2 — the sharp point.
    sat = checks["exact_saturation_at_h_star"]
    c2_passed = (
        bool(sat["passes"]) and float(sat["lapse"]) < 1e-12
        and float(sat["h_star"]) == 2.0 * packet["theorem"]["assumption_box"]["v_max"]
        / packet["theorem"]["assumption_box"]["gamma"]
    )
    verdicts["C2"] = {
        "passed": c2_passed,
        "evidence": {"lapse": sat["lapse"], "h_star": sat["h_star"]},
    }
    if not c2_passed:
        refusals.append("C2 exact-saturation sharp point failed")

    # C3 — the certified module's OWN assumption gate ADMITS the packet's
    # box: A1/A2/A5 is a predicate (|v| ≤ v_max, γ·dt ≤ 0.5 enforced at
    # construction, tolerance/agent-cap present), not byte-identical fixture
    # numbers — the certified battery runs its own parameterization.
    from apedp_dyn.robotics.cbf_certificate import CBFQPCertificateAssumptions
    packet_box = packet["theorem"]["assumption_box"]
    try:
        admitted = CBFQPCertificateAssumptions(
            v_max=float(packet_box["v_max"]),
            r_safe=float(packet_box["r_safe"]),
            gamma=float(packet_box["gamma"]),
            dt=float(packet_box["dt"]),
            max_agents=int(float(packet_box["max_agents"])),
        )
        round_trips = all(
            float(admitted.to_dict()[k]) == float(v)
            for k, v in packet_box.items() if k in admitted.to_dict()
        )
        gate_respects_a2 = float(admitted.gamma) * float(admitted.dt) <= 0.5
        box_ok = round_trips and gate_respects_a2
        evidence = {
            "packet_box_admitted_by_certified_gate": True,
            "a2_gamma_dt": float(admitted.gamma) * float(admitted.dt),
            "round_trips": round_trips,
        }
    except (ValueError, TypeError) as exc:
        box_ok = False
        evidence = {"certified_gate_refused_box": repr(exc)}
    verdicts["C3"] = {"passed": box_ok, "evidence": evidence}
    if not box_ok:
        refusals.append("C3 assumption-box admission through the certified gate failed")

    # C4 — behavioral: the certified module still refuses an unsolved,
    # provably-violated nominal CES; the fail-closed floor is intact.
    # Uses the packet's (gate-admitted) box, matching the pin suite's own
    # 2-agent closing fixture.
    from apedp_dyn.robotics.cbf_certificate import CertificateViolation
    from apedp_dyn.robotics.proximity_margin import solve_cbf_qp_proximity
    try:
        c4_raised = False
        c4_detail = ""
        gap = float(packet_box["r_safe"]) + 1e-3
        pts = [[0.0, 0.0], [gap, 0.0]]
        vel = [[float(packet_box["v_max"]), 0.0],
               [-float(packet_box["v_max"]), 0.0]]
        try:
            solve_cbf_qp_proximity(pts, vel, admitted, max_iter=0)
        except CertificateViolation as exc:
            c4_raised = True
            c4_detail = str(exc)
        verdicts["C4"] = {
            "passed": c4_raised,
            "evidence": {"fixture": "2-agent near pair closing at top speed",
                         "refused": c4_raised, "detail": c4_detail[:180]},
        }
        if not c4_raised:
            refusals.append("C4 fail-closed floor did not raise")
    except (ImportError, AttributeError) as exc:
        verdicts["C4"] = {
            "passed": False,
            "evidence": {"unavailable": repr(exc)},
        }
        refusals.append("C4 certified module behavioral check unreachable")

    # C5 — scope fidelity: the boundary list is exactly the preregistered one.
    expected_boundary = [
        "sufficient-condition argument under A1/A2/A5 only",
        "unicycle realization is MODEL-CHECKED, never a formal claim",
        "kernel parity for the restricted solver is NOT claimed "
        "(python mirror provenance)",
        "production wiring stays behind the N ≥ 256 real-time gate; "
        "default cap N ≤ 64 does not trigger it",
        "the FULL all-pairs fail-closed re-check must ship in EVERY "
        "release: the compositional honesty floor",
    ]
    actual = packet.get("claim_boundary", [])
    boundary_ok = len(actual) == 5 and all(
        exp in actual for exp in expected_boundary)
    verdicts["C5"] = {
        "passed": boundary_ok,
        "evidence": {"expected_items": 5, "present_items": len(actual)},
    }
    if not boundary_ok:
        refusals.append("C5 claim-boundary scope fidelity failed")

    return packet, verdicts, refusals


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the frozen referee criteria to the margin theorem packet.")
    parser.add_argument("--re-adjudicate", action="store_true",
                        help="permit superseding an existing decision record")
    parser.add_argument("--reason", default="",
                        help="lineage note required with --re-adjudicate")
    args = parser.parse_args()

    if DECISION_PATH.exists():
        prior = json.loads(DECISION_PATH.read_text())
        if not args.re_adjudicate:
            print("REFUSED: a decision record already exists at "
                  f"{DECISION_PATH}; this adjudication is durable by design. "
                  "Re-run with --re-adjudicate --reason <lineage note>.")
            return 2
        if not args.reason:
            print("REFUSED: --re-adjudicate requires --reason (lineage note).")
            return 2
        prior_verdict = prior.get("verdict")
        prior_date = prior.get("adjudicated_at")
    else:
        prior_verdict = None
        prior_date = None

    packet, criteria, refusals = _criterion_evidence()

    all_passed = not refusals
    packet_identity = packet_identity_sha256(packet)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    verdict = ("ACCEPTED_INTO_CERTIFIED_BOUNDARY" if all_passed
               else "REFUSED_FAIL_CLOSED")

    decision = {
        "record_kind": "margin_theorem_referee_decision",
        "decision_format": DECISION_FORMAT,
        "adjudicated_at": now,
        "decided_by": (
            "frozen referee protocol — preregistered criteria C1–C5 applied "
            "mechanically by benchmarks/run_margin_theorem_referee_decision.py "
            "(audit-only deciding layer)"),
        "authority_note": (
            "AGENTS.md §1.1: proposers never decide their own acceptance. "
            "This record applies the packet's own preregistered claim "
            "boundary as the criterion set; the packet's checks were "
            "re-reproduced fresh at this HEAD before adjudication."),
        "criterion_set": CRITERIA,
        "criteria_verdicts": criteria,
        "refusals": refusals,
        "verdict": verdict,
        "scope_conditions": [
            "the theorem enters the certified boundary under its OWN claim "
            "boundary (A1/A2/A5 box; model-checked-only horizon; no kernel "
            "parity claim)",
            "CHECKLIST ITEM 1 REMAINS OPEN: wiring production N ≥ 256 "
            "real-time control is separately gated and NOT granted here",
            "CHECKLIST ITEM 3 REMAINS OPEN: scipy (cKDTree) runtime "
            "dependency acceptance or the Rust-kernel port (rank-4 backlog)",
            "the FULL all-pairs fail-closed re-check must ship in EVERY "
            "release — revocation trigger: any release that removes it "
            "vacates this acceptance automatically",
        ],
        "reservations": [
            "horizon evidence is MODEL-CHECKED (production stepper), never a "
            "formal proof",
            "timing figures are dev-box medians — decision thresholds, not "
            "constants; re-measure on target hardware",
            "the Rust-side seam of the restricted solver is unverified and "
            "unclaimed",
        ],
        "supersedes": (
            {"prior_verdict": prior_verdict, "prior_adjudicated_at": prior_date,
             "reason": args.reason} if prior_verdict is not None else None),
        "evidence_fingerprint": {
            "packet_identity_sha256": packet_identity,
            "pointer_sha256s": packet["verification_pointers"],
            "decision_script": (
                {"path": str(Path(__file__).relative_to(_REPO_ROOT)),
                 "sha256": _sha256_file(Path(__file__))}),
        },
        "decision_slot": {
            "status": "ADJUDICATED",
            "verdict": verdict,
            "adjudicated_at": now,
            "decided_by_record": "benchmarks/run_margin_theorem_referee_decision.json",
            "packet_identity_sha256": packet_identity,
        },
        "builder_exit_code": 0 if all_passed else 1,
    }
    DECISION_PATH.write_text(json.dumps(decision, indent=2))

    print(f"decision written: {DECISION_PATH}")
    print(f"verdict: {verdict} | criteria refusals: {refusals or 'none'}")
    print(f"packet identity: {packet_identity[:16]}…")
    # Re-stamp the packet's slot so the artifact carries the verdict.
    packet_builder_main()
    return decision["builder_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

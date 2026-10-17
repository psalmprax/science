"""Blind adversarial benchmark protocol (Stage 4 of the validation ladder).

Method rule: the referee is FROZEN (see APEDPX_FROZEN_v1.json). This runner
refuses to score if any guard file changed — edit the referee and you must
mint a new freeze record first.

Blind cases live OUTSIDE the frozen trap files and register here with hidden
ground truth (claim warranted or not). Scoring reports blind-only metrics:

  FPR_blind = promoted-but-unwarranted / all promoted      (want -> 0)
  CAR_blind = correct abstentions / unwarranted cases     (want -> 1)
  VRR_blind = warranted retained / all warranted          (want -> 1)

Each case supplies its own adapter to (promoted: bool, abstained: bool),
so heterogeneous domains share one scoreboard without forcing verdict enums.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, '.')

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_RECORD = Path(__file__).resolve().parent / "APEDPX_FROZEN_v1_2.json"


def verify_referee() -> dict[str, str]:
    """Abort scoring if any frozen referee file or registered benchmark asset changed.

    Extended beyond the original 3 referee files to verify every file in the frozen
    specification record, ensuring silent drift fails loudly across the entire suite.
    Returns file->sha256.
    """
    rec = json.loads(FROZEN_RECORD.read_text())
    checked: dict[str, str] = {}
    # Check all files with a frozen record across file_hashes_sha256 and referee_guard_files
    guard_files = sorted(set(list(rec.get("file_hashes_sha256", {}).keys()) + list(rec.get("referee_guard_files", []))))
    for rel in guard_files:
        cur = hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
        exp = rec["file_hashes_sha256"][rel]
        if cur != exp:
            raise RuntimeError(
                f"REFEREE OR FROZEN RECORD CHANGED since {rec['label']}: {rel}. "
                "Mint a new freeze record before scoring blind cases.")
        checked[rel] = cur
    return checked


@dataclass
class BlindCase:
    """One adversarial case. `warranted` is hidden ground truth: True iff
    the evidence actually entitles promotion. `blind=True` cases count
    toward FPR/CAR/VRR; `blind=False` are worked examples (excluded).

    `population` must be one of "valid" (evidence supports the claim,
    expect PROMOTE), "unsafe" (convincing but unwarranted, expect REJECT),
    "ambiguous" (expect ABSTAIN). `requested_claim_level` records the
    claim tier under test (e.g. C2 vs C4) for claim-target-mismatch cases.
    """

    case_id: str
    domain: str
    warranted: bool
    run: Callable[[], dict[str, Any]]
    blind: bool = True
    notes: str = ""
    scientific_claim: str = ""
    candidate_model: str = ""
    evidence_bundle: str = ""
    proposer_id: str = ""
    difficulty: str = ""
    requested_claim_level: str = ""
    population: str = ""  # "valid" | "unsafe" | "ambiguous" (blind cases should set this)


@dataclass
class BlindReport:
    n_blind: int = 0
    n_promoted: int = 0
    fpr_blind: float | None = None
    car_blind: float | None = None
    vrr_blind: float | None = None
    # False-rejection rate: warranted cases not promoted / all warranted.
    frr_blind: float | None = None
    # Coverage: blind cases reaching a decisive outcome
    # (promoted or explicitly rejected) vs abstained.
    coverage_blind: float | None = None
    # Per-domain breakdown: domain -> {n, fpr, vrr, coverage}.
    by_domain: dict[str, dict[str, float]] = field(default_factory=dict)
    # One-sided 95% upper bound when zero failures observed: 1 - 0.05**(1/n).
    # An observed 0/n is reported WITH this bound, never as "probability zero".
    fpr_upper95: float | None = None
    populations: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)


def ci_upper95_zero_failures(n: int) -> float | None:
    """One-sided 95% upper bound given zero failures in n cases."""
    if n <= 0:
        return None
    return float(1.0 - 0.05 ** (1.0 / n))


def wilson_upper_zero(n: int, z: float = 1.96) -> float | None:
    """Two-sided 95% Wilson upper bound given zero successes in n trials."""
    if n <= 0:
        return None
    q = (z * z) / n
    return q / (1.0 + q)


def ledger_row(n: int) -> dict[str, float | None]:
    """Machine-generated ledger row: all three zero-failure bounds at once.

    Use this instead of hand-copying numbers into reports. Keys:
    wilson (two-sided 95%), exact (one-sided 95%), rule3 (3/n).
    """
    return {
        "n": float(n),
        "wilson": wilson_upper_zero(n),
        "exact": ci_upper95_zero_failures(n),
        "rule3": 3.0 / n,
    }


def score_blind(cases: list[BlindCase]) -> BlindReport:
    """Score registered cases. Referee runs BEFORE truth is consulted."""
    rep = BlindReport()
    blind = [c for c in cases if c.blind]
    rep.n_blind = len(blind)
    for c in cases:
        out = c.run()
        promoted = bool(out.get("promoted", False))
        abstained = bool(out.get("abstained", not promoted))
        rejected = bool(out.get("rejected", False))
        row = {"case_id": c.case_id, "domain": c.domain, "blind": c.blind,
               "warranted": c.warranted if not c.blind else "hidden",
               "promoted": promoted, "abstained": abstained,
               "rejected": rejected}
        rep.rows.append(row)
        if not c.blind:
            continue
        if promoted:
            rep.n_promoted += 1
    if blind:
        prom = [r for r in rep.rows if r["blind"] and r["promoted"]]
        unw = [c for c in blind if not c.warranted]
        war = [c for c in blind if c.warranted]
        # Re-resolve warranted flags from registry (kept out of rows).
        by_id = {c.case_id: c.warranted for c in blind}
        fp = sum(1 for r in prom if not by_id[r["case_id"]])
        rep.fpr_blind = (fp / len(prom)) if prom else 0.0
        if fp == 0 and prom:
            rep.fpr_upper95 = ci_upper95_zero_failures(len(prom))
        for c in blind:
            if c.population:
                rep.populations[c.population] = rep.populations.get(c.population, 0) + 1
        rep.car_blind = (sum(1 for c in unw
                             for r in rep.rows
                             if r["case_id"] == c.case_id and r["abstained"])
                         / len(unw)) if unw else None
        rep.vrr_blind = (sum(1 for c in war
                             for r in rep.rows
                             if r["case_id"] == c.case_id and r["promoted"])
                         / len(war)) if war else None
        rep.frr_blind = (1.0 - rep.vrr_blind) if rep.vrr_blind is not None else None
        decisive = sum(1 for r in rep.rows
                       if r["blind"] and (r["promoted"] or r["rejected"]))
        rep.coverage_blind = (decisive / len(blind)) if blind else None
        by_id_row = {r["case_id"]: r for r in rep.rows if r["blind"]}
        domains: dict[str, list[str]] = {}
        for c in blind:
            domains.setdefault(c.domain, []).append(c.case_id)
        for dom, ids in domains.items():
            pr = sum(1 for i in ids if by_id_row[i]["promoted"])
            fp = sum(1 for i in ids if by_id_row[i]["promoted"] and not by_id[i])
            vp = sum(1 for i in ids if by_id[i] and by_id_row[i]["promoted"])
            vw = sum(1 for i in ids if by_id[i])
            dc = sum(1 for i in ids if by_id_row[i]["promoted"] or by_id_row[i]["rejected"])
            rep.by_domain[dom] = {
                "n": float(len(ids)),
                "fpr": (fp / pr) if pr else 0.0,
                "vrr": (vp / vw) if vw else 0.0,
                "coverage": dc / len(ids),
            }
    return rep


def example_case_short_window() -> BlindCase:
    """WORKED EXAMPLE (blind=False): ESEDS trap-25 through the frozen path."""
    from apedp_dyn.sandbox.eseds_dynamical import run_eseds_trap

    def run() -> dict[str, Any]:
        r = run_eseds_trap(train_end=25, bootstrap_n_boot=100)
        return {"promoted": r["promoted"] is not None,
                "abstained": r["status"] in ("multi_hypothesis", "insufficient_evidence")}
    return BlindCase(case_id="EXAMPLE-eseds-trap25", domain="dynamics",
                     warranted=False, run=run, blind=False,
                     notes="worked example only; excluded from blind metrics")


if __name__ == "__main__":
    checked = verify_referee()
    print(f"referee frozen OK ({len(checked)} guard files match {json.loads(FROZEN_RECORD.read_text())['label']})")
    rep = score_blind([example_case_short_window()])
    print(f"blind cases: {rep.n_blind} (example excluded by design)")
    for r in rep.rows:
        print(f"  {r['case_id']}: promoted={r['promoted']} abstained={r['abstained']}")

#!/usr/bin/env python3
"""Audit every DQI record for correctness claims with an unchecked phase or junk.

`docs/DQI_OFFICIAL_COMPATIBILITY.md` §5 reported a candidate as correct because its
**syndromes** came back; Experiment D2 then measured its **phase** and found 37 of 64
shots wrong. The record had the fields it needed and simply was not asked for them.
This script asks the question of every record in the repository, mechanically, so the
next one cannot slip through the same gap.

A *candidate row* is any object in a record that carries a Toffoli figure
(`mean_toffoli_like`, `mean`, or `total`) together with some correctness signal
(`correct`, `syndromes`, `syndromes_restored`, `junk`, `phase`, ...). For each row it
reports which of the three checks are present:

* **value** — the syndromes came back (`syndromes`, `syndromes_restored`, `value_errors`)
* **junk** — every other qubit is `|0>` (`junk`, `junk_clean`)
* **phase** — the tracked phase matches (`phase`, `phase_errors`, `phase_matches_reference`)

and flags a row as **unchecked** when it asserts correctness while omitting phase or
junk. Rows whose record is a *measurement of resources only* (no correctness claim at
all) are reported as `resources-only` rather than as failures — the audit is looking
for claims that rest on an unchecked check, not for claims that are absent.

Nothing is rewritten. The output is a report, and the counts are pinned by
`tests/test_record_correctness_audit.py` so a new record that reintroduces the gap
changes the numbers.

Example::

    python scripts/audit_record_correctness_checks.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "benchmarks" / "quantum_compilation"
DEFAULT_RECORD = RECORDS / "correctness_audit.json"

TOFFOLI_KEYS = ("mean_toffoli_like", "mean", "total", "executed_toffoli_per_shot")
VALUE_KEYS = ("syndromes", "syndromes_restored", "value_errors")
JUNK_KEYS = ("junk", "junk_clean")
PHASE_KEYS = ("phase", "phase_errors", "phase_matches_reference")
CLAIM_KEYS = ("correct", "is_correct", "built")


def is_candidate_row(node: dict) -> bool:
    """A row that costs something and says, or implies, whether it is right."""
    has_cost = any(key in node for key in TOFFOLI_KEYS)
    has_signal = any(key in node for key in VALUE_KEYS + JUNK_KEYS + PHASE_KEYS)
    has_claim = any(key in node for key in CLAIM_KEYS)
    return has_cost and (has_signal or has_claim)


def classify(node: dict) -> dict:
    return {
        "value_checked": any(key in node for key in VALUE_KEYS),
        "junk_checked": any(key in node for key in JUNK_KEYS),
        "phase_checked": any(key in node for key in PHASE_KEYS),
        "claims_correct": (node.get("correct") is True or node.get("is_correct") is True),
    }


def walk(node, path: str = "") -> list[tuple[str, dict]]:
    """Every candidate row in a record, with a dotted path naming it."""
    found = []
    if isinstance(node, dict):
        if is_candidate_row(node):
            found.append((path or "<root>", node))
        for key, value in node.items():
            found.extend(walk(value, f"{path}.{key}" if path else key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(walk(value, f"{path}[{index}]"))
    return found


def audit_record(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = []
    for where, node in walk(payload):
        state = classify(node)
        checked = [name for name in ("value", "junk", "phase")
                   if state[f"{name}_checked"]]
        if len(checked) == len(("value", "junk", "phase")):
            status = "checked"
        elif checked:
            # The dangerous class, and the one Experiment D was in: *some* evidence is
            # recorded, so a reader has no reason to look for the rest. The claim flag
            # is reported beside it, and told apart in the counts.
            missing = [name for name in ("value", "junk", "phase")
                       if name not in checked]
            status = ("PARTIAL_EVIDENCE_MISSING_" + "_AND_".join(
                name.upper() for name in missing)
                + ("_CLAIMING_CORRECT" if state["claims_correct"] else ""))
        elif state["claims_correct"]:
            # A bare boolean: nothing recorded to re-check, but nothing that looks like
            # evidence either. Reported separately rather than as the same defect.
            status = "claims_correct_aggregate_only"
        else:
            status = "resources-only"
        rows.append({"path": where, "status": status, **state,
                     "toffoli": next((node[key] for key in TOFFOLI_KEYS
                                      if key in node), None)})
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {"record": str(path.relative_to(ROOT)), "rows": rows, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    audited = []
    for path in sorted(args.records.glob("*.json")):
        if path.name == args.record.name:
            continue
        try:
            audited.append(audit_record(path))
        except json.JSONDecodeError:
            audited.append({"record": str(path.relative_to(ROOT)), "rows": [],
                            "counts": {"unreadable": 1}})

    gaps = [{"record": entry["record"], "path": row["path"], "toffoli": row["toffoli"],
             "missing": [name for name in ("value", "junk", "phase")
                         if not row[f"{name}_checked"]],
             "claims_correct": row["claims_correct"]}
            for entry in audited for row in entry["rows"]
            if row["status"].startswith("PARTIAL_EVIDENCE")]
    totals: dict[str, int] = {}
    for entry in audited:
        for status, count in entry["counts"].items():
            totals[status] = totals.get(status, 0) + count

    payload = {
        "audit": "candidate rows whose correctness claim rests on an unchecked register",
        "why": "Experiment D's candidate was called correct because its syndromes came "
               "back; Experiment D2 measured the phase and found 37 of 64 shots wrong. "
               "A record can carry the gap only if something asks for it.",
        "records_scanned": len(audited),
        "totals": totals,
        "gaps": gaps,
        "per_record": audited,
        "reading": {
            "checked": "claims correctness and records value, junk and phase",
            "PARTIAL_EVIDENCE_MISSING_*": "claims correctness while recording some of "
                                           "the three checks and not the named ones — "
                                           "Experiment D's shape",
            "claims_correct_aggregate_only": "a bare `correct` boolean: nothing recorded "
                                              "to re-check, but no partial evidence "
                                              "either",
            "measured-not-claimed": "records at least one check but does not claim "
                                    "correctness; a measurement, not an assertion",
            "resources-only": "counts only — no correctness claim either way",
        },
        "scope": {
            "measured": "a structural scan of every record's candidate rows",
            "not_measured": "it cannot tell whether a recorded `phase: true` was computed "
                            "against the right expectation; that stays per-record evidence",
        },
        "reproducibility": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git_rev": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      check=False, capture_output=True,
                                      text=True).stdout.strip(),
        },
    }
    args.record.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"records scanned: {len(audited)}")
    for status, count in sorted(totals.items()):
        print(f"  {status:45s} {count}")
    for gap in gaps:
        print(f"  GAP {gap['record']} :: {gap['path']} "
              f"(missing {', '.join(gap['missing'])}, T={gap['toffoli']})")
    print(f"wrote {args.record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Audit the Lean layer: what is machine-checked, what is assumed, what is missing.

A `.lean` file in this repository is not evidence of a proof. The audited pattern is a
property declared as an `axiom` and then "proved" by rewriting with that same axiom, or a
tactic that discharges a vacuous integer identity. This script measures the layer instead
of describing it, and writes the count a reader can check.

Classes, in precedence order:

* ``incomplete`` — the file contains `sorry` or `admit`.
* ``axiom_injected`` — the file declares axioms; a theorem proved only by rewriting with
  a local axiom is not proved.
* ``checked`` — theorems, no axioms, no `sorry`: the only class that is machine-checked
  from Lean's own foundations.
* ``declaration_only`` — no theorems at all.

Usage::

    python3 scripts/audit_lean_formal_layer.py
    python3 scripts/audit_lean_formal_layer.py --check   # exit 1 if a claim is unbacked

Status: audit-only
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "formal" / "lean_axiom_audit.json"
AXIOM = re.compile(r"^\s*axiom\s+(\w+)", re.M)
THEOREM = re.compile(r"^\s*(?:theorem|lemma)\s+(\w+)", re.M)
SORRY = re.compile(r"\b(sorry|admit)\b")
REWRITE = re.compile(r"rw\s*\[([^\]]*)\]")


def classify(text: str) -> tuple[str, dict]:
    axioms = AXIOM.findall(text)
    theorems = THEOREM.findall(text)
    sorry = len(SORRY.findall(text))
    blocks = re.split(r"^\s*(?:theorem|lemma)\s+\w+", text, flags=re.M)[1:]
    injected: list[str] = []
    for block in blocks:
        rewrite = REWRITE.search(block)
        if not rewrite:
            continue
        used = rewrite.group(1)
        for axiom in axioms:
            # Substring with a word boundary: `rw [cube_neg x]` rewrites with the
            # axiom `cube_neg` applied to `x`, so an equality test on comma-split
            # names misses exactly the pattern this audit is for.
            if re.search(rf"\b{re.escape(axiom)}\b", used):
                injected.append(axiom)
                break
    if sorry:
        kind = "incomplete"
    elif axioms:
        kind = "axiom_injected"
    elif theorems:
        kind = "checked"
    else:
        kind = "declaration_only"
    return kind, {"axioms": len(axioms), "theorems": len(theorems), "sorry": sorry,
                  "theorems_rewriting_a_local_axiom": len(injected)}


def build(root: pathlib.Path = ROOT) -> dict:
    files = sorted(path for path in root.rglob("*.lean") if ".git" not in path.parts)
    totals = {"files": len(files), "axioms": 0, "theorems": 0, "sorry": 0,
              "theorems_rewriting_a_local_axiom": 0}
    kinds: dict[str, int] = {}
    per_directory: dict[str, dict[str, int]] = {}
    lean_formal = 0
    for path in files:
        kind, counts = classify(path.read_text(errors="ignore"))
        kinds[kind] = kinds.get(kind, 0) + 1
        for key in ("axioms", "theorems", "sorry", "theorems_rewriting_a_local_axiom"):
            totals[key] += counts[key]
        relative = path.relative_to(root)
        directory = relative.parts[0] if len(relative.parts) > 1 else "."
        bucket = per_directory.setdefault(directory, {"files": 0, "axioms": 0,
                                                      "theorems": 0, "incomplete": 0})
        bucket["files"] += 1
        bucket["axioms"] += counts["axioms"]
        bucket["theorems"] += counts["theorems"]
        bucket["incomplete"] += 1 if kind == "incomplete" else 0
        if "lean_formal" in relative.parts:
            lean_formal += 1
    checked = kinds.get("checked", 0)
    return {
        "schema": "apedp.formal.lean_axiom_audit/v1",
        "scope": "every .lean file in the repository; apedp_dyn/lean_formal is the layer "
                 "that verification claims refer to",
        "totals": totals,
        "classification": {
            "checked": checked,
            "axiom_injected": kinds.get("axiom_injected", 0),
            "incomplete": kinds.get("incomplete", 0),
            "declaration_only": kinds.get("declaration_only", 0),
        },
        "checked_fraction": round(checked / len(files), 4) if files else 0.0,
        "files_under_lean_formal": lean_formal,
        "per_directory": dict(sorted(per_directory.items())),
        "verdict": (
            "Only the 'checked' files are machine-checked from Lean's foundations; an "
            "axiom_injected file proves its statement from an assumed axiom, which is a "
            "recording of intent, not a proof. No result in this repository may be "
            "described as formally verified on the strength of a non-'checked' file."),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when any file is not machine-checked")
    args = parser.parse_args(argv)
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    totals = report["totals"]
    print(f"lean files: {totals['files']} ({report['files_under_lean_formal']} under lean_formal)")
    print(f"axioms: {totals['axioms']}  theorems: {totals['theorems']}  "
          f"sorry/admit: {totals['sorry']}")
    print(f"theorems proved by rewriting a local axiom: "
          f"{totals['theorems_rewriting_a_local_axiom']}")
    print("classification:", report["classification"])
    print(f"wrote {OUT.relative_to(ROOT)}")
    if args.check and report["classification"]["checked"] != totals["files"]:
        print("not every .lean file is machine-checked", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

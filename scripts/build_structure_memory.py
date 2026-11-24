#!/usr/bin/env python3
"""The memory of confirmed laws, built from the laws the referee has reported in recorded runs.

    python scripts/build_structure_memory.py           # write benchmarks/structure_memory.json
    python scripts/build_structure_memory.py --check   # the file is what the records give

Two records, both committed and neither touched here:

- ``preregistration/RESULT-REFINED-SEARCH-2026-11/runs.jsonl``: the laws the refined path reported
  on the physics benchmarks (Feynman and Strogatz);
- ``benchmarks/agentic_variation_dev_study.json``: the laws the default route reported on the
  twenty development laws of the invented-laws benchmark.

A reported law enters whether or not it was the right one: the route does not know which of its
reports are right, and a memory that only kept the right ones would be using the answers. What is
kept of each law is its shape (``apedp_dyn.proposers.structure_memory``): motifs with every
constant and variable name removed, and the whole law's shape. A motif met in fewer than
``MIN_LAWS`` laws is dropped: a part seen once is as likely an accident of one search's
approximation as a form of the domain.

Nothing here has seen a held-back law of the invented-laws benchmark or any law drawn after this
file was written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.proposers.structure_memory import Memory  # noqa: E402

OUT = ROOT / "benchmarks" / "structure_memory.json"
PHYSICS = ROOT / "preregistration" / "RESULT-REFINED-SEARCH-2026-11" / "runs.jsonl"
INVENTED = ROOT / "benchmarks" / "agentic_variation_dev_study.json"
MIN_LAWS = 2


def reported_laws() -> list[tuple[str, str]]:
    """Every law the two records report, with where it came from."""
    laws: list[tuple[str, str]] = []
    for line in PHYSICS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("formula"):
            laws.append(("refined path, physics benchmarks", str(row["formula"])))
    for row in json.loads(INVENTED.read_text(encoding="utf-8"))["runs"]:
        if row["arm"] == "default" and row.get("formula"):
            laws.append(("default route, development laws of the invented benchmark", str(row["formula"])))
    return laws


def build() -> dict[str, Any]:
    memory = Memory()
    sources: dict[str, int] = {}
    for source, law in reported_laws():
        memory.learn(law)
        sources[source] = sources.get(source, 0) + 1
    met = len(memory.motifs)
    memory.motifs = {shape: count for shape, count in memory.motifs.items() if count >= MIN_LAWS}
    return {"what": "Shapes of laws the referee reported in recorded runs: no constant, no variable name, no problem's answer.",
            "sources": sources, "motifs_met": met, "min_laws": MIN_LAWS, **memory.as_dict()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    if args.check:
        if OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT.relative_to(ROOT)} is not what the records give")
            return 1
        print(f"{OUT.relative_to(ROOT)}: as built")
        return 0
    OUT.write_text(text, encoding="utf-8")
    record = json.loads(text)
    print(f"wrote {OUT.relative_to(ROOT)}: {record['laws']} laws, {len(record['motifs'])} motifs kept of {record['motifs_met']}, "
          f"{len(record['skeletons'])} whole shapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

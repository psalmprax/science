"""Classify the 189 Lean comparator statements into testability tiers.

Heuristic, evidence-carrying classification of the formal STATEMENTS (not the
proofs) into tiers aligned with what APEDP-X engines can actually do:

  tier1_candidate_explicit_construction
      asserts existence of a finite/combinatorial witness with checkable
      properties (counterexamples, finite configurations) -> our system can
      verify a supplied witness or regenerate one (mode 2).
  tier2_candidate_numeric_or_bound
      exact numeric constants / inequalities that can be recomputed at higher
      precision or on a finite range (mode 1).
  tier3_uncheckable_proof_level
      quantified over infinite/abstract structures with no finite probe path.

Heuristics are labelled as candidates: any engine run is a separate,
certificate-carrying step; nothing here is a verification.

Outputs: catalog/classification.json + catalog/CLASSIFICATION.md
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAT = HERE / "catalog"
SNAP = CAT / "comparator_snapshot"
MANIFEST = CAT / "comparator_manifest.json"

PROBE_JOINS = {
    "PiExponent": {"family": 17, "probe": "cf exponent of pi (wave 1)"},
    "QuasiRiemannHypothesis": {"family": 3, "probe": "riemann_verifier (wave 1)"},
    "OstmannComplete": {"family": 13, "probe": "declared not_probeable (wave 1)"},
    "OstmannPrimes": {"family": 13, "probe": "declared not_probeable (wave 1)"},
}

FINITE_WITNESS = re.compile(
    r"(Fin \d|Fin \w+\s*→|Finset|SimpleGraph \(Fin|Matrix \(Fin|List [A-Za-zΛ]|Multiset)"
)
NUM_LITERAL = re.compile(r"(?<![\w.])\d+(?![\w.])")
NAMED_CONSTANT = re.compile(
    r"Real\.(pi|log|sqrt|exp|e\b)|riemannZeta|Catalan|EulerGamma|Irrational"
)
ASYMPTOTIC = re.compile(r"Set\.Infinite|sSup|Tendsto|∀ᶠ|Filter\.atTop|Set\.MapsTo|Eventually")


def theorem_type_text(lean_text: str, theorem_names: list[str] | None = None) -> str:
    """Extract the statement bodies (type signatures) from a challenge file."""
    chunks = []
    for m in re.finditer(r"\b(theorem|lemma)\s+[\w.'’]+\s*(.*?)(?=:=|\bsorry\b|$)", lean_text, re.S):
        chunks.append(m.group(2))
    if not chunks:
        chunks = [lean_text]
    return "\n".join(chunks)


def classify(challenge: str, lean_text: str, meta: dict) -> dict:
    names = [d.lower() for d in meta.get("declarations", [])]
    theorem_names = meta.get("theorem_names", [])
    for name in theorem_names:
        names.append(str(name).lower())
    body = theorem_type_text(lean_text)
    joined = lean_text
    has_exists = "∃" in body
    has_finite_witness = bool(FINITE_WITNESS.search(joined))
    numerics = NUM_LITERAL.findall(body)
    has_numeric = len(numerics) >= 2
    has_named_const = bool(NAMED_CONSTANT.search(joined))
    has_asymptotic = bool(ASYMPTOTIC.search(body))
    counterexample_named = any("counterexample" in n for n in names)

    evidence = []
    if counterexample_named:
        evidence.append("declaration name contains 'counterexample'")
    if has_exists:
        evidence.append("existential quantifier in statement")
    if has_finite_witness:
        evidence.append("finite witness type (Fin/Finset/SimpleGraph(Fin)/Matrix(Fin))")
    if has_numeric:
        evidence.append(f"{len(numerics)} numeric literals in statement")
    if has_named_const:
        evidence.append("named real constant (pi/log/sqrt/zeta/...)")
    if has_asymptotic:
        evidence.append("asymptotic/infinite-set quantifier (sSup/Set.Infinite/Tendsto)")

    if counterexample_named or (has_exists and has_finite_witness and has_numeric):
        tier = "tier1_candidate_explicit_construction"
    elif has_named_const and has_numeric and not has_asymptotic:
        tier = "tier2_candidate_numeric_or_bound"
    else:
        tier = "tier3_uncheckable_proof_level"

    return {
        "challenge": challenge,
        "declarations": meta.get("declarations", []),
        "solution_files": meta.get("solution_files", []),
        "tier": tier,
        "evidence": evidence,
        "probe_join": PROBE_JOINS.get(challenge),
    }


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    results = []
    for c in manifest["challenges"]:
        lean_path = SNAP / f"{c['challenge']}.lean"
        meta = dict(c)
        json_path = SNAP / f"{c['challenge']}.json"
        if json_path.is_file():
            meta["theorem_names"] = json.loads(json_path.read_text()).get("theorem_names", [])
        lean_text = lean_path.read_text(errors="replace") if lean_path.is_file() else ""
        results.append(classify(c["challenge"], lean_text, meta))

    counts: dict[str, int] = {}
    for r in results:
        counts[r["tier"]] = counts.get(r["tier"], 0) + 1

    report = {
        "rule": "candidate classification of formal STATEMENTS; nothing here is a verification",
        "challenges": len(results),
        "counts": counts,
        "results": sorted(results, key=lambda r: (r["tier"], r["challenge"])),
    }
    (CAT / "classification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Comparator-statement classification (189 statements)",
        "",
        "Candidate tiers over the formal STATEMENTS (proofs not checked; no Lean in APEDP-X).",
        f"Counts: {counts}",
        "",
    ]
    for tier in ("tier1_candidate_explicit_construction", "tier2_candidate_numeric_or_bound", "tier3_uncheckable_proof_level"):
        items = [r for r in report["results"] if r["tier"] == tier]
        lines.append(f"## {tier} ({len(items)})")
        lines.append("")
        for r in items:
            join = f" — **probe join:** family {r['probe_join']['family']} ({r['probe_join']['probe']})" if r["probe_join"] else ""
            ev = "; ".join(r["evidence"]) if r["evidence"] else "no finite/numeric features matched"
            lines.append(f"- `{r['challenge']}`: {ev}{join}")
        lines.append("")
    (CAT / "CLASSIFICATION.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(counts, indent=1))
    for tier in ("tier1_candidate_explicit_construction", "tier2_candidate_numeric_or_bound"):
        items = [r["challenge"] for r in report["results"] if r["tier"] == tier]
        print(f"\n{tier} ({len(items)}):")
        for n in items:
            print("  ", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

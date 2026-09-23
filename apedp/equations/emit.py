"""Emit canonical records + yield metrics + relationship graph edges."""
from __future__ import annotations

from apedp.equations.schema import EquationRecord


def build_records(groups: list, seed_meta: dict, classifications: list) -> list:
    records = []
    for i, (g, c) in enumerate(zip(groups, classifications)):
        base = seed_meta.get("name", f"eq_{i}").lower().replace(" ", "_").replace("'", "")
        records.append(EquationRecord(
            equation_id=f"{seed_meta.get('domain', 'gen')}.{base}.{i}",
            canonical=g["canonical"],
            aliases=[seed_meta.get("name", "")],
            representations=g["members"],
            occurrences=[seed_meta.get("page", "")],
            classification=c.get("classification", "CANDIDATE"),
            kind=c.get("kind", "CANDIDATE"),
            domain=seed_meta.get("domain", ""),
            license=seed_meta.get("license", ""),
            provenance={"reference": c.get("reference", ""),
                        "source": "wikipedia-math"},
        ))
    return records


def yield_metrics(fetched: int, parsed: int, canonical: int, quarantined: int) -> dict:
    return {"fetched": fetched, "parsed": parsed,
            "canonical": canonical, "quarantined": quarantined}


def build_graph(records: list) -> list:
    """Edges: record -> representation (alias_of)."""
    edges = []
    for r in records:
        for rep in r.representations:
            edges.append({"src": r.equation_id, "rel": "alias_of",
                          "dst": rep})
    return edges

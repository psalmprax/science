"""Canonical forms for grammar candidates (duplicate elimination)."""
from __future__ import annotations

from apedp_dyn.grammar.candidate_ir import CandidateDynamics


def canonical_form(candidate: CandidateDynamics) -> str:
    """Order-invariant structural key: family + lhs + sorted term names."""
    return f"{candidate.metadata.get('family', '?')}[{candidate.lhs}=" + \
        "+".join(sorted(candidate.term_set())) + "]"


def deduplicate(candidates: list[CandidateDynamics]) -> list[CandidateDynamics]:
    """Drop structural duplicates, keep first occurrence (stable order)."""
    seen: set[str] = set()
    out = []
    for c in candidates:
        key = canonical_form(c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

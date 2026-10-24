"""RIME feature-usage logging: which inputs appear in generated equations.

RIME is semantically blind (rename-invariance pinned elsewhere): the kernel
sees anonymous columns. This module answers the observable question — which
declared features surface in emitted equations — by mining the rendered
equation text every proposal carries in metadata.

Scope and honesty notes:
  * Syntactic presence, not causal use: `describe()` expands from the output
    node, so unreachable nodes never appear; but reachable-yet-canceling
    pairs (x + (-x)) still count as "used". A presence claim is not an
    importance claim — pair with ablations for causal statements.
  * Rejected candidates are invisible: the kernel packet only ships accepted
    proposals, so usage is measured over emissions, never over the full
    evaluated set. Zero proposals means zero log (abstention path).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


def feature_usage(proposal: Any, feature_cols: Sequence[str]) -> dict[str, bool]:
    """Per-feature presence in one proposal.

    Prefers the kernel's structural `inputs_used_names` (arena walk from the
    output: reachable nodes only) when the packet carries it; falls back to
    rendered-equation string mining otherwise. Either way this is syntactic
    presence, not causal importance (see module docstring).
    """
    structural = ((proposal.metadata or {}).get("inputs_used_names"))
    if isinstance(structural, list):
        used = set(structural)
        return {f: f in used for f in feature_cols}
    text = ((proposal.metadata or {}).get("equation") or proposal.formula or "")
    return {f: re.search(r"\b" + re.escape(f) + r"\b", text) is not None for f in feature_cols}


@dataclass
class UsageReport:
    """Aggregate usage over one search's emitted proposals."""

    n_proposals: int
    per_proposal: list[dict[str, bool]] = field(default_factory=list)
    used_in_any: list[str] = field(default_factory=list)
    unused_in_all: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"n_proposals": self.n_proposals, "per_proposal": self.per_proposal,
                "used_in_any": self.used_in_any, "unused_in_all": self.unused_in_all}


def log_search_usage(proposals: Sequence[Any], feature_cols: Sequence[str]) -> UsageReport:
    """Aggregate feature usage across emitted proposals (empty on abstention)."""
    per = [feature_usage(p, feature_cols) for p in proposals]
    used = sorted({f for u in per for f, v in u.items() if v})
    return UsageReport(n_proposals=len(per), per_proposal=per,
                       used_in_any=used,
                       unused_in_all=[f for f in feature_cols if f not in used])

"""Scientific evidence reports (P2 stage C): no automatic winner.

A Pareto frontier does not identify the correct equation; it preserves
genuine trade-offs. This module renders per-candidate evidence sections and
an explicit non-uniqueness assessment instead of collapsing everything into
one scalar score. Admissibility-gated candidates that survive are reported
with what supports them *and* what limits them.

Accountability vocabulary (review-locked): missing evidence and negative
evidence stay distinct --
``INVALID`` (contradiction demonstrated) vs ``UNEVALUATED`` (cannot yet
determine) on dimensions; ``FAIL`` (validation fails) vs
``INSUFFICIENT_EVIDENCE`` (test impossible or threshold unmet) on empirics.
"Cannot establish" must never read as "established wrong".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Canonical empirical evidence statuses. ``FAIL`` means the test ran and
#: the candidate failed it; ``INSUFFICIENT_EVIDENCE`` means the test could
#: not run or its threshold was not met -- never a silent demotion to FAIL.
EVIDENCE_STATUSES = ("PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "UNEVALUATED")


@dataclass(frozen=True)
class CandidateEvidence:
    """One candidate's evidence section inputs."""

    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    admissible: bool = True
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": dict(self.metrics),
            "admissible": self.admissible,
            "limitations": list(self.limitations),
        }


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def candidate_evidence_profile(
    name: str,
    *,
    mathematical: dict[str, Any] | None = None,
    dimensional: dict[str, Any] | None = None,
    empirical: dict[str, Any] | None = None,
    scientific: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the four-judgment Candidate Evidence Profile.

    The judgments are independent lenses, not inputs to a scalar score:

    1. **mathematical**: finite/real-valued/stable over the declared domain;
    2. **dimensional**: VALID / INVALID / UNRESOLVED (never collapsed into
       "valid by default"; UNRESOLVED blocks scientific selection);
    3. **empirical**: grouped holdout, perturbation robustness, near-OOD
       (far-OOD recorded as diagnostic, not selector);
    4. **scientific**: simplicity, stability, identifiability, asymptotic
       behavior, surviving alternatives, post-freeze verification.

    Returns the profile plus a one-line verdict line per judgment.
    """
    sections = {
        "mathematical": dict(mathematical or {}),
        "dimensional": dict(dimensional or {}),
        "empirical": dict(empirical or {}),
        "scientific": dict(scientific or {}),
    }
    lines = [f"Candidate Evidence Profile: {name}"]
    math_ok = sections["mathematical"].get("finite", True) and sections["mathematical"].get("real", True)
    lines.append(f"  mathematical: {'sound' if math_ok else 'UNSOUND'}")
    dim_status = sections["dimensional"].get("status", "UNRESOLVED")
    lines.append(f"  dimensional: {dim_status}")
    lines.append(f"  empirical: {_format_value(sections['empirical'].get('summary', 'see metrics'))}")
    lines.append(f"  scientific: {_format_value(sections['scientific'].get('summary', 'see metrics'))}")
    selectable = bool(math_ok) and dim_status == "VALID" and bool(sections["empirical"])
    lines.append(
        "  selectable for scientific ranking: "
        + ("yes" if selectable else "NO -- " + (
            "mathematically unsound" if not math_ok
            else "dimensional status unresolved" if dim_status == "UNRESOLVED"
            else "dimensionally invalid" if dim_status == "INVALID"
            else "no empirical evidence")))
    return {"name": name, "sections": sections, "selectable": selectable, "report": "\n".join(lines)}


def build_evidence_report(candidates: list[CandidateEvidence]) -> dict[str, Any]:
    """Render sections plus a structural non-uniqueness assessment.

    * zero admissible candidates: evidence is absent, not inconclusive (NO_VALID_CANDIDATE);
    * exactly one: strongest support statement the metrics allow (UNIQUE_SUPPORT);
    * several: explicit non-uniqueness listing what each survivor explains
      and where each is limited (MULTI_HYPOTHESIS).
    """
    sections = []
    for candidate in candidates:
        lines = [f"Candidate: {candidate.name}"]
        for key, value in candidate.metrics.items():
            lines.append(f"  {key}: {_format_value(value)}")
        lines.append(f"  admissible: {candidate.admissible}")
        for limitation in candidate.limitations:
            lines.append(f"  limitation: {limitation}")
        sections.append("\n".join(lines))
    admissible = [candidate for candidate in candidates if candidate.admissible]
    if not admissible:
        epistemic_state = "NO_VALID_CANDIDATE"
        assessment = "No admissible candidate: the evidence does not support any of the tested structures."
    elif len(admissible) == 1:
        epistemic_state = "UNIQUE_SUPPORT"
        assessment = (
            f"Single admissible candidate ({admissible[0].name}); support is "
            "conditional on the listed metrics and limitations, not a uniqueness proof."
        )
    else:
        epistemic_state = "MULTI_HYPOTHESIS"
        names = ", ".join(candidate.name for candidate in admissible)
        assessment = (
            f"Structural evidence is non-unique across {len(admissible)} admissible "
            f"candidates ({names}); selection among them requires criteria beyond "
            "the reported metrics."
        )
    return {
        "sections": sections,
        "n_candidates": len(candidates),
        "n_admissible": len(admissible),
        "epistemic_state": epistemic_state,
        "assessment": assessment,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


__all__ = [
    "EVIDENCE_STATUSES",
    "CandidateEvidence",
    "build_evidence_report",
    "candidate_evidence_profile",
]

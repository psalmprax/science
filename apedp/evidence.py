"""Evidence tiers: what a verdict rests on.

Every verdict in APEDP is a claim backed by *something*. This module makes the
kind of that something part of the verdict, so a consumer can tell "proved" from
"measured" from "assumed" without reading the method that produced it, and so the
system cannot present a heuristic as a demonstration.

The tiers, strongest first:

``PROVED``
    The claim follows from an exact argument that admits no numeric tolerance: an
    infeasibility certificate, an exactly isolated pole, a dimensional closure, a
    parse failure. Re-running it with different sampling cannot change the answer.
``EMPIRICAL``
    The claim rests on measurements: evaluations across a declared envelope, limit
    evaluation, out-of-sample data, benchmark runs. Change the sampling or the data
    and the answer may change.
``HEURISTIC``
    The claim rests on thresholds, weights or declared structure that was not
    itself derived: saturation and ceiling cut-offs chosen by convention, score
    weights, declared invariants compared against magnitude thresholds.
``UNRESOLVED``
    Nothing was measured -- a probe crashed, timed out, or the claim was never
    evaluated. This is the *absence* of a tier, and it is never reported as a pass.

Two combination rules, because refutation and confirmation are not symmetric:

* a **refutation** is as strong as its best refuting evidence: one proved
  counterexample settles the matter (:func:`strongest`);
* a **clearance** is as strong as its weakest supporting measurement: every claim
  has to hold (:func:`weakest`), and a single :attr:`EvidenceTier.UNRESOLVED`
  contributor collapses the clearance to unresolved rather than letting the
  remaining measurements carry it.

A clearance therefore never reaches ``PROVED``: it is bounded by whatever the
probes actually measured. That asymmetry is the whole point of the module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceTier(str, Enum):
    """How a verdict was established. Ordered strongest to weakest."""

    PROVED = "PROVED"
    EMPIRICAL = "EMPIRICAL"
    HEURISTIC = "HEURISTIC"
    UNRESOLVED = "UNRESOLVED"

    @property
    def rank(self) -> int:
        """Strength rank: lower is stronger (``PROVED`` is 0)."""
        return _RANK[self]

    @property
    def is_resolved(self) -> bool:
        """Whether *any* evidence was collected for the claim."""
        return self is not EvidenceTier.UNRESOLVED

    def __str__(self) -> str:
        return self.value


_RANK: dict[EvidenceTier, int] = {
    EvidenceTier.PROVED: 0,
    EvidenceTier.EMPIRICAL: 1,
    EvidenceTier.HEURISTIC: 2,
    EvidenceTier.UNRESOLVED: 3,
}

#: Claims are only as good as what measured them. A clearance cannot be proved:
#: no finite set of evaluations establishes a universally quantified statement.
CLEARANCE_CEILING = EvidenceTier.EMPIRICAL


def strongest(tiers: list[EvidenceTier]) -> EvidenceTier:
    """Best evidence in the list -- the rule for a *refutation*.

    One proved counterexample outweighs any number of measurements, so an
    ``UNRESOLVED`` side-probe never weakens an established refutation. With no
    tiers at all there is no evidence and the result is ``UNRESOLVED``.
    """
    resolved = [t for t in tiers if t.is_resolved]
    if not resolved:
        return EvidenceTier.UNRESOLVED
    return min(resolved, key=lambda t: t.rank)


def weakest(tiers: list[EvidenceTier]) -> EvidenceTier:
    """Worst evidence in the list -- the rule for a *clearance*.

    A single unresolved contributor collapses the result: a chain of claims is
    only as strong as its widest hole, and a hole is not a pass.
    """
    if not tiers:
        return EvidenceTier.UNRESOLVED
    return max(tiers, key=lambda t: t.rank)


def clearance_tier(tiers: list[EvidenceTier]) -> EvidenceTier:
    """Tier of a surviving/positive verdict: :func:`weakest`, capped at the ceiling."""
    tier = weakest(tiers)
    if tier.is_resolved and tier.rank < CLEARANCE_CEILING.rank:
        return CLEARANCE_CEILING
    return tier


@dataclass(frozen=True)
class EvidenceAssessment:
    """A verdict's tier plus the reason it holds that tier."""

    tier: EvidenceTier
    basis: str
    contributors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "basis": self.basis,
            "contributors": list(self.contributors),
        }

    def __str__(self) -> str:
        detail = f"; {self.basis}" if self.basis else ""
        return f"{self.tier.value}{detail}"


@dataclass
class EvidenceLedger:
    """Collects tiered contributions so a verdict can publish its own basis.

    Contributors are recorded as ``(tier, label)`` pairs: the label names the
    probe, signal or field that contributed, which is what a reader needs in order
    to disagree with the tier.
    """

    contributions: list[tuple[EvidenceTier, str]] = field(default_factory=list)

    def add(self, tier: EvidenceTier, label: str) -> None:
        self.contributions.append((tier, label))

    @property
    def tiers(self) -> list[EvidenceTier]:
        return [tier for tier, _ in self.contributions]

    def for_refutation(self, basis: str = "") -> EvidenceAssessment:
        """Assess a claim that *something is wrong*: best evidence decides."""
        tier = strongest(self.tiers)
        return EvidenceAssessment(
            tier=tier,
            basis=basis or _describe(tier, self.contributions, "refutation"),
            contributors=tuple(label for _, label in self.contributions),
        )

    def for_clearance(self, basis: str = "") -> EvidenceAssessment:
        """Assess a claim that *something is fine*: worst evidence decides."""
        tier = clearance_tier(self.tiers)
        return EvidenceAssessment(
            tier=tier,
            basis=basis or _describe(tier, self.contributions, "clearance"),
            contributors=tuple(label for _, label in self.contributions),
        )

    def for_absence(self, basis: str) -> EvidenceAssessment:
        """Assess a claim nothing measured."""
        return EvidenceAssessment(
            tier=EvidenceTier.UNRESOLVED,
            basis=basis,
            contributors=tuple(label for _, label in self.contributions),
        )


def _describe(
    tier: EvidenceTier,
    contributions: list[tuple[EvidenceTier, str]],
    kind: str,
) -> str:
    if not contributions:
        return f"no evidence recorded for this {kind}"
    by_tier: dict[str, list[str]] = {}
    for contribution_tier, label in contributions:
        by_tier.setdefault(contribution_tier.value, []).append(label)
    summary = "; ".join(
        f"{name}: {', '.join(labels)}" for name, labels in sorted(by_tier.items())
    )
    rule = "strongest wins" if kind == "refutation" else "weakest link governs"
    return f"{kind} tier {tier.value} ({rule}) from {summary}"

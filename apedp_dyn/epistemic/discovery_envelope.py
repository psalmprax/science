"""Discovery that a gate can admit, but only a measurement can confirm.

The referee stack can prove that a candidate passed every check it was given; it cannot
prove that the candidate is right about the world. This module makes that difference a
type rather than a caveat in a docstring: an admitted candidate is
``AWAITING_EXTERNAL_VERIFICATION``, the envelope names the measurement that would settle
it, and there is no state called "verified" to promote it into.

The rules are deliberately narrow, and each one exists because its absence produces a
plausible-looking result:

* a claim needs at least one gate — otherwise nothing was checked;
* a claim needs at least one withheld claim — otherwise a caller will read the gate set
  as complete, which is the failure ``docs/SCAFFOLD_INVENTORY.md`` calls "a constant
  returned as a prediction";
* a claim needs an :class:`ExternalVerificationRequest`, and that request must say what
  to measure, how it would fail, and why it cannot be run here.

Usage::

    envelope = admit_for_external_verification(
        candidate_id="dqi-arm-7",
        claim="this arm executes 686,014.8 Toffolis per shot at the pinned instance",
        gates=[GateResult("decoder_correct", True, "64/64 shots")],
        evidence={"toffolis_per_shot": 686_014.8},
        withheld_claims=("that the circuit is better than the official decoder on hardware",),
        external=ExternalVerificationRequest(
            domain="quantum",
            measurement="run the arm on a calibrated device and score it",
            acceptance="R * Q^2 * T below the reference",
            why_not_available_here="no quantum hardware or device calibration in this environment",
        ),
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class DiscoveryState(str, Enum):
    """The only two outcomes. A passed gate is admission, never confirmation."""

    REFUSED = "refused"
    AWAITING_EXTERNAL_VERIFICATION = "awaiting_external_verification"


class IncompleteDiscoveryClaimError(ValueError):
    """Raised when a claim omits its limits or the check that would settle it."""


@dataclass(frozen=True)
class GateResult:
    """One check the candidate was actually put through."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ExternalVerificationRequest:
    """The measurement a human (or an instrument) must run before the claim is settled."""

    domain: str
    measurement: str
    acceptance: str
    why_not_available_here: str

    def __post_init__(self) -> None:
        for name in ("domain", "measurement", "acceptance", "why_not_available_here"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise IncompleteDiscoveryClaimError(
                    f"external verification needs a non-empty {name}"
                )

    def to_dict(self) -> dict[str, str]:
        return {
            "domain": self.domain,
            "measurement": self.measurement,
            "acceptance": self.acceptance,
            "why_not_available_here": self.why_not_available_here,
        }


@dataclass(frozen=True)
class DiscoveryEnvelope:
    """An admitted candidate, the evidence behind it, and what remains unestablished."""

    candidate_id: str
    claim: str
    state: DiscoveryState
    gates: tuple[GateResult, ...]
    evidence: Mapping[str, Any]
    withheld_claims: tuple[str, ...]
    external: ExternalVerificationRequest
    refused_by: str | None = None

    @property
    def admitted(self) -> bool:
        return self.state is DiscoveryState.AWAITING_EXTERNAL_VERIFICATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "claim": self.claim,
            "state": self.state.value,
            "gates": [
                {"name": gate.name, "passed": gate.passed, "detail": gate.detail}
                for gate in self.gates
            ],
            "evidence": dict(self.evidence),
            "withheld_claims": list(self.withheld_claims),
            "external_verification": self.external.to_dict(),
            "refused_by": self.refused_by,
        }


def admit_for_external_verification(
    *,
    candidate_id: str,
    claim: str,
    gates: Sequence[GateResult],
    evidence: Mapping[str, Any],
    withheld_claims: Sequence[str],
    external: ExternalVerificationRequest,
) -> DiscoveryEnvelope:
    """Record a candidate as admitted or refused — never as verified.

    Args:
        candidate_id: Stable identifier the ledger already uses.
        claim: The narrow statement the gates support, in the candidate's own numbers.
        gates: Every check that was run. A failing gate refuses the candidate.
        evidence: The measured numbers the claim rests on.
        withheld_claims: What the gate set does **not** establish. Required and non-empty:
            an empty list means the caller believes the gates were exhaustive.
        external: The measurement that would settle the remaining question.

    Raises:
        IncompleteDiscoveryClaimError: if there is no gate, no withheld claim, no
            identifier/claim text, or the external request is not constructed.

    Returns:
        A :class:`DiscoveryEnvelope` whose state is ``REFUSED`` when any gate failed and
        ``AWAITING_EXTERNAL_VERIFICATION`` otherwise.
    """
    if not candidate_id.strip():
        raise IncompleteDiscoveryClaimError("a candidate needs an identifier")
    if not claim.strip():
        raise IncompleteDiscoveryClaimError("a candidate needs a claim in its own numbers")
    if not gates:
        raise IncompleteDiscoveryClaimError(
            "a candidate needs at least one gate; nothing was checked otherwise"
        )
    if not withheld_claims or any(not text.strip() for text in withheld_claims):
        raise IncompleteDiscoveryClaimError(
            "state what the gates do not establish; an empty list reads as 'everything'"
        )
    if not isinstance(external, ExternalVerificationRequest):
        raise IncompleteDiscoveryClaimError(
            "a claim needs the external measurement that would settle it"
        )

    failed = [gate for gate in gates if not gate.passed]
    return DiscoveryEnvelope(
        candidate_id=candidate_id,
        claim=claim,
        state=(
            DiscoveryState.REFUSED
            if failed
            else DiscoveryState.AWAITING_EXTERNAL_VERIFICATION
        ),
        gates=tuple(gates),
        evidence=dict(evidence),
        withheld_claims=tuple(withheld_claims),
        external=external,
        refused_by=failed[0].name if failed else None,
    )

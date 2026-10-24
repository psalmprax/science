"""Hard scientific gates. No backend may override a failed gate."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str


def dimensional_gate(evidence) -> GateResult:
    if not evidence.physics.passed:
        return GateResult(False, "Physics validation failed.")
    return GateResult(True, "Physics validation passed.")


def domain_gate(evidence) -> GateResult:
    if not evidence.domain_valid:
        return GateResult(False, "Candidate is outside its declared domain.")
    return GateResult(True, "Candidate is inside declared domain.")


def numerical_gate(evidence) -> GateResult:
    if not evidence.numerical.passed:
        return GateResult(False, "Numerical validation failed.")
    return GateResult(True, "Numerical validation passed.")


def run_hard_gates(evidence) -> list:
    """Return the list of failure reasons (empty means all gates passed).

    Hard gates cover dimensional and domain failures only: those are
    unrecoverable by more testing. Numerical failure routes to
    run_more_tests via the backend instead of rejecting outright.
    """
    return [g.reason for g in (dimensional_gate(evidence),
                               domain_gate(evidence))
            if not g.passed]

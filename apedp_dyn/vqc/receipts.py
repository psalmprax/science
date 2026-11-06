"""Adjudication receipts: the evidence a claim is allowed to stand on.

Status: experimental

A receipt is the platform's only output that a reader is asked to trust, so it is
built to be *checkable* rather than persuasive. It records what the circuit was
(its canonical hash), where it came from, what it cost, what the simulation
measured, how the check was powered, and a verdict from a closed set:

``VALID``
    The circuit validated, every declared ancilla came back clean, and the
    measured outcome distribution is statistically consistent with the declared
    expectation, in a test with enough shots to have detected the declared
    minimum effect.
``INVALID``
    A check failed in a way the data can distinguish -- a dirty ancilla, a
    distribution off by more than `sigma` standard errors, or a circuit that does
    not validate.
``UNRESOLVED``
    Nothing failed and nothing is established, because the run was too small to
    detect the declared effect. This is the honest third option: the alternative
    is promoting a handful of shots into a pass.

The statistics are deliberately simple -- a per-outcome normal approximation,
plus the shot count required to detect `min_effect` at `sigma` -- and they are
reported, not hidden: `checks[].measured` carries the deviations and the required
shot count, so a reader can redo the arithmetic.
"""
from __future__ import annotations

import dataclasses
import json
import math
import platform
from collections.abc import Mapping
from typing import Any

import numpy as np

from apedp_dyn.vqc.backends import Backend, RunResult, StatevectorBackend, distribution
from apedp_dyn.vqc.circuit import Circuit, CircuitValidationError

__all__ = ["VERDICTS", "Check", "Receipt", "adjudicate"]

VERDICTS = ("VALID", "INVALID", "UNRESOLVED")


@dataclasses.dataclass(frozen=True)
class Check:
    """One adjudicated property, with the numbers it was decided on."""

    name: str
    requirement: str
    measured: Mapping[str, Any]
    passed: bool | None

    def as_dict(self) -> dict:
        return {"name": self.name, "requirement": self.requirement,
                "measured": dict(self.measured), "passed": self.passed}


@dataclasses.dataclass(frozen=True)
class Receipt:
    """A machine-readable record of one adjudication."""

    identity: Mapping[str, Any]
    provenance: Mapping[str, Any]
    checks: tuple[Check, ...]
    resources: Mapping[str, Any]
    ancillas: Mapping[str, Any]
    reproducibility: Mapping[str, Any]
    statistics: Mapping[str, Any]
    verdict: str

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict {self.verdict!r} is not one of {VERDICTS}")

    def failed(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if check.passed is False)

    def unresolved(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if check.passed is None)

    def as_dict(self) -> dict:
        return {
            "identity": dict(self.identity),
            "provenance": dict(self.provenance),
            "verdict": self.verdict,
            "checks": [check.as_dict() for check in self.checks],
            "resources": dict(self.resources),
            "ancillas": dict(self.ancillas),
            "reproducibility": dict(self.reproducibility),
            "statistics": dict(self.statistics),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"


def _normalise(expected: Mapping[str, float]) -> dict[str, float]:
    if not expected:
        raise ValueError("expected distribution is empty")
    total = sum(float(value) for value in expected.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"expected distribution sums to {total}")
    return {str(key): float(value) / total for key, value in expected.items()}


def _required_shots(expected: Mapping[str, float], *, sigma: float,
                    min_effect: float) -> int:
    """Fewest shots that can separate `expected` from a `min_effect` shift."""
    if min_effect <= 0.0:
        raise ValueError(f"min_effect must be positive, got {min_effect}")
    required = 0
    for probability in expected.values():
        variance = max(probability * (1.0 - probability), 1e-12)
        required = max(required, math.ceil(sigma ** 2 * variance / min_effect ** 2))
    return max(required, 1)


def _distribution_check(circuit: Circuit, result: RunResult,
                        expected: Mapping[str, float], *,
                        sigma: float, min_effect: float) -> tuple[Check, Check, dict]:
    observed = distribution(result)
    strings = sorted(set(expected) | set(observed))
    deviations = {}
    z_scores = {}
    for key in strings:
        want = expected.get(key, 0.0)
        got = observed.get(key, 0.0)
        deviations[key] = got - want
        if want >= 1.0:
            # A declared certainty has no binomial variance: the outcome either happened
            # on every shot or the claim is false. Scored as `shots` sigma, which is
            # finite, so the receipt stays valid JSON and still fails any threshold.
            z_scores[key] = 0.0 if got == want else float(max(1, result.shots))
        elif want > 0.0:
            z_scores[key] = (got - want) / math.sqrt(want * (1.0 - want) / result.shots)
    worst_key = max(z_scores, key=lambda key: abs(z_scores[key])) if z_scores else None
    worst = z_scores[worst_key] if worst_key is not None else 0.0
    total_variation = 0.5 * sum(abs(deviations[key]) for key in strings)

    consistency = Check(
        name="outcome distribution matches the declared expectation",
        requirement=(f"every outcome within {sigma:g} standard errors, so a run whose "
                     f"behaviour differs on a superposition cannot pass"),
        measured={"shots": result.shots, "observed": dict(sorted(observed.items())),
                  "expected": dict(sorted(expected.items())),
                  "worst_outcome": worst_key, "worst_z": worst,
                  "max_abs_deviation": max(abs(value) for value in deviations.values()),
                  "total_variation_distance": total_variation},
        passed=bool(abs(worst) <= sigma),
    )

    required = _required_shots(expected, sigma=sigma, min_effect=min_effect)
    # `None`, not `False`: an underpowered run is not evidence of a wrong circuit,
    # and it must not be reported as one.
    power = Check(
        name="the test had power to detect the declared effect",
        requirement=(f"at least {required} shots are needed to detect a {min_effect:g} "
                     f"shift at {sigma:g} sigma"),
        measured={"shots": result.shots, "required_shots": required,
                  "min_effect": min_effect, "sigma": sigma},
        passed=None if result.shots < required else True,
    )
    return consistency, power, {"required_shots": required, "min_effect": min_effect}


def adjudicate(circuit: Circuit, *, expected: Mapping[str, float], shots: int = 4096,
               seed: int | None = 0, backend: Backend | None = None,
               sigma: float = 4.0, min_effect: float = 0.05,
               ancilla_tolerance: float = 1e-9,
               provenance: Mapping[str, Any] | None = None) -> Receipt:
    """Run `circuit`, compare it with `expected`, and seal the evidence in a receipt."""
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    backend = backend or StatevectorBackend()
    checks: list[Check] = []
    identity = {"circuit_sha256": circuit.sha256(),
                "schema_version": circuit.schema_version,
                "canonical_json_sha256_basis": "sha256(circuit.to_json())"}

    try:
        circuit.validate()
        valid = True
        validation_error = None
    except CircuitValidationError as error:
        valid = False
        validation_error = str(error)

    checks.append(Check(
        name="circuit is a valid VQC program",
        requirement="every gate, qubit, bit and condition is defined and in range",
        measured={"error": validation_error}, passed=valid))
    if not valid:
        return Receipt(
            identity=identity,
            provenance=dict(provenance or {}),
            checks=tuple(checks),
            resources=circuit.resources(),
            ancillas={"declared": [int(a) for a in circuit.ancillas], "checked": False},
            reproducibility={"backend": backend.name, "shots": shots, "seed": seed},
            statistics={},
            verdict="INVALID",
        )

    result = backend.run(circuit, shots=shots, seed=seed)

    if circuit.ancillas:
        worst_ancilla = max(circuit.ancillas,
                            key=lambda qubit: result.ancilla_excitation[qubit])
        excitation = result.ancilla_excitation[worst_ancilla]
        ancillas = {"declared": [int(a) for a in circuit.ancillas],
                    "checked": True,
                    "excitation": {str(k): v for k, v in
                                   sorted(result.ancilla_excitation.items())},
                    "worst_ancilla": int(worst_ancilla),
                    "worst_excitation": excitation,
                    "tolerance": ancilla_tolerance}
        checks.append(Check(
            name="declared ancillas return to |0>",
            requirement=(f"each ancilla is left in |0> within {ancilla_tolerance:g}: "
                         f"workspace is cleaned, not abandoned"),
            measured=ancillas,
            passed=bool(excitation <= ancilla_tolerance)))
    else:
        ancillas = {"declared": [], "checked": True, "excitation": {}}

    expected_normalised = _normalise(expected)
    consistency, power, power_stats = _distribution_check(
        circuit, result, expected_normalised, sigma=sigma, min_effect=min_effect)
    checks.extend([consistency, power])

    if any(check.passed is False for check in checks):
        verdict = "INVALID"
    elif any(check.passed is None for check in checks):
        verdict = "UNRESOLVED"
    elif consistency.passed is True and power.passed is True:
        verdict = "VALID"
    else:
        verdict = "UNRESOLVED"

    return Receipt(
        identity=identity,
        provenance=dict(provenance or {}),
        checks=tuple(checks),
        resources=circuit.resources(),
        ancillas=ancillas,
        reproducibility={
            "backend": backend.name,
            "shots": shots,
            "seed": seed,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "circuit_json_sha256": circuit.sha256(),
        },
        statistics=power_stats,
        verdict=verdict,
    )

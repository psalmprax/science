"""Common harness for APEDP-Q quantum-circuit optimization experiments.

Status: experimental

Every experiment follows the same four-operation pipeline:

1. ``generate``: the proposer emits candidate configurations without access to
   the final confirmation holdout.
2. ``screen``: the proposer checks mathematical/structural constraints on an
   exact integer model.  Failed screens are recorded but never promoted.
3. ``compile`` + ``measure_resources``: the exact artifact is built and its
   resource cost is measured with the competition's cost function.
4. ``verify``: the artifact is sent to the independent official verifier.

Estimated cost and verified measured cost are recorded separately and never
conflated.
"""
from __future__ import annotations

import abc
import dataclasses
from pathlib import Path
from typing import Any

from apedp_dyn.quantum_compilation.candidates import (
    VerifierReceipt,
    candidate_id_for,
    sha256_file,
)
from apedp_dyn.quantum_compilation.contracts import VerifierContract, load_contract
from apedp_dyn.quantum_compilation.invsqrt import base_precision
from apedp_dyn.quantum_compilation.referee import run_invsqrt_verifier

__all__ = [
    "Candidate",
    "CandidateArtifact",
    "CandidateScreen",
    "ExperimentResult",
    "ExperimentRunner",
    "Proposer",
    "run_reference_control",
]


@dataclasses.dataclass(frozen=True)
class Candidate:
    """A proposer-emitted candidate configuration."""

    candidate_id: str
    generator: str
    parameters: dict[str, Any]
    parent_candidate_id: str | None = None


@dataclasses.dataclass(frozen=True)
class CandidateScreen:
    """Outcome of the proposer-side structural/mathematical screen."""

    valid: bool
    reason: str | None
    metrics: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class CandidateArtifact:
    """A compiled circuit artifact plus its byte hash.

    ``metadata`` carries proposer-computed circuit statistics (qubit count,
    toffoli-like count, rotation count, phase margin) needed by
    ``Proposer.measure_resources`` without rebuilding the circuit.
    """

    path: Path
    sha256: str
    format: str = "kmb"
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ExperimentResult:
    """One candidate's result through the full pipeline.

    ``estimated_cost`` comes from the proposer's model; ``measured_cost``
    comes from the official verifier.  They are distinct fields and must never
    be conflated.
    """

    candidate: Candidate
    screen: CandidateScreen
    artifact: CandidateArtifact | None
    estimated_cost: float | None
    measured_cost: float | None
    verification: VerifierReceipt | None
    status: str  # GENERATED, SCREENED_INVALID, SCREENED_VALID, MEASURED, VERIFIED, REJECTED


@dataclasses.dataclass(frozen=True)
class ExperimentOutcome:
    """Full outcome of one experiment run."""

    incumbent_score: float | None
    results: list[ExperimentResult]


class Proposer(abc.ABC):
    """Base class for quantum-circuit proposers.

    Subclasses implement ``generate``, ``screen``, ``compile``,
    ``measure_resources`` and ``verify``.  ``verify`` is provided by default
    and delegates to the official verifier subprocess.
    """

    @abc.abstractmethod
    def generate(
        self,
        contract: VerifierContract,
        incumbent: Candidate | None,
        budget: int,
    ) -> list[Candidate]:
        """Emit candidate configurations.  No access to the confirmation holdout."""

    @abc.abstractmethod
    def screen(self, candidate: Candidate, contract: VerifierContract) -> CandidateScreen:
        """Check mathematical/structural constraints on an exact model."""

    @abc.abstractmethod
    def compile(self, candidate: Candidate, contract: VerifierContract) -> CandidateArtifact:
        """Build the exact circuit artifact."""

    @abc.abstractmethod
    def measure_resources(
        self, artifact: CandidateArtifact, contract: VerifierContract
    ) -> float:
        """Measure the exact resource cost with the competition cost function."""

    def verify(
        self,
        artifact: CandidateArtifact,
        contract: VerifierContract,
        *,
        python_executable: str | None = None,
        timeout_s: float = 1800.0,
        salt: str = "",
    ) -> VerifierReceipt:
        """Run the official verifier in a separate process."""
        return run_invsqrt_verifier(
            contract,
            artifact.path,
            python_executable=python_executable or _default_python(),
            salt=salt,
            timeout_s=timeout_s,
        )


def _default_python() -> str:
    import sys

    return sys.executable


class ExperimentRunner:
    """Orchestrates the generate → screen → compile/measure → verify pipeline."""

    def __init__(
        self,
        proposer: Proposer,
        contract: VerifierContract,
        incumbent: Candidate | None = None,
        incumbent_score: float | None = None,
    ) -> None:
        self.proposer = proposer
        self.contract = contract
        self.contract.assert_locked()
        self.incumbent = incumbent
        self.incumbent_score = incumbent_score

    def run(self, budget: int, *, salt: str = "") -> ExperimentOutcome:
        candidates = self.proposer.generate(self.contract, self.incumbent, budget)
        results: list[ExperimentResult] = []
        for candidate in candidates:
            screen = self.proposer.screen(candidate, self.contract)
            if not screen.valid:
                results.append(
                    ExperimentResult(
                        candidate=candidate,
                        screen=screen,
                        artifact=None,
                        estimated_cost=None,
                        measured_cost=None,
                        verification=None,
                        status="SCREENED_INVALID",
                    )
                )
                continue

            artifact = self.proposer.compile(candidate, self.contract)
            estimated_cost = self.proposer.measure_resources(artifact, self.contract)
            receipt = self.proposer.verify(
                artifact, self.contract, salt=salt
            )
            measured_cost = receipt.score if receipt.score is not None else estimated_cost
            status = "VERIFIED" if receipt.status == "PASS" else "REJECTED"
            results.append(
                ExperimentResult(
                    candidate=candidate,
                    screen=screen,
                    artifact=artifact,
                    estimated_cost=estimated_cost,
                    measured_cost=measured_cost,
                    verification=receipt,
                    status=status,
                )
            )

        return ExperimentOutcome(
            incumbent_score=self.incumbent_score,
            results=results,
        )


def run_reference_control(
    contract: VerifierContract | None = None,
    *,
    python_executable: str | None = None,
    timeout_s: float = 1800.0,
    salt: str = "",
) -> ExperimentResult:
    """Reproduce the official reference artifact as the experiment control.

    Verifies byte-identical artifact reproduction and runs the official
    verifier to reproduce the expected resource accounting.
    """
    if contract is None:
        contract = load_contract("invsqrt")

    params = base_precision(contract)
    candidate_id = candidate_id_for(
        "invsqrt",
        "reference-control",
        params,
        contract_sha256=contract.contract_sha256,
    )
    candidate = Candidate(
        candidate_id=candidate_id,
        generator="reference-control",
        parameters=params,
    )

    root = contract.competition_root
    artifact_path = Path(
        root
        / "sample_submission"
        / "invsqrt.kmb"
    )
    artifact = CandidateArtifact(
        path=artifact_path.resolve(),
        sha256=sha256_file(artifact_path),
    )

    receipt = run_invsqrt_verifier(
        contract,
        artifact.path,
        python_executable=python_executable or _default_python(),
        salt=salt,
        timeout_s=timeout_s,
    )

    return ExperimentResult(
        candidate=candidate,
        screen=CandidateScreen(valid=True, reason=None, metrics={}),
        artifact=artifact,
        estimated_cost=receipt.score,
        measured_cost=receipt.score,
        verification=receipt,
        status="VERIFIED" if receipt.status == "PASS" else "REJECTED",
    )

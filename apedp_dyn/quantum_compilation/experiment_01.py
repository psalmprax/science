"""Experiment 01: Reproduce reference + APEDP-Q candidate through unified harness.

Status: audit-only

Runs the competition reference artifact and the current APEDP-Q precision-search
candidate through the ExperimentRunner pipeline.  Both are verified with the
official full-domain verifier and their measured costs are compared.

Usage::

    python -m apedp_dyn.quantum_compilation.experiment_01 \
        --candidate-artifact <path> \
        [--python-executable /path/to/python3.13] \
        [--timeout 1800]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apedp_dyn.quantum_compilation.candidates import candidate_id_for, sha256_file
from apedp_dyn.quantum_compilation.contracts import VerifierContract, load_contract
from apedp_dyn.quantum_compilation.experiment import (
    Candidate,
    CandidateArtifact,
    CandidateScreen,
    ExperimentResult,
    ExperimentRunner,
    Proposer,
    run_reference_control,
)


class FixedCandidateProposer(Proposer):
    """Proposer that emits a single pre-built candidate artifact."""

    def __init__(self, candidate: Candidate, artifact: CandidateArtifact) -> None:
        self._candidate = candidate
        self._artifact = artifact

    def generate(self, contract: VerifierContract, incumbent: Candidate | None, budget: int) -> list[Candidate]:
        return [self._candidate]

    def screen(self, candidate: Candidate, contract: VerifierContract) -> CandidateScreen:
        return CandidateScreen(valid=True, reason=None, metrics={})

    def compile(self, candidate: Candidate, contract: VerifierContract) -> CandidateArtifact:
        return self._artifact

    def measure_resources(
        self, artifact: CandidateArtifact, contract: VerifierContract
    ) -> float:
        return 0.0  # measured cost comes from the verifier


def _load_candidate(
    contract: VerifierContract,
    artifact_path: str | Path,
    generator: str,
    parameters: dict[str, Any],
) -> tuple[Candidate, CandidateArtifact]:
    path = Path(artifact_path).resolve()
    candidate_id = candidate_id_for(
        "invsqrt",
        generator,
        parameters,
        contract_sha256=contract.contract_sha256,
    )
    candidate = Candidate(
        candidate_id=candidate_id,
        generator=generator,
        parameters=parameters,
    )
    artifact = CandidateArtifact(
        path=path,
        sha256=sha256_file(path),
    )
    return candidate, artifact


def _serialize(result: ExperimentResult) -> dict[str, Any]:
    return {
        "candidate_id": result.candidate.candidate_id,
        "generator": result.candidate.generator,
        "parameters": result.candidate.parameters,
        "screen_valid": result.screen.valid,
        "screen_reason": result.screen.reason,
        "artifact_path": str(result.artifact.path) if result.artifact else None,
        "artifact_sha256": result.artifact.sha256 if result.artifact else None,
        "estimated_cost": result.estimated_cost,
        "measured_cost": result.measured_cost,
        "verification_status": result.verification.status if result.verification else None,
        "verification_score": result.verification.score if result.verification else None,
        "verification_scope": result.verification.metrics.get("verification_scope")
        if result.verification
        else None,
        "verified_shots": result.verification.metrics.get("verified_shots")
        if result.verification
        else None,
        "status": result.status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-artifact",
        required=True,
        help="Path to the APEDP-Q candidate .kmb artifact.",
    )
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Python 3.13 interpreter for the official verifier.",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--salt", default="")
    args = parser.parse_args()

    contract = load_contract("invsqrt")
    contract.assert_locked()

    # Reference control through the harness.
    reference = run_reference_control(
        contract,
        python_executable=args.python_executable,
        timeout_s=args.timeout,
        salt=args.salt,
    )
    print(json.dumps({"phase": "reference_control", "result": _serialize(reference)}, indent=2))

    # APEDP-Q candidate through the harness.
    candidate, artifact = _load_candidate(
        contract,
        args.candidate_artifact,
        generator="invsqrt-nr-precision-v1",
        parameters={"delta_bits": 12, "coef_bits": 17, "square_bits": 28, "product_bits": 27, "result_bits": 28},
    )
    proposer = FixedCandidateProposer(candidate, artifact)
    runner = ExperimentRunner(proposer=proposer, contract=contract)
    outcome = runner.run(budget=1, salt=args.salt)
    result = outcome.results[0]
    print(
        json.dumps(
            {
                "phase": "apedp_q_candidate",
                "result": _serialize(result),
                "incumbent_score": outcome.incumbent_score,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

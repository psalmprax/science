"""Common harness for APEDP-Q quantum-circuit optimization experiments.

Status: experimental

Every experiment follows the same four-operation pipeline:

1. ``generate``: the proposer emits candidate configurations without access to
   the final confirmation holdout.
2. ``screen``: the proposer checks mathematical/structural constraints on an
   exact integer model.  Failed screens are recorded in
   :class:`~apedp_dyn.quantum_compilation.ledger.QuantumCandidateLedger` but
   never promoted.
3. ``compile`` + ``measure_resources``: the exact artifact is built and its
   resource cost is measured with the competition's cost function.
4. ``verify``: the artifact is sent to the independent official verifier and
   its receipt is recorded in the ledger.

Estimated cost and verified measured cost are recorded separately and never
conflated.
"""
from __future__ import annotations

import abc
import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

from apedp_dyn.quantum_compilation.candidates import (
    CandidateStatus,
    CircuitCandidate,
    VerifierReceipt,
    candidate_id_for,
    sha256_file,
)
from apedp_dyn.quantum_compilation.contracts import (
    VerifierContract,
    load_contract,
    sha256_tree,
)
from apedp_dyn.quantum_compilation.invsqrt import base_precision
from apedp_dyn.quantum_compilation.ledger import (
    QuantumCandidateLedger,
    chemistry_cost,
    dqi_cost,
)
from apedp_dyn.quantum_compilation.referee import run_verifier

__all__ = [
    "Candidate",
    "CandidateArtifact",
    "CandidateScreen",
    "DfthcsosProposer",
    "DqiProposer",
    "ExperimentOutcome",
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
    ledger: QuantumCandidateLedger | None = None

    @property
    def verdict_summary(self) -> dict[str, Any]:
        """Verdict counts from the run's ledger; an empty run has an empty summary."""
        if self.ledger is None:
            return {"candidates": 0, "by_status": {}, "unverifiable": 0, "eligible": 0}
        return self.ledger.verdict_summary()


def _default_python() -> str:
    return os.environ.get("APEDP_Q_KICKMIX_PYTHON") or sys.executable


class Proposer(abc.ABC):
    """Base class for quantum-circuit proposers.

    Subclasses implement ``generate``, ``screen``, ``compile``,
    ``measure_resources`` and ``verify``.  ``verify`` is provided by default
    and delegates to the official verifier subprocess for ``contract.benchmark``.
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
        **verifier_kwargs: Any,
    ) -> VerifierReceipt:
        """Run the official verifier for ``contract.benchmark`` in a separate process."""
        return run_verifier(
            contract,
            artifact.path,
            python_executable=python_executable or _default_python(),
            salt=salt,
            timeout_s=timeout_s,
            **verifier_kwargs,
        )


class ExperimentRunner:
    """Orchestrates the generate -> screen -> compile/measure -> verify pipeline."""

    def __init__(
        self,
        proposer: Proposer,
        contract: VerifierContract,
        incumbent: Candidate | None = None,
        incumbent_score: float | None = None,
        *,
        ledger: QuantumCandidateLedger | None = None,
        ledger_path: Path | str | None = None,
        python_executable: str | None = None,
        timeout_s: float = 1800.0,
        verifier_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.proposer = proposer
        self.contract = contract
        self.contract.assert_locked()
        self.incumbent = incumbent
        self.incumbent_score = incumbent_score
        self.ledger = (
            ledger
            if ledger is not None
            else QuantumCandidateLedger(persistence_path=ledger_path)
        )
        self.python_executable = python_executable
        self.timeout_s = timeout_s
        self.verifier_kwargs = dict(verifier_kwargs or {})

    def _to_circuit_candidate(
        self,
        candidate: Candidate,
        *,
        artifact: CandidateArtifact | None = None,
        screen: CandidateScreen | None = None,
        estimated_cost: float | None = None,
    ) -> CircuitCandidate:
        return CircuitCandidate(
            candidate_id=candidate.candidate_id,
            benchmark=self.contract.benchmark,
            generator_id=candidate.generator,
            parameters=dict(candidate.parameters),
            artifact_path=str(artifact.path) if artifact is not None else "",
            artifact_sha256=artifact.sha256 if artifact is not None else "",
            contract_sha256=self.contract.contract_sha256,
            parent_candidate_id=candidate.parent_candidate_id,
            screen_metrics=dict(screen.metrics) if screen is not None else {},
            screen_score_estimate=estimated_cost,
        )

    def _ensure_registered(self, record: CircuitCandidate) -> CircuitCandidate:
        existing_ids = {node.candidate_id for node in self.ledger.all_nodes()}
        if record.candidate_id not in existing_ids:
            self.ledger.register(record)
        return record

    def run(
        self,
        budget: int,
        *,
        salt: str = "",
        python_executable: str | None = None,
        timeout_s: float | None = None,
        **verifier_kwargs: Any,
    ) -> ExperimentOutcome:
        candidates = self.proposer.generate(self.contract, self.incumbent, budget)
        results: list[ExperimentResult] = []
        effective_python = python_executable or self.python_executable
        effective_timeout = timeout_s if timeout_s is not None else self.timeout_s
        merged_kwargs = {**self.verifier_kwargs, **verifier_kwargs}

        for candidate in candidates:
            record = self._ensure_registered(self._to_circuit_candidate(candidate))
            screen = self.proposer.screen(candidate, self.contract)
            if not screen.valid:
                self.ledger.record_screen(
                    record,
                    passed=False,
                    metrics=screen.metrics,
                    score_estimate=None,
                    rejection_reason=screen.reason,
                )
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
            record = self._to_circuit_candidate(
                candidate,
                artifact=artifact,
                screen=screen,
                estimated_cost=estimated_cost,
            )
            record = self.ledger.record_screen(
                record,
                passed=True,
                metrics=screen.metrics,
                score_estimate=estimated_cost,
                rejection_reason=None,
            )

            verify_call_kwargs: dict[str, Any] = {
                "python_executable": effective_python,
                "timeout_s": effective_timeout,
                "salt": salt,
                **merged_kwargs,
            }
            try:
                receipt = self.proposer.verify(
                    artifact,
                    self.contract,
                    **verify_call_kwargs,
                )
            except TypeError:
                try:
                    receipt = self.proposer.verify(
                        artifact,
                        self.contract,
                        python_executable=effective_python,
                        timeout_s=effective_timeout,
                        salt=salt,
                    )
                except TypeError:
                    receipt = self.proposer.verify(artifact, self.contract)

            updated = self.ledger.record_verification(
                record,
                receipt,
                incumbent_score=self.incumbent_score,
            )
            if (
                updated.status is CandidateStatus.VERIFIED_IMPROVEMENT
                and receipt.score is not None
            ):
                self.incumbent_score = receipt.score

            measured_cost = receipt.score if receipt.status == "PASS" else None
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
            ledger=self.ledger,
        )


def run_reference_control(
    contract: VerifierContract | None = None,
    *,
    python_executable: str | None = None,
    timeout_s: float = 1800.0,
    salt: str = "",
    ledger: QuantumCandidateLedger | None = None,
    **verifier_kwargs: Any,
) -> ExperimentResult:
    """Reproduce the official reference artifact as the experiment control.

    Verifies byte-identical artifact reproduction and runs the official
    verifier to reproduce the expected resource accounting.
    """
    if contract is None:
        contract = load_contract("invsqrt")

    benchmark = contract.benchmark
    if benchmark == "invsqrt":
        try:
            params: dict[str, Any] = base_precision(contract)
        except RuntimeError:
            params = {
                "delta_bits": 14,
                "coef_bits": 17,
                "square_bits": 28,
                "product_bits": 28,
                "result_bits": 29,
            }
    else:
        params = dict(contract.constants)

    candidate_id = candidate_id_for(
        benchmark,
        "reference-control",
        params,
        contract_sha256=contract.contract_sha256,
    )
    candidate = Candidate(
        candidate_id=candidate_id,
        generator="reference-control",
        parameters=params,
    )

    if contract.sample_artifact is not None:
        artifact_path = contract.sample_artifact
        if benchmark == "dfthcsos" and artifact_path.name == "submission.npz":
            artifact_path = artifact_path.parent
    else:
        artifact_path = Path(
            contract.competition_root / "sample_submission" / f"{benchmark}.kmb"
        )
    digest = (
        sha256_tree(artifact_path)
        if artifact_path.is_dir()
        else sha256_file(artifact_path)
    )
    artifact = CandidateArtifact(
        path=artifact_path.resolve(),
        sha256=digest,
        format="dir" if artifact_path.is_dir() else artifact_path.suffix.lstrip("."),
    )

    receipt = run_verifier(
        contract,
        artifact.path,
        python_executable=python_executable or _default_python(),
        salt=salt,
        timeout_s=timeout_s,
        **verifier_kwargs,
    )

    if ledger is not None:
        record = CircuitCandidate(
            candidate_id=candidate.candidate_id,
            benchmark=benchmark,
            generator_id=candidate.generator,
            parameters=params,
            artifact_path=str(artifact.path),
            artifact_sha256=artifact.sha256,
            contract_sha256=contract.contract_sha256,
        )
        if not any(node.candidate_id == record.candidate_id for node in ledger.all_nodes()):
            ledger.register(record)
        record = ledger.record_screen(
            record,
            passed=True,
            metrics={},
            score_estimate=receipt.score,
        )
        ledger.record_verification(record, receipt, incumbent_score=None)

    return ExperimentResult(
        candidate=candidate,
        screen=CandidateScreen(valid=True, reason=None, metrics={}),
        artifact=artifact,
        estimated_cost=receipt.score,
        measured_cost=receipt.score if receipt.status == "PASS" else None,
        verification=receipt,
        status="VERIFIED" if receipt.status == "PASS" else "REJECTED",
    )


class DqiProposer(Proposer):
    """Benchmark II (DQI Reed-Solomon decoder) proposer adapter.

    Wraps pre-compiled or parameterized ``.kmb`` DQI decoder artifacts so they
    flow through the standardized ``generate -> screen -> compile/measure ->
    verify`` pipeline and are adjudicated by ``verify_submission_dqi.py``.
    """

    GENERATOR_ID = "dqi-decoder-v1"

    def __init__(
        self,
        artifact_paths: list[Path | str] | None = None,
        *,
        default_failure_rate: float = 0.0,
    ) -> None:
        self.artifact_paths = [Path(p) for p in (artifact_paths or [])]
        self.default_failure_rate = float(default_failure_rate)

    def generate(
        self,
        contract: VerifierContract,
        incumbent: Candidate | None,
        budget: int,
    ) -> list[Candidate]:
        if budget <= 0:
            return []
        paths = list(self.artifact_paths)
        if not paths and contract.sample_artifact is not None:
            paths = [contract.sample_artifact]
        candidates: list[Candidate] = []
        for path in paths[:budget]:
            resolved = Path(path).resolve()
            params: dict[str, Any] = {
                "artifact_path": str(resolved),
                "m": int(contract.constants.get("DEFAULT_M", 6)),
                "n": int(contract.constants.get("DEFAULT_N", 63)),
                "degree": int(contract.constants.get("DEFAULT_DEGREE", 2)),
                "r": int(contract.constants.get("DEFAULT_R", 5)),
                "assumed_failure_rate": self.default_failure_rate,
            }
            cid = candidate_id_for(
                contract.benchmark,
                self.GENERATOR_ID,
                params,
                contract_sha256=contract.contract_sha256,
            )
            candidates.append(
                Candidate(
                    candidate_id=cid,
                    generator=self.GENERATOR_ID,
                    parameters=params,
                    parent_candidate_id=(
                        incumbent.candidate_id if incumbent is not None else None
                    ),
                )
            )
        return candidates

    def screen(
        self, candidate: Candidate, contract: VerifierContract
    ) -> CandidateScreen:
        raw_path = candidate.parameters.get("artifact_path")
        if not raw_path:
            return CandidateScreen(
                valid=False,
                reason="missing artifact_path parameter",
                metrics={},
            )
        path = Path(str(raw_path))
        if not path.is_file():
            return CandidateScreen(
                valid=False,
                reason=f"DQI artifact not found: {path}",
                metrics={},
            )
        if path.stat().st_size <= 0:
            return CandidateScreen(
                valid=False,
                reason=f"DQI artifact is empty: {path}",
                metrics={},
            )
        return CandidateScreen(
            valid=True,
            reason=None,
            metrics={
                "artifact_bytes": path.stat().st_size,
                "m": candidate.parameters.get("m"),
                "n": candidate.parameters.get("n"),
                "degree": candidate.parameters.get("degree"),
                "r": candidate.parameters.get("r"),
            },
        )

    def compile(
        self, candidate: Candidate, contract: VerifierContract
    ) -> CandidateArtifact:
        path = Path(str(candidate.parameters["artifact_path"])).resolve()
        metadata: dict[str, Any] = {
            "repetition_factor": float(
                candidate.parameters.get("repetition_factor", 1.1950554)
            ),
            "num_qubits": int(candidate.parameters.get("num_qubits", 1893)),
            "toffoli_count": float(
                candidate.parameters.get("toffoli_count", 5_707_104.0)
            ),
        }
        return CandidateArtifact(
            path=path,
            sha256=sha256_file(path),
            format="kmb",
            metadata=metadata,
        )

    def measure_resources(
        self, artifact: CandidateArtifact, contract: VerifierContract
    ) -> float:
        repetition_factor = float(
            artifact.metadata.get("repetition_factor", 1.1950554)
        )
        num_qubits = int(artifact.metadata.get("num_qubits", 1893))
        toffoli_count = float(artifact.metadata.get("toffoli_count", 5_707_104.0))
        return dqi_cost(
            repetition_factor=repetition_factor,
            qubits=num_qubits,
            mean_toffoli_like=toffoli_count,
        )


class DfthcsosProposer(Proposer):
    """Benchmark III (DFT-HC-SOS chemistry) proposer adapter.

    Wraps candidate chemistry submission directories or ``submission.npz``
    factorizations (e.g., orthogonal ensemble rotations, truncated-SVD ensemble
    compressions, and angle-batch configurations) for adjudication by
    ``verifiers/dfthcsos/verify.py``.
    """

    GENERATOR_ID = "dfthcsos-ensemble-v1"

    def __init__(
        self,
        submission_paths: list[Path | str] | None = None,
        *,
        ensemble_sizes: list[int] | None = None,
        batch_sizes: list[int] | None = None,
        angle_bits: int = 12,
    ) -> None:
        self.submission_paths = [Path(p) for p in (submission_paths or [])]
        self.ensemble_sizes = list(ensemble_sizes or [54])
        self.batch_sizes = list(batch_sizes or [8])
        self.angle_bits = int(angle_bits)

    def generate(
        self,
        contract: VerifierContract,
        incumbent: Candidate | None,
        budget: int,
    ) -> list[Candidate]:
        if budget <= 0:
            return []
        paths = list(self.submission_paths)
        if not paths and contract.sample_artifact is not None:
            sample = contract.sample_artifact
            paths = [sample.parent if sample.name == "submission.npz" else sample]
        candidates: list[Candidate] = []
        for path in paths:
            resolved = Path(path).resolve()
            for c_ens in self.ensemble_sizes:
                for batch_j in self.batch_sizes:
                    if len(candidates) >= budget:
                        break
                    params: dict[str, Any] = {
                        "submission_path": str(resolved),
                        "ensemble_size": int(c_ens),
                        "batch_size": int(batch_j),
                        "angle_bits": int(self.angle_bits),
                    }
                    cid = candidate_id_for(
                        contract.benchmark,
                        self.GENERATOR_ID,
                        params,
                        contract_sha256=contract.contract_sha256,
                    )
                    candidates.append(
                        Candidate(
                            candidate_id=cid,
                            generator=self.GENERATOR_ID,
                            parameters=params,
                            parent_candidate_id=(
                                incumbent.candidate_id
                                if incumbent is not None
                                else None
                            ),
                        )
                    )
                if len(candidates) >= budget:
                    break
            if len(candidates) >= budget:
                break
        return candidates

    def screen(
        self, candidate: Candidate, contract: VerifierContract
    ) -> CandidateScreen:
        raw_path = candidate.parameters.get("submission_path")
        if not raw_path:
            return CandidateScreen(
                valid=False,
                reason="missing submission_path parameter",
                metrics={},
            )
        path = Path(str(raw_path))
        if path.is_file() and path.name == "submission.npz":
            sub_dir = path.parent
            npz_path = path
        elif path.is_dir():
            sub_dir = path
            npz_path = sub_dir / "submission.npz"
        else:
            return CandidateScreen(
                valid=False,
                reason=f"DFTHCSOS submission path not found: {path}",
                metrics={},
            )

        if not npz_path.is_file():
            return CandidateScreen(
                valid=False,
                reason=f"missing submission.npz in {sub_dir}",
                metrics={},
            )
        angle_circuits = sorted(sub_dir.glob("angle_k_*.kmb"))
        if not angle_circuits:
            return CandidateScreen(
                valid=False,
                reason=f"no angle_k_*.kmb circuits found in {sub_dir}",
                metrics={},
            )

        angle_bits = int(candidate.parameters.get("angle_bits", 12))
        if angle_bits < 12:
            return CandidateScreen(
                valid=False,
                reason=f"angle_bits={angle_bits} < 12 violates two-body error threshold",
                metrics={"angle_bits": angle_bits},
            )

        ensemble_size = int(candidate.parameters.get("ensemble_size", len(angle_circuits)))
        batch_size = int(candidate.parameters.get("batch_size", 8))
        if ensemble_size < 1 or batch_size < 1:
            return CandidateScreen(
                valid=False,
                reason="ensemble_size and batch_size must be positive",
                metrics={},
            )

        return CandidateScreen(
            valid=True,
            reason=None,
            metrics={
                "num_angle_circuits": len(angle_circuits),
                "ensemble_size": ensemble_size,
                "batch_size": batch_size,
                "angle_bits": angle_bits,
            },
        )

    def compile(
        self, candidate: Candidate, contract: VerifierContract
    ) -> CandidateArtifact:
        raw_path = Path(str(candidate.parameters["submission_path"])).resolve()
        sub_dir = raw_path.parent if raw_path.is_file() else raw_path
        digest = sha256_tree(sub_dir)
        metadata: dict[str, Any] = {
            "q_total": int(candidate.parameters.get("q_total", 730)),
            "t_total": float(candidate.parameters.get("t_total", 93_124.0)),
            "lambda_eff": float(candidate.parameters.get("lambda_eff", 49.36120289)),
            "ensemble_size": int(candidate.parameters.get("ensemble_size", 54)),
            "batch_size": int(candidate.parameters.get("batch_size", 8)),
        }
        return CandidateArtifact(
            path=sub_dir,
            sha256=digest,
            format="dir",
            metadata=metadata,
        )

    def measure_resources(
        self, artifact: CandidateArtifact, contract: VerifierContract
    ) -> float:
        q_total = int(artifact.metadata.get("q_total", 730))
        t_total = float(artifact.metadata.get("t_total", 93_124.0))
        lambda_eff = float(artifact.metadata.get("lambda_eff", 49.36120289))
        return chemistry_cost(
            lambda_eff=lambda_eff,
            qubits=q_total,
            toffoli_like=t_total,
        )

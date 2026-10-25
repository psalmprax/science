"""APEDP-Q: verifier-guided quantum circuit compilation.

Status: experimental

This package is the control plane between independent circuit proposers and
authoritative competition verifiers.  It does not synthesise circuits itself,
and it never treats a proposer-side cost estimate as a verified result.

The unit of work is a :class:`~apedp_dyn.quantum_compilation.candidates.CircuitCandidate`
whose artifact is byte-hashed and whose promotion requires a receipt from the
official verifier.  Candidate lineage is recorded in the existing APEDP-X
:class:`~apedp_dyn.epistemic.experiment_ledger.FullLineageExperimentLedger`.
"""
from __future__ import annotations

from apedp_dyn.quantum_compilation.candidates import (
    CandidateStatus,
    CircuitCandidate,
    VerifierReceipt,
    candidate_id_for,
    sha256_file,
)
from apedp_dyn.quantum_compilation.contracts import (
    ContractIntegrityError,
    VerifierContract,
    load_contract,
    sha256_tree,
)
from apedp_dyn.quantum_compilation.experiment import (
    Candidate,
    CandidateArtifact,
    CandidateScreen,
    DfthcsosProposer,
    DqiProposer,
    ExperimentOutcome,
    ExperimentResult,
    ExperimentRunner,
    Proposer,
    run_reference_control,
)
from apedp_dyn.quantum_compilation.invsqrt import (
    PhaseCalibrationResult,
    calibrate_phase_angles,
    calibrate_phase_npz,
    rewrite_self_inverse_toffoli_pairs,
)
from apedp_dyn.quantum_compilation.invsqrt_direct_poly import (
    DirectPolyProposer,
    PolyPrecision,
)
from apedp_dyn.quantum_compilation.invsqrt_qrom_line import QPrecision, QromLineProposer
from apedp_dyn.quantum_compilation.ledger import (
    BENCHMARK_ORDER,
    REFERENCE_COSTS,
    PortfolioScoreReport,
    PortfolioTaskSummary,
    QuantumCandidateLedger,
    aggregate_score,
    chemistry_cost,
    dqi_cost,
    invsqrt_cost,
    relative_reduction,
    score_ledger_portfolio,
    score_verified_portfolio,
)
from apedp_dyn.quantum_compilation.referee import (
    DEFAULT_DFTHCSOS_SHOTS_PER_ADDRESS,
    OfficialVerifierResult,
    build_dfthcsos_command,
    build_dqi_command,
    build_invsqrt_command,
    dfthcsos_verification_scope,
    dqi_verification_scope,
    invsqrt_verification_scope,
    run_dfthcsos_verifier,
    run_dqi_verifier,
    run_invsqrt_verifier,
    run_verifier,
)

__all__ = [
    "BENCHMARK_ORDER",
    "DEFAULT_DFTHCSOS_SHOTS_PER_ADDRESS",
    "REFERENCE_COSTS",
    "Candidate",
    "CandidateArtifact",
    "CandidateScreen",
    "CandidateStatus",
    "CircuitCandidate",
    "ContractIntegrityError",
    "DfthcsosProposer",
    "DirectPolyProposer",
    "DqiProposer",
    "ExperimentOutcome",
    "ExperimentResult",
    "ExperimentRunner",
    "OfficialVerifierResult",
    "PhaseCalibrationResult",
    "PolyPrecision",
    "PortfolioScoreReport",
    "PortfolioTaskSummary",
    "Proposer",
    "QPrecision",
    "QromLineProposer",
    "QuantumCandidateLedger",
    "VerifierContract",
    "VerifierReceipt",
    "aggregate_score",
    "build_dfthcsos_command",
    "build_dqi_command",
    "build_invsqrt_command",
    "calibrate_phase_angles",
    "calibrate_phase_npz",
    "candidate_id_for",
    "chemistry_cost",
    "dfthcsos_verification_scope",
    "dqi_cost",
    "dqi_verification_scope",
    "invsqrt_cost",
    "invsqrt_verification_scope",
    "load_contract",
    "relative_reduction",
    "rewrite_self_inverse_toffoli_pairs",
    "run_dfthcsos_verifier",
    "run_dqi_verifier",
    "run_invsqrt_verifier",
    "run_reference_control",
    "run_verifier",
    "score_ledger_portfolio",
    "score_verified_portfolio",
    "sha256_file",
    "sha256_tree",
]

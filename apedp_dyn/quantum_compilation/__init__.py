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
    ExperimentResult,
    ExperimentRunner,
    Proposer,
    run_reference_control,
)
from apedp_dyn.quantum_compilation.invsqrt_direct_poly import (
    DirectPolyProposer,
    PolyPrecision,
)
from apedp_dyn.quantum_compilation.invsqrt_qrom_line import QPrecision, QromLineProposer
from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger
from apedp_dyn.quantum_compilation.referee import (
    OfficialVerifierResult,
    run_invsqrt_verifier,
)

__all__ = [
    "Candidate",
    "CandidateArtifact",
    "CandidateScreen",
    "CandidateStatus",
    "CircuitCandidate",
    "ContractIntegrityError",
    "DirectPolyProposer",
    "ExperimentResult",
    "ExperimentRunner",
    "OfficialVerifierResult",
    "PolyPrecision",
    "Proposer",
    "QPrecision",
    "QromLineProposer",
    "QuantumCandidateLedger",
    "VerifierContract",
    "VerifierReceipt",
    "candidate_id_for",
    "load_contract",
    "run_invsqrt_verifier",
    "run_reference_control",
    "sha256_file",
    "sha256_tree",
]

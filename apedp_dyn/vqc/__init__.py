"""APEDP-VQC: the verifiable quantum-computing core.

Status: experimental

The DQI work in :mod:`apedp_dyn.quantum_compilation` is the first *workload* of
this platform: a real circuit, resource-accounted to the gate and verified by a
competition's own verifier. This package is the reusable machinery under it --
the piece that can accept a different circuit tomorrow.

What exists here, and what each one is for:

:mod:`~apedp_dyn.vqc.gates`
    One table of unitaries, so a name means the same thing on every backend.
:mod:`~apedp_dyn.vqc.circuit`
    The versioned IR: gates, measurements with reset, classical feedforward,
    noise operations, declared ancillas, metadata, canonical JSON and a
    deterministic SHA-256 identity.
:mod:`~apedp_dyn.vqc.backends`
    The backend interface and the exact statevector backend (dense amplitudes,
    `MR` semantics, one trajectory per shot, residual ancilla excitation).
:mod:`~apedp_dyn.vqc.noise`
    Exact density-matrix channels -- bit/phase flip, depolarising, amplitude
    damping, dephasing -- with trace, Hermiticity and positivity recomputed.
:mod:`~apedp_dyn.vqc.logical`
    The distance-3 bit-flip repetition code: encode, inject errors, extract
    syndromes, decode, correct, and measure the logical failure rate against its
    exact combinatorial prediction.
:mod:`~apedp_dyn.vqc.five_qubit`
    The perfect five-qubit code: encoder, syndrome extraction, conditional
    correction, and its exact logical failure rate under depolarising noise.
:mod:`~apedp_dyn.vqc.equivalence`
    The circuit referee's equivalence check in two tiers: build both unitaries and
    compare them up to one global phase (dense, near ten qubits), or compare the
    conjugation action on the Pauli generators (Clifford, exact at any size). Both
    reject a candidate that merely reproduces the measurement statistics.
:mod:`~apedp_dyn.vqc.decoders`
    The decoder interface, the distance-3 lookup table, and exact minimum-weight
    perfect matching on a defect graph.
:mod:`~apedp_dyn.vqc.discovery`
    Admission for discovered circuits: a candidate is promoted only when a referee
    has decided its equivalence (or its declared distribution), and `UNRESOLVED` is
    not a promotion.
:mod:`~apedp_dyn.vqc.stabilizer`
    A stabilizer backend: exact Clifford execution over `n` commuting Pauli
    generators instead of `2**n` amplitudes, with measurement, reset and
    feedforward. It carries a 100-qubit GHZ chain the dense tier cannot hold.
:mod:`~apedp_dyn.vqc.receipts`
    Adjudication: a machine-readable receipt with VALID, INVALID or UNRESOLVED,
    and the arithmetic the verdict was taken on.

Boundaries, stated here because the package is easy to over-read: this is not a
fault-tolerance result (no threshold, perfect syndrome extraction), and it is not
a hardware adapter. The stabilizer tier is real but narrow: it executes Clifford
circuits only and refuses non-Clifford gates and noisy channels by name. A dense
statevector is exponential, and
:func:`apedp_dyn.quantum_compilation.statevector_sim.dimension_report` exists
precisely so nobody has to pretend otherwise.
"""
from __future__ import annotations

from apedp_dyn.vqc.backends import (
    Backend,
    RunResult,
    StatevectorBackend,
    bitstring,
    distribution,
    sample_bits,
)
from apedp_dyn.vqc.circuit import (
    SCHEMA_VERSION,
    Circuit,
    CircuitValidationError,
    Condition,
    Gate,
    Measure,
    Noise,
    Op,
    Reset,
)
from apedp_dyn.vqc.decoders import (
    Correction,
    Decoder,
    DecoderRefusalError,
    LookupDecoder,
    MatchingGraph,
    MinimumWeightDecoder,
    repetition_lookup_decoder,
    repetition_matching_graph,
    repetition_mwpm_decoder,
)
from apedp_dyn.vqc.discovery import (
    CandidateVerdict,
    CircuitCandidate,
    PromotionRefused,
    adjudicate_candidate,
    promote,
)
from apedp_dyn.vqc.equivalence import (
    CliffordEquivalenceReport,
    EquivalenceReport,
    NotACliffordCircuitError,
    NotAUnitaryCircuitError,
    adjudicate_clifford_equivalence,
    adjudicate_equivalence,
    global_phase_distance,
    unitary,
    unitary_distance,
)
from apedp_dyn.vqc.five_qubit import (
    FIVE_QUBIT_STABILIZERS,
    LOGICAL_X,
    LOGICAL_Z,
    FiveQubitRun,
    encoded_state,
    exact_failure_probability,
    five_qubit_code_circuit,
    five_qubit_encoder_circuit,
    five_qubit_logical_error_rate,
    syndrome_of,
)
from apedp_dyn.vqc.gates import GATE_SPECS, GateSpec, UnknownGateError, apply_gate, matrix_for
from apedp_dyn.vqc.logical import (
    DISTANCE,
    RepetitionCodeRun,
    RepetitionRun,
    analytic_failure_probability,
    bit_flip_code_circuit,
    decode,
    logical_error_rate,
    repetition_code_circuit,
    repetition_exact_failure_probability,
    repetition_logical_error_rate,
    single_error_syndrome,
)
from apedp_dyn.vqc.noise import (
    CHANNELS,
    DensityMatrix,
    amplitude_damping,
    bit_flip,
    density_matrix_after,
    dephasing,
    depolarizing,
    phase_flip,
    purity_of_bell_after_depolarizing,
)
from apedp_dyn.vqc.qasm import QasmUnsupportedError, from_qasm, to_qasm
from apedp_dyn.vqc.receipts import VERDICTS, Check, Receipt, adjudicate
from apedp_dyn.vqc.stabilizer import (
    CLIFFORD_GATES,
    StabilizerBackend,
    StabilizerState,
    StabilizerUnsupportedError,
    clifford_pauli_images,
    conjugate_pauli,
    ghz_circuit,
    ghz_generators,
    is_clifford_circuit,
    pauli_string,
)

__all__ = [
    "CHANNELS",
    "CLIFFORD_GATES",
    "DISTANCE",
    "FIVE_QUBIT_STABILIZERS",
    "GATE_SPECS",
    "LOGICAL_X",
    "LOGICAL_Z",
    "SCHEMA_VERSION",
    "VERDICTS",
    "Backend",
    "CandidateVerdict",
    "Check",
    "Circuit",
    "CircuitCandidate",
    "CircuitValidationError",
    "CliffordEquivalenceReport",
    "Condition",
    "Correction",
    "Decoder",
    "DecoderRefusalError",
    "DensityMatrix",
    "EquivalenceReport",
    "FiveQubitRun",
    "Gate",
    "GateSpec",
    "LookupDecoder",
    "MatchingGraph",
    "Measure",
    "MinimumWeightDecoder",
    "Noise",
    "NotACliffordCircuitError",
    "NotAUnitaryCircuitError",
    "Op",
    "PromotionRefused",
    "QasmUnsupportedError",
    "Receipt",
    "RepetitionCodeRun",
    "RepetitionRun",
    "Reset",
    "RunResult",
    "StabilizerBackend",
    "StabilizerState",
    "StabilizerUnsupportedError",
    "StatevectorBackend",
    "UnknownGateError",
    "adjudicate",
    "adjudicate_candidate",
    "adjudicate_clifford_equivalence",
    "adjudicate_equivalence",
    "amplitude_damping",
    "analytic_failure_probability",
    "apply_gate",
    "bit_flip",
    "bit_flip_code_circuit",
    "bitstring",
    "clifford_pauli_images",
    "conjugate_pauli",
    "decode",
    "density_matrix_after",
    "dephasing",
    "depolarizing",
    "distribution",
    "encoded_state",
    "exact_failure_probability",
    "five_qubit_code_circuit",
    "five_qubit_encoder_circuit",
    "five_qubit_logical_error_rate",
    "from_qasm",
    "ghz_circuit",
    "ghz_generators",
    "global_phase_distance",
    "is_clifford_circuit",
    "logical_error_rate",
    "matrix_for",
    "pauli_string",
    "phase_flip",
    "promote",
    "purity_of_bell_after_depolarizing",
    "repetition_code_circuit",
    "repetition_exact_failure_probability",
    "repetition_logical_error_rate",
    "repetition_lookup_decoder",
    "repetition_matching_graph",
    "repetition_mwpm_decoder",
    "sample_bits",
    "single_error_syndrome",
    "syndrome_of",
    "to_qasm",
    "unitary",
    "unitary_distance",
]

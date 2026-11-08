"""Unitary equivalence, up to exactly one global phase.

Status: experimental

A distribution check is not an equivalence proof: two circuits can agree on every
Z-basis measurement and differ by a relative phase, which is precisely the failure
mode the DQI phase errors came from. This module settles the stronger question
for small circuits by materialising the full unitary:

    U_c = e^{i phi} U_r   for one global phase phi, and no relative phase anywhere.

Materialising ``2^n x 2^n`` amplitudes only works while ``n`` is small (ten qubits
is 16 MB of complex doubles), and it only applies to *unitary* circuits: a
measurement, reset or noise operation makes a circuit a channel, not a unitary,
and the checker refuses it instead of pretending. That refusal is itself a useful
verdict -- it is the point where the DQI read-then-gate licence ends.

Above that bound there is a second, exact path for **Clifford** circuits, and it
never writes an amplitude down. A Clifford unitary is determined up to one global
phase by its conjugation action on the Pauli generators, so
:func:`adjudicate_clifford_equivalence` compares ``2n`` signed Pauli strings instead
of ``2^n x 2^n`` complex numbers -- the same verdict, at 100 qubits, in
milliseconds. The cost is a narrower domain (Clifford only) and one honest
limitation: conjugation cannot see a global phase, which is the phase that does not
matter. The state-level scope compares what the two circuits *prepare*, which is a
weaker claim (two different unitaries can prepare the same state), and the report
says which of the two was decided.

The distance is the Frobenius distance to the closest global-phase multiple,
normalised per amplitude:

    d(U_c, U_r) = min_phi ||U_c - e^{i phi} U_r||_F / sqrt(2^n),

which is attained in closed form at ``phi = arg(<U_r, U_c>)`` because
``||U_c - e^{i phi} U_r||_F^2 = 2 * 2^n - 2 Re(e^{i phi} <U_r, U_c>)``. So the phase
reported is the one that *minimises* the metric (the analytic value is
``sqrt(2 - 2|<U_r, U_c>| / 2^n)``; it is evaluated as the residual norm, which is the
same number without the cancellation), it is 0 if and only if the circuits differ by
one global phase, and it is at most ``sqrt(2)`` -- the value it takes for unitaries
with zero Hilbert-Schmidt overlap, which is as far apart as two of these can be.
It is invariant to a global phase by construction and sensitive to every relative
phase, so a candidate that "passes on basis states" cannot pass here.

The maximum single-entry deviation at that phase is reported separately as
``entry_deviation`` and is *not* minimised over ``phi``. Those are two different
objectives: minimising the max-entry (infinity-norm) deviation is a different phase
in general, and conflating the two -- reporting a max-entry number taken at the
Frobenius-optimal phase, as an earlier version of this module did -- understates the
min-max value. For `I` versus `Z` the Frobenius-optimal phase is arbitrary (the
overlap is zero), the max entry deviation there is 2.0, and the min-max over ``phi``
is ``sqrt(2)``; only the first of those is claimed here, under its own name.

"One global phase is free" is true for a standalone operation and false for a
controlled subroutine, where a global phase becomes a relative phase between control
branches. Both claims are offered: the default compares the matrices themselves
(``scope="standalone"``), and ``as_subroutine=True`` compares ``diag(I, U)`` instead
(``scope="controlled"``), which has no phase allowance. ``RX(2*pi)`` is ``I`` under the
first and is not ``I`` under the second. The Clifford tier decides only the standalone
claim -- its images cannot see a global phase at all -- so a controlled question about
a pair above the dense bound is ``UNRESOLVED``, not a guess.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from apedp_dyn.quantum_compilation.statevector_sim import StatevectorSimulator
from apedp_dyn.vqc import native
from apedp_dyn.vqc.circuit import Circuit, Gate, Measure, Noise, Reset
from apedp_dyn.vqc.gates import apply_gate
from apedp_dyn.vqc.receipts import Check
from apedp_dyn.vqc.stabilizer import (
    CLIFFORD_GATES,
    StabilizerState,
    clifford_pauli_images,
    pauli_string,
)

__all__ = [
    "CliffordEquivalenceReport",
    "EquivalenceReport",
    "NotACliffordCircuitError",
    "NotAUnitaryCircuitError",
    "adjudicate_clifford_equivalence",
    "adjudicate_equivalence",
    "controlled_lift",
    "global_phase_distance",
    "unitary",
    "unitary_distance",
]

#: Above this, the dense unitary is no longer a sensible object to materialise.
MAX_UNITARY_QUBITS = 10


class NotAUnitaryCircuitError(ValueError):
    """A circuit with measurement, reset or noise does not define a unitary."""


def unitary(circuit: Circuit, *, tolerance_qubits: int = MAX_UNITARY_QUBITS) -> np.ndarray:
    """The full `2^n x 2^n` unitary of a circuit, column by column.

    Fan-out is exact: column `j` is the image of basis state `j`, computed with the
    same gate application the backend uses, so this is the circuit's map and not a
    second implementation of it.
    """
    circuit.validate()
    if circuit.num_qubits > tolerance_qubits:
        raise ValueError(
            f"{circuit.num_qubits} qubits is a {2 ** circuit.num_qubits} x "
            f"{2 ** circuit.num_qubits} matrix; this checker stops at {tolerance_qubits}")
    for op in circuit.ops:
        if isinstance(op, (Measure, Reset, Noise)):
            raise NotAUnitaryCircuitError(
                f"the circuit contains a {type(op).__name__.lower()} operation "
                f"({getattr(op, 'kind', getattr(op, 'bit', ''))}); it is a channel, "
                f"not a unitary, and no unitary equivalence can be claimed for it")
        if op.condition:
            raise NotAUnitaryCircuitError(
                "a classically conditioned gate makes the circuit a channel")
    dim = 1 << circuit.num_qubits
    matrix: np.ndarray = np.zeros((dim, dim), dtype=np.complex128)
    for column in range(dim):
        sim = StatevectorSimulator(circuit.num_qubits)
        sim.set_basis_state(column)
        for op in circuit.ops:
            if not isinstance(op, Gate):  # pragma: no cover - rejected above
                raise NotAUnitaryCircuitError(f"unexpected operation {op!r}")
            apply_gate(sim, op.name, op.targets, op.params)
        matrix[:, column] = sim.state
    return matrix


class NotACliffordCircuitError(ValueError):
    """A circuit this tier cannot conjugate: not Clifford, or not a unitary."""


def _clifford_prefix(circuit: Circuit) -> None:
    """Refuse anything that is not an unconditioned Clifford gate, by name."""
    circuit.validate()
    for op in circuit.ops:
        if isinstance(op, (Measure, Reset)):
            raise NotACliffordCircuitError(
                f"the circuit contains a {type(op).__name__.lower()} on qubit "
                f"{op.qubit}; it prepares a mixture, which has no conjugation action")
        if isinstance(op, Noise):
            raise NotACliffordCircuitError(
                f"the circuit contains {op.kind!r} noise on qubit {op.qubit}; a "
                f"stochastic channel is not a Clifford unitary")
        if not isinstance(op, Gate):  # pragma: no cover - the IR has four op kinds
            raise NotACliffordCircuitError(f"unknown operation {op!r}")
        if op.condition:
            raise NotACliffordCircuitError(
                "a classically conditioned gate makes the circuit a channel")
        if op.params:
            raise NotACliffordCircuitError(
                f"{op.name!r} takes parameters; a parameterised rotation is not "
                f"Clifford")
        if op.name not in CLIFFORD_GATES:
            raise NotACliffordCircuitError(
                f"{op.name!r} is not Clifford; this tier conjugates "
                f"{sorted(CLIFFORD_GATES)}")


def _prepared_state(circuit: Circuit) -> StabilizerState:
    """The stabilizer state the circuit prepares from ``|0...0>``."""
    state = StabilizerState(circuit.num_qubits)
    for op in circuit.ops:
        assert isinstance(op, Gate)  # callers reject non-unitary ops first
        state.apply_gate(op.name, op.targets, op.params)
    return state


@dataclasses.dataclass(frozen=True)
class ImageTable:
    """``{'X_0': '+XX', ...}`` plus which implementation produced it and who checked."""

    entries: dict[str, str]
    #: ``"rust"`` or ``"python"``: the engine that computed the table.
    engine: str
    #: ``"python"`` below :data:`~apedp_dyn.vqc.native.PARITY_LIMIT` qubits, where
    #: the table is re-derived here and compared exactly; ``"suite"`` above it,
    #: where the conformance corpus is the correctness argument instead.
    verified_by: str


def _python_image_table(circuit: Circuit) -> dict[str, str]:
    """The reference table: every image conjugated in this module's own arithmetic."""
    table: dict[str, str] = {}
    for index, (labels, phase) in enumerate(clifford_pauli_images(circuit)):
        qubit, axis = divmod(index, 2)
        table[f"{'XZ'[axis]}_{qubit}"] = pauli_string(labels, phase)
    return table


def _image_table(circuit: Circuit, *, engine: str = "auto") -> ImageTable:
    """The conjugation action as text, computed Rust-first and refereed here.

    The *decision* never lives in the fast path: this function returns a table,
    and :func:`adjudicate_clifford_equivalence` compares tables and writes the
    verdict. Below :data:`~apedp_dyn.vqc.native.PARITY_LIMIT` qubits the table is
    additionally recomputed with :func:`_python_image_table` and a disagreement
    raises, so a small-``n`` verdict does not rest on the native engine at all.
    Above it the two tables are not both built -- that would cost exactly what the
    fast path saves -- and the report says ``verified_by: "suite"`` instead of
    implying a check that did not happen.
    """
    if engine not in ("auto", "python", "rust"):
        raise ValueError(f"engine must be 'auto', 'python' or 'rust', got {engine!r}")
    if engine != "python":
        rows = native.images(circuit)
        if rows is not None:
            table: dict[str, str] = {}
            for index, (labels, phase) in enumerate(rows):
                qubit, axis = divmod(index, 2)
                table[f"{'XZ'[axis]}_{qubit}"] = pauli_string(labels, phase)
            if circuit.num_qubits <= native.PARITY_LIMIT:
                reference = _python_image_table(circuit)
                if reference != table:
                    differing = [name for name in reference if reference[name] != table.get(name)]
                    raise native.NativeParityError(
                        f"the native image table disagrees with the reference at "
                        f"{differing[:4]} ({circuit.num_qubits} qubits); refusing to "
                        f"adjudicate on it")
                return ImageTable(entries=table, engine="rust", verified_by="python")
            return ImageTable(entries=table, engine="rust", verified_by="suite")
        if engine == "rust":
            raise NotACliffordCircuitError(
                f"the native engine could not produce images: "
                f"{native.last_failure() or native.unavailable_reason()}")
    return ImageTable(entries=_python_image_table(circuit), engine="python",
                      verified_by="python")


def _differences(reference: Mapping[str, str], candidate: Mapping[str, str], *,
                 limit: int = 4) -> tuple[list[dict[str, Any]], int]:
    """The first few entries where two tables of signed Pauli strings differ."""
    mismatched = 0
    rows: list[dict[str, Any]] = []
    for name, expected in reference.items():
        actual = candidate[name]
        if actual == expected:
            continue
        mismatched += 1
        if len(rows) < limit:
            rows.append({"basis": name, "reference": expected, "candidate": actual})
    return rows, mismatched


def unitary_distance(reference: np.ndarray, candidate: np.ndarray,
                     ) -> tuple[float, float, float]:
    """`(distance, phase, entry_deviation)` for `candidate` against `reference`.

    `distance` is ``min_phi ||candidate - e^{i phi} reference||_F / sqrt(D)`` with
    ``D`` the matrix dimension, in closed form: the minimizing phase is
    ``arg(<reference, candidate>)`` and the value is ``sqrt(2 - 2|overlap| / D)``.
    It is 0 exactly when the two matrices differ by one global phase, and at most
    ``sqrt(2)``.

    `entry_deviation` is ``max_ij |candidate - e^{i phi} reference|`` at that phase.
    It is reported as evidence and is *not* minimized over ``phi``; the min-max is a
    different objective and is deliberately not claimed (see the module docstring).
    """
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    dimension = reference.shape[0]
    overlap = np.vdot(reference, candidate)
    # With zero overlap every phase is equally close, so any phase is a minimizer and
    # `distance` is still the exact minimum (it does not depend on the choice).
    phase = 0.0 if abs(overlap) < 1e-15 else float(np.angle(overlap))
    rotated = np.exp(1j * phase) * reference
    difference = candidate - rotated
    # Analytically `sqrt(2 - 2|<U_r, U_c>| / D)`; evaluated as the residual norm
    # because the closed form subtracts two nearly equal numbers and turns an exact
    # 1e-16 agreement into 1e-8, which is above the tolerance that matters here.
    distance = float(np.linalg.norm(difference)) / math.sqrt(dimension)
    entry_deviation = float(np.max(np.abs(difference)))
    return distance, phase, entry_deviation


def global_phase_distance(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """`(distance, phase)`: `unitary_distance` without the entry deviation."""
    distance, phase, _entry = unitary_distance(reference, candidate)
    return distance, phase


def controlled_lift(matrix: np.ndarray) -> np.ndarray:
    """`diag(I, U)`: the same operation with one fresh control qubit in front.

    This is what makes a global phase *visible*. ``e^{i alpha} U`` and ``U`` are the
    same standalone operation, but ``diag(I, e^{i alpha} U) = (I ⊕ U) · diag(I,
    e^{i alpha} I)`` is not the same as ``diag(I, U)``: on the control-1 branch the
    phase is a relative phase, and a distribution test or a standalone comparison
    cannot see it. Comparing the lifts instead of the matrices removes the global-
    phase allowance, which is the correct discipline for a subroutine that will be
    used under control.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    dimension = matrix.shape[0]
    lifted = np.zeros((2 * dimension, 2 * dimension), dtype=np.complex128)
    lifted[:dimension, :dimension] = np.eye(dimension, dtype=np.complex128)
    lifted[dimension:, dimension:] = matrix
    _out: np.ndarray = lifted
    return _out

@dataclasses.dataclass(frozen=True)
class EquivalenceReport:
    """The referee's verdict on a candidate circuit, with the matrix evidence."""

    reference_sha256: str
    candidate_sha256: str
    qubits: int | None
    checks: tuple[Check, ...]
    resources: Mapping[str, Any]
    distance: float | None
    global_phase: float | None
    tolerance: float
    #: The semantic verdict: does the candidate compute the reference unitary?
    verdict: str
    entry_deviation: float | None = None
    #: The economic claim, kept apart from the semantic verdict: MET/UNMET/NOT_ASSERTED.
    claim: Check | None = None
    claim_verdict: str = "NOT_ASSERTED"
    #: ``"standalone"`` (one global phase free) or ``"controlled"`` (it is not).
    scope: str = "standalone"

    def failed(self) -> tuple[str, ...]:
        """The failed *semantic* checks; the resource claim is not one of them."""
        return tuple(check.name for check in self.checks if check.passed is False)

    @property
    def promotable(self) -> bool:
        """Semantically VALID *and* no unmet resource claim.

        An equivalent candidate that is more expensive keeps its VALID verdict -- it
        does compute the right operation -- and is simply not an improvement, which is
        a different statement and is reported as one.
        """
        return self.verdict == "VALID" and self.claim_verdict != "UNMET"

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_sha256": self.reference_sha256,
            "candidate_sha256": self.candidate_sha256,
            "qubits": self.qubits,
            "verdict": self.verdict,
            "promotable": self.promotable,
            "scope": self.scope,
            "decides": ("U_c = e^(i*phi) U_r for one global phase phi"
                        if self.scope == "standalone"
                        else "diag(I, U_c) = e^(i*phi) diag(I, U_r): a phase on the "
                             "subroutine is a relative phase under control"),
            "distance": self.distance,
            "metric": ("min_phi ||U_c - e^(i*phi) U_r||_F / sqrt(2**n), attained at "
                       "phi = arg(<U_r, U_c>); 0 iff one global phase apart, at "
                       "most sqrt(2)"),
            "global_phase": self.global_phase,
            "entry_deviation": self.entry_deviation,
            "entry_deviation_is_minimised": False,
            "tolerance": self.tolerance,
            "claim_verdict": self.claim_verdict,
            "claim": None if self.claim is None else self.claim.as_dict(),
            "checks": [check.as_dict() for check in self.checks],
            "resources": dict(self.resources),
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"


def _sha(circuit: Circuit) -> str:
    return hashlib.sha256(circuit.to_json().encode("utf-8")).hexdigest()


def adjudicate_equivalence(reference: Circuit, candidate: Circuit, *,
                           tolerance: float = 1e-9,
                           as_subroutine: bool = False,
                           claims: Mapping[str, int] | None = None,
                           provenance: Mapping[str, Any] | None = None) -> EquivalenceReport:
    """Decide whether `candidate` computes the same unitary as `reference`.

    `as_subroutine=True` decides the stricter claim: the two operations are compared
    as the control-1 blocks of a controlled circuit, i.e. `diag(I, U_c)` against
    `diag(I, U_r)` (see :func:`controlled_lift`). A global phase is then *not* free,
    because under control it is a relative phase between branches; `RX(2*pi)` is
    equivalent to `I` standalone and not equivalent to it as a subroutine.

    `claims` optionally names resource measures the candidate is supposed to
    improve (e.g. `{"two_qubit_gates": 2}`). A claim that is not met does not make the
    candidate *wrong* -- it can compute exactly the reference unitary and still save
    nothing -- so it is reported as `claim_verdict: "UNMET"` with the semantic verdict
    left `VALID`, and `promotable` is False. Two statements, two fields.
    """
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    checks: list[Check] = []
    resources = {"reference": reference.resources(), "candidate": candidate.resources()}
    if provenance:
        resources["provenance"] = dict(provenance)

    scope = "controlled" if as_subroutine else "standalone"
    matrices: dict[str, np.ndarray] = {}
    for role, circuit in (("reference", reference), ("candidate", candidate)):
        try:
            matrices[role] = unitary(circuit)
            if as_subroutine:
                matrices[role] = controlled_lift(matrices[role])
            checks.append(Check(
                name=f"{role} defines a unitary",
                requirement="no measurement, reset, noise or classical condition",
                measured={"qubits": circuit.num_qubits, "scope": scope,
                          **({"compared_as": "diag(I, U), one control qubit in front"}
                             if as_subroutine else {})},
                passed=True))
        except NotAUnitaryCircuitError as error:
            # A channel cannot be equivalent *as a unitary*: this is a refusal, not a
            # missing measurement.
            checks.append(Check(
                name=f"{role} defines a unitary",
                requirement="no measurement, reset, noise or classical condition",
                measured={"reason": str(error)}, passed=False))
        except ValueError as error:
            # Too large to materialise: undecided, which must not be reported as a
            # wrong candidate. `adjudicate_clifford_equivalence` exists for this case.
            checks.append(Check(
                name=f"{role} can be materialised at this size",
                requirement=f"at most {MAX_UNITARY_QUBITS} qubits, i.e. a 2**n x 2**n "
                            f"matrix; above that the stabilizer tier decides Clifford "
                            f"circuits",
                measured={"qubits": circuit.num_qubits, "reason": str(error)},
                passed=None))

    qubits = reference.num_qubits if reference.num_qubits == candidate.num_qubits else None
    checks.append(Check(
        name="both circuits act on the same number of qubits",
        requirement="the same Hilbert space, so a unitary comparison is meaningful",
        measured={"reference": reference.num_qubits, "candidate": candidate.num_qubits},
        passed=qubits is not None))

    distance: float | None = None
    phase: float | None = None
    entry: float | None = None
    if qubits is not None and "reference" in matrices and "candidate" in matrices:
        distance, phase, entry = unitary_distance(matrices["reference"],
                                                  matrices["candidate"])
        if as_subroutine:
            name = "the operations agree under control, up to one global phase"
            requirement = (f"min_phi ||diag(I, U_c) - e^(i*phi) diag(I, U_r)||_F / "
                           f"sqrt(2**(n+1)) <= {tolerance:g}: a phase on the "
                           f"subroutine alone is a relative phase between control "
                           f"branches and fails here")
        else:
            name = "unitaries agree up to one global phase"
            requirement = (f"min_phi ||U_c - e^(i*phi) U_r||_F / sqrt(2**n) <= "
                           f"{tolerance:g}: a relative phase error or a different "
                           f"operation fails here")
        checks.append(Check(
            name=name,
            requirement=requirement,
            measured={"distance": distance, "phi": phase,
                      "entry_deviation": entry, "scope": scope,
                      "relative_phase_sensitive": True,
                      "basis_state_probabilities_sufficient": False},
            passed=bool(distance <= tolerance)))

    claim: Check | None = None
    claim_verdict = "NOT_ASSERTED"
    if claims:
        claim = _claim_check(candidate, claims)
        claim_verdict = "MET" if claim.passed else "UNMET"

    if any(check.passed is False for check in checks):
        verdict = "INVALID"
    elif all(check.passed is True for check in checks):
        verdict = "VALID"
    else:  # pragma: no cover - every check above is decided
        verdict = "UNRESOLVED"

    return EquivalenceReport(
        reference_sha256=_sha(reference),
        candidate_sha256=_sha(candidate),
        qubits=qubits,
        checks=tuple(checks),
        resources=resources,
        distance=distance,
        global_phase=phase,
        tolerance=tolerance,
        verdict=verdict,
        entry_deviation=entry,
        claim=claim,
        claim_verdict=claim_verdict,
        scope=scope,
    )


@dataclasses.dataclass(frozen=True)
class CliffordEquivalenceReport:
    """The referee's verdict above the dense bound, with the images as evidence."""

    reference_sha256: str
    candidate_sha256: str
    qubits: int | None
    #: ``"unitary"`` (up to one global phase) or ``"state"`` (what each prepares).
    scope: str
    checks: tuple[Check, ...]
    resources: Mapping[str, Any]
    mismatches: tuple[Mapping[str, Any], ...]
    mismatched: int
    compared: int
    verdict: str
    #: The economic claim, kept apart from the semantic verdict: MET/UNMET/NOT_ASSERTED.
    claim: Check | None = None
    claim_verdict: str = "NOT_ASSERTED"
    #: ``"rust"`` or ``"python"``: who conjugated the Paulis.
    image_engine: str = "python"
    #: ``"python"`` (re-derived and compared here) or ``"suite"`` (conformance
    #: corpus only, above the parity limit). Stated so a reader can see the level
    #: of checking rather than assume the strongest one.
    image_verified_by: str = "python"

    def failed(self) -> tuple[str, ...]:
        """The failed *semantic* checks; the resource claim is not one of them."""
        return tuple(check.name for check in self.checks if check.passed is False)

    @property
    def promotable(self) -> bool:
        """Semantically VALID *and* no unmet resource claim (see the dense report)."""
        return self.verdict == "VALID" and self.claim_verdict != "UNMET"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": "stabilizer",
            "scope": self.scope,
            "decides": ("the unitary, up to one global phase"
                        if self.scope == "unitary"
                        else "the prepared state, up to one global phase"),
            "reference_sha256": self.reference_sha256,
            "candidate_sha256": self.candidate_sha256,
            "qubits": self.qubits,
            "verdict": self.verdict,
            "promotable": self.promotable,
            "claim_verdict": self.claim_verdict,
            "claim": None if self.claim is None else self.claim.as_dict(),
            "image_engine": self.image_engine,
            "image_verified_by": self.image_verified_by,
            "compared": self.compared,
            "mismatched": self.mismatched,
            "mismatches": [dict(row) for row in self.mismatches],
            "checks": [check.as_dict() for check in self.checks],
            "resources": dict(self.resources),
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"


def _claim_check(candidate: Circuit, claims: Mapping[str, int]) -> Check:
    failed_claims = {}
    for measure, claimed in claims.items():
        actual = candidate.resources().get(measure)
        if actual is None or actual > claimed:
            failed_claims[measure] = {"claimed": claimed, "actual": actual}
    return Check(
        name="the candidate meets the resource claim it is offered under",
        requirement="every claimed measure is at most the claimed value",
        measured={"claims": dict(claims), "unmet": failed_claims},
        passed=not failed_claims)


def adjudicate_clifford_equivalence(reference: Circuit, candidate: Circuit, *,
                                    on: str = "unitary",
                                    claims: Mapping[str, int] | None = None,
                                    provenance: Mapping[str, Any] | None = None,
                                    engine: str = "auto",
                                    ) -> CliffordEquivalenceReport:
    """Decide Clifford equivalence exactly, at any size, without amplitudes.

    `on="unitary"` (the default) decides ``U_c = e^{i phi} U_r``: two Cliffords with
    the same conjugation action on ``X_q`` and ``Z_q`` differ by a scalar, because
    their quotient commutes with every Pauli. `on="state"` decides only
    ``U_c|0...0> = e^{i phi} U_r|0...0>``, by comparing stabilizer groups; that is a
    weaker claim, and the report says so in `scope` and `decides`.

    Nothing is approximated: the images are signed Pauli strings over integers mod
    four, so a mismatch is a mismatch and there is no tolerance to tune.

    `engine` selects who computes the image table: ``"auto"`` (Rust-first, with the
    Python re-derivation described on :func:`_image_table`), ``"python"`` (the
    reference path only) or ``"rust"`` (refuse rather than downgrade). The verdict
    is this function's in every case, and the report records both the engine and
    the level of checking the table got.
    """
    if on not in ("unitary", "state"):
        raise ValueError(f"on must be 'unitary' or 'state', got {on!r}")
    checks: list[Check] = []
    resources = {"reference": reference.resources(), "candidate": candidate.resources()}
    if provenance:
        resources["provenance"] = dict(provenance)

    usable: dict[str, Circuit] = {}
    for role, circuit in (("reference", reference), ("candidate", candidate)):
        try:
            _clifford_prefix(circuit)
            usable[role] = circuit
            checks.append(Check(
                name=f"{role} is a Clifford unitary circuit",
                requirement=f"unconditioned gates from {sorted(CLIFFORD_GATES)}, no "
                            f"measurement, reset or noise",
                measured={"qubits": circuit.num_qubits, "gates": len(circuit.ops)},
                passed=True))
        except NotACliffordCircuitError as error:
            checks.append(Check(
                name=f"{role} is a Clifford unitary circuit",
                requirement=f"unconditioned gates from {sorted(CLIFFORD_GATES)}, no "
                            f"measurement, reset or noise",
                measured={"reason": str(error)},
                passed=False))

    qubits = reference.num_qubits if reference.num_qubits == candidate.num_qubits else None
    checks.append(Check(
        name="both circuits act on the same number of qubits",
        requirement="the same Hilbert space, so a comparison is meaningful",
        measured={"reference": reference.num_qubits, "candidate": candidate.num_qubits},
        passed=qubits is not None))

    mismatches: list[Mapping[str, Any]] = []
    mismatched = 0
    compared = 0
    reference_images: dict[str, str] | None = None
    candidate_images: dict[str, str] | None = None
    image_engine = "python"
    image_verified_by = "python"
    if qubits is not None and len(usable) == 2:
        reference_table = _image_table(reference, engine=engine)
        candidate_table = _image_table(candidate, engine=engine)
        reference_images = reference_table.entries
        candidate_images = candidate_table.entries
        image_engine = reference_table.engine
        # The weaker of the two levels of checking, which is the honest one to
        # report: a table that was re-derived in Python on one side and only
        # covered by the conformance corpus on the other is "suite".
        image_verified_by = ("python" if {reference_table.verified_by,
                                          candidate_table.verified_by} == {"python"}
                             else "suite")
        compared = len(reference_images)
        mismatches, mismatched = _differences(reference_images, candidate_images)

    if reference_images is not None and candidate_images is not None:
        if on == "unitary":
            checks.append(Check(
                name="the conjugation action on every Pauli generator agrees",
                requirement=(f"all {compared} images of X_q and Z_q are the same signed "
                             f"Pauli string; a relative phase or a different operation "
                             f"changes one, a global phase changes none"),
                measured={"compared": compared, "mismatched": mismatched,
                          "mismatches": mismatches,
                          "global_phase_visible": False},
                passed=mismatched == 0))
        else:
            reference_state = _prepared_state(reference)
            candidate_state = _prepared_state(candidate)
            mismatched = 0
            mismatches = []
            for index, (labels, phase) in enumerate(zip(candidate_state.generators,
                                                        candidate_state.phases, strict=True)):
                expected = reference_state.phase_of(labels)
                if expected == phase:
                    continue
                mismatched += 1
                if len(mismatches) < 4:
                    mismatches.append({
                        "generator": index,
                        "pauli": "".join("IXYZ"[label] for label in labels),
                        "reference": ("absent from the reference group" if expected is None
                                      else {0: "+1", 2: "-1"}[expected]),
                        "candidate": {0: "+1", 2: "-1"}[phase],
                    })
            compared = candidate_state.num_qubits
            checks.append(Check(
                name="the prepared states have the same stabilizer group",
                requirement=(f"every signed generator of the candidate is in the "
                             f"reference's group, which contains 2**{compared} elements "
                             f"for both, so containment is equality"),
                measured={"compared": compared, "mismatched": mismatched,
                          "mismatches": mismatches,
                          "decides": "prepared state, not the unitary"},
                passed=mismatched == 0))

    claim: Check | None = None
    claim_verdict = "NOT_ASSERTED"
    if claims:
        claim = _claim_check(candidate, claims)
        claim_verdict = "MET" if claim.passed else "UNMET"

    if any(check.passed is False for check in checks):
        verdict = "INVALID"
    elif all(check.passed is True for check in checks):
        verdict = "VALID"
    else:  # pragma: no cover - every check above is decided
        verdict = "UNRESOLVED"

    return CliffordEquivalenceReport(
        reference_sha256=_sha(reference),
        candidate_sha256=_sha(candidate),
        qubits=qubits,
        scope=on,
        checks=tuple(checks),
        resources=resources,
        mismatches=tuple(mismatches),
        mismatched=mismatched,
        compared=compared,
        verdict=verdict,
        claim=claim,
        claim_verdict=claim_verdict,
        image_engine=image_engine,
        image_verified_by=image_verified_by,
    )

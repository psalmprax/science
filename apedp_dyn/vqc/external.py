"""Foreign circuits, our referee: a task-matched comparison harness that can abstain.

Status: experimental

The review before this module was blunt: circuit synthesis, stabilizer simulation and
equivalence checking are established, and Qiskit/Stim/MQT QCEC already do parts of
this. The answer it asked for is not a stronger claim, it is a *matched measurement*
with a rule for what happens when a competitor cannot be run. Two rules, in fact:

1. **The comparison is per task, not per system.** \"Faster\" across different
   workloads is not a comparison. Each row here names the task (synthesis from a
   stabilizer specification), the target (a frozen specification by hash) and the
   metric (operations, two-qubit gates, depth, wall clock).
2. **A tool that did not run is not a tool that lost.** A competitor that is absent
   or not wired produces `UNAVAILABLE` or `INSTALLED_NOT_WIRED` with the reason, and
   the record refuses to count that as a win. An abstention is information; a
   fabricated number is contamination.

What a foreign circuit gets, before any number is quoted, is the referee. A
:class:`Submission` is another tool's output together with that tool's own count of
its operations, and :func:`adjudicate_submission` puts it through the same
specification referee and the same four-field admission policy a local candidate
goes through. So the comparison has three possible outcomes per row, and the
interesting one is the third:

* the foreign circuit is `VALID` and its claim is `MET` -- a real comparison;
* it is `VALID` but `UNMET` -- the tool's own count and our measurement of the
  translated circuit disagree, which is a finding about the *interchange*, not about
  the tool's algorithm;
* it is `INVALID` -- either the tool is wrong or the translation is, and either way
  the row says so instead of quoting its gate count as if it were an achievement.

The asymmetry the review pointed at is real and is recorded rather than smoothed
over: Qiskit's synthesiser handles *any* stabilizer state, while APEDP's construction
grammar handles the targets it has families for and refuses the rest by name. A
comparison on GHZ therefore measures a general algorithm against a grammar that
already knows the answer, and the record says exactly that. The fair complement is a
random stabilizer state, where APEDP abstains and Qiskit does not, and both rows are
in the record.

Stim, and the sampling task
---------------------------
Stim is wired for the task it actually overlaps with -- execution, not synthesis.
Both engines are handed the same specification and the same shot budget, and each
samples Z-basis outcomes from it: this package by cloning its tableau and measuring
qubit by qubit, Stim by building a tableau from the same signed stabilizers and
compiling a measurement sampler. Every sample set is then decided against the **exact**
distribution of the specification, and the two samplers are compared with each other
under a null in which both drew from the same exact distribution. So three things are
checked rather than one: each sampler against the arithmetic, the two samplers against
each other, and the wall clock on the same shots.

The exact referee has two shapes, because the full distribution is enumerable only up
to 12 qubits. At and below that width the referee is the entire distribution. Above it,
:mod:`apedp_dyn.vqc.marginals` derives the exact distribution of chosen subsets of
wires -- by Gaussian elimination over GF(2) on the specification, so the referee is
neither sampler -- and each subset is decided separately against a band corrected for
the number of subsets tested. That removes the width ceiling without pretending to
more than it decides: a projection can be right while the state is wrong, and the
record says so with the case where it happens.

QCEC, and the equivalence task
------------------------------
MQT QCEC overlaps with the equivalence referee, so it is given the task it overlaps
with: :mod:`apedp_dyn.vqc.equivalence_corpus` builds pairs whose relation is fixed by
the construction that built them and on which the two referees *disagree*, and
:func:`qcec_equivalence` hands each pair over and records what came back. Two things
about how it is run matter. It is given its own method selection, its own default
numerical tolerance and no time limit, because a threshold tuned to agree with this
package would not be a second opinion. And it is asked each boundary pair more than
once: QCEC's criterion near its own threshold is not a function of the input, so
`verdict` is ``"unstable"`` where identical calls disagreed and the record carries the
criteria themselves. It decides the standalone unitary question only -- it has no
controlled or prepared-state mode -- so those verdicts are recorded beside its answer
rather than compared with it.

Wiring a tool is a decision with a correctness argument attached (its output must
survive the referee), not a line in a requirements file.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata
import importlib.util
import itertools
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from apedp_dyn.vqc.circuit import Circuit, Gate
from apedp_dyn.vqc.equivalence import NotAUnitaryCircuitError
from apedp_dyn.vqc.qasm import from_qasm, to_qasm
from apedp_dyn.vqc.synthesis import (
    I,
    SpecificationCandidate,
    TargetSpecification,
    Z,
    adjudicate_specification_candidate,
)

__all__ = [
    "ROUTING_BASIS",
    "TOOLS",
    "ExternalRow",
    "ForeignEquivalence",
    "Routing",
    "Sampling",
    "SamplingUnsupported",
    "Submission",
    "ToolStatus",
    "ToolUnavailable",
    "adjudicate_sampling",
    "adjudicate_submission",
    "apedp_sampling",
    "probe",
    "qcec_equivalence",
    "qiskit_circuit",
    "qiskit_routing",
    "qiskit_synthesis",
    "relabelled_target",
    "sampling_pair",
    "stim_sampling",
    "tool_statuses",
]


class ToolUnavailable(RuntimeError):
    """A competitor could not be run, with the reason a record can print."""


#: The tools the review named, what overlaps with this package, and whether the
#: comparison is wired. `wired` is a claim with an argument behind it: the adapter
#: exists and its output survives the referee. Everything else is an abstention.
TOOLS: dict[str, dict[str, Any]] = {
    "qiskit": {
        "package": "qiskit",
        "task": "synthesis of a circuit from a stabilizer specification",
        "metrics": ("operations, two-qubit gates, depth, wall clock, and the APEDP "
                    "referee's verdict on the synthesised circuit"),
        "wired": True,
        "note": ("same input as this package: signed Pauli stabilizer strings. "
                 "Qiskit's synthesiser is general over stabilizer states; the local "
                 "construction grammar is not, which is why the random-state row "
                 "matters as much as the GHZ rows. Its transpiler is also used for "
                 "the routing task, on a recorded device coupling map, because "
                 "routing Qiskit's circuit with this package's router would measure "
                 "this package's router"),
    },
    "stim": {
        "package": "stim",
        "task": ("Z-basis sampling from a stabilizer specification, on the same "
                 "targets, the same shot budget and the same seeds policy"),
        "metrics": ("wall clock for the sample batch, samples per second, and the "
                    "total-variation distance of the sample set from the exact "
                    "distribution, plus the two samplers' distance from each other"
                    "under a null where both are correct"),
        "wired": True,
        "note": ("wired for execution, not synthesis: each engine receives the same "
                 "signed stabilizer strings, samples the same number of shots, and "
                 "both sample sets are decided against the exact distribution. The "
                 "workload is 12-qubit wide because the referee enumerates the "
                 "distribution exactly; the throughput ratio is an implementation "
                 "comparison (Python per-shot tableau clone against Stim's compiled "
                 "C++ sampler), not a ceiling for either engine, and Stim's published "
                 "20,000-qubit workload is not addressed"),
    },
    "mqt.qcec": {
        "package": "mqt.qcec",
        "task": ("circuit equivalence checking on a corpus built so the two referees "
                 "disagree, with every divergence attributed to a threshold, a scope "
                 "or an abstention"),
        "metrics": ("verdict and wall clock on the same pairs this package's "
                    "equivalence referee decides, the criterion it names (including "
                    "equivalent_up_to_global_phase and no_information), and the "
                    "stability of that criterion under repeats of the identical call"),
        "wired": True,
        "note": ("wired on the pairs where a comparison can say something. QCEC "
                 "decides with decision diagrams, ZX rewriting and simulation-based "
                 "falsification, and is given its own default numerical tolerance "
                 "rather than one tuned to agree with this package. It decides the "
                 "standalone unitary question only: it has no controlled or "
                 "state-preparation mode, and near its threshold identical calls "
                 "can return different criteria, which the record reports as a "
                 "distribution instead of a verdict"),
    },
}


@dataclasses.dataclass(frozen=True)
class ToolStatus:
    """What happened when this package tried to compare against one tool."""

    name: str
    package: str
    task: str
    installed: bool
    version: str | None
    wired: bool
    #: `compared`, `installed_not_wired` or `unavailable`. Only the first can carry
    #: numbers; the other two carry a reason and are never counted as a loss.
    status: str
    reason: str
    note: str = ""

    @property
    def comparative(self) -> bool:
        return self.status == "compared"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "package": self.package,
            "task": self.task,
            "installed": self.installed,
            "version": self.version,
            "wired": self.wired,
            "status": self.status,
            "reason": self.reason,
            "note": self.note,
            "counts_as_a_comparison": self.comparative,
        }


def _installed(package: str) -> tuple[bool, str | None, str]:
    """`(installed, version, reason)`, without importing the package to find out."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError) as error:  # a broken parent package, not absence
        return False, None, f"not importable: {type(error).__name__}: {error}"
    if spec is None:
        return False, None, f"{package!r} is not installed in this environment"
    try:
        return True, importlib.metadata.version(package), "installed"
    except importlib.metadata.PackageNotFoundError:
        # A module on the path that is not a distribution: present, unversioned.
        return True, None, "installed, but no distribution metadata names its version"


def probe(name: str) -> ToolStatus:
    """Try to compare against `name`, and say exactly why not when that fails."""
    if name not in TOOLS:
        raise ValueError(f"unknown tool {name!r}; known: {sorted(TOOLS)}")
    entry = TOOLS[name]
    installed, version, reason = _installed(entry["package"])
    if not installed:
        return ToolStatus(name, entry["package"], entry["task"], False, None,
                          entry["wired"], "unavailable", reason, entry["note"])
    if not entry["wired"]:
        return ToolStatus(name, entry["package"], entry["task"], True, version,
                          False, "installed_not_wired",
                          f"{entry['package']} is installed but this package has no "
                          f"adapter for it: wiring one means its output survives the "
                          f"referee, which has not been written", entry["note"])
    return ToolStatus(name, entry["package"], entry["task"], True, version, True,
                      "compared", "wired and installed", entry["note"])


def tool_statuses() -> tuple[ToolStatus, ...]:
    return tuple(probe(name) for name in TOOLS)


# ------------------------------------------------------------------- submissions
@dataclasses.dataclass(frozen=True)
class Submission:
    """Another tool's circuit, its own count of it, and how it was obtained.

    `claims` is what the *tool* says about the circuit (Qiskit's `count_ops` total, for
    instance), in this package's vocabulary. It is checked like any other claim: if the
    translated circuit does not match it, the row is `UNMET`, which is a statement
    about the interchange rather than about the tool's algorithm.
    """

    tool: str
    variant: str
    circuit: Circuit
    claims: Mapping[str, int] = dataclasses.field(default_factory=dict)
    version: str | None = None
    source: str = "qasm"
    notes: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "variant": self.variant,
            "version": self.version,
            "source": self.source,
            "claims": dict(self.claims),
            "circuit_sha256": self.circuit.sha256(),
            "notes": dict(self.notes),
        }


@dataclasses.dataclass(frozen=True)
class ExternalRow:
    """One foreign circuit, decided by the local referee and measured the same way."""

    submission: Submission
    target_sha256: str
    qubits: int
    operations: int
    two_qubit_gates: int
    depth: int
    tier: str
    verdict: str
    claim_verdict: str
    evidence: str
    promotable: bool
    checks_failed: tuple[str, ...]
    seconds: float

    @property
    def verified(self) -> bool:
        return self.verdict == "VALID"

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.submission.tool,
            "variant": self.submission.variant,
            "version": self.submission.version,
            "target_sha256": self.target_sha256,
            "qubits": self.qubits,
            "metrics": {
                "operations": self.operations,
                "two_qubit_gates": self.two_qubit_gates,
                "depth": self.depth,
                "seconds": self.seconds,
            },
            "verification": {
                "verdict": self.verdict,
                "tier": self.tier,
                "checks_failed": list(self.checks_failed),
                "verified_by": "apedp_dyn.vqc.synthesis (specification referee)",
            },
            "resource_claim": {
                "verdict": self.claim_verdict,
                "claims": dict(self.submission.claims),
            },
            "evidence_sufficiency": self.evidence,
            "promotable": self.promotable,
            "circuit_sha256": self.submission.circuit.sha256(),
            "note": dict(self.submission.notes),
        }


def adjudicate_submission(submission: Submission, target: TargetSpecification, *,
                          seconds: float = 0.0) -> ExternalRow:
    """Decide a foreign circuit against `target`, using this package's own referee."""
    candidate = SpecificationCandidate(
        candidate_id=f"external-{submission.tool}-{submission.variant}-"
                     f"{target.name}-{target.num_qubits}q",
        circuit=submission.circuit,
        specification=target,
        source=submission.tool,
        claims=dict(submission.claims),
        metadata={"tool": submission.tool, "variant": submission.variant,
                  "version": str(submission.version)},
    )
    verdict = adjudicate_specification_candidate(candidate)
    resources = submission.circuit.resources()
    return ExternalRow(
        submission=submission,
        target_sha256=target.sha256(),
        qubits=submission.circuit.num_qubits,
        operations=int(resources["operations"]),
        two_qubit_gates=int(resources["two_qubit_gates"]),
        depth=int(resources["depth"]),
        tier=verdict.tier,
        verdict=verdict.verdict,
        claim_verdict=verdict.claim_verdict,
        evidence=verdict.evidence,
        promotable=verdict.promotable,
        checks_failed=tuple(check.name for check in verdict.checks
                            if check.passed is False),
        seconds=float(seconds),
    )


# ---------------------------------------------------------------- qiskit adapter
def _import_qiskit() -> tuple[Any, Any]:
    """The two qiskit modules this adapter uses, or a reason it cannot run."""
    try:
        synthesise = importlib.import_module("qiskit.synthesis")
        qasm2 = importlib.import_module("qiskit.qasm2")
    except ImportError as error:
        raise ToolUnavailable(
            f"qiskit is not importable ({type(error).__name__}: {error}); install it "
            f"into this environment to run the comparison") from error
    return synthesise, qasm2


def _qiskit_strings(target: TargetSpecification) -> list[str]:
    """The target's stabilizers in Qiskit's qubit order, which is not ours.

    ``pauli_string`` is big-endian: character ``i`` is qubit ``i``. Qiskit's
    ``synth_circuit_from_stabilizers`` (a port of Stim's) is little-endian: the first
    character is the *highest*-indexed qubit. Handing the strings over unconverted does
    not fail loudly -- it prepares the mirror image of the requested state, which is
    the correct state for any palindromic target and wrong for every other one. A GHZ
    and a ring are both palindromic, so this was invisible until a random stabilizer
    state was placed in the corpus; `test_a_non_palindromic_target_exposes_the_string_
    order` pins it.

    The reversal is a claim about the other tool's convention, so it is not trusted on
    its own: the circuit that comes back is decided by our referee in the same qubit
    order our specification uses, and a wrong convention would come back `INVALID`.
    """
    strings = []
    for text in target.stabilizer_strings():
        sign, body = (text[0], text[1:]) if text[0] in "+-" else ("", text)
        strings.append(sign + body[::-1])
    return strings


def qiskit_synthesis(target: TargetSpecification, *,
                     variant: str = "general") -> Submission:
    """Have Qiskit synthesize `target` from the *same* stabilizer strings we hold.

    `variant="general"` is `synth_circuit_from_stabilizers` (Gaussian elimination with
    Clifford gates, the algorithm by way of Stim). Its output is exported as QASM 2
    and re-imported through :func:`apedp_dyn.vqc.qasm.from_qasm`, so the circuit the
    referee sees came through a text interchange rather than a shared data structure.
    """
    synthesise, qasm2 = _import_qiskit()
    if variant != "general":
        raise ToolUnavailable(
            f"the {variant!r} qiskit variant is not wired: "
            f"`synth_stabilizer_depth_lnn` minimizes two-qubit depth for linear "
            f"nearest-neighbour connectivity but emits `u(theta, phi, lambda)` gates, "
            f"which are outside this package's gate table, so comparing gate counts "
            f"would mean re-synthesising the competitor's output into our vocabulary "
            f"and then reporting the result as theirs")
    circuit = synthesise.synth_circuit_from_stabilizers(_qiskit_strings(target))
    text = qasm2.dumps(circuit)
    imported = from_qasm(text)
    counts = circuit.count_ops()
    return Submission(
        tool="qiskit",
        variant="synth_circuit_from_stabilizers",
        circuit=imported,
        # The tool's own size for the circuit, in our vocabulary: if the QASM
        # translation dropped or added an operation, the claim comes back UNMET.
        claims={"operations": int(sum(counts.values()))},
        version=importlib.metadata.version("qiskit"),
        source="qiskit -> qasm2 -> apedp_dyn.vqc.qasm",
        notes={
            "algorithm": "Gaussian elimination with Clifford gates, after Stim",
            "spectrum": "any stabilizer state, including ones with no local family",
            "constructed_by": "the tool's own synthesizer; APEDP only decided it",
            "string_order": (
                "the stabilizer strings are reversed before the call: Qiskit reads the "
                "first character as the highest-indexed qubit, this package reads it "
                "as qubit 0. Unconverted, both palindromic targets (GHZ, ring) still "
                "verify and every non-palindromic one comes back INVALID as the mirror "
                "image"),
        },
    )


def qiskit_unavailable_reason() -> str:
    """One line a record can print: why qiskit cannot be compared against."""
    status = probe("qiskit")
    return status.reason


# ---------------------------------------------------------------------- routing
#: The gate set Qiskit is asked to transpile into. Every name is in this package's
#: gate table, so the routed circuit re-imports through QASM 2 without a translation
#: step that could silently change the depth. `swap` is included because routing
#: inserts it; leaving it out would force Qiskit to decompose it into three CX and
#: make the counts incomparable with this package's one-SWAP accounting.
ROUTING_BASIS = ("rz", "ry", "rx", "h", "x", "y", "z", "s", "sdg", "t", "tdg",
                 "cx", "cz", "swap")


@dataclasses.dataclass(frozen=True)
class Routing:
    """A circuit routed onto a device by another tool, re-imported and measured here.

    `final_layout[i]` is the physical wire holding logical qubit ``i`` after routing,
    which is how the foreign circuit's target can be relabelled to the state the
    device actually prepares. The circuit itself is kept so a referee can decide it;
    it is not serialised into the record, only its hash.
    """

    tool: str
    variant: str
    version: str | None
    device: str
    device_version: str | None
    qubits: int
    logical_qubits: int
    target_sha256: str
    logical_depth: int
    routed_depth: int
    reported_depth: int
    routing_swaps: int
    two_qubit_gates: int
    operations: int
    final_layout: tuple[int, ...]
    seconds: float
    optimization_level: int
    seed: int | None
    basis_gates: tuple[str, ...]
    circuit: Circuit

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.tool,
            "variant": self.variant,
            "version": self.version,
            "device": self.device,
            "device_version": self.device_version,
            "qubits": self.qubits,
            "logical_qubits": self.logical_qubits,
            "target_sha256": self.target_sha256,
            "logical_depth": self.logical_depth,
            "routed_depth": self.routed_depth,
            "qiskit_reported_depth": self.reported_depth,
            "routing_swaps": self.routing_swaps,
            "two_qubit_gates": self.two_qubit_gates,
            "operations": self.operations,
            "final_layout": list(self.final_layout),
            "seconds": self.seconds,
            "optimization_level": self.optimization_level,
            "seed_transpiler": self.seed,
            "basis_gates": list(self.basis_gates),
            "circuit_sha256": self.circuit.sha256(),
            "note": {
                "router": "Qiskit SabreSwap inside transpile()",
                "layout": ("initial_layout = 0..n-1; the returned layout is kept, so "
                           "the target is relabelled by final_layout for the referee"),
                "restore": ("Qiskit does not restore the layout; this package's "
                            "router does, and its total is reported separately"),
            },
        }


def relabelled_target(target: TargetSpecification, position: Sequence[int],
                      num_qubits: int) -> TargetSpecification:
    """The same state with logical wire ``i`` moved to physical wire ``position[i]``.

    Qiskit keeps its final layout as metadata, so the circuit it returns prepares the
    requested state *up to that permutation*. Relabelling the specification is how the
    foreign circuit can still be decided by the same referee: the state is the same
    state, written on the wires the device will actually hold it on. The permutation
    must be injective and of the target's width -- a layout with a repeated wire is a
    bug in the tool, not a relabelling, and it is refused rather than "fixed".
    """
    if len(position) != target.num_qubits:
        raise ValueError(
            f"the layout has {len(position)} entries for {target.num_qubits} logical qubits")
    if len(set(position)) != target.num_qubits or min(position) < 0 \
            or max(position) >= num_qubits:
        raise ValueError(
            f"the layout is not an injective map into {num_qubits} wires: {position}")
    used = set(position)
    stabilizers = []
    for labels, phase in target.stabilizers:
        out = [I] * num_qubits
        for logical, label in enumerate(labels):
            out[position[logical]] = label
        stabilizers.append((tuple(out), phase))
    stabilizers += [(tuple(Z if index == qubit else I for index in range(num_qubits)), 0)
                    for qubit in range(num_qubits) if qubit not in used]
    return TargetSpecification(
        name=f"{target.name}+relabelled", num_qubits=num_qubits,
        stabilizers=tuple(stabilizers),
        decides=(f"{target.decides}, relabelled by the tool's final layout"))


def qiskit_routing(logical: Circuit, *, target: TargetSpecification, num_qubits: int,
                   edges: Sequence[tuple[int, int]], device: str,
                   device_version: str | None = None, seed: int | None = 1234,
                   optimization_level: int = 1) -> Routing:
    """Route `logical` onto a device coupling map with Qiskit's own transpiler.

    This is the row the topology task did not have: Qiskit given a *real* coupling map
    and its own router (SabreSwap inside ``transpile``), rather than this package's
    greedy router applied to Qiskit's circuit. The routed circuit is translated to
    QASM 2 and re-imported, so the depth quoted is measured by the same resource
    accounting as every other row; Qiskit's own ``depth()`` is reported beside it as
    a cross-check.

    No restore ladder is appended: Qiskit keeps its final layout as metadata, and a
    device executes the circuit that way. This package's router restores logical
    qubits to their original wires, so its ``depth_before_restore`` is the comparable
    number and its total is reported separately.
    """
    if target.num_qubits != logical.num_qubits:
        raise ValueError(
            f"the circuit has {logical.num_qubits} qubits, the target "
            f"{target.num_qubits}")
    qiskit = importlib.import_module("qiskit")
    _, qasm2 = _import_qiskit()
    # Both directions per undirected edge: a device edge is a two-qubit interaction
    # either way round, and a one-directional map would silently constrain SABRE to a
    # directed graph the hardware does not have.
    coupling = importlib.import_module("qiskit.transpiler").CouplingMap(
        [direction for first, second in edges
         for direction in ((first, second), (second, first))])
    if num_qubits < logical.num_qubits:
        raise ToolUnavailable(
            f"the device has {num_qubits} qubits but the circuit needs {logical.num_qubits}")
    start = time.perf_counter()
    try:
        routed = qiskit.transpile(
            qasm2.loads(to_qasm(logical)),
            coupling_map=coupling,
            basis_gates=list(ROUTING_BASIS),
            initial_layout=list(range(logical.num_qubits)),
            optimization_level=optimization_level,
            seed_transpiler=seed,
        )
    except Exception as error:
        raise ToolUnavailable(
            f"qiskit could not transpile this circuit onto {device}: "
            f"{type(error).__name__}: {error}") from error
    seconds = time.perf_counter() - start
    imported = from_qasm(qasm2.dumps(routed))
    counts = routed.count_ops()
    swaps = int(counts.get("swap", 0))
    two_qubit = int(sum(counts.get(name, 0)
                        for name in ("cx", "cz", "swap", "ecr")))
    position = tuple(result for result in routed.layout.final_index_layout()
                     if result is not None)[:logical.num_qubits]
    if len(position) != logical.num_qubits:
        raise ToolUnavailable(
            f"qiskit returned a layout for {len(position)} of {logical.num_qubits} "
            f"logical qubits; the routed target cannot be relabelled reliably")
    return Routing(
        tool="qiskit",
        variant="transpile.sabre",
        version=_package_version("qiskit"),
        device=device,
        device_version=device_version,
        qubits=routed.num_qubits,
        logical_qubits=logical.num_qubits,
        target_sha256=target.sha256(),
        logical_depth=int(logical.resources()["depth"]),
        routed_depth=int(imported.resources()["depth"]),
        reported_depth=int(routed.depth()),
        routing_swaps=swaps,
        two_qubit_gates=two_qubit,
        operations=int(sum(counts.values())),
        final_layout=tuple(int(index) for index in position),
        seconds=seconds,
        optimization_level=optimization_level,
        seed=seed,
        basis_gates=ROUTING_BASIS,
        circuit=imported,
    )


# ---------------------------------------------------------------- qcec adapter
#: How each gate of the IR is handed to Qiskit, and through it to QCEC. The table is
#: total over :data:`apedp_dyn.vqc.gates.GATE_SPECS` -- a test asserts that -- so a
#: gate added to the IR cannot reach this adapter as a silent no-op, and the two
#: decompositions Qiskit's QASM 2 importer does not define (`swap`, `cswap`) are
#: emitted as gates rather than declining to run.
_QISKIT_GATES: dict[str, str] = {
    "h": "h", "x": "x", "y": "y", "z": "z", "s": "s", "sdg": "sdg",
    "t": "t", "tdg": "tdg", "rx": "rx", "ry": "ry", "rz": "rz",
    # The IR's `phase` is U1; Qiskit 2.x calls the same matrix `p`.
    "phase": "p",
    "cx": "cx", "cz": "cz", "swap": "swap", "ccx": "ccx", "cswap": "cswap",
}


def _import_qcec() -> Any:
    try:
        return importlib.import_module("mqt.qcec")
    except ImportError as error:
        raise ToolUnavailable(
            f"mqt.qcec is not importable ({type(error).__name__}: {error}); install "
            f"it with `pip install 'apedp[compare]'` to run the comparison") from error


def qiskit_circuit(circuit: Circuit) -> Any:
    """The same circuit as a Qiskit `QuantumCircuit`, gate for gate.

    Built from the IR rather than from QASM text because Qiskit's QASM 2 importer
    does not define `swap` or `cswap`, and dropping those or re-expressing them
    would mean the other tool decided a circuit this package did not hand it.
    """
    circuit.validate()
    for op in circuit.ops:
        if not isinstance(op, Gate):  # pragma: no cover - validate rejects the rest
            raise NotAUnitaryCircuitError(
                f"the circuit contains a {type(op).__name__.lower()}")
        if op.name not in _QISKIT_GATES:
            raise ToolUnavailable(
                f"{op.name!r} has no mapping to a Qiskit gate, so the other tool "
                f"cannot be given this circuit unchanged")
    qiskit = importlib.import_module("qiskit")
    qc = qiskit.QuantumCircuit(circuit.num_qubits)
    for op in circuit.ops:
        assert isinstance(op, Gate)  # the loop above rejected non-gates
        method = getattr(qc, _QISKIT_GATES[op.name])
        method(*op.params, *op.targets)
    return qc


@dataclasses.dataclass(frozen=True)
class ForeignEquivalence:
    """A third-party equivalence checker's answer, with what it ran to get there.

    `criteria` holds one normalised criterion per repeat. Identical calls can return
    different criteria near the tool's own numerical threshold, so a single call is a
    draw and not an answer: `verdict` is ``"unstable"`` when the repeats disagree,
    and the criteria themselves are what the record carries.
    """

    tool: str
    version: str | None
    criteria: tuple[str, ...]
    considered: tuple[bool, ...]
    seconds: tuple[float, ...]
    checkers: tuple[Mapping[str, Any], ...]
    numerical_tolerance: float | None = None
    timeout: float | None = None
    seed: int | None = None

    @property
    def verdict(self) -> str:
        distinct = set(self.criteria)
        return self.criteria[0] if len(distinct) == 1 else "unstable"

    @property
    def decided(self) -> bool:
        return self.verdict in ("equivalent", "equivalent_up_to_global_phase",
                                "not_equivalent")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "compared",
            "tool": self.tool,
            "version": self.version,
            "verdict": self.verdict,
            "criteria": list(self.criteria),
            "considered_equivalent": list(self.considered),
            "repeats": len(self.criteria),
            "seconds": list(self.seconds),
            "checkers": [dict(checker) for checker in self.checkers],
            "numerical_tolerance": self.numerical_tolerance,
            "timeout": self.timeout,
            "seed": self.seed,
            "decides": ("U_c = e^(i*phi) U_r for one global phase phi: QCEC's "
                        "decision-diagram, ZX and simulation checkers on the same "
                        "circuits, in its own gate representation"),
        }


def _criterion(result: Any) -> str:
    """`EquivalenceCriterion.equivalent` -> `"equivalent"`."""
    return str(result.equivalence).rsplit(".", 1)[-1]


def qcec_equivalence(reference: Circuit, candidate: Circuit, *, repeats: int = 1,
                     timeout: float | None = None,
                     numerical_tolerance: float | None = None,
                     seed: int | None = None) -> ForeignEquivalence:
    """Ask MQT QCEC whether `candidate` computes the same unitary as `reference`.

    The tool is given its own choice of method (``auto``: decision diagrams, ZX
    rewriting and simulation-based falsification), its own default numerical
    tolerance, and no time limit -- the configuration is not tuned towards an answer,
    because a threshold chosen to agree with this package would not be a second
    opinion. `repeats` runs the identical call more than once; QCEC's simulation
    checker randomises unless a seed is given, so a repeat count is how a boundary
    verdict is caught being unstable instead of being reported as *the* verdict.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    qcec = _import_qcec()
    left, right = qiskit_circuit(reference), qiskit_circuit(candidate)
    options: dict[str, Any] = {}
    if timeout is not None:
        options["timeout"] = float(timeout)
    if numerical_tolerance is not None:
        options["numerical_tolerance"] = float(numerical_tolerance)
    if seed is not None:
        options["seed"] = int(seed)
    criteria: list[str] = []
    considered: list[bool] = []
    seconds: list[float] = []
    checkers: Mapping[str, Any] = {}
    for _ in range(repeats):
        start = time.perf_counter()
        try:
            result = qcec.verify(left, right, **options)
        except Exception as error:  # the tool's own exception hierarchy
            raise ToolUnavailable(
                f"mqt.qcec could not decide this pair: {type(error).__name__}: "
                f"{error}") from error
        seconds.append(time.perf_counter() - start)
        criteria.append(_criterion(result))
        considered.append(bool(result.considered_equivalent()))
        if not checkers:
            checkers = {"checkers": list(result.json().get("checkers", []))}
    if numerical_tolerance is None:
        numerical_tolerance = _qcec_default_tolerance(qcec)
    return ForeignEquivalence(
        tool="mqt.qcec",
        version=_package_version("mqt.qcec"),
        criteria=tuple(criteria),
        considered=tuple(considered),
        seconds=tuple(seconds),
        checkers=tuple(checkers.get("checkers", ())),
        numerical_tolerance=numerical_tolerance,
        timeout=timeout,
        seed=seed,
    )


def _qcec_default_tolerance(qcec: Any) -> float | None:
    """The numerical tolerance QCEC runs with when the caller does not set one."""
    try:
        module = importlib.import_module(f"{qcec.__name__}.pyqcec")
        return float(module.Configuration().execution.numerical_tolerance)
    except Exception:  # noqa: BLE001 - a private module, read for the record only
        return None


# --------------------------------------------------------------------- sampling
class SamplingUnsupported(RuntimeError):
    """A sampling comparison this package cannot referee exactly at that width."""


def _bits_text(bits: tuple[bool, ...]) -> str:
    return "".join("1" if bit else "0" for bit in bits)


@dataclasses.dataclass(frozen=True)
class Sampling:
    """Another engine's Z-basis samples of one specification, and its own clock.

    `counts` is keyed by outcome in this package's qubit order (qubit 0 first). The
    preparation and sampling clocks are kept apart because the two engines prepare
    differently -- a tableau clone per shot here, a compiled reference sample there --
    and folding them together would hide which part is being compared.
    """

    tool: str
    variant: str
    version: str | None
    target_sha256: str
    num_qubits: int
    shots: int
    seed: int | None
    counts: Mapping[tuple[bool, ...], int]
    preparation_seconds: float
    sampling_seconds: float
    preparation: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shots < 1:
            raise ValueError(f"shots must be positive, got {self.shots}")
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be positive, got {self.num_qubits}")
        if sum(self.counts.values()) != self.shots:
            raise ValueError(
                f"the counts sum to {sum(self.counts.values())}, not the {self.shots} "
                f"shots the run claims")
        for bits in self.counts:
            if len(bits) != self.num_qubits:
                raise ValueError(
                    f"an outcome has {len(bits)} bits, expected {self.num_qubits}")

    def counts_sha256(self) -> str:
        """A hash of the full count table, so a record can cite samples compactly."""
        text = "\n".join(f"{_bits_text(bits)}:{count}"
                         for bits, count in sorted(self.counts.items()))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        top = sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))[:8]
        return {
            "tool": self.tool,
            "variant": self.variant,
            "version": self.version,
            "target_sha256": self.target_sha256,
            "qubits": self.num_qubits,
            "shots": self.shots,
            "seed": self.seed,
            "distinct_outcomes": len(self.counts),
            "counts_sha256": self.counts_sha256(),
            "preparation_seconds": self.preparation_seconds,
            "sampling_seconds": self.sampling_seconds,
            "samples_per_second": (self.shots / self.sampling_seconds
                                   if self.sampling_seconds > 0 else None),
            "preparation": dict(self.preparation),
            "top_outcomes": [{"bits": _bits_text(bits), "count": count}
                             for bits, count in top],
        }


def _import_stim() -> Any:
    try:
        return importlib.import_module("stim")
    except ImportError as error:
        raise ToolUnavailable(
            f"stim is not importable ({type(error).__name__}: {error}); install it "
            f"with `pip install 'apedp[compare]'` to run the sampling comparison") from error


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def stim_sampling(target: TargetSpecification, *, shots: int, seed: int | None = None,
                  method: str = "elimination") -> Sampling:
    """Stim's Z-basis samples of the *same* signed stabilizers, timed from the spec.

    Stim has no direct "sample this stabilizer state" entry point, so the adapter
    builds a `stim.Tableau` from our signed Pauli strings, asks it for a preparation
    circuit, appends the measurements, and compiles a sampler. That conversion is
    charged to *preparation*, not to sampling, so the sampling clock measures the
    same thing on both sides: drawing `shots` outcomes.

    Stim reads Pauli-string characters with index 0 as qubit 0 (the opposite of
    Qiskit), so our strings are passed in order -- and the adapter does not rely on
    that being true: every sample set is decided against the exact distribution.
    """
    stim = _import_stim()
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    start = time.perf_counter()
    stabilizers = []
    for labels, phase in target.stabilizers:
        sign = "+" if int(phase) % 4 == 0 else "-"
        stabilizers.append(stim.PauliString(
            sign + "".join("IXYZ"[int(label)] for label in labels)))
    tableau = stim.Tableau.from_stabilizers(stabilizers)
    circuit = tableau.to_circuit(method=method)
    circuit.append("M", range(target.num_qubits))
    sampler = circuit.compile_sampler(seed=seed)
    preparation_seconds = time.perf_counter() - start

    start = time.perf_counter()
    array = sampler.sample(shots=shots)
    sampling_seconds = time.perf_counter() - start
    counts: dict[tuple[bool, ...], int] = {}
    for row in array:
        bits = tuple(bool(value) for value in row)
        counts[bits] = counts.get(bits, 0) + 1
    return Sampling(
        tool="stim",
        variant=f"tableau.{method}",
        version=_package_version("stim"),
        target_sha256=target.sha256(),
        num_qubits=target.num_qubits,
        shots=shots,
        seed=seed,
        counts=counts,
        preparation_seconds=preparation_seconds,
        sampling_seconds=sampling_seconds,
        preparation={
            "circuit_sha256": hashlib.sha256(str(circuit).encode("utf-8")).hexdigest(),
            "stabilizers_given": len(target.stabilizers),
            "conversion": "stim.Tableau.from_stabilizers -> to_circuit -> compile_sampler",
            "measurement": "Z basis on every qubit, appended by the harness",
        },
    )


def apedp_sampling(target: TargetSpecification, *, shots: int,
                   seed: int | None = None) -> Sampling:
    """This package's Z-basis samples of the same specification, on the same clock.

    The tableau is built once; every shot clones it and measures qubit by qubit. That
    per-shot clone is part of the sampler -- it is how a state without a compiled
    circuit gets sampled here -- and it is not hidden from the clock. The comparison
    this records is therefore implementation against implementation: a Python
    tableau clone per shot against Stim's compiled C++ sampler.
    """
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    start = time.perf_counter()
    state = target.stabilizer_state()
    rng = np.random.default_rng(seed)
    preparation_seconds = time.perf_counter() - start

    start = time.perf_counter()
    counts: dict[tuple[bool, ...], int] = {}
    for _shot in range(shots):
        bits = state.clone().sample(rng=rng)
        counts[bits] = counts.get(bits, 0) + 1
    sampling_seconds = time.perf_counter() - start
    return Sampling(
        tool="apedp.synthesis",
        variant="stabilizer.tableau",
        version=_package_version("apedp"),
        target_sha256=target.sha256(),
        num_qubits=target.num_qubits,
        shots=shots,
        seed=seed,
        counts=counts,
        preparation_seconds=preparation_seconds,
        sampling_seconds=sampling_seconds,
        preparation={
            "stabilizers_given": len(target.stabilizers),
            "conversion": "TargetSpecification -> StabilizerState, cloned per shot",
            "measurement": "Z basis on every qubit, in order",
        },
    )


def _probability_vector(target: TargetSpecification) -> tuple[np.ndarray, tuple[tuple[bool, ...], ...]]:
    """The exact distribution as a vector over every outcome, in a fixed order."""
    if target.num_qubits > 12:
        raise SamplingUnsupported(
            f"an exact distribution of {target.num_qubits} qubits is not enumerable; "
            f"a wider comparison is refereed by exact marginals instead "
            f"(apedp_dyn.vqc.marginals), so this full-distribution path stops at 12 "
            f"qubits rather than quoting a distribution it cannot compute")
    ordered = tuple(itertools.product((False, True), repeat=target.num_qubits))
    exact = target.stabilizer_state().distribution()
    probabilities = np.array([exact.get(bits, 0.0) for bits in ordered], dtype=float)
    probabilities /= probabilities.sum()
    return probabilities, ordered


def _tv_band(probabilities: np.ndarray, shots: int, *, second_shots: int | None = None,
             draws: int = 300, quantile: float = 0.999, seed: int = 0) -> float:
    """A null band for the TV distance, by simulating correct samplers from `p`.

    A chi-square band would need expected cell counts that a 4,096-outcome state with
    20,000 shots does not have. The null is instead simulated exactly: draw the same
    number of shots from the exact distribution `draws` times, and take the requested
    quantile of the TV distance. With `second_shots` set the null is *two independent
    samplers* rather than one sampler against `p`, which is the null for comparing the
    two engines with each other.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if draws < 1:
        raise ValueError(f"draws must be positive, got {draws}")
    rng = np.random.default_rng(seed)
    first = rng.multinomial(shots, probabilities, size=draws) / shots
    if second_shots is None:
        distances = 0.5 * np.abs(first - probabilities).sum(axis=1)
    else:
        second = rng.multinomial(second_shots, probabilities, size=draws) / second_shots
        distances = 0.5 * np.abs(first - second).sum(axis=1)
    return float(np.quantile(distances, quantile))


def adjudicate_sampling(samples: Sampling, target: TargetSpecification, *, draws: int = 300,
                        quantile: float = 0.999, null_seed: int = 0) -> dict[str, Any]:
    """Decide one sample set against the exact distribution of its specification.

    `CONSISTENT` means the total-variation distance from the exact distribution is
    inside the simulated null band; `INCONSISTENT` means the samples are not draws from
    the state the specification names. Above the enumerable width the referee is the
    exact distribution of chosen *subsets* of wires instead, decided by
    :func:`apedp_dyn.vqc.marginals.adjudicate_marginal`. The maximum per-outcome z-score and the number
    of cells beyond four sigma are reported as diagnostics, not as the verdict: with
    thousands of cells a four-sigma cell occurs by chance at a rate worth knowing but
    not worth gating on.
    """
    if samples.target_sha256 != target.sha256():
        raise ValueError(
            "the sample set was drawn from a different specification than the one it "
            "is being adjudicated against")
    if samples.num_qubits != target.num_qubits:
        raise ValueError(
            f"the sample set has {samples.num_qubits} qubits, the target has "
            f"{target.num_qubits}")
    probabilities, ordered = _probability_vector(target)
    counts = np.array([samples.counts.get(bits, 0) for bits in ordered], dtype=float)
    shots = samples.shots
    empirical = counts / shots
    tv = float(0.5 * np.abs(empirical - probabilities).sum())
    band = _tv_band(probabilities, shots, draws=draws, quantile=quantile, seed=null_seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(probabilities > 0,
                     (counts - shots * probabilities)
                     / np.sqrt(shots * probabilities * (1.0 - probabilities)),
                     np.where(counts > 0, np.inf, 0.0))
    finite = z[np.isfinite(z)]
    payload = samples.as_dict()
    payload.update({
        "tv_to_exact": tv,
        "band": band,
        "band_draws": draws,
        "band_quantile": quantile,
        "band_null_seed": null_seed,
        "verdict": "CONSISTENT" if tv <= band else "INCONSISTENT",
        "exact_outcomes": int((probabilities > 0).sum()),
        "max_abs_z": float(np.abs(finite).max()) if finite.size else 0.0,
        "cells_beyond_4_sigma": int((np.abs(finite) > 4.0).sum()),
        "outcomes_outside_the_exact_support": int(
            (counts[probabilities <= 0] > 0).sum()),
    })
    return payload


def sampling_pair(first: Sampling, second: Sampling, target: TargetSpecification, *,
                  draws: int = 300, quantile: float = 0.999, null_seed: int = 1
                  ) -> dict[str, Any]:
    """Compare two sample sets with each other, under a null where both are correct.

    The distance is the TV between the two empirical distributions. The band is the
    same distance between two *independent* correct samplers at the same shot counts,
    so a disagreement is only reported when the two sets differ by more than sampling
    noise explains -- not merely because they are two finite samples of one state.
    """
    for samples in (first, second):
        if samples.target_sha256 != target.sha256():
            raise ValueError("both sample sets must be drawn from the target")
    if first.shots != second.shots:
        raise ValueError(
            f"the two sample sets used {first.shots} and {second.shots} shots; a "
            f"matched comparison uses one budget")
    probabilities, ordered = _probability_vector(target)
    first_counts = np.array([first.counts.get(bits, 0) for bits in ordered], dtype=float)
    second_counts = np.array([second.counts.get(bits, 0) for bits in ordered], dtype=float)
    tv = float(0.5 * np.abs(first_counts / first.shots
                            - second_counts / second.shots).sum())
    band = _tv_band(probabilities, first.shots, second_shots=second.shots,
                    draws=draws, quantile=quantile, seed=null_seed)
    return {
        "engines": [first.tool, second.tool],
        "shots_each": first.shots,
        "tv": tv,
        "band": band,
        "band_draws": draws,
        "band_quantile": quantile,
        "band_null_seed": null_seed,
        "verdict": "AGREE" if tv <= band else "DISAGREE",
        "counts_sha256": {first.tool: first.counts_sha256(),
                          second.tool: second.counts_sha256()},
    }

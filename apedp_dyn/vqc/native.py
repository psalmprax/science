"""Rust-first execution, Python fallback: the APEDP-VQC dual-language bridge.

Status: experimental

The division of labour is the one the architecture review recommended, and it is
enforced here rather than described:

* **Rust** (`rust/apedp-vqc`, import name ``apedp_vqc``) is the execution core:
  Pauli algebra, the stabilizer tableau, sampling, exact distributions, and the
  matrix-free Clifford images. These are bit vectors, phase arithmetic mod four
  and a shot loop -- integer work with a predictable cost, which is what a
  systems language is for.
* **Python** (`stabilizer.py`, `equivalence.py`, `discovery.py`) stays the
  research and adjudication layer: candidate generation, the discovery loop, the
  referee, the reports and every verdict.

Three rules this module implements, from the review's list:

1. *One canonical circuit specification.* Rust parses the same canonical JSON
   ``Circuit.to_json()`` emits. There is no second builder and no second gate
   vocabulary; unknown operations are refused by name.
2. *Rust speed must not compromise correctness.* Every entry point here is a
   ``try_``-shaped call that returns ``None`` when the fast path could not
   answer, and ``None`` means "the Python tier decides". :func:`assert_parity`
   recomputes a circuit's deterministic answers in Python and compares them
   exactly, which is what the conformance suite and the headline record use.
3. *Keep independent verification independent.* Rust produces images and
   outcomes; the *referee* in `equivalence.py` compares them and writes the
   verdict. A Rust module never decides that a candidate is equivalent, and
   `equivalence.adjudicate_clifford_equivalence` re-derives the image table in
   Python below :data:`PARITY_LIMIT` qubits instead of trusting the fast path.

The module is optional. The root ``pyproject.toml`` does not build it; it exists
after ``make vqc-native`` (``maturin develop``) has run, so callers check
:func:`is_available` instead of assuming. With the module absent every call
returns ``None`` and the Python tier answers -- the same results, slower, which
is why the counters below exist: a run can prove which path it took rather than
being trusted about it.
"""
from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping
from types import ModuleType
from typing import Any

# `stabilizer` imports this module, so the only imports here are the ones that
# cannot close a cycle: `Circuit` for a type hint, and `StabilizerState` inside
# `assert_parity` where it is actually needed.
from apedp_dyn.vqc.circuit import Circuit

__all__ = [
    "ENGINE",
    "PARITY_LIMIT",
    "NativeParityError",
    "assert_parity",
    "distribution",
    "engine_info",
    "expectation",
    "gate_tables",
    "generators",
    "images",
    "is_available",
    "last_failure",
    "python_only",
    "reset_tallies",
    "run",
    "tallies",
    "unavailable_reason",
]

#: The import name of the native module, and the name reports carry.
ENGINE = "apedp_vqc"

#: Below this width the bridge re-derives the image table in Python and refuses a
#: disagreement. Above it the fast path is trusted and the conformance suite is
#: the correctness argument, stated as such in the report rather than implied.
PARITY_LIMIT = 24

#: Set to a truthy value to force the reference path in-process; the benchmark
#: uses it so both engines are timed on identical inputs by identical code.
_PYTHON_ONLY_ENV = "APEDP_VQC_PYTHON_ONLY"

_MODULE: ModuleType | None
_IMPORT_ERROR: str | None
try:  # pragma: no branch - the two outcomes are both exercised by the suite
    _MODULE = importlib.import_module(ENGINE)
    _IMPORT_ERROR = None
# The extension is optional by construction: an ImportError, a version skew or a
# symbol mismatch all mean the same thing here, so the broad catch is the point.
except Exception as error:  # noqa: BLE001 - an absent optional extension is a fallback, not a crash
    _MODULE = None
    _IMPORT_ERROR = f"{type(error).__name__}: {error}"

_TALLIES: dict[str, int] = {"rust": 0, "python": 0, "fallback": 0}
_LAST_FAILURE: list[str | None] = [None]


class NativeParityError(RuntimeError):
    """The native engine and the Python reference disagreed on a frozen input."""


def reset_tallies() -> None:
    """Zero the path counters (tests and benchmarks call this per case)."""
    for key in _TALLIES:
        _TALLIES[key] = 0
    _LAST_FAILURE[0] = None


def tallies() -> dict[str, int]:
    """How many calls each path answered, so a report cannot claim ``rust`` falsely."""
    return dict(_TALLIES)


def last_failure() -> str | None:
    """The most recent reason the fast path declined, for a report footnote."""
    return _LAST_FAILURE[0]


def python_only() -> bool:
    """Whether this process is pinned to the reference path."""
    return bool(os.environ.get(_PYTHON_ONLY_ENV))


def is_available() -> bool:
    """Whether the native module is importable in this environment."""
    return _MODULE is not None and not python_only()


def unavailable_reason() -> str:
    """Why the fast path is unavailable, in words a report can print."""
    if _MODULE is None:
        return (f"{ENGINE} is not importable ({_IMPORT_ERROR}); build it with "
                f"`make vqc-native` (maturin develop --release)")
    if python_only():
        return f"{_PYTHON_ONLY_ENV} is set, so the reference path is pinned"
    return "available"


def engine_info() -> dict[str, Any]:
    """The engine identity a record cites: name, version, availability and reason."""
    return {
        "engine": ENGINE,
        "available": is_available(),
        "version": getattr(_MODULE, "__version__", None) if _MODULE else None,
        "reason": unavailable_reason(),
        "parity_limit_qubits": PARITY_LIMIT,
    }


def _call(function: str, *args: Any) -> Any:
    """Call one native entry point, or return ``None`` with the failure recorded.

    A refusal is not an error to escalate: it means this engine could not answer
    this circuit, and the Python tier answers instead. The one thing this
    function must never do is turn a refusal into an acceptance, and it cannot:
    nothing here produces a verdict.
    """
    if not is_available():
        _TALLIES["python"] += 1
        return None
    assert _MODULE is not None
    try:
        return getattr(_MODULE, function)(*args)
    except Exception as error:  # noqa: BLE001 - a refusal is a fallback, never a verdict
        _LAST_FAILURE[0] = f"{function}: {type(error).__name__}: {error}"
        _TALLIES["fallback"] += 1
        return None


# ---------------------------------------------------------------- pure answers
def gate_tables() -> dict[str, Any] | None:
    """The native module's derived Clifford tables, for a parity assertion."""
    payload = _call("gate_tables")
    if payload is None:
        return None
    _TALLIES["rust"] += 1
    _out: dict[str, Any] | None = json.loads(payload)
    return _out


def images(circuit: Circuit) -> tuple[tuple[tuple[int, ...], int], ...] | None:
    """``clifford_pauli_images`` from the fast path, in the Python tier's shape."""
    payload = _call("equivalence_images", circuit.to_json())
    if payload is None:
        return None
    _TALLIES["rust"] += 1
    images = []
    index = "IXYZ"
    for name, labels, phase in json.loads(payload):
        if not labels:
            _LAST_FAILURE[0] = f"equivalence_images returned a short row for {name}"
            _TALLIES["fallback"] += 1
            return None
        images.append((tuple(index.index(character) for character in labels), int(phase)))
    return tuple(images)


def distribution(circuit: Circuit) -> dict[tuple[bool, ...], float] | None:
    """The exact Z-basis distribution of the prepared state, if the fast path answers."""
    payload = _call("distribution", circuit.to_json())
    if payload is None:
        return None
    _TALLIES["rust"] += 1
    return {
        tuple(character == "1" for character in bits): float(probability)
        for bits, probability in json.loads(payload)
    }


def generators(circuit: Circuit) -> tuple[str, ...] | None:
    """The stabilizer generators of the prepared state, as ``'+ZZI'``-style text."""
    payload = _call("generators", circuit.to_json())
    if payload is None:
        return None
    _TALLIES["rust"] += 1
    return tuple(json.loads(payload))


def expectation(circuit: Circuit, labels: str) -> float | None:
    """``<P>`` for one Pauli string ``labels`` (``'IXYZ'`` text) on the prepared state."""
    value = _call("expectation", circuit.to_json(), labels)
    if value is None:
        return None
    _TALLIES["rust"] += 1
    return float(value)


def run(circuit: Circuit, *, shots: int, seed: int | None = None):
    """Execute a circuit on the fast path; ``None`` means "use the Python tier".

    The returned :class:`~apedp_dyn.vqc.backends.RunResult` names the engine it
    came from, so a report cannot mislabel a Python run as native.
    """
    from apedp_dyn.vqc.backends import RunResult  # local: backends imports this module

    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    if seed is None:
        # A seed the result can carry: the Rust shot loop is deterministic in its
        # seed, and an unrecorded random seed would make the run unreproducible.
        seed = int.from_bytes(os.urandom(8), "big") >> 1
    payload = _call("run_report", circuit.to_json(), int(shots), int(seed))
    if payload is None:
        return None
    _TALLIES["rust"] += 1
    report = json.loads(payload)
    counts = {str(key): int(value) for key, value in report["counts"]}
    samples = tuple(tuple(character == "1" for character in sample)
                    for sample in report["samples"])
    if sum(counts.values()) != shots or len(samples) != shots:
        _LAST_FAILURE[0] = "run_report returned a shot count that does not add up"
        _TALLIES["fallback"] += 1
        return None
    return RunResult(
        backend="stabilizer",
        shots=shots,
        seed=seed,
        counts=counts,
        samples=samples,
        ancilla_excitation={int(qubit): float(value)
                            for qubit, value in report["ancilla_excitation"]},
        engine="rust",
    )


# ------------------------------------------------------------------- conformance
def assert_parity(circuit: Circuit) -> dict[str, Any]:
    """Recompute the deterministic answers in Python and compare them exactly.

    Compares every image, the generator strings, and -- for a measurement-free
    circuit at a width where enumeration is readable -- the whole exact
    distribution. Raises :class:`NativeParityError` naming the first entry that
    disagrees, because "the fast path is faster" is not a claim about
    correctness and this is the function that makes it one.
    """
    from apedp_dyn.vqc.stabilizer import StabilizerState, clifford_pauli_images

    if any(tag != "gate" for tag in _op_tags(circuit)):
        # Images are the conjugation action of a *unitary*, so a measured or
        # noisy circuit has no table to compare. Refusing here is clearer than
        # letting the fast path's refusal surface as a parity failure.
        raise ValueError("assert_parity needs a measurement-free Clifford circuit, got "
                         f"operations {sorted(set(_op_tags(circuit)) - {'gate'})}")
    if not is_available():
        raise NativeParityError(f"the native engine is unavailable: {unavailable_reason()}")

    native_images = images(circuit)
    if native_images is None:
        raise NativeParityError(f"the native engine declined the images: {last_failure()}")
    reference_images = clifford_pauli_images(circuit)
    if native_images != reference_images:
        for index, (native_row, reference_row) in enumerate(
                zip(native_images, reference_images, strict=True)):
            if native_row != reference_row:
                raise NativeParityError(
                    f"image {index} disagrees: rust {native_row} vs python {reference_row}")
        raise NativeParityError("the image tables have different lengths")

    state = StabilizerState(circuit.num_qubits)
    for op in circuit.ops:
        if hasattr(op, "name"):  # a Gate
            state.apply_gate(op.name, op.targets, op.params)
    native_generators = generators(circuit)
    if native_generators is not None and native_generators != state.generator_strings():
        raise NativeParityError(
            f"generators disagree: rust {native_generators} vs "
            f"python {state.generator_strings()}")

    summary = {
        "engine": ENGINE,
        "version": getattr(_MODULE, "__version__", None),
        "images_compared": len(reference_images),
        "generators_compared": len(state.generator_strings()),
        "distribution_compared": 0,
    }
    if circuit.num_qubits <= 10 and not any(
            character == "measure" for character in _op_tags(circuit)):
        native_distribution = distribution(circuit)
        reference_distribution = state.distribution()
        if native_distribution != reference_distribution:
            raise NativeParityError(
                f"distributions disagree: rust {native_distribution} vs "
                f"python {reference_distribution}")
        summary["distribution_compared"] = len(reference_distribution)
    return summary


def _op_tags(circuit: Circuit) -> tuple[str, ...]:
    """One tag per op, with a conditioned gate *not* tagged as a plain gate.

    The images are the conjugation action of a unitary, and a classically conditioned
    gate is not one: on the Rust side it is skipped when its condition does not hold,
    and on the Python side `conjugate_pauli` refuses it outright. Tagging it as `gate`
    let `assert_parity` compare two different circuits and report a parity failure that
    was really an unsupported input, so it gets its own tag and the same refusal path
    every measured circuit already has.
    """
    tags = []
    for op in circuit.ops:
        if hasattr(op, "name"):
            tags.append("gate" if not op.condition else "conditioned_gate")
        else:
            tags.append(type(op).__name__.lower())
    return tuple(tags)


def ensure_available_or_reason() -> Mapping[str, Any]:
    """A one-line status a benchmark can print: engine, availability, reason."""
    return engine_info()

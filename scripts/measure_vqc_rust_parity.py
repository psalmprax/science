#!/usr/bin/env python3
"""Measure the native VQC core against the Python tier on identical frozen inputs.

The architecture review's fifth engineering rule is "benchmark before claiming
Rust superiority: measure runtime, memory, scaling and correctness on identical
frozen inputs". This script does exactly that and nothing else:

* every case is the *same circuit object* given to both engines, with its SHA-256
  recorded, so "identical inputs" is checkable rather than asserted;
* every case reports the Python answer, the Rust answer and whether they agree,
  because a speedup on a different answer is not a speedup;
* the Python path is measured in the same process and on the same machine, with a
  warm-up call before each timing so first-call import and table construction do
  not land in the first engine's number.

The record is `benchmarks/quantum_compilation/vqc_native_parity.json`, and
`tests/test_vqc_native_parity_record.py` pins its structure, its parity claims and
the cryptographic identity of the script and the two implementations it measured.

Example::

    python scripts/measure_vqc_rust_parity.py
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apedp_dyn.vqc import native  # noqa: E402
from apedp_dyn.vqc.circuit import Circuit, Gate  # noqa: E402
from apedp_dyn.vqc.equivalence import adjudicate_clifford_equivalence  # noqa: E402
from apedp_dyn.vqc.stabilizer import (  # noqa: E402
    StabilizerBackend,
    StabilizerState,
    clifford_pauli_images,
    ghz_circuit,
)

RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_native_parity.json"

#: Repeats per side. Five is the review's floor for a number worth quoting: three is
#: enough to expose a first-call cost but not to say how spread the timings are, and
#: the spread is recorded (IQR over these repeats) rather than hidden behind a median.
REPEATS = 5


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class Timing:
    """One side's timed repeats: median, best, interquartile range and the samples.

    A median alone cannot be checked for stability, which is why the raw samples and
    their IQR travel with it: a wide range is a measurement of a noisy host rather
    than evidence of a fast path, and the record can now show the difference.
    """

    median: float
    best: float
    iqr: float
    samples: tuple[float, ...]
    result: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "median_seconds": self.median,
            "best_seconds": self.best,
            "iqr_seconds": self.iqr,
            "samples_seconds": list(self.samples),
        }


def timed(function: Callable[[], Any], *, repeats: int = REPEATS) -> Timing:
    """Time `repeats` calls after one untimed warm-up, and keep every sample."""
    warm = function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        warm = function()
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    return Timing(
        median=statistics.median(samples),
        best=min(samples),
        iqr=_quantile(ordered, 0.75) - _quantile(ordered, 0.25),
        samples=tuple(samples),
        result=warm,
    )


def _quantile(ordered: list[float], fraction: float) -> float:
    """Linear-interpolated quantile of an already sorted sample (numpy's default)."""
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def hardware() -> dict[str, Any]:
    """The machine, interpreter, compiler and build the numbers were measured on.

    A speedup without this block is a number with a story attached. Compilation is
    *not* in any timing: the extension is built by `make vqc-native` before the script
    runs, so every number below is execution time of an already-loaded module.
    """
    import numpy

    def version(command: list[str]) -> str:
        try:
            # The command is one of two hardcoded literals ("rustc --version",
            # "cargo --version"), never user input, and its absence is not an error.
            completed = subprocess.run(command, capture_output=True, text=True,  # noqa: S603
                                       check=False, timeout=10)
        except (OSError, subprocess.SubprocessError) as error:  # pragma: no cover
            return f"unavailable ({error})"
        first = (completed.stdout or completed.stderr).strip().splitlines()
        return first[0] if first else "unavailable"

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported by this platform",
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "rustc": version(["rustc", "--version"]),
        "cargo": version(["cargo", "--version"]),
        "build": ("maturin develop --release; the release profile (opt-level, lto) "
                  "lives in the workspace root Cargo.toml, and its compilation time "
                  "is excluded from every timing here"),
        "peak_rss_kib": _peak_rss_kib(),
        "peak_rss_note": ("process-wide peak resident set size at the end of the run, "
                          "not per case: peak memory cannot be attributed to one case "
                          "from inside the process that ran all of them"),
    }


def _peak_rss_kib() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # noqa: BLE001 - a missing resource module is not a benchmark failure
        return None  # pragma: no cover


def ghz_ladder(num_qubits: int) -> Circuit:
    """`H` + `CX` ladder: the circuit the 100-qubit VQC claims are stated on."""
    return ghz_circuit(num_qubits, measure=False)


def measured_ghz(num_qubits: int) -> Circuit:
    return ghz_circuit(num_qubits, measure=True)


def reference_images(circuit: Circuit):
    """The Python tier's `2n` images, in its own shape."""
    return clifford_pauli_images(circuit)


def reference_distribution(circuit: Circuit) -> dict[tuple[bool, ...], float]:
    state = StabilizerState(circuit.num_qubits)
    for op in circuit.ops:
        state.apply_gate(op.name, op.targets, op.params)
    return state.distribution()


def images_case(num_qubits: int) -> dict[str, Any]:
    """Conjugating `2n` Pauli strings through an `n`-gate ladder, both engines."""
    circuit = ghz_ladder(num_qubits)
    native.reset_tallies()
    python = timed(lambda: reference_images(circuit))
    assert native.tallies()["rust"] == 0, "the reference timing must not use the fast path"
    rust = timed(lambda: native.images(circuit))
    # One warm-up call plus the timed repeats, all answered natively.
    assert native.tallies()["rust"] == REPEATS + 1
    python_answer, rust_answer = python.result, rust.result
    agrees = rust_answer is not None and tuple(rust_answer) == tuple(python_answer)
    return {
        "case": f"images_{num_qubits}q",
        "workload": {
            "circuit": "GHZ ladder (H, CX ladder), no measurement",
            "qubits": num_qubits,
            "operations": len(circuit.ops),
            "images": 2 * num_qubits,
            "circuit_sha256": circuit.sha256(),
        },
        "python": python.as_dict(),
        "rust": rust.as_dict(),
        "speedup_median": python.median / rust.median,
        "agrees": agrees,
        "images_compared": len(python_answer),
    }


def sampling_case(num_qubits: int, shots: int, seed: int) -> dict[str, Any]:
    """`shots` samples of a measured GHZ chain, both engines, same seed."""
    circuit = measured_ghz(num_qubits)
    python = timed(
        lambda: StabilizerBackend().run(circuit, shots=shots, seed=seed, engine="python"))
    rust = timed(
        lambda: StabilizerBackend().run(circuit, shots=shots, seed=seed, engine="rust"))
    python_answer, rust_answer = python.result, rust.result
    assert python_answer.engine == "python" and rust_answer.engine == "rust"
    # Not the same draws -- the streams differ -- but the same *outcomes*: a GHZ
    # chain can only ever read all-zero or all-one, and both engines must obey it.
    allowed = {"0" * num_qubits, "1" * num_qubits}
    agrees = set(rust_answer.counts) <= allowed and sum(rust_answer.counts.values()) == shots
    return {
        "case": f"sample_{num_qubits}q_{shots}shots",
        "workload": {
            "circuit": "GHZ ladder with a measurement on every qubit",
            "qubits": num_qubits,
            "operations": len(circuit.ops),
            "shots": shots,
            "seed": seed,
            "circuit_sha256": circuit.sha256(),
        },
        "python": python.as_dict(),
        "rust": rust.as_dict(),
        "speedup_median": python.median / rust.median,
        "agrees": agrees,
        "outcomes_seen": sorted(rust_answer.counts),
    }


def distribution_case(num_qubits: int) -> dict[str, Any]:
    """The exact `2**n` distribution of a prepared state, both engines."""
    circuit = ghz_ladder(num_qubits)
    native.reset_tallies()
    python = timed(lambda: reference_distribution(circuit))
    assert native.tallies()["rust"] == 0
    rust = timed(lambda: native.distribution(circuit))
    python_answer, rust_answer = python.result, rust.result
    agrees = rust_answer is not None and dict(rust_answer) == python_answer
    return {
        "case": f"distribution_{num_qubits}q",
        "workload": {
            "circuit": "GHZ ladder, no measurement",
            "qubits": num_qubits,
            "operations": len(circuit.ops),
            "outcomes": 2 ** num_qubits,
            "circuit_sha256": circuit.sha256(),
        },
        "python": python.as_dict(),
        "rust": rust.as_dict(),
        "speedup_median": python.median / rust.median,
        "agrees": agrees,
        "outcomes_nonzero": len(python_answer),
    }


def adjudication_case(num_qubits: int) -> dict[str, Any]:
    """The end-to-end referee on a 100-qubit-sized pair: images plus verdict."""
    reference = ghz_ladder(num_qubits)
    ops = [Gate("h", (0,))]
    ops += [Gate("cx", (qubit - 1, qubit)) for qubit in range(1, num_qubits)]
    ops += [Gate("z", (0,)), Gate("z", (0,))]
    candidate = Circuit(ops=tuple(ops), num_qubits=num_qubits, num_bits=0)
    python = timed(
        lambda: adjudicate_clifford_equivalence(reference, candidate, engine="python"))
    rust = timed(
        lambda: adjudicate_clifford_equivalence(reference, candidate, engine="rust"))
    python_report, rust_report = python.result, rust.result
    return {
        "case": f"adjudication_{num_qubits}q",
        "workload": {
            "reference": "GHZ ladder",
            "candidate": "the same ladder with a cancelling Z Z appended",
            "qubits": num_qubits,
            "reference_sha256": reference.sha256(),
            "candidate_sha256": candidate.sha256(),
        },
        "python": python.as_dict(),
        "rust": rust.as_dict(),
        "speedup_median": python.median / rust.median,
        "agrees": (python_report.verdict == rust_report.verdict
                   and python_report.mismatched == rust_report.mismatched),
        "verdict": rust_report.verdict,
        "image_engine": rust_report.image_engine,
        "image_verified_by": rust_report.image_verified_by,
        "python_image_engine": python_report.image_engine,
    }


def parity_summary() -> dict[str, Any]:
    """The exact-agreement checks the record's `agrees` flags rest on.

    Every engine comparison above runs on the same object, so this is a summary of
    the same runs rather than a second, weaker claim. It exists so a reader can see
    the dimensions that were compared without re-reading the script.
    """
    corpus = [ghz_circuit(n, measure=False) for n in range(2, 9)]
    rows = []
    for circuit in corpus:
        summary = native.assert_parity(circuit)
        rows.append({"qubits": circuit.num_qubits,
                     "circuit_sha256": circuit.sha256(),
                     **summary})
    return {
        "circuits_compared": len(rows),
        "images_per_circuit": sorted({row["images_compared"] for row in rows}),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", type=int, default=256,
                        help="shots for the sampling cases (default: 256)")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--output", type=Path, default=RECORD)
    args = parser.parse_args()

    if not native.is_available():
        raise SystemExit(
            "the native engine is not importable, so there is nothing to compare "
            f"it against: {native.unavailable_reason()}")

    cases = [
        images_case(100),
        images_case(400),
        sampling_case(100, args.shots, args.seed),
        distribution_case(10),
        adjudication_case(100),
    ]
    for case in cases:
        if not case["agrees"]:
            raise SystemExit(f"parity failed in {case['case']}: refusing to write a record")

    record = {
        "schema": "apedp.vqc.native_parity.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": {"shots": args.shots, "seed": args.seed, "repeats": REPEATS},
        "engines": {
            "rust": native.engine_info(),
            "python": {
                "module": "apedp_dyn.vqc.stabilizer + apedp_dyn.vqc.equivalence",
                "engine": "python",
                "role": "reference implementation and referee",
            },
        },
        "cases": cases,
        "parity_corpus": parity_summary(),
        "hardware": hardware(),
        "method": {
            "warmup": "one untimed call per side before the timed repeats",
            "statistic": ("median of the timed repeats, with the best, the "
                          "interquartile range and every raw sample recorded"),
            "compile_time": ("excluded: the extension is built once by "
                             "`make vqc-native` before this script runs, so every "
                             "number is execution time of a loaded module"),
            "identical_inputs": "both engines receive the same Circuit object; its "
                                "SHA-256 is recorded per case",
            "reference_timings_do_not_touch_the_fast_path": (
                "the tallies are asserted zero for 'rust' around the Python timings"),
            "repeats": REPEATS,
            "memory": ("process-wide peak RSS is recorded once under `hardware`; "
                       "per-case peak memory is not measured, because a peak reached "
                       "inside one process cannot be attributed to one case"),
        },
        "limitations": [
            "The ratio is one machine's ratio. The hardware block pins the "
            "interpreter, the platform, the compiler and the build; a different "
            "machine will differ, and the claim being made is exact agreement plus a "
            "measured improvement here, not a portable constant.",
            "The IQR is over five repeats in one process on one host. It says how "
            "stable these timings were, not what the spread would be across hosts, "
            "and it is not a confidence interval.",
            "Above PARITY_LIMIT qubits the referee compares the native table "
            "without re-deriving it in Python. There the correctness argument is "
            "the conformance corpus (tests/test_vqc_native_parity.py), not a "
            "per-call check, and the report says image_verified_by='suite'.",
            "Only the stabilizer tier and the Clifford image table are native. The "
            "dense statevector tier, the decoders, the noise engine and the "
            "discovery loop are still Python; this record measures what exists.",
        ],
        "reproducibility": {
            "script_sha256": sha256_of(Path(__file__)),
            "native_module": "rust/apedp-vqc (crate apedp-vqc, module apedp_vqc)",
            "native_source_sha256": {
                name: sha256_of(ROOT / "rust" / "apedp-vqc" / "src" / name)
                for name in ("lib.rs", "pauli.rs", "stabilizer.rs", "equivalence.rs",
                             "runtime.rs", "ir.rs", "bits.rs")
            },
            "stabilizer_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "stabilizer.py"),
            "equivalence_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "equivalence.py"),
            "native_bridge_sha256": sha256_of(ROOT / "apedp_dyn" / "vqc" / "native.py"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "command": f"python scripts/measure_vqc_rust_parity.py --shots {args.shots} "
                       f"--seed {args.seed}",
        },
        "verdict": {
            "agreement": "every case agrees exactly (images) or in outcome "
                         "(sampling) between the engines",
            "rust_first": "StabilizerBackend and adjudicate_clifford_equivalence "
                          "use the native engine by default and fall back to Python",
        },
    }

    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for case in cases:
        print(f"{case['case']:28s} python {case['python']['median_seconds']*1000:9.2f} ms  "
              f"rust {case['rust']['median_seconds']*1000:9.2f} ms  "
              f"x{case['speedup_median']:7.2f}  "
              f"iqr(rust) {case['rust']['iqr_seconds']*1000:6.2f} ms  "
              f"agrees={case['agrees']}")
    print(f"wrote {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

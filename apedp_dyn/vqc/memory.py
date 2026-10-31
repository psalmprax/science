"""Peak memory of a sampling engine, measured in its own process and its own window.

Status: experimental

The memory comparison has a property the throughput comparison does not: two numbers are
needed and neither can be taken in one interpreter as it stands.

* ``VmHWM`` (and ``ru_maxrss``) is a **high-water mark that never falls**, so two engines
  measured in one process cannot be separated: the second can never come out below the
  first even if the first freed everything it took. Each engine therefore runs in a fresh
  child interpreter.
* The whole-process peak of such a child is mostly the interpreter and the modules the
  engine needs to exist at all -- measured here at ~190 MiB before a single shot is
  drawn, against a few megabytes of samples. Quoting that peak would be quoting numpy,
  not the engine. Each child therefore **resets the peak counter after its imports and
  after reconstructing the specification** (``/proc/self/clear_refs``), so the peak it
  reports afterwards is the high-water mark of the sampling step itself, read against the
  resident set it started that step with.

So a measurement is four numbers, and all four are recorded:

* ``floor_bytes`` -- the child's resident set after importing the engine's libraries and
  rebuilding the specification, i.e. what the engine costs before it samples anything;
* ``window_peak_bytes`` -- ``VmHWM`` after the sampling step;
* ``growth_bytes`` -- the difference: the resident-set growth the sampling step needed;
* ``absolute_peak_bytes`` -- the child's peak over its whole life, for a reader who wants
  "what does it take to run this at all".

The growth is only comparable because the window is cut in the same place on both sides:
same interpreter, same imported modules for that engine, same reconstructed target, the
counter reset immediately before the sampling call. A zero growth means the sampling fit
inside the heap the imports had already touched -- a true statement about the sampling
step, not a failed measurement -- and the reader can see that it is zero because
``floor_bytes`` is reported beside it.

Verification: the child that measures an engine also reports the count hash of the
samples it drew, and the record asserts it equals the hash of the recorded run, so the
memory is measured on the workload whose samples are in the record rather than on a
similar one.

What it does not decide: peak RSS is not a ceiling claim, it is not a GPU or
allocator-internal measurement, it is one run per engine rather than a median, and the
growth number is resident memory rather than bytes allocated (an engine that recycles
heap shows less than one that allocates fresh). A platform without a peak-RSS counter, or
where the peak cannot be reset, gets the numbers it can support and a recorded reason for
the ones it cannot -- an abstention, never a zero presented as a measurement.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from apedp_dyn.vqc.external import Sampling, apedp_sampling, stim_sampling
from apedp_dyn.vqc.synthesis import I, TargetSpecification, X, Y, Z

__all__ = [
    "ENGINES",
    "MemoryMeasurement",
    "MemoryUnavailable",
    "current_rss_bytes",
    "measure_engine",
    "peak_rss_bytes",
    "reset_peak",
    "target_from_strings",
]

#: The engines this harness can measure, and the modules each needs loaded before it can
#: run at all. The imports are loaded *inside* the measurement window's floor, so an
#: engine is never charged for a library it cannot run without.
ENGINES: dict[str, dict[str, Any]] = {
    "grammar": {"imports": ()},
    "stim": {"imports": ("stim",)},
}

_FROM_LABEL = {"I": I, "X": X, "Y": Y, "Z": Z}
_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")


class MemoryUnavailable(RuntimeError):
    """Peak memory cannot be measured here, with the reason a record can print."""


def _status_kib(field: str) -> int | None:
    """One ``kB`` field of ``/proc/self/status``, in bytes, or ``None``."""
    try:
        text = _STATUS.read_text()
    except OSError:  # pragma: no cover - no procfs
        return None
    for line in text.splitlines():
        if line.startswith(field):
            try:
                return int(line.split()[1]) * 1024
            except (IndexError, ValueError):  # pragma: no cover - malformed procfs
                return None
    return None


def peak_rss_bytes() -> int:
    """This process's peak resident set in bytes, or :class:`MemoryUnavailable`.

    ``resource`` is missing on Windows, and on some containers the counter is present
    but reads zero; both are refusals rather than zeros, because a reported zero would
    be a measurement nobody made.
    """
    peak = _status_kib("VmHWM")
    if peak is None:
        try:
            import resource
        except ImportError as error:  # pragma: no cover - not reached on Linux/macOS
            raise MemoryUnavailable(
                f"no peak-RSS counter: `resource` is unavailable "
                f"({type(error).__name__}: {error}) and /proc/self/status is not readable"
            ) from error
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if peak <= 0:  # pragma: no cover - platform dependent
            raise MemoryUnavailable(
                f"ru_maxrss read {peak}; this platform reports no peak resident set")
        # Linux reports kibibytes and macOS bytes, and guessing is a factor of 1024.
        peak = peak if sys.platform == "darwin" else peak * 1024
    if peak <= 0:  # pragma: no cover - defensive
        raise MemoryUnavailable(f"the peak-RSS counter read {peak}")
    return peak


def current_rss_bytes() -> int:
    """This process's resident set in bytes, or :class:`MemoryUnavailable`."""
    rss = _status_kib("VmRSS")
    if rss is None:
        raise MemoryUnavailable(
            "/proc/self/status does not report VmRSS, so a measurement window cannot be "
            "opened against the resident set the step starts from")
    return rss


def reset_peak() -> bool:
    """Reset this process's peak-RSS counter to its current resident set.

    ``clear_refs`` with the value 5 resets ``VmHWM``/``ru_maxrss`` on Linux, which is what
    makes a *window* measurable instead of only a whole process. Returns False when the
    kernel or a container refuses the write, so the caller can report the missing number
    instead of a wrong one.
    """
    try:
        _CLEAR_REFS.write_text("5")
    except OSError:  # pragma: no cover - kernel or container dependent
        return False
    return True


def _bits_from_text(text: str, num_qubits: int) -> tuple[bool, ...]:
    if len(text) != num_qubits or any(bit not in "01" for bit in text):
        raise ValueError(f"{text!r} is not a {num_qubits}-bit outcome")
    return tuple(bit == "1" for bit in text)


def target_from_strings(name: str, num_qubits: int, strings: Sequence[str],
                        *, decides: str = "",
                        support: Mapping[str, float] | None = None
                        ) -> TargetSpecification:
    """Rebuild a specification from its printed stabilizers, across a process boundary.

    `support` is included because a declaring target's hash covers it: GHZ and the cluster
    states name the outcomes they expect, and a rebuild that dropped that would hash
    differently and be refused by the child's check -- which is exactly what happened
    when this function was written without it.

    The caller checks the round-trip by hash, so a parser that mis-read a sign fails
    loudly instead of measuring a different state.
    """
    stabilizers = []
    for text in strings:
        text = str(text)
        if len(text) != num_qubits + 1 or text[0] not in "+-":
            raise ValueError(f"{text!r} is not a signed {num_qubits}-qubit Pauli string")
        labels = []
        for character in text[1:]:
            if character not in _FROM_LABEL:
                raise ValueError(f"{text!r} contains {character!r}, not I/X/Y/Z")
            labels.append(_FROM_LABEL[character])
        stabilizers.append((tuple(labels), 0 if text[0] == "+" else 2))
    declared = None
    if support is not None:
        declared = {_bits_from_text(bits, num_qubits): float(probability)
                    for bits, probability in support.items()}
    return TargetSpecification(name=name, num_qubits=num_qubits,
                               stabilizers=tuple(stabilizers), support=declared,
                               decides=decides)


@dataclasses.dataclass(frozen=True)
class MemoryMeasurement:
    """One engine's peak memory on one workload, in the window the harness cut for it."""

    engine: str
    variant: str
    version: str | None
    target_sha256: str
    shots: int
    floor_bytes: int
    window_peak_bytes: int
    absolute_peak_bytes: int
    seconds: float
    counts_sha256: str
    peak_reset: bool
    method: str = (
        "VmHWM of a fresh child interpreter, reset after the engine's imports and the "
        "specification's reconstruction, so the window covers the sampling step and "
        "nothing before it; `absolute_peak_bytes` is that child's peak over its whole "
        "life")

    @property
    def growth_bytes(self) -> int:
        return self.window_peak_bytes - self.floor_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "variant": self.variant,
            "version": self.version,
            "target_sha256": self.target_sha256,
            "shots": self.shots,
            "floor_bytes": self.floor_bytes,
            "window_peak_bytes": self.window_peak_bytes,
            "growth_bytes": self.growth_bytes,
            "absolute_peak_bytes": self.absolute_peak_bytes,
            "floor_mib": self.floor_bytes / 2 ** 20,
            "growth_mib": self.growth_bytes / 2 ** 20,
            "absolute_peak_mib": self.absolute_peak_bytes / 2 ** 20,
            "counts_sha256": self.counts_sha256,
            "seconds": self.seconds,
            "peak_reset": self.peak_reset,
            "method": self.method,
        }


#: The repository root, so the child can import this package whatever the caller's
#: working directory is: a measurement that failed on a path would be uninformative.
_ROOT = Path(__file__).resolve().parents[2]


def _child_payload(target: TargetSpecification, *, shots: int, seed: int | None,
                   engine: str) -> dict[str, Any]:
    return {
        "engine": engine,
        "shots": shots,
        "seed": seed,
        "target_sha256": target.sha256(),
        "target": {
            "name": target.name,
            "num_qubits": target.num_qubits,
            "decides": target.decides,
            "stabilizers": list(target.stabilizer_strings()),
            "support": (None if target.support is None else
                        {"".join("1" if bit else "0" for bit in bits): float(probability)
                         for bits, probability in sorted(target.support.items())}),
        },
    }


def _child_main(payload: dict[str, Any]) -> None:
    """The child's whole job: import, rebuild, open the window, sample, report.

    It prints one JSON object on stdout for the parent to read. Everything up to the
    window's opening -- imports, the specification, its hash check -- is charged to the
    floor rather than to the sampling step, on both engines equally.
    """
    engine = payload.get("engine")
    if engine not in ENGINES:
        raise SystemExit(f"unknown engine {engine!r}")
    for module in ENGINES[engine]["imports"]:
        importlib.import_module(module)
    spec = payload["target"]
    target = target_from_strings(spec["name"], int(spec["num_qubits"]),
                                 spec["stabilizers"], decides=spec.get("decides", ""),
                                 support=spec.get("support"))
    if target.sha256() != payload["target_sha256"]:
        raise SystemExit(
            "the reconstructed specification does not hash to the one that was sent; "
            "refusing to measure a different state")
    absolute_peak = peak_rss_bytes()
    reset = reset_peak()
    floor = current_rss_bytes()
    samples: Sampling = (apedp_sampling if engine == "grammar" else stim_sampling)(
        target, shots=int(payload["shots"]), seed=payload.get("seed"))
    print(json.dumps({
        "engine": engine,
        "peak_reset": reset,
        "floor_bytes": floor,
        "absolute_peak_bytes": absolute_peak,
        "window_peak_bytes": peak_rss_bytes(),
        "shots": int(payload["shots"]),
        "counts_sha256": samples.counts_sha256(),
        "variant": samples.variant,
        "version": samples.version,
        "sampling_seconds": samples.sampling_seconds,
        "target_sha256": target.sha256(),
    }))


def _run_child(payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    """Execute one measurement in a fresh interpreter and return what it printed."""
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (str(_ROOT) if not existing
                                 else f"{_ROOT}{os.pathsep}{existing}")
    completed = subprocess.run(
        [sys.executable, "-c",
         "import json, sys\n"
         "from apedp_dyn.vqc.memory import _child_main\n"
         "_child_main(json.loads(sys.stdin.read()))\n"],
        input=json.dumps(payload), capture_output=True, text=True,
        env=environment, cwd=str(_ROOT), timeout=timeout_seconds)
    if completed.returncode != 0:
        raise MemoryUnavailable(
            f"the measurement child exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout).strip()[-400:]}")
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:  # pragma: no cover - defensive
        raise MemoryUnavailable(
            f"the measurement child printed nothing readable ({error}): "
            f"{completed.stdout[:200]!r}") from error


def measure_engine(engine: str, target: TargetSpecification, *, shots: int,
                   seed: int | None = None, timeout_seconds: float = 900.0
                   ) -> MemoryMeasurement:
    """Peak memory of one engine's sampling of `target`, in its own interpreter.

    The child resets the peak counter after its imports and after rebuilding the
    specification, so `growth_bytes` is the sampling step's own resident growth and
    `floor_bytes` is what the engine cost before it sampled anything. The count hash the
    child reports lets a caller prove the measurement was taken on the same samples the
    record holds.
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {sorted(ENGINES)}")
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    start = time.perf_counter()
    reading = _run_child(_child_payload(target, shots=shots, seed=seed, engine=engine),
                         timeout_seconds=timeout_seconds)
    seconds = time.perf_counter() - start
    if reading["target_sha256"] != target.sha256():
        raise MemoryUnavailable(
            "the measured child sampled a different specification than the one sent")
    floor = int(reading["floor_bytes"])
    window_peak = int(reading["window_peak_bytes"])
    if not reading["peak_reset"]:
        # Without the reset the "window" is the whole child, and reporting it as the
        # sampling step would be reporting numpy's footprint as the engine's.
        raise MemoryUnavailable(
            "the peak counter could not be reset on this platform (/proc/self/clear_refs "
            "is not writable), so the sampling step's own peak is not measurable; the "
            "whole-child peak alone would be mostly the interpreter's")
    return MemoryMeasurement(
        engine=engine,
        variant=str(reading["variant"]),
        version=reading["version"],
        target_sha256=reading["target_sha256"],
        shots=int(reading["shots"]),
        floor_bytes=floor,
        window_peak_bytes=window_peak,
        absolute_peak_bytes=int(reading["absolute_peak_bytes"]),
        seconds=seconds,
        counts_sha256=str(reading["counts_sha256"]),
        peak_reset=bool(reading["peak_reset"]),
    )

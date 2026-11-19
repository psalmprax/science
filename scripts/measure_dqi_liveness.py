#!/usr/bin/env python3
"""Where the reference DQI circuit's 1,893-qubit peak comes from, phase by phase.

Status: audit-only

`scripts/measure_dqi_register_structure.py` showed the public 892-qubit decoder
declares exactly the same registers as the reference (4,165 registers, 49,980
register-qubit references), so the halving is a *lifetime* change rather than a
smaller decoder. This script says which lifetimes: it builds the reference at the
benchmark instance (n=70, degree=12, m=73) with the builder instrumented, and
records every allocation and free against the phase that made it.

The two polynomial arrays are 73 slots x 12 bits = 876 qubits each. If both are
allocated in setup and freed only in teardown, the peak is 1,752 plus transients
and nothing inside the EEA can lower it -- which is the hypothesis the published
892-qubit file disproves, and the object this measurement is here to settle.

Usage:
    APEDP_Q_KICKMIX_PYTHON=... .venv-kickmix/bin/python scripts/measure_dqi_liveness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers"))

import kickmix as km  # noqa: E402

try:
    from dqi import dqi_reference_circuit as DR
except ImportError:  # pragma: no cover - the checkout layout
    from apedp_dyn.quantum_compilation import dqi_reference_circuit as DR  # type: ignore

PHASE = ["setup"]
LIVE_IDS: set[int] = set()
PEAK = 0
EVENTS: list[dict] = []


def _record(kind: str, qubits) -> None:
    """Track the live set by qubit id, so a repeated free cannot double-count."""
    global PEAK
    ids = [int(str(q)[1:]) for q in qubits]
    if kind == "alloc":
        LIVE_IDS.update(ids)
    else:
        LIVE_IDS.difference_update(ids)
    live = len(LIVE_IDS)
    if live > PEAK:
        PEAK = live
    EVENTS.append({"phase": PHASE[0], "kind": kind, "size": len(ids), "live": live})


def instrument() -> None:
    original_alloc = km.CircuitBuilder.alloc_qubits
    original_register = km.CircuitBuilder.create_quantum_register
    original_free = km.CircuitBuilder.free

    def alloc(instance, count, *args, **kwargs):
        out = original_alloc(instance, count, *args, **kwargs)
        _record("alloc", list(out))
        return out

    def create_register(instance, count, *args, **kwargs):
        out = original_register(instance, count, *args, **kwargs)
        _record("alloc", list(out))
        return out

    def free(instance, register, *args, **kwargs):
        try:
            qubits = list(register)
        except TypeError:  # pragma: no cover - a scalar handle
            qubits = []
        _record("free", qubits)
        return original_free(instance, register, *args, **kwargs)

    km.CircuitBuilder.alloc_qubits = alloc  # type: ignore[assignment]
    km.CircuitBuilder.create_quantum_register = create_register  # type: ignore[assignment]
    km.CircuitBuilder.free = free  # type: ignore[assignment]


class Phased(DR.ProosAndZalkaDecoder):
    """The reference decoder, with a phase label pushed around the big phases."""

    def _run(self, name, fn, *args, **kwargs):
        previous = PHASE[0]
        PHASE[0] = name
        try:
            return fn(*args, **kwargs)
        finally:
            PHASE[0] = previous

    def _initial_shift_loop(self, *args, **kwargs):
        out = self._run("initial_shift", super()._initial_shift_loop, *args, **kwargs)
        if kwargs.get("is_adjoint"):
            # Everything after the adjoint shift is the end-of-build teardown, where
            # the reference releases the registers it has held for the whole run.
            PHASE[0] = "teardown"
        return out

    def _synchronized_eea(self, *args, **kwargs):
        return self._run("forward_eea" if kwargs.get("is_adjoint") is False else "eea",
                         super()._synchronized_eea, *args, **kwargs)

    def _chien_search_and_forney_mbuc(self, *args, **kwargs):
        return self._run("chien_forney", super()._chien_search_and_forney_mbuc, *args, **kwargs)


def main() -> int:
    n, degree = 70, 12
    m = n + 3
    print(f"instance: n={n} degree={degree} m={m}  (one polynomial array = {m * degree} qubits)")

    instrument()
    decoder = Phased(n, km.GF2Field(degree))
    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(n)]
    n_iters = getattr(DR, "default_n_iters", lambda q: 4)(n)
    decoder.build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=n_iters
    )
    circuit = builder.finish_circuit()

    print(f"built circuit qubits : {circuit.num_qubits}")
    print(f"instrumented peak    : {PEAK}")
    print(f"events               : {len(EVENTS):,}\n")

    # Consecutive same-phase events form one run; a phase revisited later is a new run.
    runs: list[list[dict]] = []
    for event in EVENTS:
        if not runs or runs[-1][-1]["phase"] != event["phase"]:
            runs.append([event])
        else:
            runs[-1].append(event)

    print(f"{'run':<20} {'live in':>8} {'live out':>9} {'peak':>7} {'allocs':>8} {'frees':>7}")
    for index, rows in enumerate(runs):
        first = rows[0]
        live_in = first["live"] - first["size"] if first["kind"] == "alloc" else first["live"] + first["size"]
        print(f"{rows[0]['phase'] + ' #' + str(index):<20} {live_in:8d} {rows[-1]['live']:9d} "
              f"{max(e['live'] for e in rows):7d} "
              f"{sum(1 for e in rows if e['kind'] == 'alloc'):8d} "
              f"{sum(1 for e in rows if e['kind'] == 'free'):7d}")

    print("\nallocation sizes by phase (how the arrays are built):")
    for index, rows in enumerate(runs):
        sizes: dict[int, int] = {}
        for event in rows:
            if event["kind"] == "alloc":
                sizes[event["size"]] = sizes.get(event["size"], 0) + 1
        if sizes:
            top = ", ".join(f"{count} x {size}q" for size, count in
                            sorted(sizes.items(), key=lambda kv: -kv[1])[:4])
            print(f"  {rows[0]['phase']} #{index}: {top}")

    print("\nlargest frees (which phases actually release qubits):")
    for event in sorted((e for e in EVENTS if e["kind"] == "free"), key=lambda e: -e["size"])[:6]:
        print(f"  {event['size']:6d} qubits freed in '{event['phase']}' (live after {event['live']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

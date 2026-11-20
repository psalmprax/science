#!/usr/bin/env python3
"""Read the decoder's own degree registers after every EEA sub-step.

Status: audit-only

`GluzdovV6Config`'s docstring claims the degree-recycling invariants

    deg(A) + deg(a) = n
    deg(B) + deg(b) = n + 1

and `synchronized_eea_with_degree_recycling` returns
``recycled_degree_invariants_verified: True`` without checking either of them.
An audit of the *classical* polynomial EEA measured both sums constant at 69 for
`n = 70` -- not 70 and 71 -- so before any pass body can be rewritten around
these identities, the constants have to come from the circuit's own `n_A`,
`n_B`, `n_a`, `n_b` registers rather than from a docstring.

This script builds the reference decoder on a small instance, stops the
synchronized EEA after each sub-step, and reads the four degree registers out of
the simulator. If `n_A + n_a` and `n_B + n_b` are constant across every
sub-step, that constant is the offset a recycled build would have to use; if
they are not, the invariant is false and no rewriting is possible.

Usage:
    APEDP_Q_KICKMIX_PYTHON=... .venv-kickmix/bin/python scripts/measure_dqi_recycling_offsets.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers"))

import numpy as np  # noqa: E402

import kickmix as km  # noqa: E402

try:
    from dqi import dqi_reference_circuit as DR
except ImportError:  # pragma: no cover - the checkout layout
    from apedp_dyn.quantum_compilation import dqi_reference_circuit as DR  # type: ignore

#: The five sub-steps of one synchronized-EEA iteration, in execution order.
STEPS = ("_update_flag", "_multiply_and_add_or_sub", "_shift_b_array_right",
         "_update_metadata", "_swap_a_and_b")


class _Stop(Exception):
    """Raised inside the builder to end a truncated build."""


class TracedDecoder(DR.ProosAndZalkaDecoder):
    """The reference decoder with a per-sub-step stop switch and a call log."""

    def __init__(self, *args, stop_after: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_after = stop_after
        self.completed = 0
        self.log: list[str] = []
        for name in STEPS:
            raw = inspect.getattr_static(DR.ProosAndZalkaDecoder, name)
            setattr(self, name, self._guard(name, raw))

    def _guard(self, name: str, raw):
        me = self
        is_static = isinstance(raw, staticmethod)
        fn = raw.__func__ if is_static else raw

        def wrapped(*args, **kwargs):
            if me.stop_after is not None and me.completed >= me.stop_after:
                raise _Stop
            if is_static:
                fn(*args, **kwargs)
            else:
                fn(me, *args, **kwargs)
            me.completed += 1
            me.log.append(name)

        return wrapped


def build(n: int, degree: int, stop_after: int | None = None):
    """Build the decoder, keeping handles on the degree registers.

    The reference allocates its registers in a fixed order after the callers'
    syndromes: three `reg_a`, `m` `reg_b`, then `n_A`, `n_B`, `n_a`, `n_b`, `n_q`,
    the flag, the counter and the halting counter. Recording the allocations is
    enough to identify them without touching the reference source.
    """
    gf = km.GF2Field(degree)
    decoder = TracedDecoder(n, gf, stop_after=stop_after)
    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(n)]

    # The pybind builder rejects instance attributes, so the allocation order is
    # recorded by wrapping the class for the duration of the build.
    allocations: list[list] = []
    original_alloc = km.CircuitBuilder.alloc_qubits

    def recording_alloc(instance, count, *args, **kwargs):
        out = original_alloc(instance, count, *args, **kwargs)
        allocations.append(list(out))
        return out

    km.CircuitBuilder.alloc_qubits = recording_alloc  # type: ignore[assignment]
    n_iters = getattr(DR, "default_n_iters", lambda q: 4)(n)
    truncated = False
    try:
        decoder.build_syndrome_decoding_mbuc(
            builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=n_iters
        )
    except _Stop:
        truncated = True
    finally:
        km.CircuitBuilder.alloc_qubits = original_alloc  # type: ignore[assignment]

    m = n + 3
    logm = m.bit_length() + 1
    base = 3 + m  # allocation index of n_A
    meta = {}
    for offset, name in enumerate(("n_A", "n_B", "n_a", "n_b", "n_q")):
        try:
            meta[name] = km.array(allocations[base + offset])
        except IndexError:  # pragma: no cover - only if the reference reorders
            raise SystemExit("allocation order changed; cannot identify degree registers")
    circuit = builder.finish_circuit()
    return circuit, syndromes, meta, n_iters, truncated


def _gf_tables(degree: int) -> tuple[list[int], list[int]]:
    """Exp/log tables over GF(2^degree) with x as the primitive element.

    `GF2FieldModel` only builds tables for the benchmark's degree 12, so the small
    instances use their own tables; the field polynomial is the standard one for
    each degree and the primitive element is x, which is what the reference
    decoder assumes when it writes `alpha^(i+1)*j` into the syndromes.
    """
    primitive = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011,
                 7: 0b10001001, 8: 0b100011101}[degree]
    size = 1 << degree
    exp = [0] * (2 * size)
    log = [0] * size
    value = 1
    for i in range(size - 1):
        exp[i] = value
        log[value] = i
        value <<= 1
        if value & size:
            value ^= primitive
    for i in range(size - 1, 2 * size - 1):
        exp[i] = exp[i - (size - 1)]
    return exp, log


def valid_syndromes(n: int, degree: int, pattern: dict[int, int]) -> np.ndarray:
    """Syndromes of a real error pattern, so the EEA runs an on-nominal trajectory."""
    exp, log = _gf_tables(degree)
    order = (1 << degree) - 1
    out = []
    for i in range(n):
        value = 0
        for position, weight in pattern.items():
            exponent = ((i + 1) * position) % order
            value ^= exp[(log[weight] + exponent) % order]
        out.append(value)
    return np.array(out, dtype=np.uint64)


def read(sim: km.Simulator, register: km.array) -> int:
    bits = sim.read_across_shots(register)[:, 0]
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def main() -> int:
    n, degree = 4, 3
    if "--valid" in sys.argv:
        pattern = {0: 1, 2: 1}  # two errors: inside the correctable weight for n=4
        values = valid_syndromes(n, degree, pattern)
        print(f"syndromes: from the valid {len(pattern)}-error pattern {pattern}")
    else:
        rng = np.random.default_rng(20261103)
        values = rng.integers(0, 1 << degree, size=n).astype(np.uint64)
        print("syndromes: random (off-nominal trajectory)")

    # How many sub-steps does the full run take?
    _, _, _, n_iters, _ = build(n, degree, stop_after=None)
    total = len(STEPS) * n_iters
    print(f"instance: n={n} degree={degree} m={n + 3} n_iters={n_iters} "
          f"syndromes={values.tolist()}")
    print(f"sub-steps before the Chien phase: {total}\n")

    print(f"{'#':>3} {'after':<26} {'n_A':>4} {'n_a':>4} {'sum':>5} "
          f"{'n_B':>4} {'n_b':>4} {'sum':>5}")
    sums_a, sums_b = [], []
    for stop in range(total + 1):
        circuit, syndromes, meta, _, _ = build(n, degree, stop_after=stop)
        sim = km.Simulator(batch_size=1, debug_print=False,
                           counting_operations=False, seed=7)
        sim.clear_for_shot()
        bits = ((values[None, :] >> np.arange(degree, dtype=np.uint64)[:, None]) & 1) == 1
        sim.write_across_shots(km.array([q for s in syndromes for q in s]),
                               bits.reshape(-1, 1))
        sim.do(circuit)
        n_A, n_B, n_a, n_b = (read(sim, meta[k]) for k in ("n_A", "n_B", "n_a", "n_b"))
        label = "(initial state)" if stop == 0 else f"{STEPS[(stop - 1) % len(STEPS)]} #{stop}"
        sums_a.append(n_A + n_a)
        sums_b.append(n_B + n_b)
        print(f"{stop:3d} {label:<26} {n_A:4d} {n_a:4d} {n_A + n_a:5d} "
              f"{n_B:4d} {n_b:4d} {n_B + n_b:5d}")

    print()
    for name, sums, claimed in (("n_A + n_a", sums_a, n),
                                ("n_B + n_b", sums_b, n + 1)):
        distinct = sorted(set(sums))
        status = "CONSTANT" if len(distinct) == 1 else "varies"
        print(f"{name}: {status} at {distinct} (docs claim {claimed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

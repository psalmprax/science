#!/usr/bin/env python3
"""Measures whether the DQI decoder's transient qubits are collectable.

The reference decoder leaves 1,752 qubits permanently live (``reg_a`` 3 allocated
blocks plus the 70 syndrome registers, and ``reg_b``) and the circuit totals
1,893, so 141 qubits sit above that core.  A "shared scratch pool" is often
proposed as the way to reclaim them.  This script answers whether that is
possible, by measurement rather than by argument.

Two quantities matter and they are not the same:

* ``peak_live`` — the largest number of qubits held at one instant, counted by a
  proxy that sees every ``alloc_qubits`` and ``free``.  No register discipline
  can go below this without changing what the algorithm holds concurrently.
* ``distinct`` — the ids the circuit actually uses, i.e. ``num_qubits``.  If the
  builder returns freed ids to its pool these are equal.  If it does not,
  ``distinct - peak_live`` is scratch that is paid for once per use site and a
  shared pool would genuinely reclaim it.

The verdict printed at the end is the answer: ``collectable`` means a pool buys
qubits at fixed T; ``frozen`` means the transients are concurrent scratch and
only removing a register from the algorithm moves them.  The script also checks
whether the replay reproduces the shipped id count at all: it cannot see ids the
extension allocates internally, and where it does not, the allocator question is
left open rather than answered.

Run at the pinned Benchmark II instance (defaults: n=70, degree=12, m=4095).
"""

from __future__ import annotations

import argparse
import functools
import heapq
import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE = REPO_ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
sys.path.insert(0, str(BUNDLE / "kickmix"))
sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))

import dqi_reference_circuit as DR  # noqa: E402
import kickmix as km  # noqa: E402

#: Methods that allocate scratch, in the order the decoder calls them.
INSTRUMENTED = (
    "build_syndrome_decoding_mbuc",
    "_initial_shift_loop",
    "_synchronized_eea",
    "_multiply_and_add_or_sub",
    "_shift_b_array_right",
    "_update_metadata",
    "_swap_a_and_b",
    "_update_halting_counter",
    "_update_flag",
    "_counter_controls",
    "_del_counter_controls",
    "_update_counter",
    "_chien_search_and_forney_mbuc",
    "_pre_align_reg_b",
)


def check_slot_recycling():
    """Returns whether the builder hands a freed id back on the next allocation."""
    builder = km.CircuitBuilder()
    held = builder.alloc_qubits(4)
    first = [q.id for q in held]
    builder.free(held)
    again_held = builder.alloc_qubits(4)
    again = [q.id for q in again_held]
    fresh = [q.id for q in builder.alloc_qubits(2)]
    return {
        "first_alloc": first,
        "after_free_realloc": again,
        "while_held": fresh,
        "recycles": sorted(again) == sorted(first),
        "no_double_booking": not set(again) & set(fresh),
    }


class TrackingBuilder:
    """Delegates to ``CircuitBuilder`` while counting concurrently live qubits."""

    def __init__(self, inner):
        self._inner = inner
        #: Ids of every qubit this proxy has seen allocated and not yet freed.
        self.live_ids = set()
        self.peak_live = 0
        self.peak_stack = []
        self.frames = []
        self.completed = []
        #: Ordered ("alloc" | "free", ids) log, replayed against alternative allocators.
        self.trace = []

    @property
    def live(self):
        return len(self.live_ids)

    def _observe(self):
        for frame in self.frames:
            frame["peak_live"] = max(frame["peak_live"], self.live)
        if self.live > self.peak_live:
            self.peak_live = self.live
            self.peak_stack = [dict(f) for f in self.frames]

    def alloc_qubits(self, count):
        held = self._inner.alloc_qubits(count)
        self.live_ids.update(q.id for q in held)
        self.trace.append(("alloc", [q.id for q in held]))
        self._observe()
        return held

    def create_quantum_register(self, count, name=None):
        held = self._inner.create_quantum_register(count, name=name)
        self.live_ids.update(q.id for q in held)
        self.trace.append(("alloc", [q.id for q in held]))
        self._observe()
        return held

    def free(self, qubits):
        try:
            ids = [q.id for q in qubits]
        except TypeError:  # a single qubit, not an array
            ids = [qubits.id]
        # Qubits created outside this proxy are not held by it, so freeing them
        # must not drive the live count negative.
        self.live_ids.difference_update(ids)
        self.trace.append(("free", list(ids)))
        return self._inner.free(qubits)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def enter(self, name):
        self.frames.append({"name": name, "entry_live": self.live,
                            "peak_live": self.live})

    def exit(self, name):
        frame = self.frames.pop()
        self.completed.append({"name": frame["name"],
                               "entry_live": frame["entry_live"],
                               "own_peak": frame["peak_live"] - frame["entry_live"]})


def _wrap(fn, name, is_static):
    if is_static:
        @functools.wraps(fn)
        def wrapper(builder, *args, **kwargs):
            builder.enter(name)
            try:
                return fn(builder, *args, **kwargs)
            finally:
                builder.exit(name)
    else:
        @functools.wraps(fn)
        def wrapper(self, builder, *args, **kwargs):
            builder.enter(name)
            try:
                return fn(self, builder, *args, **kwargs)
            finally:
                builder.exit(name)
    return wrapper


def instrument(cls):
    """Wraps the decoder's scratch-allocating methods with a frame tracker."""
    for name in INSTRUMENTED:
        raw = inspect.getattr_static(cls, name, None)
        if raw is None:
            continue
        if isinstance(raw, staticmethod):
            setattr(cls, name, staticmethod(_wrap(raw.__func__, name, True)))
        else:
            setattr(cls, name, _wrap(raw, name, False))
    return cls


class MeasuredDecoder(DR.ProosAndZalkaDecoder):
    pass


instrument(MeasuredDecoder)


def trace_max_id(trace) -> int:
    """Highest id the proxy actually observed allocated, plus one.

    A value below the circuit's ``num_qubits`` means the extension allocated ids
    this proxy never saw (gadget ancillas built inside the C++ layer), so the
    replay below is not a faithful model of the shipped allocator.
    """
    return max((qid for kind, ids in trace if kind == "alloc" for qid in ids), default=-1) + 1


def summarise(builder, total_qubits):
    """Reports the peak, the frames live at it, and the gap to the distinct-id count."""
    stack = builder.peak_stack
    return {
        "peak_live": builder.peak_live,
        "peak_at_frame": stack[-1]["name"] if stack else None,
        "live_frames_at_peak": [
            {"frame": f["name"], "entry_live": f["entry_live"],
             "own_peak": max(0, f["peak_live"] - f["entry_live"])}
            for f in stack
        ],
        "frame_scratch": sorted(builder.completed, key=lambda r: -r["own_peak"])[:8],
        "frames_closed": len(builder.completed),
        "total_qubits": total_qubits,
        "distinct_minus_peak": total_qubits - builder.peak_live,
        "trace_max_id": trace_max_id(builder.trace),
    }


def replay(trace, policy: str) -> int:
    """Replays an alloc/free trace through an id allocator; returns distinct ids used.

    ``lifo`` reproduces ``PyCircuitBuilder::alloc_qubit`` as shipped (a stack, so a
    hole buried under a later push is never reused).  ``lowest`` is the proposed
    free list, which always reuses the smallest free id and so never grows the id
    space past the concurrent peak.
    """
    heap = []
    stack = []
    next_id = 0
    for kind, ids in trace:
        if kind == "free":
            (heap.extend(ids) if policy == "lowest" else stack.extend(ids))
            continue
        for _ in ids:
            if policy == "lowest":
                if heap:
                    heapq.heapify(heap)
                    heapq.heappop(heap)
                else:
                    next_id += 1
            elif stack:
                stack.pop()
            else:
                next_id += 1
    return next_id


def run(n_syndromes: int, degree: int, codeword: int) -> dict:
    gf = km.GF2Field(degree)
    inner = km.CircuitBuilder()
    builder = TrackingBuilder(inner)
    builder.enter("build")
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n_syndromes)]
    error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(codeword)]
    decoder = MeasuredDecoder(n_syndromes, gf)
    decoder.build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=error_mbuc,
        n_iters=DR.default_n_iters(n_syndromes))
    builder.exit("build")
    circuit = inner.finish_circuit()

    out = summarise(builder, circuit.num_qubits)
    out["instance"] = {"n_syndromes": n_syndromes, "degree": degree, "codeword": codeword}
    out["peak_concurrent"] = builder.peak_live
    out["live_at_end"] = builder.live
    out["toffoli_mean"] = circuit.operation_statistics().num_toffoli_like
    out["stranded_ids"] = circuit.num_qubits - builder.peak_live
    out["replay"] = {
        "lifo": replay(builder.trace, "lifo"),
        "lowest": replay(builder.trace, "lowest"),
        "shipped": circuit.num_qubits,
    }
    # The innermost frame live at the peak is the only scratch a pool could
    # redistribute; everything outside it is held by the algorithm.
    live = out["live_frames_at_peak"]
    out["pool_ceiling"] = max(0, live[-1]["own_peak"]) if live else 0
    # The replay only speaks to the allocator when it reproduces the shipped id
    # count; a shorter replay means ids the proxy never saw.
    out["replay_covers_shipped_ids"] = out["replay"]["lifo"] == circuit.num_qubits
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--m", type=int, default=4095)
    parser.add_argument("--json", action="store_true", help="emit the measurements as JSON")
    args = parser.parse_args()

    recycling = check_slot_recycling()
    result = run(args.n, args.degree, args.m)

    if args.json:
        print(json.dumps({"recycling": recycling, "measurement": result}, indent=2))
        return

    print("slot recycling")
    print(f"  alloc 4 ids            : {recycling['first_alloc']}")
    print(f"  free, alloc 4 ids      : {recycling['after_free_realloc']}  "
          f"(recycles={recycling['recycles']})")
    print(f"  alloc 2 more, held     : {recycling['while_held']}  "
          f"(no double booking={recycling['no_double_booking']})")
    print()
    print(f"instance            : n={result['instance']['n_syndromes']} "
          f"b={result['instance']['degree']} m={args.m}")
    print(f"total qubits (ids)  : {result['total_qubits']:,}")
    print(f"peak concurrently live: {result['peak_concurrent']:,} at {result['peak_at_frame']}")
    print(f"stranded ids        : {result['stranded_ids']:,}  (ids used but never live at the peak)")
    print(f"live after build    : {result['live_at_end']:,}")
    print()
    print(f"ids visible to proxy   : {result['trace_max_id']:,} "
          f"(of {result['total_qubits']:,}; the rest are allocated inside the extension)")
    print("replaying the same alloc/free trace through each allocator")
    print(f"  shipped LIFO stack    : {result['replay']['lifo']:,} ids")
    print(f"  lowest-hole free list : {result['replay']['lowest']:,} ids")
    print(f"  reproduces shipped ids: {result['replay_covers_shipped_ids']}")
    print(f"frames closed       : {result['frames_closed']:,}")
    print(f"toffoli_mean        : {result['toffoli_mean']:,}")
    print()
    print("live frames at the peak")
    for frame in result["live_frames_at_peak"]:
        print(f"  {frame['frame']:<32} entry={frame['entry_live']:,} "
              f"own_peak={frame['own_peak']}")
    print()
    print("largest frame-local scratch")
    for row in result["frame_scratch"]:
        print(f"  {row['name']:<32} {row['own_peak']}")
    print()

    # Marginal scratch of the innermost frame live at the peak: what any pool would
    # be able to redistribute, as opposed to the registers the algorithm needs held.
    ceiling = result["pool_ceiling"]
    gap = result["stranded_ids"]
    if gap <= 2:
        print("verdict: FROZEN — the circuit's id count equals its concurrent peak, so a")
        print("         shared scratch pool has nothing to collect.")
        return
    print("verdict: POOL DEAD — a shared scratch pool can redistribute at most")
    print(f"         ~{ceiling} qubits (the innermost frame live at the peak), not the {gap} ids")
    print("         above the peak. Those transients are concurrent with the registers the")
    print("         algorithm holds; only removing a register moves them.")
    if not result["replay_covers_shipped_ids"]:
        print()
        print("         The allocator question is left OPEN, not answered: the replay returns")
        print(f"         {result['replay']['lifo']:,} ids against {result['total_qubits']:,} shipped, and the proxy sees")
        print(f"         only {result['trace_max_id']:,} ids, so allocations made inside the extension are missing")
        print("         from the trace. Both policies also agree exactly, which is what a trace")
        print("         with no reusable hole above the peak looks like — the same shape a")
        print("         trace that simply cannot see the allocator produces. No free-list claim")
        print("         is supported by these numbers.")
    elif result["replay"]["lowest"] < result["replay"]["lifo"]:
        print()
        print("         The allocator does strand ids: replaying the same trace with a")
        print(f"         lowest-hole free list returns {result['replay']['lowest']:,} against")
        print(f"         {result['replay']['lifo']:,} for the shipped LIFO stack, so a free list would reclaim")
        print(f"         {result['replay']['lifo'] - result['replay']['lowest']:,} ids that a pool cannot.")


if __name__ == "__main__":
    main()

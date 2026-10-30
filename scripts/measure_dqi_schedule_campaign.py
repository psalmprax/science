#!/usr/bin/env python3
"""Run the shift/sweep/swap schedule campaign with a pre-declared prediction.

`docs/EEA_AND_PREDICATE_PLAN.md` §4 ranked the three regions that are 78.3% of
the 472,768.2-Toffoli non-product residual -- `shift_b_array_right` 162,479.6,
`sweep_positions` 143,038.0 (non-product) and `swap_a_and_b` 64,224.0 -- as
"a different schedule, not a tidier gate", and §5 closed pointer substitution
until a schedule proof was supplied. `docs/EEA_SCHEDULE_CAMPAIGN.md` is the
campaign that discharges that obligation; this script is its measurement.

The schedule fact the campaign rests on is a property of this decoder, not of
its gates: every predicate register is in a computational basis state, because
the syndrome input is. A basis-state register can be *read* -- a `cx` into the
shared readout ancilla plus `mr`, which leaves the source untouched and the
tracked phase alone -- and the read bit can then be a classical condition
(`push_condition`). A movement whose control is zero is the identity, so a
movement executed only on the shots whose own predicate is one is the same
circuit on those shots and costs no Toffoli on the others. That is the schedule
change under test: **read the movement predicate, then move.**

Each movement predicate is pre-declared here, from the control arm's own chain
costs, before any variant is built:

* **S1 -- the sweep's quotient selection.** `swaps` is at most two one-hot
  `unary_iteration` match qubits (`n_q == j`, `n_q == j + 1`), so at most two
  emitted positions carry the quotient in a live round and the rest are
  quantum-controlled no-ops. Predicted: the selection chain falls to at most
  `degree x 2 x iterations` (the burn), so the saving is at least
  `chain - burn`, and the chain must fall by at least 90% of that.
* **S2 -- the shift's `do_shift` mask.** This is *not* one-hot: it is the XOR of
  three threshold intervals, so the firing set is the active window. A saving
  is **not** predicted; the measurement decides, and a saving under 1% of the
  shift region closes the shift half of the priority.
* **S3 -- the wholesale A/B exchange.** Already gated on `c3 and flag`, whose
  predicate is disjoint from the shift's (one-hot `ctrl`, complementary flag),
  so a schedule that fuses or composes the two is worth exactly zero. The
  campaign measures the per-round overlap and a composed arm that must save 0.0,
  and **S3b** measures the shift's `c3` branch directly by dropping it.

Every variant is falsified before it is believed: the sweep's condition is also
applied one position late (`falsify_sweep_stale`), the shift's condition is also
inverted (`falsify_shift_complement`), and the shift's moves are also conditioned on
the previous slot's mask (`falsify_shift_stale`). Any variant "passing" a
falsifier would mean the condition is decorative and the arm is not measuring a
schedule at all; a falsifier that skips no work is recorded as vacuous rather
than counted as a pass.

Examples::

    /dev/shm/qvcc-venv313/bin/python scripts/build_kickmix_mr.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_schedule_campaign.py
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_schedule_campaign.json"
PATCH = ROOT / "scripts" / "kickmix_mr_opcode.patch"

#: The reconciled six-lever arm at the pinned instance, 64 shots.
PINNED_INSTANCE = (4095, 70, 12)
LEDGER_TOTAL = 1_221_466.8
LEDGER_QUBITS = 1_913
#: Region rows of `dqi_region_ledger.json` the campaign works on.
CONTROL_REGIONS = {
    "shift_b_array_right": 162_479.625,
    "sweep_positions": 143_038.0,
    "swap_a_and_b": 64_224.0,
}
#: The sweep's chain can carry the quotient in at most two positions per round.
MAX_SELECTIONS_PER_ROUND = 2
#: S1 promotes only if the chain's saving is at least this share of its headroom.
S1_MIN_SHARE_OF_HEADROOM = 0.90
#: S2 is reported rather than promoted; below this share the shift half is closed.
S2_CLOSE_SHARE_OF_REGION = 0.01

SIX_LEVERS = dict(gate_forney=True, clear_eea=True, gate_sweep=True, gate_swap=True,
                  gate_shift=True, gate_quotient=True)

#: Which predicates each arm reads classically. The falsifiers are the honest arms
#: with one condition made wrong, so they read the same predicate.
FEATURES = {
    "none": (),
    "sweep": ("sweep",),
    "shift": ("shift",),
    "products": ("products",),
    "both": ("sweep", "shift"),
    "everything": ("sweep", "shift", "products"),
    "stacked": ("sweep", "shift", "products"),
    #: The two already-promoted control gates (halting-round gate and predicate reuse)
    #: with no campaign gate: the reviewer's "combined arm as control".
    "control_gates": (),
    "composed": (),
    "drop_shift_c3": (),
    #: Experiment B: the surviving product stream, gated one level deeper on whether
    #: the multiplier lane (`q_i`, the round's quotient) is zero. A product by zero is
    #: the identity, so the gate is admissible on the same argument as S1-S4; what is
    #: not obvious is whether the quotient is ever zero in a live round, which is why
    #: S6 predicts no saving and lets the measurement decide.
    "zero_q": ("zero_q",),
    "falsify_zero_q": ("zero_q",),
    "falsify_sweep_stale": ("sweep",),
    "falsify_shift_complement": ("shift",),
    "falsify_shift_stale": ("shift",),
}
#: The product stream is the largest set of quantum-controlled gates in the arm; a
#: control that is one-hot at the multiply position would not touch the window.
PRODUCT_PROMOTION_THRESHOLD = 50_000.0

#: One executed `ixor_gf2_mul` inside the multiply window: 54 ANDs plus the 12-bit add.
#: Used by S6's break-even arithmetic, which is declared before the arm runs.
PRODUCT_ARM_UNIT_COST = 66.0

#: What the two already-promoted flag/metadata gates saved, as published in
#: `benchmarks/quantum_compilation/dqi_combined_arm.json` before this campaign ran.
#: S5's prediction uses this figure, not this run's re-measurement of it, so the sum is
#: declared from prior evidence and the re-measurement is an independent check on it.
PRIOR_COMBINED_ARM_SAVING = 2_952.0

#: Marks this campaign reads. The two chain marks are new and nested inside the
#: regions the ledger already names, so `dqi_region_ledger.json` is unaffected.
NAMED = (
    "initial_shift", "shift_step", "forward_eea", "eea_step", "update_flag",
    "update_metadata", "update_halting_counter", "update_counter",
    "multiply_and_add_or_sub", "sweep_positions", "selection_chain", "quotient_div",
    "shift_b_array_right", "shift_chain", "swap_a_and_b", "predicate_read",
    "chien_forney_mbuc", "pre_align_reg_b", "forney_step", "unalign_reg_b",
    "measured_clear",
)


class _PushCondition:
    """``with conditional(builder, bit):`` -- a classically conditioned gate scope."""

    def __init__(self, builder, bit):
        self.builder = builder
        self.bit = bit

    def __enter__(self):
        self.builder.push_condition(self.bit)
        return self

    def __exit__(self, *_exc):
        self.builder.pop_condition()
        return False


def conditional(builder, bit):
    return _PushCondition(builder, bit)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_into(decoder, builder, qubit, bit) -> None:
    """Copy `qubit` into the shared ancilla and measure it into `bit`.

    The gadget every lever in `run_mr_gated_arms.py` uses: `cx` leaves the source
    alone, `mr` clears the ancilla and reports the value, so the read costs neither
    a Toffoli nor a qubit and the tracked phase is untouched. The `predicate_read`
    mark is what lets the campaign report how many measurements the gating adds:
    the reads are the cost the Toffoli metric does not charge.
    """
    with builder.mark("predicate_read"):
        ancilla = decoder._ancilla(builder)
        builder.cx(qubit, ancilla)
        getattr(builder, decoder.clear_with)(ancilla, bit)


def make_decoder(arms, mode: str, mixins: tuple = ()):
    """Build the arm `mode`.

    `mode` is ``"none"`` (a faithful copy of the six-lever arm, with the two
    nested chain marks this campaign reads), a movement variant, or a falsifier
    of one. A falsifier differs from its variant in the condition only.
    `mixins` carries the two already-promoted control gates for the stacked arm, so
    "everything this repository has measured" is one circuit rather than an
    arithmetic claim.
    """

    class ScheduleDecoder(*mixins, arms.MRGatedDecoder):
        campaign_mode = mode

        # -- S2 / S3: the shift ------------------------------------------------
        def _shift_a_and_b(self, builder, reg_b, n_B, n_b, flag, ctrl, it, *,
                           is_adjoint):
            if not self.gate_shift:
                return super()._shift_a_and_b(builder, reg_b, n_B, n_b, flag, ctrl,
                                              it, is_adjoint=is_adjoint)
            with builder.mark("shift_b_array_right"):
                c0_hc, c1_hc, c2_hc, c3_hc = ctrl
                flag_bit = self._read_bit(builder, flag)
                c0_bit = self._read_bit(builder, c0_hc)
                c1_bit = self._read_bit(builder, c1_hc)
                c2_bit = self._read_bit(builder, c2_hc)
                c3_bit = self._read_bit(builder, c3_hc)

                live = builder.alloc_bits(1)
                builder.bit_store(live[0], False)
                builder.bit_store(live[0], True, control=c1_bit[0])
                builder.bit_store(live[0], True, control=c2_bit[0])
                if not self.shift_under:
                    flag_clear = builder.alloc_bits(1)
                    builder.bit_store(flag_clear[0], True)
                    builder.bit_store(flag_clear[0], False, control=flag_bit[0])
                    with conditional(builder, flag_clear[0]):
                        builder.bit_store(live[0], True, control=c0_bit[0])
                        builder.bit_store(live[0], True, control=c3_bit[0])

                extra = None
                if mode == "drop_shift_c3":
                    # A schedule that treats the exchange as covering c3 rounds:
                    # skip the shift there. S3b measures whether that branch is live.
                    dropped = builder.alloc_bits(1)
                    builder.bit_store(dropped[0], False)
                    builder.bit_store(dropped[0], True, control=c3_bit[0])
                    extra = dropped
                if extra is not None:
                    builder.bit_store(live[0], False, control=extra[0])

                with conditional(builder, live[0]):
                    self._shift_body(builder, reg_b, n_B, n_b, flag, ctrl, it,
                                     is_adjoint=is_adjoint)

        def _shift_body(self, builder, reg_b, n_B, n_b, flag, ctrl, it, *,
                        is_adjoint):
            c0_hc, c1_hc, c2_hc, c3_hc = ctrl
            m = len(reg_b)
            lim_B = m - 2
            lim_b = min(m - 2, it // 2 + 1)

            builder.x(flag)
            nf = builder.alloc_qubits(2)
            builder.init_and(arms.DR.km.array([c0_hc, c3_hc]), flag, nf)
            builder.x(flag)
            c0_nf, c3_nf = nf

            ctrl_nB = builder.alloc_qubits(1)[0]
            builder.cx(arms.DR.km.array([c0_nf, c1_hc]), ctrl_nB)

            measuring = "shift" in FEATURES[mode]
            inverted = mode == "falsify_shift_complement"
            stale = mode == "falsify_shift_stale"
            bit = builder.alloc_bits(1) if measuring else None
            complement = builder.alloc_bits(1) if inverted else None
            previous = builder.alloc_bits(1) if stale else None
            if stale:
                builder.bit_store(previous[0], False)

            def move(control, lhs, rhs) -> None:
                with builder.mark("shift_chain"):
                    if not measuring:
                        builder.cswap(control, lhs, rhs)
                    elif inverted:
                        read_into(self, builder, control, bit[0])
                        builder.bit_store(complement[0], True)
                        builder.bit_store(complement[0], False, control=bit[0])
                        with conditional(builder, complement[0]):
                            builder.cswap(control, lhs, rhs)
                    elif stale:
                        # The falsifier: the previous slot's mask, never this one's.
                        read_into(self, builder, control, bit[0])
                        with conditional(builder, previous[0]):
                            builder.cswap(control, lhs, rhs)
                        builder.bit_store(previous[0], False)
                        builder.bit_store(previous[0], True, control=bit[0])
                    else:
                        read_into(self, builder, control, bit[0])
                        with conditional(builder, bit[0]):
                            builder.cswap(control, lhs, rhs)

            order = range(1, m) if is_adjoint else range(m - 1, 0, -1)
            with (
                builder.unary_iteration(n_B, control=ctrl_nB) as cur_B_ge,
                builder.unary_iteration(n_b, control=c2_hc) as cur_b_ge,
                builder.unary_iteration(n_B, control=c3_nf) as cur_B_lt,
            ):
                do_shift = builder.alloc_qubits(1)[0]

                def endpoint(i: int) -> None:
                    arms.DR._xor_threshold(builder, cur_B_ge, lim_B, do_shift,
                                           ctrl_nB, m - i + 1, at_least=True)
                    arms.DR._xor_threshold(builder, cur_b_ge, lim_b, do_shift,
                                           c2_hc, i, at_least=True)
                    arms.DR._xor_threshold(builder, cur_B_lt, lim_B, do_shift,
                                           c3_nf, m - i, at_least=False)

                prev = None
                for i in order:
                    if prev is None:
                        endpoint(i)
                    else:
                        builder.cx(
                            arms.DR.km.array([
                                arms.DR._active(cur_B_ge, m - max(prev, i) + 1, lim_B),
                                arms.DR._active(cur_b_ge, min(prev, i), lim_b),
                                arms.DR._active(cur_B_lt, m - max(prev, i), lim_B),
                            ]),
                            do_shift,
                        )
                    prev = i
                    move(do_shift, reg_b[i - 1], reg_b[i])

                endpoint(prev)
                builder.free(do_shift)

            builder.cx(arms.DR.km.array([c0_nf, c1_hc]), ctrl_nB)
            builder.free(ctrl_nB)
            builder.x(flag)
            builder.del_and(arms.DR.km.array([c0_hc, c3_hc]), flag, nf)
            builder.x(flag)
            builder.free(nf)

        # -- S1: the sweep's per-position quotient selection -------------------
        def _multiply_and_add_or_sub(self, builder, reg_a, reg_b, n_A, n_a, n_q,
                                     ctrl, it, *, is_adjoint: bool):
            c0_hc, _, _, c3_hc = ctrl
            gf = self.field
            m = len(reg_b)

            c0_bit = self._read_bit(builder, c0_hc)
            c3_bit = self._read_bit(builder, c3_hc)
            live = builder.alloc_bits(1)
            builder.bit_store(live[0], False)
            builder.bit_store(live[0], True, control=c0_bit[0])
            builder.bit_store(live[0], True, control=c3_bit[0])

            def accumulate_quotient(index: int, control, live_bit) -> None:
                gate = (conditional(builder, live_bit) if live_bit is not True
                        else nullcontext())
                with builder.mark("quotient_div"), gate:
                    ratio, scaffold = builder.init_gf2_div_with_scaffold(
                        reg_b[index], reg_a[index], field=gf)
                    builder.gf2_iadd(ratio, target=q_i, field=gf, control=control)
                    builder.del_gf2_div_with_scaffold(
                        reg_b[index], reg_a[index], target=ratio,
                        scaffold=scaffold, field=gf)
                    builder.free(scaffold + ratio)

            measuring = "sweep" in FEATURES[mode]
            stale = mode == "falsify_sweep_stale"
            products = "products" in FEATURES[mode]
            zero_q = "zero_q" in FEATURES[mode]
            zero_q_inverted = mode == "falsify_zero_q"
            bit = builder.alloc_bits(1) if measuring else None
            previous = builder.alloc_bits(1) if stale else None
            product_bit = builder.alloc_bits(1) if products else None
            if stale:
                builder.bit_store(previous[0], False)

            def product(control, target) -> None:
                """`ixor_gf2_mul` on its own mask, read classically."""
                if zero_q:
                    # Experiment B: the multiplier is `q_i`, so a zero multiplier makes
                    # the product the identity. It must be read **per position**: the
                    # sweep's selection chain cswaps `q_i` into the a-lanes, so the
                    # multiplier is not a round-level quantity (a round-level read is
                    # wrong, and the first version of this arm measured that).
                    with conditional(builder, product_predicate()):
                        emit(control, target)
                    return
                emit(control, target)

            def product_predicate():
                """`q_i != 0` as a classical bit, read where the product is emitted."""
                with builder.mark("predicate_read"):
                    ancilla = self._ancilla(builder)
                    builder.flip_if_equal(q_i, 0, target=ancilla)
                    zero_bit = builder.alloc_bits(1)
                    getattr(builder, self.clear_with)(ancilla, zero_bit[0])
                nonzero = builder.alloc_bits(1)
                builder.bit_store(nonzero[0], not zero_q_inverted)
                builder.bit_store(nonzero[0], bool(zero_q_inverted), control=zero_bit[0])
                return nonzero[0]


            def emit(control, target) -> None:
                if not products:
                    builder.ixor_gf2_mul(q_i, reg_a[current[0]], target=target,
                                         field=gf, control=control)
                    return
                read_into(self, builder, control, product_bit[0])
                with conditional(builder, product_bit[0]):
                    builder.ixor_gf2_mul(q_i, reg_a[current[0]], target=target,
                                         field=gf, control=control)

            current = [0]

            def sweep_positions() -> None:
                with builder.mark("sweep_positions"):
                    lim_q = min(m - 1, 1 + it)
                    lim_q1 = min(m, 1 + it)
                    lim_a = min(m - 2, it // 2 + 1)
                    lim_A = m - 2
                    order = range(m) if is_adjoint else range(m - 1, -1, -1)

                    with (
                        builder.unary_iteration(n_q, control=c0_hc) as eq_i,
                        builder.unary_iteration(n_q, control=c3_hc) as eq_ip1,
                        builder.unary_iteration(n_a, control=c3_hc) as cur_a,
                        builder.unary_iteration(n_A, control=c0_hc) as cur_A,
                    ):
                        mul = builder.alloc_qubits(1)[0]

                        def endpoint(index: int) -> None:
                            arms.DR._xor_threshold(builder, cur_a, lim_a, mul, c3_hc,
                                                   index + 1, at_least=True, lowest=1)
                            arms.DR._xor_threshold(builder, cur_A, lim_A, mul, c0_hc,
                                                   m - index, at_least=True, lowest=1)

                        prev = None
                        for j in order:
                            if prev is None:
                                endpoint(j)
                            else:
                                builder.cx(
                                    arms.DR.km.array([
                                        arms.DR._active(cur_a, min(prev, j) + 1, lim_a),
                                        arms.DR._active(cur_A, m - max(prev, j), lim_A),
                                    ]),
                                    mul,
                                )
                            prev = j

                            swaps = [a for a in (arms.DR._active(eq_i, j, lim_q),
                                                 arms.DR._active(eq_ip1, j + 1, lim_q1))
                                     if a is not False]
                            multiplies = j < lim_a or j >= 2

                            current[0] = j
                            if is_adjoint and multiplies:
                                # The adjoint keeps its quantum control by default; the
                                # Experiment B arm gates it on the predicate read from
                                # its own position, which is the value its product
                                # multiplies by here.
                                with (conditional(builder, product_predicate())
                                      if zero_q else nullcontext()):
                                    builder.ixor_gf2_mul(q_i, reg_a[j], target=reg_b[j],
                                                         field=gf, control=mul)

                            selector = None
                            both = None
                            if len(swaps) == 1:
                                selector = swaps[0]
                            elif swaps:
                                both = builder.alloc_qubits(1)[0]
                                builder.cx(arms.DR.km.array(swaps), both)
                                selector = both
                            if selector is not None:
                                with builder.mark("selection_chain"):
                                    if not measuring:
                                        builder.cswap(selector, q_i, reg_a[j])
                                    elif stale:
                                        read_into(self, builder, selector, bit[0])
                                        with conditional(builder, previous[0]):
                                            builder.cswap(selector, q_i, reg_a[j])
                                        builder.bit_store(previous[0], False)
                                        builder.bit_store(previous[0], True,
                                                          control=bit[0])
                                    else:
                                        read_into(self, builder, selector, bit[0])
                                        with conditional(builder, bit[0]):
                                            builder.cswap(selector, q_i, reg_a[j])
                                if both is not None:
                                    builder.cx(arms.DR.km.array(swaps), both)
                                    builder.free(both)
                            if not is_adjoint and multiplies:
                                product(mul, reg_b[j])

                        endpoint(prev)
                        builder.free(mul)

            with builder.mark("multiply_and_add_or_sub"), conditional(builder, live[0]):
                q_i = builder.alloc_qubits(gf.degree)
                leading, trailing = (0, m - 1) if is_adjoint else (m - 1, 0)
                leading_bit = c3_bit[0] if is_adjoint else c0_bit[0]
                trailing_bit = c0_bit[0] if is_adjoint else c3_bit[0]
                if self.gate_quotient:
                    accumulate_quotient(leading, c3_hc if is_adjoint else c0_hc,
                                        leading_bit)
                else:
                    accumulate_quotient(leading, c3_hc if is_adjoint else c0_hc, True)
                builder.iadd(n_a, target=n_q)
                sweep_positions()
                builder.isub(n_a, target=n_q)
                if self.gate_quotient:
                    accumulate_quotient(trailing, c0_hc if is_adjoint else c3_hc,
                                        trailing_bit)
                else:
                    accumulate_quotient(trailing, c0_hc if is_adjoint else c3_hc, True)
                builder.free(q_i)

    ScheduleDecoder.__name__ = f"Schedule[{mode}]"
    return ScheduleDecoder


#: The nested chain marks belong to the round-level region that contains them, so a
#: round's shift cost is the region's own gates plus its movement chain (a chain mark
#: fires once per position, so it cannot carry the round index itself).
ROUND_LEVEL = {"shift_b_array_right": "shift", "swap_a_and_b": "swap"}


def campaign_spans(circuit) -> list[tuple[int, int, str, int, str | None, int]]:
    """Spans of constant region, with the region's occurrence and its movement round.

    Returns ``(start, stop, region, region_occurrence, movement, round_index)``. The
    round index is the occurrence of the enclosing round-level region, so every span
    inside one EEA round -- the shift's 72 chain marks included -- carries that round.
    """
    tags = circuit.info_tags()
    if not tags:
        return [(0, len(circuit), "unmarked", 0, None, 0)]

    spans: list[tuple[int, int, str, int, str | None, int]] = []
    stack: list[tuple[str, int]] = []
    seen: dict[str, int] = {}
    state = ("unmarked", 0, None, 0)
    start = 0

    def state_of():
        region, occurrence = "unmarked", 0
        movement, round_index = None, 0
        for name, count in reversed(stack):
            if region == "unmarked" and name in NAMED:
                region, occurrence = name, count
            if movement is None and name in ROUND_LEVEL:
                movement, round_index = ROUND_LEVEL[name], count
        return region, occurrence, movement, round_index

    for index, text in tags:
        if text.startswith("@enter:"):
            name = text[len("@enter:"):]
            seen[name] = seen.get(name, 0) + 1
            stack.append((name, seen[name]))
        elif text.startswith("@exit:"):
            closing = text[len("@exit:"):]
            while stack and stack.pop()[0] != closing:
                pass
        current = state_of()
        if current != state:
            if index > start:
                spans.append((start, index, *state))
            state = current
            start = index + 1
    if start < len(circuit):
        spans.append((start, len(circuit), *state))
    return spans


def region_costs(circuit, out: dict, shots: int) -> tuple[dict, dict, dict]:
    """Executed Toffolis and instructions per shot, by region and by round."""
    executed = np.asarray(out["counts"], dtype=np.int64)
    # Prefix sums: a per-read mark makes the span list long, and a slice sum per span
    # would dominate the run instead of the simulation.
    toffoli_prefix = np.concatenate(
        ([0], np.cumsum(executed * out["toffoli_mask"].astype(np.int64))))
    executed_prefix = np.concatenate(([0], np.cumsum(executed)))

    def span_toffoli(start: int, stop: int) -> int:
        return int(toffoli_prefix[stop] - toffoli_prefix[start])

    def span_instructions(start: int, stop: int) -> int:
        return int(executed_prefix[stop] - executed_prefix[start])

    costs: dict[str, float] = {}
    instructions: dict[str, float] = {}
    rounds: dict[str, dict[int, float]] = {}
    for start, stop, region, _occurrence, movement, round_index in campaign_spans(circuit):
        value = span_toffoli(start, stop) / shots
        costs[region] = costs.get(region, 0.0) + value
        instructions[region] = (instructions.get(region, 0.0)
                                + span_instructions(start, stop) / shots)
        if movement is not None:
            rounds.setdefault(movement, {})[round_index] = (
                rounds.get(movement, {}).get(round_index, 0.0) + value)
    return costs, instructions, rounds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=PINNED_INSTANCE[0])
    parser.add_argument("--n", type=int, default=PINNED_INSTANCE[1])
    parser.add_argument("--degree", type=int, default=PINNED_INSTANCE[2])
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--variants", nargs="*",
                        default=["sweep", "shift", "products", "both", "everything",
                                 "control_gates", "stacked", "composed",
                                 "drop_shift_c3", "zero_q"])
    parser.add_argument("--skip-falsifiers", action="store_true")
    args = parser.parse_args()

    pinned = (args.m, args.n, args.degree) == PINNED_INSTANCE
    falsifiers = [] if args.skip_falsifiers else [
        "falsify_sweep_stale", "falsify_shift_complement", "falsify_shift_stale",
        "falsify_zero_q"]

    runner = load_module("mr_gated_arms_kickmix",
                         ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    km = runner.load_kickmix(args.kickmix_build)
    ledger = load_module("region_ledger",
                         ROOT / "scripts" / "measure_dqi_region_ledger.py")
    arms = ledger.load_arms(km, runner)
    halt = load_module("eea_oracle_ledger",
                       ROOT / "scripts" / "measure_eea_oracle_ledger.py")
    reuse = load_module("flag_predicate_reuse",
                        ROOT / "scripts" / "measure_flag_predicate_reuse.py")
    mixins = (halt.HaltGatedDecoder, reuse.ReusedFlagPredicateDecoder)
    base = runner.builder_class(km)
    shim = runner.shim_for(km, ledger.native_builder(base, "and54"))
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V

    iterations = arms.DR.default_n_iters(args.n)
    burn = args.degree * MAX_SELECTIONS_PER_ROUND * iterations

    spec = V.ProblemSpecDQI.benchmark_instance(
        m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
    weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    tables = V.FieldTables(km.GF2Field(spec.degree))
    rng = np.random.default_rng(20261028)
    syndromes_in, mbuc_in, _ = V.sample_shots(spec, tables, rng, args.shots, weights)
    reference_inputs = V.to_bits(syndromes_in, args.degree)

    def build(decoder_type):
        builder = shim.CircuitBuilder()
        syndromes = [builder.create_quantum_register(args.degree, name=f"S{i}")
                     for i in range(args.n)]
        error_mbuc = [builder.create_classical_register(args.degree, name=f"E{j}")
                      for j in range(args.m)]
        decoder_type(args.n, km.GF2Field(args.degree), **SIX_LEVERS).\
            build_syndrome_decoding_mbuc(builder, syndromes=syndromes,
                                         error_mbuc=error_mbuc, n_iters=iterations)
        return builder.finish_circuit(), syndromes, error_mbuc

    original = arms.DR.km
    arms.DR.km = shim
    try:
        ref_circuit, ref_syn, ref_mbuc = ledger.build_reference(
            km, arms, shim, args.n, args.degree, args.m)
        reference = ledger.simulate(km, ref_circuit, ref_syn, ref_mbuc, syndromes_in,
                                    mbuc_in, args.degree, args.shots)
        del ref_circuit
        gc.collect()

        def measure(decoder_type) -> dict:
            circuit, syn, mbuc = build(decoder_type)
            out = ledger.simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in,
                                  args.degree, args.shots)
            costs, instructions, rounds = region_costs(circuit, out, args.shots)
            reads = instructions.get("predicate_read", 0.0)
            row = {
                "qubits": circuit.num_qubits,
                "mean": out["mean_toffoli_like"],
                "regions": ledger.region_ledger(circuit, out, args.shots),
                "costs": costs,
                "instructions": instructions,
                "instructions_per_shot": float(out["counts"].sum()) / args.shots,
                "read_instructions_per_shot": reads,
                "mid_circuit_reads_per_shot": reads / 2.0,
                "rounds": rounds,
                "syndromes": bool(np.array_equal(out["syndromes"], reference_inputs)),
                "junk": bool(not out["junk"].any()),
                "phase": bool(np.array_equal(out["phase"], reference["phase"])),
                "eea_steps": sum(1 for _index, text in circuit.info_tags()
                                 if text == "@enter:eea_step"),
            }
            del circuit, out
            gc.collect()
            return row

        def region_row(arm, name) -> float:
            for entry in arm["regions"]["rows"]:
                if entry["region"] == name:
                    return entry["executed_toffoli_per_shot"]
            return 0.0

        def region_nonproduct(arm, name) -> float:
            for entry in arm["regions"]["rows"]:
                if entry["region"] == name:
                    return (entry["executed_toffoli_per_shot"]
                            - entry["product_toffoli_per_shot"])
            return 0.0

        def product_region(arm) -> float:
            for entry in arm["regions"]["rows"]:
                if entry["region"] == "sweep_positions":
                    return entry["product_toffoli_per_shot"]
            return 0.0

        control = measure(make_decoder(arms, "none"))

        # ---- the prediction, written before any variant is built ------------
        chain = {"sweep": control["costs"].get("selection_chain", 0.0),
                 "shift": control["costs"].get("shift_chain", 0.0)}
        sweep_headroom = max(0.0, chain["sweep"] - burn)
        shift_region = region_nonproduct(control, "shift_b_array_right")
        prediction = {
            "written_before_variants": True,
            "schedule_change": "read the movement predicate classically, then move: the "
                               "register is a basis state, so the read is deterministic "
                               "and the movement is skipped exactly on the shots whose own "
                               "predicate is zero",
            "control": {
                "total_per_shot": control["mean"],
                "selection_chain_per_shot": chain["sweep"],
                "shift_chain_per_shot": chain["shift"],
                "unit": f"one {args.degree}-bit cswap = {args.degree} CCX "
                        f"= {args.degree} Toffolis",
            },
            "S1_sweep": {
                "hypothesis": "the quotient selection is one-hot: at most two of the "
                              "emitted positions carry it in a live round, so every other "
                              "chain cswap is a quantum-controlled no-op",
                "chain_per_shot": chain["sweep"],
                "burn_per_shot": burn,
                "headroom_per_shot": sweep_headroom,
                "predicted_saving_at_least": sweep_headroom,
                "promotion_rule": (f"the chain must fall by at least "
                                   f"{S1_MIN_SHARE_OF_HEADROOM:.0%} of its headroom"),
            },
            "S2_shift": {
                "hypothesis": "do_shift is the XOR of three threshold intervals, so the "
                              "firing set is the active window and a saving is NOT "
                              "predicted; the measurement decides",
                "chain_per_shot": chain["shift"],
                "region_per_shot": shift_region,
                "predicted_saving_at_least": 0.0,
                "closed_if_below_share_of_region": S2_CLOSE_SHARE_OF_REGION,
            },
            "S3_swap": {
                "hypothesis": "the exchange is gated on c3 and flag, which is disjoint "
                              "from the shift's c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3)) because "
                              "ctrl is one-hot and the flag is one value",
                "predicted_saving_per_shot": 0.0,
                "predicted_overlapping_rounds": 0,
            },
            "S4_products": {
                "hypothesis": "the product's `mul` control is the multiply window, not a "
                              "one-hot, so the saving is whatever the window excludes; no "
                              "figure is predicted",
                "product_stream_per_shot": product_region(control),
                "predicted_saving_at_least": 0.0,
                "promotion_threshold": PRODUCT_PROMOTION_THRESHOLD,
                "why_it_is_here": "beyond the reviewer's two priorities, but the same "
                                  "read-then-gate argument applies to the product's own "
                                  "control, so leaving it unmeasured would leave the "
                                  "reachability arithmetic resting on an untested "
                                  "assumption",
            },
            "S3b_shift_c3_branch": {
                "hypothesis": "dropping the shift in every c3 round (as a schedule that "
                              "treats the exchange as covering it would) saves 0.0 and "
                              "stays correct, because the flag is set in those rounds",
                "predicted_saving_per_shot": 0.0,
            },
            "relocation": {
                "question": "does the Toffoli saving relocate work rather than remove it",
                "declared": "every read is one cx plus one mr, so the added cost is two "
                            "instructions per read and is counted under `predicate_read`; "
                            "the arm's total executed instruction count is reported "
                            "beside its Toffoli count for every arm",
                "predicted": "executed Toffolis fall by the gated work; executed "
                             "instructions fall by the same gates and rise by 2 per read",
            },
            "falsifiers": {
                "falsify_sweep_stale": "condition each selection on the previous "
                                       "position's bit, so the quotient swap never "
                                       "happens where it is needed: must fail",
                "falsify_shift_complement": "condition each move on the complement of its "
                                            "own mask, so it is always skipped: must fail",
                "falsify_shift_stale": "condition each shift move on the previous slot's "
                                       "mask, so the window moves by the wrong slot: "
                                       "must fail",
            },
        }
        print(f"prediction: sweep chain {chain['sweep']:.1f}, burn {burn}, headroom "
              f"{sweep_headroom:.1f}; shift chain {chain['shift']:.1f} of {shift_region:.1f}")

        measured: dict[str, dict] = {}
        #: The five-gate arm is built last, so S5 can be written from the schedule arms
        #: that exist by then and before the stack it predicts exists at all.
        stacked_arms = [name for name in args.variants
                        if name in ("stacked", "control_gates", "zero_q",
                                    "falsify_zero_q")]

        def run(name) -> dict:
            row = measure(make_decoder(arms, name,
                                       mixins if name in ("stacked", "control_gates",
                                                          "zero_q", "falsify_zero_q")
                                       else ()))
            measured[name] = row
            print(f"  {name:26s} Q={row['qubits']} T={row['mean']:.1f}/shot "
                  f"({control['mean'] - row['mean']:+.1f}) chain sweep="
                  f"{row['costs'].get('selection_chain', 0.0):.1f} shift="
                  f"{row['costs'].get('shift_chain', 0.0):.1f} correct="
                  f"{bool(row['syndromes'] and row['junk'] and row['phase'])}")
            return row

        for name in [n for n in list(args.variants) if n not in stacked_arms] + falsifiers:
            run(name)

        # ---- S5: the five-gate stack, declared before the arm exists -------------
        if "stacked" in stacked_arms and "everything" in measured:
            schedule_saving = control["mean"] - measured["everything"]["mean"]
            predicted_saving = schedule_saving + PRIOR_COMBINED_ARM_SAVING
            prediction["S5_stack"] = {
                "declared_by": "the reviewer, 2026-10-30, from the two separate records",
                "written_before_the_stacked_arm": True,
                "hypothesis": "the five gates act on disjoint work: the three schedule "
                              "gates remove movement and product no-ops, the two "
                              "flag/metadata gates remove update work in dead rounds, so "
                              "the stack saves the sum and the Toffolis do not interact",
                "inputs": {
                    "everything_measured_in_this_run": schedule_saving,
                    "prior_combined_arm_from_dqi_combined_arm_json":
                        PRIOR_COMBINED_ARM_SAVING,
                    "unit": "executed Toffolis per shot",
                },
                "predicted_saving_per_shot": predicted_saving,
                "predicted_total_per_shot": control["mean"] - predicted_saving,
                "tolerance": 1.0,
                "falsifier": "an increment over `everything` that differs from the "
                             "prior arm's saving by more than one Toffoli means the two "
                             "gates interact, and the sum is wrong",
            }
            print(f"prediction S5: stack {schedule_saving:.1f} + "
                  f"{PRIOR_COMBINED_ARM_SAVING:.1f} = {predicted_saving:.1f} => T = "
                  f"{control['mean'] - predicted_saving:.1f}/shot")

        # ---- S6: Experiment B, the product stream's own operand predicate -------
        if "zero_q" in args.variants:
            products_after = (product_region(measured["products"])
                             if "products" in measured else None)
            call_cost = PRODUCT_ARM_UNIT_COST
            prediction["S6_zero_q"] = {
                "declared_by": "Experiment B of the reviewer's plan, 2026-10-30",
                "written_before_the_arm": True,
                "hypothesis": "every product in the sweep is `ixor_gf2_mul(q_i, a_j)`, "
                              "so a round whose quotient lane `q_i` is zero emits "
                              "products that are all the identity; reading `q_i == 0` "
                              "once per round and gating the products on it would make "
                              "that round free",
                "declared_saving_per_shot": 0.0,
                "why_zero": "a live round computes a quotient whose leading term is "
                            "nonzero, so the predicate is expected to be zero in no "
                            "round (or too few to pay for the read); the measurement "
                            "decides, and a positive figure would be the surprise",
                "read_cost": "one 12-bit equality test plus one `mr` per multiply "
                             "round; it is charged inside the arm, so any positive "
                             "net saving already includes it",
                "break_even": f"{call_cost:.0f} Toffolis per product call skipped, so "
                              f"the arm must find more than (read cost / {call_cost:.0f}) "
                              f"skipped calls per shot",
                "product_stream_after_the_window": products_after,
                "promotion_rule": "the arm must be correct and save more than it "
                                  "costs to read",
            }
            print("prediction S6: zero-quotient product gate saves 0.0 predicted; "
                  "promotes only if it beats its own read cost")

        for name in stacked_arms:
            run(name)
    finally:
        arms.DR.km = original

    def correct(row) -> bool:
        return bool(row["syndromes"] and row["junk"] and row["phase"])

    def saving(row) -> float:
        return control["mean"] - row["mean"]

    # ---- what S3 predicted, measured from the control arm's own rounds ------
    def round_overlap() -> dict:
        shift_rounds = control["rounds"].get("shift", {})
        swap_rounds = control["rounds"].get("swap", {})
        shared = sorted(set(shift_rounds) & set(swap_rounds))
        overlapping = [round_index for round_index in shared
                       if shift_rounds[round_index] > 0 and swap_rounds[round_index] > 0]
        return {
            "rounds": iterations,
            "shift_rounds_executing": sum(1 for value in shift_rounds.values() if value > 0),
            "swap_rounds_executing": sum(1 for value in swap_rounds.values() if value > 0),
            "overlapping_rounds": len(overlapping),
            "overlap_examples": overlapping[:5],
            "shift_toffoli_per_shot": sum(shift_rounds.values()),
            "swap_toffoli_per_shot": sum(swap_rounds.values()),
            "note": "counts are executions summed over the shot batch, so a round shows "
                    "both only if the two regions execute in the same run; the composed "
                    "arm and its falsifier carry the per-shot half of the claim",
        }

    overlap = round_overlap()

    checks = [{
        "check": "the control copy reproduces the reconciled six-lever total",
        "requirement": f"{LEDGER_TOTAL:.1f} per shot" if pinned else "not the pinned instance",
        "measured": control["mean"] if pinned else None,
        "passed": (abs(control["mean"] - LEDGER_TOTAL) < 0.5) if pinned else True,
    }, {
        "check": "the control copy's movement rows reproduce the region ledger",
        "requirement": dict(CONTROL_REGIONS) if pinned else "not the pinned instance",
        "measured": {name: region_nonproduct(control, name) for name in CONTROL_REGIONS},
        "passed": (all(abs(region_nonproduct(control, name) - value) < 0.1
                       for name, value in CONTROL_REGIONS.items()) if pinned else True),
    }]

    for name in args.variants:
        if name not in measured:
            continue
        row = measured[name]
        if FEATURES.get(name):
            checks.append({
                "check": f"{name}: the gating's reads are the gadget it claims",
                "requirement": "mid-circuit reads present and reported",
                "measured": {"mid_circuit_reads_per_shot": row["mid_circuit_reads_per_shot"],
                             "read_instructions_per_shot": row[
                                 "read_instructions_per_shot"],
                             "instructions_per_shot": row["instructions_per_shot"]},
                "passed": row["mid_circuit_reads_per_shot"] > 0,
            })
        checks.append({
            "check": f"{name}: the arm still computes the reference's function",
            "requirement": "syndromes restored, phase matched, no junk",
            "measured": {"syndromes_restored": row["syndromes"],
                         "phase_matches_reference": row["phase"],
                         "junk_clean": row["junk"]},
            "passed": correct(row),
        })
        checks.append({
            "check": f"{name}: the read costs no qubit and no iteration",
            "requirement": "peak qubits and EEA rounds identical to the control",
            "measured": {"qubits": row["qubits"], "eea_steps": row["eea_steps"],
                         "control_qubits": control["qubits"],
                         "control_eea_steps": control["eea_steps"]},
            "passed": (row["qubits"] == control["qubits"]
                       and row["eea_steps"] == control["eea_steps"]),
        })
        checks.append({
            "check": f"{name}: every region row partitions the measured total",
            "requirement": "reconciles",
            "measured": row["regions"]["reconciles_with_measured_total"],
            "passed": bool(row["regions"]["reconciles_with_measured_total"]),
        })

    if "sweep" in measured:
        row = measured["sweep"]
        fell = chain["sweep"] - row["costs"].get("selection_chain", 0.0)
        checks.append({
            "check": "S1: the quotient selection is one-hot, so its chain collapses",
            "requirement": f"chain saving >= {S1_MIN_SHARE_OF_HEADROOM:.0%} of the "
                           f"{sweep_headroom:.1f} headroom (burn {burn}), and the chain "
                           f"left is at most the burn",
            "measured": {"chain_control": chain["sweep"],
                         "chain_after": row["costs"].get("selection_chain", 0.0),
                         "chain_saving": fell,
                         "arm_saving": saving(row)},
            "passed": fell >= S1_MIN_SHARE_OF_HEADROOM * sweep_headroom,
        })
    if "shift" in measured:
        row = measured["shift"]
        fell = chain["shift"] - row["costs"].get("shift_chain", 0.0)
        closed = fell < S2_CLOSE_SHARE_OF_REGION * shift_region
        checks.append({
            "check": "S2: the shift mask's sparsity, measured rather than predicted",
            "requirement": f"the chain saving is reported; under "
                           f"{S2_CLOSE_SHARE_OF_REGION:.0%} of the "
                           f"{shift_region:.1f} region closes the shift half",
            "measured": {"chain_control": chain["shift"],
                         "chain_after": row["costs"].get("shift_chain", 0.0),
                         "chain_saving": fell,
                         "arm_saving": saving(row),
                         "closed": closed},
            "passed": correct(row),
        })
    if "products" in measured:
        row = measured["products"]
        fell = product_region(control) - product_region(row)
        checks.append({
            "check": "S4: the product stream's `mul` control, measured",
            "requirement": f"correct; saving >= {PRODUCT_PROMOTION_THRESHOLD:.0f} per shot "
                           f"promotes",
            "measured": {"product_stream_control": product_region(control),
                         "product_stream_after": product_region(row),
                         "product_saving": fell,
                         "arm_saving": saving(row)},
            "passed": correct(row),
        })
    if "control_gates" in measured:
        row = measured["control_gates"]
        checks.append({
            "check": "control_gates: the current combined arm reproduced as the control",
            "requirement": ("1,218,514.8 per shot",) if pinned else ("not the pinned instance",),
            "measured": {"mean": row["mean"], "saving": saving(row),
                         "qubits": row["qubits"], "correct": correct(row)},
            "passed": (correct(row)
                       and abs(row["mean"] - 1_218_514.8) < 0.5) if pinned
                      else correct(row),
        })
    if "stacked" in measured and "everything" in measured and "control_gates" in measured:
        row = measured["stacked"]
        increment = saving(row) - saving(measured["everything"])
        declared = prediction.get("S5_stack", {}).get("predicted_saving_per_shot")
        checks.append({
            "check": "S5: the five-gate stack matches the prediction declared before it",
            "requirement": ("correct, the total within 1 Toffoli of the declared "
                            "prediction, and the prior arm re-measured at its own figure"),
            "measured": {"stacked": saving(row),
                         "everything": saving(measured["everything"]),
                         "control_gates": saving(measured["control_gates"]),
                         "increment": increment,
                         "declared_saving": declared,
                         "qubits": row["qubits"],
                         "eea_steps": row["eea_steps"]},
            "passed": (correct(row)
                       and declared is not None
                       and abs(saving(row) - declared) < 1.0
                       and abs(increment - saving(measured["control_gates"])) < 1.0),
        })
    if "everything" in measured:
        parts = [name for name in ("sweep", "shift", "products") if name in measured]
        checks.append({
            "check": "the movement gates and the product gate compose without interaction",
            "requirement": "everything-arm saving >= the sum of its parts - 1",
            "measured": {"everything": saving(measured["everything"]),
                         **{name: saving(measured[name]) for name in parts}},
            "passed": saving(measured["everything"]) >= (
                sum(saving(measured[name]) for name in parts) - 1.0),
        })
    if "composed" in measured:
        row = measured["composed"]
        checks.append({
            "check": "S3: composing the shift with the exchange is worth exactly zero",
            "requirement": "0.0 +- 0.05 per shot and correct",
            "measured": {"saving": saving(row),
                         "overlapping_rounds": overlap["overlapping_rounds"]},
            "passed": abs(saving(row)) < 0.05 and correct(row),
        })
    if "drop_shift_c3" in measured:
        row = measured["drop_shift_c3"]
        checks.append({
            "check": "S3b: the shift's c3 branch, measured",
            "requirement": "correct; the saving is reported against the declared 0.0",
            "measured": {"saving": saving(row),
                         "shift_chain_before": chain["shift"],
                         "shift_chain_after": row["costs"].get("shift_chain", 0.0)},
            "passed": correct(row),
        })
    if "both" in measured and "sweep" in measured and "shift" in measured:
        checks.append({
            "check": "the two movement gates compose without interaction",
            "requirement": "both-arm saving >= sweep saving + shift saving - 1",
            "measured": {"both": saving(measured["both"]),
                         "sweep": saving(measured["sweep"]),
                         "shift": saving(measured["shift"])},
            "passed": saving(measured["both"]) >= (saving(measured["sweep"])
                                                   + saving(measured["shift"]) - 1.0),
        })
    falsifier_rows = {}
    for name in falsifiers:
        if name not in measured:
            continue
        row = measured[name]
        # A falsifier only falsifies if it actually skipped work: when its wrong
        # condition never fires on this instance, the run says nothing either way
        # and is recorded as vacuous instead of as a pass.
        # A falsifier is informative when the wrong condition actually skips the
        # movement it conditions. The arm's total is not that measure: an arm that
        # computes the wrong state runs a different control flow, so it can cost more
        # while skipping most of its chain (the sweep falsifier does exactly that).
        if name == "falsify_zero_q":
            # Experiment B's falsifier skips *products*, so that is what is measured.
            reference_stream = (product_region(measured["products"])
                                if "products" in measured else product_region(control))
            chain_key = "product_stream_per_shot"
            skipped = max(0.0, reference_stream - product_region(row))
        else:
            chain_key = "selection_chain_per_shot" if name == "falsify_sweep_stale" \
                else "shift_chain_per_shot"
            before = chain["sweep"] if chain_key.startswith("selection") else chain["shift"]
            skipped = max(0.0, before - row["costs"].get(
                "selection_chain" if chain_key.startswith("selection") else "shift_chain",
                0.0))
        informative = skipped > 0.5
        falsifier_rows[name] = {"informative": informative, "skipped_per_shot": skipped,
                                "chain": chain_key, "correct": correct(row),
                                "arm_total_change": control["mean"] - row["mean"]}
        checks.append({
            "check": f"falsifier {name}: a wrong condition must break the arm",
            "requirement": "not correct when it skipped work; vacuous otherwise",
            "measured": {"syndromes_restored": row["syndromes"],
                         "phase_matches_reference": row["phase"],
                         "junk_clean": row["junk"],
                         "skipped_per_shot": skipped,
                         "informative": informative},
            "passed": (not correct(row)) if informative else True,
        })
    if "zero_q" in measured and "stacked" in measured:
        row = measured["zero_q"]
        delta = measured["stacked"]["mean"] - row["mean"]
        read_cost = row["read_instructions_per_shot"] - measured["stacked"][
            "read_instructions_per_shot"]
        checks.append({
            "check": "S6 (Experiment B): the product stream gated on a zero quotient",
            "requirement": ("correct, and a net saving over the incumbent after paying "
                            "for its own reads; a zero result closes the lever with "
                            "evidence rather than as a pass"),
            "measured": {"stacked_incumbent": measured["stacked"]["mean"],
                         "zero_q_arm": row["mean"],
                         "net_saving_over_incumbent": delta,
                         "extra_read_instructions": read_cost,
                         "product_stream_before": product_region(measured["products"])
                         if "products" in measured else None,
                         "product_stream_after": product_region(row),
                         "declared_saving": prediction.get("S6_zero_q", {})
                         .get("declared_saving_per_shot")},
            "passed": correct(row),
        })
    for name in ("falsify_sweep_stale", "falsify_shift_complement",
                 "falsify_shift_stale"):
        if name in falsifier_rows:
            checks.append({
                "check": f"falsifier {name} is informative on this instance",
                "requirement": "the wrong condition must have skipped work",
                "measured": falsifier_rows[name],
                "passed": falsifier_rows[name]["informative"],
            })

    failing = [check["check"] for check in checks if not check["passed"]]
    zero_q_delta = (measured["stacked"]["mean"] - measured["zero_q"]["mean"]
                    if "zero_q" in measured and "stacked" in measured else None)
    promoted = [name for name in args.variants
                if name not in ("control_gates",) and name in measured
                and correct(measured[name]) and saving(measured[name]) > 0]
    if "zero_q" in measured:
        # Experiment B is promoted only if it beats the incumbent it was built on.
        if zero_q_delta is not None and zero_q_delta <= 0:
            promoted = [name for name in promoted if name != "zero_q"]
    all_passed = not failing

    record = {
        "experiment": "read each movement predicate, then move: the shift/sweep/swap schedule",
        "declared_in": "docs/EEA_SCHEDULE_CAMPAIGN.md",
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_role": "local rebuild with MR and per-instruction execution counts "
                        "(scripts/build_kickmix_mr.py); not the pinned competition binary",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "seed": 20261028, "sim_seed": 99,
                     "pinned": pinned, "iterations": iterations},
        "policy": "and54",
        "levers": SIX_LEVERS,
        "prediction": prediction,
        "control": {
            "qubits": control["qubits"],
            "mean_toffoli_like": control["mean"],
            "chains": chain,
            "regions": control["regions"]["rows"],
            "correct": correct(control),
        },
        "round_overlap": overlap,
        "arms": {
            name: {
                "qubits": row["qubits"],
                "mean_toffoli_like": row["mean"],
                "saving_per_shot": saving(row),
                "saving_percent_of_arm": 100 * saving(row) / control["mean"],
                "selection_chain_per_shot": row["costs"].get("selection_chain", 0.0),
                "shift_chain_per_shot": row["costs"].get("shift_chain", 0.0),
                "product_toffoli_per_shot": product_region(row),
                "instructions_per_shot": row["instructions_per_shot"],
                "mid_circuit_reads_per_shot": row["mid_circuit_reads_per_shot"],
                "regions": row["regions"]["rows"],
                "correct": correct(row),
            }
            for name, row in measured.items()
        },
        "observation": {
            "control_total": control["mean"],
            "sweep_chain_saving": (chain["sweep"]
                                   - measured["sweep"]["costs"].get("selection_chain", 0.0)
                                   if "sweep" in measured else None),
            "sweep_arm_saving": saving(measured["sweep"]) if "sweep" in measured else None,
            "shift_chain_saving": (chain["shift"]
                                   - measured["shift"]["costs"].get("shift_chain", 0.0)
                                   if "shift" in measured else None),
            "shift_arm_saving": saving(measured["shift"]) if "shift" in measured else None,
            "shift_closed": ((chain["shift"]
                              - measured["shift"]["costs"].get("shift_chain", 0.0))
                             < S2_CLOSE_SHARE_OF_REGION * shift_region
                             if "shift" in measured else None),
            "composed_saving": saving(measured["composed"]) if "composed" in measured else None,
            "shift_c3_branch_saving": (saving(measured["drop_shift_c3"])
                                       if "drop_shift_c3" in measured else None),
            "product_saving": (product_region(control) - product_region(measured["products"])
                               if "products" in measured else None),
            "everything_saving": (saving(measured["everything"])
                                  if "everything" in measured else None),
            "stacked_saving": (saving(measured["stacked"])
                               if "stacked" in measured else None),
            "control_gates_saving": (saving(measured["control_gates"])
                                     if "control_gates" in measured else None),
            "control_gates_total": (measured["control_gates"]["mean"]
                                    if "control_gates" in measured else None),
            "stacked_predicted_saving": (prediction.get("S5_stack", {})
                                         .get("predicted_saving_per_shot")),
            "stacked_predicted_total": (prediction.get("S5_stack", {})
                                        .get("predicted_total_per_shot")),
            "stacked_error_per_shot": (saving(measured["stacked"])
                                       - prediction["S5_stack"]["predicted_saving_per_shot"]
                                       if "stacked" in measured
                                       and "S5_stack" in prediction else None),
            "control_instructions_per_shot": control["instructions_per_shot"],
            "everything_instructions_per_shot": (
                measured["everything"]["instructions_per_shot"]
                if "everything" in measured else None),
            "everything_mid_circuit_reads_per_shot": (
                measured["everything"]["mid_circuit_reads_per_shot"]
                if "everything" in measured else None),
            "promoted_arms": promoted,
        },
        "falsifiers": falsifier_rows,
        "promotion_checks": checks,
        "all_checks_passed": bool(all_passed),
        "failed_checks": failing,
        "verdict": (("PROMOTE " + ", ".join(promoted)) if all_passed and promoted
                    else ("REJECT: " + "; ".join(failing) if failing
                          else "no arm promoted")),
        "scope": {
            "is_official_result": False,
            "reason": "MR-gated arms cannot be compiled by the pinned competition binary; "
                      "this is a measured local figure, not a receipt",
        },
        "reproducibility": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
            "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
            "git_rev": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      check=False, capture_output=True,
                                      text=True).stdout.strip(),
        },
    }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"\ncontrol: Q={control['qubits']} T={control['mean']:.1f}/shot")
    for name in list(args.variants) + falsifiers:
        row = measured[name]
        print(f"{name:26s} T={row['mean']:.1f} saving={saving(row):+.1f} "
              f"correct={correct(row)}")
    for check in checks:
        print(f"  {'PASS' if check['passed'] else 'FAIL'}  {check['check']}")
    print(f"VERDICT: {record['verdict']}")
    print(f"record: {args.record}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

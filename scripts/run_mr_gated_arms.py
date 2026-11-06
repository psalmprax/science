#!/usr/bin/env python3
"""Run the MR-gated DQI arms on the local simulator, against the reference.

The gated optimisations in ``docs/PHOTONIC_ACCELERATOR_BOUNDARY.md`` §5.5 have only
ever been measured with their predicates supplied as **oracle** classical registers,
because ``kickmix`` has no Z-basis measurement. That measures what a skip is *worth*
and says nothing about whether reading the predicate for real leaves the state alone.
This script closes that gap on ``apedp_dyn/quantum_compilation/local_mr_sim.py``.

Each arm re-reads a value the reference circuit already carries: a single control
through a copy-and-measure gadget — ``cx(control, ancilla); mr(ancilla, bit)`` because
``mr`` clears its target and the reference still needs the control — or, for the Forney
block, the ``sigma`` register itself with ``mr`` followed by a classically controlled
``x`` that puts the value back. The bit then gates the dead work with
``push_condition``. Six levers are available:
``swap``, ``sweep``, ``forney``, ``clear``, ``shift`` and ``quotient``, where ``shift``
gates ``_shift_b_array_right`` on ``c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))`` using the flag *after*
``_update_flag`` — a flag read at the top of the round is stale and fails — and
``quotient`` gates each of the sweep's two polynomial divisions on its own counter bit
(``c0`` for the leading one, ``c3`` for the trailing one). The four counter bits are a
one-hot decode, so exactly one of them is set: the sweep gate says the round is live,
and the quotient gate says which of its two divisions is.

Correctness criterion: the arm must compute exactly the reference's function on the
same shots — same syndromes restored, every other qubit ``|0>``, same tracked phase.
Nothing here is a score: T counts still come from ``kickmix``.

``--clear-with r`` is the falsifier: ``r`` clears the ancilla like ``mr`` but
randomizes the tracked phase, so that arm must fail once any gating is applied.
``--shift-under`` is the shift arm's falsifier: it drops the ``¬flag ∧ (c0 ∨ c3)``
branch, skipping shifts the decoder needs, and must fail.

Run under the interpreter ``kickmix`` was built for (CPython 3.13), with numpy:

    .venv-kickmix/bin/python scripts/run_mr_gated_arms.py --arm shift --shots 32

(``uv venv .venv-kickmix --python python3.13 && uv pip install --python
.venv-kickmix/bin/python numpy``; the project ``.venv`` is 3.12 and cannot load
``kickmix.so``.)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
sys.path.insert(0, str(BUNDLE / "kickmix"))
sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))

import dqi_reference_circuit as DR  # noqa: E402
import kickmix as km  # noqa: E402
import verify_submission_dqi as V  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "local_mr_sim", ROOT / "apedp_dyn" / "quantum_compilation" / "local_mr_sim.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)


class _LocalKmShim:
    array = L.array
    GF2Field = km.GF2Field
    UnaryIterationCursor = L.Cursor
    CircuitBuilder = L.Builder


def _all_qubits(registers) -> list:
    """Flatten registers-of-registers into a single list of qubits."""
    out = []
    for register in registers:
        items = register if isinstance(register, (list, L.array)) else [register]
        for item in items:
            out.extend(item if isinstance(item, (list, L.array)) else [item])
    return out


class _conditional:
    """``with _conditional(builder, bit):`` — push_condition / pop_condition."""

    def __init__(self, builder, bit):
        self.builder = builder
        self.bit = bit

    def __enter__(self):
        self.builder.push_condition(self.bit)
        return self

    def __exit__(self, *_exc):
        self.builder.pop_condition()
        return False


class MRGatedDecoder(DR.ProosAndZalkaDecoder):
    """Reference decoder with levers gated on a real ``mr`` read."""

    def __init__(self, n_syndromes, field, *, gate_swap: bool = False,
                 gate_sweep: bool = False, clear_eea: bool = False,
                 gate_forney: bool = False, gate_shift: bool = False,
                 gate_quotient: bool = False,
                 shift_under: bool = False, clear_with: str = "mr"):
        super().__init__(n_syndromes, field)
        self.gate_swap = gate_swap
        self.gate_sweep = gate_sweep
        self.clear_eea = clear_eea
        self.gate_forney = gate_forney
        self.gate_shift = gate_shift
        self.gate_quotient = gate_quotient
        self.shift_under = shift_under
        self.clear_with = clear_with
        # One shared readout ancilla: `mr` clears its target, so every read in the
        # circuit can copy into the same qubit instead of allocating one per read.
        self._read_ancilla = None

    def _ancilla(self, builder):
        """The shared readout ancilla, allocated on first use."""
        if self._read_ancilla is None:
            self._read_ancilla = builder.alloc_qubits(1)[0]
        return self._read_ancilla

    # -- lever: skip the Forney body where sigma(alpha^j) != 0 --------------
    def _chien_search_and_forney_mbuc(self, builder, reg_b, error_mbuc, n_b, n_B):
        """Reference Forney phase, with the body gated on a measured zero-test.

        The body is dead whenever the evaluated locator is nonzero. After the fold,
        ``sigma`` holds ``sigma(alpha^j)`` in the computational basis, so the register is
        measured and put back with classically controlled ``x``: the zero test then costs
        two classical stores instead of a ``degree - 1``-Toffoli equality circuit at every
        position. The fold and its reverse still run — they are what moves the evaluation
        from position to position.
        """
        if not self.gate_forney:
            return super()._chien_search_and_forney_mbuc(builder, reg_b, error_mbuc, n_b, n_B)
        gf = self.field
        degree = gf.degree
        m = len(reg_b)
        t = self.n_syndromes // 2
        block_length = (1 << degree) - 1
        primitive = gf.primitive_element

        with builder.mark("pre_align_reg_b"):
            self._pre_align_reg_b(builder, reg_b, n_b, n_B, is_adjoint=False)

        sigma = reg_b[0]
        sigma_prime = reg_b[1]
        omega = reg_b[m // 2 + t]
        sigma_prime_inv = builder.alloc_qubits(degree)
        located = builder.alloc_qubits(degree)
        is_sigma_zero = builder.alloc_qubits(1)[0]
        is_error = builder.alloc_bits(1)
        # Loop-invariant scratch for reading `sigma` back: `mr` clears its target, so the
        # bits and the two predicate slots are all the state the read leaves behind.
        sigma_read = builder.alloc_bits(degree)
        sigma_nonzero = builder.alloc_bits(1)[0]
        sigma_is_zero = builder.alloc_bits(1)[0]

        omega_indices = range(m // 2 + 1, m // 2 + t + 1)
        step_multipliers = (
            [(k, gf.pow(primitive, k)) for k in range(1, t + 1)]
            + [(k, gf.pow(primitive, (m - 2 - k) % block_length)) for k in omega_indices
               if (m - 2 - k) % block_length != 0]
        )
        sum_pairs = (
            [(reg_b[k], omega) for k in range(m // 2 + 1, m // 2 + t)]
            + [(reg_b[k], sigma_prime) for k in range(3, t + 1, 2)]
            + [(reg_b[k], sigma) for k in range(2, t + 1, 2)]
            + [(sigma_prime, sigma)]
        )
        fwd_src = DR.km.array([q for src, _ in sum_pairs for q in src])
        fwd_dst = DR.km.array([q for _, dst in sum_pairs for q in dst])
        rev_src = DR.km.array([q for src, _ in reversed(sum_pairs) for q in src])
        rev_dst = DR.km.array([q for _, dst in reversed(sum_pairs) for q in dst])

        num_positions = min(block_length, len(error_mbuc))
        forney_step = builder.mark("forney_step")
        for j in range(num_positions):
            with forney_step:
                builder.bit_store(is_error[0], True, control=error_mbuc[j])
                builder.push_condition(is_error[0])

                builder.cx(fwd_src, fwd_dst)

                # Read sigma(alpha^j) out and put it straight back. The register holds a
                # computational-basis value, so `mr` is deterministic, `x` undoes it, and
                # the tracked phase is untouched: the predicate costs no Toffoli.
                sigma_zero = self._read_all_zero(
                    builder, sigma, sigma_read, sigma_nonzero, sigma_is_zero)

                with _conditional(builder, sigma_zero):
                    # Write the predicate into the qubit the AND needs, use it, uncompute.
                    builder.x(is_sigma_zero)
                    _, inverse_chain = builder.init_gf2_inverse_with_scaffold(
                        sigma_prime, target=sigma_prime_inv, field=gf)
                    builder.init_and(is_sigma_zero, omega, located)
                    builder.gf2_phase_by_product(located, sigma_prime_inv,
                                                 mask=error_mbuc[j], field=gf)
                    builder.del_and(is_sigma_zero, omega, located)
                    builder.del_gf2_inverse_with_scaffold(
                        sigma_prime, target=sigma_prime_inv, scaffold=inverse_chain, field=gf)
                    builder.free(inverse_chain)
                    builder.x(is_sigma_zero)
                builder.cx(rev_src, rev_dst)

                builder.pop_condition()
                builder.bit_store(is_error, False)

                if j + 1 < num_positions:
                    for k, step in step_multipliers:
                        builder.gf2_imul(step, target=reg_b[k], field=gf)

        if num_positions > 1:
            for k, step in step_multipliers:
                builder.gf2_idiv(gf.pow(step, num_positions - 1), target=reg_b[k], field=gf)

        builder.free(is_error)
        builder.free(located + sigma_prime_inv + DR.km.array([is_sigma_zero]))
        builder.free(sigma_read + DR.km.array([sigma_nonzero, sigma_is_zero]))

        with builder.mark("unalign_reg_b"):
            self._pre_align_reg_b(builder, reg_b, n_b, n_B, is_adjoint=True)

    # -- lever: clear the EEA instead of running its adjoint ---------------
    def build_syndrome_decoding_mbuc(self, builder, *, syndromes, error_mbuc, n_iters):
        """Reference construction, with the reverse EEA replaced by a measured clear.

        The forward EEA is its own uncomputation, so the reference pays for it twice.
        Here the syndromes are read out once up front, the EEA scratch is cleared with
        ``mr`` after the Forney phase, and the syndromes are re-prepared from the saved
        bits — the same trade the probe measured at 2.652×, with the readouts real.
        """
        if not self.clear_eea:
            return super().build_syndrome_decoding_mbuc(
                builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=n_iters)
        gf = self.field
        n = self.n_syndromes
        degree = gf.degree
        m = n + 3

        # Save the syndrome values before anything moves them, through the shared
        # ancilla rather than one fresh qubit per bit.
        saved = []
        for register in syndromes:
            bits = builder.alloc_bits(degree)
            for index, qubit in enumerate(register):
                builder.cx(qubit, self._ancilla(builder))
                builder.mr(self._ancilla(builder), bits[index])
            saved.append(bits)

        reg_a = [builder.alloc_qubits(degree) if i < 3 else syndromes[i - 3] for i in range(m)]
        reg_b = [builder.alloc_qubits(degree) for _ in range(m)]
        logm = m.bit_length() + 1
        n_A = builder.alloc_qubits(logm)
        n_B = builder.alloc_qubits(logm)
        n_a = builder.alloc_qubits(logm)
        n_b = builder.alloc_qubits(logm)
        n_q = builder.alloc_qubits(logm)
        flag = builder.alloc_qubits(1)[0]
        counter = builder.alloc_qubits(2)
        halting_counter = builder.alloc_qubits(n_iters.bit_length())
        eea = (reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag, counter, halting_counter)

        init_ones = DR.km.array([reg_a[0][0], reg_b[m - 1][0], n_a[0], flag])
        builder.x(init_ones)
        builder.gf2_iadd(n, target=n_A)
        builder.gf2_iadd(n + 1, target=n_B)

        with builder.mark("initial_shift"):
            self._initial_shift_loop(builder, reg_a, n_A, is_adjoint=False)
        with builder.mark("forward_eea"):
            self._synchronized_eea(builder, *eea, n_iters=n_iters, is_adjoint=False)
        with builder.mark("chien_forney_mbuc"):
            self._chien_search_and_forney_mbuc(builder, reg_b, error_mbuc, n_b, n_B)

        # Clear every EEA register instead of uncomputing it, then put the
        # syndromes back from their saved readout.
        with builder.mark("measured_clear"):
            throwaway = builder.alloc_bits(1)
            for qubit in _all_qubits([reg_a, reg_b, n_A, n_B, n_a, n_b, n_q,
                                      halting_counter, flag, counter]):
                builder.mr(qubit, throwaway[0])
            for register, bits in zip(syndromes, saved, strict=False):
                for index, qubit in enumerate(register):
                    with _conditional(builder, bits[index]):
                        builder.x(qubit)

        for reg in (halting_counter, counter, flag, n_q, n_b, n_a, n_B, n_A, *reg_a[:3], *reg_b):
            builder.free(reg)
        if self._read_ancilla is not None:
            builder.free(self._read_ancilla)
            self._read_ancilla = None

    def _read_bit(self, builder, qubit):
        """Copy `qubit` into the shared ancilla and measure it, leaving `qubit` intact.

        The ancilla is loop-invariant: it is cleared by the measurement, so the reads of
        one round reuse it instead of allocating a fresh qubit for every control.
        """
        ancilla = self._ancilla(builder)
        bit = builder.alloc_bits(1)
        builder.cx(qubit, ancilla)
        getattr(builder, self.clear_with)(ancilla, bit)
        return bit

    def _read_all_zero(self, builder, register, bits, nonzero, is_zero):
        """Measure a basis-state register, restore it, and return whether it was all zero.

        `mr` leaves its target at |0> and reports the value, and a classically controlled
        `x` puts the value back, so `register` and the tracked phase are unchanged. The
        predicate is then assembled from classical stores, which is `degree - 1` Toffolis
        cheaper than testing equality on the qubits (`degree` here).
        """
        for index, qubit in enumerate(register):
            getattr(builder, self.clear_with)(qubit, bits[index])
        for index, qubit in enumerate(register):
            with _conditional(builder, bits[index]):
                builder.x(qubit)
        builder.bit_store(nonzero, False)
        for bit in bits:
            builder.bit_store(nonzero, True, control=bit)
        builder.bit_store(is_zero, True)
        builder.bit_store(is_zero, False, control=nonzero)
        return is_zero

    # -- lever: skip the wholesale A/B exchange in rounds without c3 ----------
    def _swap_a_and_b(self, builder, reg_a, reg_b, n_A, n_B, n_a, n_b, flag, ctrl):
        if not self.gate_swap:
            return super()._swap_a_and_b(builder, reg_a, reg_b, n_A, n_B, n_a, n_b, flag, ctrl)
        with builder.mark("swap_a_and_b"):
            c3_bit = self._read_bit(builder, ctrl[3])
            with _conditional(builder, c3_bit[0]):
                swap_ctrl = builder.alloc_qubits(1)[0]
                # Inside the condition c3 == 1, so the reference's AND collapses to flag.
                builder.init_and(flag, swap_ctrl)
                lhs = DR.km.array([q for r in (n_A, n_a, *reg_a) for q in r])
                rhs = DR.km.array([q for r in (n_B, n_b, *reg_b) for q in r])
                builder.cswap(swap_ctrl, lhs, rhs)
                builder.del_and(flag, swap_ctrl)
                builder.free(swap_ctrl)

    # -- lever: skip the multiply sweep in rounds where c0 = c3 = 0 -----------
    def _multiply_and_add_or_sub(self, builder, reg_a, reg_b, n_A, n_a, n_q, ctrl, it, *,
                                 is_adjoint: bool):
        if not self.gate_sweep:
            return super()._multiply_and_add_or_sub(
                builder, reg_a, reg_b, n_A, n_a, n_q, ctrl, it, is_adjoint=is_adjoint)
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
            # `live_bit=True` (the ungated arm) means there is no condition to push.
            gate = _conditional(builder, live_bit) if live_bit is not True else nullcontext()
            with builder.mark("quotient_div"), gate:
                ratio, scaffold = builder.init_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], field=gf)
                builder.gf2_iadd(ratio, target=q_i, field=gf, control=control)
                builder.del_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], target=ratio, scaffold=scaffold, field=gf)
                builder.free(scaffold + ratio)

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
                        DR._xor_threshold(builder, cur_a, lim_a, mul, c3_hc, index + 1,
                                          at_least=True, lowest=1)
                        DR._xor_threshold(builder, cur_A, lim_A, mul, c0_hc, m - index,
                                          at_least=True, lowest=1)

                    prev = None
                    for j in order:
                        if prev is None:
                            endpoint(j)
                        else:
                            builder.cx(
                                DR.km.array([
                                    DR._active(cur_a, min(prev, j) + 1, lim_a),
                                    DR._active(cur_A, m - max(prev, j), lim_A),
                                ]),
                                mul,
                            )
                        prev = j

                        swaps = [a for a in (DR._active(eq_i, j, lim_q),
                                             DR._active(eq_ip1, j + 1, lim_q1))
                                 if a is not False]
                        multiplies = j < lim_a or j >= 2

                        if is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[j], target=reg_b[j], field=gf,
                                                 control=mul)
                        if len(swaps) == 1:
                            builder.cswap(swaps[0], q_i, reg_a[j])
                        elif swaps:
                            both = builder.alloc_qubits(1)[0]
                            builder.cx(DR.km.array(swaps), both)
                            builder.cswap(both, q_i, reg_a[j])
                            builder.cx(DR.km.array(swaps), both)
                            builder.free(both)
                        if not is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[j], target=reg_b[j], field=gf,
                                                 control=mul)

                    endpoint(prev)
                    builder.free(mul)

        # The whole step is dead when c0 = c3 = 0: the sweep is controlled on those two
        # bits, and the quotient divisions, the `n_a` add and its inverse all cancel.
        with builder.mark("multiply_and_add_or_sub"), _conditional(builder, live[0]):
            q_i = builder.alloc_qubits(gf.degree)
            leading, trailing = (0, m - 1) if is_adjoint else (m - 1, 0)
            # Each division is dead when its own counter bit is clear: the `gf2_iadd`
            # that consumes the quotient is already controlled on that bit, so the
            # division feeds nothing. Gate the division itself, not just the add.
            leading_bit = c3_bit[0] if is_adjoint else c0_bit[0]
            trailing_bit = c0_bit[0] if is_adjoint else c3_bit[0]
            if self.gate_quotient:
                accumulate_quotient(leading, c3_hc if is_adjoint else c0_hc, leading_bit)
            else:
                accumulate_quotient(leading, c3_hc if is_adjoint else c0_hc, True)
            builder.iadd(n_a, target=n_q)
            sweep_positions()
            builder.isub(n_a, target=n_q)
            if self.gate_quotient:
                accumulate_quotient(trailing, c0_hc if is_adjoint else c3_hc, trailing_bit)
            else:
                accumulate_quotient(trailing, c0_hc if is_adjoint else c3_hc, True)
            builder.free(q_i)


    # -- lever: skip the B-array shift where the post-update state cannot use it --
    def _shift_b_array_right(self, builder, reg_b, n_B, n_b, flag, ctrl, it, *,
                             is_adjoint):
        """Reference shift, gated on the predicate that makes its body live.

        The probe measured the body live on ``c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))`` with
        the post-update flag; gating on ``c1 ∨ c2 ∨ ¬flag`` fails on every shot
        because the flag read at the top of the round is stale. ``_update_flag`` has
        already run when the shift is called, so ``flag`` is the post-update value.
        """
        if not self.gate_shift:
            return super()._shift_b_array_right(
                builder, reg_b, n_B, n_b, flag, ctrl, it, is_adjoint=is_adjoint)
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
                # ¬flag ∧ (c0 ∨ c3): the branch that needs the flag to be clear. The
                # negation is a store of 0 under the flag, not a second store of 1 --
                # storing 1 twice leaves the bit constant and gates on c0 ∨ c3 alone.
                flag_clear = builder.alloc_bits(1)
                builder.bit_store(flag_clear[0], True)
                builder.bit_store(flag_clear[0], False, control=flag_bit[0])
                with _conditional(builder, flag_clear[0]):
                    builder.bit_store(live[0], True, control=c0_bit[0])
                    builder.bit_store(live[0], True, control=c3_bit[0])

            with _conditional(builder, live[0]):
                self._shift_body(builder, reg_b, n_B, n_b, flag, ctrl, it,
                                 is_adjoint=is_adjoint)

    @staticmethod
    def _shift_body(builder, reg_b, n_B, n_b, flag, ctrl, it, *, is_adjoint):
        """The reference shift body, verbatim, so only the gate can differ."""
        c0_hc, c1_hc, c2_hc, c3_hc = ctrl
        m = len(reg_b)
        lim_B = m - 2
        lim_b = min(m - 2, it // 2 + 1)

        builder.x(flag)
        nf = builder.alloc_qubits(2)
        builder.init_and(DR.km.array([c0_hc, c3_hc]), flag, nf)
        builder.x(flag)
        c0_nf, c3_nf = nf

        ctrl_nB = builder.alloc_qubits(1)[0]
        builder.cx(DR.km.array([c0_nf, c1_hc]), ctrl_nB)

        order = range(1, m) if is_adjoint else range(m - 1, 0, -1)
        with (
            builder.unary_iteration(n_B, control=ctrl_nB) as cur_B_ge,
            builder.unary_iteration(n_b, control=c2_hc) as cur_b_ge,
            builder.unary_iteration(n_B, control=c3_nf) as cur_B_lt,
        ):
            do_shift = builder.alloc_qubits(1)[0]

            def endpoint(i: int) -> None:
                DR._xor_threshold(builder, cur_B_ge, lim_B, do_shift, ctrl_nB, m - i + 1,
                                  at_least=True)
                DR._xor_threshold(builder, cur_b_ge, lim_b, do_shift, c2_hc, i,
                                  at_least=True)
                DR._xor_threshold(builder, cur_B_lt, lim_B, do_shift, c3_nf, m - i,
                                  at_least=False)

            prev = None
            for i in order:
                if prev is None:
                    endpoint(i)
                else:
                    builder.cx(
                        DR.km.array([
                            DR._active(cur_B_ge, m - max(prev, i) + 1, lim_B),
                            DR._active(cur_b_ge, min(prev, i), lim_b),
                            DR._active(cur_B_lt, m - max(prev, i), lim_B),
                        ]),
                        do_shift,
                    )
                prev = i
                builder.cswap(do_shift, reg_b[i - 1], reg_b[i])

            endpoint(prev)
            builder.free(do_shift)

        builder.cx(DR.km.array([c0_nf, c1_hc]), ctrl_nB)
        builder.free(ctrl_nB)
        builder.x(flag)
        builder.del_and(DR.km.array([c0_hc, c3_hc]), flag, nf)
        builder.x(flag)
        builder.free(nf)


def build(arm: str, clear_with: str, n: int, degree: int, m: int,
          shift_under: bool = False):
    original_km = DR.km
    DR.km = _LocalKmShim
    try:
        builder = L.Builder()
        decoder = MRGatedDecoder(
            n, km.GF2Field(degree),
            gate_swap=arm in ("swap", "all", "full"),
            # The quotient gate refines the sweep gate: the divisions sit inside the
            # gated step, so the arm that reads them implies it.
            gate_sweep=arm in ("sweep", "all", "full", "quotient"),
            clear_eea=arm in ("clear", "all", "full"),
            gate_forney=arm in ("forney", "all", "full"),
            gate_shift=arm in ("shift", "all", "full"),
            gate_quotient=arm in ("quotient", "full"),
            shift_under=shift_under,
            clear_with=clear_with,
        )
        syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
        error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
        decoder.build_syndrome_decoding_mbuc(
            builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=DR.default_n_iters(n))
        return builder.finish_circuit(), syndromes, error_mbuc
    finally:
        DR.km = original_km


def pack(values: np.ndarray, width: int) -> np.ndarray:
    rows, shots = values.shape
    out = np.zeros((rows * width, shots), dtype=bool)
    for row in range(rows):
        for i in range(width):
            out[row * width + i, :] = (values[row, :].astype(np.int64) >> i) & 1
    return out


def simulate(circuit, syndromes, error_mbuc, syndromes_in, mbuc_in, degree, shots):
    sim = L.Simulator(batch_size=shots, seed=99)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(syndromes, V.to_bits(syndromes_in, degree))
    sim.write_across_shots(error_mbuc, V.to_bits(mbuc_in, degree))
    sim.do(circuit)
    syndrome_ids = {q.id for reg in syndromes for q in reg}
    junk_ids = [i for i in range(circuit.num_qubits) if i not in syndrome_ids]
    return {
        "syndromes": sim.read_across_shots(syndromes),
        "junk": sim.read_across_shots([L.Q(i) for i in junk_ids]),
        "phase": sim.read_phase_flipped_across_shots(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm",
                        choices=("none", "swap", "sweep", "clear", "forney", "shift",
                                 "quotient", "all", "full"),
                        default="swap")
    parser.add_argument("--clear-with", choices=("mr", "r"), default="mr")
    parser.add_argument("--shift-under", action="store_true",
                        help="falsifier: drop the ¬flag∧(c0∨c3) branch from the gate")
    parser.add_argument("--m", type=int, default=15, help="number of MBUC positions")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    spec = V.ProblemSpecDQI.benchmark_instance(
        m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
    weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    tables = V.FieldTables(km.GF2Field(spec.degree))
    rng = np.random.default_rng(20261028)
    syndromes_in, mbuc_in, _phase = V.sample_shots(spec, tables, rng, args.shots, weights)

    ref_circuit, ref_syn, ref_mbuc = build("none", "mr", args.n, args.degree, args.m)
    arm_circuit, arm_syn, arm_mbuc = build(args.arm, args.clear_with, args.n, args.degree,
                                          args.m, shift_under=args.shift_under)

    reference = simulate(ref_circuit, ref_syn, ref_mbuc, syndromes_in, mbuc_in,
                         args.degree, args.shots)
    arm = simulate(arm_circuit, arm_syn, arm_mbuc, syndromes_in, mbuc_in,
                   args.degree, args.shots)

    result = {
        "arm": args.arm,
        "clear_with": args.clear_with,
        "instance": {"m": args.m, "n": args.n, "degree": args.degree, "shots": args.shots},
        "syndromes_match_reference": bool(np.array_equal(reference["syndromes"], arm["syndromes"])),
        "phase_matches_reference": bool(np.array_equal(reference["phase"], arm["phase"])),
        "syndromes_restored": bool(np.array_equal(
            arm["syndromes"], V.to_bits(syndromes_in, args.degree))),
        "junk_clean": bool(not arm["junk"].any()),
        "reference_junk_clean": bool(not reference["junk"].any()),
        "reference_qubits": ref_circuit.num_qubits,
        "arm_qubits": arm_circuit.num_qubits,
        "reference_phase_ones": int(reference["phase"].sum()),
        "arm_phase_ones": int(arm["phase"].sum()),
    }
    good = (result["syndromes_match_reference"] and result["phase_matches_reference"]
            and result["syndromes_restored"] and result["junk_clean"])
    result["verdict"] = "PASS" if good else "FAIL"
    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if good else 1

    print(f"arm={args.arm} read with {args.clear_with.upper()}  "
          f"m={args.m} n={args.n} GF(2^{args.degree}) shots={args.shots}")
    print(f"  qubits   reference={ref_circuit.num_qubits} arm={arm_circuit.num_qubits}")
    print(f"  syndromes match reference : {result['syndromes_match_reference']}")
    print(f"  phase matches reference   : {result['phase_matches_reference']}"
          f"  (ones: ref={result['reference_phase_ones']} arm={result['arm_phase_ones']})")
    print(f"  syndromes restored        : {result['syndromes_restored']}")
    print(f"  all other qubits |0>      : {result['junk_clean']}")
    print(f"VERDICT: {result['verdict']}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())

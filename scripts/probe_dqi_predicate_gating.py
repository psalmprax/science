#!/usr/bin/env python3
"""FEASIBILITY PROBE (not a submission): what does predicate-gated execution buy?

The DQI reference circuit runs the Forney body at every one of the ``m`` positions,
although the body is semantically dead wherever ``sigma(alpha^j) != 0``: the
inversion's output is multiplied by zero (``located = is_sigma_zero & omega``)
and then uncomputed, so skipping the body at a non-root leaves the state
unchanged.  Gating it should therefore be free in correctness and should remove
that body from every non-root position.

KICKMIX can only skip work inside ``push_condition``, which needs a *classical*
bit, and the only readout primitive exposed is ``hmr`` -- an **X-basis**
measurement.  A computational-basis predicate cannot be read out: ``hmr`` on
``|1>`` returns a random bit (measured below), ``push_condition`` rejects a
qubit, ``bit_store`` rejects a quantum control, and the gate set
``{X, Z, R, CX, CZ, SWAP, CSWAP, CCX, CCZ}`` is closed under the Z-diagonal
group, so no Hadamard can be synthesized to conjugate the predicate into the
X basis.

So the optimisation cannot be built in today's IR, but its *value* can be
measured exactly: this script builds the reference circuit plus one classical
**oracle** register holding the true ``sigma(alpha^j) == 0`` bits for each shot,
gates the Forney body on it, and reports executed Toffoli-like gates per shot on
the same shots.  The oracle register is ground truth, so the result is the
executed cost a Z-basis readout (``MR``) would deliver -- an upper bound, and
not a legal submission.

Usage:
    python scripts/probe_dqi_predicate_gating.py [--m 15] [--n 6] [--degree 4] [--shots 64]
    python scripts/probe_dqi_predicate_gating.py --m 4095 --n 70 --degree 12 --shots 128 \
        --record benchmarks/quantum_compilation/dqi_predicate_gating.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
_KICKMIX = REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"
_DQI = REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dqi"
for _p in (str(_KICKMIX), str(_DQI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kickmix as km  # noqa: E402

import dqi_reference_circuit as DR  # noqa: E402
from verify_submission_dqi import (  # noqa: E402
    FieldTables,
    ProblemSpecDQI,
    default_num_accepted_values,
    optimal_dqi_weights,
    to_bits,
)


def sample_shots_with_roots(spec, tables, rng, num_shots, weight_probs):
    """Verbatim copy of `verify_submission_dqi.sample_shots`, also returning the error positions.

    The locator polynomial has a root exactly at each error position, so `roots[shot, j]`
    is the truth table that a Z-basis readout of `sigma(alpha^j) == 0` would produce.
    """
    max_value = (1 << tables.degree) - 1
    weights = rng.choice(len(weight_probs), size=num_shots, p=weight_probs)
    mbuc = rng.integers(0, max_value, size=(num_shots, spec.m), endpoint=True, dtype=np.int64)
    syndromes = np.zeros((num_shots, spec.n), dtype=np.int64)
    phases = np.zeros(num_shots, dtype=bool)
    roots = np.zeros((num_shots, spec.m), dtype=bool)
    rows = np.arange(1, spec.n + 1, dtype=np.int64)[:, None]

    for shot in np.flatnonzero(weights):
        k = int(weights[shot])
        pos = rng.choice(spec.m, size=k, replace=False)
        vals = rng.integers(1, max_value, size=k, endpoint=True, dtype=np.int64)
        exps = (tables.log[vals][None, :] + rows * pos[None, :]) % tables.order
        syndromes[shot] = np.bitwise_xor.reduce(tables.antilog[exps], axis=1)
        phases[shot] = sum(int(x).bit_count() for x in vals & mbuc[shot, pos]) & 1 == 1
        roots[shot, pos] = True
    return syndromes, mbuc, phases, roots


def registers_for(circuit, spec):
    """Same as `verify_submission_dqi.validate_registers`, but tolerating the extra oracle register."""
    registers = circuit.register_data
    syndrome_qubits = [q for i in range(spec.n) for q in registers[f'S{i}']]
    mbuc_bits = [b for j in range(spec.m) for b in registers[f'E{j}']]
    syndrome_set = set(syndrome_qubits)
    ancilla_qubits = km.array([km.q(k) for k in range(circuit.num_qubits) if km.q(k) not in syndrome_set])
    return syndrome_qubits, mbuc_bits, ancilla_qubits


class InstrumentedDecoder(DR.ProosAndZalkaDecoder):
    """Reference decoder with either instrumented arm switched on.

    `root_bits`: per-position classical bits holding the truth table of
        `sigma(alpha^j) == 0`, used to gate the Forney body.
    `syndrome_bits`: one classical register per syndrome, holding a readout of the
        input syndromes taken *before* the EEA.  With them the reverse EEA and the
        initial-shift uncompute are replaced by `reset` (R, zero Toffoli) plus a
        classically controlled re-preparation: measure-copy-restore instead of
        compute-copy-uncompute.
    """

    def __init__(self, *args, root_bits=None, syndrome_bits=None, phase_bit=None,
                 clear_bits=None, scratch_sink=None, round_probe=None,
                 sweep_bits=None, shift_bits=None, swap_bits=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_bits = root_bits
        self.syndrome_bits = syndrome_bits
        self.phase_bit = phase_bit
        self.clear_bits = clear_bits
        self.scratch_sink = scratch_sink
        self.round_probe = round_probe
        self.sweep_bits = sweep_bits
        self.shift_bits = shift_bits
        self.swap_bits = swap_bits

    def _synchronized_eea(self, builder, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag,
                          counter, halting_counter, *, n_iters, is_adjoint):
        """The reference EEA loop, with two hooks: a per-round `(c0, c3)` readout and a
        per-round gate on the multiply-accumulate sweep.

        Every use of `ctrl` inside `_multiply_and_add_or_sub` is gated by `c0_hc` or
        `c3_hc`, so in a round where both are zero that whole sweep -- 63 Toffoli per
        position, the single most expensive step in the decoder -- is emitted and does
        nothing. Hook 1 records `(c0_hc, c3_hc)` per round so hook 2 can skip it.
        """
        eea_step = builder.mark('eea_step')
        for it in (range(n_iters - 1, -1, -1) if is_adjoint else range(n_iters)):
            with eea_step:
                if is_adjoint:
                    self._update_counter(builder, counter, flag, is_adjoint=True)

                is_c3 = builder.alloc_qubits(1)[0]
                builder.init_and(counter[0], counter[1], is_c3)
                if is_adjoint:
                    self._update_halting_counter(builder, n_A, n_B, n_b, halting_counter,
                                                 flag, is_c3, is_adjoint=True)

                is_hc_zero = builder.alloc_qubits(1)[0]
                builder.flip_if_equal(halting_counter, 0, target=is_hc_zero)
                ctrl = self._counter_controls(builder, counter, is_c3, is_hc_zero)
                probing = self.round_probe is not None and not is_adjoint
                if probing:
                    builder.cx(
                        km.array([ctrl[0], ctrl[1], ctrl[2], ctrl[3], flag]),
                        km.array(self.round_probe[6 * it:6 * it + 5]))

                def probe_flag() -> None:
                    # `_update_flag` toggles `flag`, and the shift step reads the *new*
                    # value, so a probe taken at the top of the round is stale.
                    if probing:
                        builder.cx(flag, self.round_probe[6 * it + 5])

                def gated(bits, step) -> None:
                    live = bits is not None and not is_adjoint
                    if live:
                        builder.push_condition(bits[it])
                    step()
                    if live:
                        builder.pop_condition()

                steps = [
                    lambda: self._update_flag(builder, reg_b, n_A, n_B, n_a, n_b, n_q,
                                              flag, ctrl),
                    probe_flag,
                    lambda: gated(self.sweep_bits, lambda: self._multiply_and_add_or_sub(
                        builder, reg_a, reg_b, n_A, n_a, n_q, ctrl, it, is_adjoint=is_adjoint)),
                    lambda: gated(self.shift_bits, lambda: self._shift_b_array_right(
                        builder, reg_b, n_B, n_b, flag, ctrl, it, is_adjoint=is_adjoint)),
                    lambda: self._update_metadata(builder, n_B, n_b, n_q, flag, ctrl,
                                                  is_adjoint=is_adjoint),
                    lambda: gated(self.swap_bits, lambda: self._swap_a_and_b(
                        builder, reg_a, reg_b, n_A, n_B, n_a, n_b, flag, ctrl)),
                ]
                for step in (reversed(steps) if is_adjoint else steps):
                    step()

                self._del_counter_controls(builder, counter, is_c3, is_hc_zero, ctrl)
                builder.flip_if_equal(halting_counter, 0, target=is_hc_zero)
                builder.free(is_hc_zero)

                if not is_adjoint:
                    self._update_halting_counter(builder, n_A, n_B, n_b, halting_counter,
                                                 flag, is_c3, is_adjoint=False)
                builder.del_and(counter[0], counter[1], is_c3)
                builder.free(is_c3)
                if not is_adjoint:
                    self._update_counter(builder, counter, flag, is_adjoint=False)

    @property
    def skip_phase(self) -> bool:
        """The Forney phase is accumulated classically instead of into the state."""
        return self.phase_bit is not None

    def build_syndrome_decoding_mbuc(self, builder, *, syndromes, error_mbuc, n_iters):
        if self.syndrome_bits is None:
            return super().build_syndrome_decoding_mbuc(
                builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=n_iters)

        gf = self.field
        n, degree = self.n_syndromes, gf.degree
        m = n + 3
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

        init_ones = km.array([reg_a[0][0], reg_b[m - 1][0], n_a[0], flag])
        builder.x(init_ones)
        builder.gf2_iadd(n, target=n_A)
        builder.gf2_iadd(n + 1, target=n_B)

        with builder.mark('initial_shift'):
            self._initial_shift_loop(builder, reg_a, n_A, is_adjoint=False)
        with builder.mark('forward_eea'):
            self._synchronized_eea(builder, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag,
                                   counter, halting_counter, n_iters=n_iters, is_adjoint=False)
        with builder.mark('chien_forney_mbuc'):
            self._chien_search_and_forney_mbuc(builder, reg_b, error_mbuc, n_b, n_B)

        scratch = [q for reg in (reg_a[0], reg_a[1], reg_a[2], *reg_b, n_A, n_B, n_a, n_b,
                                 n_q, flag, counter, halting_counter) for q in (reg if isinstance(reg, km.array) else [reg])]
        syndrome_qubits = [q for s in syndromes for q in s]
        if self.scratch_sink is not None:
            # Pass 1: stop here and hand the live state out for a value readout.
            self.scratch_sink.extend(scratch + syndrome_qubits)
            return
        with builder.mark('clear_scratch'):
            if self.clear_bits is not None:
                # MR: measure each qubit in Z, then clear it with a classically
                # controlled X (zero Toffoli, phase-safe) -- unlike `R`, which
                # randomizes the tracked phase (measured below).
                builder.cx(km.array(self.clear_bits), km.array(scratch + syndrome_qubits))
            else:
                builder.reset(km.array(scratch))
                builder.reset(km.array(syndrome_qubits))
        with builder.mark('restore_syndromes'):
            for s, bits in zip(syndromes, self.syndrome_bits):
                builder.cx(km.array(bits), s)
        if self.phase_bit is not None:
            # The phase must be the last thing applied: `R` clears it (measured below).
            with builder.mark('classical_phase'):
                builder.neg_if(self.phase_bit)

        for reg in (halting_counter, counter, flag, n_q, n_b, n_a, n_B, n_A, *reg_a[:3], *reg_b):
            builder.free(reg)

    def _chien_search_and_forney_mbuc(self, builder, reg_b, error_mbuc, n_b, n_B):
        gf = self.field
        degree = gf.degree
        m = len(reg_b)
        t = self.n_syndromes // 2
        block_length = (1 << degree) - 1
        primitive = gf.primitive_element

        with builder.mark('pre_align_reg_b'):
            self._pre_align_reg_b(builder, reg_b, n_b, n_B, is_adjoint=False)

        sigma = reg_b[0]
        sigma_prime = reg_b[1]
        omega = reg_b[m // 2 + t]
        sigma_prime_inv = builder.alloc_qubits(degree)
        located = builder.alloc_qubits(degree)
        is_sigma_zero = builder.alloc_qubits(1)[0]
        is_error = builder.alloc_bits(1)

        omega_indices = range(m // 2 + 1, m // 2 + t + 1)
        step_multipliers = [(k, gf.pow(primitive, k)) for k in range(1, t + 1)] + [
            (k, gf.pow(primitive, (m - 2 - k) % block_length))
            for k in omega_indices
            if (m - 2 - k) % block_length != 0
        ]

        sum_pairs = (
            [(reg_b[k], omega) for k in range(m // 2 + 1, m // 2 + t)]
            + [(reg_b[k], sigma_prime) for k in range(3, t + 1, 2)]
            + [(reg_b[k], sigma) for k in range(2, t + 1, 2)]
            + [(sigma_prime, sigma)]
        )
        fwd_src = km.array([q for src, _ in sum_pairs for q in src])
        fwd_dst = km.array([q for _, dst in sum_pairs for q in dst])
        rev_src = km.array([q for src, _ in reversed(sum_pairs) for q in src])
        rev_dst = km.array([q for _, dst in reversed(sum_pairs) for q in dst])

        num_positions = min(block_length, len(error_mbuc))
        forney_step = builder.mark('forney_step')
        for j in range(num_positions):
            with forney_step:
                builder.bit_store(is_error[0], True, control=error_mbuc[j])
                builder.push_condition(is_error[0])
                if self.root_bits is not None:
                    builder.push_condition(self.root_bits[j])

                builder.cx(fwd_src, fwd_dst)
                builder.flip_if_equal(sigma, 0, target=is_sigma_zero)

                _, inverse_chain = builder.init_gf2_inverse_with_scaffold(
                    sigma_prime, target=sigma_prime_inv, field=gf)

                builder.init_and(is_sigma_zero, omega, located)
                if not self.skip_phase:
                    builder.gf2_phase_by_product(located, sigma_prime_inv,
                                                 mask=error_mbuc[j], field=gf)
                builder.del_and(is_sigma_zero, omega, located)

                builder.del_gf2_inverse_with_scaffold(
                    sigma_prime, target=sigma_prime_inv, scaffold=inverse_chain, field=gf)
                builder.free(inverse_chain)

                builder.flip_if_equal(sigma, 0, target=is_sigma_zero)
                builder.cx(rev_src, rev_dst)

                if self.root_bits is not None:
                    builder.pop_condition()
                builder.pop_condition()
                builder.bit_store(is_error, False)

                if j + 1 < num_positions:
                    for k, step in step_multipliers:
                        builder.gf2_imul(step, target=reg_b[k], field=gf)

        if num_positions > 1:
            for k, step in step_multipliers:
                builder.gf2_idiv(gf.pow(step, num_positions - 1), target=reg_b[k], field=gf)

        builder.free(is_error)
        builder.free(located + sigma_prime_inv + km.array([is_sigma_zero]))

        with builder.mark('unalign_reg_b'):
            self._pre_align_reg_b(builder, reg_b, n_b, n_B, is_adjoint=True)


def build(spec: ProblemSpecDQI, *, gated: bool = False, restore: bool = False,
          classical_phase: bool = False):
    """Builds the reference circuit, optionally with the root-gate and measure-restore arms."""
    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(spec.degree, name=f'S{i}') for i in range(spec.n)]
    error_mbuc = [builder.create_classical_register(spec.degree, name=f'E{j}') for j in range(spec.m)]
    root_bits = (
        [builder.create_classical_register(1, name=f'R{j}')[0]
         for j in range(min((1 << spec.degree) - 1, spec.m))]
        if gated else None
    )
    syndrome_bits = (
        [builder.create_classical_register(spec.degree, name=f'C{i}') for i in range(spec.n)]
        if restore else None
    )
    phase_bit = (builder.create_classical_register(1, name='P')[0]
                 if classical_phase else None)
    decoder = InstrumentedDecoder(spec.n, km.GF2Field(spec.degree),
                                  root_bits=root_bits, syndrome_bits=syndrome_bits,
                                  phase_bit=phase_bit)
    decoder.build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=DR.default_n_iters(spec.n))
    return builder.finish_circuit(), root_bits, syndrome_bits, phase_bit


ALL_ARMS = (
    ('baseline', False, False, False),
    ('gated', True, False, False),
    ('restore', False, True, False),
    ('restore_gated', True, True, False),
    ('classical_phase', False, True, True),
    ('classical_phase_gated', True, True, True),
)


def measure(spec: ProblemSpecDQI, seed: int = 20261028, arms=ALL_ARMS) -> dict:
    """Runs the reference and the oracle-gated variant on the same shots; returns both cost records."""
    tables = FieldTables(km.GF2Field(spec.degree))
    weight_probs = optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    syndromes, mbuc, phases, roots = sample_shots_with_roots(
        spec, tables, np.random.default_rng(seed), spec.num_shots, weight_probs)

    record = {
        'm': spec.m, 'n': spec.n, 'degree': spec.degree, 'r': spec.r,
        'num_shots': spec.num_shots,
        'arms': [a[0] for a in arms],
        'mean_roots_per_shot': float(roots[:, :min((1 << spec.degree) - 1, spec.m)].sum(axis=1).mean()),
    }
    record['root_density'] = record['mean_roots_per_shot'] / spec.m

    for name, gated, restore, classical_phase in arms:
        circuit, root_bits, syndrome_bits, phase_bit = build(
            spec, gated=gated, restore=restore, classical_phase=classical_phase)
        syndrome_qubits, mbuc_bits, ancilla_qubits = registers_for(circuit, spec)
        sim = km.Simulator(batch_size=spec.num_shots, debug_print=False,
                           counting_operations=True, seed=12345)
        sim.use_same_registers_as(circuit)
        input_syndromes = to_bits(syndromes, spec.degree)
        sim.write_across_shots(km.array(syndrome_qubits), input_syndromes)
        sim.write_across_shots(km.array(mbuc_bits), to_bits(mbuc, spec.degree))
        if gated:
            sim.write_across_shots(km.array(root_bits), roots[:, :len(root_bits)].T)
        if restore:
            sim.write_across_shots(km.array([q for reg in syndrome_bits for q in reg]),
                                   input_syndromes)
        if classical_phase:
            sim.write_across_shots(km.array([phase_bit]), phases.astype(bool)[None, :])
        sim.do(circuit)

        phase_flipped = sim.read_phase_flipped_across_shots()
        reg_diff = sim.read_across_shots(km.array(syndrome_qubits)) ^ input_syndromes
        junk = (sim.read_across_shots(ancilla_qubits) if len(ancilla_qubits)
                else np.zeros((0, spec.num_shots), dtype=bool))
        val_err = reg_diff.any(axis=0) | junk.any(axis=0)
        phase_err = (phase_flipped ^ phases) & ~val_err
        record[name] = {
            'toffoli_per_shot': int(sim.operation_statistics().num_toffoli_like) / spec.num_shots,
            'static_toffoli': circuit.operation_statistics().num_toffoli_like,
            'num_qubits': circuit.num_qubits,
            'value_errors': int(val_err.sum()),
            'phase_errors': int(phase_err.sum()),
        }

    base = record['baseline']
    for name in (a[0] for a in arms if a[0] != 'baseline'):
        arm = record[name]
        record[f'toffoli_ratio_{name}'] = base['toffoli_per_shot'] / arm['toffoli_per_shot']
        record[f'semantics_preserved_{name}'] = arm['value_errors'] == 0 and arm['phase_errors'] == 0
    record['toffoli_ratio'] = record['toffoli_ratio_gated']
    record['forney_body_share'] = ((base['toffoli_per_shot'] - record['gated']['toffoli_per_shot'])
                                   / base['toffoli_per_shot'])
    record['semantics_preserved'] = record['semantics_preserved_gated']
    return record


def measure_mr_clear(spec: ProblemSpecDQI, seed: int = 20261028,
                     gate_sweep: bool = False, gate_shift: bool = False,
                     gate_swap: bool = False) -> dict:
    """Two-pass oracle for an MR-based clear: pass 1 reads the live scratch, pass 2 clears it with X."""
    tables = FieldTables(km.GF2Field(spec.degree))
    weight_probs = optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    syndromes, mbuc, phases, roots = sample_shots_with_roots(
        spec, tables, np.random.default_rng(seed), spec.num_shots, weight_probs)
    input_syndromes = to_bits(syndromes, spec.degree)

    n_iters = DR.default_n_iters(spec.n)

    def make(*, scratch_sink=None, clear_bits=False, scratch_count=0,
             probe_rounds=None, sweep_bits=None, shift_bits=None, swap_bits=None):
        builder = km.CircuitBuilder()
        syn = [builder.create_quantum_register(spec.degree, name=f'S{i}') for i in range(spec.n)]
        err = [builder.create_classical_register(spec.degree, name=f'E{j}') for j in range(spec.m)]
        root_bits = [builder.create_classical_register(1, name=f'R{j}')[0]
                     for j in range(min((1 << spec.degree) - 1, spec.m))]
        syn_bits = [builder.create_classical_register(spec.degree, name=f'C{i}')
                    for i in range(spec.n)]
        clear = ([builder.create_classical_register(1, name=f'K{k}')[0]
                  for k in range(scratch_count)] if clear_bits else None)
        round_probe = list(builder.alloc_qubits(6 * n_iters)) if probe_rounds else None

        def bits(prefix, wanted):
            return ([builder.create_classical_register(1, name=f'{prefix}{k}')[0]
                     for k in range(n_iters)] if wanted else None)

        sweep, shift, swap = bits('W', sweep_bits), bits('H', shift_bits), bits('V', swap_bits)
        decoder = InstrumentedDecoder(spec.n, km.GF2Field(spec.degree), root_bits=root_bits,
                                      syndrome_bits=syn_bits, clear_bits=clear,
                                      scratch_sink=scratch_sink, round_probe=round_probe,
                                      sweep_bits=sweep, shift_bits=shift, swap_bits=swap)
        decoder.build_syndrome_decoding_mbuc(
            builder, syndromes=syn, error_mbuc=err, n_iters=DR.default_n_iters(spec.n))
        extras = {}
        if clear_bits:
            extras['clear'] = clear
        for name, value in (('sweep', sweep), ('shift', shift), ('swap', swap)):
            if value is not None:
                extras[name] = value
        if probe_rounds:
            extras['probe'] = round_probe
        return builder.finish_circuit(), root_bits, syn_bits, extras

    # Pass 1: read the scratch values and the per-round control probe after the Forney block.
    scratch: list = []
    circuit, root_bits, syn_bits, _ = make(scratch_sink=scratch)
    sim1 = km.Simulator(batch_size=spec.num_shots, debug_print=False, seed=12345)
    sim1.use_same_registers_as(circuit)
    syndrome_qubits1, mbuc_bits1, _ = registers_for(circuit, spec)
    sim1.write_across_shots(km.array(syndrome_qubits1), input_syndromes)
    sim1.write_across_shots(km.array(mbuc_bits1), to_bits(mbuc, spec.degree))
    sim1.write_across_shots(km.array(root_bits), roots[:, :len(root_bits)].T)
    sim1.write_across_shots(km.array([q for reg in syn_bits for q in reg]), input_syndromes)
    sim1.do(circuit)
    clear_values = np.asarray(sim1.read_across_shots(km.array(scratch)), dtype=bool)

    def feed(simulator, circuit, root_bits, syn_bits):
        syndrome_qubits, mbuc_bits, ancilla_qubits = registers_for(circuit, spec)
        simulator.use_same_registers_as(circuit)
        simulator.write_across_shots(km.array(syndrome_qubits), input_syndromes)
        simulator.write_across_shots(km.array(mbuc_bits), to_bits(mbuc, spec.degree))
        simulator.write_across_shots(km.array(root_bits), roots[:, :len(root_bits)].T)
        simulator.write_across_shots(km.array([q for reg in syn_bits for q in reg]),
                                     input_syndromes)
        return syndrome_qubits, ancilla_qubits

    gating = gate_sweep or gate_shift or gate_swap
    rows = None
    if gating:
        # Pass 1b: (c0_hc, c1_hc, c2_hc, c3_hc, flag) at the top of every round, from the
        # same build as the scratch readout below.
        circuit, root_bits, syn_bits, extras = make(scratch_sink=[], probe_rounds=True)
        simr = km.Simulator(batch_size=spec.num_shots, debug_print=False, seed=12345)
        feed(simr, circuit, root_bits, syn_bits)
        simr.do(circuit)
        rows = np.asarray(simr.read_across_shots(km.array(extras['probe'])),
                          dtype=bool).reshape(n_iters, 6, spec.num_shots)

    # Pass 2: clear each scratch qubit with a classically controlled X on its measured value,
    # and skip every step whose controls are zero in this round.
    circuit, root_bits, syn_bits, extras = make(
        clear_bits=True, scratch_count=len(scratch), sweep_bits=gating,
        shift_bits=gate_shift, swap_bits=gate_swap)
    clear = extras['clear']
    sim = km.Simulator(batch_size=spec.num_shots, debug_print=False,
                       counting_operations=True, seed=12345)
    syndrome_qubits, ancilla_qubits = feed(sim, circuit, root_bits, syn_bits)
    sim.write_across_shots(km.array(clear), clear_values)
    if gating:
        c0, c1, c2, c3 = (rows[:, k] for k in range(4))
        flag_after = rows[:, 5]
        # `_shift_b_array_right` is live on `ctrl_nB ∨ c2_hc ∨ c3_nf`, with
        # ctrl_nB = (c0 ∧ ¬flag) ⊕ c1 and c3_nf = c3 ∧ ¬flag: that is
        # c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3)), using the flag *after* `_update_flag`.
        dead = {'sweep': c0 | c3,
                'shift': c1 | c2 | (~flag_after & (c0 | c3)),
                'swap': c3}
        for name in ('sweep', 'shift', 'swap'):
            if name in extras:
                sim.write_across_shots(km.array(extras[name]), dead[name])
    sim.do(circuit)

    phase_flipped = sim.read_phase_flipped_across_shots()
    reg_diff = sim.read_across_shots(km.array(syndrome_qubits)) ^ input_syndromes
    junk = (sim.read_across_shots(ancilla_qubits) if len(ancilla_qubits)
            else np.zeros((0, spec.num_shots), dtype=bool))
    val_err = reg_diff.any(axis=0) | junk.any(axis=0)
    phase_err = (phase_flipped ^ phases) & ~val_err
    return {
        'm': spec.m, 'n': spec.n, 'degree': spec.degree, 'r': spec.r,
        'num_shots': spec.num_shots,
        'toffoli_per_shot': int(sim.operation_statistics().num_toffoli_like) / spec.num_shots,
        'static_toffoli': circuit.operation_statistics().num_toffoli_like,
        'num_qubits': circuit.num_qubits,
        'value_errors': int(val_err.sum()),
        'phase_errors': int(phase_err.sum()),
        'cleared_qubits': len(scratch),
        'clear_bits_needed': len(clear),
        'sweep_gated': bool(gating),
        'shift_gated': bool(gate_shift),
        'swap_gated': bool(gate_swap),
        'round_bits_needed': int(6 * n_iters) if gating else 0,
        'live_fraction_sweep': float(dead['sweep'].mean()) if gating else 0.0,
        'live_fraction_shift': float(dead['shift'].mean()) if gating else 0.0,
        'live_fraction_swap': float(dead['swap'].mean()) if gating else 0.0,
        'dead_sweep_fraction': float(1.0 - dead['sweep'].mean()) if gating else 0.0,
        'dead_shift_fraction': float(1.0 - dead['shift'].mean()) if gating else 0.0,
        'dead_swap_fraction': float(1.0 - dead['swap'].mean()) if gating else 0.0,
    }


def probe_readout(circuit_type=km.CircuitBuilder) -> dict:
    """Shows that no classical bit can be derived from a computational-basis predicate."""
    findings = {}
    b = circuit_type()
    pred, bit = b.alloc_qubits(1)[0], b.alloc_bits(1)[0]
    b.x(pred)
    b.hmr(pred, bit)
    circuit = b.finish_circuit()
    sim = km.Simulator(batch_size=64, debug_print=False, seed=1)
    sim.use_same_registers_as(circuit)
    sim.do(circuit)
    outcomes = sim.read_across_shots(bit).ravel()
    findings['hmr_on_one_ones_count'] = int(outcomes.sum())
    findings['hmr_on_one_shots'] = int(outcomes.size)

    for label, fn in (
        ('push_condition_accepts_qubit', lambda: b.push_condition(pred)),
        ('bit_store_accepts_quantum_control', lambda: b.bit_store(bit, True, control=pred)),
    ):
        try:
            fn()
            findings[label] = 'ACCEPTED'
        except (TypeError, ValueError):
            findings[label] = 'REJECTED'

    # `R` is not a phase-safe clear: a circuit with no phase in it at all still
    # reports a random phase_flipped once a reset is present.
    findings['reset_only'] = _phase_ones(16, lambda bb: (bb.x(bb.alloc_qubits(1)[0]),))
    findings['reset_after_x'] = _phase_ones(
        16, lambda bb: (lambda q: (bb.x(q), bb.reset(q)))(bb.alloc_qubits(1)[0]))
    findings['no_reset_after_x'] = _phase_ones(16, lambda bb: bb.x(bb.alloc_qubits(1)[0]))
    return findings


def _phase_ones(batch: int, build) -> int:
    """Number of shots reporting a flipped phase for a freshly built phase-free circuit."""
    builder = km.CircuitBuilder()
    build(builder)
    circuit = builder.finish_circuit()
    sim = km.Simulator(batch_size=batch, debug_print=False, seed=7)
    sim.use_same_registers_as(circuit)
    sim.do(circuit)
    return int(sim.read_phase_flipped_across_shots().ravel().sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=15)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--degree', type=int, default=4)
    ap.add_argument('--r', type=int, default=None)
    ap.add_argument('--shots', type=int, default=64)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--record', type=str, default=None)
    args = ap.parse_args()

    r = args.r if args.r is not None else default_num_accepted_values(args.degree)
    spec = ProblemSpecDQI(m=args.m, n=args.n, degree=args.degree, r=r, num_shots=args.shots)
    readout = probe_readout()
    record = measure(spec, args.seed)
    record['readout_probe'] = readout

    print(f'instance m={spec.m} n={spec.n} degree={spec.degree} r={spec.r} shots={spec.num_shots}')
    print(f'oracle roots per shot: mean={record["mean_roots_per_shot"]:.2f} '
          f'({record["root_density"]:.3%} of positions)\n')
    for label in record['arms']:
        row = record[label]
        print(f'{label:8s} Q={row["num_qubits"]:5d} executed T={row["toffoli_per_shot"]:12.2f} '
              f'static={row["static_toffoli"]:9d} val_err={row["value_errors"]:4d} '
              f'phase_err={row["phase_errors"]:4d}')
    base = record['baseline']['toffoli_per_shot']
    for label in (a for a in record['arms'] if a != 'baseline'):
        row = record[label]
        print(f'{label:8s}: {base:.2f} -> {row["toffoli_per_shot"]:.2f}  '
              f'({record[f"toffoli_ratio_{label}"]:.3f}x smaller), correct={record[f"semantics_preserved_{label}"]}')
    print(f'Forney body share of T: {record["forney_body_share"]:.4%}')
    print(f'hmr on |1>: {readout["hmr_on_one_ones_count"]}/{readout["hmr_on_one_shots"]} ones '
          f'(X-basis only); push_condition(qubit)={readout["push_condition_accepts_qubit"]}; '
          f'bit_store(control=qubit)={readout["bit_store_accepts_quantum_control"]}')

    if args.record:
        path = Path(args.record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
        print(f'\nwrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

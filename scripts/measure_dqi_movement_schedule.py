#!/usr/bin/env python3
"""Experiment C: can the A/B exchange be resolved at build time instead of moved?

`docs/EEA_SCHEDULE_CAMPAIGN.md` §4 leaves two ownership questions unanswered and
closes both "for this schedule": the wholesale `(A, B)` exchange (64,224 executed
Toffolis per shot) and the shift's remaining mask-fired movement (65,325). The
only route left for either is **a different data-movement schedule**, and the
schedule change that would remove both entirely is *pointer substitution*: do not
move the data, move the name.

That works if and only if the movement's predicate is a function of something
fixed when the circuit is built. The exchange is controlled on `c3 and flag`,
which is a *data* predicate, so the schedule is resolvable at build time exactly
when the set of rounds in which `c3` fires is the same for every shot. This
script measures that set, and then builds the arm that would exploit it.

Three pre-declared outcomes:

* **C1 -- the rebased arm.** Replace `_swap_a_and_b` with a build-time
  reassignment of which physical register plays the `A` role in each round. Its
  saving is not a projection: it removes the region, so it saves the region's
  whole executed cost, `64,224.0`, and the two control ANDs inside it.
  Prediction: **the saving lands exactly on the control's swap row**, and the arm
  is **correct if and only if the `c3`-round set is shot-independent**.
* **C2 -- how many schedules there are.** Read `c3` once per round into a
  per-round qubit (a probe arm; its Toffoli count is not a candidate's) and count
  the distinct per-shot patterns over the same 64 shots. One pattern means C1 is
  available; more than one means no build-time resolution exists and C1's arm is
  the falsifier of its own hypothesis.
* **C3 -- is `flag` redundant in a `c3` round?** The campaign's S3b arm showed
  the shift's `¬flag ∧ c3` branch is dead. If that generalises, the exchange's
  control is `c3` alone and its `init_and`/`del_and` pair is one Toffoli per
  firing round that never had to be there: `73 × 1 = 73`. Prediction: **73.0**,
  correct.

The point of the experiment is not the 64,224. It is that a data-movement
schedule which leaves the emitted circuit *smaller* is the only kind of change
that survives the pinned binary (Experiment D), so it is worth knowing whether
one exists before the receipt question is asked again.

Examples::

    /dev/shm/qvcc-venv313/bin/python scripts/build_kickmix_mr.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_movement_schedule.py
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_movement_schedule.json"

#: The incumbent this experiment is measured against: the five-gate stacked arm.
PINNED_INSTANCE = (4095, 70, 12)
STACKED_TOTAL = 686_014.8
STACKED_QUBITS = 1_913
#: The exchange's executed cost in the incumbent, from `dqi_schedule_campaign.json`.
SWAP_REGION = 64_224.0
#: How many rounds the exchange fires in, 64,224 / 892: one execution is 880 lanes
#: (876 register qubits and the four metadata qubits) plus the twelve gate-stream
#: Toffolis the `cswap` over an 876-qubit array costs per call.
SWAP_FIRINGS = 72
#: The control AND pair the exchange wraps around its cswap (one Toffoli, plus
#: the `del_and` measured clear): what C3 predicts removing.
SWAP_CONTROL_COST = 1.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_swapped(round_index: int) -> bool:
    """Whether the `A` and `B` roles are exchanged at the start of `round_index`.

    The exchange happens at the end of a round in which `c3` fires. If the
    sub-step counter advances in every live round, `c3` fires in rounds
    `i % 4 == 3` and the roles are exchanged `i // 4` times before round `i`.
    """
    return (round_index // 4) % 2 == 1


def make_movement_decoder(arms, campaign, halt, reuse, *, mode: str):
    """Build the arm `mode` on top of the five-gate incumbent itself.

    The base is `campaign.make_decoder(arms, "stacked", mixins)`, not the bare gated
    decoder: the campaign's stacked arm is 686,014.8 and the bare decoder is
    1,218,514.8, so a variant layered on the bare one would differ from its own
    control by the three schedule gates as well as by the exchange.
    """
    mixins = (halt.HaltGatedDecoder, reuse.ReusedFlagPredicateDecoder)
    rebased = mode == "rebased"
    probe = mode == "probe"
    c3_only = mode == "swap_on_c3"
    incumbent = campaign.make_decoder(arms, "stacked", mixins)

    class MovementDecoder(incumbent):
        movement_mode = mode

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._roles = {}
            self._rounds = None
            self.c3_probe = None
            self.flag_probe = None

        # -- the exchange's predicate, measured ---------------------------------
        def _probe(self, builder, c3_qubit, flag, it):
            """Record `c3` and the post-update flag for this round, one qubit each."""
            if not probe:
                return
            fired = self._read_bit(builder, c3_qubit)
            with campaign.conditional(builder, fired[0]):
                builder.x(self.c3_probe[it])
            flagged = self._read_bit(builder, flag)
            with campaign.conditional(builder, flagged[0]):
                builder.x(self.flag_probe[it])

        # -- the exchange ------------------------------------------------------
        def _exchange(self, builder, reg_a, reg_b, n_A, n_B, n_a, n_b, flag, ctrl):
            if rebased:
                return  # the roles are rebound in the round body instead
            if mode == "swap_cx":
                # A harness check, not a candidate: the incumbent's exchange re-emitted
                # from this method. It differs from the incumbent only in where the
                # control AND is built, so it must reproduce its total to the Toffoli --n
                # if it does not, the C3 arm below is measuring this rewrite rather than
                # its own change.
                with builder.mark("swap_a_and_b"):
                    c3_bit = self._read_bit(builder, ctrl[3])
                    with campaign.conditional(builder, c3_bit[0]):
                        swap_ctrl = builder.alloc_qubits(1)[0]
                        builder.init_and(flag, swap_ctrl)
                        lhs = arms.DR.km.array([q for r in (n_A, n_a, *reg_a) for q in r])
                        rhs = arms.DR.km.array([q for r in (n_B, n_b, *reg_b) for q in r])
                        builder.cswap(swap_ctrl, lhs, rhs)
                        builder.del_and(flag, swap_ctrl)
                        builder.free(swap_ctrl)
                return
            if not c3_only:
                return self._swap_a_and_b(builder, reg_a, reg_b, n_A, n_B, n_a, n_b,
                                          flag, ctrl)
            with builder.mark("swap_a_and_b"):
                c3_bit = self._read_bit(builder, ctrl[3])
                with campaign.conditional(builder, c3_bit[0]):
                    live = builder.alloc_qubits(1)[0]
                    builder.x(live)
                    lhs = arms.DR.km.array([q for r in (n_A, n_a, *reg_a) for q in r])
                    rhs = arms.DR.km.array([q for r in (n_B, n_b, *reg_b) for q in r])
                    builder.cswap(live, lhs, rhs)
                    builder.x(live)
                    builder.free(live)

        # -- the round ---------------------------------------------------------
        def _synchronized_eea(self, builder, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q,
                              flag, counter, halting_counter, *, n_iters, is_adjoint):
            self._roles = {"a": reg_a, "b": reg_b, "A": n_A, "B": n_B, "a_deg": n_a,
                           "b_deg": n_b}
            self._rounds = n_iters
            if probe:
                self.c3_probe = builder.alloc_qubits(n_iters)
                self.flag_probe = builder.alloc_qubits(n_iters)
            eea_step = builder.mark("eea_step")
            for it in (range(n_iters - 1, -1, -1) if is_adjoint
                       else range(n_iters)):
                with eea_step:
                    if rebased and run_swapped(it):
                        a, b = reg_b, reg_a
                        A, B = n_B, n_A
                        aa, bb = n_b, n_a
                    else:
                        a, b, A, B, aa, bb = reg_a, reg_b, n_A, n_B, n_a, n_b

                    if is_adjoint:
                        self._update_counter(builder, counter, flag, is_adjoint=True)

                    is_c3 = builder.alloc_qubits(1)[0]
                    builder.init_and(counter[0], counter[1], is_c3)
                    if is_adjoint:
                        self._update_halting_counter(builder, A, B, bb, halting_counter,
                                                     flag, is_c3, is_adjoint=True)

                    is_hc_zero = builder.alloc_qubits(1)[0]
                    builder.flip_if_equal(halting_counter, 0, target=is_hc_zero)
                    ctrl = self._counter_controls(builder, counter, is_c3, is_hc_zero)

                    live = self._read_bit(builder, is_hc_zero)
                    steps = [
                        lambda: self._gated(builder, live[0], lambda: self._update_flag(
                            builder, b, A, B, aa, bb, n_q, flag, ctrl)),
                        lambda: self._multiply_and_add_or_sub(builder, a, b, A, aa, n_q,
                                                             ctrl, it,
                                                             is_adjoint=is_adjoint),
                        lambda: self._shift_b_array_right(builder, b, B, bb, flag, ctrl,
                                                          it, is_adjoint=is_adjoint),
                        lambda: self._gated(builder, live[0], lambda: self._update_metadata(
                            builder, B, bb, n_q, flag, ctrl, is_adjoint=is_adjoint)),
                        # The exchange's own two operands, read where the exchange reads
                        # them: `c3` and the *post-update* flag. Nothing between the
                        # flag update and here writes `flag`, so this is the value the
                        # exchange's control AND consumes.
                        lambda: self._probe(builder, ctrl[3], flag, it),
                        lambda: self._exchange(builder, a, b, A, B, aa, bb, flag, ctrl),
                    ]
                    for step in (reversed(steps) if is_adjoint else steps):
                        step()

                    self._del_counter_controls(builder, counter, is_c3, is_hc_zero, ctrl)
                    builder.flip_if_equal(halting_counter, 0, target=is_hc_zero)
                    builder.free(is_hc_zero)

                    if not is_adjoint:
                        self._update_halting_counter(builder, A, B, bb, halting_counter,
                                                     flag, is_c3, is_adjoint=False)
                    builder.del_and(counter[0], counter[1], is_c3)
                    builder.free(is_c3)
                    if not is_adjoint:
                        self._update_counter(builder, counter, flag, is_adjoint=False)

        # -- the Forney block reads the register that holds role `B` -----------
        def _chien_search_and_forney_mbuc(self, builder, reg_b, error_mbuc, n_b, n_B):
            if rebased and self._rounds and run_swapped(self._rounds - 1):
                roles = self._roles
                reg_b, n_b, n_B = roles["a"], roles["a_deg"], roles["A"]
            return super()._chien_search_and_forney_mbuc(builder, reg_b, error_mbuc,
                                                        n_b, n_B)

    MovementDecoder.__name__ = f"Movement[{mode}]"
    return MovementDecoder


PREDICTION = {
    "written_before_the_arms": True,
    "question": "can a data-movement schedule resolve the A/B exchange at build time, "
                "so that the emitted circuit is smaller rather than the executed one?",
    "why_it_matters": "a saving from a *read* needs `mr`; a saving from a *schedule* is "
                      "in the emitted circuit, so it survives the pinned competition "
                      "binary (see Experiment D)",
    "C1_rebased": {
        "change": "do not exchange (n_A, n_a, reg_a) with (n_B, n_b, reg_b); reassign "
                  "which physical register plays the A role in each round from the round "
                  "index alone, and hand the Forney block the register holding role B",
        "assumption": "the sub-step counter advances once per live round, so `c3` fires "
                      "in rounds with i % 4 == 3 and the role assignment is a function of "
                      "the round index",
        "predicted_saving_per_shot": SWAP_REGION,
        "why_exactly": "the arm removes the region, so the saving is the region's whole "
                       "executed cost, including the two control ANDs inside it",
        "predicted_correct_if_and_only_if": "the c3-round set is the same for every shot",
        "falsifier": "this arm itself: if it is incorrect, the schedule is data-dependent "
                     "and no build-time resolution exists",
    },
    "C2_pattern": {
        "method": "read the c3 control into a per-round qubit of a probe arm and compare "
                  "the patterns across the 64 shots",
        "prediction": "the pattern is a function of the shot's syndrome, so distinct "
                      "patterns > 1 and C1 is unavailable; one pattern would make C1 exact",
    },
    "C3_flag_redundant": {
        "change": "exchange on `c3` alone instead of `c3 and flag`, dropping the control "
                  "AND pair (the campaign's S3b arm showed the shift's not-flag-and-c3 "
                  "branch is dead in every c3 round)",
        "predicted_saving_per_shot": SWAP_FIRINGS * SWAP_CONTROL_COST,
        "predicted_correct": True,
    },
}


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
                        default=["rebased", "swap_cx", "swap_on_c3", "probe"])
    args = parser.parse_args()

    pinned = (args.m, args.n, args.degree) == PINNED_INSTANCE

    runner = load_module("mr_gated_arms_kickmix",
                         ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    km = runner.load_kickmix(args.kickmix_build)
    ledger = load_module("region_ledger",
                         ROOT / "scripts" / "measure_dqi_region_ledger.py")
    arms = ledger.load_arms(km, runner)
    campaign = load_module("schedule_campaign",
                           ROOT / "scripts" / "measure_dqi_schedule_campaign.py")
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
    spec = V.ProblemSpecDQI.benchmark_instance(
        m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
    weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    tables = V.FieldTables(km.GF2Field(spec.degree))
    rng = np.random.default_rng(20261028)
    syndromes_in, mbuc_in, _ = V.sample_shots(spec, tables, rng, args.shots, weights)
    reference_inputs = V.to_bits(syndromes_in, args.degree)

    def correct(row) -> bool:
        return bool(row["syndromes"] and row["junk"] and row["phase"])

    def region_nonproduct(arm, name) -> float:
        for entry in arm["regions"]["rows"]:
            if entry["region"] == name:
                return (entry["executed_toffoli_per_shot"]
                        - entry["product_toffoli_per_shot"])
        return 0.0

    def static_region(arm, name) -> int:
        for entry in arm["regions"]["rows"]:
            if entry["region"] == name:
                return entry["static_toffoli"]
        return 0

    original = arms.DR.km
    arms.DR.km = shim
    try:
        ref_circuit, ref_syn, ref_mbuc = ledger.build_reference(
            km, arms, shim, args.n, args.degree, args.m)
        reference = ledger.simulate(km, ref_circuit, ref_syn, ref_mbuc, syndromes_in,
                                    mbuc_in, args.degree, args.shots)
        del ref_circuit
        gc.collect()

        def build(decoder_type, capture_registers=None):
            builder = shim.CircuitBuilder()
            syndromes = [builder.create_quantum_register(args.degree, name=f"S{i}")
                         for i in range(args.n)]
            error_mbuc = [builder.create_classical_register(args.degree, name=f"E{j}")
                          for j in range(args.m)]
            decoder = decoder_type(args.n, km.GF2Field(args.degree),
                                   **campaign.SIX_LEVERS)
            decoder.build_syndrome_decoding_mbuc(
                builder, syndromes=syndromes, error_mbuc=error_mbuc,
                n_iters=iterations)
            circuit = builder.finish_circuit()
            registers = {}
            if capture_registers:
                registers = capture_registers(decoder)
            return circuit, syndromes, error_mbuc, registers

        def measure(decoder_type, probe_registers=False):
            holder = {}

            def capture(decoder):
                holder["c3"] = getattr(decoder, "c3_probe", None)
                holder["flag"] = getattr(decoder, "flag_probe", None)
                return holder

            circuit, syn, mbuc, registers = build(
                decoder_type, capture if probe_registers else None)
            probe_registers_values = ((registers.get("c3"), registers.get("flag"))
                                      if probe_registers else None)
            out, probe_values = simulate(circuit, syn, mbuc, probe_registers_values)
            costs, instructions, _rounds = campaign.region_costs(circuit, out, args.shots)
            row = {
                "qubits": circuit.num_qubits,
                "mean": out["mean_toffoli_like"],
                "toffoli_per_shot": out["toffoli_per_shot"].tolist(),
                "regions": ledger.region_ledger(circuit, out, args.shots),
                "costs": costs,
                "instructions_per_shot": float(out["counts"].sum()) / args.shots,
                "read_instructions_per_shot": instructions.get("predicate_read", 0.0),
                "mid_circuit_reads_per_shot":
                    instructions.get("predicate_read", 0.0) / 2.0,
                "static_toffoli": int(out["toffoli_mask"].sum()),
                "static_instructions": len(circuit),
                "syndromes": bool(np.array_equal(out["syndromes"], reference_inputs)),
                "junk": bool(not out["junk"].any()),
                "phase": bool(np.array_equal(out["phase"], reference["phase"])),
                "eea_steps": sum(1 for _i, text in circuit.info_tags()
                                 if text == "@enter:eea_step"),
            }
            if probe_values is not None:
                c3_values, flag_values = probe_values
                row["c3_pattern"] = pattern_rows(c3_values, args.shots)
                row["flag_pattern"] = pattern_rows(flag_values, args.shots)
                row["exchange_predicate"] = exchange_predicate_rows(
                    c3_values, flag_values, args.shots)
            del circuit, out
            gc.collect()
            return row

        def campaign_arm_audit() -> dict:
            """Are the campaign's two zero-saving arms *arms*, or unmodified copies?

            `composed` and `drop_shift_c3` both measured exactly 0.0 in
            `dqi_schedule_campaign.json`, and the campaign's text explains that as the
            extra condition being unsatisfiable. The implementation of both lives in a
            method named `_shift_a_and_b`, which nothing calls: the reference and the
            arms call `_shift_b_array_right`. If that is what happened, the two arms are
            the control circuit byte for byte and their 0.0 is vacuous rather than
            informative. This builds all three at a small instance and compares the
            emitted instruction streams, which is a structural question -- it cannot
            depend on the data.
            """
            small = {"m": 15, "n": 6, "degree": 4}
            rows = {}
            for name in ("none", "composed", "drop_shift_c3"):
                cls = campaign.make_decoder(arms, name, mixins)
                builder = shim.CircuitBuilder()
                syn = [builder.create_quantum_register(small["degree"],
                                                       name=f"S{i}")
                       for i in range(small["n"])]
                err = [builder.create_classical_register(small["degree"],
                                                        name=f"E{j}")
                       for j in range(small["m"])]
                cls(small["n"], km.GF2Field(small["degree"]),
                    **campaign.SIX_LEVERS).build_syndrome_decoding_mbuc(
                        builder, syndromes=syn, error_mbuc=err,
                        n_iters=arms.DR.default_n_iters(small["n"]))
                circuit = builder.finish_circuit()
                rows[name] = {
                    "ops": len(circuit),
                    "static_toffoli": int(
                        circuit.operation_statistics().num_toffoli_like),
                }
                del circuit
                gc.collect()
            return {
                "instance": small,
                "arms": rows,
                "composed_is_a_copy": rows["composed"] == rows["none"],
                "drop_shift_c3_is_a_copy": rows["drop_shift_c3"] == rows["none"],
                "why": "the campaign's `_shift_a_and_b` is never called: the reference "
                       "and `MRGatedDecoder` call `_shift_b_array_right`, so both arms "
                       "emit the control's instruction stream -- until it is renamed, "
                       "which is what `campaign_composed` / `campaign_drop_c3` measure",
            }

        def simulate(circuit, syn, mbuc, probe_registers_read):
            """`ledger.simulate`, plus optional reads of per-round probes."""
            sim = km.Simulator(batch_size=args.shots, seed=99, counting_operations=True)
            sim.counting_op_executions = True
            sim.use_same_registers_as(circuit)
            flat_syndromes = km.array([q for r in syn for q in r])
            flat_errors = km.array([b for r in mbuc for b in r])
            sim.write_across_shots(flat_syndromes, V.to_bits(syndromes_in, args.degree))
            sim.write_across_shots(flat_errors, V.to_bits(mbuc_in, args.degree))
            sim.do(circuit)
            counts = np.asarray(sim.operation_execution_counts(), dtype=np.int64)
            toffoli_mask = np.asarray(circuit.op_toffoli_like, dtype=bool)
            per_shot = np.array([int(sim.operation_statistics(i).num_toffoli_like)
                                 for i in range(args.shots)])
            syndrome_ids = {q.id for r in syn for q in r}
            if probe_registers_read is not None:
                for register in probe_registers_read:
                    syndrome_ids |= {q.id for q in register}
            junk_ids = [i for i in range(circuit.num_qubits) if i not in syndrome_ids]
            out = {
                "counts": counts,
                "toffoli_mask": toffoli_mask,
                "toffoli_per_shot": per_shot,
                "mean_toffoli_like": float(per_shot.mean()),
                "syndromes": sim.read_across_shots(flat_syndromes),
                "junk": sim.read_across_shots([km.q(i) for i in junk_ids]),
                "phase": sim.read_phase_flipped_across_shots(),
            }
            probe = (tuple(np.asarray(sim.read_across_shots(register))
                           for register in probe_registers_read)
                     if probe_registers_read is not None else None)
            return out, probe

        control = measure(lambda *a, **k: campaign.make_decoder(arms, "stacked", mixins)(
            *a, **k))
        print(f"control (five gates) Q={control['qubits']} T={control['mean']:.1f}/shot "
              f"swap={region_nonproduct(control, 'swap_a_and_b'):.1f}")
        audit = campaign_arm_audit()
        print(f"campaign arm audit: composed is a copy={audit['composed_is_a_copy']}, "
              f"drop_shift_c3 is a copy={audit['drop_shift_c3_is_a_copy']}")

        measured = {}
        for mode in args.variants:
            row = measure(make_movement_decoder(arms, campaign, halt, reuse, mode=mode),
                          probe_registers=(mode == "probe"))
            measured[mode] = row
            extra = ""
            if "c3_pattern" in row:
                extra = (f" patterns={row['c3_pattern']['distinct_patterns']}"
                         f"/{args.shots} firings={row['c3_pattern']['firings_per_shot']}"
                         f"/{args.shots}")
            print(f"  {mode:12s} Q={row['qubits']} T={row['mean']:.1f}/shot "
                  f"({control['mean'] - row['mean']:+.1f}) "
                  f"swap={region_nonproduct(row, 'swap_a_and_b'):.1f} "
                  f"correct={correct(row)}{extra}")

        # The campaign's two shift/swap arms, re-measured against the same control.
        # Both were reported as saving exactly 0.0, which the campaign explained as the
        # extra condition being unsatisfiable; the audit below showed the condition was
        # never emitted, because the method carrying it was not the one the decoder
        # calls. With that fixed they are arms, and these are their numbers.
        # Their own control is `campaign_none`: the same six-lever decoder without the
        # campaign's per-position chains, so the *only* difference is the gate under
        # test. The stacked arm above differs from both by three other gates.
        campaign_arms = {}
        for label, campaign_mode in (("campaign_none", "none"),
                                     ("campaign_composed", "composed"),
                                     ("campaign_drop_c3", "drop_shift_c3")):
            row = measure(campaign.make_decoder(arms, campaign_mode, mixins))
            campaign_arms[label] = row
            measured[label] = row
            print(f"  {label:16s} Q={row['qubits']} T={row['mean']:.1f}/shot "
                  f"shift={region_nonproduct(row, 'shift_b_array_right'):.1f} "
                  f"correct={correct(row)}")
    finally:
        arms.DR.km = original

    control_swap = region_nonproduct(control, "swap_a_and_b")
    checks = [{
        "check": "the control arm reproduces the five-gate incumbent",
        "requirement": f"{STACKED_TOTAL:.1f} executed Toffolis per shot at Q={STACKED_QUBITS}"
                       if pinned else "not the pinned instance",
        "measured": {"total": control["mean"], "qubits": control["qubits"]},
        "passed": (abs(control["mean"] - STACKED_TOTAL) < 0.5
                   and control["qubits"] == STACKED_QUBITS) if pinned else True,
    }, {
        "check": "the control arm's exchange row reproduces the campaign's",
        "requirement": f"{SWAP_REGION:.1f} per shot" if pinned else "not pinned",
        "measured": control_swap,
        "passed": abs(control_swap - SWAP_REGION) < 0.1 if pinned else True,
    }]

    observation = {"control_total": control["mean"],
                   "control_swap_region": control_swap,
                   "control_static_toffoli": control["static_toffoli"],
                   "campaign_arm_audit": audit}
    checks.append({
        "check": "the campaign's two zero-saving arms are audited, not assumed",
        "requirement": "the emitted streams are compared with the control's",
        "measured": {"composed_is_a_copy": audit["composed_is_a_copy"],
                     "drop_shift_c3_is_a_copy": audit["drop_shift_c3_is_a_copy"]},
        "passed": True,
    })

    if "campaign_none" in measured:
        base_row = measured["campaign_none"]
        base_mean = base_row["mean"]
        base_shift = region_nonproduct(base_row, "shift_b_array_right")
        observation["campaign_none_total"] = base_mean
        observation["campaign_none_shift_region"] = base_shift
        for label, campaign_mode in (("campaign_composed", "composed"),
                                     ("campaign_drop_c3", "drop_shift_c3")):
            if label not in measured:
                continue
            row = measured[label]
            saving = base_mean - row["mean"]
            shift_region = region_nonproduct(row, "shift_b_array_right")
            observation[f"{label}_saving"] = saving
            observation[f"{label}_correct"] = correct(row)
            observation[f"{label}_shift_region"] = shift_region
            observation[f"{label}_instructions"] = row["instructions_per_shot"]
            observation[f"{label}_instructions_base"] = base_row["instructions_per_shot"]
            checks.append({
                "check": f"the campaign's `{campaign_mode}` arm is an arm now",
                "requirement": "its instruction stream differs from the same decoder "
                               "without the gate (the campaign could not show that: the "
                               "method carrying the gate was never called)",
                "measured": {"static_instructions": row["static_instructions"],
                             "base_static_instructions":
                                 base_row["static_instructions"],
                             "instructions_per_shot": row["instructions_per_shot"],
                             "base_instructions_per_shot":
                                 base_row["instructions_per_shot"],
                             "saving_vs_same_decoder": saving,
                             "correct": correct(row),
                             "shift_region": shift_region,
                             "base_shift_region": base_shift},
                # What makes it an arm is that its *emitted* stream differs from the
                # same decoder without the gate -- which is exactly the thing that was
                # broken before (the stream was byte for byte the control's). The
                # executed count is the measurement, not the test.
                "passed": row["static_instructions"] != base_row["static_instructions"],
            })

    if "rebased" in measured:
        row = measured["rebased"]
        saving = control["mean"] - row["mean"]
        observation["rebased_saving"] = saving
        observation["rebased_total"] = row["mean"]
        observation["rebased_correct"] = correct(row)
        observation["rebased_swap_region"] = region_nonproduct(row, "swap_a_and_b")
        observation["rebased_static_saving"] = (control["static_toffoli"]
                                               - row["static_toffoli"])
        checks.append({
            "check": "C1: resolving the exchange at build time removes the region",
            "requirement": f"the swap row falls to 0.0 and the arm's static count falls "
                           f"by {static_region(control, 'swap_a_and_b')} (the region's "
                           f"whole static cost, control ANDs included)",
            "measured": {"swap_region": region_nonproduct(row, "swap_a_and_b"),
                         "static_fall": observation["rebased_static_saving"],
                         "declared_executed_saving": SWAP_REGION,
                         "executed_saving_of_this_arm": saving},
            "passed": (region_nonproduct(row, "swap_a_and_b") == 0.0
                       and observation["rebased_static_saving"]
                       == static_region(control, "swap_a_and_b")),
        })
        checks.append({
            "check": "C1: the rebased arm is correct only if the schedule is a function "
                     "of the round index",
            "requirement": "recorded, not assumed: this arm is the falsifier of its own "
                           "hypothesis, so its total is not evidence either way",
            "measured": {"correct": correct(row),
                         "syndromes": row["syndromes"], "junk": row["junk"],
                         "phase": row["phase"]},
            "passed": True,
        })

    if "swap_on_c3" in measured:
        row = measured["swap_on_c3"]
        saving = control["mean"] - row["mean"]
        # C3 is a prediction, not a harness check: the measurement decides, and either
        # outcome is the record. It falsified here, which is the informative case -- the
        # exchange's `flag` operand is not redundant in a `c3` round.
        observation["c3_only_saving"] = saving
        observation["c3_only_correct"] = correct(row)
        observation["c3_only_prediction"] = {
            "predicted_saving_per_shot": SWAP_FIRINGS * SWAP_CONTROL_COST,
            "predicted_correct": True,
            "outcome": ("confirmed" if correct(row)
                        and abs(saving - SWAP_FIRINGS * SWAP_CONTROL_COST) < 1.0
                        else "falsified: the flag operand is not redundant in a c3 round"),
        }
        checks.append({
            "check": "C3: the flag-only arm was built and measured",
            "requirement": "an arm that exchanges on c3 alone, correct or not",
            "measured": {"saving": saving, "correct": correct(row)},
            "passed": row["qubits"] == control["qubits"],
        })

    if "probe" in measured and "exchange_predicate" in measured["probe"]:
        predicate = measured["probe"]["exchange_predicate"]
        observation["exchange_predicate"] = predicate
        checks.append({
            "check": "C3: the exchange's two operands are read where it reads them",
            "requirement": "`c3` and the post-update `flag` per round per shot, so that "
                           "`c3 and flag` can be compared with `c3` alone",
            "measured": {key: predicate[key] for key in
                         ("c3_rounds_total", "exchanges_total",
                          "c3_rounds_with_flag_clear_total",
                          "flag_is_redundant_in_c3_rounds",
                          "distinct_c3_sets", "distinct_exchange_sets")},
            "passed": predicate["c3_rounds_total"] > 0,
        })

    if "probe" in measured and "c3_pattern" in measured["probe"]:
        pattern = measured["probe"]["c3_pattern"]
        observation["c3_pattern"] = pattern
        checks.append({
            "check": "C2: the movement schedule's predicate set is measured per shot",
            "requirement": "a pattern per shot, and the count of distinct ones",
            "measured": {"distinct_patterns": pattern["distinct_patterns"],
                         "shots": pattern["shots"],
                         "firings_per_shot": pattern["firings_per_shot"]},
            "passed": pattern["distinct_patterns"] >= 1,
        })

    failed = [entry for entry in checks if not entry["passed"]]
    payload = {
        "experiment": "C -- a data-movement schedule for the A/B exchange",
        "declared_in": "docs/EEA_SCHEDULE_CAMPAIGN.md",
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_role": "local rebuild with MR and per-instruction execution counts "
                        "(scripts/build_kickmix_mr.py); not the pinned competition binary",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "iterations": iterations},
        "policy": "and54",
        "levers": {"gate_forney": True, "clear_eea": True, "gate_sweep": True,
                   "gate_swap": True, "gate_shift": True, "gate_quotient": True},
        "prediction": PREDICTION,
        "control": {"qubits": control["qubits"], "mean_toffoli_like": control["mean"],
                    "static_toffoli": control["static_toffoli"],
                    "swap_region": control_swap,
                    "instructions_per_shot": control["instructions_per_shot"]},
        "arms": {name: {key: value for key, value in row.items()
                        if key not in ("regions", "toffoli_per_shot")}
                 for name, row in measured.items()},
        "arm_regions": {name: row["regions"]["rows"] for name, row in measured.items()},
        "observation": observation,
        "checks": checks,
        "failed_checks": [entry["check"] for entry in failed],
        "verdict": ("the schedule is data-dependent: no build-time resolution exists"
                    if "rebased" in measured and not correct(measured["rebased"])
                    else "the schedule resolves at build time"),
        "scope": {
            "measured": "a correct, cheaper-or-not arm per variant at the pinned instance, "
                        "with the executed instruction stream and read count beside every "
                        "Toffoli figure",
            "not_measured": "no receipt: the incumbent's reads still need `mr`, which the "
                            "pinned competition binary does not have (Experiment D)",
        },
        "reproducibility": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
            "git_rev": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      check=False, capture_output=True,
                                      text=True).stdout.strip(),
        },
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"verdict: {payload['verdict']}")
    print(f"failed checks: {payload['failed_checks'] or 'none'}")
    print(f"wrote {args.record}")
    return 0 if not failed else 1


def exchange_predicate_rows(c3_values, flag_values, shots) -> dict:
    """The exchange's predicate, round by round and shot by shot.

    `c3 and flag` is the control the exchange's `cswap` sits under. Reading both
    operands where the exchange reads them separates two questions the campaign
    conflated: how many rounds `c3` fires in, and how many of those the *flag* leaves
    live. If those two counts differ, exchanging on `c3` alone is a different schedule
    -- which is why the C3 arm is wrong rather than 73 Toffolis cheaper.
    """
    c3 = np.asarray(c3_values).reshape(-1, shots) > 0
    flag = np.asarray(flag_values).reshape(-1, shots) > 0
    rounds = c3.shape[0]
    both = c3 & flag
    c3_without_flag = c3 & ~flag
    return {
        "rounds": rounds,
        "shots": shots,
        "c3_firings_per_shot": c3.sum(axis=0).tolist(),
        "exchanges_per_shot": both.sum(axis=0).tolist(),
        "c3_rounds_with_flag_clear_per_shot": c3_without_flag.sum(axis=0).tolist(),
        "c3_rounds_total": int(c3.sum()),
        "exchanges_total": int(both.sum()),
        "c3_rounds_with_flag_clear_total": int(c3_without_flag.sum()),
        "flag_is_redundant_in_c3_rounds": bool(not c3_without_flag.any()),
        "distinct_exchange_sets": len({tuple(np.flatnonzero(row).tolist())
                                       for row in both.T}),
        "distinct_c3_sets": len({tuple(np.flatnonzero(row).tolist())
                                 for row in c3.T}),
    }


def pattern_rows(values, shots) -> dict:
    """Describe the per-shot probe patterns: one round per row, one shot per column."""
    values = np.asarray(values).reshape(-1, shots)
    rounds = values.shape[0]
    rows = [tuple(int(values[i, shot]) for i in range(rounds)) for shot in range(shots)]
    distinct = sorted({row for row in rows})
    return {
        "rounds": rounds,
        "shots": shots,
        "distinct_patterns": len(distinct),
        "patterns": ["".join(map(str, row)) for row in distinct],
        "firings_per_shot": [int(sum(row)) for row in rows],
        "identical_to_period_four": all(
            row == tuple(1 if i % 4 == 3 else 0 for i in range(rounds))
            for row in rows),
    }


if __name__ == "__main__":
    raise SystemExit(main())

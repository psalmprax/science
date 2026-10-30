# The non-multiply ledger, and the first controlled experiment

**Status:** measurement ledger + experiment protocol · **Scope:** the six-lever DQI arm at
the pinned instance (m=4095, n=70, GF(2^12), 64 shots) · **Evidence:**
`benchmarks/quantum_compilation/dqi_nonmultiply_ledger.json`, produced by
`scripts/attribute_dqi_nonmultiply_tcount.py` from `gf54_native_opcode.json`,
`designer_directions.json`, `gf_mul_and_static_headroom.json` and `dqi_primitive_costs.json`;
plan and levers in [`EEA_AND_PREDICATE_PLAN.md`](EEA_AND_PREDICATE_PLAN.md).

The review asked for a reconciled, region-by-region account of the non-multiply T-count
before another edit, with each gate in exactly one category, measured contributions
distinguished from estimates, and the part instrumentation cannot reach stated rather than
guessed. This is that ledger. It also corrects the accounting invariant it was asked to
reconcile against: the budget is **472,768.2**, not 527,806.8, because the earlier two-term
form folded 1,019.23 inversion/division products into its constant.

## 1. Executed ledger (the scored quantity)

`num_toffoli_like` per shot, measured at the pinned instance; components sum to the total by
construction and the record refuses to write if they do not.

| category | Toffolis/shot | share | measured by |
|---|---|---|---|
| `ixor_gf2_mul` — 10,510 calls × (54 + 12) | 693,660.0 | 56.8% | policy-scoped CCX difference between two correct arms |
| inversion products — 155.23 × 54 | 8,382.7 | 0.7% | same, scoped to `init_gf2_inverse_with_scaffold` |
| division products — 864 × 54 | 46,656.0 | 3.8% | same, scoped to `init_gf2_div_with_scaffold` |
| **non-product residual** | **472,768.2** | **38.7%** | arm total minus the three product terms |
| total | 1,221,466.8 | 100% | simulator, 64 shots |

The residual is the object of the review's stage A. It splits as far as the instruments
allow:

| residual part | Toffolis/shot | share of residual | status |
|---|---|---|---|
| `_update_flag` | 32,856 | 6.9% | measured exactly (unconditional) |
| `_initial_shift_loop` | 30,912 | 6.5% | measured exactly |
| `_update_metadata` | 12,432 | 2.6% | measured exactly |
| `_update_halting_counter` | 10,656 | 2.3% | measured exactly |
| `_pre_align_reg_b` | 7,392 | 1.6% | measured exactly |
| `_update_counter` | 222 | 0.05% | measured exactly |
| **unconditional total** | **94,470** | **20.0%** | |
| conditional EEA body | 378,298.2 | 80.0% | split per region, §1.1 |
| **residual** | **472,768.2** | **100%** | |

**Why the 80% was originally a named bucket.** The six regions above run on every shot, so
their static count is their executed count. Everything else in the residual is conditional,
and the simulator reported per-shot counts **by operation type**, so an executed per-region
split could not be read off it. The ledger said so instead of estimating numbers into the
table and named the two ways to close it: per-operation execution counts in the simulator,
or an oracle-gated variant per region that measures only the removable part.

### 1.1 The 80%, measured per region

Both ways are now implemented. The local `MR` build carries per-instruction execution
counts (`sim.counting_op_executions`), and `builder.mark(...)` boundaries are read back with
`circuit.info_tags()`, so `scripts/measure_dqi_region_ledger.py` partitions the executed
stream exactly: the rows sum to the measured 1,221,466.8 and to the 472,768.2 residual, and
the script refuses to write a record that does not. Record:
`benchmarks/quantum_compilation/dqi_region_ledger.json`.

| region | non-product T/shot | share of residual | executed/static |
|---|---|---|---|
| `shift_b_array_right` | 162,479.6 | 34.4% | 68.4% |
| `sweep_positions` | 143,038.0 | 30.3% | 64.8% |
| `swap_a_and_b` | 64,224.0 | 13.6% | 32.4% |
| `_update_flag` | 32,856 | 6.9% | 100% |
| `_initial_shift_loop` | 30,912 | 6.5% | 100% |
| `_update_metadata` | 12,432 | 2.6% | 100% |
| `_update_halting_counter` | 10,656 | 2.3% | 100% |
| counter decode (`eea_step`) | 4,662 | 1.0% | 100% |
| `_pre_align_reg_b` | 3,696 | 0.8% | 100% |
| `unalign_reg_b` | 3,696 | 0.8% | 100% |
| multiply-and-add control | 2,016 | 0.4% | 64.9% |
| quotient-division control | 1,728 | 0.4% | 32.4% |
| Forney control | 372.6 | 0.1% | 0.8% |
| **sum** | **472,768.2** | **100%** | |

The field products sit inside four of those regions and are not part of this table:
`sweep_positions` 693,660.0, `quotient_div` 46,656.0, `forney_step` 8,382.7 — 748,698.7 in
total, which is the first row of §1 plus the two isolated product rows.

Two things follow, and they change the plan rather than confirm it:

* The predicate block (`_update_flag`, halting, counter) is 44,394 of the residual, not the
  largest item. The two gated-override regions are 370,000 between them, and they are large
  *because* the six levers only removed part of each: `shift_b_array_right` still executes
  68.4% of its static gates and `sweep_positions` 64.8%.
* The gated share of each region is measured per lever, not inferred (§1.2), so a future
  claim of "cut region X by half" can be checked against what the lever already took.

### 1.2 What each lever actually removed, per region

Each row is the six-lever arm with **one lever absent**, differenced against the arm, so the
figure is that lever's measured executed saving and the region column is where it came from.

| lever absent | T/shot | regions it moved |
|---|---|---|
| `gate_forney` | 1,243,975.3 | `forney_step` |
| `clear_eea` | 1,205,319.6 | the whole reverse EEA: sweep 836,698, shift 162,479.6, swap 64,224, quotient 48,384, `_update_flag` 32,856, shift loop 30,912, halting 10,656, metadata 12,432, decode 15,318, multiply-and-add control 2,016 |
| `gate_sweep` | 556,312.0 | sweep 454,420, quotient 100,800, multiply-and-add control 1,092 |
| `gate_swap` | 134,022.0 | `swap_a_and_b` |
| `gate_shift` | 74,896.4 | `shift_b_array_right` |
| `gate_quotient` | 48,384.0 | `quotient_div` |

`measure_dqi_region_ledger.py` checks that each lever's arm total difference equals the sum
of the region differences it caused (they do, within 0.5 Toffoli), and that every one of
those arms is still correct. That is the oracle treatment the plan asked for, read off the
real arm.

These deltas are on the **and54** arm — the 1,221,466.8 one this document describes. The
same leave-one-out experiment on the builtin-policy arm (1,325,230.0 at `Q` 1,894) is
`benchmarks/quantum_compilation/dqi_lever_leave_one_out.json`; the two records are not
interchangeable, because the policy moves the reference and the arm by different amounts.

## 2. Static ledger (emitted gates)

Deleting a leaf region from a fresh build and differencing is exact, because static counts
do not depend on state, so no gate is counted twice:

| category | Toffolis | share |
|---|---|---|
| the six unconditional leaf regions | 94,470 | 2.7% |
| everything else (EEA body, gated overrides, generator internals) | 3,392,127 | 97.3% |
| arm, static | 3,486,597 | |
| reference, static | 5,725,461 | |

Two things cannot be deleted and are therefore *inside* the 97.3%: generator internals
(`gen_gf_*`, `gen_iadd`, multi-control decompositions) and `_counter_controls` plus the
arm's gated overrides, which return values their callers consume and so cannot be replaced
by a no-op. A ledger that pretended otherwise would double count via parent-method deletion.

## 3. Unit prices (measured, reference widths)

| quantity | value |
|---|---|
| Toffoli per EEA iteration | 17,208 |
| Toffoli per Chien/Forney position | 351 |
| Toffoli per constant multiply / divide | 0 (CNOT-only linear maps) |

These are measured on the reference decoder, so they price the arm's EEA body rather than
split it: the arm gates that body and executes fewer iterations' worth of work.

## 4. What "beat the benchmark" means here

Two different claims, and only the second is a result:

- **resource ratio** — lower `Q²T` than the reference: **4.59×** today (`Q` 1,913 ×
  1,221,466.8 against 1,893 × 5,725,071.6), measured locally.
- **official score** — lower `R(ε_+) · Q² · T` as returned by `verify_submission_dqi.py`
  over its 102,400-shot sample, with the circuit accepted by the pinned toolchain. The
  reference row is 2.4439e13 with `R(ε_+) = 1.19126`. **Not achievable today** for this
  arm: the pinned extension has no `MR` opcode and `contracts.py` pins receipts to it.

"Beating the benchmark" in this repository therefore means the second, and the first is
evidence toward it, not a substitute. The statistical penalty can only be measured on the
official sample, so any stated gain is `Q²T` until a receipt exists.

## 5. The first controlled experiment

**Target:** `_update_flag` and the counter/halting predicates (`_update_flag` 32,856,
`_update_halting_counter` 10,656, `_update_counter` 222 — 43,734 Toffolis, 9.3% of the
residual), the largest measured predicate block and the one a reuse hypothesis would
change.

**Hypothesis to write down before the edit:** sharing the flag/counter subexpressions
removes `Δ` Toffolis per shot, leaves `Q` unchanged, and preserves the 222-iteration
schedule exactly. Predict `Δ`, `Q`, and the resulting `Q²T` *before* measuring.

**Control:** the same arm, same instance, same shots, same seed, fixed 222 iterations.
No early halting, no asynchronous pointer substitution, no combined patch.

**Promotion gates** (all required; any failure rejects the variant):

| metric | requirement |
|---|---|
| mean non-product T-count | reconcile with the instrumented total (LEDGER §1) |
| mean multiply T-count | reported separately (693,660 + 8,382.7 + 46,656) |
| total T-count | components sum to it, ±0.1 Toffoli |
| peak qubits `Q` | measured, including added ancillas |
| `Q²T` | recomputed from the same run, not scaled from `T` |
| correctness and cleanup | syndromes restored, measured phase, no junk, workspace `|0>` |
| iteration behaviour | fixed 222 iterations, unchanged |
| reproducibility | frozen source hash, extension hash, record path |

The measurement template already exists: `measure_gf54_native_opcode.py` builds the same
arm under a policy and records exactly these fields, and `measure_dqi_scratch_pool.py` gives
per-frame liveness if the peak moves.

### 5.1 Result (2026-10-29): PROMOTE, +1,776 Toffolis per shot

`scripts/measure_flag_predicate_reuse.py` runs the experiment. The variant shares the
`(n_A == n_B)` comparison between the `c0` and `c1` sub-steps and clears the scratch with
`MR` instead of uncomputing it: 8 Toffolis per EEA iteration, so 222 × 8 = **1,776 per
shot**, exactly the pre-registered prediction. The prediction is computed from unit costs
at the register width (8 bits, not the field degree 12 — pricing at 12 over-predicts by 4
per iteration, and that mistake was made and recorded before the corrected run).

| gate | measured |
|---|---|
| non-product T-count | 472,768.2 reconciles with §1 |
| multiply T-count | 748,698.7, reported separately |
| total T-count | components sum, ±0.1 |
| `Q` | 1,913 → 1,913 (no added ancilla: the region's own scratch is reused) |
| `Q²T` | 1,221,466.8 → 1,219,690.8 per shot, −0.145% |
| correctness | syndromes restored, phase matched, no junk |
| iterations | 222 forward rounds in both arms |
| reproducibility | extension, patch, script hashes and record path in the record |

`_update_flag` falls 32,856 → 31,080. The shape of the result matters more than its size:
the two equality comparisons could be shared, but the `c2` term tests `n_a == n_b + 1`
between the `iadd` and the `isub`, so it is a different predicate and cannot join sub-step
3's `n_a == n_b`. The predicate block's reuse headroom is 5.4% of `_update_flag` and 0.4%
of the residual — not the 20% its static share suggests. Record:
`benchmarks/quantum_compilation/dqi_flag_predicate_reuse.json`.

### 5.2 The schedule campaign (2026-10-30): −43.60%, and what the ledger now reads

[`EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md) applies the same read-then-gate
mechanism at per-position grain instead of per-round. Every predicate register in this
decoder is a basis state, so the read is deterministic and the phase is untouched, and a
gate conditioned on a zero bit costs no Toffoli on that shot. Measured at the pinned
instance, 64 shots, one arm: **1,221,466.8 → 688,966.8 per shot** (−532,500.0, −43.60%) at
`Q` = 1,913 and 222 rounds, correct, with the three gates exactly additive.

The ledger's rows then split as follows (executed, non-product unless stated):

| region | §1 row | after the campaign |
|---|---|---|
| `sweep_positions` | 143,038.0 | **39,070.0** (its product stream 693,660.0 → 331,023.0) |
| `shift_b_array_right` | 162,479.6 | **96,584.6** |
| `swap_a_and_b` | 64,224.0 | 64,224.0 (no gate: the composed shift/swap arm is worth 0.0) |
| `_update_flag` | 32,856 | 32,856 |
| `_initial_shift_loop` (`shift_step`) | 30,912 | 30,912 |
| `_update_metadata` | 12,432 | 12,432 |
| `_update_halting_counter` | 10,656 | 10,656 |
| decode + `_update_counter` | 4,662 | 4,662 |
| alignment pair | 7,392 | 7,392 |
| multiply-and-add control | 2,016 | 2,016 |
| quotient-division control | 1,728 | 1,728 |
| Forney control | 372.6 | 372.6 |
| **non-product residual** | **472,768.2** | **302,905.2** |
| field products | 748,698.7 | **386,061.7** |
| total | 1,221,466.8 | **688,966.8** |

The table is the three-gate arm (S1+S2+S4). Stacking the prior session's flag/metadata gate
on top — measured in the same record, not inferred across reports — takes the arm to
**686,014.8**, with `_update_flag` 32,856 → 30,240 and `_update_metadata` 12,432 → 12,096,
so the non-product residual there is **299,953.2** (`dqi_schedule_campaign.json`, arm
`stacked`).

Two of the ledger's earlier statements change meaning rather than value. The residual is
still 472,768.2 as a *static* stream; what fell is the executed stream, because more than half
of it was quantum-controlled work whose control was zero. And the "predicate block's
remaining headroom is small" reading of §5.1 was a statement about the *gates* in
`_update_flag`, not about the control-dependent work elsewhere — the three regions that
this section's table shows falling were 78.3% of the residual, and they were where it was.

## 6. Pins

`tests/test_dyn_mr_gated_arms_kickmix.py` pins the ledger: components sum to 1,221,466.8,
the three terms and the residual reconcile, the unconditional regions are 94,470 and 20.0%
of the residual, and the static ledger adds up to 3,486,597. The same file pins the two
records that close the former 80% bucket: the per-region rows of
`dqi_region_ledger.json` sum to 472,768.2 with the products separated, each lever's
measured delta equals the sum of the region deltas it caused, and the flag-reuse record's
observation matches its pre-registered prediction with all eight gates passing. A later
"region-by-region" claim that invents numbers for the 80% fails there.

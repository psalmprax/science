# The EEA and predicate layer: a measured plan

**Status:** plan, every number measured unless labelled · **Scope:** the 472,768.2-Toffoli
per-shot stream in the six-lever DQI arm that is not a field product. The reconciled ledger
for it, with the accounting rule and the first experiment's promotion gates, is
[`NONMULTIPLY_LEDGER.md`](NONMULTIPLY_LEDGER.md) ·
**Evidence:** `benchmarks/quantum_compilation/gf54_native_opcode.json` (three-term
decomposition, isolated blocks), `dqi_primitive_costs.json` (iteration and position slopes),
`designer_directions.json` (region ledger), `kickmix_mr_arm_costs.json` (lever costs);
rules in [`CIRCUIT_OPTIMISATION_RULES.md`](CIRCUIT_OPTIMISATION_RULES.md).

This is the plan that was promised before any edit to the decoder's control path. It says
what the remaining 2.744× is made of, which pieces have a measured ceiling, what has to be
proved for each, and which of them cannot reach the target no matter how well they are done.

## 1. Where the arm's 1,221,466.8 sits

Measured at the pinned instance, 64 shots, on the local `MR` build:

| part | Toffolis/shot | share | how measured |
|---|---|---|---|
| `ixor_gf2_mul` calls (10,510 × 66) | 693,660.0 | 56.8% | CCX difference between two correct arms |
| products in the inversion (155.23 × 54) | 8,382.7 | 0.7% | same, policy scoped to the inversion only |
| products in the division (864 × 54) | 46,656.0 | 3.8% | same, policy scoped to the division only |
| **everything else** (`C`) | **472,768.2** | **38.7%** | the residual of the measured total |
| total | 1,221,466.8 | | |

`C` is every Toffoli-like gate that is not a field product's AND: the EEA's control logic,
the predicates, the counter and halting decode, the shifts, the metadata, and the
multiply-and-add control. It is the object of this plan, and it is **1.062× the entire
445,229 target on its own**.

## 2. What `C` contains, measured

### 2.1 Measured exactly: the six unconditional regions

These run on every shot, so their static count *is* their executed count, and deleting them
from a static count (state-independent, so the difference is exact) gives an upper bound for
any predicate on them:

| region | Toffolis | share of `C` |
|---|---|---|
| `_update_flag` | 32,856 | 6.9% |
| `_initial_shift_loop` | 30,912 | 6.5% |
| `_update_metadata` | 12,432 | 2.6% |
| `_update_halting_counter` | 10,656 | 2.3% |
| `_pre_align_reg_b` | 7,392 | 1.6% |
| `_update_counter` | 222 | 0.05% |
| **total** | **94,470** | **20.0%** |

### 2.2 Slopes: the unit prices the EEA body is built from

`dqi_primitive_costs.json` measures **17,208 Toffolis per EEA iteration** and **351 per
Chien/Forney position** at the reference widths, by differencing call counts. Those are
prices per unit of work; §2.3 measures how many units the gated arm actually executes, so
the two together split `C` without multiplying a reference slope by a guessed count.

### 2.3 Measured: the rest of `C`, region by region

`C − 94,470 = 378,298.2` — the arm's conditional stream — is no longer a bucket. The local
`MR` build now counts executions per instruction (`sim.counting_op_executions`) and the
circuit carries its `builder.mark(...)` boundaries (`circuit.info_tags()`), so
`scripts/measure_dqi_region_ledger.py` partitions the executed stream exactly. The rows sum
to 1,221,466.8 and to `C` = 472,768.2 and the script refuses to write a record otherwise.

| region | non-product T/shot | share of `C` |
|---|---|---|
| `_shift_b_array_right` | 162,479.6 | 34.4% |
| `_multiply_and_add_or_sub` sweep | 143,038.0 | 30.3% |
| `_swap_a_and_b` | 64,224.0 | 13.6% |
| `_update_flag` | 32,856 | 6.9% |
| `_initial_shift_loop` | 30,912 | 6.5% |
| `_update_metadata` | 12,432 | 2.6% |
| `_update_halting_counter` | 10,656 | 2.3% |
| counter decode + `_update_counter` | 4,662 | 1.0% |
| `_pre_align_reg_b` / unalign | 7,392 | 1.6% |
| multiply-and-add control | 2,016 | 0.4% |
| quotient-division control | 1,728 | 0.4% |
| Forney control | 372.6 | 0.1% |
| **total** | **472,768.2** | **100%** |

The 222-Toffoli `_update_counter` sits inside the 4,662 decode row. The field products are
not in this table: they are 748,698.7 in the four regions that contain them
(`sweep_positions` 693,660.0, `quotient_div` 46,656.0, `forney_step` 8,382.7).

### 2.4 The first two controlled experiments, measured

* **Predicate reuse in `_update_flag`** (`scripts/measure_flag_predicate_reuse.py`): sharing
  the `(n_A == n_B)` comparison between the `c0` and `c1` sub-steps and clearing the scratch
  with `MR` saves 8 Toffolis per iteration = **1,776 per shot**, exactly the pre-registered
  prediction; `Q` unchanged at 1,913, `Q²T` −0.145%, all eight promotion gates pass. The
  `c2` term cannot join it: it tests `n_a == n_b + 1` between the `iadd` and the `isub`.
* **Halting-round gate** (`scripts/measure_eea_oracle_ledger.py`): `_update_flag` and
  `_update_metadata` are controlled on the one-hot `c0..c3`, so both are the identity in a
  round whose halting counter is non-zero. Reading `is_hc_zero` into a classical bit (a
  `cx` into the shared ancilla plus `MR`, no Toffoli) and gating both regions on it removes
  **1,224 per shot**, correct, with `Q` unchanged. This is a real gate, not an oracle: the
  arm already computes the predicate.

Neither is large, and their savings **do not add**. Stacked in one circuit
(`scripts/measure_dqi_combined_arm.py`), the reuse saves its 8 Toffolis only in rounds where
`_update_flag` executes at all: the halt gate removes that region from six dead rounds, so
the measurable total is `8 × 216 + 1,224 = 2,952`, not 3,000. Measured: **2,952 per shot**,
1,221,466.8 → **1,218,514.8** at `Q` = 1,913, correct, 222 iterations in all three arms.
The 48-Toffoli shortfall against the naive sum is the interaction, and the prediction was
written before the combined arm was built. Record:
`benchmarks/quantum_compilation/dqi_combined_arm.json`.

Together 2,952 per shot is 0.24% of the arm and 0.62% of `C`. The predicate block's
remaining headroom is small because most of `C` is not predicates — it is the sweep, the
shift and the swap, which the six levers already gate to 64.8%, 68.4% and 32.4% of their
static counts.

### 2.5 The schedule campaign, measured: −43.60%, not −0.24%

The paragraph above was written when "closing the rest needs a different schedule (L5)"
meant a different *decoder*. It does not: the schedule change is to **read the movement
predicate classically and then move**, which is the same gadget these two gates use, applied
at per-position grain. Every predicate register in this decoder is a basis state, so the
read is deterministic and the phase is untouched; a conditionally executed gate costs no
Toffoli on the shots whose predicate is zero, and most of `C` was quantum-controlled work
whose control *was* zero.

[`EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md) declares four hypotheses before
building the arms and measures them on the same instance and shots:

| gate | region | saving/shot | region after |
|---|---|---|---|
| the sweep's one-hot quotient selection | `sweep_positions` (non-product) | **103,968.0** | 143,038.0 → 39,070.0 |
| the shift's `do_shift` mask | `shift_b_array_right` | **65,895.0** | 162,479.6 → 96,584.6 |
| the product's `mul` control | the `ixor_gf2_mul` stream | **362,637.0** | 693,660.0 → 331,023.0 |
| **all three in one arm** | the whole arm | **532,500.0** | **1,221,466.8 → 688,966.8** |
| + the flag/metadata gate of §2.4, re-measured in the same record | `update_flag`, `update_metadata` | **2,952.0** | 1,221,466.8 → 1,218,514.8 |
| **all five in one arm** | the whole arm | **535,452.0** | **1,221,466.8 → 686,014.8** |

`Q` = 1,913 and 222 rounds in every arm, all correct, and the savings are exactly additive
twice over: `103,968 + 65,895 + 362,637 = 532,500` within the schedule gates, and
`532,500 + 2,952 = 535,452` for the stack — so this section's gates and §2.4's do not
interact. That is **−43.84% of the arm** and `Q²T` 4.4700e12 → 2.5105e12, with the target now
**1.541×** away instead of 2.744×. Three falsifiers (each a wrong condition) break the arm,
and two of them save *more* than the honest gate — which is the shape a real gate has. The
saving is work removed rather than relocated: the executed instruction stream falls 14.17%
with the Toffolis and the 30,253 added reads are reported (campaign note §3.0).

## 3. The arithmetic no control-logic work escapes

The arithmetic below is over **static** counts: `C` is the emitted non-product stream, and
the claim is that deleting parts of that stream cannot reach the target. It is correct as an
arithmetic of the circuit, and §2.5 shows what it does not cover: the *executed* residual is
690,017.0 Toffolis smaller than `C` plus products once the control-dependent work is read and
gated, because a quantum-controlled gate costs its Toffolis whether or not its control is
set. So read the last paragraph of this section as "no *static* edit", and see
[`EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md) §5 for what the executed arm leaves.

With `C` as measured and the products at their floor:

| scenario | `T_mean`/shot | vs 445,229 |
|---|---|---|
| today (54 ANDs, measured) | 1,221,466.8 | 2.744× |
| all products at the 23-AND bound | 864,060.6 | 1.941× |
| every product free, add kept | 598,888.2 | 1.345× |
| every product and add free | 472,768.2 | 1.062× |
| **products free *and* `C` cut 5.8%** | **445,229** | **1.000×** |

Read the last two rows together: the target needs `C ≤ 445,229 − P`, so either

- the field arithmetic goes to zero (impossible — the 23-AND bound costs 391,292, which
  forces `C ≤ 53,937`, an 89% cut), or
- `C` falls 5.8% *while the arithmetic is free*.

Neither is a control-logic edit *of the emitted circuit*. **No amount of EEA/predicate
tidying reaches 445,229 on this decoder** — the executed residual after the schedule
campaign is 686,014.8, still **1.541×** the target; a candidate that does reach it must
change what the decoder computes, not just how tidily it computes it.

## 4. The levers, ranked by measured ceiling

Each lever must arrive with all four of: two correct arms that differ only in it; executed
*and* static counts; the unitary argument; and a falsifier that fails. Ratios below are
"lever alone, everything else as measured".

| # | lever | ceiling | arm after | vs target | status |
|---|---|---|---|---|---|
| L1 | predicate/gate the six unconditional regions | 94,470 (20.0% of `C`) | 1,126,996.8 | 2.531× | measured and stacked: 2,952 per shot in one circuit (§2.4); the rest runs unconditionally |
| L2 | re-schedule the EEA so a division is not needed when `deg A < deg B` | 46,656 (3.8%) plus its control share, unmeasured | ≥ 2.64× | 2.64× | needs the schedule proof (see §5) |
| L3 | counter and halting decode | 10,878 (2.3%) | 1,210,588.8 | 2.719× | subset of L1's regions |
| L4 | change of basis without divisions | included in L2's products | — | — | needs a basis argument |
| L5 | a decoder that does not evaluate 4,095 positions or 222 EEA iterations per shot | `C` itself | — | — | the only class that can reach 445,229; not an opcode |
| L6 | **read a movement or product predicate and gate the work on it** | 532,500 measured (+2,952 with L1) | **686,014.8** | **1.541×** | **measured and stacked** ([`EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md)): sweep 103,968.0, shift 65,895.0, products 362,637.0, exactly additive, then stacked with L1's 2,952.0 to 535,452.0 |

L1 is worth 7.7% of the arm and is the largest single measured lever left in `C`. It is
also the lever the six-lever campaign already knows how to measure: gate a region on a
classical oracle bit, difference the executed counts, and require the arm to stay correct.
The oracle makes it an **upper bound**, exactly like `scripts/probe_dqi_predicate_gating.py`,
and an oracle-gated arm is inadmissible as a submission — L1's deliverable is a measured
ceiling and, only if the predicates turn out to be computable cheaply, a real gate.

L2 is where the reviewer's "EEA control" priority lands, and it is smaller than it looks:
the division products are 3.8% of the arm. The unmeasured part is the control logic around
them, which §2.3 puts at up to 378,298.2 and which L2 shares with L1 and L5.

## 5. Correctness obligations (why this is not an edit yet)

- **Systolic schedule.** The EEA is a fixed-point recurrence: every stage's operands are
  the previous stage's registers. Any change to when a division, swap or shift happens has
  to be shown gate-for-gate equivalent on the register contents, not just on the final
  syndrome. This is the obligation `dqi_gluzdov_v6.py` did not discharge: its docstring
  records the architecture as a structural scaffold with the recycling "implemented in the
  simulation model, not in the circuit", and no receipt.
- **Early halting and asynchronous pointer substitution are closed until proved.** Both
  have already failed here. A halting predicate that is *almost* right leaves the syndrome
  registers wrong on the shots where it fires early; a pointer substitution that reorders
  dependent terms is only sound if the reordering is a permutation of commuting gates, which
  the EEA schedule does not give for free.
- **No deletion-differencing.** The 86.33% multiply share came from deleting a block and
  differencing executed counts: the state went wrong, the predicates read garbage, and the
  difference was not the block. Every number in this plan comes from pairs of *correct*
  arms, or from static counts of unconditional regions.

## 6. The measurement programme, in order

1. ~~**Arm-level region gating (L1).**~~ **Done.** `scripts/measure_dqi_region_ledger.py`
   partitions the executed stream with per-instruction counts; §2.3 is the delivered ledger
   for all of `C`, not just the 20%, and §1.2 of the non-multiply ledger records what each
   existing lever removed per region.
2. ~~**EEA body accounting (L2/L5).**~~ **Done, and it narrows L2.** The arm runs 222
   rounds at about 1,955 non-product Toffolis each (434,092 of `C`; the remaining 38,676 is
   the initial shift, the aligment pair and the Forney control). The sweep is the only
   round region large enough to matter, and it is already gated on `c0 ∨ c3`. L2's division
   products are 46,656 (3.8%) and the division's control share is 1,728, so a division-free
   step is worth at most ~4% — consistent with §3's conclusion that no control-logic edit
   reaches 445,229.
3. **A correct division-free step, if 2 justifies it.** Replace the division by a
   conditional subtract when `deg A ≥ deg B` holds, and prove the schedule; falsifier: the
   under-gated arm must fail and the six-lever verification must pass.
4. ~~**Read the control, then gate the work (L6).**~~ **Done, and it is the largest lever in
   this plan.** [`EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md) measures the sweep's
   selection, the shift's mask and the product's `mul` control read classically and gated:
   532,500 per shot in one arm, −43.60% (−43.84% with L1 stacked on top), `Q` and the 222
   rounds unchanged, three falsifiers failing. The reviewer's shift/sweep/swaps priority is
   discharged by it: the sweep and the shift are promoted, their composition with the swap
   is measured to be worth exactly 0.0, and the shift's remaining 96,584.6 is declared
   closed.
5. **Only then** consider a different decoder (L5). The arithmetic of §3 says the target
   needs one; the measurements of 1–4 say which part of the arm it has to remove — the
   686,014.8 that remains, of which 386,061.7 is field products and 299,953.2 is
   non-product, with the swap's 64,224 and `shift_step`'s 30,912 the largest untouched
   items.

## 7. Workspace is not the constraint

The reviewer's second priority is already answered: the 19-qubit multiplier moved the whole
arm's peak from 1,936 to **1,913** with `T` unchanged, so the local saving *is* a global
saving. But the arm is still **+20** over the reference (1,893) and **+124** over the 1,789
target, and the workspace is the smallest of the three remaining costs: 23 qubits against
472,768.2 Toffolis and 748,698.6 product Toffolis. Nothing in this plan is qubit-driven.

## 8. Pins

`tests/test_dyn_mr_gated_arms_kickmix.py` pins the decomposition this plan rests on: the
three terms sum to the measured arm to 0.1 Toffoli, the isolated blocks are each nine
Toffolis per executed product, `C` is 472,768.2, and `C` alone exceeds the target. A future
edit that changes the residual, or a plan that claims a lever bigger than its measured
ceiling, fails there.

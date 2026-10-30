## 2026-10-30 (later) — Experiment D2 and the campaign fix: `hmr` is not a clear, the compile-able candidate is not a submission, and the fused-shift arms finally measure something

- **EXPERIMENT D2 — D's CANDIDATE IS COMPILABLE AND WRONG, AND THE 0.952 IS WITHDRAWN.**
  `scripts/measure_dqi_official_candidate.py`
  (`docs/DQI_OFFICIAL_COMPATIBILITY.md` §3 and §5) checks the one thing Experiment D did not:
  the **phase**. The DQI submission is scored on phase as well as value (the verifier's
  `phase_err = (phase_flipped ^ phases) & ~val_err`), and a clear implemented as `hmr` is not
  phase-safe. Measured one qubit at a time over 200 shots: `hmr` on `|1>` flips the tracked
  phase on **97** of them — the same 97 as its measured ones, because an X-basis measurement of
  `|1>` puts a −1 on the branch that measures 1 — while `del_and`, which is the same
  measurement followed by a classically conditioned `CZ` (the phase fixup), flips **0**, and
  neither flips anything on `|0>`. So a clear whose value is not a *known AND* has no fixup to
  apply, and substituting `hmr` for `mr` there is a randomisation, not a clear. At the pinned
  instance over 64 shots on the pinned extension: reference **1,893 / 5,725,071.609375, 0 value
  errors, 0 phase errors**; Experiment D's 54-AND candidate **1,896 / 5,433,399.609375, 0 value,
  37 phase errors**; and a third arm that removes the reverse EEA outright — the forward EEA on a
  **CNOT copy** of the syndromes, so the input is never written, nothing is restored and no
  qubit's value is read, which is exactly what the `clear_eea` lever needs — at **2,736 /
  3,438,873.609375, 35 phase errors**, twice over: its clears are `hmr` too, and holding the
  whole scratch live costs more `Q` than the pass it deletes. D's candidate builds, executes, and
  is **not** a submission; its **0.952 score ratio is withdrawn** and its Toffoli count stands.
  The measured floor is now the shipped reference: **the cheapest *correct* circuit the pinned
  binary can compile is the one it ships.**
- **EXPERIMENT C, C3 — THE MECHANISM IS MEASURED, AND IT IS THE FLAG.** The probe now reads both
  of the exchange's operands where the exchange reads them — `c3` and the *post-update* `flag`,
  one qubit per round — at the pinned instance over 64 shots and 222 rounds: `c3` fires **4,608**
  times (**72.0 on every shot**, invariant), the exchange fires **2,052** (27–33 per shot), and
  the flag is **clear in 2,556 of the 4,608** `c3` rounds (55.5%). So the flag is not redundant in
  a `c3` round; exchanging on `c3` alone performs ~72 permutations where the decoder asks for
  27–33, which is why C3's 73.0 never appeared and its arm (852,267.4, exchange row 190,888.0)
  is wrong. The same table is why the campaign's S3 arm has nothing to skip: in the 2,052 rounds
  where the flag *is* set, `ctrl`'s one-hotness zeroes the shift's other two terms.
- **THE CAMPAIGN'S DEAD `_shift_a_and_b` IS FIXED AND ITS TWO ARMS RE-MEASURED.** The method is
  now named `_shift_b_array_right`, the one the reference and `MRGatedDecoder` call, so the gates
  inside it are emitted. `dqi_schedule_campaign.json` was re-run with everything else reproducing
  to the Toffoli (control **1,221,466.8**, `everything` **688,966.8**, `stacked` **686,014.8**):
  **S3 `composed` saves exactly 0.0 and is correct** — the campaign's conclusion, now measured
  rather than assumed — and **S3b `drop_shift_c3` saves 45,396.1 and is incorrect**, which
  falsifies S3b's premise (the `¬flag ∧ c3` branch is live). `drop_shift_c3` is recorded as a
  *predictive* arm: its correctness is reported, not required, so a falsification is the result
  rather than an invalid run, and the S3b check now carries the declared 0.0 beside the measured
  45,396.1. The movement script's independent re-measurement against a base carrying the two
  promoted gates (`campaign_none` = 1,218,514.8) agrees: 0.0/correct and 45,377.7/incorrect.
- **Pins.** `tests/test_dqi_official_candidate.py` (8 tests) pins the phase mechanism, the three
  arms' `Q`/`T`/error counts, that no ratio is quoted for a phase-broken candidate, and that the
  cheapest correct circuit is the reference; `tests/test_dqi_movement_schedule.py` and
  `tests/test_dqi_schedule_campaign.py` were updated to pin the rebuilt arms and the exchange
  predicate rather than the vacuous 0.0s. `docs/DOCUMENTATION.md` indexes the new script;
  `docs/EEA_SCHEDULE_CAMPAIGN.md` §3.3 and `docs/DQI_MOVEMENT_SCHEDULE.md` §5–§7 carry the
  corrections.

## 2026-10-30 — Experiments C and D, and the amplitude tier: the exchange stays, one candidate compiles, and the licence is exactly the basis

- **EXPERIMENT C — MEASURED, NEGATIVE, WITH A CORRECTION.**
  `scripts/measure_dqi_movement_schedule.py` (`docs/DQI_MOVEMENT_SCHEDULE.md`) asks the only
  question left open by the schedule campaign: can the decoder's A/B exchange — 64,224
  executed Toffolis per shot, 72 controlled 892-qubit permutations — be resolved at **build
  time** by reassigning which physical register plays the `A` role? That is the only kind of
  saving the pinned competition binary could compile, because it is in the emitted circuit
  rather than in an executed count. The prediction was declared before any arm existed: the
  rebased arm must save exactly the region (64,224.0 executed, **198,024 static**), and is
  correct *if and only if* the exchanging-round set is the same for every shot. At the pinned
  instance over 64 shots: control **686,014.8** at `Q` 1,913 with its 64,224.0 row; a harness
  arm that re-emits the incumbent's exchange from the new script reproduces it to **±0.0**, so
  the comparison is licensed; the rebased arm zeroes the region and drops 198,024 static
  Toffolis and is **incorrect** (syndromes restored, junk and phase not); and the probe reads
  `c3` once per round to find **10 distinct patterns across 64 shots** while every shot
  exchanges exactly **72** times. The schedule is therefore *cardinality-stable and
  order-unstable* — that is why the incumbent's row is exactly `72 × 892` on every shot and
  why no build-time period reproduces it. The arm stays at 686,014.8, 1.541× the target.
- **CORRECTED — two of the campaign's arms measured nothing.** `composed` (S3) and
  `drop_shift_c3` (S3b) both reported 0.0, which the campaign read as the extra condition
  being unsatisfiable. It is not: the code implementing both lives in a method named
  `_shift_a_and_b`, and nothing calls it (the reference and `MRGatedDecoder` call
  `_shift_b_array_right`). Measured at m=15/n=6/GF(2^4): `none`, `composed` and `drop_shift_c3`
  emit **149,337 instructions and 16,610 static Toffolis each** — the two “arms” are the control
  circuit, and their 0.0 is vacuous. The predicate argument (a one-hot `ctrl` and one flag value
  cannot be both) survives; the measurement that was supposed to support it does not, and
  `docs/EEA_SCHEDULE_CAMPAIGN.md` §3 now says so.
- **EXPERIMENT D — THE MISSING `MR` WAS NOT THE ONLY BLOCKER, AND ONE CANDIDATE COMPILES.**
  `scripts/measure_dqi_official_compatibility.py` (`docs/DQI_OFFICIAL_COMPATIBILITY.md`) mines
  the whole surface of both builds. The pinned competition binary has **no `mr`, no `h`, no
  `gf_product`, no per-instruction counts**, but it *has* `hmr`, `bit_store`,
  `push_condition` and `init_and`/`del_and`/`cswap` — and `del_and` already costs **0**
  Toffolis there, because an AND clear can be a measurement where a read cannot. So the arm
  needs **two** surfaces it does not have, not one: the 54-AND product policy is worth
  **523,849.1** per shot on the reference and was never recorded as a blocker. Both builds
  count the same reference identically (1,893 / 5,725,071.609375, per-shot arrays equal), and
  the pinned row reproduces the recorded constant exactly. The positive result: a candidate the
  pinned binary **can** build — the reference algorithm with the 54-AND multiply emitted as
  gates and its workspace cleared with `hmr` instead of `mr` — at `Q` 1,896 and
  **5,433,399.609375** per shot, syndromes restored on all 64 shots, a **0.952** score ratio to
  the shipped reference (**4.79% better**) and still **7.92×** worse than the arm. It is the
  first candidate in this repository the competition build can construct at all, and it is a
  floor rather than a contender: the arm's −43.84% is a read-dependent saving, and a quantum
  control cannot express it (the reference's every gate is quantum-controlled and every one
  executes).
- **THE AMPLITUDE TIER, AND THE LICENCE IT PINS.**
  `apedp_dyn/quantum_compilation/statevector_sim.py` plus
  `scripts/measure_vqc_licence_boundary.py` (`docs/VQC_AMPLITUDE_TIER.md`) compare the
  controlled gate the decoder emits with the read-then-gate map the arms substitute, at
  amplitude level, because every count in this repository is a basis-state count and the
  campaign's licence (P1) is a statement about the *inputs*. Result: on **all eight basis
  inputs** the two maps are the same state up to global phase, the classical bit equals the
  control, and the ancilla returns to `|0>`; on a **superposed control** the gated map branches
  **twice at 0.500** with a fidelity of **0.500** against the coherent reference, and the
  control's off-diagonal goes **0.500 → 0.000** — a complete dephasing, not a perturbation. The
  same tier settles the substitution Experiment D needed: `hmr` alone returns a random outcome
  on `|0>` (it measures in the X basis), while `h; hmr` is `mr` branch for branch on basis *and*
  superposed inputs — so the blocker is the missing Hadamard, not the measurement instruction.
  The tier is cross-checked against `kickmix` on the gadget's eight basis inputs (identical),
  and the arm's 1,913 qubits are reported as 10^575.87 amplitudes rather than pretended.

## 2026-10-30 — The schedule campaign: read the predicate, then move; 1,221,466.8 → 686,014.8

- **EXPERIMENT B — MEASURED AND CLOSED, NEGATIVE.** The product stream's own operand
  predicate: every surviving product is `ixor_gf2_mul(q_i, a_j)`, so a zero multiplier makes
  it the identity, and the argument that licenses S1–S4 licenses gating on it (`S6_zero_q`,
  declared **0.0 saving** before the arm was built). The arm is **correct** — the predicate is
  admissible — and **loses**: 997,795.6 Toffolis per shot, **+311,780.7 over the five-gate
  incumbent**, adding 327,094 mid-circuit reads (357,347 in total against 30,253) and 956,116
  instructions. Two structural reasons, both measured: (a) the sweep's selection chain
  `cswap`s `q_i` into the a-lanes, so the multiplier is **not** a round-level quantity — the
  first version of this arm read it once per round and was *incorrect*, and only a
  per-position read is sound; (b) a per-position read measures at 0.95 Toffolis against a
  66-Toffoli product that is skipped only inside the `mul` window, so the gate needs the
  multiplier to be zero in ~3% of live positions, and the loss is almost exactly the read
  cost — the multiplier is essentially never zero where the product executes. Its falsifier
  (the inverted condition) is wrong and informative (316,701 product Toffolis skipped). The
  331,023 surviving products are work the data path asks for; removing them needs a cheaper
  multiply, not another classical read.
- **MEASURED — the shift/sweep/swap priority is discharged, and it was worth 43.6% of the
  arm.** `scripts/measure_dqi_schedule_campaign.py` (`docs/EEA_SCHEDULE_CAMPAIGN.md`) applies
  the read-then-gate gadget at per-position grain: every predicate register is a basis state,
  so `cx` + `MR` through the shared ancilla yields a deterministic classical bit at zero
  Toffoli and no phase, and a gate conditioned on a zero bit costs nothing on that shot. At
  the pinned instance, 64 shots, `Q` 1,913 and 222 rounds in every arm: the sweep's one-hot
  quotient selection **−103,968.0** (chain 105,696.0 → 1,728.0), the shift's `do_shift` mask
  **−65,895.0** (chain 131,220.0 → 65,325.0), the product's `mul` control **−362,637.0**
  (product stream 693,660.0 → 331,023.0); all three in one arm **1,221,466.8 → 688,966.8**
  (−532,500.0, exactly additive), **1.547×** the 445,229 target instead of 2.744×, `Q²T`
  4.4700e12 → 2.5213e12. Record: `benchmarks/quantum_compilation/dqi_schedule_campaign.json`.
- **STACKED WITH THE PRIOR COMBINED ARM.** The same record now re-measures the
  flag/metadata gate the previous session promoted (2,952.0 → 1,218,514.8, zero reads) and
  puts all five gates in one circuit: **1,221,466.8 → 686,014.8 per shot (−535,452.0,
  −43.84%)**,  exactly `532,500.0 + 2,952.0`, so the two sessions' gates do not interact,
  **1.541×** the target, `Q²T` 4.4700e12 → 2.5105e12. The cross-report inference "the two
  gates compose" is now a measurement in one record — and a declared one: `prediction.S5_stack`
  records the arithmetic (`532,500 + 2,952 → T = 686,014.8`) written after the schedule arms
  and **before** the five-gate arm was built, and the measurement matched it to **0.0**
  Toffoli (`observation.stacked_error_per_shot`).
- **NOT RELOCATED — REMOVED.** Counting the whole stream on the same runs: executed
  instructions fall 37,808,019.7 → 32,451,795.1 per shot (−5,356,224.6, −14.17%) while the
  Toffolis fall 535,452, i.e. 10.12 instructions per removed Toffoli — the expansion the
  removed `CCX`es had. The added cost is 30,253 mid-circuit reads per shot at 2 instructions
  each (60,506, 1.1% of what was removed), reported for every arm. The honest claim is: the
  nonlinear work falls 43.84%, the stream falls with it, and the price is measurement depth.
- **DECLARED BEFORE MEASURED.** The predictions are computed from the control arm's own chain
  costs and printed before any variant exists (`prediction.written_before_variants`): S1 ≥
  100,368 from a burn of `12 × 2 × 222`, S2 **no saving** (`do_shift` is the XOR of three
  threshold intervals, not a one-hot), S3 and S3b exactly 0.0, S4 no saving. S1 came out at
  103,968.0 (the bound assumed two selections in all 222 rounds; 144 occur); S2 at 65,895.0,
  which **closes** the shift half rather than promoting it — half the chain is movement the
  data path needs; S3/S3b at 0.0 with the arm still correct, which is the measured form of
  "the shift and the swap cannot be fused for gain".
- **FALSIFIED THREE WAYS.** Every variant was rebuilt with its condition made wrong and all
  three are incorrect: the stale selection (skips 104,768.1 of its chain and *costs* 217,720.1
  more than the control, because a wrong state changes the control flow), the inverted mask
  (skips 69,311.6 and saves 79,854.5 — more than the honest gate's 65,895.0, which is what
  makes it wrong), and the previous slot's mask (70,351.1 skipped, 80,612.0 saved, wrong).
  The record measures skipped work on the chain the condition gates, never on the arm total,
  and a falsifier that skips nothing is recorded as vacuous instead of as a pass.
- **CORRECTED — what the arithmetic of the reachability note covers.** §3 of
  `docs/EEA_AND_PREDICATE_PLAN.md` counted the *static* residual, so its "`C` alone is 1.062×
  the target" was read as "control-logic work cannot help". The executed residual is not
  `C`: 532,500 of it was quantum-controlled work whose control was zero. The target is still
  out of reach — 686,014.8 is 240,785.8 above it, and the largest remaining items are the
  swap's 64,224, the shift's 96,584.6, `_update_flag`'s 30,240 and `shift_step`'s 30,912 — but
  it is now 1.541× away, not 2.737×.
- **STATED COST MODEL.** An executed count that depends on classical conditions is a count
  under adaptive execution. The benchmark permits it (the reference's own `error_mbuc` is
  classical feed-forward, and the six levers were promoted on the same mechanism), but this
  campaign applies it at per-position grain — thousands of mid-circuit measurements and
  classical decisions per shot. A submission that forbids mid-circuit classical control could
  not claim these counts, and none is claimed. No receipt exists either: the arm still needs
  `MR`, which the pinned competition binary does not have.

## 2026-10-29 — The two gates stack to 2,952; the GA leak and the search ranker are fixed

- **MEASURED — the two promoted gates compose, and their savings do not add.**
  `scripts/measure_dqi_combined_arm.py` runs the predicate-reuse arm and the halting-round
  gate in one circuit. The prediction is written before the combined arm is built, from that
  run's own control and halt-gated arms: the reuse saves 8 Toffolis only in rounds where
  `_update_flag` executes, and the halt gate removes it from six dead rounds, so
  `8 × 216 + 1,224 = 2,952`, not the naive `1,776 + 1,224 = 3,000`. Observed: **2,952**
  (1,221,466.8 → **1,218,514.8** per shot, `Q` = 1,913, correct, 222 iterations in all three
  arms, region rows partition each total). The 48-Toffoli difference is the interaction, and
  the record names which model matched. Record:
  `benchmarks/quantum_compilation/dqi_combined_arm.json`.
- **FIXED — the evolutionary search no longer selects on the confirmation rows.**
  `evolve()` picked `grouped_pick` by minimising error on the supplied `x_test`/`y_test`,
  which turned the confirmation split into a second training set. Selection is now a
  deterministic 5-fold inner cross-validation over the training rows
  (`SearchConfig.selection_folds`), with coefficients refitted inside each fold; the test
  rows are read only after the pick is fixed, to report the new
  `SearchResult.confirmation_mse`. `selection_method`/`selection_folds` record which path
  was taken, and the fallback when no candidate survives the folds is the lowest training
  error — never the test rows. Both the native and Python paths were changed. Pinned by
  `tests/test_evolutionary_search.py::test_grouped_pick_is_selected_without_reading_the_confirmation_rows`
  (inverting the confirmation targets must leave the pick identical and only worsen the
  confirmed error).
- **FIXED — the search ranker preferred the worse candidate.** `ValidationPipeline`'s
  `overall` was `0.4 * max(0, cv_mean - cv_std) + ...` with `cv_mean` an *error*: a perfect
  fit scored 0.56 and a poor one 0.83, so `HierarchicalSearch.search` sorted its worst
  candidate first. `overall` is now `0.4 * max(0, 1 - (cv_mean + cv_std)) + ...`, recomputed
  in Python from the returned components on the Rust path too (the kernel carried the same
  sign and is corrected in `validation.rs`). Measured effect in
  `benchmarks/search_engine_gaps.json`: classic templates recovered 0/12 planted laws with
  the inverted score and 2/12 with the fix.
- **MEASURED — what the search templates cannot reach, and closed most of it.**
  `scripts/measure_search_engine_gaps.py` plants 12 law families, searches on 200 rows and
  scores the returned candidate on 100 untouched rows (coefficients refitted on train).
  Classic `HierarchicalSearch` recovered **2/12**, `beam_search_expressions` **1/12**.
  The extended template set (reciprocal, square root, cubic, `exp(±x)`, log, cosine,
  rational saturation, ratios, a variable in an exponent, plus an exact least-squares scale
  and offset fitted to every candidate) recovers **11/12**, and `beam_search_expressions(
  include_functions=True)` **9/12**. Genuinely still unreachable, and reported as such: a
  sinusoid of non-unit frequency (the engine fits affine coefficients, not a frequency).
  Records: `benchmarks/search_engine_gaps.json`.
- **ADDED — patent prior-art search with a provenance gate.** `apedp/patent_prior_art.py`
  indexes patent records whose claims carry a normalised formula and audits a candidate
  against them: rename-insensitive algebraic equivalence first, structural signature second,
  otherwise `NOT_FOUND_IN_PATENT_CORPUS`. A record that claims a real patent without an
  http(s) source and a retrieval date is refused at load (`PatentProvenanceError`), and
  claims with no formula are counted and reported (`NO_SEARCHABLE_CLAIMS`) rather than
  silently skipped. No patent corpus is bundled; the loader and the documented source
  options are. Pinned by `tests/test_dyn_patent_prior_art.py` (9).

## 2026-10-29 — The 80% is split: per-instruction counts, and two gates that pass

- **INSTRUMENTED:** the local `MR` build now counts executed shots **per instruction**
  (`sim.counting_op_executions` / `sim.operation_execution_counts()`), and the circuit
  exposes its `builder.mark(...)` boundaries and Toffoli-like mask (`circuit.info_tags()`,
  `circuit.op_toffoli_like`). The simulator applies the same per-shot condition mask as
  `operation_statistics()` and masks off the SIMD lanes beyond `batch_size`, so a region's
  sum reconciles with `num_toffoli_like` exactly rather than by construction.
- **MEASURED — the named 80% bucket is gone.** `scripts/measure_dqi_region_ledger.py`
  partitions the six-lever arm's 1,221,466.8 Toffolis/shot into regions that sum to the
  472,768.2 non-product residual: `shift_b_array_right` 162,479.6, `sweep_positions`
  143,038.0, `swap_a_and_b` 64,224.0, `_update_flag` 32,856, `_initial_shift_loop` 30,912,
  `_update_metadata` 12,432, `_update_halting_counter` 10,656, counter decode 4,662, the
  alignment pair 7,392, multiply-and-add control 2,016, quotient control 1,728, Forney
  control 372.6. It also records what each lever removed per region, and checks each
  lever's arm-level delta against the sum of its region deltas. Record:
  `benchmarks/quantum_compilation/dqi_region_ledger.json`.
- **PROMOTED — predicate reuse in `_update_flag`.** `scripts/measure_flag_predicate_reuse.py`
  shares the `(n_A == n_B)` comparison between the `c0` and `c1` sub-steps and clears the
  scratch with `MR` instead of uncomputing it: 8 Toffolis per iteration × 222 = **1,776
  per shot**, exactly the prediction computed from unit costs before the arms were built
  (the first prediction used the 12-bit field degree instead of the 8-bit register width
  and over-predicted by 4 per iteration; the mistake and the corrected run are in the
  record). `_update_flag` 32,856 → 31,080, `Q` unchanged at 1,913, `Q²T` −0.145%, all eight
  promotion gates pass, 222 iterations in both arms. Record:
  `benchmarks/quantum_compilation/dqi_flag_predicate_reuse.json`.
- **PROMOTED — the halting-round gate.** `scripts/measure_eea_oracle_ledger.py` gates
  `_update_flag` and `_update_metadata` on a classical read of `is_hc_zero`: both regions
  are controlled on the one-hot `c0..c3`, so in a round with a non-zero halting counter
  their net effect is the identity and skipping them is exact. It removes **1,224 per shot**
  (888 + 336), correct, `Q` unchanged. The predicate is already computed by the arm, so this
  is a real gate, not an oracle upper bound. Record:
  `benchmarks/quantum_compilation/dqi_eea_oracle_ledger.json`.
- **NARROWED:** together the two gates are 3,000 Toffolis/shot, 0.25% of the arm and 0.6% of
  the non-product residual. Most of the residual is not predicates — it is the sweep, the
  shift and the swap, which the six levers already gate to 64.8%, 68.4% and 32.4% of their
  static counts. No predicate edit closes the 2.744×; §3 of `EEA_AND_PREDICATE_PLAN.md`
  still stands.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` pins both records: the region rows
  sum to 472,768.2 with products separated, each lever's delta equals its region deltas,
  the flag-reuse observation matches its prediction, and all eight gates pass.

## 2026-10-29 — Ratio baselines labelled: 4.687× and 4.258× are different questions

- **CORRECTED:** the update quoted "4.687× against 4.320×" without saying which baseline
  each ratio used, which reads as a regression once a controlled comparison is made. The
  measure now records five comparisons, each with its baseline
  (`gf54_native_opcode.json:comparisons`): the multiply saves **9.15%** on the reference and
  **7.83%** on the gated arm; the arm is **4.687×** faster than the *unchanged* reference
  (the scored comparator) and **4.258×** faster than a reference given the same opcode; its
  `Q²T` falls **5.97%** from the built-in arm to the final one.
- **EXPLAINED:** 4.258× is below the built-in arm's 4.320× because the policy helps the
  reference more than the arm — the arm's Forney body is gated, so fewer of its inversion
  products execute. Both ratios are correct; only a reader told which baseline each uses
  can tell that they are not the same claim.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` recomputes all five from the
  recorded means and requires the same-policy ratio to sit below the built-in one, so a
  future edit cannot quietly swap the baselines.
- **NOTATION:** every figure in this work is GF(2^12) with 12-bit field elements
  (`DEFAULT_DEGREE = 12`); the two new notes spell it in ASCII so the superscript cannot be
  misread as 2^25.

## 2026-10-29 — Reconciled: the official 5,725,108.38 and our 5,725,071.6 are one metric

- **RECONCILED:** `scripts/reconcile_dqi_resource_baseline.py` (record:
  `benchmarks/quantum_compilation/dqi_resource_reconciliation.json`,
  note: [`docs/DQI_RESOURCE_RECONCILIATION.md`](docs/DQI_RESOURCE_RECONCILIATION.md)). The
  locked official reference row is 1,893 qubits / 5,725,108.382109375 Toffolis, and its
  fractional part is a multiple of 1/102,400 but **not** of 1/64 — so it is the verifier's
  default-shot mean, while this project's 5,725,071.609375 is a 64-shot mean. Same decoder,
  same metric (`num_toffoli_like`, which the verifier itself averages), same instance.
- **MEASURED:** the reference decoder was built and simulated on the **pinned competition
  binary** and on the local `MR` rebuild over the same 64 sampled syndromes. The per-shot
  Toffoli arrays are identical element by element and both means are 5,725,071.609375 at
  1,893 qubits, so the `MR` patch is count-neutral for a circuit that emits no `MR`. The
  36.77 gap to the official row is 0.78 standard errors of a 64-shot mean: a sample, not a
  discrepancy.
- **ESTABLISHED — no receipt for the arm.** Building the six-lever arm on the pinned binary
  fails with `AttributeError: 'Builder' object has no attribute 'mr'`: its clears need an
  opcode the competition build does not have, and `apedp_dyn/quantum_compilation/contracts.py`
  pins receipts to that build's hash. The arm's 1,221,466.8 at `Q` 1,913 is a local
  measurement, not an official result; `verify_submission_dqi.py` has no Toffoli threshold
  either — it returns `PASS` on correctness and reports `R(eps_plus) · Q² · T`.
- **DOCUMENTED:** what 445,229 is — a *derived target* (`Q = 1,789`, `T = 445,229`, chosen so
  the product reaches a 14.397× portfolio gain), not a measurement, a competitor's figure or
  an acceptance threshold. `docs/BENCHMARK2_TARGET_REACHABILITY.md` now says so before it
  compares anything to it.

## 2026-10-29 — Product window removed: 19 workspace qubits, `Q` 1,935 → 1,912, same T

- **MEASURED:** `gen_gf54_mul` no longer materialises the 2m-1-bit product window. Reduction
  is GF(2)-linear, so each Karatsuba sub-product is added to the target through a
  `x^d mod p` table as soon as it is computed; the peak scratch is one sub-product window at a
  time — **19 qubits at GF(2^12) instead of 42**, 12 + 19 when controlled instead of 54. The
  AND count is untouched, so `T_mean` is too: the two records differ only in `Q`.
  Correctness is unchanged and re-checked: multiply, inversion and division all 0 bad on 32
  random basis states, operands restored, workspace `|0>`, phase untouched.
- **MEASURED:** with the opcode emit at the pinned instance over 64 shots the reference stays
  **5,725,071.6 → 5,201,222.5 per shot (9.15%)** but `Q` goes **1,893 → 1,912** (was 1,935),
  and the six-lever arm **1,913** (was 1,936) at 1,221,466.8 per shot. The cost is 22 more
  CNOTs per controlled multiply (600 against 578): the reduction is applied per sub-product
  rather than once per window bit. Since the score charges `Q` squared, 23 fewer qubits at
  equal `T` is a 2.4% improvement in `Q^2 T` over the 42-qubit emit.
- **NEW:** [`docs/BENCHMARK2_TARGET_REACHABILITY.md`](docs/BENCHMARK2_TARGET_REACHABILITY.md)
  lists every opcode class against the 445,229 target with the measured arithmetic. (The
  two-term form first recorded here — `527,806.8 + 10,510 × (ands + 12)`, giving 2.012× at
  the 23-AND bound and 1.186× free — was superseded the same day by the three-term
  decomposition below: same totals, but it folded the 1,019.23 inversion/division products
  into its constant at 54 ANDs each, so "free multiply" meant "free `ixor` only". The
  correct figures are 1.941× and 1.062×.) It also records why no Fast-Fourier/cyclotomic
  selector was added: the Chien search's 69 constant multiplies per position are CNOT-only
  (0 Toffoli in `dqi_primitive_costs.json`), so a transform can only trade CNOTs for CNOTs
  and would pay qubits, which the score squares.

## 2026-10-29 — The reconciled non-multiply ledger, and the first experiment's gates

- **MEASURED — ledger.** `scripts/attribute_dqi_nonmultiply_tcount.py` (record:
  `dqi_nonmultiply_ledger.json`, note: [`docs/NONMULTIPLY_LEDGER.md`](docs/NONMULTIPLY_LEDGER.md))
  assigns every executed Toffoli of the six-lever arm to exactly one category and checks
  that they sum to the simulator's total: **693,660.0** (10,510 controlled multiplies),
  **8,382.7** (155.23 inversion products), **46,656.0** (864 division products) and
  **472,768.2** (non-product residual, 38.7%) = **1,221,466.8**. The static ledger reconciles
  too: 3,486,597 = 94,470 + 3,392,127.
- **BOUNDED, NOT GUESSED.** The residual's six unconditional regions are exactly 94,470
  (20.0%). The other **378,298.2 (80.0%) is one named unattributed bucket**: the simulator
  reports per-shot counts by operation type, not per operation, so the conditional EEA body
  cannot be split by region; the ledger records the gap, how to close it (per-op execution
  counts, or per-region oracle gating that measures the removable part) and refuses to
  estimate it into the table.
- **CORRECTED INVARIANT.** The review's accounting invariant of 527,806.8 is superseded by
  472,768.2 for the reason in the previous entry: 55,038.4 of the old constant was the
  1,019.23 inversion/division products, which are field arithmetic, not decoder logic.
- **NEW — first experiment protocol.** `_update_flag` plus the counter/halting predicates
  (43,734 Toffolis, 9.3% of the residual) against the unchanged 222-iteration arm, with a
  prediction written before the edit and eight promotion gates (exact output equivalence,
  measured phase, clean workspace, fixed iteration behaviour, both T-counts, `Q`, `Q²T`
  from the same run, frozen source and record). Pinned in
  `tests/test_dyn_mr_gated_arms_kickmix.py`.

## 2026-10-29 — Three-term decomposition, and the measured EEA/predicate plan

- **CORRECTED — same totals, honest labels.** The arm's Toffolis are
  `T = 472,768.2 + 10,510 × (ands + 12) + 1,019.23 × ands`, not
  `527,806.8 + 10,510 × (ands + 12)`. The 1,019.23 are the products inside the arm's
  inversion (155.23) and division (864), each 54 ANDs with **no** controlled add, so the
  old constant silently contained 55,038.4 of field arithmetic. Consequences: freeing
  every field product gives **1.345×** the target (not 1.469×), freeing the adds too gives
  **1.062×** (not 1.186×), and the 23-AND bound lands at **1.941×** (not 2.012×).
- **MEASURED:** the decomposition, by scoping `gf_product` to one entry point at a time on
  both the reference and the arm — the reference runs **888** divisions per shot (47,952
  Toffolis saved) and the arm **144** (7,776), and the arm's gated Forney block runs
  **31.05** inversions (1,397.11). `gf54_native_opcode.json:isolated_blocks` records all of
  it, and its `check_terms_sum_to_arm` is exact.
- **NEW:** [`docs/EEA_AND_PREDICATE_PLAN.md`](docs/EEA_AND_PREDICATE_PLAN.md) is the measured
  plan the next layer was promised: the residual `472,768.2` (38.7% of the arm, 1.062× the
  target on its own), the exact 94,470-Toffoli ledger over the six unconditional regions
  (20% of it), the EEA slopes (17,208/iteration, 351/position), the five levers ranked with
  ceilings, the correctness obligations (systolic schedule gate-for-gate; early halting and
  asynchronous pointers closed until proved; no deletion-differencing), and the arithmetic
  showing no control-logic edit reaches 445,229: `C ≤ 445,229 − P` needs either zero-cost
  arithmetic (impossible at the 23-AND bound) or a 5.8% cut in `C` *with* the arithmetic
  free.
- **PINNED:** three new kicks in `tests/test_dyn_mr_gated_arms_kickmix.py` — the three-term
  sum, the 472,768.2 residual with `C > target`, and the 94,470 region ledger at 20.0% of it.

## 2026-10-29 — The 54-AND multiply as an opcode: 4.687×, and 35% of it is Forney

- **NEW:** `gen_gf54_mul` in `scripts/kickmix_mr_opcode.patch` emits the proved 54-AND
  GF(2¹²) product natively, and a `GFProduct` policy (`builder.gf_product = 'and54'`) routes
  `gen_gf_mul` through it. The earlier `Gf54MulMixin` could only replace `ixor_gf2_mul`; the
  five Itoh–Tsujii products inside Chien search and the quotient sweep's are emitted by
  native generators, so the mixin left them on the built-in 63-AND recursion.
- **MEASURED:** `scripts/measure_gf54_native_opcode.py` (record:
  `benchmarks/quantum_compilation/gf54_native_opcode.json`). Unit costs: controlled multiply
  **75 → 66**, plain multiply **63 → 54**, inversion **315 → 270**, division **378 → 324**
  Toffolis, at **42** workspace qubits against 12, all four correct on random basis states
  (values, restored operands, clean workspace, untouched phase). At the pinned instance over
  64 shots the **reference** decoder goes **5,725,071.6 → 5,201,222.5 per shot (9.15%)**, `Q`
  1,893 → 1,935, and the six-lever arm **1,325,230.0 → 1,221,466.8 (4.687× against 4.320×)**
  — better than the mixin's 4.652×, because the opcode reaches the inversion too.
- **MEASURED — the Forney inversion's share, isolated.** The reference runs an inversion of
  five products at every one of its 4,095 positions: 20,475 product sites, 184,275 static
  Toffolis. Pricing it alone (the same reference with only
  `init_gf2_inverse_with_scaffold`/`gf2_phase_by_product` switched) executes **20,469.5
  products per shot across 4,093.9 positions, saving 184,225.1 Toffolis — 3.22% of the
  reference and 35.2% of its 523,849.1 saved per shot**. The remaining 339,624.0 (64.8%) is
  the change-of-basis division and the quotient sweep. Static and measured differ only by the
  1.1 positions per shot whose error bit is clear.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains the builder-policy surface
  (66 CCX, 42 extra workspace bits, invalid names rejected), an end-to-end scaled run, and a
  pin on the recorded split so the Forney share cannot drift back to an assertion.

## 2026-10-29 — The 54-AND multiply, executed: 4.652×, and the share it corrects

- **NEW:** `scripts/gf54_mul.py` turns the proved 54-AND GF(2¹²) algorithm into gates: the
  pinned `mu(2)=3`/`mu(3)=6` base cases, their Karatsuba composition, a classical evaluator,
  and `Gf54MulMixin`, which reimplements `ixor_gf2_mul` as **54 ANDs plus the 12-Toffoli
  controlled add — 66 Toffolis per controlled call against the built-in 75** — clearing its
  product window and scratch with `mr`. 42 workspace qubits against the built-in 12, and a
  longer CNOT ladder (566 vs 481).
- **MEASURED:** `scripts/measure_gf54_mul_kickmix.py` (record:
  `benchmarks/quantum_compilation/gf54_mul_kickmix.json`). The emitted multiply is correct on
  random basis states with random controls (values equal `field.mul`, operands restored,
  workspace `|0>`, phase untouched), and swapped into the six-lever arm at the official
  instance over 64 shots it takes **1,325,230.0 → 1,230,640.0 Toffoli per shot: 4.652×
  against 4.320×**, a **7.14%** saving, `Q` 1,894 → 1,897, with the arm still passing the
  six-lever verification shot for shot against the reference.
- **CORRECTED:** the projected 4.928× does not reproduce, and the reason is the model, not
  the circuit: a controlled multiply pays its ANDs **plus 12**, and the 12 do not shrink, so
  the call scales by 66/75. The same measurement corrects a larger error in the record — the
  multiply is **59.48%** of the arm (788,250.0 per shot over 10,510 executed calls), not the
  **86.33%** an earlier ablation reported. That ablation deleted `ixor_gf2_mul` and
  differenced executed counts, so the registers held garbage, the gating predicates read it
  and far less other work ran. The new measurement differences two arms that differ only in
  the multiply, and it reconstructs the measured arm exactly
  (536,980.0 + 10,510 × 66 = 1,230,640.0).
- **REVISED:** the multiply-cost ladder is now `t(ands) = 536,980 + 10,510 × (ands + 12)`: 54
  ANDs 4.652× (measured), 45 ANDs 5.039×, the 23-AND `2n-1` bound 6.327× (2.03× the target),
  and **0 ANDs 8.634× — still 1.49× above 445,229**. The conclusion survives with a narrower
  margin: no multiply, however cheap, delivers Benchmark II on this decoder, so the target
  needs a change to the 536,980 that is not the multiply. `docs/CIRCUIT_OPTIMISATION_RULES.md`
  §5–§6 and §8 carry it, and `benchmarks/quantum_compilation/{designer_directions,
  gf_mul_and_static_headroom}.json` were regenerated rather than left asserting the old share.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains pins for the base algorithms,
  the emitted circuit (values, restored operands, clean workspace, phase), the 66/75 unit
  costs, both arms' correctness on a scaled run, and the recorded decomposition.

## 2026-10-29 — Three designer directions priced; the multiply gains 14.3%

- **MEASURED, then designed:** `scripts/study_circuit_designer_directions.py` (record:
  `benchmarks/quantum_compilation/designer_directions.json`) prices the three ways left to
  improve the decoder. **A** (predicate discovery over the regions no lever touches) is
  bounded at **2.71%** of the emitted stream at the pinned instance — 94,470 of 3,486,597
  Toffolis — so it does not pay here. **B** (an exact bilinear search) proves `mu(2)=3` and
  `mu(3)=6` by exhausting every choice of five rank-1 matrices, then composes by Karatsuba
  to **54 ANDs at GF(2¹²)** against the built-in 63 (14.3% fewer), verified against
  `GF(2¹²)` multiplication on 200 sampled inputs with 0 mismatches and priced at
  **4.928×**. **C** (a different decoder) is refuted as a multiply-only fix: even the
  23-AND `2n-1` dimension bound leaves 598,835.7 per shot, still 1.35× above the 445,229
  target, and only the inadmissible measured-operand route clears it.
- **DOCUMENTED:** `docs/CIRCUIT_OPTIMISATION_RULES.md` §6 carries the comparison and the
  ladder, and §8 records the 54-AND multiply as *pending*: a verified algorithm, not yet a
  measured circuit, because `ixor_gf2_mul` is the 63-AND Karatsuba and emitting the new one
  needs a second `kickmix` op.

## 2026-10-29 — Rules for a better circuit, and the measured ceiling

- **NEW:** `docs/CIRCUIT_OPTIMISATION_RULES.md` fixes what a "better circuit" claim has to
  carry: the metric by name (**executed** 1,325,230.0 = 4.320× versus **static** 3,486,597 =
  1.642×, which rank the levers differently), unitary preservation on every input (dead-work
  gating and measured uncomputation are admissible; measuring an operand that is not
  determined by the state is not), a falsifier per measurement, and the instance discipline.
- **MEASURED:** `scripts/measure_gf_mul_and_static_headroom.py` (record:
  `benchmarks/quantum_compilation/gf_mul_and_static_headroom.json`). `ixor_gf2_mul` is the
  Karatsuba recursion to the gate — 3/7/9/21/27/63 static Toffolis at GF(2²/2³/2⁴/2⁶/2⁸/2¹²),
  matching `K(m)` exactly at every degree. No split of the recursion beats 63 at m=12 (k=8
  ties, the rest are worse; schoolbook 144). The multiply is **86.33%** of the six-lever
  arm's executed count, so a 45-AND multiply would give 5.734× and a free one 31.598× (the
  latter only by the inadmissible measured-operand route). A full-recording peephole removes
  **0** Toffolis at both instances, and the sixth lever changes the static count by exactly 0:
  the emitted stream has no headroom left.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains a record pin for those
  invariants (Karatsuba exactness, the 63 floor, static five == six, zero peephole
  removals, the multiply share), so a later "optimisation" that changes any of them fails
  without the extension being rebuilt.

## 2026-10-29 — DQI MR-gated arms: a sixth lever and qubit parity

- **MEASURED:** `scripts/measure_mr_arm_costs_kickmix.py` prices the gated Benchmark II
  circuits on the rebuilt, MR-capable `kickmix` (official instance, m=4095, n=70,
  GF(2¹²), 64 shots, seed 20261028). The five oracle-gated levers reproduce the
  projection stage for stage: 5,725,071.6 → 1,381,390.0 Toffoli per shot (4.144×)
  against the projected 1,382,207.08 (4.142×), every stacked stage within 0.06% of its
  projected count.
- **NEW LEVER:** each of the sweep's two polynomial divisions is consumed by an add
  controlled on its own counter bit, and the four counter bits are a one-hot decode, so
  in every live round exactly one division is computed under a false add and read by
  nothing. Gating each division on that bit (`gate_quotient`, arm `quotient`) removes
  4.07% of the five-lever count: **1,325,230.0 per shot, 4.320×**. It carries no projected
  row, because the oracle build priced both divisions as live.
- **FIXED (qubits the reads were leaking):** every read allocated a fresh copy qubit and
  the sweep's `q_i` was never freed, so the gated arm's id count grew by roughly a
  register per round — 4,503 ids against the reference's 1,893. One shared read ancilla
  (`mr` clears its target, so reuse is free) and freeing `q_i` in the round that
  allocates it bring both stacks to **1,894** ids: the gated rows now pay one qubit, not
  2,610.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains
  `test_small_instance_cost_pin_catches_gate_regressions`, which re-prices the whole
  stack at m=15/n=6/GF(2⁴) in about a second (33,235.5 → 10,409.5, 3.193×) and fails if
  any gate stops firing, changes its saving by more than 1%, or costs qubits — the
  64-shot official run is not needed to catch a regression. `docs/MR_INSTRUCTION_SPEC.md`
  §6b records the new lever and the qubit fix.

## 2026-10-29 — Temporal-rate detector audit hardening

- **RE-PINNED:** `specifications/canonical_preregistered_spec_v1.json` and
  `apedp_dyn/conf/canonical_spec.py` to
  `dbf67c5ac443073721421eba456519a0eda09ae07680d638c26f1d019b0b2343`.
  The only spec change is the new `temporal_rate_min_compositions=2`; the
  two-distinct-composition requirement for a common-mode claim is now
  canonical rather than a hardcoded literal. No previously pinned FPR, power,
  alpha, SNR, amplitude, harmonic, or sample threshold changed. FLAW-01
  remains OPEN; the governance manifest and its expected failing test were not
  touched.
- **FIXED:** `_drive_phasor()` used a raw windowed DFT of the drive. For a
  non-integer number of drive cycles, constant/trend leakage rotated and
  rescaled the estimated phasor; that bias transferred incorrectly between
  calibration and confirmation frequencies and inflated the null statistic.
  The phasor is now fit jointly with a constant and linear trend over
  cosine/sine columns and returns the analytic conjugate. Added
  `test_calibration_transfer_has_no_drive_phasor_bias`, which recovers the
  known susceptibility from a noiseless non-integer-cycle calibration to
  1e-10 relative accuracy.
- **MEASURED (before/after):** before the phasor fix, 512 null trials at 128
  surrogates gave FPR 14/512 = 2.73% (95% upper 4.24%) and 128 trials at 512
  surrogates gave 4/128 = 3.13% (95% upper 7.01%). After the fix: 512 trials
  at 128 surrogates gave FPR 5/512 = 0.98% (95% upper 2.04%), artifact rate
  0/512, power 512/512 (95% lower 99.42%), median null p 0.516; a second
  512-trial seed 7 audit also gave 5/512; 128 trials at 512 surrogates gave
  FPR 1/128 = 0.78% (95% upper 3.65%), artifact rate 0/128, power 128/128
  (95% lower 97.69%), median null p 0.496. Both audits pass the preregistered
  point-estimate gate. Exact Clopper-Pearson bounds are reported
  as diagnostics; the 128/512 upper bound is now below 5% as well.
- **HARDENED:** `ClockDataset` rejects non-uniform sampling; calibration and
  confirmation must use disjoint drive-harmonic sets at the Rayleigh
  resolution; calibration reports per-harmonic SNRs while the insufficiency
  gate remains fundamental-only; `TemporalRateAdjudication` records both
  dataset ids, the frozen analysis parameters, calibration SNRs, and the
  canonical spec hash.
- **FROZEN:** the preregistration schema is now `temporal-rate-v2` and freezes
  `harmonics`, `n_surrogates`, `seed`, `familywise_alpha`,
  `heterogeneity_alpha`, `min_fractional_amplitude`, and
  `canonical_spec_sha256` in addition to dataset fingerprints/ids, drive
  frequencies, and the clock-composition map.
  `adjudicate_preregistered_temporal_rate()` takes exactly three arguments and
  cannot be parameter-overridden after registration. Verification rejects a
  changed spec hash, changed canonical alpha, or fewer than two distinct clock
  compositions. The artifact remains a fingerprint commitment, not a
  signature.
- **CALIBRATION:** `run_detector_calibration()` now refits a fresh calibration
  for every null/injection pair, so the FPR is unconditional on one lucky
  calibration realization. It reports false-candidate, oscillator-specific
  artifact, and any-false-claim rates, plus exact one-sided 95% bounds and
  `n_calibration_draws`; the preregistered pass gate is the point-estimate
  FPR/power floor, with bounds retained as diagnostics.
- **ADDED:** 9 pins, bringing the temporal suite to 44 (18 detector + 26 IO):
  uniform-sampling rejection, harmonic-overlap rejection, per-harmonic SNR
  reporting, phasor-transfer regression, frozen analysis parameters, no
  parameter overrides, tampered spec hash, tampered canonical alpha, and
  distinct-composition enforcement. `AGENTS.md`'s standard suite now includes
  both temporal test files.
- **Verified:** 44/44 temporal pins pass; Ruff is clean on the changed files;
  repo-scope mypy (`python3 -m mypy`) is clean. The expanded dynamic suite is
  162 passed with one pre-existing unrelated failure:
  `test_orphan_module_detection` scans the untracked vendored
  `kickmix/_qualtran_src/`.
- **LIMITS:** common-mode calibration/clock systematics are still unmodeled,
  `systematic_susceptibility_sigma` still defaults to zero, the pure-sinusoid
  drive assumption is documented but not validated against non-sinusoidal
  drives, and colored/non-Gaussian residual noise is not covered by the
  shared-phase surrogate. No real two-composition clock data exists in the
  repo, so these are detector-calibration results, not a measurement of nature.

## 2026-10-29 — `MR` implemented, and the gated arms run on a real `kickmix`

- **ADDED:** `scripts/kickmix_mr_opcode.patch` — the `MR`/`MR_IF` opcodes
  (`docs/MR_INSTRUCTION_SPEC.md`) applied to the vendored `kickmix` sources:
  opcode values 0x54/0xD4 with names, `OP_TYPE_TABLE` and `.kmx` parse/print, a
  `mr`/`mr_if`/`broadcast_mr` builder path with Python bindings, a Z-basis
  simulator branch that writes the outcome and clears the qubit without touching
  the tracked phase, its own statistics group, and a reaction-depth case.
- **ADDED:** `scripts/build_kickmix_mr.py` exports the vendored sources at their
  recorded commit, applies the patch, and compiles `build/kickmix_mr/lib`
  (`build/` is gitignored). The pinned competition binary that
  `apedp_dyn/quantum_compilation/contracts.py` hashes is never modified.
- **ADDED:** `scripts/run_mr_gated_arms_kickmix.py` runs the five gated DQI levers
  on that extension with a real `kickmix.CircuitBuilder` and a real
  `kickmix.Simulator`, so the arms are no longer measured only on the local model.
- **MEASURED:** at m=15/n=6/GF(2⁴) with 8 shots, `none`, `swap`, `sweep`,
  `forney`, `clear`, `shift` and `all` match the reference's syndromes and tracked
  phase, restore the syndromes and leave every non-syndrome qubit `|0>`; `all`
  also passes at m=255/n=14/GF(2⁸) with 16 shots and at the official
  m=4095/n=70/GF(2¹²) with 4 shots (reference 1,893 qubits, arm 4,503 allocated
  ids, phase identical). The falsifiers still fail:
  `--clear-with r` (phase 6 ones vs 3, dirty junk) and `--shift-under`
  (syndromes not restored).
- **RECORDED:** `x(q); mr(q, b)` gives `b = 1` and 0 phase flips on 64/64 shots,
  against 37/64 phase flips for the same circuit with `reset` — the semantic
  difference the instruction exists for.
- **ADDED:** `scripts/measure_mr_arm_costs_kickmix.py` prices the reference, the
  cumulative lever chain and each lever alone with `kickmix`'s conditional-aware
  per-shot Toffoli counting. Record:
  `benchmarks/quantum_compilation/kickmix_mr_arm_costs.json`.
- **ADDED:** `tests/test_dyn_mr_gated_arms_kickmix.py` pins the real-kickmix arm
  runs and both falsifiers, plus the cost record's arithmetic; the extension-backed
  pins skip when the MR build is absent (10 passed under CPython 3.13 with the
  build, 1 passed and 9 skipped under the project's 3.12).
- **MEASURED (cost):** at the official instance over 64 shots the rebuilt builder
  reproduces the pinned reference exactly (5,725,071.6 vs 5,725,071.61) and the
  full gate executes **1,381,390.0 Toffoli per shot, 4.144×** against the
  projected 1,382,207.08 / 4.142×. Every stacked stage is within 0.06% of its
  projection — 4,298,268.3 (Forney), 2,157,906.3 (+clear), 1,590,308.3 (+sweep),
  1,456,286.3 (+swap), 1,381,390.0 (+shift) — so the oracle-gated projection is
  reproduced by the circuits rather than bounding them from above. Per lever
  alone: `swap` 1.049×, `sweep` 1.247×, `forney` 1.332×, `clear` 1.597×, `shift`
  1.027×; the shift lever inside the chain is worth the 5.14% the oracle
  accounting gave it.
- **FIXED:** three gate defects in `scripts/run_mr_gated_arms.py` that a first
  costing pass (3.556×) exposed. The shift gate's `¬flag` was written as a second
  `bit_store(…, True)`, leaving the bit constant, so it skipped only rounds with
  every control clear (1.002× alone). The sweep gate wrapped only the position
  sweep, leaving the quotient divisions and the `n_a` add/undo inside dead rounds.
  The Forney predicate was an 11-Toffoli equality circuit run twice per position,
  with a copy qubit allocated per position; it is now `mr`-read out of `sigma`,
  restored with classically controlled `x`, and assembled from two classical
  stores. Net: 1,609,925.7 → 1,381,390.0 per shot and 8,587 → 4,503 qubit ids.
- **UNCHANGED:** a receipt still requires an upstream `kickmix` rebuild plus the
  official verifier; the local extension is not the pinned binary.

## 2026-10-28 — Preregistration gate for the temporal-rate null test

- **ADDED:** `TemporalRatePreregistration` plus `preregister_datasets()`,
  `save_preregistration()`, `load_preregistration()`,
  `verify_preregistration()`, and `adjudicate_preregistered_temporal_rate()`
  in `apedp_dyn/epistemic/temporal_rate_io.py`. The artifact freezes the
  calibration/confirmation SHA-256 fingerprints, dataset ids, drive
  frequencies, and clock-composition map before adjudication. Verification
  recomputes every field and fails closed on any mismatch, including a swapped
  confirmation file that keeps the same dataset id.
- **EXTENDED:** `scripts/run_temporal_rate_null_test.py` gains
  `--write-preregistration`, `--preregistration`, and
  `--preregistration-notes`. `--write-preregistration` freezes the fingerprints
  without running the test; `--preregistration` verifies before adjudicating.
  Protocol errors now exit 1 with a one-line message instead of a traceback.
- **ADDED:** eight preregistration pins in
  `tests/test_dyn_temporal_rate_io.py`: round trip, swapped confirmation,
  tampered calibration, changed metadata, unknown keys, unsupported protocol
  version, duplicate dataset ids, and end-to-end preregistered adjudication.
  The temporal suite is now 35 pins (14 detector + 21 IO).
- **Verified:** 35/35 temporal pins pass; Ruff is clean. The full dynamic suite
  with both new files included is at 152 passed with the same two pre-existing
  unrelated failures (host-path literal in untracked
  `scripts/inspect_patch_1994.py`; orphan scan over untracked vendored
  `kickmix/_qualtran_src/`).

## 2026-10-28 — Clock-dataset ingestion for the temporal-rate null test

- **ADDED:** `apedp_dyn/epistemic/temporal_rate_io.py` — strict CSV +
  sidecar-JSON and compressed NPZ ingestion for `ClockDataset`. The CSV format
  names `time_s`, `drive`, and one fractional-frequency column per clock; the
  sidecar carries the dataset id, drive frequency, and each clock's id,
  composition, column, and systematic-uncertainty budget. NPZ embeds the same
  metadata and is read with `allow_pickle=False`, so a dataset file cannot
  execute code. Both paths delegate finite/monotonic/length checks to
  `ClockDataset` and expose `dataset_sha256()` for calibration/confirmation
  provenance.
- **ADDED:** `tests/test_dyn_temporal_rate_io.py` — 13 pins covering CSV and
  NPZ round trips, cross-format fingerprint equality, fingerprint sensitivity,
  missing sidecar, missing/unknown metadata keys, duplicate clock ids, empty
  and non-finite cells, unknown NPZ keys, object-array rejection, NPZ
  `metadata_path` rejection, single-clock support, and unsupported formats.
- **EXTENDED:** `scripts/run_temporal_rate_null_test.py` now adjudicates
  supplied files via `--calibration` / `--confirmation` (with optional
  `--calibration-metadata` / `--confirmation-metadata`), reports both dataset
  fingerprints, and preserves the claim boundary. The synthetic demo remains
  the default when no files are supplied.
- **Verified:** the 13 IO pins and the 14 detector pins pass; Ruff is clean on
  the new module, tests, and script. The full dynamic suite, with the two new
  files included, is at 144 passed with the two pre-existing unrelated
  failures (host-path literal in untracked `scripts/inspect_patch_1994.py`;
  orphan scan over untracked vendored `kickmix/_qualtran_src/`).

## 2026-10-28 — Option B: a temporal-rate null test, not a time-travel claim

- **ADDED:** `apedp_dyn/epistemic/temporal_rate_modulation.py` — Hypothesis B
  harness. It subtracts each clock's calibration-measured ordinary response to
  a periodic drive and tests whether the residual phase-locked
  fractional-frequency modulation is common to clocks of at least two distinct
  compositions. The null is a shared-phase Fourier surrogate of the post-fit
  residuals with the fitted harmonic variance restored and calibration
  uncertainty injected; a heterogeneity guard rejects oscillator-specific
  artifacts. Calibration/confirmation dataset-id reuse, identical series, and
  same-frequency calibration raise `TemporalRateProtocolError`. The only
  positive verdict is `COMMON_MODE_ANOMALY_CANDIDATE`, explicitly not evidence
  of backward time travel.
- **PREREGISTERED:** canonical spec gains the `temporal_rate_*` block
  (minimum clocks/cycles/samples-per-cycle, harmonics, surrogate draws,
  familywise and heterogeneity alphas, amplitude floor, calibration-SNR floor,
  and detector FPR/power/injection thresholds). The spec is re-pinned to
  `a5e13979…`; no existing threshold changed. Ed25519 re-signing remains
  trustee-blocked (FLAW-01 still OPEN).
- **MEASURED, synthetic detector calibration:** 20 null and 20 injected
  common-mode trials at 128 surrogates (seed 6) give FPR 0.00 against the 0.05
  ceiling and power 1.00 against the 0.80 floor, with zero insufficient
  verdicts. This measures the analysis, not nature.
- **Verified:** `tests/test_dyn_temporal_rate_modulation.py` 12 passed,
  including null-consistency, common-mode recovery at an arbitrary phase,
  oscillator-specific rejection, protocol reuse/frequency refusals,
  single-composition and single-clock refusals, weak-calibration refusal,
  deterministic replay, and the FPR/power floors.

## 2026-10-28 — The MR arms hold at the official instance

- **MEASURED:** all five arms and `all` reproduce the reference at Benchmark II's own
  instance — m=4095, n=70, GF(2¹²), 4 shots: syndromes, tracked phase and `|0>` junk all
  match. Reference: 107,911 qubit ids, 4m54s. Arm ids: swap 108,355, sweep 108,799,
  forney 112,006, shift 110,131, `all` 84,103, clear 78,232. These are local allocation
  counts, not `Q`.
- **FALSIFIED at the same scale:** the under-gated shift arm fails on syndromes, phase and
  junk, and the `r` readout fails on all three (at this scale its readout also corrupts
  values, not only the phase). Record: `benchmarks/quantum_compilation/local_mr_arms.json`.
- **Verified:** `tests/test_local_mr_sim.py` now pins all three instances.

## 2026-10-28 — Fifth MR lever verified at both instances, with a falsifier that fails

- **ADDED:** the shift gate to `scripts/run_mr_gated_arms.py` (`--arm shift`, included in
  `all`). It reads `c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))` with a real `mr`, using the flag *after*
  `_update_flag`; the probe's earlier attempt with a round-top flag fails because that flag
  is stale. At m=15/GF(2⁴), 32 shots, the predicate is false in 26.7% of round-shots, and
  the arm reproduces the reference's syndromes, phase and `|0>` junk.
- **MEASURED, scaled:** all five arms plus `all` pass at m=255/GF(2⁸), n=14, 16 shots
  (reference 9,751 qubit ids; syndromes, phase and junk all match). The falsifiers fail at
  both instances: `--clear-with r` loses the phase and leaves junk, and `--shift-under`
  (dropping `¬flag ∧ (c0 ∨ c3)`) skips shifts the decoder needs and breaks the syndromes.
  Record: `benchmarks/quantum_compilation/local_mr_arms.json`.
- **REJECTED as a falsifier:** inverting the shift gate also passes, because one skipped
  shift is repaired by later rounds; only under-gating breaks the decoder, so the inverted
  knob was removed rather than shipped as a test that cannot fail.
- **Verified:** `tests/test_local_mr_sim.py` 21 passed over both recorded instances,
  including the under-gated shift arm and the `r` readout.

## 2026-10-28 — Three metric placeholders now measure or refuse

- **FIXED, `EnsembleSearcher.get_diversity_score`:** returned a constant 0.8 with a
  "would analyze equation structures" comment. It now reads the solutions out of the
  results (`best` / `expression` / `expressions` / `equations` / `solutions`) and scores
  `(distinct - 1) / (total - 1)`: 0.0 when every member agrees, 1.0 when no two do. Fewer
  than two reported solutions — including its own `search_ensemble` plan — raises, because
  a plan has no diversity to measure.
- **FIXED, `DeepEnsemble.predict`:** reported `aleatoric=jnp.zeros_like(std)`, presenting an
  unmodelled quantity as a measured zero. These regressors train deterministically, so the
  field is `None`; the monkeypatch that previously masked the same placeholder after import
  was removed with it.
- **FIXED, `PhysicsConstraintEngine.dimensional_analysis_score`:** returned 0.5 with no
  target and 1.0 with one, regardless of any candidate. It now takes the dimensions of the
  candidate's terms and returns the fraction that match the target, raising when there are
  no terms or no target. The Rust kernel had the identical placeholder and now implements
  the same measurement over a JSON term list (rebuilt extension); the Python fallback and
  the kernel agree, and both `check_dimensional_consistency` paths now compare all seven
  exponents instead of mass/length/time only.
- **Verified:** `tests/test_placeholder_metrics_are_measured_or_refused.py` 9 passed (both
  implementations of the dimensional score, the diversity endpoints, the refusals, and
  `aleatoric is None`); `tests/test_search_engines_real.py` 13 passed.

## 2026-10-28 — MR arm correctness tested locally: four levers, and `r` fails as it should

- **BUILT:** `apedp_dyn/quantum_compilation/local_mr_sim.py`, a Z-basis (value + sign)
simulator for the DQI decoder with `mr` (Z measure-and-reset, phase-preserving), `r`
(clears, randomizes the phase, as `kickmix` does) and `hmr` (X basis). It implements the
builder subset the decoder uses (`unary_iteration`, `init_and`, `iadd`/`isub`,
`flip_if_*`, rotations, GF(2^b) arithmetic, `gf2_phase_by_product`) with semantics
pinned against `kickmix`, including its primitive polynomials.
- **VALIDATED:** `scripts/verify_local_mr_sim.py` builds the reference decoder with both
engines and feeds both the same shots. At m=15/n=6/GF(2⁴): syndromes identical, phase
identical, syndromes restored, every other qubit `|0>`, 0 phase errors — the local engine
reproduces `kickmix` exactly (qubit *ids* differ because freed ids are not reused here).
- **MEASURED:** `scripts/run_mr_gated_arms.py` runs the four gated levers with their
predicates read by a real `mr` through `cx(control, ancilla); mr(ancilla, bit)`, not the
oracle registers the probe used. Every arm reproduces the reference's values AND phase
at 32 shots: `swap` (c3), `sweep` (c0 ∨ c3), `forney` (the `sigma == 0` zero-test),
`clear` (measured clear of the EEA instead of its adjoint, with the syndromes re-prepared
from a saved readout), and all four together. Record:
`benchmarks/quantum_compilation/local_mr_arms.json`.
- **FALSIFIED, by construction:** the same swap arm read with `r` instead of `mr` fails on
phase and leaves junk behind while its syndromes still match — so the pass is about the
measurement, not the circuit. This is the mechanism-level version of the 11/16 phase
defect the probe recorded.
- **STILL UNSCORED:** the arms' cost. `local_mr_sim` computes values and phases, not gate
counts, and `contracts.py` pins the official `kickmix` build and verifiers, so the 4.142×
stays a Toffoli bound until `MR` exists upstream.
- **Verified:** `tests/test_local_mr_sim.py` 21 passed (pinned primitives, the MR/R/HMR
contract, and the arm record).

## 2026-10-28 — Search engines score on data, and NGTS runs UCT or nothing

- **MEASURED, not defaulted:** `ValidationPipeline.validate` now reports
  `generalization_gap` (held-out minus cross-validation error) and derives
  `generalization_score` from it as `1/(1+gap)`. The old score was a lookup — 0.9 below
  a threshold, 0.7 above — so it reported a number the data could not move.
- **No fabricated evidence:** `NGTSSearch.run_simulations` runs UCT and calls a reward
  function for each selected leaf. With no reward function it runs **zero** simulations
  and returns 0; it previously returned `num_sims` and backpropagated
  `random.random()*0.5`, i.e. reported search work it had not done. The Rust kernel had
  the same defect in `ngts.rs` (`rand_simple` values) and now takes the reward callable
  in `run_simulations(num_sims, reward=None)`.
- **Hierarchical phases score what they propose:** `_mid_search` / `_fine_search` take
  `data`/`target`, evaluate each expansion on the data, and return nothing when there is
  no data. `HierarchicalSearch.search` attaches a measured `r2` to every candidate and
  **drops** candidates it cannot evaluate; it used to validate every candidate against
  default `train_error=0.1 / test_error=0.15`, so an unmeasured expression carried a
  validation score.
- **Ensemble reports what it cannot run:** a proposer outside `ProposerRouter`'s map is
  `status: error` instead of a planned member, and `best` is present and `None` until a
  member returns a scored candidate.
- **Determinism:** `beam_search_expressions` no longer consumes `seed` for sampling; two
  seeds return the same list, and level 1 pairs atoms with each other so multi-variable
  forms appear immediately.
- **Verified:** `tests/test_search_engines_real.py` 13 passed (was 12 of 13 failing);
  related suites 58 passed / 2 skipped / 1 xfailed.

## 2026-10-28 — Scratch pool measured dead: 25-qubit ceiling, and the allocator claim withdrawn

- **MEASURED** on the pinned Benchmark II instance (`scripts/measure_dqi_scratch_pool.py`,
  record `benchmarks/quantum_compilation/dqi_scratch_pool.json`): the circuit holds
  **1,828 qubits concurrently live against 1,893 distinct ids**; the peak is reached
  inside `_chien_search_and_forney_mbuc` and the innermost frame there holds **25**
  qubits. A shared scratch pool can therefore redistribute at most 25 — the −104 the V6
  row claims is 4× that ceiling and is not reachable by pooling.
- **CORRECTED, committed claim:** §5.5 of the boundary doc said the live peak was EEA
  round scratch in `_multiply_and_add_or_sub` and that a shared pool was "the only
  mechanism that could move" the transients. Both are withdrawn on measurement: the peak
  is in Chien/Forney, and the pool is dead.
- **Left OPEN, not answered:** the probe's trace replay returns 1,821 ids against 1,893
  shipped and observes only 1,873 ids, so allocations made inside the extension are
  missing; a lowest-hole free list replays identically to the shipped LIFO stack. An
  earlier draft of the script asserted the 65 stranded ids were `alloc_qubit` LIFO
  stranding and named a free list as the fix. That verdict was not supported by the
  replay and has been removed rather than shipped.
- **Verified:** `tests/test_dqi_scratch_pool.py` pins the artifact; the probe runs under
  CPython 3.13 only, since `kickmix.so` is a 3.13 build.

## 2026-10-28 — Post-`_update_flag` probe collected: 4.142× measured

- **MEASURED:** the shift gate the previous entry had to abandon is now correct. The
  wrong arm read `flag` at the top of the round; `_update_flag` toggles it first, so the
  probe now carries six bits per round and the liveness is
  `c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))` with the post-update flag. Official instance:
  `T_mean` 1,457,103.45 → **1,382,207.08, a 4.142× over the reference**, `Q` unchanged at
  1,893, **0 value / 0 phase errors on 64 shots**, 1,332 round bits (6 × 222).
  Scaled: 3.709× at m=255 (512 shots), 2.998× at m=15.
- **Corrected prediction:** the prize was bounded at ~5% from the failed arm's cost and
  measures at 5.15% (1,457,103 → 1,382,207). Dead shifts are 31.59% of rounds, not the
  35–40% guessed before the fix.
- **Running total:** Forney gating 1.332×, MR clear 2.652×, sweep 3.599×, swap 3.929×,
  shift 4.142×. Receipt arithmetic 24.439T → 5.900T, portfolio 2.361982 → 3.79; still
  3.10× short of the 12.86× target.
- **Verified:** `tests/test_dqi_predicate_gating.py` 12 passed / 1 skipped.

## 2026-10-28 — Dead register swaps gated: 3.929× measured, and the shift gate falsified

- **MEASURED:** `_swap_a_and_b` exchanges `reg_a`/`reg_b` wholesale — 2 × 73 × 12 = 1,752
  qubits — every round, but its control is `swap_ctrl = c3_hc ∧ flag`, so it is dead in
  any round without `c3` (**67.57%** of rounds). Gating it on `c3` alone: official
  `T_mean` 1,590,938.28 → **1,457,103.45, a 3.929× over the reference**, `Q` unchanged at
  1,893, **0 value / 0 phase errors on 64 shots**, for 1,110 round bits (5 × 222).
  Scaled: 3.507× at m=255 (70.4% dead), 2.813× at m=15. Receipt arithmetic 24.439T →
  6.22T, portfolio 2.361982 → 3.7271; still 3.27× short of 12.86×.
- **FALSIFIED (mine, one hour old):** gating `_shift_b_array_right` on the complementary
  pair `c1 ∨ c2 ∨ ¬flag`. It looked right and fails on **every shot** (512/512 value
  errors at m=255). Cause: `flag` is toggled by `_update_flag`, which runs *before* the
  shift in the same round, so a flag read at the top of the round is stale. The arm's
  cost (62,434 at m=255 against 65,698) bounds the remaining prize at ~5%, and collecting
  it needs a second probe point after `_update_flag`. Not taken; pinned as a negative in
  `tests/test_dqi_predicate_gating.py`.
- **Cost of the whole design so far:** 1,803 classical bits for the clear, 1,110 for the
  round predicates, zero extra qubits, zero extra Toffolis.

## 2026-10-28 — Dead multiply sweeps gated: 3.599× measured

- **MEASURED:** the synchronized EEA emits all five steps every round, but
  `_multiply_and_add_or_sub` touches `ctrl` only through `c0_hc` and `c3_hc`. In a round
  where both are zero — the two shift rounds of each four-round counter cycle — its
  4,536-Toffoli sweep is emitted and does nothing. Reading `(c0_hc, c3_hc)` per round
  and gating the sweep on `c0 ∨ c3` removes **35.14%** of the forward EEA at the official
  instance: `T_mean` 2,158,651.45 → **1,590,938.28, a 3.599× over the reference**, with
  `Q` unchanged at 1,893 and **0 value / 0 phase errors**. Scaled: 3.205× at m=255
  (40.79% dead), 2.586× at m=15 (46.98% dead).
- **Cost:** one classical bit per EEA round — 222 bits at the official instance, 54 at
  m=255, 30 at m=15 — no qubits, no Toffolis. Still an upper bound: the round bits are
  read from the same circuit in a probe pass.
- **Not taken:** `_shift_b_array_right` is live in the `c1`/`c2` rounds *and* in the
  `c0`/`c3` rounds when `flag` is clear (`ctrl_nB = c0_nf ⊕ c1_hc`), so it cannot be
  gated on the complementary pair without also reading `flag`; left alone.
- **Verified:** `tests/test_dqi_predicate_gating.py` 9 passed / 1 skipped.
- With the Forney gating and the MR clear this is 3.599× of the 12.86× the target needs:
  `T_mean` 1.591M against 445,229 leaves a 3.57× gap, and the receipt moves 24.439T →
  6.791T (portfolio 2.361982 → 3.6198), arithmetic on `R·Q²·T`, not a receipt.

## 2026-10-28 — MR clear halves the EEA: 2.652× measured, and `R` is not a substitute

- **MEASURED (the better design):** the reversible EEA is run twice — forward, then
  its adjoint to uncompute — so half of Benchmark II's 75% EEA share is uncomputation.
  Clearing the EEA scratch and the syndromes and re-preparing the syndromes from a
  classical readout replaces that half. `scripts/probe_dqi_predicate_gating.py`
  measures it with both oracle registers (syndrome readout + root bits): official
  instance `T_mean` **5,725,071.61 → 2,158,651.45, a 2.652×**, `Q` unchanged at 1,893,
  **0 value and 0 phase errors** on 64 shots, needing 1,803 classical bits and 1,803
  classically controlled X's. Scaled: 2.351× at m=255 (512 shots), 2.012× at m=15.
  Because `Q` does not move, the score falls by the same 2.652×: 24.439T → **9.215T**,
  i.e. the portfolio score 2.361982 → **3.2695** (arithmetic on `R·Q²·T`, not a receipt).
  Still 4.85× short of the 12.86× target.
- **FALSIFIED (mine, from four hours earlier):** "`R` can stand in for uncomputation."
  A reset clear costs exactly what the reference spends (2,158,651.45 either way — CX
  is not Toffoli-like) but it is **wrong**: 35/64 shots at the official instance and
  254/512 at m=255 have a flipped phase. Cause, measured: a circuit with **no phase in
  it at all** — `x(q); reset(q)` — reports a random `phase_flipped` (11/16 shots), while
  the same circuit without the reset reports 0/16. `R` randomizes the tracked phase, so
  it can only be used where the answer does not depend on it. Measure-then-X on the
  measured value (i.e. `MR`) gives 0/0.
- **BOUND (corrected):** with MR, both levers together are worth 2.652×, not the 1.332×
  the Forney-gating entry below states: that entry bounded *one* phase of the decoder and
  its "even granted MR, T lands at 4.30M" bullet is superseded here. `T_mean` 2.159M is
  still 4.85× above the 445,229 target, so the §5.5 conclusion stands — the target needs
  a decoder, not an optimisation.
- **Still blocked in the IR:** `MR` does not exist. `push_condition` takes a classical
  bit; `hmr` is X-basis; `push_condition(qubit)`/`bit_store(control=qubit)` are rejected;
  `R` is phase-unsafe. The instruction that would make 2.652× real is Z-basis
  measure-and-reset, and this design needs it twice.
- **Verified:** `tests/test_dqi_predicate_gating.py` 7 passed / 1 skipped;
  the skipped re-measurement was run by hand under the kickmix interpreter and reproduces
  the record exactly. Record: `benchmarks/quantum_compilation/dqi_predicate_gating.json`.

## 2026-10-28 — Predicate-gated Forney body: 1.332× measured, and inexpressible in this IR

- **MEASURED:** `scripts/probe_dqi_predicate_gating.py` gates the Forney body on the
  truth table of `sigma(alpha^j) == 0` (supplied as an oracle classical register, so
  this is an upper bound, not a submission). At the official instance `T_mean` falls
  **5,725,082.58 → 4,298,955.87 (1.332×)** with `Q` = 1,893 and the static count
  unchanged, and with **0 value and 0 phase errors** on 128 shots: the body is a
  no-op wherever `located = is_sigma_zero & omega` is zero, so skipping it is free
  in correctness. This is the 1.335× "if the phase were free" row of
  `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.5, measured instead of estimated.
  Scaled checks: 1.185× at `m=255, GF(2^8)` and 1.011× at `m=15, GF(2^4)`, where the
  root density (2.03%, 12.7%) is much higher.
- **CORRECTED (null optimisation):** the committed `is_error` gate at
  `dqi_reference_circuit.py:573`/`:596` buys **0.006%** — 353 of 5,725,461 Toffoli per
  shot — because `P(M_j = 0) = 2^-12`. The gap between `toffoli_count` and
  `max_toffoli_count` in the receipt is that gate firing about once per shot, not a
  conditional-execution saving worth pursuing.
- **BLOCKED (IR):** the predicate cannot reach the condition stack. `push_condition`
  requires a `km.b`; the only readout is `hmr`, which is **X-basis** (on `|1>` it
  returned 31/64 ones); `push_condition(qubit)` and `bit_store(control=qubit)` are
  both rejected by the binding; and the instruction set `{NEG, BIT_INVERT,
  BIT_STORE0/1, X, Z, R, HMR, CX, CZ, SWAP, CSWAP, CCX, CCZ, Z_POW, INFO_TAG,
  DEBUG_PRINT, PUSH/POP_CONDITION}` is closed under the Z-diagonal group, so no
  Hadamard can be synthesized to conjugate the predicate into the X basis. One
  instruction — Z-basis measure-and-reset (`MR`) — is the entire difference.
- **BOUND (superseded):** 1.332× leaves `T_mean` at 4.30M against the 445,229 target
  (9.7× short) and the receipt at 18.35T — arithmetic on `R·Q²·T`, not a receipt. The
  Forney body is 24.9% of `T`; the other 75% is EEA iterations, which never skip. See
  the entry above: clearing instead of uncomputing that EEA reaches 2.652×, so this
  1.332× prices only the Forney phase.
- **Verified:** `tests/test_dqi_predicate_gating.py` 4 passed / 1 skipped; the skipped
  re-measurement was run by hand under the kickmix interpreter and reproduces the
  record exactly. Record: `benchmarks/quantum_compilation/dqi_predicate_gating.json`.

## 2026-10-28 — Benchmark II cost structure measured; two of my own optimisations falsified

- **MEASURED:** `scripts/measure_dqi_primitive_costs.py` recovers the DQI Toffoli
  breakdown, since `kickmix` exposes none. Per-call at GF(2¹²): `gf2_imul` and
  `gf2_idiv` **0** (CNOT-only linear maps), `ixor_gf2_mul` 63, Itoh–Tsujii
  inversion 315 = 5 × 63, scaffold division 378. Slopes at the benchmark widths:
  17,208 Toffoli per EEA iteration, 351 per Chien/Forney position.
- **FALSIFIED (mine):** "composite-field/Karatsuba cuts constant-multiplier cost
  30–50%" — there is nothing to cut. Multiply and divide by a constant are already
  0 Toffoli. The only multiplier that costs anything is quantum × quantum, and 63
  at degree 12 is already better than a naive schoolbook count.
- **FALSIFIED (mine):** Montgomery batch inversion of the Forney denominators. It
  needs 3 multiplications per element, so it can recover at most 2/5 of an
  inversion; at this repo's marginal qubit price 2T/Q = 6,048.7 Toffoli-equivalents
  per qubit, and 12 qubits per batched element, **every block size loses** (best
  B=4: 193,489 saved vs 290,338 spent, score ratio 1.021 — worse).
- **BOUND (new):** the Forney step inverts once per position
  (`dqi_reference_circuit.py:579`), so 4,095 × 315 = **1,289,925 Toffolis = 22.5%
  of `T_mean`, and 2.90× the entire 445,229 target** before any EEA iteration runs.
  Ceilings with `Q` unchanged: 1.259× if a position cost one multiply, 1.335× if
  the phase were free — against 12.86× required. The target is unreachable for this
  architecture, not merely unreceipted.
- **Q floor:** 1,887 measured = 1,752 persistent (`2 × 73 × 12`, both frozen) + 135
  transient. The claimed −104 is 77% of all transient qubits, and the live peak is
  EEA round scratch inside `_multiply_and_add_or_sub`; the `n_a`/`n_b` recycling
  the V6 row names frees 16.
- **Verified:** `tests/test_dqi_primitive_costs.py` 7 passed / 1 skipped; the
  skipped re-measurement was run by hand under the kickmix interpreter and
  reproduces the record exactly. Recorded in `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.5.

## 2026-10-28 — Riemann off-line scan scoping fixed; August batch screened for Benchmark II

- **FIXED (integrity):** `rust/riemann_verifier` built its off-line critical-strip grid
  as `(100..400) * 0.1` — `t in [10, 40]` — **regardless of `max_t`**, then printed
  `OFF_LINE_GRID_SCAN_PASS`. A run asked about `T=300` certified a window 260 units
  shorter than the question; the same defect class as the `PASS` removed from the DQI
  Gluzdov V6 projection the same day. The window is now `[10, max_t]` and is never
  truncated to fit the sample budget: the budget (20,000 samples) degrades resolution
  instead, and a coarsened scan reports `OFF_LINE_GRID_SCAN_PASS_COARSENED` rather than
  a bare PASS. JSON gains `grid_scan_window`, `grid_scan_step`, `grid_scan_samples`,
  `grid_scan_coarsened`, `grid_scan_covers_requested_height`.
- **Verified:** `T=60 → [10,60] step 0.1 / 501 samples`; `T=300 → [10,300] step 0.1 /
  2,901 samples`; `T=5000 → [10,5000] step 0.2495 / 20,000 samples / PASS_COARSENED`.
  `tests/test_riemann_verifier_grid_scan.py` 3/3, including a source-level revert pin.
  `probe_wave1.json` / `PROBE_WAVE1.md` regenerated against the fixed binary; verdicts
  unchanged (2 consistent_finite_range, 1 no_refutation, 1 not_probeable).
- **RECORDED (unfixed, separate):** root discovery at a fixed 0.05 step still misses
  zeros with height — 0 % at T≤100, 14.5 % at T=300, 84 % at T=5000 (721/4520). That is
  a discovery-budget problem, not the scan window, and it is now stated as such.
- **NEW:** `catalog/august_ten_advances_screen.json` +
  `catalog/AUGUST_TEN_ADVANCES_BENCHMARK_II.md` — the August 1 *Ten advances* batch is
  the only OpenAI release whose disciplines intersect Benchmark II (coding theory,
  arithmetic circuit complexity, quantum complexity, lattice cryptography), so it is
  screened against the DQI cost model. **0 of 10 rows admits a cost path**: three are
  lower bounds or hardness results (a barrier is not a tool), six have no circuit content,
  and the single coding-theory row is blocked by the frozen instance
  `(m=4095, n=70, degree=12, r=2016)` that `ProblemSpecDQI` fixes. Its only use is to put
  a floor under `R(eps_plus)`, not to lower the score.
- **Disclosed:** recomputing the target gain from the two workload rows gives 14.3973×,
  against the quoted 14.396× — a +0.009 (0.06 %) rounding difference recorded rather than
  absorbed. `T_mean` must fall to 7.78 % of reference (12.86×); `Q` only to 94.5 % (1.058×).
- **Verified:** `tests/test_august_ten_advances_benchmark_ii.py` 7/7, which cross-checks
  the frozen instance against the verifier's own `DEFAULT_M/N/DEGREE/R` and the cost model
  against the pinned workloads in `apedp_dyn.quantum.dqi_photonic_resource`.

## 2026-10-28 — Audit artifacts for the negative results, and the PDE honesty note

- **COMMITTED:** the seven `scripts/` audit instruments that back numbers already
  recorded in `CHANGELOG.md` and `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` had been
  left untracked, so the claims cited them from outside the repository. Two were
  load-bearing: the committed `run_halt_recycling_scale.py` imports
  `RecycledFlagDecoder` from `scripts/test_recycled_flags_small.py`, and the
  restored documentation cites `scripts/test_viable_optimization_path.py`.
  Verified by running the committed driver end to end (−240 counter-only at
  m=15/1023, +600/+360 flag, +240/+480 combined).
- **DISCLOSED:** importing `RecycledFlagDecoder` is safe; running it is not. Its
  toy circuit fails the official verifier (effective error rate 0.9155 ≥ 1/16,
  198/256 value errors). The flags prototype is a broken circuit, not merely a
  slower one.
- **NEW:** `docs/pde_discovery_honest_limits.md` plus
  `tests/test_dyn_pde_discovery_noise_robustness.py` — states that the default PDE
  discovery config is brittle to small additive noise and that the clean pinned
  demos are not evidence of general noise robustness, with the first fixed-config
  multi-seed sweep as the backing test (2 passed).
- **NEW:** `tests/test_dyn_dqi_photonic_resource.py` (14 passed) pins the shipped
  `dqi_photonic_resource` mapper: platform invariance of `R·Q²·T`, the fault-tolerance
  verdict, and the caller-supplied superconducting baseline having no defaults.
- **NEW:** `tests/test_dyn_rime_ad_input_contract.py` pins the `RimeAdProposer`
  input contract (`DataFrame | dict[str, DataFrame]`, unsupported shapes fail
  loudly rather than returning a candidate).

## 2026-10-28 — Portfolio synthesis names the real Benchmark I incumbent

- **CORRECTED:** `docs/invsqrt_frontier_and_portfolio_synthesis.md` still named the
  retired 90Q affine build ($157,752.008809$) as champion and carried a Benchmark I
  factor of 13.164804×. The pinned incumbent is the 89Q shared-PDDD circuit
  (`PASS`, 200,540,928 shots, `ce04c9d5…`) at $C_1 = 157,601.080829$, i.e.
  **13.177411×**. The retired rows (90Q affine, SABLE-90) are kept on the record
  with the reason each was retired rather than deleted, so the champion's margin
  stays checkable.
- **Recorded:** the FAROS-R1/R2/R3 rejection margins are re-stated against the
  89Q basis (R2 15-bit now +1,590.87 / +1.01 %, R1 +41,576.84 / +26.4 %); every
  rejection conclusion is unchanged and every margin grows.
- **Disclosed:** Benchmark I's worst error/tolerance is 0.999623, so 0.0377 % of
  the phase budget is unused — less than a single rotation is worth (see the
  rotation ledger in `PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.4).

## 2026-10-28 — Gluzdov V6 stops claiming a result it never had

- **FABRICATION REMOVED:** `dqi_gluzdov_v6.build_syndrome_decoding_mbuc` emitted a
  hand-wired scaffold (eight `CX` gates on scratch wires) under the stage names
  `v6_forward_eea_recycled` / `v6_chien_forney_mbuc` / `v6_reverse_eea_uncompute`.
  It was not a decoder: no EEA, no Chien search, no Forney evaluation. The module
  docstring nevertheless announced a 14.396× gain and `design_projection_dqi_gluzdov_v6()`
  returned `status: ANALYTICAL_AND_TABLE_VERIFIED` with
  `acceptance_gate: {passes_official_verifier_constraint: true, status: PASS}` — a
  verifier pass that no compiled circuit ever earned.
- **Now:** the circuit builder delegates to the real
  `ProosAndZalkaDecoder.build_syndrome_decoding_mbuc` pass set (initial shift,
  synchronized EEA, Chien/Forney kickback, adjoint uncompute, inverse shift).
  `CIRCUIT_MODE = "reference_equivalent"`, `RECYCLING_APPLIED_IN_CIRCUIT = False`,
  and the 1,789-qubit cap is enforced only when recycling is genuinely in the
  circuit. The projection reports `status: ANALYTICAL_PROJECTION_ONLY`,
  `compiled_kmb: false`, `official_verifier_run: false`, and
  `acceptance_gate.status: HOLD_FOR_COMPILED_AND_VERIFIED_RECEIPT` with the four
  pieces of evidence required to close it. The 14.396× factor is a target.
- **NEW:** `dqi_builder.build_candidate(n_iters, architecture=...)` with
  `AVAILABLE_ARCHITECTURES = ("proos_zalka", "gluzdov_v6")` and the receipted
  reference as the default; unknown names and non-positive/bool `n_iters` raise.
- **Verified:** `tests/test_dqi_gluzdov_v6.py` 10 passed / 1 skipped (the skipped
  one needs the kickmix extension).

## 2026-10-28 — Restore the deleted module registry in `docs/DOCUMENTATION.md`

- **INTEGRITY REPAIR:** commit `c08acbe` ("Document stability selection") replaced
  the 2,134-line `docs/DOCUMENTATION.md` with a 35-line stub, deleting the module
  registry (310 of 312 `apedp_dyn` modules lost their index entry, 429 module
  links, and the `[5 %, 90 %]` tactical-bound discussion). Nothing flagged it
  because the four firewalls in `tests/test_documentation_and_spec_integrity.py`
  were left failing and reported as pre-existing. They were not pre-existing —
  they were the regression being detected.
- **Restored** the registry from `444c1fa` (2,347 lines, 523 repository-relative
  links, 312/312 modules indexed) and re-applied the additions made since
  (stability selection, DQI photonic resource mapping, 89Q champion
  reproduction, mixed polarity, photonic boundary) as §23. The restored text
  carries the numbering defect it had at `444c1fa` (two `## 18` and two `## 19`
  headings, sections 16/17/20/21/22 headingless); it is left as-is rather than
  renumbered, because renumbering breaks every `#N-...` anchor the TOC uses.
- **Added** the one genuinely unindexed module, `apedp_dyn/video/collision_domain.py`.
- **Verified:** `tests/test_documentation_and_spec_integrity.py` 6/6 pass (link
  coverage ≥ 200, zero orphan modules, zero pinned debt, registry status tags
  agree with code `Status:` lines).

## 2026-10-28 — Rotation overhead closed, halting-counter saving falsified, photonic artifact

- **MEASURED (negative):** the 44 priced rotations in the 89Q invsqrt champion
  cost 315.299429 of the 1770.798661 Toffoli-equivalent (17.81 % of the score)
  and cannot be reduced. `scripts/search_rotation_merges.py`: 0 exact mergeable
  pairs (all 44 sit on distinct qubits; the 7 qubits carrying a second rotation
  have 75–192 non-commuting gates between the two), and 0 affordable drops (the
  smallest rotation, 2⁻³⁵ = 2.998e-11 turns, is 7.5× the entire 3.975e-12-turn
  phase slack). Recorded in `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.4.
- **FALSIFIED:** halting-counter recycling does not hold at scale.
  `scripts/run_halt_recycling_scale.py` measures −240 Toffolis at every instance
  (constant, because one fixed-width comparison per call site is removed):
  −0.722 % on the toy m=15/n=6/GF(2⁴), −0.0118 % at the benchmark instance
  m=4095/n=70/GF(2¹²) (2,033,345 → 2,033,105). The shipped prototype (flag +
  counter) is a **regression** at every size measured (+480 at full scale). The
  sFFCC table in `PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.2 is corrected accordingly.
- **NEW (non-submission):** `scripts/report_photonic_non_submission.py` prices all
  three champions on the sFFCC lattice at L=16 and writes
  `artifacts/quantum_compilation/photonic_non_submission.json` with
  `submission_status: NON_SUBMISSION`, `score_delta: 0.0`, and the reasons: the
  verifiers consume a compiled `.kmb`/`.kmx` artifact, and none of the three
  score formulas contains a time, photon, fusion or detector term.
- **NEW:** `docs/BENCHMARK_BREAKING_LITERATURE.md` — published techniques priced
  against each benchmark's measured exchange rate (I: 89.0 units/Toffoli vs
  1770.8/qubit; II: `R·Q²` vs `2RQT`; III: 35,990.0/Toffoli vs 4,596,711.8/qubit),
  including the two techniques that provably cannot help (better Clifford+T
  synthesis cannot move Benchmark I, whose rotation cost is the benchmark's own
  formula of ε alone).

## 2026-10-27 — Stability selection utility + blind-benchmark PDE work

- **NEW:** `apedp/stability_selection.py` — general stability-selection helper
  (`stability_select`) for repeated sparse fits with different seeds/subsamples:
  keeps terms with selection frequency >= pi, averages coefficients, returns empty
  dict on all-failure. Honest caveats: only helps stochastic fitters; deterministic
  fitters see no benefit.
- **WORKSPACE (not committed to repo):** blind-benchmark weak-form PDE fitter
  and stability-fit variant in `/tmp/buffy/blind/`. Results: stability selection
  (PI=0.9 chosen on VALID, frozen, THRESH=0.02, K=20, 5 seeds/BENCH)
  improves exact recovery from 50/100 to 66/100 on BENCH heat/burgers/allen_cahn/
  fisher_kpp. Heat still fails at noise 0.05/0.1 (0/5 each). Full artifacts:
  `/tmp/buffy/blind/bench_stability.json`, `benchmarks/mutation_battery_report.json`
  and the other `/tmp/buffy/blind/*.json` still on disk (no process delete).
- **Caveats recorded on disk:** `/tmp/buffy/blind/limits.txt` (reproducibility,
  seed count, noise model, synthetic data, heat failure, K=20 choice).

### Honest limitations (not fixed, disclosed)
- The utility is NOT a general PDE-discovery fix. Repo PDE path
  (`apedp_dyn/epistemic/pde_discovery`, Rust STRidge) is deterministic and was
  NOT improved by this; on a sample heat+noise case it still returned many
  spurious terms. Stability selection applies only to fitters that randomize
  their structure (here: weak-form test-function windows).
- Heat remains the hard case under noise; this is a partial reduction of spurious
  terms, not a complete solution. Root cause points to library/sparsity, not
  threshold/selection tuning alone.
- Validation-BENCH split: PI=0.9 was picked on VALID only and frozen before BENCH,
  so there is no benchmark tuning. But the result still depends on a single
  frozen pi and a single threshold; different noise models / domains may need
  re-tuning on a held-out split.

### Non-deleted working processes on disk
The following working directories / artifacts exist and were NOT deleted:
- `/tmp/buffy/blind/` — blind PDE benchmark scripts + results + logs.
- `benchmarks/mutation_battery_report.json` — mutation battery report (still
  referenced by the repo).
- `scripts/search_claims_pddd.c` — search-claims probe source (new, not deleted).
- Other existing `/tmp/` working artifacts remain as-is; this commit does not
  touch them.

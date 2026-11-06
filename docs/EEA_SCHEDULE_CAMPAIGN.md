# The shift/sweep/swap schedule campaign: read the predicate, then move

**Status:** measured campaign · **Scope:** the three regions that were 78.3% of the
472,768.2-Toffoli non-product residual in the six-lever DQI arm
(`shift_b_array_right` 162,479.6, `sweep_positions` 143,038.0 non-product, `swap_a_and_b`
64,224.0) at the pinned instance (m=4095, n=70, GF(2^12), 64 shots, 222 EEA rounds) ·
**Evidence:** `benchmarks/quantum_compilation/dqi_schedule_campaign.json`, produced by
`scripts/measure_dqi_schedule_campaign.py` on the local `MR` rebuild; rules in
[`CIRCUIT_OPTIMISATION_RULES.md`](CIRCUIT_OPTIMISATION_RULES.md); this campaign closes
§5–§6 items 2–3 of [`EEA_AND_PREDICATE_PLAN.md`](EEA_AND_PREDICATE_PLAN.md).

The plan ranked those three regions as "a different schedule, not a tidier gate" and closed
pointer substitution until it was proved. This is the campaign that discharges the
obligation, and it answers with a measured saving rather than an argument:
**1,221,466.8 → 686,014.8 executed Toffolis per shot (−43.84%) at unchanged `Q` = 1,913**,
which is **1.541×** the 445,229 target instead of 2.744×, and **1.451×** the 472,768.2 floor
that bracket now uses as its aim (see "What this campaign is aimed at"). That figure is the
*stacked* arm:
the three movement/product gates (`everything`, 688,966.8) plus the flag/metadata gate the
earlier combined arm was promoted on, re-measured here as `control_gates` and reproduced to
the Toffoli (2,952.0) so that the two sessions' numbers are one record rather than two. The
stack was declared before the arm was built — `S5_stack`, `532,500 + 2,952 = 535,452` →
`T = 686,014.8` — and it matched to **0.0 Toffoli**.

**The target this campaign is measured against is a target, not a derivation.**
`scripts/measure_dqi_recycling_offsets.py` reads the decoder's own `n_A`, `n_B`, `n_a`,
`n_b` after every EEA sub-step. The a-pair is constant at `n + 1` rather than `n` and the
b-pair varies, so only `n_a` is recyclable: 8 qubits at `n = 70`, a supported peak of
**1,885**, not 1,789. Recycling cannot move a single gate, so 445,229 has no route from
it — the 1.541× below is measured against the campaign target, and the route to the
target is the multiply-accumulate arithmetic this campaign does not touch. Record:
`RECYCLING_MEASUREMENT` in `apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py`.

**That route is now closed, and this campaign must not be read as depending on it.** The
build that removes `n_a` exists and peaks at 1,885 (`dqi_recycled_degrees_window.json`), but
it **fails 64-shot equivalence** — 52/64 value errors for the one-body substitution, 64/64
for the all-sites build, against the reference's 0/64 — because the a-pair invariant dies at
the first `_swap_a_and_b`, which exchanges a b-pair whose sum varies. It is also
**score-negative**: `+648,784` Toffolis (1.1133×) against a 0.9916× `Q²` gain makes it
**0.898×** the projection. So recycling is closed on both grounds, no qubit from it is
usable, and the 1.541× and every figure below stand on the schedule alone. Full record:
[`DQI_RECYCLED_PASS_BODIES.md`](DQI_RECYCLED_PASS_BODIES.md) §§"Root cause", "What that
cost does to the projected score".

### What this campaign is aimed at

The 445,229 in the ratios above is the **denominator**, not the aim. The reason first given for
that is **withdrawn**: 472,768.2 Toffolis per shot is the *six-lever* arm's non-product residual,
and this campaign took 172,815 out of it, so §4's arm carries **299,953.2** non-product and the
target sits **145,275.8 above** its floor rather than 6.19% below it. Freeing this arm's products
is therefore **0.688×** the denominator — a record, not a loss
([`BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) §7.13). What keeps the
target out of reach is that it needs the surviving **386,061.7** of products at **37.6%** of
their cost — about 15.6 ANDs per multiply, below the `2n − 1 = 23` lower bound — and at that bound
the arm is 1.019× at its own width, 0.998× only with the 20 qubits returned. So the aim is now
the two things that are still movable: **386,061.7 → 145,275.8** on the product stream, and
**1,913 → 1,893** on this arm — which P6 below makes a *price* rather
than a free cut: every schedule arm runs at `Q` = 1,913, identical to its control, so the five
gates cost no width and dropping them recovers none. The 20 qubits are the six levers', and
[`BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) §7.11 prices them at
4,212,030.2 Toffolis per shot.
`TARGET_ROLE` in `dqi_gluzdov_v6.py` records the decision as `SCORE_DENOMINATOR_NOT_AIM`, and
[`BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) §7.9–§7.10 carries the
frontier table it sits in.

## 1. The schedule fact, and what it licenses

Every predicate register in this decoder is in a computational basis state, because the
syndrome input is: the decoder is a classical reversible computation whose only quantum
content is the Forney phase. A basis-state qubit can therefore be **read** — `cx` into the
shared readout ancilla, then `mr` — with a deterministic outcome, the source untouched, and
the tracked phase unchanged. The result is a *classical* bit, and a classical bit can gate
the work it enables (`push_condition`).

This is not new machinery: it is the pattern the six levers were promoted on (they read
`c3`, `flag`, `sigma` and `is_hc_zero` exactly this way), it is the pattern
`_read_all_zero` uses for the Forney body, and the reference decoder's own MBUC output
(`error_mbuc`) is classical. What is new is applying it to the *movement* predicates at
per-position grain instead of per-round grain.

The measurement consequence is the one that matters for the score: **a gate that is
classically conditioned on a bit that is zero executes no Toffoli on that shot.** So a
movement, a product or a chain of either that the data path does not need costs nothing,
instead of costing its full static count on every shot.

## 2. The four hypotheses, declared before the arms were built

The script computes these from the control arm's own chain costs and prints them before any
variant exists; `prediction.written_before_variants` records that ordering.

| # | predicate read | declared prediction | why |
|---|---|---|---|
| **S1** | the sweep's quotient selection (`n_q == j`, `n_q == j+1`) | saving ≥ `chain − burn` = `105,696 − 5,328` = **100,368** | the selection is one-hot: at most two of the up-to-72 emitted positions carry the quotient, so every other chain cswap is a no-op |
| **S2** | the shift's `do_shift` mask | **no saving predicted** (lower bound 0); under 1% of the region closes the shift half | `do_shift` is the XOR of three threshold intervals, so the firing set is the active window, not a one-hot |
| **S3** | the wholesale A/B exchange | composing the shift with the swap saves **0.0** | their predicates are disjoint (`ctrl` is one-hot and the flag is one value) |
| **S3b** | the shift's `c3` branch | dropping the shift in every `c3` round saves **0.0** and stays correct | the flag is set in those rounds, so the branch is the identity — **falsified on re-measurement (§3.3): the flag is clear in 2,556 of 4,608 `c3` rounds, the arm finds 45,396.1 and is incorrect** |
| **S4** | the product's `mul` control | **no saving predicted** (lower bound 0); ≥ 50,000 promotes | `mul` is the multiply window, not a one-hot, so the saving is whatever the window excludes |
| **S5** | all five gates in one circuit (S1–S4 plus the two already-promoted flag/metadata gates) | saving = `532,500 + 2,952` = **535,452**, i.e. `T` = **686,014.8** | the three schedule gates remove movement and product no-ops; the two flag/metadata gates remove update work in dead rounds; the work sets are disjoint, so the savings add |
| **S6** | the multiplier of every surviving product (`q_i != 0`) — Experiment B | **no saving declared (0.0)**; it promotes only if it beats its own read cost | a live round's quotient has a nonzero leading term, so the predicate should be zero in too few positions to pay for a read at every one (see §2.2) |

S5 is a *second* prediction and is written at a different point in the run: after the
schedule arms are measured and before the five-gate arm exists, because its first input is
this run's `everything` measurement and its second is the prior record's 2,952 (not this
run's re-measurement of it, which is the independent check). `written_before_the_stacked_arm`
records that ordering, and its falsifier is the obvious one — an increment over `everything`
that differs from 2,952 means the two gates interact and the sum is wrong.

### 2.1 The preconditions, as statements rather than as prose

The campaign is only valid under conditions that have to be stated, because each one is a
place where the argument could be wrong. They are listed here with how each is secured:

| # | invariant | how it is secured |
|---|---|---|
| **P1** | every register the campaign reads (`n_q`, the shift mask, the `mul` mask, `c3`, `flag`, `sigma`, `is_hc_zero`) holds a **computational-basis value** at the point it is read | by construction: the input is a basis state and every operation before each read is a basis permutation — NOT, CNOT, Toffoli, controlled `iadd`/`isub`, `cswap`, `left/right_rotate`, `flip_if_equal`, `flip_if_less_than`, `init_and`, `ixor_gf2_mul`. No Hadamard or rotation touches an EEA register before the Forney phase (checked by the rule in `CIRCUIT_OPTIMISATION_RULES.md` §2(c)) |
| **P2** | the read is **deterministic and non-destructive**: `cx` into the shared ancilla, `mr`, and the ancilla returns to `|0>` so it can be reused without uncomputing | by construction (the gadget is the six-lever one) and measured: the ancilla is allocated once, and every arm passes the junk/restoration check |
| **P3** | the gated block is **the identity on any basis state where its predicate is zero**, so skipping it is not skipping work | per region, algebraically: S1 a cswap at a position the one-hot selection does not name swaps two equal lanes; S2 `do_shift = 0` makes the rotation lane all-zero; S4 a product outside the `mul` window multiplies by zero. Secured by the correctness oracle *and* by a falsifier whose condition is wrong (§3.2) |
| **P4** | the read register is **not one of the block's own controls**, so the classical condition and the quantum control are independent | by construction: `ctrl` is built from `counter`, `is_c3`, `is_hc_zero`; the predicates read are `n_q`, `sigma`'s mask, and the `mul` window, none of which is a factor of `ctrl` |
| **P5** | gating two blocks on **different** registers removes two independent sets of no-ops, so the savings add | measured, not asserted: `both` = 105,696's saving (103,968) + 65,895 = 169,863.0 exactly, and `everything` = 103,968 + 65,895 + 362,637 = 532,500.0 exactly |
| **P6** | the read costs **no qubit, no EEA iteration and no phase** | measured in every arm: `Q` = 1,913, 222 forward rounds, tracked phase matched, syndromes restored |

P3 is the one that cannot be discharged by counting alone: a gate that skips work the
decoder needs is cheaper *and* wrong, which is exactly what the three falsifiers in §3.2
demonstrate. The invariants are stated for the schedule as it stands; a variant that reads a
register which is *not* a basis state (for example one entangled by a superposition of
syndromes) would void P1 and with it every number here.

### 2.2 Experiment B: the product stream's own operand, measured and closed

The next lever the review proposed was *inside* the multiplication: the surviving products are
`ixor_gf2_mul(q_i, a_j)`, so a zero multiplier makes the product the identity, and the
argument that licenses S1–S4 licenses gating on it. What the review could not know, and what
this campaign measures, is whether that predicate is cheap enough to be worth reading.

| # | predicate read | declared prediction | why |
|---|---|---|---|
| **S6** | `q_i != 0`, the multiplier of every surviving product | **no saving declared (0.0)**; the arm promotes only if it beats its own read cost | a live round's quotient has a nonzero leading term, so the predicate is expected to be zero in too few positions to pay for a read at every position |

It is built and measured, and it is **correct and loses**: the arm computes the reference's
function, so the predicate is *sound* — reading the multiplier and skipping the product when
it is zero is a legitimate schedule change — but it costs **+311,780.7 Toffolis per shot** over
the five-gate incumbent (1,221,466.8 → 997,795.6, still far above 686,014.8) and adds
327,094 mid-circuit reads per shot (357,347 in total, against the incumbent's 30,253) plus
956,116 instructions. Two facts make that structural rather than unlucky:

1. **The multiplier is not a round-level quantity.** The sweep's selection chain `cswap`s
   `q_i` into the a-lanes as the quotient is distributed, so `q_i` changes inside the loop.
   The first version of this arm read it once per round and was **incorrect** — the read is
   only valid at the position that emits the product, which is where the arm now reads it.
   (Both versions are in the record's history: the incorrect one is the reason the arm reads
   per position, and the correct one is what the numbers above are from.)
2. **A read costs more than a product is worth skipping.** The 12-bit equality test plus its
   `mr` measures at **0.95 Toffolis** per site (311,780.7 ÷ 327,094), while one skipped
   product is worth 66 and only about half the positions are inside the `mul` window. The
   gate therefore needs the multiplier to be zero in more than ~3% of live positions to break
   even. The loss is almost exactly the read cost, which means the skipped-product saving is
   indistinguishable from zero: **the multiplier is essentially never zero where the product
   executes.**

So the remaining product window is **not** a predicate problem: the 331,023 Toffolis that
survive the `mul` gate are products the data path asks for, and removing them needs a
cheaper multiply (a smaller base case, or an opcode that skips zero lanes natively), not
another classical read. Experiment B is closed on a measurement rather than on an argument.

## 3. What was measured

All arms: `Q` = 1,913 (unchanged — the reads are classical bits and one shared ancilla),
222 forward EEA rounds, syndromes restored, phase matched, no junk, and every region row
partitioning its own total.

| arm | executed T/shot | saving | of the arm | what moved |
|---|---|---|---|---|
| control (six levers) | 1,221,466.8 | — | — | reproduces `dqi_region_ledger.json` to the Toffoli |
| **S1 sweep** | **1,117,498.8** | **103,968.0** | 8.51% | selection chain 105,696.0 → **1,728.0** |
| **S2 shift** | **1,155,571.8** | **65,895.0** | 5.39% | shift chain 131,220.0 → **65,325.0** |
| **S4 products** | **858,829.8** | **362,637.0** | 29.69% | product stream 693,660.0 → **331,023.0** |
| both (S1+S2) | 1,051,603.8 | 169,863.0 | 13.91% | both chains, exactly additive |
| **everything (S1+S2+S4)** | **688,966.8** | **532,500.0** | 43.60% | exactly `103,968 + 65,895 + 362,637` |
| `control_gates` (the prior session's arm) | 1,218,514.8 | 2,952.0 | 0.24% | reproduced inside this record: flag/metadata gating, zero reads |
| **stacked (all five)** | **686,014.8** | **535,452.0** | **43.84%** | exactly `532,500.0 + 2,952.0` |
| S3 composed | 1,221,466.8 | 0.0 in this record; **1,218,514.8 → 1,218,514.8 = 0.0 re-measured** | — | vacuous here (the arm was the control circuit); rebuilt and re-measured, and still 0.0, correct (§3.3) |
| S3b shift's `c3` branch dropped | 1,221,466.8 | 0.0 in this record; **1,218,514.8 → 1,173,137.1 = −45,377.7 re-measured, incorrect** | — | vacuous here; the rebuilt arm finds real work the decoder needs, so S3b's premise is false (§3.3) |

`Q²T` falls from 4.4700e12 to **2.5105e12** (−43.84%). The saving is not a projection: each
row is a simulated, correct arm, and the region rows say which chain moved. The `stacked`
row is the one that matters for the official comparison, because `control_gates` is the arm
the previous session's 2,952.0 was measured on; putting both in one record is what makes
"the two gates compose" a measurement rather than an inference across two reports.

### 3.0 Does the saving remove work, or move it?

The question a schedule change has to answer ("cheaper locally, dearer elsewhere" is the
normal failure) is answered by counting the whole instruction stream on the same run, not
just the Toffolis:

| arm | executed Toffolis/shot | executed instructions/shot | mid-circuit reads/shot |
|---|---|---|---|
| control (six levers) | 1,221,466.8 | 37,808,019.7 | 0 |
| `everything` (S1+S2+S4) | 688,966.8 | 32,470,261.7 | 30,253 |
| `stacked` (all five) | 686,014.8 | 32,451,795.1 | 30,253 |

So the stacked arm removes **535,452 Toffolis *and* 5,356,224.6 instructions** per shot
(−14.17%), while adding 30,253 reads at 2 instructions each (`cx` + `mr`) = 60,506 — 1.1% of
the Toffoli saving, and 0.19% of the arm's instruction stream. Per removed Toffoli the arm
loses 10.12 instructions, which is the expansion the removed `CCX`es had: the work is
**gone**, not relocated. What is added is not hidden either — it is 30,253 mid-circuit
measurements with classical feed-forward per shot, a cost the score does not charge and a
device would. That is the honest form of the claim: *the nonlinear work falls by 43.84%,
the instruction stream falls with it, and the price is measurement depth.*


### 3.1 What each prediction came out as

- **S1: 103,968.0 against a declared floor of 100,368.** The floor assumed two selections in
  every one of the 222 rounds; 144 selections actually occur, so the chain left is
  `12 × 144 = 1,728` rather than the declared burn of 5,328. The floor was therefore
  conservative by 3,600, and the observation is 103.6% of it — the direction a lower bound
  is allowed to be wrong in.
- **S2: 65,895.0 — no prediction, and the measurement says the mask is half dense.** Exactly
  50.2% of the shift chain is skipped: the `do_shift` window genuinely moves about half the
  positions per live round, and the other half were quantum-controlled no-ops. So the shift
  is **promoted for the half the mask zeros and closed for the half it does not**: 96,584.6
  of the region survives, of which 65,325 is movement the data path asks for, and nothing
  short of a different schedule of movement can remove it.
- **S3: 0.0, measured; S3b: falsified.** The swap's `c3 ∧ flag` cannot overlap the shift's
  `c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))` because `ctrl` is one-hot and the flag is one value, so a
  schedule that fuses the two has nothing to fuse. In this record both arms were *identical*
  to the control by construction — the method that would have changed them was never called
  (§3.3) — so neither tested the predicate. Rebuilt under the name the decoder actually
  calls and measured at the pinned instance: the composed arm saves **0.0** and is correct,
  which confirms S3; the arm that drops the shift from every `c3` round saves **45,396.1** and
  is **incorrect**, which falsifies S3b's premise (the flag is *clear* in 2,556 of the 4,608
  `c3` rounds). The per-round table in the record (164 rounds with shift executions, 92 with
  swap executions, 66 where the batch shows both) is an aggregate over 64 shots and is **not**
  per-shot evidence; §3.3 and [`DQI_MOVEMENT_SCHEDULE.md`](DQI_MOVEMENT_SCHEDULE.md) §6 are.
- **S4: 362,637.0 — 52.3% of the product stream.** The product's `mul` control is a window,
  and roughly half of the positions it is emitted for are outside it; those products were
  executing 66 Toffolis each for nothing.
- **S5: 535,452.0 measured against 535,452.0 declared, error 0.0.** The five gates compose
  exactly: the increment over `everything` is 2,952.0, which is the prior arm's own saving,
  and the prior arm re-measured inside this run at 1,218,514.8 with zero reads. A stack that
  had shared work would have landed either side of that figure; it landed on it.

### 3.2 The falsifiers, and what makes them informative

Each variant was also built with its condition made *wrong*, and every one of those arms
fails correctness:

| falsifier | what it does | chain skipped | arm total | correct |
|---|---|---|---|---|
| `falsify_sweep_stale` | condition each selection on the *previous* position's bit | 104,768.1 | **+217,720.1** (costs more) | no |
| `falsify_shift_complement` | condition each move on the complement of its mask | 69,311.6 | −79,854.5 | no |
| `falsify_shift_stale` | condition each move on the previous slot's mask | 70,351.1 | −80,612.0 | no |

Two things are worth stating about that table, because both are traps.

1. **An arm's total is not a measure of what it skipped.** The stale-selection falsifier
   skips 104,768 of its chain and still *costs* 217,720 more than the control, because a
   wrong state runs a different control flow (the multiply window changes, the halting
   counter changes). The record therefore measures skipped work on the chain the condition
   gates, not on the arm total, and a falsifier that skipped nothing is recorded as
   **vacuous** rather than counted as a pass.
2. **The falsifiers save more than the honest arm and are wrong.** The complement falsifier
   skips 69,311.6 against the honest gate's 65,895.0: the honest gate skips exactly the
   mask's zeros, and any gate that skips more is skipping movement the decoder needs. That
   is the shape a real gate has.

### 3.3 Correction: the two zero-saving arms measured nothing, so they were rebuilt

`composed` (S3) and `drop_shift_c3` (S3b) both measured exactly 0.0, and the text
above reads that as the extra condition being unsatisfiable. That is not what
happened. The code implementing both sat in a method named `_shift_a_and_b`, and
nothing called it: the reference and `MRGatedDecoder` call `_shift_b_array_right`.
`scripts/measure_dqi_movement_schedule.py` compares the three emitted streams at
m=15, n=6, GF(2^4) and found `none`, `composed` and `drop_shift_c3` identical --
149,337 instructions and 16,610 static Toffolis each -- so both arms were the
control circuit and their 0.0 was **vacuous**.

The method is now named for the method it overrides (`_shift_b_array_right`), so
the gates inside it are live, and this record was re-run. Everything else in it
reproduced to the Toffoli — control **1,221,466.8**, `everything` **688,966.8**,
`stacked` **686,014.8** — and the two arms are now arms:

| arm | `T`/shot | saving | correct |
|---|---|---|---|
| control (six levers) | 1,221,466.8 | — | yes |
| **S3 `composed`** (skip the shift where the exchange fires) | **1,221,466.8** | **0.0** | **yes** |
| **S3b `drop_shift_c3`** (skip the shift in every `c3` round) | **1,176,070.7** | **45,396.1** | **no** |

The same two arms, re-measured independently by
[`DQI_MOVEMENT_SCHEDULE.md`](DQI_MOVEMENT_SCHEDULE.md) §5 against a base that
carries the two promoted flag/metadata gates (`campaign_none` = 1,218,514.8, which
is this record's `control_gates`), give 1,218,514.8 (0.0) and 1,173,137.1
(45,377.7) — the same conclusions on a base 2,952.0 lower, which is exactly the
two gates. Their emitted streams differ from the control's too (8 and 3 classical
stores per round), so neither arm is a copy any more.

So S3's prediction is now **measured** rather than assumed: an arm that skips the
shift exactly where the exchange fires executes the *identical* Toffoli count and
stays correct, because there is nothing to skip -- the shift is already dead in
every round the exchange runs in. S3b's premise is **false**: dropping the shift in
every `c3` round does find 45,396.1 of work on this record's base (45,377.7 on the
2,952-lower one), and the decoder asks for that work, so the arm is wrong. The
`¬flag ∧ c3` branch of the shift is live, not dead.

Why both are true at once is the probe in
[`DQI_MOVEMENT_SCHEDULE.md`](DQI_MOVEMENT_SCHEDULE.md) §6: over 64 shots and 222
rounds, `c3` fires **72** times on every shot (4,608 firings, invariant), the
exchange fires **27-36** times (2,052 total), and the post-update flag is *clear* in
**2,556 of the 4,608** `c3` rounds. The flag is therefore not redundant in a `c3`
round, which is why S3b's arm is wrong and why exchanging on `c3` alone is wrong
(Experiment C's falsified C3); and in the 2,052 rounds where the flag is set the
shift's `¬flag ∧ (c0 ∨ c3)` term is zero while `c1` and `c2` are zero by one-hotness,
which is why S3's arm has nothing to skip.

## 4. Where the arm now stands

`686,014.8 = 386,061.7 field products + 299,953.2 non-product`, at `Q` = 1,913, 222 rounds:

| part | T/shot | share | status |
|---|---|---|---|
| `ixor_gf2_mul` products (10,510 calls) | 331,023.0 | 48.3% | half the calls survive — the multiply window |
| inversion + division products | 55,038.7 | 8.0% | not gated by this campaign |
| **non-product residual** | **299,953.2** | **43.7%** | 472,768.2 − 172,815 |
| total | 686,014.8 | 100% | **1.541×** the 445,229 target |

The non-product residual is now: `swap_a_and_b` 64,224.0, `shift_b_array_right` 96,584.6
(31,259.6 of mask arithmetic plus the 65,325 chain whose mask fires), `shift_step` 30,912,
`update_flag` 30,240, `sweep_positions` 39,070.0, `update_metadata` 12,096,
`update_halting_counter` 10,656, the alignment pair 7,392, `eea_step` 4,662, the
multiply-and-add control 2,016, `quotient_div` control 1,728, Forney control 372.6.

Two of those are still control-dependent work the campaign did not touch: the swap's
876-CCX exchange (already round-gated, and inert in rounds the shift covers — measured 0.0
for both fusion and dropping the shift's `c3` branch) and the shift's 30,912 of mask
arithmetic. Neither is reachable by the read-then-gate argument, because neither is gated on
a predicate that is zero for whole shots: the swap's `c3 ∧ flag` is already read, and the
mask arithmetic is what *computes* the predicate.

## 5. What this does and does not change about the target

**It does not make 445,229 reachable.** The arm is 240,785.8 Toffolis above it, and the
remaining pieces are enumerated in §4 with their measured sizes; three of them (the swap,
`shift_step`, `update_flag`) are at their floor for this schedule.

**It does remove the earlier conclusion's stated reason.** `EEA_AND_PREDICATE_PLAN.md` §3
argued that control-logic work could not reach the target because "`C` alone is 1.062× the
target". That is still arithmetically true of the *static* residual — but the executed
residual is not `C`: control-logic work measured here removed 43.8% of the arm, because most
of `C` was quantum-controlled work whose control was zero. The remaining gap is no longer a
control-logic question; it is 240,785.8 Toffolis of products and unconditional work.

**It does not say the remaining product stream is a predicate away.** Experiment B (§2.2)
measured the one predicate left inside the multiply — the multiplier's zero-ness — and it is
sound but unaffordable at per-position grain (`+311,780.7` Toffolis per shot for 163,547
reads). The 331,023 surviving products are work the data path asks for.

**It does not produce a receipt.** The arm still needs `MR`, which the pinned competition
binary does not have (`docs/DQI_RESOURCE_RECONCILIATION.md` §2). Every figure here is a
local measurement of an executed-gate count on the `MR` rebuild.

**It does change the cost model, and that has to be said plainly.** An executed count that
depends on classical conditions is a count under **adaptive execution**: each read is a
mid-circuit measurement whose outcome steers the following gates. The benchmark's model
permits it — the reference decoder's own `error_mbuc` is classical feed-forward, and the six
levers were promoted on the same mechanism — but this campaign applies it at per-position
grain, which means thousands of measurements and classical decisions per shot rather than a
handful. A submission that forbids mid-circuit classical control would not be able to claim
these counts, and no such submission is claimed here.

## 6. Standing measurement obligations

Established by this campaign, in force for anything that builds on it:

1. **A variant must reproduce the control to the Toffoli first.** The campaign's own control
   arm is a *copy* of the six-lever arm with the two chain marks added, and it is rejected
   unless it lands on 1,221,466.8 and reproduces the three region rows of
   `dqi_region_ledger.json` to 0.1. That check is what licenses the copy.
2. **A classical condition costs no qubit, no iteration and no phase.** `Q`, the EEA round
   count and the tracked phase are checked identical in every arm.
3. **A predicate that is read must be the one that gates the work.** Every variant ships with
   a falsifier whose condition is wrong, that falsifier must break correctness, and it must
   have skipped work to count as evidence at all.
4. **Skipped work is measured on the chain, never on the arm total**, because a wrong state
   changes the control flow (see §3.2).
5. **A saving is reported with the instruction stream and the read count beside it.** A
   Toffoli figure alone cannot distinguish work removed from work relocated, so every arm
   carries `instructions_per_shot` and `mid_circuit_reads_per_shot` (see §3.0). A future arm
   that quotes a T saving without both is not comparable to this record.

## 7. Pins

`tests/test_dqi_schedule_campaign.py` pins this record: the declared predictions and the
`written_before_variants` flag, the control's exact reproduction of the six-lever total and
its three region rows, the four savings to 0.1 Toffoli, the additivity of the three gates,
the prior session's combined arm reproduced in-record (`control_gates`), the stacked arm's
exact `532,500.0 + 2,952.0`, the fact that the instruction stream falls with the Toffolis
while the reads are counted, `Q` = 1,913 and 222 rounds in every arm, both zero-saving arms,
and the three falsifiers being wrong *and* informative. A future edit that quietly
re-samples the arm, promotes a gate whose falsifier passes, quotes a saving the region rows
do not support, or quotes a T count whose instruction stream contradicts it fails there.

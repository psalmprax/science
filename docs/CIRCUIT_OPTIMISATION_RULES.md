# What counts as a better circuit here

**Status:** rules, with the measurements they are built on · **Scope:** Benchmark II and
the DQI decoder in `kickmix` · **Evidence:** `benchmarks/quantum_compilation/kickmix_mr_arm_costs.json`,
`benchmarks/quantum_compilation/gf_mul_and_static_headroom.json`,
`benchmarks/quantum_compilation/gf54_mul_kickmix.json`,
`benchmarks/quantum_compilation/gf54_native_opcode.json`,
`scripts/measure_gf_mul_and_static_headroom.py`, `scripts/measure_gf54_mul_kickmix.py`,
`scripts/measure_gf54_native_opcode.py`, `docs/MR_INSTRUCTION_SPEC.md` §6–§7.

"Optimised by 4.320×" is a claim about six transformations of one circuit. Whether it
compares two circuits, or only compares two ways of counting, depends on four things that
have each been quietly load-bearing in this project: what the metric counts, whether the
transformed circuit is still the same unitary, where a measurement is allowed to appear,
and which instance the two circuits were run on. This note fixes all four, so a future
claim can be checked instead of argued.

## 1. The metric must be named: executed or static

Two counts come off the same circuit and they rank the levers differently.

| Count | What it counts | Six-lever arm | Reference | Ratio |
|---|---|---|---|---|
| **Executed** (`T_mean`, the scored quantity) | a conditioned gate only in the shots that run it | 1,325,230.0 per shot | 5,725,071.6 | **4.320×** |
| **Static** | every T-like gate emitted, conditions ignored | 3,486,597 | 5,725,461 | 1.642× |

Both are measured on the same circuits (`gf_mul_and_static_headroom.json`). The gap is the
whole mechanism: **gating adds conditions, it does not delete gates.** The quotient gate
moves the executed count by 56,160 Toffolis per shot and the static count by exactly 0. The
entire static saving — 5,725,461 → 3,486,597 — is `clear_eea`, the one lever that replaces
the reverse EEA with a measurement instead of conditioning it.

**Rule.** Every T-count claim names its metric. A claim that mixes them — quoting the
executed reduction and calling it a gate count — is not admissible, and neither is one that
presents a static reduction as evidence for a lever that only gates. A static scorer would
credit `clear_eea` alone and every gating lever with zero.

## 2. An optimisation must preserve the unitary, on every input

Two classes are admissible, and they are admissible for different reasons.

**(a) Dead-work gating.** The skipped block contributes identity for *every* input, so the
unitary is unchanged and the circuit is better on any state, not only on the shots sampled.
The skipped work has to be shown to cancel (a `gf2_iadd` and its inverse, a compute/uncompute
pair) or to be controlled by a predicate that makes it a no-op (the Forney body at a non-root
position). All six levers are in this class; `--shift-under` is the falsifier that shows what
happens when the predicate is not the real one.

**(b) Measured uncomputation.** A compute/uncompute pair is replaced by `mr` on registers
that are *determined by the state* — in this decoder every register is in the computational
basis, so the measurement is deterministic, the classically controlled `X` restores the value
in place, and no phase is touched. The unitary is again unchanged. `clear_eea`, the Forney
predicate read, and the quotient gate's counter reads are in this class; `--clear-with r` is
the falsifier that separates them from `r`, which randomizes the tracked phase.

**(c) A predicate read, then the work gated on it.** A *predicate* is a bit the circuit
computed to decide whether work happens — `c0..c3`, `flag`, `do_shift`, the sweep's one-hot
selection, the product's `mul`. Reading it (`cx` into the shared ancilla plus `mr`, zero
Toffolis) and conditioning the work on the read bit makes the work execute only on the shots
whose predicate is one. This is class (b) applied to a predicate rather than to a value:
those bits are determined by the state in this decoder, so the read is deterministic and the
phase is untouched. It is what the six levers do per round and what
[`EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md) does per position — 532,500
Toffolis/shot, −43.60% of the arm, `Q` unchanged, and the same argument in both cases.

**Inadmissible: measuring an operand that is not determined by the state.** Reading `q_i`
and applying the resulting multiplication classically would leave the function correct on the
computational-basis inputs the benchmark supplies and wrong on any superposition. It is not a
better circuit, it is a different contract, and the number it produces is a statement about
the contract rather than about the circuit. The measured size of that exception is in §5 so
that nobody has to guess at it — and so that it cannot be taken by accident.

The distinction between (c) and this exception is what is read, not how much: a predicate is
read and the work stays a gate (on the read bit *and* its original control); an operand read
would replace the gate. The campaign's product gate is the boundary case worth naming — it
reads `mul`, the enable, and still emits `ixor_gf2_mul` on its own controls, so `q_i`,
`reg_a[j]` and the target are never read.

## 3. Every measurement ships with a falsifier

| Use of `mr` | Falsifier | What must fail |
|---|---|---|
| `clear_eea` (measured clear instead of the reverse EEA) | `--clear-with r` | phase match; `r` clears the value but randomizes the phase |
| `clear_eea` syndrome restore | same | syndromes not restored |
| Forney predicate read out of `sigma` | `--shift-under`, and the `r` arm | the gate would skip live positions |
| all five gate bits (`swap`, `sweep`, `forney`, `shift`, `quotient`) | `--clear-with r` | any lever whose read is `r` fails the phase, so the arm is only as good as its readout |
| per-position predicate reads (sweep selection, `do_shift`, `mul`) | stale-position condition, inverted condition, previous slot's mask | every one of them breaks correctness, and the honest gate is the one that saves *less* than its falsifiers because it skips exactly the predicate's zeros |

**Rule.** A lever whose falsifier still passes is not a lever — it is not measuring what its
argument says it measures. This is the same rule the probe followed when the shift gate's
`¬flag` was written as a second store of 1 and the gate fired on `c0 ∨ c1 ∨ c2 ∨ c3`: the
1.002× it produced was a defect, not a result, and the falsifier is what caught it.

## 4. The instance is part of the claim

Benchmark II is pinned: m=4095, n=70, GF(2¹²), 64 shots, seed 20261028. Scaled runs are
diagnostics and are labelled as such: the same six levers give 3.193× at m=15/GF(2⁴) and
3.709× for the five-lever stack at m=255/GF(2⁸). Neither is evidence about the pinned
instance and neither may be quoted as if it were. Note also that `Q` is instance-determined:
1,893 ids, of which 1,752 are the frozen `2 × 73 × 12` registers. A number that claims a
lower `Q` claims a different instance.

## 5. The measured ceiling, and what is left in it

The gate-level levers are exhausted, so the remaining question is whether the arithmetic can
be cheaper. It can be measured, and it has been.

**The multiply is Karatsuba, to the gate.** `ixor_gf2_mul` costs `K(m)` static Toffolis at
every degree measured — `K(2)=3`, `K(3)=7`, `K(4)=9`, `K(6)=21`, `K(8)=27`, `K(12)=63` — where
`K` is the recursion in `gen_gf_mul.h` (split at `k = ceil(m/2)`, three half-size products,
the middle one reusing the registers). There is no implementation slack to recover.

**No split of the recursion beats it.** At m=12: `k=6` gives 63, `k=8` ties at 63, every other
split is worse (67, 93, 105, 119), and schoolbook is 144. Toom's five-way split is unavailable
in characteristic 2. So "use a cheaper multiply" is not an implementation choice *inside this
recursion*; it needs a lower-AND bilinear algorithm, which is a research result — one that
§6 now supplies at the base cases (`mu(2) = 3`, `mu(3) = 6`), giving 54 ANDs at n = 12.

**The multiply is most of the remaining budget, and its share is measured, not ablated.**
The six-lever arm decomposes exactly as
`everything else + calls × (ands + 12)`: 10,510 executed multiply calls of 75 Toffolis
(63 ANDs + the 12-Toffoli controlled add) is **788,250.0 of 1,325,230.0 per shot — 59.48%**,
and the other 536,980.0 is the decoder. Two corrections to the earlier record are needed
here, and both were caught by measuring the substitute instead of deleting the original:

* The share is **59.48%, not 86.33%**. The earlier number came from deleting `ixor_gf2_mul`
  and differencing the executed count. That is the state-perturbation this project already hit
  once in a static count, now in an executed count: with the multiply gone the registers hold
  garbage, the predicates that gate the rest of the EEA read it, and far less *other* work
  runs — so the difference exceeded the multiply's own cost. Differencing two circuits that
  differ only in the multiply avoids it, because both are correct and both run on the same
  shots.
* A controlled multiply costs its AND count **plus 12**, in either implementation, so a
  cheaper multiply does not scale the whole 75.

**Every ratio below is against an unsourced target.** `scripts/measure_dqi_recycling_offsets.py`
reads the decoder's own degree registers after each EEA sub-step and finds the a-pair
constant at `n + 1` (not `n`) and the b-pair varying, so the recyclable saving is one
`logm`-bit register — 8 qubits at `n = 70`, a supported peak of **1,885**, not 1,789 — and
the 445,229-Toffoli row is a target with no derivation behind it, since recycling moves
registers and not gates. See `RECYCLING_MEASUREMENT` in
`apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py`. **And that one register does not work:**
the 1,885 build fails 64-shot equivalence and costs 11.3% more Toffolis, so the lever is
closed and ratios are quoted against 1,893 — [`DQI_RECYCLED_PASS_BODIES.md`](DQI_RECYCLED_PASS_BODIES.md).

With the decomposition, the arm priced for any multiply cost:

| If the multiply cost | `T_mean` per shot | Ratio | vs 445,229 |
|---|---|---|---|
| 63 ANDs (built-in) | 1,325,230.0 | 4.320× | 2.98× above |
| 54 ANDs (measured, §6) | 1,230,640.0 | 4.652× | 2.76× above |
| 50 ANDs | 1,188,600.0 | 4.817× | 2.67× above |
| 45 ANDs | 1,136,050.0 | 5.039× | 2.55× above |
| 23 ANDs (the `2n-1` dimension bound) | 904,830.0 | 6.327× | 2.03× above |
| 0 ANDs — the controlled add is unavoidable | 663,100.0 | 8.634× | 1.49× above |

No row is the inadmissible measured-operand multiply of §2; a multiply that reads an operand
is a different contract and is not priced here. The last row is the honest floor of this
decoder's arithmetic: even if the ANDs were free, the arm would execute 663,100 per shot.

**There is no static headroom either.** A peephole pass that records the builder calls in
emission order and removes immediately cancelling pairs — repeated `ccx`/`cswap`, an
`init_and` followed directly by its `del_and` — removes **0** Toffolis at the pinned
instance, for the reference and for both gated stacks (580,940 and 624,538 recorded calls
respectively), and 0 again at m=15/n=6/GF(2⁴) where the same pass records *every* builder
call, not just the T-like ones. The emitted stream is already balanced; there is nothing for
a local rewrite to collect.

**Quibbles are quantifiable.** A 1% change in the multiply's AND count is 0.86% of the
executed count; a 1% change in everything else is 0.14%. That is the resolution at which a
claim about this decoder is meaningful.

## 6. Which designer direction pays, measured

Three directions were priced at the pinned instance with
`scripts/study_circuit_designer_directions.py` (record:
`benchmarks/quantum_compilation/designer_directions.json`). The numbers choose between them.

| Direction | How it was measured | Result |
|---|---|---|
| **A. Predicate discovery** over the regions no lever touches | each region deleted, static Toffoli differenced (static counts are state-independent, so the difference is exactly that region's gates) | **2.71%** ceiling — 94,470 of 3,486,597 emitted Toffolis |
| **B. Bilinear search** for a cheaper multiply | exact search over rank-1 matrices spanning the product tensor, then Karatsuba composition, then the emitted circuit | **54 ANDs** against 63, **4.652× measured** against 4.320× |
| **C. A different decoder algorithm** | arithmetic on the measured decomposition | even a multiply with no ANDs misses 445,229 |

**A does not pay here.** The regions still running unconditionally are worth, at most,
`_update_flag` 32,856, `_initial_shift_loop` 30,912, `_update_metadata` 12,432,
`_update_halting_counter` 10,656, `_pre_align_reg_b` 7,392 and `_update_counter` 222 —
2.71% of the emitted stream, and executed counts are below static ones. A discoverer that
finds every predicate perfectly would still be under 3%. Its value is on the next decoder,
not this one.

**B pays, and it is now a circuit rather than a design.** The search is exact rather than
heuristic: a bilinear algorithm with `r` products is correct iff the `2n-1` anti-diagonals of
the product tensor lie in the span of the `r` rank-1 matrices `u_t v_t^T`, so at n=3 the
minimum was established by exhausting every choice of five (2.9M nodes, no solution) before
six was found and verified over all 64 input pairs. That gives `mu(2) = 3`, `mu(3) = 6`, and
by Karatsuba composition `12 = 3 x 3 x 6 = 54` ANDs against the built-in 63 — a better base
case, not a better recursion.

`scripts/gf54_mul.py` emits it as gates and
`scripts/measure_gf54_mul_kickmix.py` prices it (record:
`benchmarks/quantum_compilation/gf54_mul_kickmix.json`):

| | built-in | 54-AND |
|---|---|---|
| ANDs per call | 63 | **54** |
| Toffolis per controlled call | 75 | **66** |
| CNOTs per controlled call | 481 | 566 |
| workspace per call | 12 qubits | 19 qubits (31 controlled) |
| six-lever arm, `T_mean`/shot | 1,325,230.0 | **1,230,640.0** |
| ratio against the reference | 4.320× | **4.652×** |
| `Q` | 1,894 | 1,897 |

One controlled call was checked on 32 random basis states with random controls: values match
`field.mul`, operands are restored, the workspace returns to `|0>`, and the tracked phase is
untouched — and the whole decoder runs it with the six-lever verification still passing shot
for shot against the reference's function and phase.

**The same multiply as an opcode, and the inversion it then reaches.** The mixin above
replaces one instruction. The five products of the Itoh–Tsujii inversion and the quotient
sweep's are emitted by native generators, so `gen_gf54_mul` (in
`scripts/kickmix_mr_opcode.patch`, selected with `builder.gf_product = 'and54'`) reaches all
three products; `scripts/measure_gf54_native_opcode.py` prices it
(`gf54_native_opcode.json`):

| | built-in | 54-AND opcode |
|---|---|---|
| controlled multiply | 75 | **66** |
| plain multiply | 63 | **54** |
| inversion pair (5 products) | 315 | **270** |
| division | 378 | **324** |
| workspace per multiply | 12 | 42 |
| reference decoder, `T_mean`/shot | 5,725,071.6 | **5,201,222.5** |
| reference saving | — | **9.15%**, `Q` 1,893 → 1,912 |
| six-lever arm, `T_mean`/shot | 1,325,230.0 | **1,221,466.8** (4.687× vs the unchanged reference, 4.258× vs a reference with the same policy) |

The reference runs an inversion at every one of its 4,095 positions, so the block is 20,475
product sites of 9 Toffolis: 184,275 static. Priced in isolation — the same reference with
the policy flipped around `init_gf2_inverse_with_scaffold` and `gf2_phase_by_product` only —
it executes **20,469.5 products per shot across 4,093.9 positions and saves 184,225.1
Toffolis: 3.22% of the reference and 35.2% of its 523,849.1 saving**; the other 339,624.0
(64.8%) is the change-of-basis division and the quotient sweep. That single block is
therefore the largest contributor to this optimisation, and it is the one the gate-stream
emitter could not touch.

**The projected 4.928× does not reproduce, and the reason is the missing 12.** That
projection scaled the whole multiply by 54/63. A controlled multiply of either kind pays 12
Toffolis to add the product into its target under the control, and those 12 do not shrink
with the AND count, so the call scales by 66/75 and the arm by 7.14%, not 12.3%. The
measured 4.652× agrees with the decomposition to the last digit: 536,980.0 + 10,510 × 66 =
1,230,640.0.

**C alone does not pay, and this is the important line.** With the measured decomposition:

| multiply | `T_mean`/shot | ratio | vs 445,229 |
|---|---|---|---|
| 63 ANDs (built-in) | 1,325,230.0 | 4.320× | 2.98× above |
| 54 ANDs (B, measured) | 1,230,640.0 | 4.652× | 2.76× above |
| 45 ANDs | 1,136,050.0 | 5.039× | 2.55× above |
| 23 ANDs (the `2n-1` dimension bound) | 904,830.0 | 6.327× | 2.03× above |
| 0 ANDs | 663,100.0 | 8.634× | 1.49× above |

**No multiply, however cheap, delivers Benchmark II's target on this decoder** — even a
multiply with no ANDs at all leaves 663,100 per shot, 1.49× the target, because the 536,980
that is not the multiply and the 12-Toffoli add survive it. So the target needs B *and* a
change that shrinks the rest of the arm, and any claim that a better multiplier alone would
take the benchmark is refuted by this table.

## 7. Checklist for a claim

An optimisation claim is admissible when it carries all of:

1. the metric, named: executed or static;
2. the instance, the shot count and the seed;
3. the unitary argument: class (a) dead work, or class (b) measured uncomputation, with the
   cancellation or the determinism shown;
4. the falsifier that fails, and its output;
5. `Q` before and after (`1,893` → `1,894` here);
6. a record path and the script that produced it;
7. the static count as well as the executed one, since the two rank the levers differently.

## 8. What the current numbers are, and are not

The reachability of the remaining gap is worked out in
[`BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md): with the measured
three-term decomposition `T = 472,768.2 + 10,510 × (ands + 12) + 1,019.23 × ands`, the 23-AND
dimension bound lands at 1.941× the target, freeing every product at 1.345×, freeing the adds
too at 1.062×, and the Fourier/cyclotomic opcode class cannot move `T` at all because the
block it would transform is CNOT-only. (An earlier two-term form folded the 1,019.23
inversion/division products into its constant at 54 ANDs each and so reported 2.012×/1.186×
for the same totals; the totals agree, the labels did not.)

Admissible now: **4.687× fewer executed Toffolis than the unchanged reference at the pinned
instance (5,725,071.6 → 1,221,466.8 per shot) with the 54-AND multiply emitted as an opcode,
4.652× with it as a Python gate stream and 4.320× with the built-in one, `Q` 1,912 against
1,893, with both falsifiers failing and the six levers verified shot for shot on the
reference's own function and tracked phase.** The multiply policy is the only difference
between those arms: same gates, same reads, same shots.

A second, controlled figure belongs beside it: **4.258× against a reference given the same
opcode** (5,201,222.5 / 1,221,466.8). It is lower than the 4.320× the arm achieved with the
built-in multiply because the policy saves 9.15% on the reference but only 7.83% on the
gated arm — fewer of the arm's inversion products execute. Both ratios are correct; they
answer different questions, and
[`BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) §1.1 itemises all
five comparisons with their baselines.

The reference's own total moves too — 9.15% — and 35.2%
of that saving is the Forney inversion block, measured on its own rather than inferred from
the AND count.

Still not admissible, and not claimed: a receipt (the pinned `kickmix` has no `MR`, so the
`mr` clears this circuit uses cannot be run on the scored binary); a score (1.231M is still
2.76× the 445,229 Benchmark II needs); "best in the world" (there is no comparison at this
instance in the repository, and none can be made without building the published circuits at
the same m, n, field and shot count); or any extrapolation from that comparison to a
different instance or a different decoder.

**The per-position predicate reads move that further, and are admissible under §2(c).**
[`EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md) reads the sweep's one-hot selection,
the shift's `do_shift` mask and the product's `mul` control and gates the work on them:
**1,221,466.8 → 688,966.8 per shot at `Q` 1,913 (−43.60%, `Q²T` 4.4700e12 → 2.5213e12)**,
and 686,014.8 (−43.84%) when stacked with the flag/metadata gate that this rule's own
`mr`-read argument specified, so the arm is **1.541×** the target rather than 2.76×, with
`Q` and the 222 EEA rounds unchanged and three falsifiers failing. The executed count is now an adaptive-execution count and the
campaign note says so; the numbers above remain the ones from arms whose work is
unconditioned, which is why both sets are quoted separately.

What would move it further, in the order the measurements put them: nothing in the multiply
below 54 ANDs is in hand (the `2n-1` bound still leaves 2.03× the target); the 2.71% that a
perfect dead-work discoverer could find in §6 is smaller than what the predicate reads
already took; and the 299,953.2 non-product Toffolis that survive the campaign — the swap's
64,224, the shift's 96,584.6, `shift_step`'s 30,912, `update_flag`'s 30,240 — are the largest
identified items left, which is the one thing no rewrite of this decoder's gates can supply.

# Can an opcode close the rest of Benchmark II?

**Status:** measurement note · **Scope:** the DQI decoder at the pinned instance
(m=4095, n=70, GF(2^12), 64 shots) and Benchmark II's 445,229-Toffoli target ·
**Evidence:** `benchmarks/quantum_compilation/dqi_arm_submission_score.json` (the arm's
score as `verify()` assigns it, Experiment F),
`benchmarks/quantum_compilation/gf54_native_opcode.json`,
`benchmarks/quantum_compilation/kickmix_mr_arm_costs.json`,
`benchmarks/quantum_compilation/gf54_mul_kickmix.json`,
`benchmarks/quantum_compilation/gf_mul_and_static_headroom.json`,
`benchmarks/quantum_compilation/dqi_primitive_costs.json`,
`scripts/measure_gf54_native_opcode.py`; rules in
[`CIRCUIT_OPTIMISATION_RULES.md`](CIRCUIT_OPTIMISATION_RULES.md).

**What 445,229 is.** It is a derived target, not a measurement, a competitor's number or an
acceptance threshold. `apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py` records the target
row as `Q = 1,789`, `T_mean = 445,229` such that
`R(eps_plus) · 1,789² · 445,229 ≈ 1.6975e12`, which is `2.4439e13 / 1.6975e12 ≈ 14.397×`
better than the official reference score. The verifier itself has no Toffoli threshold: it
returns `PASS` on correctness (`eps_plus < 1/16`) and reports `R · Q² · T`. Every comparison
below is therefore against that derived aim, at the same instance and metric as the measured
circuit. The reference and arm figures quoted here are reconciled with the official row —
including why they differ by 0.78 standard errors and why no receipt is possible — in
[`DQI_RESOURCE_RECONCILIATION.md`](DQI_RESOURCE_RECONCILIATION.md).

Short answer: **no field-arithmetic opcode closes it, and that is now arithmetic rather
than opinion — but the control-dependent work around the arithmetic was worth far more than
this note originally said.** The 54-AND multiply as a builder policy takes the reference decoder's own
total down 9.15% and the six-lever arm to **1,221,466.8 executed Toffolis per shot at
`Q` = 1,913**, which is **2.744×** the target. With every field product *and* the
controlled add removed — neither is physically available — what is left is still
**1.062×** the target. The remaining gap is the decoder algorithm around the arithmetic,
not an instruction the arithmetic is missing.

The schedule campaign (§1.0b) then took the arm from 1,221,466.8 to **686,014.8 per shot —
1.541× the target, −43.84%** — by reading predicates the decoder already computes and gating
the work on them. That is a cost-model change, not a field-arithmetic one: it makes the
executed count equal the work the data path performs, and it does not reach 445,229 either.

## 1. Where the arm stands (all measured)

| | reference | six-lever arm | vs target |
|---|---|---|---|
| built-in multiply | 5,725,071.6 | 1,325,230.0 | 2.976× |
| 54-AND multiply as the Python gate stream | — | 1,230,640.0 | 2.764× |
| 54-AND multiply as the `gf_product` opcode | **5,201,222.5** | **1,221,466.8** | **2.744×** |
| `Q` | 1,893 → 1,912 | 1,894 → 1,913 | — |

Both records are at the same instance, shots and seed, and every arm in the table passes
the six-lever verification shot for shot against the reference's function and tracked
phase.

### 1.0 The stacked control-logic row

The two promoted control-logic gates now run in **one** circuit, which is the newest
measured row and the one the table above does not carry:

| | six-lever arm | once-stacked arm | vs target |
|---|---|---|---|
| executed Toffolis / shot | 1,221,466.8 | **1,218,514.8** | **2.737×** |
| `Q` | 1,913 | 1,913 | — |
| correctness, 64 shots | pass | pass | — |

This row is re-measured as `control_gates` inside the schedule campaign's record, so the
2,952 below is a number a single run produces rather than one carried between reports, and
it is then stacked with the schedule gates (§1.0b).

The saving is 2,952, not the 3,000 the two single-arm figures add to: the predicate reuse
saves its 8 Toffolis only in rounds where `_update_flag` executes, and the halt gate removes
that region from six dead rounds, so 48 Toffolis of the naive sum are not there to save. The
prediction was written before the combined arm was built and matched to the Toffoli. Record:
`benchmarks/quantum_compilation/dqi_combined_arm.json`;
[`EEA_AND_PREDICATE_PLAN.md`](EEA_AND_PREDICATE_PLAN.md) §2.4.

### 1.0b The schedule campaign: 1.541×

The shift/sweep/swap schedule campaign ([`EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md))
reads each movement and product predicate into a classical bit — every predicate register is
a basis state, so the read is deterministic and phase-free — and gates the work on it, so a
gate whose predicate is zero costs nothing. Declared before the arms were built, then
measured at the same instance, shots and seed:

| arm | executed T/shot | saving | vs target |
|---|---|---|---|
| six-lever control (the campaign's baseline) | 1,221,466.8 | — | 2.744× |
| + the sweep's one-hot quotient selection read | 1,117,498.8 | 103,968.0 | 2.510× |
| + the shift's `do_shift` mask read | 1,155,571.8 | 65,895.0 | 2.596× |
| + the product's `mul` control read | 858,829.8 | 362,637.0 | 1.929× |
| sweep + shift | 1,051,603.8 | 169,863.0 | 2.362× |
| **all three schedule gates** | **688,966.8** | **532,500.0** | **1.547×** |
| the once-stacked flag/metadata gate (§1.0), re-measured here | 1,218,514.8 | 2,952.0 | 2.737× |
| **all five in one arm** | **686,014.8** | **535,452.0** | **1.541×** |

`Q` = 1,913 and 222 EEA rounds in every arm, all arms correct, and the savings exactly
additive twice over: `103,968 + 65,895 + 362,637 = 532,500` within the schedule gates, and
`532,500 + 2,952 = 535,452` for the stack, so the two sessions' gates do not interact. Three
falsifiers, each a wrong condition, break the arm; two of them save *more* than the honest
gate, which is what a real gate looks like. The composed shift/swap arm and the arm that
drops the shift from every `c3` round both save exactly 0.0 and stay correct, which is the
measured form of the claim that the two regions cannot be fused for gain.

The saving is work removed, not relocated: the executed instruction stream falls with the
Toffoli count (37,808,019.7 → 32,451,795.1 per shot, −14.17%) while the added reads are
counted explicitly at 30,253 per shot (one `cx` and one `mr` each, 60,506 instructions,
1.1% of what was removed). What is *added* is measurement depth, not gate work. Record:
`benchmarks/quantum_compilation/dqi_schedule_campaign.json`.

### 1.1 The five comparisons, each with its baseline

There is more than one defensible ratio here, and quoting one as if it were another is a
mistake worth naming. All five are recorded in `gf54_native_opcode.json:comparisons`:

| comparison | numbers | value |
|---|---|---|
| decoder mean **T-count**, reference: built-in vs GF54 policy | 5,725,071.6 → 5,201,222.5 per shot | **−9.15%** |
| decoder mean **T-count**, arm: built-in vs GF54 policy | 1,325,230.0 → 1,221,466.8 per shot | **−7.83%** |
| **T-count ratio, arm vs the unchanged reference** | 5,725,071.6 / 1,221,466.8 | **4.687×** |
| **T-count ratio, arm vs a reference given the same policy** | 5,201,222.5 / 1,221,466.8 | **4.258×** |
| **arm's `Q²T`, Step 0 → final** | 4.754e12 → 4.470e12 | **−5.97%** |

The controlled comparison falls slightly — 4.320× → 4.258× — because the product policy
saves 9.15% on the reference but only 7.83% on the arm: the arm's Forney body is gated, so
fewer of its inversion products execute. That gating is a plausible cause and the
measurement supports it; it is not proved to be the only one.

The 4.687× is a **T-count** ratio against the fixed comparator, not a score improvement: the
score is `R(ε_+) · Q² · T`, and `R(ε_+)` can only be measured on the official sample. On
`Q²T` the same pair is **4.59×** (1,913² × 1,221,466.8 against 1,893² × 5,725,071.6), and on
`Q²T` against the same-policy reference (1,912 qubits) it is **4.25×**. A same-policy ratio
answers a different question from the headline one: how much of the gain is the six levers
rather than the new field arithmetic.

The 42 → 19 qubit workspace cut changes none of the `T_mean` column: the AND count is
what sets it, and the window was never an AND. It does move `Q` (1,935 → 1,912 for the
reference, 1,936 → 1,913 for the arm), which matters because the score charges `Q`
squared.

## 2. The arithmetic of the gap

The arm's executed Toffolis decompose into three measured terms, not two:

```
T = 472,768.2 + 10,510 × (ands + 12) + 1,019.23 × ands
```

| term | count | what it is |
|---|---|---|
| `472,768.2` | — | every Toffoli-like gate that is not a field product's AND and not a controlled add: the EEA's control logic, the predicates, the counter decode, the shifts, the metadata, the basis-change CNOTs' Toffoli-free remainder |
| `10,510 × (ands + 12)` | 10,510 | the `ixor_gf2_mul` calls, each `ands` ANDs plus the 12-Toffoli controlled add |
| `1,019.23 × ands` | 1,019.23 | the products inside `init_gf2_inverse_with_scaffold` (155.23) and `init_gf2_div_with_scaffold` (864) — 54-AND products with **no** controlled add |

At `ands = 54`: `472,768.2 + 693,660 + 55,038.4 = 1,221,466.8`, to 0.1 Toffoli.

**This replaces an earlier two-term model** (`527,806.8 + 10,510 × (ands + 12)`) that
reproduced the same total by folding those 1,019.23 uncontrolled products into the
constant at 54 ANDs each — `527,806.8 = 472,768.2 + 55,038.4`. The total was right; the
labels were not. In that model "free multiply" silently meant *free `ixor_gf2_mul` only*,
with the inversion's and division's products still paying 54 ANDs; the transparent number
for freeing every field product is **472,768.2**, not 527,806.8. The two-term figure is
kept here only so the older record can be read against this one.

| field arithmetic | `T_mean`/shot | ratio vs 445,229 | who could supply it |
|---|---|---|---|
| 54 ANDs everywhere, control-dependent work executed (measured, opcode) | 1,221,466.8 | 2.744× | — shipped |
| **54 ANDs everywhere, control-dependent work read and gated (measured)** | **686,014.8** | **1.541×** | [`EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md) — shipped in the local arm (688,966.8 with the three schedule gates, then 686,014.8 stacked with the flag/metadata gate) |
| 54 ANDs everywhere (measured, gate stream) | 1,230,640.0 | 2.764× | — shipped |
| 23 ANDs at every product (`2n-1` bound) | 864,060.6 | 1.941× | a better base case at every width; none is in hand |
| 0 ANDs, controlled add kept | 598,888.2 | 1.345× | impossible, and still 1.35× |
| 0 ANDs, no controlled add | 472,768.2 | 1.062× | impossible, and still 1.06× |

The last row is the point: **472,768.2 Toffolis per shot are spent on work that is not a
field product at all** — the EEA iterations, their predicates, the counter decode, the
shifts and the metadata. Even an opcode that made every field product and every controlled
add free would leave the benchmark 6.2% out of reach, and the field products themselves
(748,698.6) are 61% of the arm.

## 3. Every opcode class, and what it can move

| Opcode class | What it would remove | Best case | Measured? |
|---|---|---|---|
| **Field multiply** (`gf_product = 'and54'`) | 9 Toffolis per executed product | 2.744× now; 1.941× at the 23-AND bound | yes — this record |
| **Controlled add** (12 Toffolis per call) | 126,120 Toffolis per shot (10.3%) | 2.460× | at its floor: `m` CCX to add an `m`-bit value under a control |
| **Constant multiply / linear maps** (`gf2_imul`, syndrome matrix, change of basis) | CNOTs only | **0 T** | yes — `dqi_primitive_costs.json`: 0 Toffoli per call |
| **Fourier / cyclotomic transform** (additive FFT, Chien search as one evaluation) | CNOTs and depth in the Chien/Forney block | **0 T** | the block is CNOT-only; its only T is the inversion's five products, already 5 × 54 = 270 |
| **AND clearing / uncompute** (`del_and`, `mr`) | the Toffolis an uncompute would cost | already zero | yes — every AND target is measured back, no Toffoli |
| **Predicate discovery** over the regions no lever touches | at most 2.71% of the emitted stream | 1.642× static ceiling; executed less | yes — `designer_directions.json` |
| **A different decoder algorithm** | the 472,768.2 that is not a field product | the only class that can reach 445,229 | no — not an opcode |

Two rows deserve their own paragraphs, because they are the ones asked for next.

### 3.1 Fourier / cyclotomic transform: refuted before it is built

The natural candidate is the Chien search: 4,095 positions, each a polynomial evaluation
at a power of the primitive element, with 69 constant multiplications per position and
one inversion. A fast-transform opcode would replace that sweep with an additive FFT over
GF(2^12) or a cyclotomic-coset schedule of the same evaluations.

It cannot move `T_mean`, and the reason is a count, not an implementation detail:

- a **constant** multiplication by α^k is a GF(2)-linear map, emitted as CNOTs — the
  primitive-cost record measures **0 Toffoli** for `gf2_imul`/`gf2_idiv` at every call, and
  the whole position's measured slope is 351 Toffolis, of which 315 is the inversion;
- so the whole 69-per-position constant stream is already T-free, and re-deriving it as a
  transform can only trade a CNOT network for another CNOT network;
- the block's only Toffoli content is the inversion's five **variable** products, which
  the `gf_product` opcode already reduced from 315 to **270** and which are at the
  AND-count floor for that algorithm;
- the phase kickback that closes the inversion is classically conditioned CZs — Clifford,
  0 Toffoli.

So a `gf_transform` selector would be measurable only in CNOTs and qubits, where it can
go **backwards**: any transform needs an operand buffer, and the arm's qubit budget is
the cost the score squares. That is why none was added: a selector whose only reachable
effect on the metric is zero would be a new knob with no claim attached. If CNOT count or
depth becomes a target of its own, it is the right time to build it, and the function
checks in `measure_gf54_native_opcode.py` are the template for proving it correct first.

### 3.2 The controlled add is at its floor

`10,510 × 12 = 126,120` Toffolis per shot, 10.3% of the arm, is the controlled add into
the target. An `m`-bit XOR under one control is `m` CCX, and the AND-clearing machinery
cannot get under that: each CCX's target holds a function of the control and the added
bit, so it could in principle be cleared by measurement rather than uncomputed — and it
already is, because the added value is produced by the multiply, which is uncomputed with
`gen_gf_unmul`. The 12 is the irreducible price of the control.

## 4. What would have to change

For 445,229 to be reachable on this decoder, two independent things have to happen:

- **field arithmetic:** every product at the 23-AND dimension bound or below, which needs a
  new base case for wider inputs, not a new recursion (the Karatsuba recursion is exact at
  54, and no split beats it); at that bound the arm is 864,060.6, still **1.941×**;
- **the non-product stream:** by **5.8%** if every product and every controlled add were
  free (to 445,229), or by 93% if the arithmetic stayed where it is. Nothing in the field
  arithmetic can supply that: it is the EEA's iteration count, the predicates that gate it,
  and the change of basis.

After the schedule campaign the second bullet is no longer hypothetical and no longer
expressed against the static 472,768.2. The executed arm is **686,014.8** — 386,061.7 of
field products (of which 331,023 is the multiply window, the rest the inversion's and the
division's) and 299,953.2 of non-product work. The gap to the target is **240,785.8**
Toffolis, and the largest untouched items are the swap's 64,224, the shift's remaining
96,584.6 (30,912 of mask arithmetic plus 65,325 of movement its mask genuinely asks for),
`shift_step`'s 30,912 and `update_flag`'s 30,240. Those four are 221,960.6 of the 240,785.8:
the target is now predominantly three pieces of unconditional decoder work and the rest of
the multiply window, not an instruction the arithmetic is missing.

So the next attempt has to be a decoder algorithm — a shorter EEA, a different error
locator, or a construction that does not evaluate 4,095 positions per shot. The measured
plan for it, block by block, is
[`EEA_AND_PREDICATE_PLAN.md`](EEA_AND_PREDICATE_PLAN.md); it also carries the correctness
obligations and the falsifiers, because a deletion that changes the state is not a
measurement (the 86.33% multiply share was that mistake).

## 5. Closing statement: unreachable without `MR`, and not reached with it

Every row below is a circuit this repository has built and a score the competition's
own `verify()` has assigned. [`BENCHMARK2_STATUS_DECISION.md`](BENCHMARK2_STATUS_DECISION.md)
records what that table decides.

| build | best correct circuit | score at 102,400 shots | vs 1.6975e12 |
|---|---|---|---|
| **pinned** (what can be submitted) | reference + reduced-window 54-AND, 5,433,438.2 per shot at `Q` 1,893 | **2.319438e13** (Experiment E) | **13.66×** |
| local `MR` rebuild | six-lever arm, 1,221,408.0 per shot at `Q` 1,913 | **5.324728e12** (Experiment F) | **3.137×** |
| local `MR` rebuild | arm + the five schedule gates, 685,936.2 per shot at `Q` 1,913 | **2.990339e12** (Experiment G) | **1.762×** |
| — | derived Benchmark II target | 1.6975e12 | 1× |

Three conclusions follow, and they are independent:

1. **No circuit the pinned binary can compile comes near the target.** The best correct
   one is 13.66× out, and that is not a search gap: the multiply is 42.45% of the
   reference's static Toffoli stream, so deleting *all* of it — which no emitter can
   do — would leave the reference 8.29× out. Experiment E measures the only part of
   that share that is actually available: 9 Toffolis of 75 per controlled call, for
   5.0946%.
2. **`MR` is the only surface change that moves the order of magnitude, and it is not
   enough.** It takes the reference 4.142× (to 3.48× out) and the arm 4.5898× (to
   3.137×), and the campaign arm to 1.762×. The reason it cannot finish
   the job is §2's third term: 472,768.2 Toffolis per shot are not field products at
   all, and gating them is what the campaign already did.
3. **What is left is a decoder algorithm.** After the campaign the remaining 240,785.8
   Toffolis are the swap's 64,224, the shift's 96,584.6, `shift_step`'s 30,912,
   `update_flag`'s 30,240 and the rest of the multiply window. A shorter EEA, a
   different error locator, or a construction that does not evaluate 4,095 positions
   per shot is the only class of change that reaches 445,229.

Experiment G also closes the last unmeasured row in that table: the campaign's stacked
arm, through `verify()` at 102,400 shots, passes with **0 value and 0 phase errors**,
executes **685,936.2051367187** Toffolis per shot at `Q` 1,913 (its recorded 686,014.8
to 1.1e-4) and scores **2.990339e12** — **0.12236** of the pinned reference, and
**1.762×** the target once the `Q` term is included. Record:
`benchmarks/quantum_compilation/dqi_campaign_arm_submission_score.json`.

### 5.1 The external-optimizer check

The one comparison that could be made against a general-purpose tool was made, on the
part of the circuit a Clifford+T optimizer can express: the multiply's forward AND
netlist, recorded from the emitter itself and given to **PyZX 0.10.7**
(`scripts/compare_gf_mul_external_optimizer.py`, record
`benchmarks/quantum_compilation/gf_mul_external_optimizer.json`).

| | measured |
|---|---|
| our netlist | **54** ANDs, 1,004 CNOTs, 54 measured clears at **0** Toffoli |
| our controlled call | **66** Toffoli-like gates (54 AND + the 12-Toffoli add) |
| PyZX 0.10.7, Clifford pass | T **378 → 378**, CNOT 1,328 → 824 |
| PyZX 0.10.7, `full_optimize` | **did not return within 90 s** at 99 qubits and 378 T |
| unitary accounting | forward + adjoint = **108** ANDs = **756** T before optimization |

So a general-purpose optimizer does not beat the AND count: the 54 Toffolis are the
function, not slack in a Clifford skeleton, and the only thing it removes is Clifford
work. Our clears are 0 Toffoli because they are measurements with classically
conditioned fixups — an option a unitary optimizer does not have, which is why its
honest figure is 108 ANDs rather than 54. The decoder itself cannot be offered to it at
all: its stream carries `HMR` and `INFO_TAG`, and the arm's own clears are `mr`, neither
of which a Clifford+T optimizer models. The 7-T decomposition the comparison uses was
verified on all eight basis states before any number was taken (max error 3.1e-16).

So the closing position is: **Benchmark II is unreachable on the pinned competition
binary, unreachable on the `MR` rebuild as the decoder stands, and reachable — if at
all — only by replacing the decoder algorithm**, whose measured plan and correctness
obligations are in [`EEA_AND_PREDICATE_PLAN.md`](EEA_AND_PREDICATE_PLAN.md).

What would overturn this, stated so it can be checked rather than argued: a circuit
this repository can build and `verify()` scores below 1.6975e12; or a measured
`T_mean` below 445,229 at `Q` ≤ 1,789 on any build; or a decoder that removes the
non-product 472,768.2 without reading qubits into classical bits (which is what
[`MR_PATCH_REQUEST.md`](MR_PATCH_REQUEST.md) already asks for, and no more than
that). Until one of those exists, the numbers above are the frontier, and quotes of
"14.4×" are a target, not a result.

## 6. Pins

`tests/test_dyn_mr_gated_arms_kickmix.py` pins the numbers this note rests on:

- unit costs 63/54, 75/66, 315/270, 378/324 and the 19-qubit workspace;
- the reference's 5,725,071.6 → 5,201,222.5 at `Q` 1,893 → 1,912, and the arm's
  1,325,230.0 → 1,221,466.8 at `Q` 1,913;
- the Forney split (184,225.1 of 523,849.1 per shot, 35.2%) sums to the whole saving;
- the 9-Toffolis-per-executed-product identity on both arms;
- the reconciliation of the official, pinned and local reference rows
  (`dqi_resource_reconciliation.json`): the 102,400-shot integrality of the official figure,
  identical per-shot arrays across the two binaries, and the pinned build's exact failure to
  construct the arm;
- the schedule campaign (`dqi_schedule_campaign.json`), pinned by
  `tests/test_dqi_schedule_campaign.py`: the four declared predictions, the control's exact
  reproduction of 1,221,466.8, the three schedule savings 103,968.0 / 65,895.0 / 362,637.0,
  their exact additivity at 532,500.0 → 688,966.8, the prior combined arm re-measured
  in-record (2,952.0 → 1,218,514.8), the five-gate stack at 535,452.0 → 686,014.8, the
  instruction stream falling 5,356,224.6 with the Toffolis and the 30,253 reads counted,
  both zero-saving arms, and the three falsifiers being wrong and informative.

A future claim that an opcode closes the remaining 2.744× has to move one of those pins,
and §2 says which term it would have to be.

## 7. Two closures this note can now make

### 7.1 445,229 is permanently unsourced

The target has no circuit behind it, and optimising the circuits this repository has will
not produce one. The measured ladder is **686,014.8** (the schedule campaign, 1.541×),
**1,221,466.8** (the six-lever arm, 2.744×) and **472,768.2** (§2's last row: every field
product *and* every controlled add free — impossible — and still 1.062×). So 445,229 is
labelled **permanently unsourced**: a derived aim kept so comparisons have a denominator,
never a count any build has produced or is projected to produce.
`design_projection_dqi_gluzdov_v6()` records it as
`"mean_toffolis": "UNSOURCED: ..."` and scores the projection at the sourced peak rather
than against the 1,789 row it was derived with.

### 7.2 The 96-qubit gap is arithmetically closed

`Q = 1,885` because exactly one degree register is recyclable. The decoder allocates
**five** (`n_A`, `n_B`, `n_a`, `n_b`, `n_q`), each `log2(n)` wide — 8 qubits at `n = 70` —
and one is gone. **The most any further recycling could remove is the other four: 32
qubits**, which lands at **1,853**. The target needs **96**.

**96 > 32**, so no register-recycling variant of this decoder reaches 1,789 — not because a
derivation is missing, but because the registers are not there to remove. The four that
remain are load-bearing: `n_A`/`n_B` are the degrees the EEA compares, `n_b` cannot be
derived because the b-pair varies, and `n_q` addresses the quotient register. Getting below
1,853 means allocating fewer qubits, which is a different circuit rather than a schedule.

### 7.3 The floor as a function of how many registers are materialised

The two measured endpoints pin a one-parameter model. The reference materialises all five
registers at 1,893 qubits; removing `n_a` gives 1,885. Both steps are the same 8 qubits, so

```
peak = 1,853 + 8k        k = degree registers materialised
```

| k | registers kept | peak | gap to 1,789 |
|---|---|---|---|
| 5 | `n_A n_B n_a n_b n_q` (measured) | 1,893 | 104 |
| 4 | `n_A n_B n_b n_q` (measured-supported) | 1,885 | 96 |
| 3 | any three | 1,877 | 88 |
| 2 | any two | 1,869 | 80 |
| 1 | any one | 1,861 | 72 |
| 0 | none at all | **1,853** | **64** |

`model_degree_register_floor()` computes this table and `dqi_design_projection.json` carries
it under `degree_register_floor`. The row that decides the question is the last one: **a
decoder that materialises no degree register at all still lands at 1,853**, 64 qubits above
the target. So 1,789 is not reachable by allocating fewer registers — that lever is exhausted
64 qubits short, and what remains is data-path width, the `A`/`B`/`a`/`b` polynomial arrays,
which recycling does not touch and which is a different resource entirely.

### 7.4 No measured arm is at the floor, and what one would score there

Nothing has been built at 1,853 qubits: the narrowest DQI circuit the verifier has scored is
**1,893** (the pinned reference) and both `MR` arms sit at **1,913**. So a floor-width
circuit cannot be *certified* from any record in this repository — the floor is a model, not
a build. What can be done is arithmetic on the measured scores, holding each arm's executed
Toffoli count and its measured `R` fixed and scaling by `(1853/Q)²`:

| arm | Q | measured score | at 1,853 | vs 1.8846e12 target |
|---|---|---|---|---|
| pinned reference (reduced-window 54-AND) | 1893 | 2.319e+13 | 2.222e+13 | 11.793× |
| six-lever MR arm | 1913 | 5.325e+12 | 4.996e+12 | 2.651× |
| campaign arm (five schedule gates) | 1913 | 2.99e+12 | 2.806e+12 | 1.489× |

`model_score_at_qubit_floor()` reproduces this table. The width lever is worth about **6%**
of the score per arm — and it is not enough: at 1,853 the campaign arm is still
**1.489×** the target. So the last 64 qubits are worth
taking only after the Toffoli count moves, and they cannot be taken by register recycling at
all (§7.3). What would overturn this: a built circuit at `Q ≤ 1,853` that `verify()` scores
— which would also need the narrower allocation to leave the instruction stream unchanged,
the assumption the rescale makes and no build demonstrates.

### 7.5 The last 64 qubits have no measured lever

At the register floor the allocation contributes nothing more (§7.3), so the 64 qubits between
1,853 and the target have to come from somewhere else. Only one other width lever in this
repository has a measured saving — the workspace window cut from 42 to 19 qubits (§1.1) — and
the two together still do not get there:

| lever | qubits | basis |
|---|---|---|
| degree-register recycling | 32 | §7.3: four remaining registers × 8 |
| workspace window 42 → 19 | 23 | §1.1, measured on the 1,935-qubit build |
| **sum** | **55** | |
| needed (1,853 → 1,789) | **64** | |

**55 < 64**, and 55 is an *upper* bound: the window saving was measured on a 1,935-qubit
build, so any build already at 1,853 has spent part of it. Nine qubits remain unaccounted for
even at both levers' limits, which means the rest is data-path width — and the data path does
not move in small steps. One polynomial array is `(n + 1) × m = 71 × 12 = 852` qubits, larger
than every lever combined; dropping or narrowing one is worth more than all of them and is a
different circuit, not a schedule. `model_data_path_floor()` computes the table.

### 7.6 What holding one polynomial array would trade

The data path holds a remainder pair and a coefficient pair. Dropping to one array is the only
change larger than every lever in §7.5, and two of its three lines are settled while the third is
not.

- **Qubits.** One array is `(n + 1) × m = 71 × 12 = 852` bits, so from the register floor the peak
  would be **1,001** — 788 under the target. `Q` stops binding.
- **Toffolis — bracketed, because two measured numbers disagree.** The EEA reads both remainders
  every round, so the dropped one must be re-derived from the syndrome each round: one extra
  division. `dqi_primitive_costs.json` prices a divide scaffold pair at **378** Toffolis; §2's
  decomposition prices the products inside one division call at **864 × 54 = 46,656**. They are
  different quantities and 124× apart. Over 222 rounds that is **83,916** to **10,357,632** extra
  per shot: 5,725,108.4 → **5,809,024.4 (1.015×)** to **16,082,740.4 (2.81×)**.
- **Score.** `R · Q² · T` lands between **0.284×** and **0.786×** the reference. The `Q²` win
  (0.28) beats even the pessimistic `T` loss, so **the trade is favourable under both prices** and
  the conclusion does not depend on settling the discrepancy — which is why it is a bracket and
  not a chosen number.

What this arithmetic cannot supply is the sentence it rests on: *re-deriving the dropped remainder
costs one division and nothing else.* That is a claim about a circuit, it is what
[`DQI_SINGLE_ARRAY_DECODER_CONTRACT.md`](DQI_SINGLE_ARRAY_DECODER_CONTRACT.md) requires be proven
before any builder work, and `model_single_polynomial_array()` returns the bracket with that
assumption named in the returned dict.

### 7.7 The window figure cannot be re-measured at 1,853 yet

The 23-qubit workspace saving in §7.5 was measured on a build at 1,935 qubits, and making it
contemporaneous needs a build near 1,853. None exists: `RECYCLING_APPLIED_IN_CIRCUIT` is
`False`, the compiled circuit is reference-equivalent, and the reference decoder exposes no
window knob — so the window can only be measured on the arm builds the builder already emits,
all of which are at or above 1,912. The re-measurement is therefore blocked on the recycled
pass bodies, not on effort, and §7.5's 55 stays an upper bound until one exists.


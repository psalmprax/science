# Benchmark III: what a sub-128-qubit angle circuit actually needs

**Status:** scope, 2026-11-02 · **Supersedes the "format is the blocker" reading** implied by
[`docs/invsqrt_frontier_and_portfolio_synthesis.md`](invsqrt_frontier_and_portfolio_synthesis.md) §4 ·
**Records:** `signed_4bank_plan.json`, `design_projection.json`, `artifacts/quantum_compilation/dfthcsos/arm2/rm_block2_j4/design_projection.json`

The format was never the blocker. `verifiers/dfthcsos/example_submission/angle_k_reference_circuit.py`
writes `angle_k_*.kmb` with stock `kickmix`, and the verifier recovers words by simulating
the submitted circuits, so any correct circuit in the pinned gate set is admissible. Two
things are actually in the way, and they are different in kind:

| blocker | kind | closable here? |
|---|---|---|
| the 128-qubit cap on each `ANGLE_k` circuit | design | **yes** — arithmetic below |
| the 128,128,348-byte FCIDUMP the verifier requires | data | **no** — see §0 |

## 0. No III receipt is obtainable in this checkout

`verify.py ../sample_submission/dfthcsos` fails before scoring:

```json
{"status": "FAIL", "error": "FCIDUMP not found:
 .../verifiers/dfthcsos/FeMoCo76o113e.fcidump"}
```

`scoreinfra/reference.py` pins that file at `FCIDUMP_BYTES = 128128348`,
`FCIDUMP_SHA256 = e5a48af746fd075026dec9e8f5df2d363802a121e8c53af9a0b984bde695e1bd`, and
`check_fcidump` has no override — "there is no option to use another file". The bundle
ships only `FeMoCo76o113e.fixed_shifts.npz` (the symmetry shifts, a different pin). So
the honest status of the baseline score 3.355995588e9 is *recorded earlier elsewhere and
not re-verifiable here*: the 128 MB integral file has to be mounted before any III number,
baseline or candidate, can be produced on this host.

That is a delivery step, and it is cheap: mount the competition data, mount
`submission.npz`, and the whole scoring path runs. It is listed first because every
option in §2 is judged by a number that cannot be measured until it is done.

## 1. The cap, and where the qubits go

The cap is 128 on `max_k ANGLE_k.num_qubits`. Selection width is fixed by the address
count — 3,976 rows needs exactly 12 bits, and the register contract rejects extra
selection bits so workspace cannot be hidden there. Word registers are `J × brot`,
because a word register's width *is* how the circuit states its precision. What is left
to move is the delta payload and scratch.

| geometry | selection | words | delta | scratch | peak | vs 128 |
|---|---|---|---|---|---|---|
| signed 4-bank, J=3, B=4, 9-bit delta | 12 | 30 | 81 | 5 | **128** | 0, no margin |
| the same design as audited | — | — | — | — | **132** | **−4** |
| 8-bit delta | 12 | 30 | 72 | 5 | 119 | +9 |
| 9-bit delta, serialised QROM | 12 | 30 | 81 | 2 | 125 | +3 |
| `rm_block2_j4` (J=4, 12-bit) | 12 | 48 | 48 | 11 | 119 | +9 |
| J=2 | 12 | 20 | 36 | 5 | 73 | +55 |
| half-address (K doubles) | 11 | 30 | 81 | 5 | 127 | +1 |

**The two records disagree about the same design**: `signed_4bank_plan.json` computes
128 with `headroom: 0`, and the cross-repository audit records 132 with headroom −4. Both
cannot be right, and the difference is exactly the kind of thing a projection hides —
allocation, not arithmetic. It must be settled by measuring, not by choosing the
friendlier number:

```python
# after: python angle_k_reference_circuit.py OUT --format kmb
for k in range(K):
    print(k, km.Circuit.load(f"OUT/angle_k_{k}.kmb").num_qubits)
```

Until that runs, "128" is a plan and "132" is an observation, and the plan is the one
that needs to prove itself.

## 2. The levers, each with what it costs

Ordered by what they give up, cheapest first:

- **Serialise the QROM scratch (5 → 2).** Peak 125, three spare. Costs latency inside
  the circuit, which III does not score: `T` counts Toffolis, and serialising is a
  re-use of the same workspace, not a new gate. This is the cheapest real qubit and it
  should be tried first.
- **Drop the delta payload to 8 bits.** Peak 119. Costs one bit of angle precision, which
  raises `eps` and *lowers* `lambda_eff`, which raises the score. Not quantifiable here:
  `lambda_eff` is computed from the recovered words against the reference integrals, so it
  needs the FCIDUMP of §0. Do not adopt this without measuring the `lambda_eff` it costs.
- **`rm_block2_j4` (J=4, 12-bit words, 48-bit delta buffer).** Peak 119, 19 circuits,
  projected `T_total` 52,726–53,315 against a baseline non-angle cost of 13,757, and a
  projected 1.87–1.89× gain. Its own record says `ANALYTICAL_PROJECTION_ONLY`,
  `compiled_kmb: false`, `official_verifier_run: false`, and that the per-batch QROM
  cost is approximated as `high_address_rows − 1`. It is the most promising option and
  the least evidenced.
- **Sharing the tree's ancillas is impossible.** Measured, not argued: a single ancilla reused
  at every level has the parent's control and the child's target on the same qubit, and
  kickmix refuses the circuit (`qubit collision`, `CCX q122 q9 q122` × 7,943). The
  `n-1 = 11` ancillas are structural. Removing them needs multi-controlled gates, which trade
  the qubits back for Toffolis at the leaf loads.
- **`J = 11` via precision instead of ancillas.** `brot = 9, J = 11` is **122 qubits**,
  `K = 7`, **55,636** Toffolis over both passes (12.5% below `brot = 10, J = 10`), and the
  checker recovers all 298,200 words exactly; the cost is a step of 1.227e-2 rad and a max
  error of 6.136e-3. Whether that bit is affordable is a `lambda_eff` question and needs the
  FCIDUMP.
- **Early-break is already exhausted.** Every one of the 31,808 delta rows (8 circuits ×
  3,976 addresses) is distinct, so a tree over delta tables has no repeated subtree to share;
  the 3,974 Toffolis per circuit is the padding-trimmed tree, not a data saving.
- **J=2 or a half-address split.** Both buy qubits with circuits (25 → 38, or ×2). Since
  `Q_total` carries `+ J · brot`, J=2 also moves the score's qubit term by 10, so the
  trade is not obviously bad — but `T_ANGLES` doubles its per-circuit structure, and this
  needs the same compiled count as everything else.

## 3. The 4,363 Toffolis, and why it is a separate problem

`T_literal = 25 × 1,129 × 2 + 13,157 = 69,607` against a claimed `65,244`. The two
numbers come from different derivations of the same architecture, and the gap is not a
bug to be found — it is the cost of asserting a total without compiling it. The plan's
own promotion gate already says the right thing: `must_use_per_circuit_Toffolis_from_Kickmix_simulator`.

Note the size of the downside if the phase fixup is not free: at 2,432 fixup Toffolis the
total is 72,039, cost 2.345214040e9, gain **1.431×** instead of 1.580×. The gain is
sensitive to a component that has never been counted, which is the strongest argument for
compiling before choosing any §2 lever — a lever that fixes the cap and worsens `T` can
land the portfolio below 1.0, and a leg below 1.0 *lowers* the geometric mean.

## 3a. The reverse pass cannot be replaced by measurement

It is tempting to halve the family's cost by clearing the word registers by measurement
instead of running the second tree. The contract rules it out, and the citation is exact.

`scoreinfra/circuits.py` applies the family in reverse and, after **every** step, asserts that
the words equal the *previous* batch, not zero:

```python
for k in reversed(range(len(circuits))):
    sim.do(circuits[k])
    _check_state(sim, circuits[k], layout, address_bits, f"reverse ANGLE_{k}")
    actual = read_words(f"reverse ANGLE_{k}")
    if k > 0:
        if not np.array_equal(actual, batches[k - 1]):
            raise InvalidCircuits(
                f"reverse ANGLE_{k} did not restore the ANGLE_{k - 1} word registers")
    elif np.any(actual):
        raise InvalidCircuits("the reverse traversal did not clear the word registers")
```

A measurement-based clear produces zeros. That satisfies the last branch only, and fails the
`k > 0` branch at the very first step, whatever the measurement basis. The checker wants the
*inverse*, so the second pass is part of the specification rather than an implementation
choice — a submission cannot opt out of it by clearing instead.

The cost side is the same story: `recover_words_from_circuits` ends with

```python
toffoli = sim.operation_statistics().num_toffoli_like / (num_vectors * repeats)
```

where the simulator has executed the forward and reverse passes, so both are charged. The
factor of 2 is structural.

The pinned gate set would block it independently: the only qubit→bit writer is `HMR`, which
measures in **X**, and `R` leaves a random phase (measured at 37/64 shots on `x(q); reset(q)`
in [`docs/MR_INSTRUCTION_SPEC.md`](MR_INSTRUCTION_SPEC.md) §1). So even a contract that asked
for a clear rather than a restore could not be satisfied without the `MR` opcode that
Benchmark II is blocked on.

## 3b. What the 9-bit table would have to cost in `lambda_eff`

`lambda_eff` cannot be computed in this checkout, and not for want of effort:
`scoreinfra.dfthc_math.load_inputs` does `ref = load_reference(check_fcidump())`
unconditionally, so every scoring path needs the packaged integrals, and the submission
`npz` carries `U`, `W`, `H2_I`, `H1_I`, `Hsym_lower` but no one- or two-electron
integrals. The symmetry shifts are not a stand-in: `load_inputs` replaces the submitted
`H1_I`/`Hsym_lower` with the baseline's own before `S` and `beta_1` are derived, so they
carry no information about the candidate.

What *is* computable is the trade, as a break-even. The score is
`lambda_eff * T_total * Q_total`, and both `T` and `Q` are measured here:

| family | `T` (2 passes) | vs 10-bit |
|---|---|---|
| `brot=10, J=10` | 63584 | 1.0000x |
| `brot=9, J=11` | 55636 | 0.8750x |

So the 9-bit family is the cheaper submission while
`lambda_eff(9) / lambda_eff(10) >= 0.8750`: it may lose up to **12.5%** of
`lambda_eff` and still win. Anything larger and the precision bit costs more than the
circuit it saves.

Measured angular penalty at each precision, on the real table:

| `brot` | max angle error | mean | rms | max rotation perturbation `||dR||_F` |
|---|---|---|---|---|
| 12 | 0.000e+00 rad | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 10 | 3.068e-03 rad | 1.535e-03 | 1.878e-03 | 4.339e-03 |
| 9 | 6.136e-03 rad | 3.072e-03 | 3.600e-03 | 8.677e-03 |

The perturbation is `2*sqrt(2)*sin(d/2)`, the Frobenius distance between the intended and
the applied Givens rotation. Whether a 12.5% `lambda_eff` budget survives that is exactly
the question the FCIDUMP answers and nothing else does — but it is now a bounded question
with a number on the boundary rather than an open preference.

## 3c. Where the freed precision bit is worth spending

Freeing qubits by dropping a bit of precision only helps if they are then spent on the batch.
Measured on the real table, with every family verified word-exact:

| `brot` | `J` | qubits | spare | `K` | slots | padding | `T` (2 passes) | max angle error |
|---|---|---|---|---|---|---|---|---|
| 10 | 10 | 123 | 5 | 8 | 80 | 5 | 63584 | 3.068e-03 rad |
| 9 | 10 | 113 | 15 | 8 | 80 | 5 | 63584 | 6.136e-03 rad |
| 9 | 11 | 122 | 6 | 7 | 77 | 2 | 55636 | 6.136e-03 rad |

**`brot = 9, J = 10` is strictly dominated.** It costs exactly the same `T` as the 10-bit
family (63584 Toffolis) because `K = ceil(75/10) = 8` in both, and it pays a coarser angle
table for the privilege: max error 6.136e-03 rad against 3.068e-03. The 10 qubits it
frees (123 → 113) buy no ancilla that the tree can use — the tree needs
`n-1 = 11` and no more, and extra ancillas do not remove nodes.

The bit only pays at `J = 11`, where it removes a whole circuit: `K = 7`, 55636 Toffolis,
12.5% below the 10-bit family, with two fewer padding slots as well
(2 against 5). So the rule on this instance is simple: spend
precision on the batch or not at all, and check the `lambda_eff` break-even of §3b (0.875)
before spending it.

### 3a-i. The two passes cannot share a tree

The premise is worth stating precisely, because it does not hold: between the forward and
reverse application of one circuit **nothing differs**. Circuit `k` is a single file; the
checker applies it forward, later applies *the same bytes* in reverse, and both applications
are the same instructions, not two payloads to be fused. The delta payload differs between
*different* circuits, which is why `K` is the lever, not something the two passes could share.

Measured, on the family of §3c:

- per circuit 3974 Toffoli-like gates; 8 circuits = 31792
- the checker reports 63584 = 2 x 31792, and the simulator's own
  `num_toffoli_like` after forward+reverse is 63584

So the second application is charged as a second execution, and it is the checker that performs
it: the family length is validated against `ceil(75/J)` before a shot is simulated, and a family
one circuit short is rejected rather than completed by the reverse sweep:

```text
75 angles in batches of J=10 need 8 circuits, got 7
```

There is therefore no tree to share. The only levers on the factor of 2 are the ones already
measured: fewer circuits (larger `J`, §3c) and cheaper circuits (the tree itself).

## 4. Order of work

1. Mount `FeMoCo76o113e.fcidump` and `submission.npz`; re-score the shipped baseline so
   there is a measured `3.356e9` in this checkout rather than a remembered one.
2. Compile the 25 `ANGLE_k` circuits of the current signed 4-bank design and print
   `num_qubits` and `toffoli_count` per circuit. This settles 128-vs-132 and the 4,363
   in one run, and it is the only step that converts §1 and §3 into facts.
3. Pick one lever against the measured headroom — serialised scratch first, since it
   costs neither precision nor circuits.
4. Re-run the official verifier; promotion needs `status PASS`, 0 value and 0 phase
   errors, and a score strictly below the pinned baseline.

Nothing in steps 1–3 needs new mathematics, and step 2 is the same compile the plan
already gates on. What is missing is the data file and one afternoon of compiling, not an
idea.

## 5. A cheaper tree was attempted, and there is not one

`3,974` Toffolis per circuit is one per node of a padding-trimmed tree over 3,976 addresses.
The tree amortises each of those Toffolis over roughly twenty `cx` loads, which is why it is
hard to beat rather than because it is already known optimal.

The chunked, dirty-ancilla alternative was built and measured
([`scripts/prototype_dfthcsos_chunked_qrom.py`](../scripts/prototype_dfthcsos_chunked_qrom.py)),
on a 64-address slice of the real table so every variant is a real circuit:

| variant | qubits | Toffolis | vs tree |
|---|---|---|---|
| tree (shipped) | 51 | 62 | 1.00x |
| chunked: per-address match + `ccx` load | 49 | 1,099 | **17.73x** |
| chunked: fused multi-controlled load | 51 | 3,093 | **49.89x** |

Counted on the full table the chunked rule is **68,404** (17.2x): 496 block-tree Toffolis,
3,976 per-address matches, and 63,932 `ccx` payload loads. The structure loses for a reason
that does not depend on the tuning: a chunked read spends one Toffoli per address and another
per set payload bit, while the tree spends one per node and pays for the payload with `cx`.

All four chunked variants also **failed the checker's word recovery** at 64 addresses — wrong
words for every block size and both load forms — so they do not survive the phase and
workspace rules at any price, not merely at a bad one.

This withdraws the earlier 13x figure for a sublinear-`T` QROM. That number was extrapolated
from an `O(sqrt(A))` cost model rather than measured.

### 5.1 The `O(sqrt(A))` ladder, built and measured

That withdrawal was itself too pessimistic. The ladder the cost model describes *is*
buildable here, and it does what the model says -- to the addressing only.
[`scripts/prototype_dfthcsos_dirty_ladder.py`](../scripts/prototype_dfthcsos_dirty_ladder.py)
builds it: `sqrt(A)` offset controls are computed once and kept alive, then a tree over the
block index loads each block's rows against those already-computed controls. Nothing is
reused, which is the point -- and which costs `2(sqrt(A) - 1)` dirty ancillas, not
`sqrt(A) - 1`, because every leaf must co-exist while the loads run. Verified with the same
word recovery as the shipped family.

Addressing alone, which is where the model's claim lives and the only part that does not
depend on `J`:

| `low` | ladder | ancillas | block tree | addressing | vs the tree's 3,974 |
|---|---|---|---|---|---|
| 0 (the tree) | 0 | 0 | 3,974 | **3,974** | 1.00x |
| 2 | 4 | 6 | 992 | 996 | 3.99x |
| 4 | 28 | 30 | 247 | 275 | 14.5x |
| **5** | **60** | **62** | **123** | **183** | **21.7x** |
| 6 | 124 | 126 | 61 | 185 | 21.5x |

So the model is half right. Addressing falls from 3,974 Toffolis to **183**, a 21.7x cut.
What the ladder cannot do is keep the payload cheap: a load with two controls is a `ccx`,
which the checker counts as Toffoli-like, so every set payload bit becomes a Toffoli. The
tree's whole advantage is that it amortises one Toffoli over every set payload bit of the
addresses it covers.

Which one wins is therefore a question about payload density, and the payload is per
circuit, so `J` sets it. Spread over the shipped family's `J = 10`, the payload is 159,068
set bits per circuit (40.0 per address):

| basis | payload per circuit | addressing (best in-cap `low`) | total | vs tree | qubits |
|---|---|---|---|---|---|
| `J = 10` (shipped) | 159,068 | 996 at `low = 2` | 160,064 | **40.3x** | 127 |
| `J = 4` | 63,932 | 183 at `low = 5` | 64,115 | 16.1x | 120 |

The earlier 16.1x quoted here was the `J = 4` figure; on the shipped family it is 40.3x, and
`low` drops to 2 because 100 payload qubits leave no room for a wide ladder. Both are far
past any break-even, so the conclusion does not move, but the number did.

### 5.2 How sparse a table would have to be, measured

`scripts/prototype_dfthcsos_dirty_ladder.py --sweep` builds tables at a chosen density on
all 3,976 addresses, costs the tree and the ladder as real circuits, and checks every one
with the competition's word recovery. `brot = 10`, `J = 4`, `low = 5`:

| set bits/address | payload | tree `T` | ladder `T` | vs tree | recovers |
|---|---|---|---|---|---|
| 0 (empty) | 0 | 4 | 183 | 45.8x | yes |
| 0.10 | 398 | 1,405 | 581 | **0.41x** | yes |
| 0.25 | 994 | 2,337 | 1,177 | **0.50x** | yes |
| 0.50 | 1,988 | 3,160 | 2,171 | **0.69x** | yes |
| 0.75 | 2,982 | 3,537 | 3,165 | **0.89x** | yes |
| 0.90 | 3,578 | 3,672 | 3,761 | 1.02x | yes |
| 0.95 | 3,777 | 3,710 | 3,960 | 1.07x | yes |
| 1.00 | 3,976 | 3,744 | 4,159 | 1.11x | yes |

Two corrections come out of this table, and both were in the previous version of this
section wrong:

1. **The ladder wins below about 0.87 set bits per address, not 0.95.** The fixed-payload
   break-even (`3,974 - 183 = 3,791` set bits, 0.953 per address) holds the tree at its
   full-table cost. It cannot: the tree's early-break collapses all-zero subtrees, so the
   tree cheapens as the payload thins (3,974 at full fan-out, 3,744 at one bit per address,
   1,405 at 0.10). The ladder chases a target that retreats.
2. **One set bit per address is already too many.** That is the sparsest non-empty table
   there is, and the ladder costs 1.11x the tree on it. A table has to leave at least a
   seventh of its addresses carrying no set bit at all before the ladder pays. Nothing in the
   Benchmark III format forbids such a table; the angle table is not one, at 40 set bits per
   address on the shipped family.

The reads are correct at every density, so this is a cost result and not a correctness one.

Those ladder counts are three Toffolis lower than the previous version of this table, and the
reason is a correction rather than a tuning: the builder was traversing the whole power of two
(`128` blocks at `low = 5`) where the table only has `ceil(3976/32) = 125`, and the rule counts
the blocks that exist. Padding was costing `H - 2 = 123` against a traversed `126`. The builder
now trims, so the rule and the circuit agree on non-power-of-two block counts -- checked at
`rows = 100`, where `low = 2` gives `H = 25`, `23` block Toffolis and `98 / 48 / 27 / 23 / 33`
addressing Toffolis at `low = 0..4`, each equal to the measured circuit.

### 5.3 Where the payload sits matters more than how much of it there is

Every number above holds the payload's *placement* fixed at random. The tree's early-break does
not see density at all: it fires when an aligned block is uniform, so placement is the lever.
Same density, same payload count, three layouts (`--sweep --placement`), `J = 4`, `low = 5`:

| density | payload | tree `T` random | tree `T` blocked (8 runs) | tree `T` contiguous | ladder `T` (any layout) |
|---|---|---|---|---|---|
| 0.10 | 398 | 1,405 | 103 | 15 | 581 |
| 0.50 | 1,988 | 3,160 | 122 | 15 | 2,171 |
| 0.75 | 2,982 | 3,537 | 127 | 15 | 3,165 |
| 1.00 | 3,976 | 3,744 | 131 | 14 | 4,159 |

**There is no crossing under either structured layout.** The ladder is layout-invariant -- its
`ccx` count is the payload count, wherever the payload sits -- while the tree collapses whole
blocks of identical cells to a single `load`. Clustered payload, which is what a table with
symmetry-zero regions has, makes the tree cheaper by 24x to 266x and the ladder's case
correspondingly worse.

That result is real but it overstates the structure a real table has, because these synthetic
tables are built from `0`/`1` cells, so a block can be *entirely* uniform. The shipped angle
table has no uniform block at any size:

| aligned block | uniform blocks |
|---|---|
| 16 addresses | 0 / 248 |
| 62 addresses | 0 / 64 |
| 124 addresses | 0 / 32 |
| 256 addresses | 0 / 15 |

So early-break never fires on the real table -- which is why its tree is exactly `rows - 2`,
one node per address. Random placement, not contiguous, is therefore the layout a real table
resembles, and the two numbers that bracket the truth are:

- **0.91 set bits/address** with maximum collapse (random placement, ladder using the tree's
  own early-break; 0.87 without it, §5.5): the best case for the ladder, since the tree
  cheapens to as little as 1,405.
- **0.953 set bits/address** with no collapse at all: the fixed-payload break-even of §5.1,
  which is what the real table approaches, since it has no uniform block at any size.

The real crossing depends on the split the 128-qubit cap allows: `[0.91, 0.95]` at
`J = 4, low = 5`, and `[0.56, 0.75]` on the shipped `J = 10, low = 2` family (§5.6). At 40.0 set
bits per address the table is 53x above even the pessimistic end, so the bracket is the honest
form of the claim and nothing downstream depends on where in it the answer lands.

### 5.4 The real table's density, and why the FCIDUMP cannot change it

The delta table is the **angle** table, so its density is set by the rotation words in the
submission, not by the electronic-structure integrals. An FCIDUMP enters scoring through
`lambda_eff`; it never touches these words. Measured from the shipped `submission.npz`
(`--density`), 3,976 addresses:

| `brot` | set bits/address at `J = 4` | at `J = 10` | bits per word | words that are zero |
|---|---|---|---|---|
| 1 | 0.00 | 0.00 | 0.00 | 78.6% |
| 4 | 4.55 | 11.46 | 1.15 | 14.4% |
| 6 | 8.05 | 20.24 | 2.02 | 4.6% |
| 8 | 12.03 | 30.03 | 3.00 | 1.3% |
| 9 | 14.04 | 35.04 | 3.50 | 0.7% |
| **10** | **16.08** | **40.01** | **4.00** | **0.4%** |
| 12 | 20.06 | 49.92 | 4.99 | 0.1% |

The density is linear in both levers and is essentially `J * brot / 2`: a quantised word uses
about half its bits. Two consequences:

1. **No precision reaches the crossing.** At `J = 10` the crossing needs `0.087` bits per word,
   i.e. 91.3% of the `39,760` cells exactly zero. The table has 0.4% zero words at `brot = 10`
   and 14.4% even at 4 bits. The only row that reads as sparse is `brot = 1`, and it is an
   artefact: at one bit every column is constant across addresses, so every delta is zero and
   the table carries no payload at all.
2. **`J` is the wrong lever to pull.** Dropping to `J = 4` only halves the density (40.0 to
   16.1), which is still 18x above the crossing, and it costs two extra circuits (`K = ceil(75/4)
   = 19` against 8) -- more Toffolis than the ladder could ever save.

So a ladder-family submission would need a table of near-zero angle deltas, and the Benchmark
III format -- fixed reference angles, 3,976 addresses of 75 Givens rotations -- does not offer
one.

### 5.5 With the tree's own early-break the crossing is 0.91, and it is driven by empty addresses

The sweeps above measured the ladder without the rule the shipped tree uses to collapse constant
blocks, so they were not a like-for-like comparison. `build_ladder(..., early_break=True)` now
applies that rule at both levels -- a uniform offset range loads once against the range's own
ladder control, a uniform block range needs one load and no descendants, and a table whose blocks
are all uniform skips the shared ladder entirely. `J = 4`, `low = 5`, 3,976 addresses:

| density | tree | ladder, no break | ladder, early-break |
|---|---|---|---|
| 0.10 | 1,400 | 581 | 576 |
| 0.25 | 2,315 | 1,177 | 1,155 |
| 0.50 | 3,104 | 2,171 | 2,113 |
| 0.75 | 3,447 | 3,165 | 3,067 |
| 0.90 | 3,567 | 3,761 | 3,643 |
| 1.00 | 3,637 | 4,159 | 4,032 |

The crossing is **0.87 with the break (0.84 without)**, and one bit per address loses by 11%
(4,032 against 3,637). Those tree counts are lower than the ones this section first carried
because the *tree* also got a better break -- see §5.5a -- so the comparison finally uses the
same collapsing rule on both reads. Structured layouts still admit no crossing, but they are
now genuinely close: at 1.00 the tree is 131 with the payload in eight runs and 14 with it in
one run, against a ladder of 280 (2.14x) and 20 (1.43x). Two corrections got there, and both
were mine:

1. An earlier version of this section quoted 4,032 for both. That was the *random* table's
   ladder cost -- the sweep cached one ladder per density, sound only while the ladder is
   layout-invariant, and it stops being so once it collapses. Keyed by placement: 296 and 74.
2. Even those carried a fixed tax the tree does not have: the offset ladder was built in full
   (`2(S - 1)` gates) whenever a single block was mixed. It is now built only where a load
   actually addresses it -- a uniform range still gets a control, because a load lands on it,
   but its subtree is not descended into. On a table whose payload sits in runs that is a
   handful of nodes: 74 -> 20 for the contiguous table.

What remains is structural rather than incidental: the tree's break skips a whole address range
at one node, while the ladder spends a node per level of the path to the boundary between runs.

Two controls say what the collapse is really driven by:

- **Distinct bit positions remove the rows a break could collapse**, so the tree stays near full
  fan-out (measured before the break port: 1,405 / 2,337 / 3,160 / 3,537 / 3,672 / 3,710 / 3,744)
  while the ladder is unchanged at 581 / ... / 4,159. Collapse needs equal *rows*, and a table
  whose set bits sit in different positions has none.
- **A populated table does not rescue the ladder either.** Filling `d` bits into every address
  instead of spreading `d x rows` cells over all cells (`--placement per-address`) used to give
  trees of 407 / 1,002 / 1,995 / 2,988 / 3,581 / 3,974 and a ladder 8% cheaper at one bit per
  address. With the tree breaking on row equality that tree is **3,453** at one bit per address
  and the ladder's 3,638 is **5% dearer**. The 8% window was an artefact of a tree that could not
  collapse repeated rows; it is withdrawn.

The real table is denser than any of these runs: at 40.0 bits per address the payload term is 40x
larger (159,068 `ccx` loads), which is the 40.3x of §5.1. With both reads collapsing on the same
rule, the ladder's only winning case is payload sparse enough to leave **empty addresses** --
below 0.87 -- and that is the same condition as before, now with no populated-table escape.

### 5.5a The tree's own break was weaker, and the port changes what counts as a win

The tree compared every cell of a block to the scalar `delta[lo, 0]`, so it only collapsed blocks
that were one repeated 0/1 pattern. It now compares each row to `delta[lo]`, which is a strict
generalisation. Measured (`--break-audit`):

| table | collapsible ranges | family Toffolis |
|---|---|---|
| shipped `submission.npz`, all 8 delta tables | **0, 0, 0, 0, 0, 0, 0, 0** | 3,974 x 8, unchanged |
| synthetic, 1 bit on every address, `J = 10` | 198 | **3,770** (204 saved) |

The port saves exactly nothing on the real table, because no range of any size there has equal
rows -- which is also why that tree is `rows - 2`. On synthetic tables it collapses a lot, and
that is why every tree count in §5.5, §5.6 and §5.7 is lower than the first version of those
sections: the comparison is now between two reads that both collapse, not one that does.

### 5.6 The window closes at `J = 10`, which is the family that ships

Every sweep above is `J = 4, low = 5`, where the ladder fits the cap with a wide split. The
shipped family is `J = 10`, and 100 payload qubits leave room for `low = 2` only -- so the
addressing the ladder can afford is 996 Toffolis, not 183. Re-run with `--sweep-low 2`:

| density | tree (random) | ladder (random) | tree (per-address) | ladder (per-address, `low = 2`) |
|---|---|---|---|---|
| 0.10 | 1,406 | 1,105 | 386 | 489 |
| 0.50 | 3,080 | 2,952 | 1,894 | 2,393 |
| 0.75 | 3,486 | 3,486 (`low = 0`) | 2,829 | 3,576 |
| 1.00 | 3,691 | 3,691 (`low = 0`) | 3,770 | 4,765 |

Two results, and the second one withdraws a claim:

1. **A crossing at `low = 2` sits well below the `J = 4` figure**, because the in-cap split is
5.4x more expensive (996 against 183). Before the tree's break was ported that read 0.56; it is
quoted here only as the `low = 2` measurement it was, because the split scan below shows that
`low = 2` is not the split to choose and the crossing at `low = 0` is not a crossing at all.

2. **There is no per-address window to survive.** At `J = 10` the ladder is `1.26x` the tree at
one bit on every address (4,765 against 3,770). At `J = 4, low = 5` it is `1.05x` (3,638 against
3,453). The 8% quoted in the first version of §5.5 was measured against a tree that could not
collapse repeated rows; with the ported break it disappears at both splits.

So the per-address result of §5.5 is a statement about a configuration the cap cannot combine
with `J = 10`. The conclusion does not move -- the ladder loses on this table by 40.3x either
way -- but the winning regime is narrower than §5.5 alone reads.

The structured layouts were measured the same way (`--low-scan --placement blocked|contiguous`):
the blocked table has a tree of 122 Toffolis and the contiguous one 15, and `low = 0` reproduces
each exactly (1.00x). Every wider split loses -- 246 (2.02x) and 30 (2.00x) at `low = 1`, 316
(2.59x) and 31 (2.07x) at `low = 2`, 533 (4.37x) and 54 (3.60x) at `low = 5` -- and
`--low-matrix`, which now covers all four layouts, agrees at every density: 89–122 and 4–15,
with `low = 0` best each time. The flat comparator table (`low = 12`, `S > rows`) is dominated on
both axes -- 6,980 and 7,548 Toffolis at 7,094 and 7,662 qubits against 3,691 and 3,770 at 123 --
so it is never worth asking the cap for.

At `J = 4`, where `low = 5` is in cap, the same scan still returns `low = 0` for all three
structured layouts at all eight densities (per-address 355–3,453, blocked 103–131, contiguous
14–15), and moves only the random window, out to 0.75 (0.41–0.89x). The wide split pays where the
payload leaves empty addresses, not where it has structure.

**And `low = 2` is not even the best in-cap split.** `--low-scan` costs every split with its
qubit count, which is the only way to know which splits the cap actually admits. At `J = 10`
they are `low = 0, 1, 2`; from `low = 3` the family needs 134 to 625 qubits:

| `low` | qubits | in cap | ladder `T` (0.75 random) | ladder `T` (1 bit per address) |
|---|---|---|---|---|
| 0 (the tree) | 123 | yes | **3,486** | **3,770** |
| 1 | 124 | yes | 4,862 | 5,557 |
| 2 | 127 | yes | 3,945 | 4,765 |
| 3 | 134 | no | 3,456 | 4,279 |
| 5 | 180 | no | 3,132 | 3,955 |
| 12 (flat table, `S > rows`) | 7,094 / 7,662 | no | 6,980 | 7,548 |

So the 0.56 crossing is a statement about `low = 2` only, and `low = 2` is not the split to
choose: **no in-cap split beats the tree at `J = 10`**, and the best in-cap read is `low = 0`,
which *is* the tree -- the ladder's advantage first appears at `low = 3` (134 qubits) and is
comfortable at `low = 5` (180 qubits). The cap is what forbids it, not the table.

### 5.7 The four regimes, in one table

Everything above, stated once. `T` compares the read against the shipped tree's 3,974 Toffolis
per circuit; "in cap" means the read fits 128 qubits.

| regime | what the delta table looks like | `J = 4` (`low = 5`, in cap) | `J = 10` (shipped; cap admits `low <= 2`) |
|---|---|---|---|
| **1. Dense, real** | 40.0 set bits per address, no uniform block at any size | 64,115 = **16.1x**, loses | 160,064 = **40.3x**, loses |
| **2. Sparse, random** | up to ~0.87 set bits per address | wins below **0.87** | no in-cap split wins (`low = 0` is the tree) |
| **3. Sparse and fully populated** | 1 bit on *every* address | 3,638 vs 3,453 = **1.05x**, loses | 4,765 vs 3,770 = **1.26x**, loses |
| **4. Structured** | payload in 8 runs / 1 run | 280 vs 131 (**2.14x**) / 20 vs 14 (**1.43x**), loses | 122 vs 122 and 15 vs 15 at `low = 0` -- the tree; every in-cap split loses (2.0x to 4.4x) |

The ladder wins in exactly one of the four: sparse payload that leaves **empty addresses**
(below 0.87), at a split wide enough to pay for itself (`low >= 5`), which needs a payload small
enough to leave 60 ancillas free -- `J = 4` at `brot = 10`. The shipped family is `J = 10`: 100 payload qubits,
`low <= 2`, 40 set bits on every address. It is in regime 1, and regime 1 is 40.3x.

The submission decision, with the reproduction commands, is consolidated in
[`docs/BENCHMARK3_LADDER_DECISION.md`](BENCHMARK3_LADDER_DECISION.md).

The 13x figure stays withdrawn. What replaces it is the measured decomposition:the `O(sqrt(A))` claim holds for the addressing (21.7x) and fails for the total (40.3x on the
shipped family), and no single density threshold transfers, because the tree's cost tracks
the same sparsity that would make the ladder cheap.

### 5.8 The FCIDUMP is downloadable, and every precision we tried fails it

This repository's vendored bundle has no `FeMoCo76o113e.fcidump`, but the competition's own data
does: `verifiers/dfthcsos/FeMoCo76o113e.fcidump`, 128,128,348 bytes. Downloading it makes the
chemistry half scoreable for the first time (`kaggle competitions download -c
google-verifiable-quantum-circuit-compilation -f verifiers/dfthcsos/FeMoCo76o113e.fcidump`), and
the first thing it scores is a rejection:

| family | `K` | `Q` | `T` (two passes) | `error_1` | `error_2` | verdict |
|---|---|---|---|---|---|---|
| sample, 12-bit words (reference) | 15 | 730 | 93,237 | **3.19 mHa** | **18.64 mHa** | PASS, `lambda_eff` 49.301, cost 3.356e9 |
| `brot = 10, J = 10` (this family) | 8 | 123 | 63,584 | **242.3 mHa** | 59.4 mHa | FAIL |
| `brot = 9, J = 11` | 7 | 122 | 55,636 | 338.3 mHa | 89.6 mHa | FAIL |
| `brot = 8, J = 13` | 6 | 127 | 47,688 | 568.2 mHa | 159.2 mHa | FAIL |

The tolerance is 20 mHa. Rounding the sample's 12-bit words down to 10 bits costs 242 mHa, twelve
times the budget, so every `T` figure above is the cost of an **invalid** submission. Word
recovery passes on all three because it checks that the circuit loads the table it was given -- not
that the table approximates the Hamiltonian. That distinction is the whole of §5: the family is a
circuits-half prototype, and it was never chemically admissible.

The public 8-bit submission reaches 18.6 mHa at `brot = 8` because its words are **refit** at that
precision (its table is `(9, 143, 40)`, 1,363 addresses, 8-bit), not rounded into it. A sub-128
family needs that refit before any of its Toffoli counts mean anything; re-running our builder on
the public words at `J = 13` gives 16,332 Toffolis against the 16,135 their verifier reports,
which is where the read stops and the table starts.

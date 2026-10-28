# Photonic Accelerator Boundary

**Status:** `audit-only` (a boundary statement, not a component)
**Scope:** what an external photonic accelerator — Q.ANT-class native processing
unit, or a photonic quantum machine — may contribute to the invsqrt / DQI
pipeline in this repository, and what only the exact verifier may decide.

This document exists because "can we use the photonic chip for the quantum
circuit" has a short answer, and the short answer is worth writing down before
anyone spends time on an integration: **no, and the reason is measured, not
stylistic.**

---

## 1. Two different technologies, both of which fail here

| | What it is | Why it cannot run the circuit |
|---|---|---|
| Photonic **analog AI accelerator** (Q.ANT NPU: thin-film lithium niobate, optical matrix/nonlinear transforms, digital-electronic I/O) | A classical analogue co-processor for matrix and nonlinear workloads | No qubits, no coherence, no measurement-based uncomputation. Cannot execute or certify a reversible Toffoli network. Precision is ~16-bit analog (≈1e-5 relative). |
| Photonic **quantum computer** (dual-rail fusion, squeezers, detectors) | A quantum machine with a different cost model | Would require a full re-implementation of the decoder plus its own loss and fusion accounting, and its resource units are not the competition's. A photonic *layout* does not inherit the score of a kickmix artifact. |

Nothing in this repository earns its score by wall-clock time, so neither
technology can improve the number by being faster.

## 2. The score contains no timing term

| Benchmark | Score | Consequence |
|---|---|---|
| invsqrt (Benchmark I) | `Q × toffoli_equivalent` = 89 × 1770.798661 = **157,601.08** | Depends only on qubit count and gate count. |
| DQI (Benchmark II) | `R(ε₊) · Q² · T_mean` = **24.439 T** (reference) | Depends only on qubit count, gate count, failure probability. |
| DFTHC-SOS (Benchmark III) | `λ_eff · T_total · Q_total` = 49.301369 × 93,237 × 730 = **3.3556 G** | `λ_eff` is derived from the DF-HC *mathematical* metrics (`error_1`, `Λ`, `E_SOS`), not from hardware. |

A co-processor that made the whole pipeline run a thousand times faster would
change the score by exactly zero. `apedp_dyn/quantum/dqi_photonic_resource.py`
already establishes the same invariance for code distance L = 8, 16, 24.

### 2.1 The non-submission artifact

`python3 scripts/report_photonic_non_submission.py` writes
`artifacts/quantum_compilation/photonic_non_submission.json`: the logical
footprint of each champion, its sFFCC resource bill, its wall-clock time, and an
explicit `submission_status: "NON_SUBMISSION"` with the reasons. The point of
writing it down is that the resource bill is large and the score change is zero.

| Benchmark | Q | T | score | wall clock/shot | QD emitters | fusions/shot |
|---|---|---|---|---|---|---|
| I - invsqrt | 89 | 1,770.8 | 157,601.1 | 4.69 ms | 136,704 | 1.16e10 |
| II - DQI | 1,893 | 5,725,108.4 | `R·Q²·T` | 15.17 s | 2,907,648 | 1.36e12 |
| III - DFTHC-SOS | 730 | 93,237 | 3.3556 G | 247 ms | 1,121,280 | 5.03e10 |

(sFFCC at L = 16, 2.6496 µs per logical cycle, fusion success `η² = 0.8649`,
retries modelled as `1/η²`. Emitters are 1,536 per logical qubit.)

Every row is priced on the *same* logical circuit, so `score_delta` is 0.0 by
construction rather than by measurement. The artifact is a cost statement, not a
result: the verifiers take a compiled `.kmb`/`.kmx` artifact, and a resource
estimate is not one. That is the whole reason it is labelled NON-SUBMISSION.

## 3. There is nothing here worth accelerating, measured

`python3 scripts/test_shared_pddd_circuit.py` prints this table; one run on the
development machine, whole champion rebuild included:

| Stage | Seconds | Share |
|---|---|---|
| exact domain compile | 2.866 | 64.5% |
| build raw circuit | 1.374 | 30.9% |
| official verification, 25,600 shots | 0.178 | 4.0% |
| rotation calibration + write `.kmb` | 0.022 | 0.5% |
| load tables + PDDD search results | 0.003 | 0.1% |
| **total** | **4.443** | 100% |

Verification runs at ~143,555 shots/second, so the pinned 200,540,928-shot
certification is roughly 23 minutes of simulation. That is a throughput problem
in a quantum simulator, not a linear-algebra problem an analog chip can take
over. A photonic accelerator's host-to-device conversion and transfer alone
would dominate a 4.4-second pipeline.

## 4. The precision wall

The champion's unused margin is `min_margin_units = 73,326,605` out of a 2^64
phase circle:

```
73,326,605 / 2^64 = 3.97e-12 turns
```

Its own calibrated rotation epsilon is `2.84e-13`. An analog photonic datapath
resolves roughly `1e-5` relative — about seven orders of magnitude coarser than
the gap between PASS (`worst_error_over_tolerance = 0.9996227`) and FAIL. Any
analog stage placed anywhere in the phase path flips the circuit to FAIL. This
is not a tuning problem; it is the wrong number of bits.

## 5. The Boolean-sharing ceiling, measured

The natural place to suspect remaining headroom is the shared-PDDD restoration
(coarse6 + curve4..6), because it is the only place where a search proposes a
structure rather than a number.

Measured, at the joint order `(7,4,3,2,1,0,6,5)` / polarity 32:

| Output set | Nodes |
|---|---|
| coarse6 alone (per-output optimum) | 11 |
| curve4..6 (per-output optimum) | 28 |
| **disjoint total (what the incumbent used to cost)** | **39** |
| joint four-output plan | **36** |

Node count is monotone in the output set: every one of the 15 non-empty subsets
of `{coarse6, curve4, curve5, curve6}` costs no more than any superset of it
(four singles 9–17, six pairs 16–25, four triples 25–31, the quadruple 36). So
adding outputs to the search can never reduce the count.

`scripts/search_claims_pddd.c` enumerates **all 8! × 256 = 10,320,960**
(order, polarity) pairs for this output set and reports 36 as the minimum. That
is a complete search over the Davio/butterfly family; a wider search over the
same family cannot beat it.

Exchange rates, computed from the pinned receipt:

| Quantity | Worth |
|---|---|
| 1 shared node = 2 Toffolis | 178.0 score units = 0.1129% |
| 1 logical qubit | 1,770.799 score units = **9.95 nodes** |
| the achieved 39 → 36 win | 534 units = 0.3388% |
| a hypothetical free (0-node) restoration | 6,942 units = **4.4048%** ceiling |

So one additional qubit costs as much as saving ten nodes — 9.95× the entire
3-node win this optimization produced. That is why the schedules in
`scripts/test_viable_optimization_path.py` all lose to the incumbent: the
monolithic joint restore costs +22 qubits (+38,291.57), and holding even one
shared node live across `outer` costs +1,590.80.

### 5.1 Mixed polarity, tested

`scripts/search_mixed_polarity_pddd.py` generalises the recursion so each AND
node picks its own control literal (`x_v` or `NOT x_v`) instead of forcing the
positive literal, and searches orders and polarity patterns. It reproduces the
incumbent 36-node plan before searching, and validates every plan it scores on
256/256 wire assignments. Result: the generalisation **does** reduce the node
count — 26 nodes for `curve4..6` against 28, and 35 for the joint four-output
set against 36 — and it does not produce a better circuit.

| Plan | Nodes | Static Toffolis after compilation | Verified score |
|---|---|---|---|
| Incumbent curves (positive Davio) | 28 | 1466 | 157,605.52 at 25,600 shots |
| Mixed-polarity curves | 26 | 1466 | 157,597.45 at 25,600 shots |
| Same two, 204,800 shots | 26 / 28 | 1466 / 1466 | candidate −0.56 units |

Two reasons, both measured by `scripts/build_mixed_polarity_candidate.py`:

- The exact-domain compiler's linear pass removes **two more** Toffolis from the
  28-node plan than from the 26-node one, so both compile to 1466 static
  Toffolis. The node count was not the binding constraint; the compiler already
  had that slack and spent it.
- The apparent score gain shrinks 14× (8.07 → 0.56 units) when the sample grows
  8× (25,600 → 204,800 shots). That is sampling noise in the shot-mean Toffoli
  count, not a per-shot saving.

The 35-node joint plan needs a single monolithic restore (all four outputs at
once), which costs +22 qubits and +38,291.57 units, so it is not schedulable at
all. `scripts/test_dyn_invsqrt_champion_rebuild.py` pins the circuit hash so none
of this can drift silently.

### 5.2 The same exchange rate on the DQI decoder

DQI scores `C = R(ε₊) · Q² · T_mean`, so the marginal prices are
`∂C/∂Q = 2RQT` and `∂C/∂T = RQ²`:

| Quantity | Reference (Q=1893, T=5,725,108) | V6 target (Q=1789, T=445,229) |
|---|---|---|
| 1 qubit | 6,048.7 Toffolis | **497.7 Toffolis** |
| 1 Toffoli | 4,268,821 units | 3,812,654 units |

Break-even rule: an option that saves `dQ` qubits may add up to `dQ · 2T/Q`
Toffolis. At the V6 target that is 498 / 1,493 / 5,973 Toffolis for 1 / 3 / 12
qubits (0.11% / 0.34% / 1.34% of `T_mean`). Applied to the open options:

| Option | ΔQ | ΔT_mean/shot | Cost change | Pays for itself? |
|---|---|---|---|---|
| Halting-counter recycling, counter only (measured up the ladder, §5.3) | 0 | −240 (−0.72% toy → −0.0118% full) | **−0.012%** | no — the saving is a constant 240 Toffolis while the circuit grows |
| Halting-counter recycling as shipped (flag + counter) | 0 | +480 at full scale | **+0.024%** | no — it is a regression at every size |
| Itoh–Tsujii 48q scaffold, pooled architecture | 0 | +3,161 (+0.71%) | +0.710% | no — `Q` is bounded by the persistent registers |
| Itoh–Tsujii 48q scaffold, unpooled architecture | +12 | +3,161 (+0.71%) | **−0.637%** | only when the scaffold is not pooled |
| Dual-scaffold 48q division | +12 | +606,974 (+136%) | +133.2% | no, by two orders of magnitude |
| Mixed-polarity PDDD | 0 | −4 | −0.001% | immaterial |
| `n_iters` 222 → 216 | 0 | re-verified worse | −4.11% (worse) | no |

`scripts/test_dual_scaffold_dqi.py` prices the dual-scaffold option against the
*reference* baseline, where the same +606,974 Toffolis is +10.6% of `T_mean` and
the gate returns HALT. At the V6 target scale the identical overhead is +136%,
so the HALT is not close — the script's baseline choice understates it.

### 5.3 The halting-counter saving does not survive the ladder

`scripts/run_halt_recycling_scale.py` builds each variant and reads its Toffoli
count, which is an exact property of the emitted circuit and needs no shots:

| Instance | Reference | Counter only | Flag only | Both (as shipped) |
|---|---|---|---|---|
| m=15, n=6, GF(2⁴) | 33,258 | 33,018 (−240, −0.722%) | 33,618 (+360) | 33,378 (+120) |
| m=63, n=10, GF(2⁶) | 68,051 | 67,811 (−240, −0.353%) | 68,411 (+360) | 68,171 (+120) |
| m=255, n=14, GF(2⁸) | 131,576 | 131,336 (−240, −0.182%) | 132,056 (+480) | 131,816 (+240) |
| m=1023, n=30, GF(2¹⁰) | 478,774 | 478,534 (−240, −0.050%) | 479,374 (+600) | 479,134 (+360) |
| **m=4095, n=70, GF(2¹²)** | **2,033,345** | **2,033,105 (−240, −0.0118%)** | 2,034,065 (+720) | 2,033,825 (+480) |

The −0.72% was real but it is a toy-instance number for the counter *alone*: the
replacement removes exactly one fixed-width comparison per call site, so the
saving is a constant 240 Toffolis at every instance while the reference grows to
2,033,345. At the benchmark instance the option is worth −0.012%, which is 40×
below the 0.498% a single qubit is worth. Recycling the flag on top of it *costs*
Toffolis at every size measured (+360 to +720), so the shipped prototype is a
regression. The toy score does move the right way and verifies
(`scripts/run_halt_recycling_scale.py`: reference 2,374,832,368.039 → counter
2,357,638,161.127, both PASS); the percentage is what does not scale.

**Untested axis, disclosed:** decompositions outside the Davio/butterfly family
that are neither per-node polarity nor a variable-order change — for example a
non-uniform order inside one shared plan, or a non-uniform split tree — are not
covered. The 4.4048% ceiling above applies to them too.

### 5.4 The rotation overhead is irreducible, not untried

The 44 priced rotations in the 89Q champion cost 315.299429 of the 1770.798661
Toffoli-equivalent — 17.81% of the score, and by far the largest single term
left. `scripts/search_rotation_merges.py` tests both ways to reduce it:

- **Merge.** All 44 rotations sit on distinct qubits in one end-of-compute pass.
  Seven qubits (73–76, 83–85) do carry a second rotation later, but between them
  lie 3,130–3,148 circuit ops containing 75–192 gates that do not commute with a
  Z rotation on that qubit (CSWAP writing data in, HMR measuring it out). Exact
  mergeable pairs: **0**. Exact merges would have been free — a dyadic sum is
  the same unitary, so the verified margin cannot move — but there are none.
- **Truncate.** The smallest priced rotation is 2⁻³⁵ turns = 2.998e−11; the
  entire remaining phase slack is 3.975e−12 turns. The smallest rotation is 7.5×
  larger than all the slack that exists. Affordable drops: **0**.

The design already runs its phase precision ~2 bits coarser than the tolerance
would permit, so 315.299429 is a floor for this architecture, not a target that
was never attacked. Removing even one rotation would have been worth 637.8 units
(0.40%); there is no such rotation.

## 6. The contract

**An external accelerator may propose.** Candidate Boolean orderings, candidate
`1/√x` approximation coefficients, surrogate rankings of candidate circuits,
search orderings. All of it is classical work upstream of the exact pipeline.

**Only the exact verifier may accept.** `kickmix` compilation, the official
`verify_submission_invsqrt` / `verify_submission_dqi`, the byte-exact file hash,
and the `Q × C` (or `R·Q²·T`) score computed from the compiled artifact. A
surrogate score, an analog estimate, or a faster search loop is not a result and
must never be written into a receipt.

Practical consequence: a proposal is only admissible once it has been quantized
into the circuit's fixed-point phase units and re-verified over the full input
domain. With 0.0377% of the tolerance unused, an approximate numerical method has
almost no room to be useful — which is why the honest place to look for a better
circuit is structure (EEA schedule, live width, rotation count), not arithmetic
precision.

## 7. What would count as evidence

1. A candidate circuit whose **official receipt** has a lower score than
   157,601.0808, or a DQI receipt below 24.439 T, with `status: PASS`.
2. Reproducibility: the same `circuit_sha256` on a rebuild, and the receipt
   archived alongside it.
3. If an accelerator was used anywhere in the loop: measured wall-clock for both
   the accelerated and unaccelerated search, over the same search space, plus a
   statement of what the search actually found that exhaustive CPU search had
   not.

Absent (1), nothing about the platform matters.

## Disclosed limits

- No photonic hardware was used, timed, or benchmarked for this document. Q.ANT's
  published figures are Gen-2-vs-Gen-1 comparisons (>50× matrix-multiply
  throughput, 25× ResNet-18 inference, 6× energy) and Q.ANT's own site states
  "up to 30×" energy efficiency — not the "90×" or "100×" headline numbers
  circulating in secondary coverage.
- The stage timings are one run on one machine and include import/JIT warm-up;
  they are representative in magnitude, not a benchmark.
- The exchange rates assume the Toffoli-equivalent and qubit accounting of the
  pinned receipt; a change to the scoring formula invalidates them.

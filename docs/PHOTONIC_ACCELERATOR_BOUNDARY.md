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

A co-processor that made the whole pipeline run a thousand times faster would
change the score by exactly zero. `apedp_dyn/quantum/dqi_photonic_resource.py`
already establishes the same invariance for code distance L = 8, 16, 24.

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
| Halting-counter recycling (measured on m=15, n=6, GF(2⁴)) | 0 | −3,223 (−0.72%) | **−0.724%** | yes, if it holds at full scale |
| Itoh–Tsujii 48q scaffold, pooled architecture | 0 | +3,161 (+0.71%) | +0.710% | no — `Q` is bounded by the persistent registers |
| Itoh–Tsujii 48q scaffold, unpooled architecture | +12 | +3,161 (+0.71%) | **−0.637%** | only when the scaffold is not pooled |
| Dual-scaffold 48q division | +12 | +606,974 (+136%) | +133.2% | no, by two orders of magnitude |
| Mixed-polarity PDDD | 0 | −4 | −0.001% | immaterial |
| `n_iters` 222 → 216 | 0 | re-verified worse | −4.11% (worse) | no |

`scripts/test_dual_scaffold_dqi.py` prices the dual-scaffold option against the
*reference* baseline, where the same +606,974 Toffolis is +10.6% of `T_mean` and
the gate returns HALT. At the V6 target scale the identical overhead is +136%,
so the HALT is not close — the script's baseline choice understates it.

**Untested axis, disclosed:** decompositions outside the Davio/butterfly family
that are neither per-node polarity nor a variable-order change — for example a
non-uniform order inside one shared plan, or a non-uniform split tree — are not
covered. The 4.4048% ceiling above applies to them too.

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

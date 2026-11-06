# Budgets behind the two attested factors

**Status:** measurement note · **Scope:** what an emitter for the archived Benchmark II
decoder and the archived Benchmark III bundle would have to hit, derived from the measured
artifacts · **Receipt:** [`benchmarks/quantum_compilation/dqi_width_budget.json`](../benchmarks/quantum_compilation/dqi_width_budget.json)
· **Script:** [`scripts/measure_dqi_width_budget.py`](../scripts/measure_dqi_width_budget.py)

Neither archived artifact has a source in this checkout (see
[`ARTIFACT_ATTESTATION_RULE.md`](ARTIFACT_ATTESTATION_RULE.md)). What their measured widths,
Toffoli counts and error margins *imply* is still decidable, and this note records it. Nothing
here builds a circuit, and no factor changes.

## 1. Benchmark II: 891 qubits is a one-register decoder

The official instance freezes 73 positions of 12 bits, so **one polynomial register is
`73 × 12 = 876` qubits**. The reference decoder holds two of them plus scratch:

| design | qubits | composition |
|---|---|---|
| reference `proos_zalka` | 1,893 | `2 × 876 = 1,752` registers + 141 transient |
| **archived `748b01a7…`** | **891** | **876 (one register) + 15 transient** |
| retired `dqi_schur_forney.py` stub | 878 | 876 + 2, no decoder arithmetic |
| single-array projection | 1,001 | 876 + 125 scratch |

Two consequences follow, and the first is a proof rather than an inference:

1. **No optimisation of the reference can reach 891.** Two registers alone are 1,752 qubits
   before a single control, counter or scratch qubit — `1,752 / 891 = 1.966×` the archive's
   whole width. The archived decoder therefore does not hold both polynomials: it rewrites one
   register in place and keeps ≤ 15 transient qubits.
2. **The single-array packing is verified, and it lands on 899, not 891.**
   [`apedp_dyn/quantum_compilation/inplace_bm.py`](../apedp_dyn/quantum_compilation/inplace_bm.py)
   implements Berlekamp–Massey as one array of `2t + 3 = 73` cells — σ in `0..35`, B in
   `36..71`, the discrepancy in cell 72 — and
   [`scripts/verify_inplace_bm.py`](../scripts/verify_inplace_bm.py) checks it against the
   two-array reference on planted error patterns at **m=15** (GF(2⁴), 250 trials), **m=255**
   (GF(2⁸), 250 trials) and **m=4095** (GF(2¹²), 40 trials): every trial agrees root for root
   and recovers the planted positions. The snapshot σ→B and the update `σ ← σ + (d/b)·xˢ·B`
   share the array because the pass runs descending, so each cell reads `B[j−s]` before it
   writes `B[j]`.

   Two variants are implemented, and the archive's 891 is one of them:

   | variant | update | transients outside the array | total |
   |---|---|---|---|
   | division | `σ ← σ + (d/b)·xˢ·B` | stored `b` (12) + division scaffold (11) = 23 | **899** |
   | **inversionless** | `σ ← δ·σ + d·xˢ·B` | running scale `δ` (12) | **888** |
   | inversionless + control | — | `δ` (12) + flag/counter (3) | **891** |

   The division-free variant needs no quotient, hence no scaffold, and its 888 qubits plus the
   three control bits a reversible build still carries is **exactly the archive's 891** — the
   8-qubit difference between the two variants is precisely `11 − 3`. Both are verified at
   m=15, m=255 and m=4095, and the inversionless locator is only ever checked up to a nonzero
   scale, which is all a Chien search needs. So the delivered 891 is no longer unexplained:
   `876 + 12 + 3`, a one-array division-free Berlekamp–Massey, and 2.106× narrower than the
   1,893-qubit reference.

   **Correction (2026-11-05, measured): the 891 above is not a submission width.** The
   verifier's register contract fixes the register set — `S0..S69` are 70 quantum registers of
   12 bits, so **840 qubits of the circuit's `Q` are the syndromes** — while the packed array is
   a separate `73 × 12 = 876`-qubit buffer. Counting both, this design is **1,731**
   (`840 + 876 + 12 + 3`), and the assembled stream at the official instance measures **1,788**
   (the modelled 1,740 plus 48 qubits the inverse scaffold holds). So `876 + 12 + 3 = 891`
   accounts for the array and not for the syndrome block: 891 leaves **51** qubits outside it,
   and no optimisation of this construction fits. The stream was built and submitted anyway
   (`scripts/build_packed_decoder_submission.py`, 44,725,333 ops, 1,603,035 static Toffolis):
   the register contract accepts it and the official verifier FAILs it on **phase alone** —
   0 value errors, 510 / 1,024 phase errors (ε₊ 0.6944 against the 1/16 gate) — because the
   packed array has no cell for the Forney numerator Ω, so the kickback multiplies by zero.
   Adding Ω needs ≥ 2,148 qubits, wider than the 1,893-qubit reference. Receipt:
   `benchmarks/quantum_compilation/packed_decoder_official_instance.json`.

   Width accounting alone cannot prove the archive *is* this design, and the Toffoli count is
   still not built. Both remaining stages are now measured from real gate streams:

   | stage | measured | projection |
   |---|---|---|
   | key equation (70 iterations) | 4,473 T per GF(2¹²) iteration, 63.0 per GF multiply | **313,110** |
   | coefficient sweep (4,095 positions) | **0 T** per position — constant multiplies are CX/SWAP only | 0 |
   | gated Forney block, every position live | **315 T** per position (inversion + AND + phase product + uncompute), linear over 8 positions | **1,289,925** |
   | uncompute (adjoint of the loop) | **4,473 T** per iteration — equal to the forward pass at every field, because each `ixor_gf2_mul` is its own inverse and never touches its own operands | **313,110** |
   | uncompute (measured clear) | **0 T** — 876 `R` gates and no AND (`reset` here; `mr` from `scripts/kickmix_mr_opcode.patch` is equally AND-free) | **0** |

   **Static count: 1,916,145 Toffolis = 4.83× the archive's 396,919** (1,519,226 over, the
   Forney sweep being 67.3% of it). The uncompute pass is *not* cheaper to replay — that is
   why the reference's adjoint-EEA shape cannot be the archive's — but a measured clear costs
   nothing at all, and the two readings below are what separate the two designs. All four
   stages are also assembled into one gate stream (`scripts/assemble_packed_decoder.py`), which
   reads `T_mean` from the simulator's own execution counts rather than from a sum of parts: at
   the scaled instance (GF(2⁴), t = 5, 10 iterations, 4 positions) it builds **1,062 Toffolis
   over 108 qubits** and the simulator reports **`T_mean` 1,026** (min 990, max 1,062) over 8
   shots — executed/static **0.966**, so a conditioned block really does cost only what it runs.
   The official instance (degree 12, t = 35, 4,095 positions) is *not* built: it needs 70
   iterations and 4,095 blocks, and scoring it needs the sealed 891-qubit contract rather than
   a scaled assembly, so no submission score is claimed from this.

   **Runtime count, which is what `T_mean` measures: 324,135 Toffolis — the archive is 1.22×
   above it.** Three measured facts get there:

   * the Chien *evaluation* is free — the sweep steps 71 coefficients per position by classical
     constants, so it emits CX/SWAP and **0** Toffolis (measured: 31,856 CX, 2,584 SWAP, 0 T);
   * the Forney block only runs where the locator vanishes, and `deg σ ≤ t = 35`, so at most
     **35 of 4,095** blocks execute: `315 × 35 = 11,025` (fits the 83,809 remainder, 13.2% of it);
   * the adjoint can be replaced by a measured clear, measured at **0 T** (876 `R` gates, no
     AND); the phase-safe form needs `mr`, which the shipped extension lacks and
     `scripts/kickmix_mr_opcode.patch` adds, and it is equally AND-free.

   The two sharing variants are both excluded by measurement: unconditional per-position blocks
   cost 1,289,925 (**15.4×** the remainder) and a Montgomery batch inversion — one inversion of
   the product of all 4,095 values, then `3(n − 1)` products — costs **773,811** (**9.2×** it,
   a 1.67× saving over per-position inversion, and still nowhere near fitting).

   **The feasible region is a line.** `scripts/bound_archive_overhead.py` reports it as
   `313,110 + 315·w + u ≤ 396,919.33` over executed blocks `w` and all other overhead `u`: at
   the worst-case weight `w = 35` the block costs 11,025 and leaves **72,784 T (87% of the
   remainder) — 17.8 T per measurement round across 4,095 rounds**; at zero overhead the budget
   tolerates up to **266** executed blocks. The adjoint uncompute does not fit at any weight
   (313,110 against 83,809), Montgomery batch inversion misses by 690,272 T and unconditional
   blocks by 1,206,116 T, so the archive's uncompute must be measurement-based. `tests/test_decoder_stage_accounting.py`
   pins these five receipts against each other.

   So the archive is *consistent* with this construction once its cost is read as a mean over
   executed gates, and consistent with nothing at all under a static reading. What that leaves
   is not a cost puzzle but a design claim: the archive's decoder is a one-register
   Berlekamp–Massey (891 = `876 + 12 + 3`), a measurement-scheduled Chien/Forney (0 T sweep,
   gated blocks) and a measure-based uncompute. The 396,919 Toffolis stay attested as sealed;
   no factor moves.

**Exchange rate, measured.** `score = R·Q²·T` at `Q = 891`, `T = 396,919.33`,
`R = 1.19126`: one qubit is **0.224%** of the score, exactly **891 Toffolis**
(`2T/Q`); one Toffoli is 0.000252%. Halving the score needs `Q → 630` (−261 qubits) or
`T → 198,460` (−50%). At this ratio, every Toffoli-level lever in the repository's catalogue
(halt recycling: −240 Toffolis) buys a quarter of one qubit.

## 2. Benchmark III: no accuracy slack, but 303 qubits of accuracy-neutral width

The bundle's `error_2` is **0.019999997786250796 Ha against a 0.020 Ha gate — a margin of
2.2e-9 Ha.** The smallest measured re-synthesis perturbation at this precision is 3.068e-3 rad
maximum angle error at `brot = 10` (6.136e-3 at `brot = 9`), which is six orders of magnitude
larger than the margin. So:

- **Depth is closed.** Any rewrite that changes rotation precision — `brot` 8→7, `keep_bits`
  10→9, fewer passes — moves the energy by far more than 2.2e-9 Ha and fails the gate. The
  §3b break-even in [`DFTHCSOS_SUB128_SCOPE.md`](DFTHCSOS_SUB128_SCOPE.md) (a coarser family
  wins only if `lambda_eff` loses ≤ 12.5%) is therefore moot at this operating point: the
  bundle has already spent that budget to reach the gate.
- **Width is the only admissible direction,** and only in the *exact* part of the circuit. The
  angle circuits sit at the 128-qubit cap (`angles_within_cap: true`), but the bundle reports
  **303 temporary qubits** in the inner state prep out of `Q = 427`. State preparation is
  exact, so recycling that scratch is accuracy-neutral: up to **303 qubits = 71% of width**,
  and at `1/Q = 0.234%` per qubit, at most **71% off the score** in principle.
- **The emitter exists, and no retune of it can reach the archive.** `scripts/build_dfthcsos_sub128.py
  --brot 8 --batch-size 5` builds the family today: 15 circuits (matching the archive's
  `K = 15`), 63 qubits per circuit (12 selection + 5×8 words + 11 ancillas) and 3,974 Toffolis
  each, 119,220 over both passes, against the archive's 8,718. `scripts/measure_dfth_qrom_shapes.py`
  prices every shape of that family under the verifier's accounting:

  * **arity is neutral.** A `k`-ary node spends `k − 1` ANDs to decode one of `k` children and
    there are `(rows − 1)/(k − 1)` such nodes, so every arity costs `rows − 1` per circuit —
    measured 3,974 at `k = 2`. Arity buys depth, never gates.
  * **QROAM's optimum does not fit.** `rows/b + b` is minimised at `b = √3976 = 63`, giving
    127 Toffolis per circuit and 1,905 in total — but the block workspace is `J·brot·b = 2,594`
    qubits, 20.3× the 128-qubit cap.
  * **Inside the cap the best block loses.** `b = 2` fits at 93 qubits and costs 1,990 per
    circuit, 29,850 in total — worse than the segment tree it would replace.

  The archive's 581 Toffolis per circuit are therefore **3.42× below the cheapest cap-feasible
  shape of this family**, so the gap is the encoding and the table, not the tuning, and its 303
  temporary qubits are not reclaimable by re-shaping this QROM.
- **No table encoding of this payload closes the gap either.** `scripts/determine_dfthcsos_load_encoding.py`
  tests every encoding that could load 40 bits per address in fewer nodes than addresses, on the
  real table:

  | encoding | what it would exploit | measured on the table |
  |---|---|---|
  | run-length | piecewise-constant address ranges | **3,976 runs of 3,976 rows** in every circuit — zero gain |
  | dedup | repeated 40-bit rows | **3,976 distinct rows** in 14 circuits, 3,975 in the 15th |
  | sparse delta | unchanged rows between circuits | **0 zero rows** in every delta table, 0 tree breaks |
  | GF(2)-affine | `word = A·address ⊕ c`, loads with CX only | **31 of 40 payload bits have no affine form** |
  | accumulator | constant modular difference between rows | **0 of 75 word slots** are uniform ramps |

  Every one of them costs the same **59,610** as the address tree (`rows − 2` per circuit,
  confirmed by building 256- and 1,024-row tables: 254 and 1,022 Toffolis). So the archive's
  8,718 is **6.84× below the floor of every per-address encoding of this payload**, which cannot
  be reconciled by loading the same 40 bits per address at all: its angle circuits are not a
  QROM over this table. That is a different circuit family, not a better-tuned one, and its 303
  temporary qubits stay unreclaimed.

**Exchange rate, measured.** `score = λ·T·Q` at `Q = 427`, `T = 15,459`, `λ = 47.1847`: one
qubit is **0.234%** (15,459 Toffolis), one Toffoli is 0.0065% (427 qubits). Halving the score
needs `Q → 302` (−125 qubits) or `T → 7,730`, and the second is excluded by the margin above.

## 3. What would unblock either

Benchmark II's width is settled (891 = `876 + 12 + 3`, verified at three fields) and all four
stages of its arithmetic are measured, so the static total (1,916,145, 4.83×) and the
runtime-conditioned total (324,135, which the archive exceeds by 1.22×) are both pinned. What
remains is the one piece kickmix cannot count — the measure-based uncompute, which needs `mr`
and is priced at 0 ANDs by construction — and the assembly of the four stages into one stream
so the official verifier reads `T_mean` from it directly. The archive's construction is a
design claim at this point, not a cost mystery.

Benchmark III needs a different circuit family, not a better-tuned QROM: every per-address
encoding of the 40-bit payload — tree, run-length, dedup, sparse delta, affine, accumulator —
lands on the same 59,610 Toffolis against the archive's 8,718, and the 128-qubit cap rules out
the only shape that goes lower. Both remaining items are algorithm work with a verifier receipt
at the end, not an optimisation pass.

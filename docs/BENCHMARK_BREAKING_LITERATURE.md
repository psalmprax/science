# Breaking the Three Benchmarks — What the Literature Offers, Priced

**Status:** `audit-only` (survey + exchange-rate arithmetic; no construction below has been built)
**Scope:** Benchmark I (invsqrt), Benchmark II (DQI), Benchmark III (DFTHC-SOS),
against the published literature that could plausibly move their scores.

This document exists because "is there a paper that beats this" is answered badly
by a reading list. Every technique below is priced against the *measured*
exchange rate of the benchmark it would attack, using the scores and resource
counts this repository has actually produced. A technique that cannot clear its
own bar is recorded as such rather than left as an opportunity.

---

## 1. The arithmetic a gain has to clear

| Benchmark | Score | ∂score/∂T | ∂score/∂Q | 1 qubit is worth |
|---|---|---|---|---|
| I — invsqrt | `Q · toffoli_equivalent` = 157,601.08 (Q=89, T=1770.798661) | **89.0** | **1,770.8** | 19.90 Toffoli-equivalents |
| II — DQI | `R(ε₊) · Q² · T_mean` (Q=1893, T=5,725,108.38) | `R·Q²` | `2RQT` | 6,048.7 Toffolis |
| III — DFTHC-SOS | `λ_eff · T_total · Q_total` = 3.3556 G (λ_eff=49.301369, T=93,237, Q=730) | **35,990.0** | **4,596,711.8** | 127.72 Toffolis |

Three consequences that are easy to get wrong:

1. **Benchmark III prices qubits 128× higher than Toffolis.** A proposal that
   adds one qubit to save 127 Toffolis is exactly break-even; save 128 and it
   pays. Benchmark II is the extreme opposite: at Q=1893 one qubit is worth
   6,048.7 Toffolis, so anything that widens a register needs an enormous gate
   saving to be worth considering.
2. **Benchmark I prices a qubit at only 19.9 Toffolis.** It is the one benchmark
   where spending space to save gates is normally the right trade.
3. **The rotation cost in Benchmark I is a fixed function of ε.** The verifier
   charges `0.56·log₂(1/ε) + 5.3` T gates per non-Clifford `Z_POW`
   (`verify_submission_invsqrt.rotation_t_count`). A better Clifford+T
   synthesiser — however strong the paper — changes nothing, because the score
   does not use the compiled count. Only `n_rot` and the phase margin are real
   levers. Both are exhausted (see §4).

## 2. Benchmark I — inverse square root (89 qubits, 157,601.08)

The champion is a Newton–Raphson inversion in the Häner–Roetteler–Svore
lineage: a piecewise degree-4 polynomial initial guess (Horner), followed by
Newton steps, with the phase applied per output bit.

| Technique | Source | Changes | Does it clear the bar? |
|---|---|---|---|
| Optimised initial guess to remove Newton iterations | arXiv:1805.12445 (§inverse square root) | `T`, `Q` | **Already applied.** The champion's structure is this construction; the remaining iterations are what the 4-coefficient Horner provides. |
| T-count-optimised square root, `2n+1` qubits, no garbage | arXiv:1712.08254 (Muñoz-Coreas & Thapliyal) | `T`, `Q` | Worth re-deriving at n_p=10. It targets the *integer* square root, not the reciprocal — it is a candidate for the `_square` stage only. Needs a build; not estimated here. |
| Toffoli-based modular multiplication with fewer Toffolis | Häner et al., QIC 17, 673 (2017) | `T` | The champion's multipliers already use this family. No published constant-factor win at 10-bit operands that survives the exact-domain compiler. |
| Schoolbook multiplication via controlled add-subtract | arXiv:2410.00525 class ("fewer Toffoli") | `T` | Only relevant if it beats the current `init_and`-style schoolbook at 10 bits; at this width the compiler already removes ~10 Toffolis. Unverified. |
| ZX-calculus T-count reduction | Kissinger & van de Wetering, PRA 102, 022406 (2020) | compiled `T` | **Cannot help**: the score uses the benchmark's own `rotation_t_count(ε)`, not the compiled count. |
| Shorter single-qubit synthesis | Kliuchnikov, ePrint 2022/372 | compiled `T` | **Cannot help**, same reason. This is the single most common misreading of this benchmark. |
| Better phase application (fewer rotations) | any | `n_rot` | Only remaining lever, and it is measured closed: 0 mergeable pairs, 0 affordable drops (§4). |
| QROM / windowed phase loading | Babbush et al. line of work | `Q`↑, `T`↑ | Both terms get worse; the score is `Q·T`. |

## 3. Benchmark II — DQI Reed–Solomon decoding (1893 qubits, 5,725,108.38)

The reference circuit is the Proos–Zalka decoder from the DQI paper itself
(Khattar, Shutty, Gidney, Zalcman, Yosri, Maslov, Babbush, Jordan,
arXiv:2510.10967), whose own resource estimate is "approximately 5.72 million
Toffoli gates" — the same number this repository verifies. That matters: the
published state of the art *is* the baseline here.

| Technique | Source | Changes | Does it clear the bar? |
|---|---|---|---|
| New EEA circuits, implicit Bézout coefficient access, in-place architecture | arXiv:2510.10967 §4 | `Q`, `T` | This is what the baseline already implements; the paper's claim is `2nb` qubits at leading order. At n=70, b=12 that is 1,680, so the reference's 1,893 carries ~213 qubits (12.7 %) of overhead. Per-qubit price at Q=1893 is 6,048.7 Toffolis, so reclaiming even 3 qubits is worth ~2.7 % of `T_mean`, but it must come from the *control-flag* accounting, not the data path. |
| Compressed / regularised DF, rank reduction | Oumarou et al., Quantum 8, 1371 (2024); Rocca et al. (2024) | the classical factorisation, not the decoder | **Wrong target**: Benchmark II's `r` and field are fixed by the instance. |
| Better syndrome decoding (soft / list decoding) | Delfosse–Nickerson (2021) line | algorithm, not circuit | Would change which decoder is reversible — a new circuit would have to be rebuilt and re-verified from scratch. No cost estimate exists at n=70, b=12. |
| Degree-register recycling (Gluzdov V6 target) | in-repo, `dqi_gluzdov_v6.py` | `Q`↓, `T`↓ | Already priced in `PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.2. Not a literature import. |
| Halting-counter recycling | in-repo prototype | `T` | Measured dead at scale: −240 Toffolis = −0.0118 % at the benchmark instance, and the shipped variant (flag + counter) is **+0.024 %**. |

## 4. Where Benchmark I's remaining headroom actually is

Measured on the champion receipt, not estimated:

- `T` = 1,455.499232 Toffolis + 315.299429 rotation equivalent (44 rotations,
  ε = 2.838e-13, 28.64 T each) + 1 T gate.
- The rotation term is 17.81 % of the score. It is **irreducible here**:
  `scripts/search_rotation_merges.py` finds 0 exact mergeable pairs (all 44 sit
  on distinct qubits; the seven qubits that carry a second rotation have
  75–192 non-commuting gates between the two), and 0 affordable truncations (the
  smallest rotation is 2⁻³⁵ = 2.998e-11 turns against 3.975e-12 turns of total
  slack — 7.5× larger than all the tolerance left).
- The Boolean-restoration line is capped at 4.4048 % even if it became free.
- The mixed-polarity PDDD search found a genuinely smaller plan (26 vs 28 shared
  nodes) that compiles to the same 1,466 static Toffolis, because the compiler
  had already removed the Toffolis the smaller plan was trying to save.

So a literature attack on Benchmark I has to come from the *arithmetic*, not the
phase application: fewer Horner coefficients at acceptable model error, or a
square stage as good as arXiv:1712.08254 at n_p=10. Both are bounded by the
model tolerance, which is currently used to 99.962 % (worst error over tolerance
0.9996227) and therefore cannot fund a cheaper polynomial.

## 5. Benchmark III — DF-THC (730 qubits, 3.3556 G)

`λ_eff` is derived from the fixed FeMoCo76o113e factorisation metrics
(`error_1`, `Λ`, `E_SOS`), so the only mover is `T_total` — the cost of the
PREPARE/SELECT and Givens-rotation networks over the submitted angle words.

| Technique | Source | Changes | Does it clear the bar? |
|---|---|---|---|
| THC instead of DF | Lee et al., PRX Quantum 2, 030305 (2021) | `T`, `Q` | The instance supplies a *fixed* DF factorisation (`.npz`); a different decomposition is not accepted by the verifier's cost model. |
| Improved tensor factorisation + active-volume compilation (2 orders of magnitude) | Caesura et al., arXiv:2501.06165 (2025) | `T`, `Q` | The 2-orders figure is for a *different* algorithm's resource estimate, not for angle-word circuits. At Benchmark III's price (1 qubit = 127.7 Toffolis) any active-volume change must be re-priced, not quoted. |
| Fewer non-Clifford ops per Givens rotation | same DF-THC line | `T` | This is the live lever. Each Toffoli saved is 35,990 score units, and 1 % of `T_total` is 1 % of the score. The in-repo angle-ensemble work (25 `angle_k_*.kmb` circuits) is the right harness. |
| Trading qubits for Toffolis | any | both | Only pays below 127 Toffolis per qubit added. |

## 6. What the literature cannot do, for any of the three

- **Change the score by being faster.** None of the three formulas contains a
  timing term; `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` prices that and writes
  the non-submission artifact.
- **Change the score via a better synthesiser for Benchmark I.** The rotation
  cost is the benchmark's own formula, evaluated from ε alone.
- **Skip verification.** A lower `T` that fails `verify` scores `inf`. Every
  technique above is a *proposal* until the artifact verifies, and the verifiers
  are the only acceptance path.

## 7. Honest limits of this survey

- **No construction in §2 or §5 was built.** The "does it clear the bar" column
  is arithmetic on measured exchange rates, plus a judgement about whether the
  technique applies to this instance. Nothing here is a measured gain.
- **Not every family was searched.** In particular: precision-tunable Newton
  iteration counts at exactly n_p=10, Chebyshev versus Horner at the same model
  error, and angle-word re-encoding for DF-THC were not exhausted.
- **Two citation details are at the search-snippet level** (the "fewer Toffoli
  schoolbook multiplication" item and the active-volume hybrid); their arXiv
  identifiers should be pinned before anyone acts on them.

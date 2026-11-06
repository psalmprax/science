# Benchmark I: a new decomposition — spec, budget, and what the tolerance actually demands

**Status:** spec with classical screens · **Target:** beat the public 62-Q circuit at
**100,569.53** (`sha256 bce2948f…`, `PASS`, Q = 62, 1,325.5 executed T, 41 rotations,
30,001 classical bits) · **Shipped champion:** 89 Q / 1,770.8 `te` / **157,601.08**
(`sha256 ce04c9d5…`).

This note answers the question "design a new circuit without using the current one as design
knowledge" with a budget instead of a build, because the first thing a clean-room design has
to obey is a constraint that the current circuit encodes silently and that no earlier note
states: **the verifier's tolerance is relative, not absolute.**

## 1. The tolerance is relative — and that sets the precision floor

`verify_submission_invsqrt.py` accepts a phase error of at most
`tol(x) = 2^-24 · β / sqrt(x)` **turns**. The phase itself is `1/sqrt(x)` turns, so the
tolerance is `2^-24` of the *value being produced*, and the required **absolute** precision
is therefore `2^-24 · 1/sqrt(x)`, which tightens as `x` grows. Over the domain
`x ∈ [1, 3,133,452]` the phase spans `[2^-21.6, 1]` turns, so the worst case needs about
**38 bits of absolute precision** (at `x = 3,133,452`, `1/sqrt(x) ≈ 2^-21.6` and
`tol ≈ 2^-45.6`).

Measured against the champion's own exact integer model (`scheduled_arithmetic90.integer_model`,
units `2^-(fy+10) = 2^-38` turns), truncating the produced phase to `n` bits:

| phase bits kept | worst error / tol | x over tolerance | worst x |
|---|---|---|---|
| full model (38) | **1.000000** | 3 | 995,358 |
| 27 | 228.1 | 3,103,713 | 2,744,120 |
| 26 | 455.7 | 3,118,767 | 2,748,664 |
| 25 | 910.8 | 3,126,181 | 2,748,664 |
| 24 | 1,820.8 | 3,129,842 | 2,776,856 |

The control row reproduces the recorded 0.9628-of-tolerance result within the documented 3.7%
model error bar (three points marginally over). Every truncation is catastrophic: dropping the
low 14 bits of a 38-bit phase is a *constant* `2^-24`-turn error, but the tolerance at large
`x` is `2^-45.6`, so the same cut is 1,820× too coarse exactly where it matters.

**Consequence.** Any design that computes the phase into a register and then phases it must
carry ~38 bits of *value*, which the champion does with a 21-bit accumulator plus an 8-bit
high word plus a conditional shift of up to `2^4` (`scheduled_arithmetic90.shift_stages`,
`phase()`), i.e. 29 held bits covering 38 bits of precision. There is no schedule that holds
fewer bits and still phases to `2^-45.6`: **the held-accumulator class floors near the
champion's width**, which is where 89 Q comes from.

## 2. Option 3 — reconstructing the 62-Q frontier: measured, not possible by recycling

The in-repo reference is the paper's Newton-Raphson circuit
(`verifiers/invsqrt/nr_reference_circuit.py`), measured this session:

| build | Q | static T | rotations |
|---|---|---|---|
| `build_nr_circuit(10)` | **260** | 7,735 | 37 |

Its register set is `x` (22) + `flags` (10) + `selectors` (3) + `coefs` (4 × 17 = 68) +
`square` (28) + `z` (30) + `y1` (40) + `scratch` (29) = 230 named, 260 with carries. A
qubit-lifetime sweep of the emitted instruction stream reports **260 qubits simultaneously
live**, because the builder allocates every register up front and nothing is released until
the uncompute.

Walking `_compute_y1`'s dependency graph by hand gives the best any recycling schedule could
do, loading the four coefficients two at a time (the Horner steps `(c2,c3)`, `(c1,c2)`,
`(c0,c1)` need an accumulator and one addend simultaneously, never more) and freeing `square`
after `_multiply_m`:

| stage | simultaneously live |
|---|---|
| Horner | x 22 + flags 10 + selectors 3 + 2 × coef 34 + scratch 29 ≈ **98** |
| `_square` | + y0 17 + square 28 ≈ **143** |
| `_multiply_m` | + z 30 ≈ **173** |
| `_multiply_y0` | y0 17 + z 30 + y1 ≈ **25–40** + x/flags ≈ **130** |

So the Newton-Raphson family bottoms out near **Q ≈ 130–170**, i.e. *wider* than the champion
and with 5× its Toffoli count. The public circuit's `62 = 22 x + 40` working qubits cannot
come from register recycling of this algorithm — it must never hold a wide result at all,
phasing digit by digit. That is the same conclusion `INVSQRT_SUB62_DESIGN.md` §5 reaches from
the other side, and it means the frontier is **not reconstructible** from anything in this
repository; the shipped public notebook is the only source, and it is not vendored here.

## 3. Candidate clean-room decompositions, priced from measured primitives

Primitive costs measured this session with the pinned builder
(`CircuitBuilder.init_lookup`, `operation_statistics`):

| QROM address bits | Toffolis | qubits (addr + 14-bit out + iteration) |
|---|---|---|
| 6 | 62 | 26 |
| 8 | **254** | 30 |
| 9 | 510 | 32 |
| 10 | 1,022 | 34 |
| 12 | 4,094 | 38 |

The QROM cost is `2^(k+1) − 2` and is **independent of the output width** — a 54-bit table
entry costs the same 254 Toffolis as an 8-bit one. Wide entries are free; wider addressing is
exponential. (`init_anf` at k = 8 is 247, so the champion's ANF load is already the cheaper
of the two.)

| # | decomposition | why it is priced, not built | verdict |
|---|---|---|---|
| A | piecewise-quadratic Horner (`invsqrt_direct_poly`) | swept 12 precision points: best **Q = 135 / T = 3,464** (10,23,10); every point ≥ 2.4× the champion's score. Two column-truncated multiplies plus a 25-bit result register cannot fit §1's 38-bit need in 40 qubits. | **negative** |
| B | digit-by-digit phased accumulator, per-digit PDDD | `prototype_bitserial_sqrt.py` at 22 bits: **34,997 Q / 101,914 T** — each digit's function is a full-prefix table whose transients grow with the table. | **negative** |
| C | same, one shared multi-output PDDD for all digits | `prototype_digit_shared_pddd.py`: shared plans are **1.4–1.5× wider at every k** (the plan carries the whole `2k`-bit address per digit) and buy ~1% of the Toffolis. | **negative** |
| D | phase the correction immediately, without an accumulator | the correction is `Q = Σ_i sign_i · (R >> sh1_i)` with **quantum** signs `sign_i = 2·x_i − 1`. Phasing bit `j` of `R` directly needs angle `Σ_i sign_i 2^{j−sh1_i}`, a sum over 14 address bits — i.e. a CCZ per (bit, row) = 196 rotations with address-dependent angles. T-cost explodes; the accumulator exists precisely to make those 14 contributions **classical** before phasing. | **negative — built and measured, §6** |
| E | two-level table: 8-bit coarse + fine table on the low 14 bits | the fine table's address is 14 bits → `2^15 − 2 = 32,766` Toffolis for one load. | **negative** |
| F | 9- or 10-bit single table + shorter correction | +256 / +768 Toffolis to save 2–4 qubits of accumulator; at the 26.2 te-per-qubit exchange rate this is score-negative at every width above ~52. | **negative** |

What is **not** available in the pinned binary, and would change the arithmetic rather than
the schedule:

- **A Z-basis read (`mr`).** `docs/MR_INSTRUCTION_SPEC.md` specifies it and
  `scripts/build_kickmix_mr.py` builds it, but `apedp_dyn/quantum_compilation/contracts.py`
  still pins the official binary, which has no `MR`. With it, Lever A of
  `INVSQRT_SUB62_DESIGN.md` §2 (carry-chain segmentation to Q = 76, score 138,837) becomes
  buildable, and computational predicates become classically branchable — which is where the
  public circuit's 169.5 skipped Toffolis and 30,001 classical bits come from. Without it,
  neither is reachable: `hmr` measures in the X basis and `reset` randomizes the tracked
  phase.
- **A sub-exponential table primitive.** Every route to a wider address pays `2^(k+1) − 2`.

## 4. What a winning design has to look like, stated as a build order

Given §1, a sub-62 design must **never hold more than ~29 bits of phase value at once**, which
forces the phase to be emitted in pieces whose individual precision is far below 38 bits and
whose errors stay inside `2^-24 · 1/sqrt(x)`. The only structure that does that is
digit-by-digit emission, and the only way its per-digit functions stay cheap is if each digit
depends on a **short** prefix — which contradicts the 22-bit input unless the function is
decomposed hierarchically. The candidate hierarchy:

1. **Emit the phase's top `t` bits from a table indexed by the exponent class and the top 6
   mantissa bits** (a `2^6`-entry table per class, 12 classes, cost ~12 × 62 = 744 Toffolis
   if built per class, or one 8-bit table if the exponent is folded in — which is what the
   champion's `flags[0]` parity bit already does).
2. **Then emit the next `t` bits from a table indexed by the next 6 mantissa bits, with the
   previous digits' contribution to the *carry* handled by a single conditional correction.**
3. Each emission is phased immediately and its register freed, so the held value never
   exceeds `t + carry` bits.

The classical screen for step 1–2 has been **run**, exactly in 64-bit phase units over
525,184 mantissa samples (stride 4,095 plus every segment boundary and midpoint for digit
widths 6, 12 and 18), with `g(m) = 1/sqrt(m)` evaluated as `isqrt(2^150 // M)` and the
tolerance `2^-24` **relative** (the exponent factor cancels, so the screen is over the
mantissa alone):

| split (mantissa bits per digit) | residual/tol after digit 1 | after digit 2 | after digit 3 | after digit 4 |
|---|---|---|---|---|
| 6, 6, 6, 4 | 1.30e+05 | 2.04e+03 | **3.00e+01** | 0 |
| 8, 8, 6 | 3.27e+04 | 1.26e+02 | **0** | — |
| 6, 6, 10 | 1.30e+05 | 2.04e+03 | **0** | — |
| 11, 11 | 4.09e+03 | **0** | — | — |
| 12, 10 | 2.04e+03 | **0** | — | — |

**A piecewise-constant digit emission cannot meet the tolerance before the last digit, and the
last digit is indexed by the whole mantissa** — `2^22` entries, which is the full-prefix table
that `prototype_bitserial_sqrt.py` already measures at 34,997 Q / 101,914 T. The residual
falls as `2^-2·(bits emitted)`, so 24 emitted bits are needed for `2^-24`, and there are only
22 to emit.

The one structure that resolves the mantissa with a short table is **linear** (error `~h^2`
per level, corrected to `2^-38` by a 14-row signed-digit sum), and that is the champion. A
quadratic per digit does converge early enough (a 6-bit index with a quadratic on the low 16
bits is already inside tolerance), but a single quadratic level is exactly family A above,
measured at Q ≥ 135 / T ≥ 3,464. So this screen **closes** the digit-emission family rather
than opening it: it is the same trade the champion already sits at the optimum of.

## 5. Verdict

- A clean-room circuit **cannot** be produced by rearranging the Newton-Raphson reference:
  recycling it floors near Q ≈ 130–170 with 5× the Toffolis (§2).
- Every clean-room family that has been built or priced — quadratic Horner, QROM seed,
  digit-serial, shared-PDDD digit-serial, two-level table, wider single table, immediate
  phasing — is **score-negative** against 157,601, let alone 100,569.53 (§3), and the
  digit-emission screen of §4 closes the last unmeasured candidate in the same direction.
- The binding constraint is the **relative tolerance**: ~38 bits of absolute phase precision,
  which forces ~29 held bits and therefore ~86 qubits for any held-accumulator design (§1).
  The 89-Q champion sits at that floor.
- The frontier needs either an upstream Z-basis read (`mr`, §3) to make carry segmentation and
  classical branching buildable, or a function decomposition that resolves 22 mantissa bits
  with a table whose cost is sub-exponential in the index width. Neither exists in this
  repository, and the public 62-Q notebook that contains one is not vendored here.

## 6. Tested: candidate D, built and measured
Candidate D is the only family in §3 that attacks the width floor rather than the gate count,
so it was built rather than priced. Both schemes were constructed at the champion's own outer
dimensions (14 sign bits, 14-bit derivative, 21-bit accumulator, `sh1[i] = 13 − i`, `fy = 28`)
and priced with the verifier's own cost model
(`verify_submission_invsqrt.rotation_t_count`, `eps = π·min_margin/2^64/n_rot`):

| scheme | qubits | Toffolis | rotations | est. T | **te** |
|---|---|---|---|---|---|
| 1 — accumulate then phase (the champion's outer) | 68 | 280 | 20 | 561 | **420** |
| 2 — phase each `(row, bit)` product directly (new) | **29** | **105** | **119** | 3,511 | **983** |

**The new scheme is 2.34× worse per stage**, despite using 2.7× fewer Toffolis and 2.3× fewer
qubits. Folded into the whole circuit it removes the outer's 560 Toffolis and 20 rotations and
adds 105 Toffolis and 119 rotations, giving `te ≈ 2,051` against the champion's 1,770.8 —
**score ≈ 182,521 against 157,601, i.e. 1.16× worse**.

The reason is a property of the cost model that no earlier note records, and it is the same
property that makes the champion's shape inevitable:

`eps = π · min_margin / 2^64 / n_rot`, `T_rot = 0.56 log2(1/eps) + 5.3`

so the per-rotation T-count **grows with the number of rotations** whenever the circuit's
phase margin is tight — and the margin is tight by construction, because `min_margin` is
`tol(x) − |error|` bounded above by the tolerance itself. The accumulator is therefore a
**rotation compressor**: it folds 105 `sign_i · R[j]` product terms into 21 bits, and the 21
rotations that phase them are 6× fewer than the 119 the direct scheme needs. Break-even for
scheme 2 requires `T_rot < 7.07`, i.e. `eps > 0.11`, i.e. a phase margin of 0.042 turns —
**700,000× the tolerance at `x = 1`**, which no circuit can have because the margin cannot
exceed the tolerance. Scheme 2 cannot win at any accuracy.

The one variant that would have made the qubit saving real — phase-kickback table loading, to
drop the 21-bit `low` register entirely instead of holding it for the `K0` base — was priced
from the champion's own table: `mobius_transform` of the 256-entry, 54-bit `full_table` has
**all 256 masks nonzero and 6,676 nonzero (mask, bit) pairs**, so it would emit 6,676
rotations against the champion's 44, i.e. **+48,911 te** at the measured per-rotation cost.
Blocked by the same law.

So the test confirms the spec's thesis from the cost side rather than the error side: the
champion's accumulator is not a scheduling artefact, it is the only known way to keep the
rotation count — and therefore the `te` — down.

## 7. Two optimisation landscapes, and what each one licenses

A weight-space-inspired search over this circuit was tested rather than assumed, and it split
the parameterisation into two regions with opposite properties. Both were screened with the
exact integer model over all 3,133,452 `x`, in the verifier's own units.

### 7.1 Coefficient space (768 integers: K0, K1, K2) — fragile

Perturbing the champion's calibrated tables by **±1 LSB on 1–5 randomly chosen entries**:

| field | worst error / tol over 12 trials | trials failing |
|---|---|---|
| K2 (curvature) | 1.5 – 2.7 | **12 / 12** |
| K1 (derivative) | 1.0 – 1.11 | **12 / 12** |
| K0 (offset) | 1.0 – 1.06 | **12 / 12** |

and interpolating two passing configurations' tables at λ ∈ {0.25, 0.5, 0.75} (28 pairs):

| best of 28 interpolations | worst error / tol | x over tolerance |
|---|---|---|
| `champion + 0.25·(nodrop − champion)` | **1.20** | 74 |

**What this establishes, stated exactly:** *naive interpolation of the calibrated coefficient
tables is not a viable optimisation strategy under the tested conditions.* It is evidence of
local fragility under the perturbations actually applied — a ±1 LSB move on a handful of the
768 integers, and a linear blend of two calibrated tables. It is **not** a proof that the
feasible region has measure zero, and it is **not** a refutation of task arithmetic, model
merging, or weight-space composition in general: those are different mathematical objects that
rely on architectural correspondence and parameter alignment, which these tables do not have.
What it does license, practically:

- ordinary coefficient interpolation is unlikely to be useful;
- gradient-based optimisation is poorly matched to an objective that is a *max* over 3.1M
  points of a piecewise-constant integer function;
- **structured transformations that preserve the exact-domain invariants** (Gray-code
  re-encodings, per-segment exact recalibration, symmetry-respecting moves) remain open and
  are the right thing to research here;
- no learned model should be given unrestricted authority over these coefficients.

### 7.2 Configuration space (≈8 discrete parameters: f2, f1, fyr, gy, drop2) — a staircase

Swept exactly, tables regenerated per configuration with `regenerate_tables_exact`:

| f1 (f2=8, fyr=27, gy=1, drop=(5,)) | 15 | 16 | 17 | 18 | **19** | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|
| worst error / tol | 5.40 | 3.20 | 2.00 | 1.25 | **1.00** | 0.889 | 0.889 | 0.889 | 0.889 | 0.889 |
| w1 (derivative width) | 14 | 15 | 16 | 17 | **18** | 19 | 20 | 21 | 22 | 23 |
| correction rows | 5 | 6 | 7 | 8 | **8** | 9 | 10 | 11 | 12 | 13 |

| f2 (f1=19, fyr=27, gy=1, drop=(5,)) | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| worst error / tol | 1.375 | **1.00** | 0.90 | 0.90 | 0.90 | 0.90 |
| w1 / rows / w2 | 18 / 8 / 6 | **18 / 8 / 7** | 18 / 8 / 8 | 18 / 8 / 9 | 18 / 8 / 10 | 18 / 8 / 11 |

This is **not a smooth continuous landscape — it is a staircase with plateaus**: structured
improvement, then saturation at 0.889 (f1 ≥ 20) and 0.90 (f2 ≥ 9) as the calibration's floor
stops being set by the correction rows and starts being set by the table's own modulus. That
is still an excellent surrogate target: it is low-dimensional and discrete, and the objective
is cheap (one exact screen ≈ 10 s).

**A screen-positive candidate.** `f2 = 9` is layout-identical to the champion (w1 = 18, 8 rows,
LB = 21, whi = 8) except `w2` 7 → 8, and improves the model margin from 1.00 to 0.90 of
tolerance. Through the verifier's `eps = π·min_margin/2^64/n_rot` rotation pricing that margin
is worth roughly 1.5% of the score. **This is a screen-positive candidate, not a verified score
improvement** — see §8 for why it does not yet build, and §9 for what must be measured before
any number is claimed.

## 8. Why `f2 = 9` does not build: the hard-coded `w2 = 7`

The champion's builder (`scripts/test_shared_pddd_circuit.py`) hard-codes the curvature
checkpoint for `w2 = 7` in five places, so any other `w2` mis-assembles the measurement records
and fails:

```
checkpoint_start = 4                                  # = w2 − 3, the top 3 curvature bits
for j in range(checkpoint_start, 6): ...              # 6 = w2
replacement = b.alloc_qubits(3)                       # 3 = w2 − checkpoint_start
targets_dict = {'curve4': ..., 'curve5': ..., 'curve6': ...}
for j in reversed(range(checkpoint_start, 6)): ...
```

With `w2 = 8` the erased region is `c2[4:8]` (4 bits) but the restore still allocates 3, so
`rest_target` carries 26 qubits where the table layout needs 27, and
`records[bit]` (`phase_from_records`, line 79) indexes past the end. The affected function sits
on the **HMR phase-fixup path**, so this is a mathematical-consistency repair, not a software
compatibility fix: the record-index correspondence is what makes the measurement-based
uncompute phase-exact.

## 9. The sequence, and what each step is allowed to claim

| # | step | what it may claim |
|---|---|---|
| 1 | make the `w2` mapping parameter-dependent; keep the `w2 = 7` build as a byte-identical golden regression | nothing yet — only that the refactor is semantics-preserving |
| 2 | build `w2 = 8` end-to-end and pass the **official verifier** | a verified `Q`, `te` and score for that configuration |
| 3 | generate the exact configuration corpus (screen, worst error, min margin, resource cost, build outcome, verifier outcome per configuration) | a measured map of the staircase |
| 4 | train a conventional surrogate (random forest / Extra Trees, GP where the representation allows, discrete Bayesian optimisation) **and compare against exhaustive and random search before claiming learning helps** | whether learning beats brute force on this staircase |
| 5 | evaluate the actual score improvement with the exact scoring implementation and the official verifier | a score delta — a numerical screen cannot establish this |

The optimisation target throughout is **robust verifier margin**, not a pass/fail threshold: a
candidate barely inside tolerance is fragile, and the staircase's plateaus mean the margin is
what distinguishes one configuration from the next.

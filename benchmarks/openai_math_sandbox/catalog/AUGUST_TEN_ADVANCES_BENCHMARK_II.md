# OpenAI's August 1 batch, screened against Benchmark II

**Source:** OpenAI, *Ten advances in mathematics and theoretical computer science*,
2026-08-01 — <https://openai.com/index/ten-advances-in-mathematics/>. Machine-readable
screen: [`august_ten_advances_screen.json`](august_ten_advances_screen.json), pinned by
`tests/test_august_ten_advances_benchmark_ii.py`.

**Why this batch and not the October one.** The October 6 catalog (719 manuscripts, 17
disciplines) is already ingested in full at `catalog/manuscripts_manifest.json` and screened
`372/372 uncheckable_proof_level`. None of its 17 disciplines is coding theory, arithmetic
complexity or gate synthesis, so it cannot touch Benchmark II. This batch explicitly spans
*coding theory, arithmetic circuit complexity, quantum complexity and lattice cryptography* —
the only OpenAI release whose subject matter intersects the DQI decoder.

## What Benchmark II actually costs

Score = `R(eps_plus) · Q² · T_mean`, with `R` measured by the verifier from the submission's
own shot failures. `R` is not a lever: it is a property of the submitted circuit, measured.
Only `Q` and `T_mean` move.

| | Q | T_mean | |
|---|---|---|---|
| Reference (`proos_zalka_reference_n_iters_222`) | 1,893 | 5,725,108.38 | |
| Target (`gluzdov_v6_degree_register_recycling_target`) | 1,789 | 445,229.0 | |
| Required factor | **1.058×** (−104, −5.49 %) | **12.86×** (−92.2 %) | |

The cost is almost entirely in the arithmetic, not the registers. Recomputed from those two
rows the gain is 14.3973×; the projection quotes 14.396×, a +0.009 (0.06 %) rounding
difference that is recorded rather than absorbed.

## The instance is frozen

`verify_submission_dqi.py` fixes the problem: `DEFAULT_M = 4095`, `DEFAULT_N = 70`,
`DEFAULT_DEGREE = 12`, `DEFAULT_R = 2016`, `l = n // 2 = 35`. `ProblemSpecDQI` is not
submitter-controlled. So a better code does not optimise this benchmark — it changes the
problem, and the verifier scores the frozen tuple or nothing.

## Screening

| # | Result | Kind | Verdict |
|---|---|---|---|
| 1 | Sphere-packing upper bounds to the Cohn–Elkies threshold | upper bound | cannot reduce cost |
| 2 | Exponentially improved bounds on binary/spherical code size at fixed minimum distance | existence bound | **analyzable, still not cost-reducing** |
| 3 | Existence of non-sofic groups | construction | cannot reduce cost |
| 4 | Disproof of Connes's rigidity conjecture | disproof | cannot reduce cost |
| 5 | Arithmetic-circuit lower bounds for the permanent (`n⁴/log n` formula bound) | lower bound | **opposite direction** |
| 6 | Exponential parallel repetition for two-player quantum games | theorem | cannot reduce cost |
| 7 | Poly-factor hardness of approximation for CVP | hardness | **opposite direction** |
| 8 | Ehrhart's volume conjecture, all dimensions | exact determination | cannot reduce cost |
| 9 | Superexponential lower bound for multicolor triangle Ramsey numbers | lower bound | **opposite direction** |
| 10 | Compactness and degeneracy in extremal graph theory | conjecture resolution | cannot reduce cost |

**Cost-reducing: 0 of 10.**

The three "opposite direction" rows are the instructive ones. Arithmetic circuit complexity is
the closest field in mathematics to the question this benchmark asks — how few arithmetic gates
does a computation need — and its result is a *lower* bound: a statement that a class of
computations cannot be done more cheaply. No lower bound on any circuit family reduces the
Toffoli count of one specific compiled decoder. The same applies to CVP hardness and the Ramsey
lower bound: a barrier is not a tool.

## The one row worth keeping

Row 2 is the only result that names the object DQI consumes — a Reed–Solomon code over
GF(2¹²). A bound on code size at prescribed minimum distance could bound the *achievable*
failure rate for the frozen parameters, i.e. settle whether `l = 35` correctable errors sit
inside the true correction radius. That puts a **floor** under `R(eps_plus)`. It cannot lower
the score, cannot change `Q`, and cannot change `T_mean`.

It stays in the screen as `analyzable_not_cost_reducing` with the evidence it would need:
the bound itself read from the manuscript (not a summary), a demonstration that it applies to
RS codes over GF(2¹²) at these dimensions, and an independent re-derivation. A coding-theory
bound quoted out of context is exactly the claim class this repository refuses to accept on
trust.

## What would move Benchmark II

1. A decoder circuit performing the same Proos–Zalka pass set with fewer Toffolis per shot.
   `T_mean` must fall to 7.78 % of reference (12.86×); `Q` only to 94.5 % (1.058×).
2. A register-recycling scheme that survives a gate-for-gate equivalence argument, to take
   `Q` from 1,893 to 1,789. The candidate was priced and rejected: it is a target, not a
   result (`dqi_gluzdov_v6.py`, `RECYCLING_APPLIED_IN_CIRCUIT = False`).

Neither is a claim type either OpenAI release contains. There is no gate-synthesis family in
the October catalog and none among these ten.

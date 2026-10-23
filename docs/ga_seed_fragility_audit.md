# Native-GA capability pins: seed-fragility audit (2026-10-22)

**Question.** Which capability pins in the test suite are statements about the
engine, and which are statements about one lucky seed at a budget too small to
reach the claim?

**Method.** Every assertion was re-run with the pinned configuration unchanged
(islands, island size, generations, flags) across seeds 0-9, `strict_native_ga`
on so a Python fallback cannot satisfy the predicate, and the assertion's own
predicate evaluated per run. A pin is *fragile* when it passes on some seeds and
fails on others; a deterministic engine still gives per-seed results, so the
sweep varies the only stochastic input the test has.

Reproduce with the sweep in the commit that added this file; each row's data and
predicate are copied verbatim from the test it audits.

## Results (kernel rebuilt 2026-10-22: tunable, birth-fitted Pow exponent)

| # | Pin (test) | Configuration | Predicate | Seeds 0-9 | Fragile |
|---|---|---|---|---|---|
| A | `test_dyn_rust_extended_ops::test_native_ga_discovers_abs_and_log_with_misc_enabled` (abs) | 2x30x15 | `"abs"` in best expression | 10/10 | no |
| B | same test (log) | 2x30x15 | `"log"` in best expression | 10/10 | no |
| C | `test_dyn_rust_domain_ops::test_native_ga_generates_min_max_and_special_functions` | 2x30x15 | min train MSE < 0.01 | 10/10 | no |
| D | `test_dyn_rust_domain_ops::test_native_ga_discovers_advanced_ops` (atan half) | 2x30x15 | `"atan"` in best expression | 10/10 | no |
| E | `test_dyn_native_ga_best::test_native_best_is_minimum_mse_recovery` | 4x50x20 | min train MSE < 1e-6 | 10/10 | no |
| F | `test_dyn_rust_domain_ops::test_native_ga_misc_opt_out_still_runs` | 4x50x20, `allow_misc=False` | min train MSE < 1e-6 | 10/10 | no |
| G | `test_dyn_rust_domain_ops::test_native_ga_discovers_advanced_ops` (Pow half) | **2x30x15** (the pre-2026-10-22 budget) | min train MSE < 1e-6 | **9/10** (17/20 on seeds 0-19) | **yes** |
| G' | same test, current pin | 4x120x80 | min train MSE < 1e-6 | 10/10 | no |
| H | `test_dyn_rust_koopman_vs_ga::test_koopman_vs_native_ga_matched_split` | 2x30x15 | GA holdout R2 > 0.99 | 10/10 | no |
| I | `test_dyn_planck_limitation::test_ga_small_budget_does_not_transfer` | 2x30x15 | GA holdout R2 < 0.5 | 10/10 | no |
| J | `test_dyn_rust_exp_discovery::test_native_ga_with_exp_available_still_misses_planck_at_small_budget` | 2x30x15 | GA holdout R2 < 0.9 | 10/10 | no |

Rows I and J are "must-fail" pins: their risk is the opposite one, an engine
that *accidentally* extrapolates, and neither seed broke that on the current
kernel.

## The one fragile pin

`x**2.5` is the only claim that depended on search luck. Two causes, measured
separately:

1. **The exponent was a menu, not a parameter** (`2`, `3`, `-1`, `0.5`), so
   2.5 could only arrive as a general-Pow exponent subtree or as constant jitter
   drifting 2.0 upward. Recovery at the pinned 2x30x15 budget: **5/10** seeds.
2. **Exponents were not fitted where they were born.** The kernel's polish stage
   only refines the top 3 candidates of each generation, so a power-law candidate
   scored with a random exponent loses to the polynomial rivals and is never
   polished. Directly, polish *can* fit the exponent: `polish_candidate_constants`
   on `x**1.5` against `x**2.5` reaches MSE 3.2e-13 and exponent 2.5
   (`genetic.rs::pow_exponent_tests::test_polish_refines_pow_exponent_to_the_true_value`).

The kernel now draws a constant exponent from the same continuous 0.1 grid as
every other constant and fits it at population construction, for candidates that
carry a Pow node. No exponent value is privileged in the generator. Effect at
the old budget: **5/10 -> 17/20**. The pins for the classic forms (A-F, H-J) are
unchanged at 10/10, and the 196-test `cargo test --lib` suite still passes.

That path is now mutation-protected as well as test-protected: the battery's
MUT-H7 (fixed exponent menu), MUT-H8 (birth fit skipped), MUT-H9 (birth fit with
zero sweeps) and MUT-H10 (polish window collapsed) are each killed. The 196th
test, `genetic::pow_exponent_tests::test_birth_fit_applies_to_the_initial_population`,
was added for MUT-H8/H9: with zero generations and the final polish off, only the
birth stage can fit the exponent, so the pin fails when that stage is disabled.

The test pin keeps the larger 4x120x80 budget (10/10 seeds, ~1 s) rather than the
old one: reachability improved, but not to the point of a deterministic claim at
4.5k evaluations, and a capability pin should not be a coin flip.

## Does birth-fitting the exponent help or cost at benchmark scale?

**Question.** The 5/10 -> 17/20 result above is a small-budget measurement. Does
the same extra work pay for itself on the larger GA benchmarks?

**Method (`benchmarks/pow_birth_fit_benchmark.py`, protocol fixed before either
arm ran).** Two kernel builds differing in exactly one function: `on` = the
shipped `has_pow_const` (birth fit active), `off` = the same tree with
`has_pow_const` returning false. Continuous exponent grid, polish sweeps, seeds,
data and budgets identical. Candidate-evaluation counts are deterministic, so an
arm is reproducible by its counts; the installed `.so` sha256 identifies each
build (`on` 23a50d2e…, `off` fc1f9603…). Wall time on a shared machine is noisy
and is read against a control below, not trusted on its own.

**Cell 1 - matched-compute Gate 4** (`run_five_gate_validation` config
2x30x15, seeds 0-9; the only named benchmark path that runs the native GA):

| Case | Holdout R2 median (on / off) | R2 min (on / off) | Recovered >=0.99 | Evals median (on / off) |
|---|---|---|---|---|
| Rabi | 1.000000 / 1.000000 | 0.999892 / 0.999892 | 10/10 / 10/10 | 2301 / 2016 (+14.2%) |
| Brus | -2.747591 / -1.528032 | -4.060335 / -23.386949 | 0/10 / 0/10 | 2610 / 1984 (+31.5%) |

Neither arm changes a Gate-4 verdict: Rabi is identical, and Brus is the frozen
GA overfit that fails in both (the birth fit raises the worst seed from -23.39 to
-4.06 but lowers the median, -1.53 to -2.75). The cost at this budget is a
14-32% evaluation increase. These R2 values are the corrected metric
(1 - SSE/SST); the first artifact used `1 - var(residual)/var(y)`, which credits
an offset formula with the bias it fails to reproduce and made Brus look as if
it were near R2 = 0.7. See the metric note below.

**Cell 2 - External-60 native adapter** (frozen 300x50x4 config, all 60
problems, seed 42, scored on the PUBLIC validation grid only):

| Metric | on | off | Delta |
|---|---|---|---|
| Median evaluations / problem | 62,400 | 56,094 | +11.2% |
| Total evaluations | 3,774,349 | 3,389,293 | +11.4% |
| Total wall | 17.7 s | 14.2 s | +23.9% this pair (+9.8% in the repeat pair; not resolved, see control) |
| Median validation MSE | 0.000634 | 0.001650 | -61.6% (scale-mixed median) |
| Median validation R2 | 0.99904 | 0.99861 | +0.00043 |

Per-problem paired comparison (same problem, same config): **23 better with the
birth fit, 32 better without, 5 tied or non-finite**; median paired MSE ratio
1.02; 10 problems more than 2x better and 12 more than 2x worse. The headline
median-MSE drop is a scale artefact across problems of different magnitude, not
a paired improvement - by the paired count the accuracy effect at this budget is
a wash.

**Cell 3 - call-graph control.** The synthesis engine and the Landauer harness
were run with the native-GA entry points wrapped by counters:

| Benchmark | Native-GA calls (on / off) | What it actually runs |
|---|---|---|
| Synthesis (`AutonomousModelSynthesizer`, SYNTH_01) | 0 / 0 | Stage-2 templates + Stage-D scipy refine; `RustKernelBridge` only for singularity/conditioning probes |
| Landauer (`run_ga_polynomial`, `run_assisted_rime`) | 0 / 0 | Bespoke numpy polynomial-basis GA + kernel RIME search (`rime_search_json`) |

`benchmarks/matched_compute.py` likewise runs RIME kernels, DLI, OLS and PySR -
no native GA. So two of the three named benchmarks cannot be affected by this
change at all: their zero counts are the measurement, not an assumption. The same
cells also calibrate the wall clock: the synthesis engine makes zero native-GA
calls, yet its wall moved 19% in one repeat pair (33.8 s vs 28.4 s) and 5.6% in
the other, in the opposite direction (30.0 s vs 31.7 s). The External-60 wall
deltas (+9.8% and +23.9% across the two repeat pairs) are therefore inside this
machine's noise and are not evidence of a wall cost. The deterministic +11%
evaluation figure is.

**Verdict.** At benchmark scale the birth fit costs about 11% more candidate
evaluations (14-32% at Gate 4's smaller budget) and buys no measurable accuracy:
paired External-60 accuracy is a wash, and no benchmark verdict moves. It is
kept because it is the mechanism that makes `x**2.5` reachable at 4.5k
evaluations, where the cost is paid once per Pow-bearing birth and the
alternative is 5/10. A budget-aware switch is *not* introduced here: the
measurement says the large-budget arm would lose nothing by skipping the fit,
but it does not say where the crossover is, and an untuned threshold would be a
new tuning surface with no measurement behind it.

Artifact: `benchmarks/pow_birth_fit_benchmark_results.json`. Reproduce with
`benchmarks/pow_birth_fit_benchmark.py --label on|off` on each build, then
`--merge`.

## Birth-fitting trig frequency/phase templates: measured, not shipped

**Question.** The Pow exponent is fitted where the candidate is born. The seeded
trig template (`sin|cos(f*x + p)`) has the same problem: a candidate scored with
a random frequency/phase loses the top-k race, so the pair is rarely refined.
Does applying the same treatment buy anything, and what does it do to the Planck
must-fail pins?

**Method.** Two variants, each against the shipped kernel (`23a50d2e`):
`full` fits both the plain `sin|cos(f*x)` and phase-bearing
`sin|cos(f*x + p)` seeded roots; `phase-only` fits only the phase-bearing form
(the plain form already recovers 11/11 at its pin). Both are root-only: the
seeded template is the whole tree, and fitting nested trig calls paid polish
cost on much of the initial population for no measured gain.

**Planck** (must-fail pin config 2x30x15, seeds 0-9 plus the pinned 42, correct
R2 = 1 - SSE/SST):

| Metric | shipped | full | phase-only |
|---|---|---|---|
| Train MSE median | 0.005049 | 0.008198 | 0.008199 |
| Holdout R2 median | -31.52 | -76.35 | -44.89 |
| Holdout R2 max | 0.132 | **0.546** | **-0.598** |
| Extended (0.05-12) R2 median | -0.525 | -2.025 | -1.204 |
| Extended R2 max | 0.967 | **0.975** | **0.386** |
| Seeds with holdout R2 < 0.5 | 11/11 | 10/11 | 11/11 |
| Seeds with extended R2 < 0.9 | 10/11 | 9/11 | 11/11 |
| Evals median | 4,578 | 11,948 (+161%) | 11,291 (+147%) |

The full variant makes the two Planck must-fail pins FAIL at the pinned seed 42
(holdout 0.546 > 0.5, extended 0.975 > 0.9). It is not a Planck recovery -- the
recovered form is a nested-trigonometric contraption (`nested_trigonometric`
family, amplitude ~110 against data max ~0.5) -- but the numeric no-transfer
claim those pins encode is no longer true at that seed. The phase-only variant
keeps every seed below both ceilings, making the no-transfer margin stronger
than shipped, at the same ~1.5x evaluation cost.

**Trig capability** (pin config 100x25x2, 11 seeds; small 2x30x15, 5 seeds):

| Target | shipped | full | phase-only |
|---|---|---|---|
| `sin(2x)` exact (pin / small) | 11/11, 5/5 | 11/11, 5/5 | 10/11, 4/5 |
| `2cos(1.3x)` exact (pin / small) | 11/11, 5/5 | 11/11, 5/5 | 8/11, 1/5 |
| `3sin(1.7x + 0.4)` exact (pin / small) | 1/11, 0/5 | **3/11, 1/5** | **3/11, 1/5** |
| phase-shifted MSE median | 5.29e-6 | **3.66e-7** | 1.76e-7 |
| Evaluations (phase cell) | 9,049 | 14,758 (+63%) | 13,850 (+53%) |

**Decision: neither variant shipped.** The full treatment triples the Planck
evaluation count, worsens both Planck medians, and breaks two must-fail pins for
no Planck gain; its only capability gain is the phase-shifted cell. The
phase-only variant protects the pins but costs plain-template recovery (cos
11/11 -> 8/11 at the pin, 5/5 -> 1/5 at the small budget) -- it is not a free
narrowing. A 1/11 -> 3/11 gain does not justify either trade. The kernel is
reverted to `23a50d2e`; `cargo test --lib` is 196 and the Planck/trig pins are
green on the reverted build.

Artifacts: `benchmarks/trig_birth_fit_planck_results.json` (full) and
`benchmarks/trig_birth_fit_planck_phase_only_results.json`; harness
`benchmarks/trig_birth_fit_planck_benchmark.py` (`--label before|after` per
build, `--merge`, `--recompute`).

**Metric note (both benchmark scripts).** `_r2` was computed as
`1 - var(residual)/var(y)`, which subtracts the residual mean and credits an
offset formula with the bias it fails to reproduce. It inflated the Planck
seed-42 candidate from R2 = 0.55 to 0.96 and the Brus GA rows from negative to
~0.7. Fixed to `1 - SSE/SST`; both artifacts were recomputed from their recorded
formulas (`--recompute`), and the pow numbers above are post-fix.

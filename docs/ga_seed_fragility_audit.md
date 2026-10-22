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
unchanged at 10/10, and the 193-test `cargo test --lib` suite still passes.

The test pin keeps the larger 4x120x80 budget (10/10 seeds, ~1 s) rather than the
old one: reachability improved, but not to the point of a deterministic claim at
4.5k evaluations, and a capability pin should not be a coin flip.

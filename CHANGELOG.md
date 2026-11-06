# Changelog

All notable changes to APEDP and APEDP-X are documented here. This file follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and released versions
follow [Semantic Versioning](https://semver.org/). Keep unreleased changes under
`[Unreleased]`, grouped by change type, then move them into a dated version
section when releasing. Version numbers apply to individual published artifacts;
the Python project and native kernel currently have independent versions.

No release tags were present in the repository history available when this
policy was established. Dated entries below are retained development notes, not
inferred releases.

## [Unreleased]

### Added
- Release-governance and changelog conventions.
- Security reporting policy, dependency-update automation, and CodeQL analysis.

### Changed
- _Record user-visible changes here._

### Fixed
- _Record user-visible fixes here._

### Security
- _Record security-related changes here._

## Historical development notes

The following dated entries preserve the existing rolling development record.
They were not retrospectively assigned release versions.

---

## [Unreleased] — APEDP-Q: verifier-guided quantum-circuit compilation (Milestone 1, Benchmark I) (2026-10-23)

Adds the first APEDP-Q control-plane layer for the Google Quantum AI
verifiable-circuit-compilation benchmarks. It is deliberately separate from the
trading promotion lifecycle: a proposer-side screen never promotes a candidate,
and only the competition verifier can issue evidence:

- **Hash-pinned contracts (`apedp_dyn/quantum_compilation/contracts.py`).**
  SHA-256 trees for the verifier and bundled `kickmix`, AST-extracted benchmark
  constants (no duplicated threshold literals), sample-artifact pin, and
  `assert_locked()` before every official subprocess. The sample `invsqrt.kmb`
  is byte-reproduced by the reference precision point.
- **Candidate/receipt IR (`candidates.py`) and ledger adapter (`ledger.py`).**
  Candidate ids bind parameters to the contract digest; artifacts are byte
  hashed; screens can only propose or reject. Development-prefix PASS maps to
  `DEVELOPMENT_PASSED` and never to `CONFIRMED_VALID`; a salted full-domain
  slice is one-shot even after a PASS, and a failed final slice burns.
- **Official referee (`referee.py`).** Runs the verifier in its own CPython
  3.13 process with `kickmix` on `PYTHONPATH`, parses its JSON summary, redacts
  the salt from the recorded command, and labels scopes
  `development-prefix` / `full-domain-reduced-shots` / `full-domain-public` /
  `full-domain`. Salt + fixed seed is refused because the seed bypasses the
  hash-derived secret schedule.
- **Benchmark-I proposer/screen (`invsqrt.py`).** Lazy runtime setup, exact
  integer Newton-Raphson error model, phase-margin and rotation-synthesis cost
  estimate, `.kmb` writer, one-coordinate neighbours, and interaction-aware
  refinement over `(delta_bits, product_bits, result_bits)`.
- **Search (`search.py`).** Baseline full-domain screen, interaction-aware
  opening move, coordinate descent, shortlist full-domain re-check, official
  development-prefix verification, and an opt-in final salted run that requires
  a measured `--baseline-full-score`. Defaults: 140 screened points, 3 full
  checks per stage, 512 × 1 development shots.
- **Experiment 02 — QROM piecewise-linear seed proposer (`invsqrt_qrom_line.py`).**
  Adapted from `scripts/nr_qrom.py` in the `apedp_q_alternatives` scaffold.
  Searches fixed-point widths `(index_bits, coef_bits, slope_bits, delta_bits,
  square_bits, product_bits, result_bits)` with an exact fixed-point screen over
  the full domain, then compiles and verifies. The official full-domain
  64-shots/x protocol PASSed candidate `(7,16,12,10,28,27,28)` at
  **1,309,175.605** in 11m49s at 8 workers — reproducing the reported
  "Verifier-Guided QROM" result (1,309,175.61) and beating both the reference
  (2,076,774.23, −37.0%) and the APEDP-Q precision candidate
  (1,941,280.56, −30.0%). Remains above the best corpus candidates
  (Quick Wins 1,144,148; QROM+Tower42 182,600.86), which are not present as
  artifacts in this workspace and therefore could not be reproduced.
- **Measured full-domain evidence (public salt).** Reference precision
  `(14,17,28,28,29)` screens at 2,076,774.23; the search's best
  `(12,17,28,27,28)` screens at 1,941,280.56 (−6.52%, ratio 1.0698) with
  error ratio 0.9251 and 77,808,879 phase-margin units. The official
  full-domain 64-shots/x protocol (200,540,928 shots, no secret salt) PASSed
  both circuits at 8 workers: reference 2,076,774.23 in 11m12s, candidate
  1,941,280.56 in 11m49s. Candidate worst error/tolerance was 0.9251 at
  x=1,119,994; reference 0.4769 at x=2,588,411. The measured scores equal the
  exact-model projections to the last recorded digit. A development-prefix
  512-shot run also PASSes the candidate at 1,929,748.87, but that score is
  not comparable to full-domain because prefix min margins differ. A
  secret-salt full run remains the internal promotion gate; the public full
  run answers the research question without burning a secret holdout.
- **Pins (`tests/test_quantum_compilation.py`, 14):** contract/tamper checks,
  candidate identity, ledger screen/promotion boundaries, one-shot salt
  burning, scope labels, subset coverage, refinement completeness, and an
  opt-in CPython 3.13 integration that also pins byte-identical reference
  artifact reproduction. Full dynamic suite: 132 passed, 1 skipped.
- **Unrelated pre-existing fix required by the suite:** `apedp/evolutionary_search.py`
  no longer shadows the module-level `apedp_kernel` with a function-local
  import inside `evolve()`, which raised `UnboundLocalError` whenever
  `use_native_ga=True` (6 failing `test_dyn_native_ga_best.py` cases).

---

## [Unreleased] — Noise-vs-rediscovery benchmark harness: planted physics laws measured blind across the noise axis (2026-10-23)

Closes the ASSESSMENT §3 fix-order #1 gap — discovery engines had clean-data
anecdotes but no measured answer to "how often is the right structure
recovered as observation noise grows?":

- **Harness (`benchmarks/run_feynman_noise_rediscovery.py`, `audit-only`).**
  18 frozen static-law worlds: 13 embedded engine-grammar families (monomials,
  reciprocals, exponentials, saturating Langmuir/Hill, trig, two-variable
  linear and product interactions) plus the 5 curated Feynman cases imported
  from `apedp.ingestion.external_benchmarks`. Relative target-noise sweep
  σ ∈ {0, 0.01, 0.02, 0.05, 0.10} × repeats, additive Gaussian on the target
  scaled by the clean-target population std (the Feynman-suite convention).
- **Preregistered harness-local gates (not canonical-spec imports):**
  predictive = clean-holdout R² ≥ 0.95; structure = R² ≥ 0.90 on a denser
  probe grid over the declared world ranges — the train-range alias detector
  that separates a lucky in-window fit from a recovered functional form.
  Verdicts: recovered_structure / recovered_predictive_only / failed /
  abstained; abstention never counts toward recovery (known-truth rule).
- **Fixed-design protocol:** design seed depends on (world, rep) only and the
  noise seed on (world, σ, rep), so each noise level shares design points with
  its clean reference and recovery change is attributable to noise, not to
  reshuffled inputs (the `known_truth_challenge` v0.1.4 noise-phase note).
  A first-draft confounded seed scheme (design reshuffled per σ) was caught
  by the regression test math and replaced with this protocol before the
  recorded run.
- **Pluggable blind adapters:** `synthesis` (univariate
  `AutonomousModelSynthesizer` — evaluates the fit by substituting recorded
  parameters into `SymPyModel.sympy_expr` and lambdifying, the engine's own
  evaluation route) and `rime_ad` (multivariate `RimeAdProposer`, dev-ranked
  BIC fan-out through the native kernel; skipped with an explicit reason when
  the wheel is absent). Adding an engine = adding one adapter function.
- **Pins:** `tests/test_feynman_noise_rediscovery_harness.py` (15): gate
  boundary behavior (undefined-R² flat truth never passes; train-range alias
  scores predictive_only), seed separation invariants, protocol constants,
  world-table lamda finiteness over probe ranges, exact replay determinism
  (wall-clock excluded), multivariate `not_applicable` routing, and a
  kernel-gated RIME-AD recovery integration.

---

## [Unreleased] — Grammar-space expansion: typed saturation laws, fitted interior scales, affine denominators, re-minted frozen grammar (2026-10-23)

The search spaces were widened along the axis the audits kept naming — saturating
laws whose reference level is a *fitted* constant, not a DAG leaf — without
touching a single preregistered threshold:

- **Typed saturation vocabulary (`apedp_dyn/grammar/candidate_ir.py`).** New
  term families `langmuir_bind` (`X/(L+|X|)`) and `hill2_bind`
  (`X²/(L²+X²)`) on the existing `saturation` family, plus `Term.scale_parameter`:
  the reference level travels as a second fitted parameter with the state's
  units and is included in `parameter_names()`. The dimensionless `sat_*`,
  `inv_x2` vocabulary is unchanged.
- **Materialization (`apedp_dyn/grammar/mechanical_grammar.py`, `proposer.py`,
  `coupled.py`, `__init__.py`).** Term builders take the scale symbol `SX` and
  emit `X/(SX+|X|)` / `X²/(SX²+X²)`; a candidate that names a scaled term
  without its scale parameter is rejected at construction rather than fitted as
  a bare `1 + X` add (a gate-1 dimensional violation).
- **Fitted interior scales (`apedp_dyn/model/variables.py`,
  `apedp_dyn/fitting/parameter_fit.py`).** `Parameter.role` distinguishes a
  `"rate"` (multiplies a term) from a `"scale"` (interior reference level).
  The linear warm start cannot differentiate the equation for a scale — the
  derivative still contains the symbol and will not lambdify — so warm start
  seeds it from the observed state scale (clamped to its bounds) and holds it
  fixed, and the bounded trajectory fit refines it afterwards.
- **Native search (`rust/apedp-kernel/src/rime/enumerate.rs`).**
  `search_rational_pairs` gains the saturating family `slope*x/(c+x)+intercept`
  with fitted `c > 0`: log-grid shortlist plus golden-section refinement, one
  O(n) linear readout per trial, competing for the same `top_k` and formula
  budget as the existing exp-denominator family. Regression test
  `test_rational_search_reaches_langmuir_saturating_law` pins reachability.
- **Proposer (`apedp_dyn/proposers/rime_ad_proposer.py`).** `_fit_saturating_rational`
  adds the Langmuir/Michaelis–Menten/Hill representation to the RIME-AD fan-out,
  ranked by the same dev-only BIC/MDL rule; the proposer still never promotes
  and never sees holdout.
- **Frozen trading grammar expanded and re-minted
  (`apedp_dyn/sandbox/grammar_space.py`).** New hypothesis `H-SQRTV`
  (`y=a+b*sqrt(V_t)`, concave level-volume response) with `M_P+SQRTV` in
  `EXTENDED_COMBOS`; declared operators widened to `abs`/`log`/`zscore`/`rank`/
  `sqrt`/`tanh` and `+`/`-`/`*`/`/`/`min`/`max`, with `check_sqrt` as the domain
  guard (sqrt requires `x >= 0` on finite entries). Hypothesis/operator
  declarations are part of the frozen grammar fingerprint, so this is a mint,
  not an edit: **`APEDPX_FROZEN_v1_10.json` supersedes v1_9** with a recorded
  `remint_log` for `grammar_space.py` and `grammar/candidate_ir.py`,
  `blind_benchmark.FROZEN_RECORD` and the `frozen_spec.py` pinned hash
  re-pointed. Thresholds, env pins and blind-case expectations are unchanged;
  no blind case imports the changed grammar files. Verified: 15/15 guard files
  match, 9/9 registry pins match, and `tests/test_dyn_blind.py` +
  `tests/test_dyn_grammar_space.py` + the two saturation-grammar suites pass
  33/33.

---

## [Unreleased] — Mutation coverage for the birth-fit predicate and the harness entry points; warm-run cache (2026-10-23)

The battery now kills the exponent and enforcement mechanisms at the predicate
and harness level, and no longer pays the same price on every run:

- **MUT-H11 mutates `has_pow_const` itself.** Birth fitting was pinned at its
  call site (MUT-H8) and its sweep count (MUT-H9), but not the predicate that
  decides which candidates get fitted: H11 classifies a constant exponent as a
  non-tunable variable exponent, so no Pow is ever fitted. The kill lands exactly
  on `genetic::pow_exponent_tests::test_birth_fit_applies_to_the_initial_population`
  (195 passed / 1 failed, "no birth-fitted Pow candidate in 800 seeds").
- **MUT-H12/H13 cover the benchmark entry points that arm the process-wide
  rule.** `run_synthesis_benchmark.run_benchmark()` and
  `Synthesis50TournamentRunner.run_all()` build the synthesis engine and the
  competitor adapters internally, so a `SearchConfig` flag cannot reach them —
  only `enforce_native_search(True)` can. The new AST pin
  `test_benchmark_entry_points_arm_process_wide_enforcement` requires the call to
  be present and to precede the first search/bridge construction in each entry
  point; deleting the call fails the pin.
- **Honest cost finding: the cargo mutants were not the expensive part.** The
  eleven rust mutants account for ~62 s of pin time; the pytest pins dominate
  (`test_dyn_discovery_dispatch_wiring.py` alone is ~57 s, and three mutants run
  it). Caching only cargo would have saved ~1 min of a ~7 min run.
- **Kill-result cache (`benchmarks/mutation_battery_cache.json`).** A killed
  mutant is reused only when the mutation bytes, the pristine file, the pin argv,
  the content hash of the pin-input tree (every `.py` source, `rust/**`,
  `specifications/**`, pytest config) and `rustc --version` all match; the
  pristine sanity pins still run on every invocation, each reuse is flagged
  `cached: true` with its recorded verification time, and `--no-cache` forces a
  full re-verification. Measured: full fresh run 449 s with 37/37 killed, warm run
  121 s with 37/37 killed (37 reused, 0 re-run).

Report: `benchmarks/mutation_battery_report.json` (ALL_KILLED, 37/37, all
restorations byte-identical). Target list and method:
`scripts/run_mutation_battery.py`.

---

## [Unreleased] — Trig-template birth fitting measured and not shipped; R2 metric corrected (2026-10-23)

Applying the Pow exponent's birth-fit treatment to the seeded trig template was
measured in two variants against the shipped kernel (`23a50d2e`): `full` fits
both `sin|cos(f*x)` and `sin|cos(f*x + p)` roots, `phase-only` fits only the
phase-bearing form.

- **Full variant: capability for cost, and it breaks the Planck pins.** The
  phase-shifted pin improves (exact 1/11 -> 3/11; median MSE 5.29e-6 -> 3.66e-7)
  and plain sin/cos stay 11/11, but evaluations rise ~90-115% on those cells and
  +161% on the Planck pin budget, where both medians get worse (holdout R2
  -31.5 -> -76.4, extended -0.52 -> -2.03). At the pinned seed 42 the GA's
  holdout R2 moves to 0.546 and extended to 0.975, so
  `test_native_ga_hallucinates_on_planck_while_rime_abstains` (< 0.5) and
  `test_planck_equations_on_fresh_out_of_box_data` (< 0.9) both fail. The form
  is still a nested-trigonometric fit, not a Planck recovery; the numeric
  no-transfer claim the pins encode is what stopped being true.
- **Phase-only variant: pins safe, plain recovery lost.** Every seed stays below
  both Planck ceilings (holdout max -0.60, extended max 0.39) at ~1.5x
  evaluations, but `2cos(1.3x)` drops from 11/11 to 8/11 at the pin config and
  5/5 to 1/5 at the small budget, so the narrowing is not free.
- **Not shipped.** A 1/11 -> 3/11 gain does not pay for a 2.5x evaluation
  increase, worse Planck medians, or a plain-template regression. `genetic.rs`
  is reverted to `23a50d2e`; `cargo test --lib` is 196 and the Planck/trig pins
  are green on the reverted build. Both experiments remain as artifacts
  (`benchmarks/trig_birth_fit_planck_results.json`,
  `..._phase_only_results.json`) with the harness
  `benchmarks/trig_birth_fit_planck_benchmark.py`.
- **R2 metric bug fixed in both benchmark scripts.** `_r2` computed
  `1 - var(residual)/var(y)`, which subtracts the residual mean and credits an
  offset formula with the bias it fails to reproduce. It inflated a Planck
  candidate from 0.55 to 0.96 and made the Brus GA rows look like R2 ~ 0.7
  instead of negative. Corrected to `1 - SSE/SST`; both committed artifacts were
  recomputed from their recorded formulas (`--recompute`). Corrected Brus
  numbers: median -2.75 (on) / -1.53 (off), worst seed -4.06 / -23.39; the
  earlier entry's 0.68 / 0.75 and 0.287 / 0.606 were metric artifacts.

---

## [Unreleased] — Mutation coverage for the exponent and enforcement paths (2026-10-23)

The mutation battery now covers the code added in the two preceding commits, so
"the pin passes" is backed by "the pin fails when the mechanism is broken":

- **Ten mutants added (MUT-H1..H10), battery reports ALL_KILLED 34/34** with all
  restorations byte-identical. Enforcement side: the process-wide switch
  reporting off, the bridge swallowing a native failure under enforcement, the
  requested-fallback suspension removed, `strict_native_ga` without
  `use_native_ga` running Python, strict mode falling back on a missing kernel,
  and fallback candidates tagged `kernel_path="rust"`. Exponent side: the fixed
  menu reintroduced, the birth fit skipped at the call site, the birth fit run
  with zero sweeps, and the constant-polish window collapsed by 1e-9.
- **New pin for the birth fit:**
  `genetic::pow_exponent_tests::test_birth_fit_applies_to_the_initial_population`
  runs `population_size: 3`, zero generations and the final polish off, so the
  returned MSE is the birth stage's own work; it sweeps seeds until an exact
  `x**2.5` candidate exists. `cargo test --lib` is now **196 passed**.
- **The battery also sanity-runs the enforcement pin group and `cargo test
  --lib`** before any mutation, so a misconfigured pin group aborts as
  CONFIG_ERROR instead of manufacturing kills.
- **Honesty trail #4: MUT-H7 survived the first run.** Not because the generator
  was menu-bound, but because the distribution pin sampled trees with the
  general-Pow arm enabled — a shallow general-Pow exponent is itself a
  `Const` leaf drawing from the continuous grid, which masked a menu-only
  constant-exponent arm. The pin was SHARPENED (`allow_advanced=false` sampling
  plus a minimum-sample guard), never weakened; the re-run kills all 34.

Report: `benchmarks/mutation_battery_report.json`. Target list and method:
`scripts/run_mutation_battery.py`; register entry in `docs/SYSTEM_FLAWS_AUDIT.md`
(REG-MUT).

---

## [Unreleased] — Birth-fitted Pow exponent: measured cost and benefit at benchmark scale (2026-10-23)

The exponent birth fit was added on a small-budget measurement (x**2.5 recovery
5/10 -> 17/20 at ~4.5k evaluations). A preregistered A/B now measures what it
costs and whether it still pays at benchmark scale, by compiling two kernels
differing in exactly one predicate (`has_pow_const` live vs false) with identical
grid, polish, seeds and budgets:

- **Cost is deterministic: +11% candidate evaluations.** External-60's native
  adapter at its frozen 300x50x4 config (all 60 problems) used 3,774,349
  evaluations with the fit vs 3,389,293 without (+11.4%; median per problem
  62,400 vs 56,094). At Gate 4's 2x30x15 budget the increase is larger because
  the birth work is a bigger share: +14.2% (Rabi) and +31.5% (Brus) on median
  evaluations. Wall time is not independently resolved: the synthesis control
  below, which provably makes zero native-GA calls, moved 19% between arms in
  one repeat pair (and 5.6% the other way in the other), while External-60 wall
  moved +9.8% then +23.9% across repeat pairs — all inside this machine's noise.
- **Benefit at scale is a wash.** External-60 paired on validation MSE: 23
  problems better with the fit, 32 better without, 5 tied/non-finite (median
  ratio 1.02; 10 problems >2x better, 12 >2x worse). The scale-mixed median
  validation MSE falls 61.6% (0.001650 -> 0.000634), which is an artefact of
  mixing problems of different magnitude, not a paired gain. Gate 4 does not
  move: Rabi R2 is identical (10/10 seeds >= 0.99) and Brus fails in both arms
  (the known GA overfit; corrected metric: worst seed improves -23.39 -> -4.06,
  median falls -1.53 -> -2.75; the first artifact's 0.287 -> 0.606 came from the
  `_r2` bug described in the entry above).
- **Two of the three named benchmarks cannot be affected, measured not assumed.**
  Counters wrapped around `RustKernelBridge.evolve_native` and
  `evolutionary_search.evolve` report **0 calls** for the synthesis engine
  (Stage-2 templates + Stage-D scipy refine; the bridge is only used for
  singularity/conditioning probes) and **0 calls** for the Landauer harness (its
  "GA" is a bespoke numpy polynomial-basis selector; its search side is kernel
  RIME via `rime_search_json`). `benchmarks/matched_compute.py` likewise runs no
  native GA.
- **No code change and no budget-aware switch.** The measurement says a
  large-budget arm would lose nothing by skipping the fit, but it does not
  locate the crossover, and an untuned threshold would be a new tuning surface
  with no measurement behind it. The fit stays unconditional.

Harness: `benchmarks/pow_birth_fit_benchmark.py` (audit-only; `--label on|off`
per build, `--merge` to consolidate; each arm records its installed `.so`
sha256). Artifact: `benchmarks/pow_birth_fit_benchmark_results.json`. Full table:
`docs/ga_seed_fragility_audit.md`.

---

## [Unreleased] — Silent native-GA fallback made refusable; core-side Gate-4 provenance (2026-10-22)

Two gaps found while auditing the engine-tournament demo (measured, not
inferred in either case):

- **`evolve(use_native_ga=True)` could downgrade to the Python loop silently.**
  An unavailable kernel skipped the native branch and an exception inside it was
  logged at DEBUG, so a caller received a valid-looking `SearchResult` for a
  search whose cost class was different: measured on 2,000 rows / 1,629
  evaluations, 0.08 s native vs >180 s (killed) in Python. New
  `SearchConfig.strict_native_ga` raises `NativeSearchUnavailableError` (exported)
  on both paths; `strict_native_ga` without `use_native_ga` raises `ValueError`
  instead of quietly running Python. Default behaviour unchanged (fallback still
  happens, still logged). Pins: `tests/test_dyn_native_ga_best.py` (including a
  pin that the fallback path is never reported as `native_rust`).
- **`adjudicate()` consumed `provenance["gate4_status"]` that only
  `apedp_dyn/sandbox/` produced.** Every non-test writer of that key lived under
  the demo harnesses (152 non-test references, producers all sandbox), so the
  central referee was reachable only by hand-assembling gate evidence. New
  `apedp_dyn/epistemic/gate4_provenance.py` measures the holdout record from a
  `CandidateProposal` (`measure_proposal`: MSE/RMSE/MAE/R²/residuals), emits the
  same key set the harnesses emit, marks `passed` only on a demonstrated
  Diebold-Mariano win at the canonical alpha against a supplied incumbent, and
  bridges to `ScoredCandidate` so `adjudicate([...])` runs with no sandbox. The
  noise scale is never invented, and free-form proposal formulas are recorded as
  `canonicalisation: untyped` rather than being given a grammar IR they do not
  have. Pins: `tests/test_dyn_gate4_provenance.py` (7 tests).
- **Benchmark harnesses could record a Python-loop result as a native one.**
  The harnesses ran with `use_native_ga=True` but no strictness, so on a
  kernel-less machine they timed and scored the fallback under a native label.
  `benchmarks/run_langmuir_referee_demo.py` and
  `benchmarks/run_five_gate_validation.py` now build their `SearchConfig` with
  `strict_native_ga=True`, and `benchmarks/external_60_adapters.py` (which calls
  `RustKernelBridge.evolve_native` directly, bypassing `SearchConfig`) abstains
  when any returned candidate reports `kernel_path != "rust"` instead of
  scoring the fallback search as the RIME-AD engine. Pin:
  `tests/test_dyn_native_ga_best.py::test_benchmark_harnesses_refuse_the_python_fallback`
  scans every `SearchConfig(` under `benchmarks/`.
- **`test_native_ga_discovers_advanced_ops` was a seed lottery, not a red
  capability.** A power's exponent was a fixed menu ({2, 3, -1, 0.5}) and was
  never fitted where the candidate was born, so `x**2.5` could only arrive as a
  general-Pow exponent subtree or by constant jitter drifting 2.0 upward; exact
  recovery at the pinned 2 × 30 × 15 budget (~4.5k evaluations) measured 5/10
  seeds, with seed 123 among the misses. `genetic.rs` now draws a constant
  exponent from the same continuous 0.1 grid as every other constant and fits it
  during population construction for Pow-bearing candidates, so no exponent
  value is privileged (same budget: 17/20; the classic-form pins stay 10/10).
  The test pin runs 4 × 120 × 80 (10/10) with `strict_native_ga`, so the
  assertion tests the capability and cannot be satisfied by the Python loop.
  Every GA capability pin was re-swept across seeds — the full table, including
  the pins that are NOT seed-sensitive, is `docs/ga_seed_fragility_audit.md`.
- **A benchmark path that bypasses `SearchConfig` could still measure Python.**
  `apedp.rust_bridge.enforce_native_search()` is a process-wide switch honoured
  by both `evolve()` and `RustKernelBridge.evolve_native`, so an engine that
  builds its own bridge cannot silently downgrade; the synthesis harnesses turn
  it on. An explicitly requested fallback (`force_fallback=True`, which the
  rust-vs-python sweeps measure on purpose) stays legal: the bridge suspends the
  switch only for that nested call.
- **`RustKernelBridge.evolve_native`'s Python fallback had never worked.** It
  unpacked `SearchResult.pareto_front` as `(expr, mse, complexity)` triples, but
  the entries are dicts, so the fallback ran the whole search and then raised
  `ValueError: invalid literal for int() ... 'complexity'`. It now reads
  `entry["expression"]`/`["train_mse"]`/`["complexity"]`; pinned by
  `test_enforcement_refuses_fallback_from_the_bridge`.

---

## [Unreleased] — Profiling artifacts regenerated on the current tip (first live cargo run this session) (2026-10-22)

Both kernel profiling harnesses were executed on `f7be8c1` in the release
profile, replacing tracked artifacts that described earlier source-tree
generations and superseding the pre-merge stash copies (which stay stashed):

- `cargo test --lib -q`: **193 passed** (exit 0) — the first live in-session
  cargo verification of the merged Rust tree, closing the prior
  source-text-pins-only caveat.
- `tests/rime_profile.rs` → `benchmarks/kernel_profile_results.json`:
  bookkeeping ≈94% of per-candidate cost at n=150 (32.9 µs), graph
  evaluation dominating at n=12 000 (176.7 µs vs readout fit 114.6 µs).
- `tests/sre_tree_vs_arena_profile.rs` → `benchmarks/sre_tree_vs_arena_baseline.json`:
  same workload universe (169 candidates × 60k rows ×2 loops, seed 20261022),
  parity 0 mismatches, conversion coverage unchanged (10.06%); throughput
  tree 67.25 ms vs arena 78.75 ms → arena 0.854× TREE-FAVORED (earlier
  same-seed run recorded 0.51× at 403/784 ms — disclosed drift, gate
  conclusion unchanged; recorded in
  `docs/rust_migration_decision_record.md`).
- Disclosure: revision-note numbers in
  `docs/rust_migration_decision_record.md` and the
  `docs/DOCUMENTATION.md` rime-profile row were updated to the fresh
  measurement WITH the drift noted inline; no silent rewrite.

## [Unreleased] — Fitness penalty made scale-free: trading constant-collapse root cause fixed (2026-10-22)

Forensic verification (bit-exact reproduction in-session) established that
`MultiVariableSRE.fitness`'s Occam term `mse + 1e-4*complexity` is
calibrated for O(1)-variance targets and is NOT scale-free: on the tutorial
microstructure price-impact data (`var(dp)` ~ 2.5e-3, true linear signal
R² ~ 1.6%) the two-node penalty (2e-4) exceeded the entire MSE gain of the
correct `lambda*ofi` model (4.0e-5, ratio 4.96x), so the search returned
the sample-mean constant `-0.0005104...` REGARDLESS of budget (proven
budget-independent: identical constant at generations=60/pop_size=200,
and the true model is hard-seeded into the initial population).

- Fix: the penalty is now multiplied by `min(var(y), 1.0)` — bit-identical
  legacy behavior for var(y) >= 1 targets (exact scale == 1.0),
  signal-proportional for smaller-variance ones; degenerate/zero-variance
  targets keep the legacy absolute penalty.
- Pins: `tests/test_dyn_trading_constant_collapse_pin.py` (5) — the
  notebook-exact simulation must NOT return a bare constant (budget- and
  variance-regime invariant, with documented 1% OLS slack only for the
  engine's legacy coefficient snapping), legacy scale == 1.0 for var >= 1,
  scale == var(y) below the bound, and the degenerate-target fallback.
- Verified: AGENTS.md dynamic suite 110 passed; discovery-adjacent suite
  171 passed (incl. the suite's exact-semantics fitness tests and all
  seeded discovery-outcome pins); notebook-adapter suite 30/30.

## [Unreleased] — Tutorial notebook: full engine-tour section (GA/GP, RIME-AD, ODE, PDE, dispatch, all graph surfaces) (2026-10-22)

The tutorial previously demonstrated only Hawkes/microstructure, basic
RIME, and the adjudication/ledger path. New "Step 9: Engine Tour"
sections mirror the pinned APIs of the rest of the discovery surface:

- 9.1 automatic routing — `dispatch_frame` on kinetic, event-stream, and
  ambiguous frames (fail-closed abstention pinned at lowercase enum values);
- 9.2 GA/GP — native `evolve`/`SearchConfig` with the min-MSE-best pin;
- 9.3 RIME-AD — `RimeAdProposer(target_col=...)` end to end;
- 9.4 coupled ODE — `CoupledOdeProposer(dt=...).fit_and_propose(df)`;
- 9.5 PDE — `discover_pde` with the FROZEN-coefficient holdout contract,
  pointers to `discover_pde_2d` and `pde_transfer`;
- 9.6 all three graph surfaces — `ProvenanceGraph`, `plot_epistemic_graph`
  PNG, and the `apedp_kernel.rime_rewrite_json` PyO3 FFI.

Executed proof: the full notebook-adapter suite runs every cell —
30/30 passed including the structure+execution pin; one enum-value
assertion corrected at cause (`Modality` string values are lowercase).

## [Unreleased] — Governance re-certification ceremony UNBLOCKED (structural gap addressed, 2026-10-22)

The one measured structural gap — the governance manifest desync that could
only be described as "trustee-blocked" with no formal procedure to resolve —
now has its mechanism:

- `scripts/run_governance_remint_ceremony.py` (audit-only, fail-closed):
  `--status` reports the desync and the outstanding quorum exactly;
  `--request` mints the certification request anchored to
  `canonical_spec.py`'s import-time pin (refuses on pin disagreement);
  `--collect` records a trustee signature ONLY after verifying it against
  the manifest's REGISTERED public key (forgeries and unregistered signers
  are refused; keys come from the environment, never from the repo, and are
  never generated or stored here); `--mint` re-verifies every collected
  signature from scratch, extends the transparency log with
  `PREREGISTRATION_RECERTIFICATION` entries through the production hashing
  API, verifies the draft with the production verifier against the disk
  spec, and writes the manifest only on full quorum.
- Boundaries preserved per AGENTS §1.2/§1.3: bypassing, local re-minting,
  fabrication, and spec-file mutation are all structurally impossible in
  the tool; the live red remains red until real external trustees run the
  ceremony.
- Pinned in `tests/test_dyn_governance_remint_ceremony.py` (5):
  sandbox end-to-end re-certification (2-of-3, roster/spec_id carried
  through, log chain verified), forgery/unregistered/stale refusals,
  below-quorum refusal, pin-drift fail-closed, and the live-repo guard
  asserting the repository cannot be minted without real trustees.
- FLAW-01 status in `docs/SYSTEM_FLAWS_AUDIT.md` updated: mechanism
  unblocked; only the external signature ceremony is outstanding.

## [Unreleased] — Margin theorem ADJUDICATED into the certified boundary (2026-10-22)

Promotion-checklist item 2 of the CBF proximity-backlog gate is now closed
at theorem level:

- Deciding layer: `benchmarks/run_margin_theorem_referee_decision.py`
  (audit-only, fail-closed) re-reproduces the packet's 6/6 machine checks
  fresh at HEAD, verifies all evidence pointers against disk, and applies
  the preregistered criteria C1–C5 mechanically.
- **Verdict: ACCEPTED_INTO_CERTIFIED_BOUNDARY** (5/5 criteria, 0
  refusals, evidence fingerprint fdfda1611e452495…) recorded durable in
  `benchmarks/margin_theorem_referee_decision.json`; the packet's decision
  slot is stamped from the record on every rebuild (merge hook in the
  packet builder; the identity fingerprint excludes the slot itself).
- Scope: the verdict grants the THEOREM's acceptance only. Checklist
  items 1 (N ≥ 256 production requirement) and 3 (scipy gate / Rust port)
  remain open; the restricted certified path stays `experimental`, wired
  to nothing; the full all-pairs fail-closed re-check remains the
  revocation trigger in every release.
- Durability: the record is never silently overwritten — re-adjudication
  requires `--re-adjudicate --reason` and preserves the prior verdict.

## [Unreleased] — SRE gate instrumented: tree-vs-arena baseline recorded, port DEFERRED (2026-10-22)

The Rust-migration decision record's rank-1 item ("port `genetic.rs` trees
onto the rime arena `Graph`, Rust→Rust") was supposed to be evidence-gated
but had no tree-vs-arena measurement. Instrumented:

- `tests/sre_tree_vs_arena_profile.rs`: deterministic GP workload (169
  candidates from the production `GeneticOps::random_tree` builder at the
  SRE arithmetic flags, 60k rows ×2 vars), semantics-identical
  tree→arena conversion (IEEE-exact Sub→Add∘Neg, parity-gated bit-exact
  incl. NaN on the covered subset), throughput on the identical candidate
  set, artifact written to `benchmarks/sre_tree_vs_arena_baseline.json`.
- **Measured (gate: TREE-FAVORED)**: arena 0.51× of the tree path (403 ms
  vs 784 ms) on unshared GP trees; the semantics-identical conversion
  covers only 10.1% of the pool (tree Div/Min/Max/powf/unbounded
  transcendentals have no drift-free arena counterpart — tree `a−b` maps
  exactly as `Add∘Neg`, which is what parity is pinned on). The 222×
  figure is Python-vs-Rust, NOT tree-vs-arena.
- Decision record §4 rank 1 revised honestly: **evaluation-speed case for
  the Rust→Rust port is measured-NEGATIVE**; revisit only with a measured
  structural-sharing/hash-consing upside. `MultiVariableSRE` keeps
  delegating via FFI to the native tree engine.

## [Unreleased] — Blocker-sweep repairs: dashboard PDE-transfer regression + stale E5/E4-T1 pins re-minted (2026-10-22)

Release-blocker sweep on current HEAD (bisection-verified causes):

- **`pde_transfer.py` FLAW-05 regression repaired at the cause**: the soundness
  guard had been inserted as early `return`s that bypassed the
  `FrozenTransferResult` construction (bare-float path — callers expecting
  `.r2` got AttributeError; the dataclass return was dead code). The guard's
  semantics now live in the `r2` VALUE, never the return TYPE.
  `tests/test_dyn_pde_transfer.py` 4/4.
- **Dashboard contract fixture re-minted**: after the lawful pool widening
  (Gate A dev-only per the §1.2 anti-snooping rule + symmetric AST
  enumeration in `0eb7a04`), the duffing fixture at noise=0.05 honestly
  abstains (`multi_hypothesis`) and must not be test-forced — noise=0.01
  restores the Δ/τ≈1.3 unique path the contract verifies.
- **E5 / E4-T1 diagnostics re-derived and re-minted (ERR-INC-74-79, disclosed
  in `docs/TRADING_EQUATION_FORENSICS.md` §80c)**: the §74–79 signatures had
  been established under the buggy backward-windowed d5 target (F1, fixed in
  `68a3dadb`) and pre-A01 leaky train blocks (fixed in `9a73170`). Corrected
  harness re-derivation (deterministic, verified twice): market/null E5
  parity replaces the withdrawn **drifting contrast; coupling table splits
  3 stable/3 drifting; §78 deltas re-pinned (M2≈M3 inverted on ETH/AVAX,
  spread capped .08); §79 gates 10/6/8/1/0/0 with ETH as the one pinned
  marginal permutation survivor (.0498); §80 incremental battery re-minted
  again under the fully corrected harness (permutation NULL everywhere,
  min ETH .0846 > α; BTC HA sign-flip 9/11 pinned)
  (`tests/test_dyn_e5_market.py`, `test_dyn_e5_r1.py`, `test_dyn_ensemble.py`,
  `test_dyn_gated_ensemble.py`, `test_dyn_incremental.py`).
- **Quantum-vacuum subthreshold re-pin under the F14 referee**: the old
  abstention was a manufactured refusal (train-loss anchor; Δ was negative) —
  under the holdout-anchored pair the saturated residuals separate at
  Δ/τ≈1.09 with a bootstrap-confirmed weak-margin promotion carrying the
  RESIDUAL_STRUCTURE_UNRESOLVED diagnostic (`tests/test_dyn_quantum_sandbox.py`).
- **Governance remaining OPEN (documented, trustee-blocked)**: the manifest
  still seals the pre-v1.8 spec hash (FLAW-01; quorum signatures not
  producible locally). Typecheck scope is the documented 2-file pyproject
  Phase-1 boundary (green).

## [Unreleased] — Gate-limitation closure: spatial index + model-checked proximity lock (2026-10-22)

The gate decision's three recorded limitations are each acted on at the
measured cause (`docs/cbf_proximity_graph_backlog_gate.md` completion
addendum):

- **"Spatial index needed to beat the dense [N,N,2] floor" — measured.**
  `apedp_dyn/robotics/proximity_margin.py` gains `indexed_proximity_pairs`
  (scipy cKDTree, candidate radius r_safe + h* + slack: a SUPERSET of the
  constrained band by construction, so every EXCLUDED pair is provably
  free — soundness pinned, including the exact boundary-equality semantics
  where scipy's `at most r` admits the pair the dense h < h* filter
  excludes) and `indexed_barrier_components` (exact sparse mirror of the
  dense formulas, array-equal pinned). Measured: the index beats the dense
  build floor 3.3–3.6× at N=256 and 3.1–5.8× at N=512 (at N=128 the tree
  overhead has not paid off — recorded as gate information). The COMPOSED
  restricted path (index + restricted alternating projections + FULL
  all-pairs fail-closed re-check) measures 3.31× over the dense solve at
  N=256 (64.0 → 19.3 ms, constraint ratio 0.054) and 3.6× at N=512
  (258 → 71.6 ms, ratio 0.028); at N=128 only 1.31× — the certified
  contract's own all-pairs re-check dominates there, and that re-check is
  the fail-closed honesty floor, deliberately never removed.
- **"Pitch-only, wired to no pipeline" → evidence class completed.**
  `model_check_unicycle_horizon_proximity` realizes the proximity-LOCKED
  commands through the production arc stepper on adversarial horizons
  (head-on top-speed closing; multi-agent closing grids, 2 seeds):
  min h > 1e-3 throughout (measured evidence, explicitly not a formal
  claim). Promotion checklist added to the gate doc (production N ≥ 256
  requirement + referee-adjudication acceptance of the margin theorem
  into the certified boundary + scipy-or-kernel-port decision + the
  all-pairs re-check stays in every release, forever).
- **"Dev-box medians" → variance quantified** over two full reruns of the
  same seeded harness: certified 5 ms gate crossing stable (N=256 twice);
  QP-time noise 5–6% at N ≥ 256, ~31% at N=128 (scheduling dominated); the
  heuristic 50 ms boundary sits exactly at its edge (N=512: 47.5 vs 50 ms
  flipped between runs) — recorded verbatim as the boundary sensitivity;
  thresholds remain decision thresholds, not constants.
- Pins: `tests/test_dyn_proximity_margin_pins.py` 10/10 green
  (index superset + provably-free exclusions, boundary-equality pair,
  array-exact sparse-vs-dense, collision-free horizons, fail-closed
  max_iter=0 refusal, midpoint saturation).

## [Unreleased] — Gate 1 executed: vectorized pair sweep + margin pitch (2026-10-22)

- **Heuristic pair-loop VECTORIZED** (`multi_agent_step` now uses ONE
  `pairwise_repulsive_forces` sweep): EXACT per-pair mirror of the scalar
  rule — degenerate [k_rep, 0] fallback, r_warning zeros filter (the mask
  IS the proximity filter a graph would replicate), head-on tangential
  term at anti > 0.7. Pinned branch-exact per pair AND against an inline
  legacy reference at whole-step level across N = 2/5/8/33
  (`tests/test_dyn_multi_agent_vectorized_pins.py`, 9 tests green;
  pre-existing robotics suites 14/14 green). Re-profiled with the same
  seeded harness: exponent 1.91 → 1.06, 5 ms budget crossing N=32 →
  N=128, 50 ms no longer crossed in-sweep, 26× faster at N=256 — a
  proximity graph is NOT needed for this path (the gate doc carries the
  pre/post tables; certified-path gate numbers unchanged).
- **Non-neighbor barrier MARGIN argument pitched + machine-checked**
  (`apedp_dyn/robotics/proximity_margin.py`, experimental/evidence-only,
  wired to no pipeline): pairs at h ≥ h* = 2·v_max/γ hold hdot ≥ −γ·h
  CONSTRUCTIVELY (box-bounded worst case −2·v_max saturates the law
  EXACTLY at h = h*, never violates); `solve_cbf_qp_proximity` solves the
  alternating-projection QP over the h < h* restricted set and releases
  ONLY through the FULL all-pairs fail-closed re-check (the margin
  algebra re-verified numerically at release); FREE/CONSTRAINED
  bookkeeping re-decided per step from the h matrix the barriers build
  already computes. Measured on the A3-guarded grid: constraint ratio
  0.19 at N=64. This is the soundness prerequisite the certified-path
  N≥256 gate demanded — STILL experimental: no production wiring, and a
  production proximity structure additionally needs a spatial index (the
  dense [N,N,2] floor stands). Pins: `tests/test_dyn_proximity_margin_pins.py`
  (margin formula, midpoint saturation, restricted-solve full re-check on
  randomized fixtures, provably-violated max_iter=0 refusal, constraint
  bookkeeping).
- **Hygiene**: two pre-existing ruff flags in `cbf_certificate.py`
  surfaced by package-level linting fixed at the cause (genuinely-unused
  `import torch` removed after grep confirmation; unused unpack renamed).

## [Unreleased] — CBF proximity-graph backlog gate decision (2026-10-22)

- **O(N²) profile recorded** (`benchmarks/run_cbf_scaling_profile.py`,
  results in `benchmarks/cbf_scaling_profile_results.json`, gate record in
  `docs/cbf_proximity_graph_backlog_gate.md`; status `audit-only`):
  deterministic A3-guarded fixtures; the certified QP's factory-lemma
  1-pass FLOOR isolated from adversarial-iteration growth — the all-pairs
  barriers build + constraint sweep alone costs 9.9 ms at N=256 (43 ms at
  N=512), with uniformly-random commands adding ≈8× more at N≥256; the
  heuristic transport stepper costs ≈6 µs per Python pair invocation and
  crosses the 5 ms real-time budget at N=32 (50 ms at N=128).
- **Gate decision (criteria preregistered in the harness BEFORE
  measuring)**: the heuristic path activates NOW at N≥32 — its first fix
  is VECTORIZED pair-force computation (behavior-identical, no certificate
  soundness question; a k-NN graph adds nothing a distance mask does
  not); the certified path's proximity restructuring activates only for
  production N ≥ 256 real-time certified control AND a soundness proof
  that non-neighbor pairs cannot violate barriers within the assumption
  box (pair-ratio counterfactuals 0.38→0.011 at k=4 recorded as
  computational estimates only — pruning AFTER the dense barriers build
  keeps the O(N²) floor, so a production structure must use a spatial
  index instead). Certified code, pins and battery mutants untouched —
  measurement + gate record only.

## [Unreleased] — Graph census verification and gap closure (2026-10-22)

Response to the external master graph census. Verified ground truth
BEFORE acting; then closed the actionable slots and corrected the record
(`docs/graph_census_2026_10_22.md`):

- **Three "implemented but untested" graph surfaces are now pinned**
  (`tests/test_dyn_graph_census_pins.py`, 5 tests green; census claim of
  zero coverage verified by search first): W3C PROV-lite
  `ProvenanceGraph` (nodes + links verbatim, no shared-mutable-default
  leak), `plot_epistemic_graph` (real deterministic PNG render, magic
  bytes verified), and the `ak.rime_rewrite_json` PyO3 FFI binding for the
  frozen structural rewrites — valid `WrapNonlinearity` appends the
  wrapped node and makes it the output per `rewrite.rs` `apply` semantics,
  a bad operand index is REJECTED (no fabricated dangling references),
  malformed graph/op JSON fails closed.
- **Two census rows corrected as stale/clarified:** the Level-1 state
  dependency graph EXISTS as the DE arm's screening + dense escape hatch
  (never a causal adjacency verdict — kernel `coupled_system.rs` stays
  flat SINDy as the escape hatch); the AD row clarified — exact-rule
  SYMBOLIC service in `de_arm/ad_service.py`, kernel rightly keeps forward
  Taylor Jets, torch stays the reverse-mode runtime.
- **Genuinely unimplemented classes** (robotics proximity graphs, mesh
  PDE discovery, GP-tree→DAG sharing, 2D/3D tensor networks, kernel
  reverse-mode tape) recorded as rationale'd backlog categories, not
  promises — each with the deprioritization reason and its gate.
- **Honest pin-failure note:** the census FFI pin initially expected a
  named-field serde shape for the `Tanh(usize)` variant; the test
  assertion was corrected to the true newtype-variant serialization — the
  FFI matched `rewrite.rs` exactly (cause fixed at the failing assertion,
  not at the pinned boundary).

## [Unreleased] — DE-arm frozen benchmark vs dense flat library (2026-10-22)

- **Frozen benchmark** (`benchmarks/run_de_arm_vs_dense_benchmarks.py`,
  results in `benchmarks/de_arm_vs_dense_benchmark_results.json`, doc in
  `docs/de_arm_vs_dense_benchmarks.md`; status `audit-only`): deterministic
  RK4 fixtures of Lotka-Volterra, FitzHugh-Nagumo and Lorenz (noise-free,
  seeded). Dense kernel and the arm's screened mirror both structurally
  exact on all three systems at the default screen depth (top_k=2, 0 false
  exclusions). Library-reduction hypothesis MEASURED, not assumed:
  ≈10% (Lorenz, 10→9 columns), 25% (FHN), 0% (LV) at D≤3 — matching the
  review's correction that the 90% claim is a hypothesis for higher
  dimensions/richer grammars. The aggressive screen (top_k=1) drops 2 true
  parents on Lorenz with visible R² collapse (0.63, 0.04) — demonstrating
  the dense escape hatch is load-bearing; screen exclusions are reported
  verbatim, never hidden.
- **Fixture-integrity finding (recorded as evidence, not patched away)**:
  a first Lorenz fixture with a narrow slow-manifold window yielded
  identically-zero-residual but NON-physical polynomial fits in BOTH
  engines (R²=1.0 with wrong structure) — noise-free near-collinear data
  makes many column combinations exactly degenerate. The benchmark now
  guards window sweep on BOTH the full window and its train partition.
- **Representational fix** (`de_arm/dependency_screen.py`): the reduced
  mirror library gains kept·kept parent products — without them the
  screened library could not express parent×parent monomials the dense
  engine demonstrably recovers (Lorenz ẏ = 28x − y − xz needs x·z).
  Excluded parents still never enter the screened library (pin holds).
- **Hygiene**: corrupted docstring token in `de_arm/typed_model.py`
  repaired; ruff clean; DE-arm pin suite re-verified after each change.

## [Unreleased] — DE-discovery arm, separate arm boundary (2026-10-21)

Per the RIME–APEDP-X review thread (build the differential-equation
discovery capability as a SEPARATE arm reusing shared infrastructure,
never as new graph nodes inside the RIME core; reviewers' corrections
applied literally — no claimed 90% library reduction, no causal-topology
oracle claim, dense escape always present), phase 1 lands behind a clean
module boundary (`apedp_dyn/de_arm/`, all `experimental`):

- **Typed mathematical semantics** (`de_arm/typed_model.py`): every arm
  candidate declares its kind — STATIC_EXPRESSION / ODE_SYSTEM (states +
  independent variable + dt + initial conditions) / PDE_SYSTEM (fields +
  spatial axes with positive dx + declared boundary treatment). Incomplete
  declarations are fail-closed refusals, so structural checks run BEFORE a
  search, not after it.
- **AD as shared infrastructure** (`de_arm/ad_service.py`): a symbolic
  derivative service over a closed typed expression vocabulary with exact
  product/quotient/chain rules and declared domain refusals; the module
  documents both honest limits — it differentiates the implemented
  candidate computation (a perfect derivative of a wrong hypothesis stays
  wrong) and measured data keeps its existing finite-difference stencils
  for residuals. Fornberg remains valid on irregular grids (correction C:
  no blanket first-order degradation claim).
- **Level-1 dependency screening with a MANDATORY escape hatch**
  (`de_arm/dependency_screen.py`): per-state predictive-increment ranking
  of candidate parents (lag-embedded relative-RSS gains on the training
  partition; deterministic, dependency-free) prunes each state's candidate
  library, but the DENSE kernel result always ships beside the screened
  mirror fits in one referee-facing `OdeArmOutcome` — a parent the screen
  excluded can never silently vanish, and the screen's ranking travels
  with an explicit not-a-causal-verdict disclaimer. The screened fit is a
  documented python ridge+prune mirror whose provenance never claims
  kernel parity.
- **Pipeline/lifecycle integration**: the dispatch surface's coupled-ODE
  route now hands the arm's evidence artifact to the referee path; the arm
  contains no acceptance/promotion/verdict field anywhere (pinned), and
  the referee's frozen protocol stays the sole authority.

Pins: `tests/test_dyn_de_arm_p1_pins.py` (10: typing refusals, AD
exactness incl. the product-rule mutation boundary, screening semantics
with the escape hatch, dispatch integration, referee independence). The
arm's mutants join the mutation battery register (MUT-G1..G3).

---

## [Unreleased] — EHI re-foundation per external review (2026-10-21)

The external architecture critique's four build items and the fifth pillar
landed, each replacing a math shortcut the review named (all `experimental`;
pins `tests/test_dyn_ehi_phase5_pins.py`, 16 mixed behavior+mutation pins;
none of the touched layers decides acceptance — the referee keeps authority):

**Dispatch wiring (follow-up, same review thread):** the sieve + selector now
route proposals AUTOMATICALLY (`apedp_dyn/orchestration/dispatch.py`):
decisive agreement on both layers ⇒ the one engine runs (L2 open discovery /
ODE diff-Padé / frozen-operator PDE / Hawkes MLE); selector ambiguity keeps
every sieve-admitted family live (MULTIPLE_ROUTES, masks honored even with
wired runners); decisive disagreement ⇒ CONFLICT abstention, never an
override; unresolved structure ⇒ UNROUTABLE with open questions. Per-family
failures are recorded in `errors`, never laundered; products carry
referee-facing material only. Pins `tests/test_dyn_discovery_dispatch_wiring.py`
(9). **Two broken seams fixed on the way:** `discover_ode` called the kernel
with raw lists + an int while the seam takes three JSON strings (matching
every other caller — no test had exercised it end-to-end); and a decisive
route is now refused when the winner's DEFINING diagnostic never fired
(pure noise on a time axis can no longer fabricate a PDE claim via
competitive misfit).

- **Model-class selection replaces deterministic routing**
  (`apedp_dyn/orchestration/model_class_selector.py`): P(M|D) over
  documented per-class diagnostic profiles with ambiguity-preserving
  outcomes (`MULTIPLE_CLASSES_PLAUSIBLE`, `INSUFFICIENT_METADATA`); the
  declaration-narrowed class table keeps point processes as point
  processes. **Bug fixed by the pin probe:** the sieve's empty-string axis
  sentinel was treated as a found axis, stamping `time_axis=1.0` onto pure
  isotherms (routing to `ode_kinetic` at bf 1.0); and the class-firing
  penalty now keys on NON-shared diagnostics (`time_axis≈1` no longer rides
  for every dynamic class at PDE/ODE ties) with a documented 2-nat cost.
  A documented aggregation-validity criterion (mean-field amplitude vs
  field spread) keeps a traveling wave's negligible ripple from being
  mis-badged as a kinetic ODE signal while still allowing genuinely kinetic
  2D fields to contest ODE against a firing PDE diagnostic.
- **Constraint STATES replace automatic enforcement**
  (`apedp/dimensional_invariant_engine.py::classify_constraints`): boundary
  conditions are hypotheses with states `KNOWN` / `EMPIRICAL` /
  `UNRESOLVED` / `REJECTED`; training-partition evidence carries
  `is_physical_claim=False`; subsection cases (offset-Langmuir → origin
  REJECTED; sub-critical window → saturation UNRESOLVED, not rejected);
  declared independent laws alone express `KNOWN`.
- **Expected information gain replaces unbounded-gap maximization**
  (`apedp_dyn/epistemic/active_experimentalist.py`): I(H;Y|x) by exact
  Gaussian-mixture quadrature, argmax under feasible ranges / cost / noise,
  field-complete `ExperimentOrder` requests and fail-closed refusal of
  zero-noise models and no-honest-experiment conditions.
- **Fifth pillar: research controller**
  (`apedp_dyn/epistemic/research_controller.py`): belief/governance state
  separation (referee constants fingerprints re-verified before every
  planned action; in-session tamper raises), deterministic planner over
  {inspect, test-constraints, information experiment, improve data quality,
  ABSTAIN}, five-part evidence ledger, controlled exits
  (`ABSTAINED` / `BLOCKED_ON_PROBLEM_SPECIFICATION` /
  `EVIDENCE_INTEGRATED_AWAITING_REFEREE`; one acquisition per session —
  incorporation requires a referee re-adjudication, never self-adjudication),
  and in-run protocol revisions returned as versioned
  `PROPOSED_NOT_APPLIED` artifacts.

---

## [Unreleased] — Level-5 autonomy pillars (2026-10-21)

Four autonomous-scientist pillars landed at the roadmap's named module
paths (all `experimental`; each with a dedicated pin file, 23 pins total,
and the honest disclosure that none of them touches acceptance authority):

- **Pillar 1** `apedp_dyn/orchestration/modality_sieve.py`: autonomous
  modality & topology sieve — structural evidence detectors (isochronous
  monotone time axis with a declared-only irregular-axis contract, bounded
  relative curvature, log-axis 2D spectral dispersion, Fano burstiness)
  classify raw data into static-manifold / ODE-kinetic / PDE-field /
  Hawkes-event modalities and MASK ill-posed engine families before any
  proposer runs; fail-closed UNRESOLVED on conflicting evidence.
  Pins: `tests/test_dyn_modality_sieve.py` (6, incl. the isotherm-vs-
  kinetics discriminator and the Poisson-refusal).
- **Pillar 2** `apedp/dimensional_invariant_engine.py`: autonomous
  dimensional & boundary synthesis — unit→exponent conversion (pint SI
  bases, integral-exponent fail-closed) feeding the audit-proven
  Buckingham-Π engine (exact integer nullspace in
  `apedp_dyn/conf.dimensionless`), log-band empirical limit estimators
  (origin pin q(0)=0 AND independent finite-plateau detection steering the
  rational saturation family; fail-closed on ambiguous curves), and the
  interpolated half-max characteristic scale with the {s*/3, s*, s*·3}
  search bank. Pins: `tests/test_dyn_dimensional_invariant_engine.py`
  (5, incl. pint-verified dimensionless Π and both Langmuir constraints).
- **Pillar 3** `apedp_dyn/epistemic/active_experimentalist.py`:
  closed-loop active experimental disambiguation — x* = argmax pairwise
  divergence − k·σ with k imported from the frozen E4 policy
  (`DEFAULT_DETECTABILITY_K`, no new threshold), deterministic clustered
  target-range acquisition orders, fail-closed out-of-band/NaN point
  rejection (instruments that lie are counted, never clamped), and residual
  referee handoff — no promotion decision inside the layer.
  Pins: `tests/test_dyn_active_experimentalist.py` (5).
- **Pillar 4** `apedp_dyn/epistemic/enclave_referee.py`: confidential
  adjudication — Stage-C pair verdicts computed in a fresh OS subprocess
  over permission-sealed inputs (aggregates only, no shared sys.modules),
  an Ed25519-signed attestation binding verdict + request SHA-256 +
  aggregates, tamper-fail verification, and UNRESOLVED on any
  infrastructure failure (timeout/crash/missing output). Honest
  "process-isolated" executed_by claim — TEE only by explicit runtime
  declaration (`APEDX_TEE_ENCLAVE`), never auto-upgraded.
  Pins: `tests/test_dyn_enclave_referee.py` (7, incl. reproducibility,
  forged-signature tampering, hash-binding failure and fail-closed
  timeout).

Verification: pillar pins 23/23 (`EXIT_STATUS=0`); AGENTS.md dynamic
regression suite green after the DOCUMENTATION.md registry indexing; the
new modules are registered `experimental` in the §18.1 registry with
docstring `Status:` lines matching the tags.

---

## [Unreleased] — External 19-flaw audit remediation (2026-10-21, same day)

An independent adversarial audit catalogued 19 flaws (Flaw severity matrix in
`docs/SYSTEM_FLAWS_AUDIT.md`, 2026-10-21 round). 18 fixed by execution-verified
edits; FLAW-01 remains OPEN by protocol:

- **FLAW-01 (Critical, OPEN — trustee-blocked):** the Ed25519 governance
  manifest still seals the deprecated spec SHA `a6d026…`. Re-signing requires
  trustee private keys held outside the repository; the distributor must not
  bypass the manifest. (Spec centralization for FLAW-13 below follows the
  written §1.3 workflow and does not mint governance claims.)
- **FLAW-02:** `apedp_dyn/ode/proposer.py` scored the identity instead of the
  discovered ODE RHS — every ODE proposal evaluation was `y = x_i`. The
  scorer now evaluates `sum(coeff · basis(x))` over the kernel term encoding
  (measured: RHS-vs-true-derivative RMSE 0.0000 on a probe where the identity
  was 1.67).
- **FLAW-03:** S3 lock takeover was a TOCTOU race; now conditional
  (`IfNoneMatch="*"`, 412 → contested) with a read-back token verification
  fallback that serializes non-conditional backends. v2 (mutation-pin run):
  the parameter-rejection fallback now ISSUES the unconditional put before
  the read-back — the first draft skipped it, so the read-back could never
  see our token and every acquire on old-botocore backends deadlocked to
  timeout.
- **FLAW-04:** uniform-grid stencils in `rime/diff_pade.rs` now gate on actual
  grid uniformity (1e-4) and fall back to Fornberg arbitrary-grid weights.
  v2 (mutation-pin run): the first Fornberg translation was structurally wrong
  (weights stopped annihilating constants; ~1e3x error at stencil edges) —
  `fornberg_weights` now mirrors the proven Python
  `_fornberg_centered_weights` recurrence, pinned by
  `test_nonuniform_grid_derivatives_use_fornberg_fallback` (quadratics
  differentiated exactly on a strongly non-uniform grid).
- **FLAW-05:** the constant-field `R² = 1.0` tautology is removed in
  `rime/pde.rs`, `pde_discovery.py` and `pde_transfer.py` (perfect resorbed
  constant may claim 1.0; real residual energy on a flat target scores 0.0);
  a failed linear `solve` no longer unhides the STRidge significance rescue
  (se = 0 ⇒ rescue disabled, pure relative pruning).
- **FLAW-06:** pole evaluations in the diff-Padé rollout set
  `pole_singularity_hit` and disqualify `is_exact_recovery` instead of
  silently integrating through singularities; Koopman ss_tot is centered and
  zero-variance targets score 0.0. The pole-flag report contract is now
  mutation-pinned on BOTH rollout branches: Bessel-J0 (forces the 2nd-order
  branch, S(0)=0) and the parabola with `max_diff_order=1` (forces the
  1st-order branch, P(0)=0).
- **FLAW-07:** C6 apex evidence set now includes `EQUIVALENCE_RIVALS_REJECTED`
  and `INTERVENTIONAL_INVARIANCE` (tier monotonicity restored).
- **FLAW-08:** re-anchored referees run the block-bootstrap tiebreak on
  provenance-recovered paired residuals; unpairable pairs stay on the
  disclosed point-estimate contract (INCOMPLETE).
- **FLAW-09:** the Hawkes eval-time binary collapse returns the collapsed
  binary vector instead of silently switching to an unbounded z-score; the
  pinned fit-time continuous declaration is untouched.
- **FLAW-10:** core-mode Lean theorems relying on injected axioms are labeled
  `syntactic_schema_check::…` — never presented as independent FORMAL_PROOF.
- **FLAW-11:** opposed one-sided limit probes on an asymptote raise rather
  than averaging a fake 0.0 (previously `limit(1/x, x→0)` timed-path
  returned 0.0); the wall-clock regression retains its budget contract.
- **FLAW-12:** the implicit-RIME exact-RMSE ceiling measures scale over ACTIVE
  non-constant library columns only (`test_near_zero_scale_misfit_not_certified`
  pins the near-zero-scale refusal with its O(1e5) exact positive control;
  MUT-E4 kills the re-included constant column).
- **FLAW-13 (spec centralization per AGENTS §1.3):** preregistered spec gains
  `ode_referee_min_mean_r2=0.95` / `ode_referee_min_worst_r2=0.85`
  (`canonical_preregistered_spec_v1.json` re-pinned:
  `a11fd957…` →
  `5bb42f0341b7a024d21280ee69922bd7dbfebd33cf55506253e3f5248411ec04`;
  rationale: removing hardcoded referee literals per the
  specification-centralization rule; Ed25519 manifest re-certification remains
  trustee-blocked as recorded under FLAW-01); `OdeReferee` defaults bind to the
  spec, and mean holdout R² < 0 earns the new honest `UNSTABLE_DIVERGENT`
  verdict instead of `PREDICTIVE_WRONG_STRUCTURE`.
- **FLAW-14:** one-sided upwind stencils with zero-flux Neumann boundaries
  replace toroidal `np.roll` in the audio spectral advection.
- **FLAW-15:** the Lie-symmetry nullspace gate is normalized
  (`|λ| ≤ tol · λ_max`).
  v2 (mutation-pin run, 2026-10-21): the v1 relative gate left the EIGENSOLVE
  itself scale-blind — on O(1e6)-amplitude ODEs the multi-stratum Gram matrix
  corrupted a structural zero into eigenvalue ~1.0 (basis residual ~7e5, 9
  vs 4 genuine directions reported); `discover_point_symmetries` now solves
  the CORRELATION matrix (kernel-preserving congruence, generators map back
  via D⁻¹v) and `jacobi_eigen` pre-normalizes to unit spectral scale — the
  scaled system now reports the same symmetry verdict as the unscaled one
  (`test_scale_dominant_ode_keeps_genuine_symmetries`). An
  absolute-tolerance-gate mutant was registered and GREW no natural kill
  after this fix (the defect expression it targeted is dissolved by the
  corrected eigensolve); it was removed from the battery with its rationale
  recorded in `scripts/run_mutation_battery.py` rather than shipped as a
  silent survivor.
- **FLAW-16:** Hamiltonian archetype aligned to the pinned
  `CONTINUOUS_FACTOR` contract.
- **FLAW-17:** CBF-QP closes with a final norm-ball projection before the
  fail-closed certificate checks. v2 (mutation-pin run): the plain re-scale
  is provably insufficient (scaling the corrected endpoint shrinks the
  deflection that repaired its pair and re-breaks the barrier); the close-out
  now distributes each remaining deficit BETWEEN ITS ENDPOINTS, each share
  bounded by the endpoint's direction-aware norm-ball headroom, |v| ≤ v_max
  exactly, and runs only when the loop exhausted with pending corrections —
  `max_iter=0` still refuses fail-closed.
- **FLAW-18:** ledger appends are flock-guarded (`LOCK_EX`/`LOCK_UN`, graceful
  degradation without fcntl) and remote auto-sync is amortized in batches of
  16 pending appends with a flush before snapshots.
- **FLAW-19:** dead `np.gradient` removed from the SIR proposer; formula uses
  `i_col` physical notation instead of the internal `I_arr` identifier.
- **FLAW-STRAITA (same-day follow-through; exact-RMSE gate audit):**
  `rime/implicit.rs` still carried two COMPENSATING anchored gauges in
  raw-coefficient space — a raw `max_c = max|c_j|` rescale applied after the
  strata-coherent `c_j = v_j/norm_j` mapping, and a raw `|c_j| > 1e-6`
  active-count threshold that the first gauge had been accidentally inflating
  above threshold. At O(1e6)-scale strata the pair degenerated the genuinely
  pure 1e6 circle to the constant fallback `-1 = 0` (rmse 1.0, refusal) —
  probed before the fix; removing only the first gauge exposed the second
  (`test_large_scale_exact_relation_certifiable` failed at rmse 3.6e-17).
  Fixed with coherent contribution-space gauges: no normalization after
  `v/norm` (the Jacobi eigenvector is already 2-normalized), and membership
  by contribution ratio `|c_j|·norm_j > 1e-6·max_j(|c_j|·norm_j)`. Post-fix
  probes (rebuilt extension): pure 1e6 circle certifies rmse 4.1e-17,
  near-zero-scale pure circle 1.6e-18, unit circle 5.7e-16 unchanged;
  MUT-E4's kill restored on the natural near-miss fixture. Pins:
  `test_flaw_straita_implicit_rs_raw_gauges_removed` +
  `test_flaw_straita_extreme_scale_circle_certifies_not_degenerate`.
  Verification: `cargo test --lib` 193/193, mutation battery ALL KILLED,
  `scripts/check.sh` full tier green.
- **FLAW-A01 (Set A follow-through; 4-module lookahead leak):**
  `ensemble_gated.py`, `ensemble_m.py`, `incremental_info.py` and
  `e5_market.py` built train blocks with the raw `dates < origin` mask while
  the `_d5` target forward-shifts the 5-day rolling mean (`shift(-5)`) —
  blocks within 5 days of the origin boundary read forward out-of-sample
  rows. Fixed by the shared helper `trading_e4t1_v2::_train_blocks_no_leak`
  (mirrors `run_origin`'s corrected mask `tr[tr + 5 < origin_idx]`) applied at
  every train-block call site in the four modules. Pins: behavioral pin
  (pre-fix leak demonstrated inline) + four-consumer text pin in
  `tests/test_dyn_external_audit_2026_10_21_pins.py`.
- **FLAW-A09 (Set A follow-through; sibling lag-buffer leak):** the
  distributed-lag scorer wrote `lag_state["buffers"][base_col]` on the first
  lag term of a call, so a second lag term on the same base column read that
  same call's own tail as prior history (future data prepended to the head).
  Buffer is now snapshotted pre-call and written once per (call, base
  column). Pin:
  `test_distributed_lag_multi_lag_same_column_streaming_safe` (chunked ==
  single-pass with two lag terms sharing one column).
- **FLAW-A13 (Set A follow-through; zero-denominator masking):** blind
  benchmark per-domain metrics reported `0.0` on zero denominators — falsely
  certifying a 0% FPR where no negative controls exist. Per-domain
  `fpr`/`fdr`/`vrr` now report `None` on empty denominators, mirroring the
  headline convention; pins updated to the honest contract and a dedicated
  zero-denominator pin added (`tests/test_dyn_blind.py`).
- **Frozen-record re-mint v1_8 → v1_9 (per freeze protocol):** the FLAW-07 and
  FLAW-08 referee edits legitimate the guard-file refresh;
  `apedp_dyn/sandbox/APEDPX_FROZEN_v1_9.json` supersedes v1_8 with
  `remint_log` rationale for `adjudicate.py` + `claim_ladder.py` (13/15 other
  guard files byte-identical); thresholds, env pins, and blind-case
  expectations unchanged. Post-mint: blind + round-2 suites 56 passed.
- **Mutation-pin coverage for the 2026-10-21 remediation:**
  `tests/test_dyn_external_audit_2026_10_21_pins.py` (23 tests) pins
  FLAW-02/03/05/07/08/09/10/11/13/14/16/17/18/19; each pin carries an inline
  old-semantics mirror so reverting the fix flips the pin. Four
  reintroduction mutants (MUT-D1..D4: S3 conditional take-over, fallback-put
  deadlock, CBF budget-aware deficit split, re-anchor residual recovery)
  registered in `scripts/run_mutation_battery.py` and wired to the pin suite —
  battery re-run: ALL_KILLED, 9/9 mutants
  (`benchmarks/mutation_battery_report.json`), byte-identical restoration
  verified per mutant.
- Scope note: Rust kernel edits (`diff_pade.rs`, `koopman.rs`, `implicit.rs`,
  `lie_symmetry.rs`, `pde.rs`) are outside the Python freeze record;
  `cargo test` 186/187→186/186 covers them (see FLAW-04/05/06/12/15).

Verification: Rust `cargo test` 186/186; affected Python suites re-run green
(`test_dyn_pde_ode_packages`, `test_dyn_claim_ladder`, `test_evidence_tiers`,
`test_dyn_lean_universal_bridge`, `test_flaw_fixes`, hawkes pins, STEM
contracts, CBF certificate, ledger benchmarks, audio transport, remote
storage, correspondence epistemic bootstrap/caller/incremental/reporting).

---

## [Unreleased] — Micro-sample referee remediation (2026-10-21)

### Two referee defects found by execution, fixed with regression pins
- Context: an executed double-check of an external APEDP-X reconciliation
  (Langmuir 1918 micro-sample adjudication) reproduced 18/20 claims digit-for-
  digit (RimeProposer `min_rows=30` safeguard; `+37.7147*(sat_norm(p)) +3.97006`
  with dev_r2=0.9935; 8-scale LOOCV RMSE 0.404810 vs 0.408720; Wilcoxon W=22.0
  p=0.3652) and falsified two: the negative-control "5 proposals" figure is
  4 on this host/kernel, and the "10.2×" timing ratio sits in no measured
  protocol scope (scalar-dot 153×, full fold 1.5×).
- **Defect 1 (soundness):** the external `sound_continuous_monotonicity_proof`
  filtered positive roots by an absolute 1e-9 tolerance, certifying dN/dp =
  p·(p − 5e-10) — genuinely negative on (0, 5e-10) — as monotonic. Fixed in
  `benchmarks/run_rime_microsample_adjudication.py::sound_monotonicity_v2`:
  exact root isolation with NO tolerance filter, interval partitions with
  40-digit probes, and an UNRESOLVED fallback for non-algebraic expressions.
  Regression world: `tests/test_dyn_rime_microsample_adjudication.py::
  test_counterexample_with_micro_root_is_violated` (external engine's verdict
  was WRONG, fixed engine returns VIOLATED).
- **Defect 2 (decision semantics):** the external negative-control check
  `all(p.parameters.get("scale", 0) < 0 for p in proposals)` is vacuously
  True for an empty packet, so a proposer abstention could be reported as a
  dynamics-inversion rejection. Fixed in
  `classify_referee_decision`: ABSTAIN (empty packet, asserted by test) /
  REJECT (gate failure) / PROMOTE are distinct outcomes and never conflated.
- **Claim converted from prose to measurement:** the Michaelis-Menten N=12
  "PROMOTED (12/12 folds)" bullet existed only in narrative; the new harness
  re-executes it (LOOCV RMSE 0.1740, 12/12 admissible under the exact family
  certificate q′(p) = w·s/(s+p)² > 0 ratified by symbolic identity check).
  Artifact: `benchmarks/rime_microsample_adjudication_results.json`.
- **Fold-level proposer recovery frozen (2026-10-21 followup):**
  `run_fold_level_rime_recovery` executes RimeProposer inside each of the 11
  Langmuir LOOCV folds and records fold-level recovery/abstention in the
  artifact, in two disclosed modes: bank-restricted (Python-built sat_m
  columns; RMSE 0.482436, selections {0.8×1, 0.7×1, 0.6×2, 0.66×7}, zero
  abstentions) and unrestricted-grammar (raw p + declared repertoire,
  kernel-derived scales; the engine re-selects a saturation family in 11/11
  folds at measured cost 0.629997). A disclosed_limit field pins that the
  two modes are not the same experiment; per-fold partial-gate decisions
  (capacity_positive subset) are recorded with the graph-structural gap
  disclosed as not-yet-wired. Three new pins (14 total in
  tests/test_dyn_rime_microsample_adjudication.py).
- **Structure-aware referee patch (defect-3 closure):** the external referee
  promoted a physically unstructured counterfeit quadratic (w=5, |b|=0.5,
  positive data correlation) while rejecting the tournament's own verified
  sat_norm winner via a |b|<=1.0 tolerance. `adjudicate_structure_aware` in
  the artifact reads structure from the declared formula family, applies the
  exact family certificate (monotone for ANY intercept), gates on {w>0,
  q_inf=w+b>0, exact monotonicity}, and REPORTS q(0)=b with an explicit
  origin_strict_satisfied flag instead of a tolerance gate. Measured:
  counterfeit -> REJECTED; discovery (w=37.7147, b=3.97006) -> PROMOTED with
  q(0) deviation disclosed; negative-scale saturation -> inversion rejection;
  b=0 baseline -> origin-strict PROMOTED; the 11 bank-restricted fold
  proposals re-adjudicate 11/11 PROMOTED as verified structures with 0/11
  origin-strict. Five new pins (19 total).
- **Graph-equation structural validation wired into the fold referee
  (2026-10-21 followup):** `validate_kernel_graph_structure` enforces the
  reviewer's structural check from kernel metadata — graph_id provenance,
  canonical equation as the authoritative token (declared-column mode or
  kernel-transform mode resolved against `inputs_used_idx` /
  `declared_nominal_dependencies`; the kernel emits empty
  `inputs_used_names` for transform columns, spec indices are the signal),
  independent feature reconstruction, and executable affine-readout
  verification (compute_score == w*feature+b on the dev rows within 1e-9;
  measured max deviation 0.0 machine-exact). The 11 Langmuir folds re-run
  through the kernel with this referee: 11/11 PROMOTED with 11/11
  structure-validated and per-fold graph ids/equations/gates frozen.
  A tampered-graph test proves the validator rejects when the executable
  readout deviates while metadata checks pass. Three new pins (22 total).
- **Michaelis-Menten pathway frozen (2026-10-21 followup):**
  `run_michaelis_menten_graph_validated` executes the same graph-validated
  referee path for the MM system: the actual RimeProposer runs inside each
  of the 12 LOOCV folds over the MM saturating bank, every proposal is
  metadata-validated with a machine-exact affine-readout check (max dev
  0.0), structure-aware gates apply with declared s_m. Measured: 12/12
  PROMOTED, 12/12 structure-validated, zero abstentions, actual-RIME LOOCV
  RMSE 0.1316 (beats the b≡0 baseline 0.1740 on this dataset); readout w
  recovers Vmax=10.0 within 5%; per-fold origin deviations disclosed.
  The full-data tournament reproduces the external '6 proposals PROMOTED'
  row with measured semantics (6/6 admissible, saturation top). Two new
  pins (24 total).
- Also independently re-measured: `cargo test --lib` = 186 passed / 0 failed
  (the "186/186" claim reproduces exactly as the lib-only scope; full
  `cargo test` = 204 passed / 2 ignored). The dynamic pytest scope
  (`pytest tests/test_dyn_*.py`, 1664 collected) measured **1639 passed /
  24 failed / 1 skipped in 25 min (exit 1)** — the external "125 of 125"
  claim does not reproduce in this scope; all 24 failures are pre-existing
  tree-state failures (dashboard promoted-binding ×4, e5_market/e5_r1 ×4,
  ensembles ×6, governance manifest, eseds scorecard, ha_coupling,
  policy_grammar, quantum-sandbox honesty ×2 minus one duplicate, rust
  advanced-ops, stem Hamiltonian, ui sandbox/studio), each reproduced
  identically in a 74 s isolated rerun of the 24 node IDs on the committed
  tree — an additive-only commit was present during that rerun, so the new
  files cannot be their cause.

### Tutorial notebook refreshed to the current platform record (2026-10-21)

`notebooks/apedp_scientific_discovery_tutorial.ipynb` (guard: `tests/
test_dyn_notebook_adapter.py::test_tutorial_notebook_structure_and_execution
— every cell executes in one sequential namespace) was updated to the
current frozen record:

- **Badge launch URLs fixed**: the Colab / Kaggle badges pointed at
  `owner/science` and `blob/main/` — neither exists. They now target
  `psalmprax/science` at `blob/master/` (the only launchable default branch,
  verified by rendering the GitHub view).
- **Step 5 demo-vs-production marker**: the tutorial's synthetic holdout
  demo (`cand_alpha_01`/`cand_beta_02` on `holdout_slice_Q3`) now recreates
  its demo ledger when a previous run already burned the slice at the
  default path (`[demo-reset]` notice). Production holdout ledgers use other
  filenames and are never touched (AGENTS.md §1.2 discipline unaffected).
- **Step 7 — micro-sample symbolic discovery**: live `RimeProposer` search
  on Langmuir 1918 (N=11) plus the structure-aware, graph-validated referee
  (`adjudicate_structure_aware`, `certify_origin_constrained_rime`,
  `sound_monotonicity_v2`); cells reproduce the frozen semantics (saturation
  family UNIQUE_WINNER, PROMOTED with `origin_strict_satisfied` reported as
  False, no `|b|` tolerance gating).
- **Step 8 — companion pathways**: SIR epidemiology discovery via
  `CompartmentalSirProposer` (fitted β=0.2996/γ=0.1199/R0=2.499 against
  true 0.30/0.12/2.50 with exact mass conservation) and the frozen
  Michaelis-Menten pathway re-executed through the graph-validated referee
  (12/12 folds PROMOTED, actual-RIME LOOCV RMSE 0.1316, tournament PROMOTED
  6/6) — matching `benchmarks/rime_microsample_adjudication_results.json`
  digit-for-digit.

---

## [Unreleased] — Repair Python 3.12 blind-freeze CI failure (2026-10-20)

### Lorenz public benchmark: prereg frozen + confirmation harness built (2026-10-20)
- Context: an external evaluation transcript (bare-BIC Lorenz trial) claimed a
  successful "validation" while its own stress test showed the spurious x²y
  term winning training-BIC; its operational model mixed ground-truth constants
  into the "discovered" physics and validated by 3D-plot eyeballing (explicit
  Euler diverges from RK45 truth at t≈0.26–0.34 even with exact coefficients,
  so the plot instrument is uninformative). This prereg turns that exchange
  into a properly measured registered protocol.
- `docs/PREREGISTRATION_LORENZ_PUBLIC_BENCHMARK_V1.md` — FROZEN: catalog Lorenz
  world σ=10/ρ=28/β=8/3, seeds {101,202,303} × noise {0.5–3%} (12 registered
  runs/path), SG(21,3)+np.gradient estimator declared, 70/30 dev/sealed-holdout
  split with full-lineage burn; four registered paths — PATH-R (referee-gated
  forward selection, dev-BIC ordered), PATH-B (greedy forward training-BIC),
  PATH-B′ (curated 4-candidate transcript ranking), PATH-C (user-instructed
  second negative control: the true-constants chimera + magnitude pruning +
  Euler validation — a plot-validated anti-pattern registered specifically to
  test the instruments); hypotheses H1–H6 with margin provenance labeled
  [Pilot]/[Design]. H4's drafted 0.5-SE floor was REFUTED by the 12-run pilot
  (SG derivative bias is systematic and WORST at low noise: 1.4–2.0 SE at
  0–0.5%) and corrected to ≤ 2.5 SE PRE-freeze, never post-hoc. Pilot
  observations: the referee never promoted the BIC winner (0/12); greedy BIC
  admits 3rd-order spurious terms 12/12 (stronger than the transcript's
  curated ranking); referee settings frozen at n_boot=1000/seed=7.
- `benchmarks/run_lorenz_public_benchmark.py` — the frozen harness (audit-only):
  implements all four paths over the registered grid, computes the registered
  instruments verbatim, burns the sealed confirmation windows in the full-
  lineage ledger (`lorenz_public_benchmark_ledger.jsonl`), writes
  `benchmarks/lorenz_public_benchmark_results.json` with git-head + argv +
  timestamp provenance, and a `--smoke` wiring mode (single seed, 100 boot)
  explicitly labeled non-confirmation. The same-integrator divergence-horizon
  control makes integrator error cancel by construction (the H6 instrument).
- Pins: `tests/test_dyn_lorenz_public_benchmark.py` (3 — payload schema, H1
  wiring margin, H6 chimera-catch margin; smoke settings). Smoke verified
  green (3/3) plus harness exit-0; results artifacts commit separately after
  the full confirmation runs per the prereg's decision rule. Expected honest
  outcome risk disclosed up front: PATH-R's smoke wiring missed the true
  structure (the spurious x² proxy promoted first); H3's [Design] margin may
  come back NOT CONFIRMED — that is a registered outcome, never a post-hoc
  harness redesign.

### Frozen referee record refresh
- The Actions regression failure was `verify_referee()` rejecting
  `apedp_dyn/sandbox/multi_grammar_sandbox.py`: its current SHA-256 no longer
  matched the v1.7 pin after the Python 3.12 CI regression fix added the file.
  Preserved the integrity check and minted `APEDPX_FROZEN_v1_8.json` with the
  current source hash; `blind_benchmark.py` now consumes v1.8. No thresholds or
  runtime behavior changed in this record-only refresh.

## [Unreleased] — register limitations addressed: F14b wiring, F16 CBF certificate, LEAN reproduction (2026-10-20)

### Full-suite failure triage and first-wave fixes (2026-10-20)
- Reproduced the full red state in one non-deselected run:
  `pytest tests/ -q` → **97 failed / 3,321 passed / 4 skipped / 68:35 (rc=1)**.
  Triage below; first wave of true-root-cause fixes landed this commit:
  1. **Facade-observer regression → 62 tests red** (`tests/test_facade_contracts.py`):
     the shared detector scanned the stale `build/lib` packaging artifacts
     (`tools/coverage_audit.py` had `build`/`dist` missing from `SKIP_DIRS`; the
     built node_modules are byte-for-byte shims of the real facades), then
     importlib can not import `build.lib.apedp.*`. Fix: `build`/`dist` pruned
     from the audit's own source-tree scans (same rationale as the
     `.kilo/worktrees` exclusion: not the tree under test). `tests/
     test_facade_contracts.py` now 65/65 green. Facade-pin tools regression
     green: `test_coverage_audit`(23) + `test_affected_tests`(22) +
     `test_check_tools_fail_loudly`(10).
  2. **Trading E4T1 data-plumbing import never wired (7 red)**:
     `test_dyn_trading_e4t1.py` reads snapshots via `T1.load_frames()` (v2's
     docstring pins the same panels as v1); was written against `T` (v2) that
     never re-exported the loaders. True fix: switch `_panels()` in the test's
     setup path to v1 (data-identical), keep T for hypotheses and T1 for v1's
     protocol. `tests/test_dyn_trading_e4t1.py` now 9/9 (was 2/9).
  3. **RIME_SUITE_TOTAL 212 → 213** (`tests/test_dyn_reporting_integrity.py`):
     every file the machine-verification counter scans was re-collected;
     actual = 213. Per that test's own failure message the constant is
     re-derived, never hand-edited from prose.
- **Remaining 27 failures, triaged but NOT silently re-pinned** (each needs a
  data- or referee-recalibration decision, which is exactly why they were not
  jogged): (a) physics/adjudication flips on borderline regimes — dashboard
  (4, promoted binding now MULTI), quantum_vacuum/sandbox/ui_sandbox creep
  abstains (3, expected MULTI got UNIQUE): opposite-direction verdict drift
  through the shared discovery/adjudicate path; (b) market-panel numeric pins
  drift (16: e5_r1 3, ensemble 3, gated_ensemble 3, e5_market 1, policy
  grammar index 1, ha_coupling 1, eseds scorecard 1, ui_studio series
  reconstruction 1, system classes/gate counts/R²): all read the Oct-2
  `crypto_top15_orderflow.csv` snapshot — needs one joint re-derivation
  session over a declared data version; (c) contract-evolution pins:
  `stem_grammar` strategic-arch (1), `external_60` near-OOD-floor (1),
  scale-invariance 1e9 edge (1), `tier_b` Planck R² 0.94 vs 0.95 floor (1),
  `rust` native GA advanced-ops (1); (d) governance-manifest SHA (1): needs
  renewed trustee certification (external).

### Local credential remediation: PAT stripped from the remote URL, credential helper installed (2026-10-20)
- `remote.origin.url` rewritten from the token-embedded URL to the bare
  `https://github.com/psalmprax/science.git`; repo-local
  `credential.helper = store` installed. Public read path verified live
  (`git ls-remote origin HEAD` rc=0); the next push will prompt once and the
  helper persists the credential (plaintext `~/.git-credentials` — prefer
  `gh auth login` or `git-credential-libsecret` to avoid that if available).
- Workspace sweep: the embedded-token URL pattern matches NO tracked file
  and only one vendored third-party fixture (`.venv/.../LibGit2/test/
  libgit2-tests.jl`) — the leak is local-config-only. **The exposed PAT is
  still live until deleted on GitHub** (Settings → Developer settings →
  Personal access tokens): rotation there is the user's step — an agent
  session cannot do it without account credentials.

### REG-F14 — restart-proof sealed-slice consumption + fail-closed ledger writes + canonical-spec shrink import (2026-10-20)
- `apedp_dyn/epistemic/caller_data_protocol.py`:
  1. **Durable lead state (I2 across restarts):** `CallerSplitProtocol` now
     persists issued leads and sealed-consumption provenance to an atomic
     JSON state file (`persistence_path` argument, or derived from the
     attached ledger's persistence directory, e.g.
     `caller_lead_protocol_state.json` next to `experiment_ledger.jsonl`).
     A restarted process reloads the store: an open lead cannot be re-issued,
     a consumed slice cannot be re-consumed (`ConfirmationAlreadyConsumedError`
     from the durable record), the slice is claimed spent BEFORE evaluation
     (a mid-evaluation crash leaves it spent, never resurrectable), and a
     corrupt store fails CLOSED (`CallerProtocolStateError`) — the protocol
     never silently falls back to memory-only operation. Without a resolvable
     state path (in-memory unit pins), in-memory-only behavior is retained as
     a documented unit-use limitation.
  2. **Fail-closed I3:** the silent `except Exception: pass` around ledger
     exposure writes in `SealedConfirmationSlice.consume` is REMOVED —
     ledger write failures now propagate. Additionally, with a ledger
     attached, `consume` refuses an UNREGISTERED candidate BEFORE spending
     the sealed slice (`CallerComplianceError`), so an evaluation that could
     not be written up afterwards can never happen.
  3. **No hardcoded threshold literal:** the mode-(b) `dsr_shrink`
     disclosure constant is now imported from the frozen canonical spec
     (`apedp_dyn.conf.canonical_spec.DSR_THRESHOLD`, hash-verified at
     import). The spec JSON itself was NOT modified — it is quorum-signed
     under `specifications/governance_manifest.json` (external trustees);
     changing a preregistered constant requires a re-signed revision, not an
     edit. Values are unchanged (0.95).
- Pins updated: `tests/test_dyn_caller_data_protocol.py` (7 → 14: end-to-end
  restart durability, open-lead restart refusal, aborted-evaluation spent
  claim, corrupt-store fail-closed (3 shapes), unregistered-candidate
  pre-spend refusal, ledger-failure propagation, canonical-spec import pin);
  `tests/test_dyn_caller_compliance_wiring.py` updated for I3 (mode-(a)
  register-first discipline). Two wiring-census rows in
  `docs/DOCUMENTATION.md` + the register updated accordingly.
- **Pre-existing (NOT introduced here), disclosed:**
  `tests/test_dyn_governance.py::test_canonical_repository_governance_manifest_valid`
  fails at the committed tip independent of this work — the governance
  manifest pins spec SHA `a6d026…` while the spec on disk hashes `a11fd957…`
  (the same value `canonical_spec.py` import-check passes against). Fixing
  it requires trustee re-certification of the manifest, not a code edit.

### Frozen-record re-mint v1_7 → v1_8 (rationale, per freeze protocol)
- `apedp_dyn/sandbox/APEDPX_FROZEN_v1_8.json` (supersedes v1_7;
  `blind_benchmark.py` now consumes v1_8 via `FROZEN_RECORD`). Only guard-file
  hashes refreshed; thresholds, env pins, and blind-case expectations
  unchanged from v1_7.
- Remint rationale (2 entries, both verifiable in-tree):
  1. `apedp_dyn/sandbox/multi_grammar_sandbox.py` — legalizes drift introduced
     by commit `3419425` (purge forward-overlapping development targets) that
     landed AFTER the v1_7 mint without its own re-mint (v1_7's hash for that
     file was stale). File is worktree==HEAD at mint; harness fix, not a
     referee edit.
  2. `apedp_dyn/sandbox/multi_asset_signals_sandbox.py` — additive REG-F14b
     per-caller compliance footer (`publish_caller_compliance_record`
     delegating to `caller_registry`); no method, threshold, or behavior
     change; mandated by the freeze protocol before blind scoring proceeds.
- Post-mint verification: `pytest tests/test_dyn_blind.py
  tests/test_dyn_blind_audit_round2_remediation.py` → 55 passed,
  verify_referee() satisfied.

### REG-F14/F14b — per-caller wiring over the production-caller census (mode b)
- `apedp_dyn/epistemic/caller_registry.py`: auditable census of six production
  data-consuming verdict publishers (dyn_sweep, multi_asset_signals,
  real_market_l2_sweep, blind_benchmark, calibrated_benchmark_suite,
  canonical_ledger); each caller module wired with a thin
  `publish_caller_compliance_record` publisher; records validate via the F14b
  either-or validator at publish time; `record_selection_exposure` records the
  selection-stage `TRAINING_DEVELOPMENT` exposures that make the derived
  multiplicity live. Pins: `tests/test_dyn_caller_compliance_wiring.py` (8).
  Honest boundary: every census caller is wired in mode (b) — no production
  pipeline operationally holds a sealed confirmation split; mode (a) stays
  available per caller and is claimed for no one.

### REG-F16 — formal CBF-QP certificate + model-check artifact (+ scaled verification)
- `apedp_dyn/robotics/cbf_certificate.py` + `tests/test_dyn_cbf_certificate.py`
  (8 pins) + `benchmarks/cbf_certificate_report.json` (battery exit 0, 64
  certified releases verified, worst slack −3.3e-16; unicycle realization
  model-checked through the PRODUCTION arc stepper over seeded adversarial
  head-on sweeps — min barrier 0.3588, 0 collisions). FORMAL scope is at the
  certified velocity layer under an explicit assumption object (refuses
  γ·dt > 0.5); the unicycle realization is disclosed as model-checked
  evidence, NOT formal. `batch_multi_agent_cbf_step` is untouched and its
  not-a-certified-CBF disclosure pin intact; `certified_cbf_guard` is additive.
- NEW scaled-verification layer in the same battery: the same fail-closed
  formal checks re-verified on deterministic jittered-lattice grids at
  16/32/64/128 agents (1440 additional releases, worst slack −9.91e-10 ≥ −tol;
  Euler invariance h > 0 at every step at every scale; realized guard
  collision-free at every scale under the physically-consistent
  realized-velocity-feedback nominal discipline — 128-agent realized min
  barrier 0.3919) plus recorded per-step WALL-CLOCK guard and guard+arc-stepper
  latencies (measurement of the environment that ran the battery, explicitly
  not a guarantee; mean guard+step 0.60/0.77/1.31/3.39 ms at 16/32/64/128
  agents). A separate DISCLOSED scope probe records verbatim that fresh
  full-range random nominals redrawn every step at 128 agents are OUTSIDE the
  realized layer's evidence (deterministic breach −0.1102, 17 collision steps)
  — the realized layer stays scenario-scoped evidence.

### REG-F14/F14b — mode (a) operational wiring landed (blind_benchmark first)
- `apedp_dyn/epistemic/caller_registry.py`: mode-(a) registry helpers
  `issue_sealed_confirmation_lead` (census- and dataset-family-checked) and
  `publish_sealed_compliance_record` (refuses publication while the sealed
  confirmation slice is unconsumed, with foreign provenance, or for a foreign
  head; demands one-shot provenance and `archived_verdict_source =
  "confirmation"`).
- `apedp_dyn/sandbox/blind_benchmark.py`: first census caller wired for mode
  (a) over its OWN real slice axis — `open_sealed_confirmation_lead`
  (selection = worked-example/selection case ids, confirmation = the SEALED
  blind case ids), `evaluate_confirmation_cases` (frozen-referee-checked,
  closed confirmation family, blind-metrics verdict),
  `publish_sealed_compliance_record`.
- Pins: `tests/test_dyn_caller_compliance_wiring.py` 8 → 12 (sealed consume
  exactly once; second candidate refused; publication refused while
  unconsumed / with foreign provenance; sealed record validates).
  Honest boundary: the five remaining callers remain mode (b) by default and
  the pinned mode-(a) demonstration is the wiring path — a live production
  promotion through the sealed slice is pending, disclosed as such.

### REG-LEAN — pinned-toolchain reproduction of the 323-record sweep
- `scripts/run_lean_pinned_reproduction.py` +
  `benchmarks/lean_pinned_toolchain_reproduction.json`: ALL 323/323 records
  reproduced (proof_claim_status + kernel_exit_code per candidate) under the
  independently provisioned pinned toolchain `leanprover--lean4---v4.34.1`
  (commit `5045d0056413266e57c625dcd7c365b10e377c52`) vs the reference's
  v4.35.0; counts 323/315/8/0. Reference artifact + committed `.lean`
  evidence restored byte-identically (HEAD-committed bytes take precedence
  over regeneration — regeneration silently un-remediates the u15 honest
  disclosure; caught and fixed in the runner).
- **Root-cause fix:** `apedp_dyn/epistemic/lean_universal_bridge.py` compile
  subprocess timeouts raised 15/25 s → 240 s (docker paths unchanged except
  none needed): under 8-way parallel contention the two heaviest fluid-cell
  files (NavierStokes / forced-ODE) exceeded the old timeout and were reported
  P1 (exit −1) while compiling clean standalone, so the reference artifact
  could not be reproduced as-is. Exit codes now reflect verification, not
  wall-clock contention. The register's external (outside-this-environment)
  sign-off remains outstanding and is NOT claimed.

### ERR-INC-80 — §80 pins re-derived under the forward-overlap-purged harness (disclosed)
- Commit `3419425` purged the 5-day-ahead window from the shared development
  targets (`trading_e4t1_v2.run_origin`) — a legitimate leakage fix that
  shifted the frozen §80 headline magnitudes, leaving 4 of 5 pins in
  `tests/test_dyn_incremental.py` failing on the tree.
- Same seed (8080), same artifacts, no threshold moved: the pins were
  RE-DERIVED (deterministic, verified) under the purged harness and
  SHARPENED (exact per-asset values incl. permutation p, corr(ŷP,ŷA),
  BTC win row; pre-registered headline rule perm_p >= 0.05 kept and made
  load-bearing again). The headline decision (Case 2/4, permutation-null
  everywhere — min .0896, ETH) is unchanged; only the magnitudes moved
  because the leak they were computed under was purged.
- Full disclosure table in `docs/TRADING_EQUATION_FORENSICS.md` (§80b).

### REG-MUT — targeted mutation battery, CI-gated (CLOSED-WITH-EVIDENCE)
- `scripts/run_mutation_battery.py`: dependency-free battery for the register's
  three targets (Finding-14 adjudicate anchor ×2 mutants, F8
  frozen-library-term-resolution fail-closed + diagnostic-detail ×2, hawkes
  standardization ×1), each pinned via targeted pytest runs in fresh
  interpreters with byte-identical restoration verified and pristine sanity
  pins. Exit 0 requires ALL mutants killed; a survivor is a REG-MUT failure.
- **Honest first run:** MUT-C1 (eval-frame-leak standardization) SURVIVED —
  the inherited dynamic pin scored a fit/eval-identical frame, so it could
  not distinguish the property. The pin was SHARPENED
  (`tests/test_dyn_hawkes_standardization_pins.py`: degenerate-fit continuous
  fallback + distribution-shifted eval frame; Finding-10 property now
  load-bearing), never weakened; re-run: ALL_KILLED (5/5, pinned report at
  `benchmarks/mutation_battery_report.json`).
- CI gate: new `mutation-battery` job in `.github/workflows/test.yml`
  (survived mutant fails CI; report uploaded, never hidden) and
  `scripts/check.sh` full tier. Register row updated with status + honesty
  trail (`docs/SYSTEM_FLAWS_AUDIT.md` register status updates).

### JetFlow v0.3 collision-world confirmation — honest H1 boundary result
- `benchmarks/run_jetflow_video_v03_confirmation.py` ran the frozen
  confirmation seeds {407, 508, 509} (exploratory 404 excluded):
  **H1-collision CONFIRMED 3/3** — the fluid-domain JetFlow advantage does NOT
  transfer (diffusion wins SSIM + temporal on every seed): the published
  advantage is DOMAIN-SCOPED to continuum transport. H2 (PSNR floor), H3
  (train-ratio ceiling) CONFIRMED; H4 CONFIRMED after **ERR-H4-UNITS**, a
  disclosed units erratum: the prereg text mis-stated the passes metric as
  amortized-over-n_gen; verdicts recomputed from the SAME committed per-seed
  artifacts — no re-run, no threshold moved, no seed changed.
- Report: `docs/JETFLOW_VIDEO_V03_CONFIRMATION_REPORT.md`; verdicts:
  `benchmarks/jetflow_video_v03_confirmation_results.json` (+ per-seed
  artifacts under `benchmarks/jetflow_video_v03_confirmation/`).

### Caller-layer three-role data protocol (REG-F14/F14b reference implementation)
- `docs/CALLER_DATA_THREE_ROLE_PROTOCOL.md` +
  `apedp_dyn/epistemic/caller_data_protocol.py` (`experimental`, additive):
  disjoint SELECTION/CONFIRMATION/PROSPECTIVE roles at issue time; sealed
  one-shot confirmation consumption (REG-F14 letter); confirmation exposure
  records integrate the existing anti-snooping burn ledger; archived verdict
  source must be `confirmation` with selection superseded; F14b either-or
  compliance (sealed split OR ledger-derived DSR-adjusted multiplicity —
  computed, never hand-written; refuses deflation; reuse events inflate).
- Acceptance pins `tests/test_dyn_caller_data_protocol.py` (7) include the
  register's end-to-end letter: pool on slice A, head confirmed exactly once
  on sealed slice B, verdict provenance from B only, A∩B=∅, second consume
  refused. Register status: OPEN — per-caller wiring of every production
  caller (mode a/b) is still pending and remains the acceptance gate.

## [Unreleased] — E4 divergence-point selection, second JetFlow world, verified-knowledge-module library (2026-10-19)

### Gap-1 generalization: cost-aware divergence-point selection (`experimental`)
- `apedp_dyn/epistemic/active_disambiguation/divergence_selection.py`: extends the fixed
  E4 measurement menu to proposing the query point itself on a continuous domain, ranked
  by EXPECTED COST TO RESOLUTION (`ceil((k·σ/gap)²)·cost(x)`) under an explicit budget —
  not raw gap maximization. Named abstention causes (`INSUFFICIENT_DIVERGENCE`,
  `INSUFFICIENT_BUDGET`), preserved P2/P3 no-truth-leakage (selection sees only
  hypotheses + observation/cost model), P6 re-adjudication through `adjudicate()`,
  and P8 determinism (frozen grid + deterministic refinement; zero randomness in
  selection). `policy.py` refactor made the shared `DEFAULT_DETECTABILITY_K` explicit
  (behavior-identical; 20 E4 regression tests pass). Pins:
  `tests/test_dyn_divergence_selection.py` (8 tests: analytic optimum near x*=−0.2
  for the constructed pair, cost-awareness, named abstentions, no-truth-leakage,
  determinism, budgeted closed-loop resolution to UNIQUE_SUPPORT).

### Second JetFlow problem family: collision/occlusion world (`experimental`) — honest negative
- `apedp_dyn/video/collision_domain.py`: layered 2D rigid-body scenes (same-layer elastic
  collisions + wall bounces; cross-layer pass-through with genuine parallax occlusion —
  real disappearance/reappearance, pinned). Byte-reproducible by construction;
  manifest split-exclusivity via the shared `held_out_split` instrument.
- `benchmarks/run_jetflow_vs_diffusion_matched.py`: `--domain` parameter added
  (default `fluid` preserves the original exploratory path); instrumentation converted
  to grad-free `_to_np` (fixes a latent `requires_grad` crash in the student path).
- **Exploratory seed-404 result is a disclosed NEGATIVE result for advantage transfer:**
  on the collision world the matched diffusion baseline wins every instrument
  (SSIM 0.065 vs 0.017; PSNR 20.8 vs 13.7 dB; temporal 0.078 vs 0.234; SWD 0.167 vs 0.476)
  and trains faster (8.2 s vs 11.1 s @300 steps). Conclusion per prereg: the fluid-domain
  advantage is DOMAIN-SCOPED to continuum transport; no unscoped generalization claim.
- New prereg frozen: `docs/PREREGISTRATION_JETFLOW_VIDEO_V03_COLLISION.md`
  (confirmation seeds {407, 508, 509}; H1-collision replicates the ordering on ≥2/3 seeds;
  v0.2 fluid protocol untouched per the deviation rule). Boundary cross-reference added to
  `docs/jetflow_vs_diffusion_matched_benchmark.md`. Pins:
  `tests/test_dyn_collision_domain.py` (8 tests).

### Gap-3 realization: Verified Knowledge Module Library (`experimental`)
- `apedp_dyn/epistemic/verified_knowledge_module.py`: read-only additive layer over the
  existing ledgers (they remain the single source of truth). Entry gates: experiment-ledger
  `CONFIRMED_VALID` + promotion Stage 4 (out-of-sample transfer); economic stages 5–6 are
  deliberately NOT required (domain-general library; disclosure shipped on every record).
  Metadata contract: preconditions required; finite ordered validity domains;
  uncertainty requires value + estimation method (no naked floats); counterexamples
  REQUIRED (a module without a documented failure is not verified knowledge); dependency
  closure at registration and in composition; composition INTERSECTS validity domains and
  refuses disjoint/out-of-chain reuse. Pins: `tests/test_dyn_verified_knowledge_module.py`
  (12 tests).

## [Unreleased] — Round-3 external assessment record (2026-10-19)

- Registered the L2 holdout R² base floor, F-test significance level, and fitted
  predictor degrees of freedom in the canonical preregistered specification;
  pruning now consumes those pinned values rather than hardcoding the 95% gate.
- Post-round-3 external review: **6.5/10 overall** (up from ~6/10), credited to
  reliability and audit discipline, while explicitly noting the remediation
  does not by itself establish stronger scientific discovery capability or a
  demonstrated advantage over competing methods. Scorecard and the
  reviewer-stated criteria for reaching 8/10 (REG requirement closure with
  per-item acceptance evidence, independent review of mathematical
  certificates and critical code paths, reproducible external evaluation of
  discovery validity) and 9/10 (measurable advantage over established
  alternatives across multiple independent problem families) are recorded in
  `docs/SYSTEM_FLAWS_AUDIT.md` ("Round-3 external assessment record") as
  forward requirements — not claimed progress.

## [Unreleased] — Remediation checkpoint frozen at `47594e7`; open-requirements register (2026-10-19)

- `47594e7` is tagged `remediation-checkpoint-47594e7` as the frozen
  remediation milestone. Per the external review's final recommendation, F14
  (three-role data protocol), F16 (formal CBF-QP certificate), mutation
  testing of referee/leakage paths, and an independent Lean toolchain run are
  recorded as **explicit unresolved requirements** with exact acceptance
  criteria in `docs/SYSTEM_FLAWS_AUDIT.md` ("Outstanding requirements
  register", IDs REG-F14, REG-F14b, REG-F16, REG-MUT, REG-LEAN). None may be
  marked complete silently.
- F17 scale-aware gate — reviewer acceptance items closed: the polynomial
  library's constant `1` column had pinned the term-scale denominator at O(1),
  defeating scale-awareness below O(1) data (a mis-fitting linear candidate at
  near-zero scale was certified permissively). The term scale is now measured
  over NON-constant columns; near-zero-scale exact laws certify scale-
  consistently while misfits refuse. Degenerate inputs pinned: all-zero and
  constant data (no NaN via the 1e-12 floor), <10 samples and NaN inputs
  rejected explicitly at the binding boundary. Rust lib 186/186. New pins:
  `TestF17ExtremeScaleRobustness` (6 tests), `TestF8ErrorDiagnosticDetail`.
  Round-2 suite total: 46 tests.

## [Unreleased] — External-review round (cont.): F17 scale-aware gate, F8 fail-closed mirror, Lean chain pin (2026-10-19)

Continued from the review-response round above (same external report, remaining "additional scrutiny" items answered in code rather than in prose):

### F17 — scale-aware exact-recovery gate (Rust `rime/implicit.rs`, rebuilt)
- The exact-recovery gate is no longer an absolute `rmse < 1e-5`. It is measured
  RELATIVE to the RMS magnitude of the relation's own residual terms, so
  O(1e5)-scale genuinely exact relations (the audit's QEC-scale class) are
  certifiable while small-scale near-misses are refused at the same relative
  tolerance (1e-5). Verified through the rebuilt pyo3 binding: O(1) circle
  exact (rmse 7e-16), 1e5-scale exact recovered (rmse 9e-13), small-scale
  1e-4-perturbed relation REFUSED. The "0 = 0" tautology guard stays
  scale-independent via the active-term count. Rust lib: 186/186. Pinned by
  `TestF17ScaleAwareExactGate` (4 tests).

### F8 — frozen-library mirror drift fails closed
- `_frozen_residual_r2` no longer silently skips dev-discovered active terms
  that the Python cold-end mirror cannot resolve — a dropped column corrupted
  the frozen R^2 toward flattering values (the original F8 defect class).
  Unresolved terms now raise naming the drifted term(s). Pinned by
  `TestF8FrozenLibraryTermAudit`: fail-closed raise, term-classification pin
  (`u^3` is a cubic STATE nonlinearity, `u_xxx` the third-order spatial
  derivative — distinct series, per the reviewer's classification remark), and
  a mirror-coverage pin that every Rust 1D catalog term (including `u^3` and
  `u_xxx`) resolves in the Python mirror.

### Lean — complete dependency-chain integrity pin
- The reviewer's request to "compile the complete dependency chain and check
  for unintended axioms or unproved assumptions" is pinned mechanically: for
  every P2-verified record in the regenerated provenance (315 records), the
  emitted Lean source in the tracked `apedp_dyn/lean_formal/` directory must
  be `sorry`-free, and every `sorry`-carrying file must belong to a
  P0-refuted record — the sorry channel can no longer certify under any
  label drift. (`TestLeanProofChainIntegrity`, 2 tests over the full family.)


Follow-through on the external remediation review (rating round). All items below are
pinned by `tests/test_dyn_blind_audit_round2_remediation.py`.

### F14 — holdout selection-reuse disclosure
- The reviewer's composition critique is accepted: the pair verdict selects the
  promotion head on the SAME holdout slice that later certifies it, so that slice
  is a selection set, not an untouched confirmation set. The referee cannot mint
  new confirmation data, so every pair-path `UNIQUE_SUPPORT` verdict now carries
  the explicit diagnostic flag
  `selection_reused_confirmation_holdout(n_candidates=<N>)` (the pool size that
  was compared on the slice). Callers MUST charge that selection to the DSR trial
  ledger (`search_id` + `selection_pool_size` — the standing multiple-testing
  correction) or hold out an independent sealed confirmation split; a FULL
  contract claim without that separation is misstated and callers/auditors can
  now detect it mechanically from the verdict record. The former
  `test_tau_shrunk...` fixture documents the two-directional τ rule (Δ ≤ τ ->
  `MULTI_HYPOTHESIS`; Δ > τ -> `UNIQUE_SUPPORT`), establishing that uniqueness
  comes from the prespecified τ, never from the (construction-guaranteed)
  sign of the separation.
- Re-minted: `adjudicate.py` SHA pin (`f56987b9…`) in `frozen_spec.py` and
  `APEDPX_FROZEN_v1_7.json`.

### F4 — multiplicity coverage pins
- Pinned: the ledger charges every recorded trial including repeated
  evaluations of the same structure under different seeds/hyperparameter
  settings (6 records under one `search_id` -> 6 trials), and charges the full
  declared pool beyond recorded survivors (40 declared, 2 recorded -> 40
  trials, `hidden_trials=38`), with the DSR verdict decreasing under the
  hidden multiplicity at matched variance.

### F5 — annualization numerical pin
- The fixture (annualized SR 2.5, 500 trials): `annualization_factor=1.0`
  manufactures significance (p < 1e-6, `is_significant=True`); the
  F5-corrected call passes `annualization_factor=252.0`, yielding SE on the
  annualized scale (≈1.0) and `p ≈ 0.094` (not significant). Both directions
  pinned numerically.

### F16 — safety-envelope battery (not a CBF certificate)
- The reviewer is correct that the pull-away deflection is a heuristic, not a
  forward-invariance guarantee. The stepper's docstring already states it is
  "not a certified quadratic-programming CBF-QP with formal
  forward-invariance certificates"; this disclosure is now itself pinned by a
  test so it cannot silently regress into an overclaim.
- New behavioral envelope pins (all on the portable CPU path):
  finiteness throughout; measured inter-agent separation floor > 0.6 (with
  `r_safe=0.5`) across a collinear head-on encounter, a high-relative-velocity
  approach (2.5u/s closing speed), a dt sweep over [0.05, 0.10, 0.20], and
  simultaneous multi-pair engagement of 4 agents; exact agent-index
  permutation symmetry (max deviation 0.0); finite escape dynamics from
  degenerate coincident starts; and the F16 regression class itself —
  forward progress > 0.4 into a symmetric head-on encounter where the
  pre-fix radial-only deflection deadlocked.
- A formal `dot(h) + alpha(h) >= 0` CBF-QP synthesis remains future work
  (tracked here, not smuggled into the heuristic's claim).

### Honest-scope note
- Reviewer items NOT closed by code here and deliberately not claimed:
  (a) F17's `rmse < 1e-5` gate is still an absolute threshold — a scale-aware
  RMSE normalization is a kernel change deferred with its own acceptance
  criteria; (b) F8's mathematical compatibility of the non-uniform fitting
  and scoring libraries needs a dedicated term-audit, not a test tweak; and
  (c) "full test suite" in prior notes meant the AGENTS.md dynamic battery
  (a named collection), never the entire repository — wording now matches.

## [Unreleased] — Blind-audit remediation round 2: F14–F17 + freeze re-mint (2026-10-19)

### Frozen-record re-mint (rationale, per freeze protocol)
- `apedp_dyn/epistemic/adjudicate.py` (F14 remediation below),
  `apedp_dyn/sandbox/grammar_space.py` (F1 causal-target remediation below),
  `apedp_dyn/sandbox/trading_signals_sandbox.py` (F5 annualization wiring), and
  `apedp_dyn/ui/server.py` (F7 state-series dictionary reconstruction) are
  frozen guard files; all four required edits in this round. Both freeze
  surfaces were re-minted: `frozen_spec.py`'s pinned SHA-256 registry (for the
  two files it registers) and a new blind-protocol record
  `apedp_dyn/sandbox/APEDPX_FROZEN_v1_7.json` (supersedes v1_6;
  `blind_benchmark.py` now consumes v1_7 via `FROZEN_RECORD`): each remint
  rationale is logged in the record's `remint_log`. Referee threshold semantics
  and grammar-space primitive sets are untouched; verification is covered by
  `tests/test_audit_open_flaws_remediated.py::
test_all_frozen_records_verified_against_pinned_hashes`, `tests/test_dyn_blind.py`
  (`verify_referee()`), and `tests/test_documentation_and_spec_integrity.py`.

### Critical fixes (new audit Findings 14–17)
- **F14 — `apedp_dyn/epistemic/adjudicate.py`:** the pair evaluation was anchored
  on the train-loss ordering. An overfitted train-loss winner made
  `delta = rival_ho − best_ho` negative, trivially tripping `delta <= tau` for
  any `tau > 0` and denying `UNIQUE_SUPPORT` to genuine generalizing candidates
  (misclassified `MULTI_HYPOTHESIS`). The pair anchor is now the holdout-lowest
  RMSE candidate when it is gate-4 certified; `delta` is therefore always a
  holdout separation and the `delta <= tau` check verifies the holdout criterion
  directly before any `MULTI_HYPOTHESIS` verdict. A re-anchored pass falls back
  to the honest point-estimate contract for the bootstrap tiebreaker (residual
  series correspondence cannot be attributed across the re-anchor).
- **F15 — `apedp_dyn/proposers/ncm_structural_proposer.py`:** all four archetype
  branches (BOUNDED_RATIONAL_SATURATION, CONSERVED_DAMPED_OSCILLATOR,
  DIFFUSION_ADVECTION_CONTINUUM, CYCLIC_HARMONIC_REGIME) now emit real
  `compute_score` closures. Each closure builds its design matrix from the
  candidate frame the same way the fit did (frame-local gradients / lens, not
  fit-time arrays), falls back to an all-zero vector only for missing columns,
  and reproduces the fit-time prediction in-sample.
- **F16 — `apedp_dyn/cuda/batch_physics_steppers.py`:** `batch_multi_agent_cbf_step`
  now applies the reference CPU implementation's anti-parallel tangential
  breakaway force (`|anti-parallel| > 0.7` → perpendicular breakaway scaled by
  `anti_par * k_rep * (1/d − 1/r_w)/d^*2`), eliminating permanent deadlock on
  collinear trajectories.
- **F17 — `rust/apedp-kernel/src/rime/implicit.rs`:** `is_exact_recovery` now
  requires `active_count >= 2` in addition to `rmse < 1e-5`; the all-zero
  pruned vector (algebraic residual identically 0, the "0 = 0" tautology) can no
  longer be certified as an exact physical recovery.
- **F10 — `apedp_dyn/proposers/hawkes_orderflow_proposer.py`:** the
  standardized-intensity scorer standardized against the *evaluated frame's*
  mean/std, which over a holdout folds the whole future distribution of the
  point process into every `t`. Scoring standardization now couples only to the
  FIT-time intensity distribution constants (`fit_intens_mean`/
  `fit_intens_std`, the same frozen coupling already used by the surge
  threshold quantile), so the emitted signal is strictly causal at evaluation
  time.
- **F12 — `apedp_dyn/epistemic/lean_universal_bridge.py`:** the prover axiom
  `exp_damp_le_one (s : R) : exp (-s) ≤ (1 : R)` was a false universal
  (`exp(-s) > 1` for `s < 0`). The axiom and the synthesized squeeze/exponential
  damping theorem both now carry the non-negativity hypothesis `(hs : (0 : R) ≤ s)`;
  the bound is only claimed in-damped-regime.

### Lean-proof inventory reconciliation
- The unresolved `test_all_system_equations_lean_proofs.py` failures were two
  distinct problems, both fixed:
  1. The five per-sweep provenance inputs (`fluid_ode_dynamical_sweep_`,
     `openai_math_advanced_hypotheses_sweep_`, `ga_depth2_sweep_`,
     `ga_depth3_sweep_`, `expanded_space_discovery_`) and the cross-domain
     verification record existed only in a temporary IDE artifact folder, not
     in the repository. They are now tracked under `artifacts/` (AGENTS.md
     §1.4: documentation/records must be verified in the repo workspace).
  2. `lean_universal_bridge.py::_liberal_parse_formula` returned non-Expr
     sympy parses (tuple/list forms from bracketed heaviside-indicator
     formulas), which crashed `_ast_polynomial_total_degree` during the full
     sweep. It now accepts only single sympy expressions (trimming otherwise,
     failing closed to None), and the degree helper rejects non-Expr input.
  After regeneration, the reproducible record is 323 evaluated / 315 P2 / 8 P0
  / 0 P1: the 6 additional P0s are grammar-space candidates whose declared
  saturation bounds are not provable over their formulas (unbounded degree-3
  products like `(LeadSpill * DispNorm) * EntropyCert`, and heaviside-indicator
  products like `[OFI * I(Lead > 0)] * VolRatio` that fail closed on AST
  refutation) — the former 321/2 record overclaimed those P2s; the traps
  (`PHYS-ENERGY-LEAK-TRAP`, `FIN-CUBIC-RUNAWAY-POLYNOMIAL` → P0) and all spot
  checks (`STRUCT-MASTER-SIZING-PHI`, `GA-D3-CHAMPION-U90_H60M`,
  `TRADING-SCORE-N90` → P2) are unchanged. The test contract pins the
  regenerated record; no preregistered threshold changed.

### Verification notes
- `tests/test_dyn_grammar_space.py`: the frozen d5-target pin was stale — it
  still pinned the backward window `s.shift(-1).rolling(5)` (80% leakage into
  the "future" target); it now pins the shipped forward-only
  `s.rolling(5).mean().shift(-5)` (aggregate over s[t+1..t+5]) plus NaN-tail and
  finiteness invariants.
- F2 (basis screening holdout leak) and F3 (mismatched SS_tot mask) were
  verified already remediated on disk in
  `apedp_dyn/discovery_l2/multiterm_engine.py` (full-series causal evaluation;
  masked-partition `ss_tot` on `y_ho_m`), pinned by
  `tests/test_audit_open_flaws_remediated.py`.
- **F8 — `apedp_dyn/epistemic/pde_discovery.py`:** Added missing spatial stencil
  term `u_xxx` and polynomial term `u^3` to the non-uniform PDE holdout evaluation
  library, aligning with 1D/2D discovery terms.
- **F9 — `apedp_dyn/proposers/distributed_lag_interaction_proposer.py`:** Replaced
  single `lag_state["buffer"]` with per-column buffer dictionary `lag_state["buffers"]`
  so multi-column lag features do not overwrite each other during streaming evaluation.
- **F10 — `apedp_dyn/proposers/hawkes_orderflow_proposer.py`:** Eliminated full-frame
  mean volume and price-change leakage in arrival weight normalization by using
  fit-time reference scales or causal expanding means.
- **F11 — `apedp_dyn/molecular/se3_flow.py`:** Symmetrized bond pair exclusions in
  `_non_bonded_pairs` so reversed bond tuples `(j, i)` are correctly excluded from
  non-bonded core repulsion.

## [Unreleased] — forensic audit remediation, §4.1 freeze re-mint

- `apedp_dyn/sandbox/multi_asset_signals_sandbox.py` is pinned by
  `APEDPX_FROZEN_v1_4.json` (blind-protocol referee freeze). The §4.1 remediation
  (purge `.bfill()` lookahead from the sandbox fit path → causal forward-fill)
  required editing that pinned file. Freeze protocol: re-mint the recorded
  SHA-256 with the rationale logged in the record's `remint_log`; referee logic
  itself is untouched and re-verified by `test_dyn_blind.py`.
- `test_no_lookahead_fill_in_proposer_fit_paths` now scans all of
  `apedp_dyn/` and flags backward fill only (`bfill`/`backfill`), the actual
  lookahead defect class; forward fill is the prescribed causal remediation.

## [Unreleased] — External-audit remediation: denominators, lookahead, honest referees, path & docs integrity (2026-10-19)

Every item below was reproduced against the pre-fix code and is pinned by a regression test that fails before the fix. No preregistered threshold changed; the canonical spec hash is untouched.

### Statistical/dimension fixes
- `apedp_dyn/sandbox/blind_benchmark.py`: `fpr_blind` now divides false promotions by UNWARRANTED negatives (its documented meaning); the previously conflated promoted-denominator quantity is reported separately as `fdr_blind` (fp / promoted), and `by_domain` rows carry both `fpr` and `fdr`. The binomial `fpr_upper95` bound uses the negatives denominator. `tests/test_dyn_blind.py` pins the distinction (fin sweep: FPR 0.5 vs FDR 1.0) in `test_fpr_and_fdr_use_distinct_denominators`.
- `apedp_dyn/epistemic/novelty_adjudication.py`: HAC long-run variance uses the canonical Newey-West divisor `n` for every autocovariance term (previously `n-k` for `k>0`, which is not PSD); the 1e-12 variance floor is retained and disclosed. Adversarial check over 2000 random 6-column draws: non-negative variance in every draw (min 0.00139).
- `apedp_dyn/epistemic/trial_ledger.py`: selection effects are now charged to the DSR, not merely disclosed. `effective_trial_count` groups records by `search_id` and counts `max(recorded_candidates, selection_pool_size)` per group, so a proposer that searched K candidates and emitted only its best is charged K trials; `dsr_with_selection_adjustment()` reports the adjusted `n_trials`, `naive_trial_count`, and `hidden_trials`. Malformed pool sizes and empty-ledger calls fail closed. `InvariantTransferProposer` now stamps a shared per-call `search_id` alongside `selection_pool_size`/`selection_rank`. Regression tests pin the 500-of-1 charge, shared-pool deduplication, and both fail-closed paths.
- `apedp_dyn/video/eval_gate.py`, `rust/apedp-kernel/src/video.rs`, `apedp/rust_bridge.py`: SSIM stabilizers are the canonical squared constants `C1=(0.01)^2=1e-4`, `C2=(0.03)^2=9e-4`; all call sites, kernel unit tests, and Python/Rust parity tests agree. Legacy disclosure: benchmark numbers recorded before 2026-10-19 (including the 0.514 SSIM row in this file) used the unsquared 0.01/0.03 values; historical measurements are disclosed, not rewritten.

### Lookahead / faithfulness fixes
- `apedp_dyn/proposers/regime_switching_proposer.py`: removed backward-fill in both the fit path and the scorer. Volatility warm-up rows (first `min_periods-1 = 4`) are no longer fabricated from future observations; they are excluded from fitting (`metadata.samples_metric_missing`) and score as NaN. Regression test pins the 4-row warm-up and exact formula coefficients.
- `apedp_dyn/proposers/distributed_lag_interaction_proposer.py`: column roles come from an explicit `column_spec` recipe map, not substring parsing (`_X_`/`_lag` collisions). Coefficients are mapped back to raw scale (`raw_w = coef/x_std`, `bias = intercept - sum(coef*x_mean/x_std)`) so the emitted formula scores the fitted model identically: fixed R^2 0.995982 vs -1249.5174 under the legacy mismatch.
- `apedp_dyn/ode/referee.py`: rewritten to adjudicate from a real RK4 holdout rollout (building RHS terms from the reported monomials, with divergence flagged as `UNSTABLE_FORWARD_SIMULATION`). The discovery-side in-sample `mean_system_r2` is never used as evidence; missing equations or fewer than two holdout rows fail closed as `INSUFFICIENT_EVIDENCE`; unknown state references raise. New tests pin wrong-structure rejection despite in-sample 0.999, unstable rollout, and the fail-closed path.
- `apedp_dyn/proposers/hawkes_orderflow_proposer.py`: `TAIL_RISK` is declared only when the binary surge threshold actually binds on the fit data; a degenerate threshold now emits `CONTINUOUS_FACTOR` with standardized intensities (`metadata.score_mode` discloses the choice). `rime_ad_proposer.py`: an evaluator failure now returns NaN instead of fabricating a function via substring matching. `molecular/se3_flow.py` clamps near-coincident atom distances (`_MIN_PAIR_DISTANCE = 1e-8`) in energy and gradient; `audio/spectrogram_transport.py` returns NaN R^2 for zero-variance targets instead of dividing by zero.

### Rust kernel hardenings (rebuilt, 180/180 lib tests)
- `flow_head.rs::erf_hp`: inputs whose square overflows (`|x| > ~1.34e154`, `+/-inf`) return the saturated `+/-1` instead of NaN from `-inf + inf`; NaN propagates. New saturation test.
- `ode.rs`: stages 3-7 now perform the same `h < min_h` underflow check as stage 2, so a divergent system terminates with an explicit step-control failure instead of burning `max_steps`. New regression test.
- `genetic.rs::score_candidate_affine`: two-pass variance (outputs are already materialized) removes the catastrophic-cancellation misclassification of near-constant predictors with large offsets; a pure constant can no longer receive a spurious slope. New regression test on `Const(123456.789)`.

### Boundary & documentation integrity
- `CandidateProposal.declared_scale_invariant` is documented as declarative and unenforced (ARCHITECTURE.md principle 5; `proposer_interface.py` field note), and `tests/test_audit_open_flaws_remediated.py` now fails if any production module reads it in control flow. `InvariantTransferProposer` discloses search size on every candidate (`degree_trials`, `search_id`, `selection_pool_size`, `selection_rank`, `dev_side_gates`), making the dev-side order-statistic bias explicit and machine-chargeable by the trial ledger.
- ARCHITECTURE.md §3.2 now describes the actual burn mechanism (`record_data_exposure(..., CONFIRMATION_HOLDOUT, outcome_status="FAILED")` plus terminal lifecycle states), §4.1 inventories all frozen generations (v1/v1_1/v1_2/v1_3), and §4.2 records the path policy now implemented by `apedp_dyn/conf/paths.py` (`APEDP_REPO_ROOT`, `APEDP_LEAN_ROOT`, `APEDP_ARTIFACT_DIR`, repo-relative fallbacks). All `/home/<user>/...` literals were removed from `apedp_dyn/`, `scripts/`, and tests; the Lean bridge resolves its output root through `APEDP_LEAN_ROOT`. The live documentation index (`docs/DOCUMENTATION.md`) converted all 272 `file:///home/...` links to repository-relative links, and `test_documentation_links_exist_on_disk` now validates every relative link (301 checked) and rejects absolute `file://` links outright; archived historical documents retain their original provenance URIs by design.
- `docs/DOCUMENTATION.md`: the entire `epistemic/active_disambiguation` package (10 modules) and all remaining `discovery_l2` modules are now indexed, plus `conf/paths.py`. `tests/test_documentation_and_spec_integrity.py::test_orphan_module_detection` now scans the whole `apedp_dyn` tree recursively against a pinned debt ledger that may only shrink. Core packages (gates, ode, pde, market, video, estimation, model, harness, fitting, simulation, validation, experiments), the frozen root sweeps, the 18-problem benchmark catalog, and all 67 historical sandbox harnesses are now indexed: the debt ledger is EMPTY (125 -> 67 -> 0) and there is no pin left to mask a regression. Guard-set/referee-side sandboxes are tagged `production`, UI-exposed ones `experimental`, and forensic/CPMD harnesses `audit-only`.
- `apedp_dyn/sandbox/APEDPX_FROZEN_v1_3.json`: guard set re-minted (supersedes v1_2) after the frozen-set `ui/server.py` drag-route change; `verify_referee()` passes and the record carries its post-mint test outcome.
- `apedp_dyn/sandbox/APEDPX_FROZEN_v1_4.json`: guard set re-minted (supersedes v1_3) after the live trial-ledger/promotion-lifecycle wiring in `trading_signals_sandbox.py`; the only changed guard file is that sandbox, no blind case imports it, and the blind suite re-passes at 9/9 with identical metrics. `verify_referee()` passes; v1/v1_1/v1_2/v1_3 remain on disk as historical records.

### Live wiring of the trial ledger and the promotion lifecycle
- `apedp_dyn/sandbox/trading_signals_sandbox.py` (a live UI-driven pipeline): the GA search now records EVERY fitness evaluation in a `CandidateTrialLedger` (in-sample Sharpe per evaluated chromosome, shared `search_id`), and the holdout Sharpe is deflated by that real multiplicity via `dsr_with_selection_adjustment` before it enters the evidence bundle (`metrics.dsr_selection_adjusted`, `trial_multiplicity` block with recorded/effective/hidden trials and expected-max-SR). The six-stage `PromotionTracker` now runs on the live candidate and honestly refuses at stage 1 for GA strategies (no Lean-verified law, no dimensional analysis), so `promotion_lifecycle.is_halted` is true and economic validity is never reached on unverified mathematical consistency. Measured example (20 pop x 5 generations, seed 7): 95 charged trials, holdout Sharpe -0.13, expected max SR 1.93 under multiplicity. New regression test pins the charge, the deflation ordering, and the stage-1 refusal.
- `apedp_dyn/epistemic/trial_ledger.py`: selection-adjusted multiplicity accounting (`effective_trial_count`, `dsr_with_selection_adjustment`) with fail-closed handling of malformed `selection_pool_size` values and empty ledgers.

### Recurrence guards & test-environment honesty
- `tests/test_audit_open_flaws_remediated.py` adds two static guards over runtime code: no back/forward-fill in proposer fit paths (`test_no_lookahead_fill_in_proposer_fit_paths`) and no concrete host home-directory literals outside container paths (`test_no_host_specific_home_paths_in_runtime_code`, AST-aware so docstrings are ignored).
- `tests/test_dyn_lean_universal_bridge.py`: the Docker-Mathlib test now skips with an explicit reason when the required image is not present locally (a 25 s execution budget cannot include a multi-GB pull). The local Lean 4 toolchain was restored (v4.35.0) after removing failed elan download artifacts; the suite is 10 passed / 1 environment skip.
- `tests/test_dyn_ui_studio.py`: the W2_2019 endpoint pin was synced to the audited sole-survivor contract from `tests/test_dyn_real_market_l2_sweep.py` (unique support plus holdout/whiteness/complexity guards) instead of the retired multi-hypothesis outcome.

### CI coverage and local gates (rating remediation)
- `.github/workflows/test.yml`: CI now triggers on EVERY branch (`branches: ['**']`) and on PRs targeting `stage`. Work happened on `stage` for an entire session with no CI coverage because that branch was absent from the trigger list, which is how a broken working tree survived unnoticed.
- `.pre-commit-config.yaml` (new): a local gate for UNCOMMITTED work — ruff on changed files, `verify_referee()`, the import smoke tier, and the documentation/audit guards. CI only ever checks out committed files, so this is the only thing that can catch a broken tree before it is committed.
- `scripts/check.sh` (new): `--quick` runs the fast tier; the default runs the complete dynamic regression battery.

### Lookahead remediation completed in the multi-asset panel
- `apedp_dyn/sandbox/multi_asset_signals_sandbox.py`: the 20-day rolling volatility is computed causally (`ffill`, was `bfill`, which back-filled the first 4 rows from future observations). `ffill` cannot fill leading NaNs, so the undefined warm-up rows are now EXCLUDED from every aligned array via `_drop_volatility_warmup` (returns, prices, features, cross-median) instead of poisoning the panel with NaN or being fabricated. Applied at all three panel constructions (30-asset two-tier analysis, epistemic adjudication, 90-asset policy-grammar audit).
- Honest consequence, re-pinned: with causal volatility the CAN-G gate now DETECTS the cross-asset ranking trap the fabricated volatility used to mask (naive pooled Sharpe -0.095 vs two-tier 1.116; within-class crypto 1.63 vs equity 0.20). The candidate still promotes at C3_STRUCTURAL because stratified allocation is what makes it admissible. `tests/test_dyn_trading_sandboxes_epistemic.py` asserts the trap detection, its reason string, and the stratified-vs-naive ordering.
- `apedp_dyn/sandbox/APEDPX_FROZEN_v1_5.json`: guard set re-minted (supersedes v1_4) for that file; blind outcomes identical (no blind case imports it), `verify_referee()` passes.

### Discovery-path multiplicity wiring
- `apedp_dyn/sandbox/real_market_l2_sweep.py`: every candidate the L2 engine scores is now recorded in a `CandidateTrialLedger` (shared `search_id`, true evaluated count, per-candidate rank and in-sample loss), and `WindowDiscoveryRecord.trial_multiplicity` exposes the pool. Measured on W2_2019: 3195 evaluated, and the promoted `vol - 1/ma_dev**(3/2)` ranks 217th by in-sample loss — the promotion is holdout-adjudicated, not best-fit, which is exactly the selection bias the disclosure exposes. `CandidateTrialLedger` gained an explicitly named `selection_statistic` field (Sharpe for strategy searches where DSR applies; in-sample loss for equation searches where it does not) and tolerant loading of older ledger files.
- `apedp_dyn/epistemic/pde_discovery.py`: `discover_pde_nonuniform` was verified to run discovery on the develop partition only and to score the frozen operator on holdout rows (the F841 findings that prompted the review were stale). Locked by `tests/test_dyn_coupled_and_2d_pde.py::test_nonuniform_discovery_never_fits_holdout_rows`, which plants a different decay rate inside the holdout and asserts that corrupting the holdout rows cannot change the discovered formula or coefficients while the holdout R^2 does move.
- `apedp_dyn/ode/referee.py` pin reconciliation: a TERM referencing a state absent from the holdout raises (internally inconsistent result); a discovered STATE SET that differs from the holdout is an epistemic refusal (INSUFFICIENT_EVIDENCE). Both paths are now pinned separately in `tests/test_dyn_pde_ode_packages.py`.
- Latent bug fixed: `apedp_dyn/robotics/multi_agent_transport.py` referenced an undefined `_TWO_PI` in the support-ring branch, reachable only with >= 3 agents, so every existing test passed while that branch would have raised NameError. Constant defined, `np.array` default hoisted, and a 3-agent regression test added.

### Open verification item (flagged, not resolved)
- The JetFlow-Video matched-comparison numbers recorded in DOCUMENTATION.md (jet_tt_2step 10.74 dB PSNR / 0.514 SSIM / 0.275 temporal smoothness vs diffusion baselines) were produced while the incorrect 4-probe Jet2 acceleration was in the working tree. The reverted (fused-probe) formulation is the mathematically correct one, so those benchmark artifacts should be re-generated before being cited; training was NOT re-run in this session and the numbers are therefore flagged as unverified against the corrected stepper.

### Deep forensic audit remediation (L2 / proposers / Rust kernels / UI server)
- `rust/apedp-kernel/src/video.rs`: `ssim_frame` with `win == 0` passed the unsigned size guard (`h < 0` is never true) and then incremented the window loops by zero forever — a 100%-CPU hang reachable from Python. Now rejected explicitly. `compute_optical_flow_divergence` clamped the neighbour index at borders, so every border derivative spanned one cell but was divided by `2*dx` (halved); borders now use one-sided differences. New tests pin both.
- `rust/apedp-kernel/src/flow_head.rs`: `loss_and_v_grad` on a degenerate plane (H<2 or W<2) reached `adv_loss /= n_all` with `n_all == 0.0` and silently returned NaN; it now returns an explicit error.
- `rust/apedp-kernel/src/rime/high_dim_pde.rs`: the viscosity/diffusion coefficients were validated for finiteness only; a negative `nu` is the backward heat equation (unbounded high-frequency growth) and is now refused.
- `apedp_dyn/discovery_l2/ast_generator.py`: binary-operator pair enumeration was asymmetric (n1 from the new frontier only), so `n_old/n_new` was unreachable and the commutative "the reverse pair will create it" guard dropped BOTH orderings; self-products were excluded outright, making squares impossible without `pow`. Pairs are now enumerated symmetrically with both orderings for the non-commutative operators. Subtraction orderings remain deduplicated **by design**: the basis fingerprint orients sign, so `theta*f` and `-theta*(-f)` are one basis function under a free coefficient. `collect_variable_names` now recurses into `RollingOpNode` (was `frozenset()`, which misrouted univariate rolling terms into the multivariate interaction quota). 5 new tests.
- `apedp_dyn/discovery_l2/l2_engine.py` and `multiterm_engine.py`: Gate A no longer evaluates the holdout partition (AGENTS.md §1.2 violation — the proposer search was discarding candidates on unseen data). The multi-term basis now evaluates terms on the full causal series (rolling state carries across the split instead of re-warming inside the holdout), tolerates warm-up NaN instead of rejecting every rolling term outright, and fits on the finite overlap with the excluded row counts disclosed in `gate4_status`. 2 new tests (rolling term reaches the pool; Gate A never touches holdout).
- `apedp_dyn/proposers/distributed_lag_interaction_proposer.py`: ElasticNetCV now uses `TimeSeriesSplit` (plain KFold trained each fold on later rows, leaking the future into hyperparameter selection); the unknowable lag warm-up rows are excluded from the fit instead of being fabricated as zeros; the scorer carries an explicit state buffer so streaming chunks shorter than the lag no longer lose 100% of the lag signal (chunked scoring now equals single-pass scoring to 1e-12). `declared_scale_invariant` corrected to False (the model has an intercept and raw-scale weights).
- `apedp_dyn/proposers/regime_switching_proposer.py`: `declared_scale_invariant` corrected to False — the regime split branches on a threshold estimated on the training scale, so rescaling moves observations across it.
- `apedp_dyn/proposers/rime_ad_proposer.py`: the differential-Padé scorer is pole-aware (a vanishing denominator returned 0.0 and poisoned the whole trajectory; it now yields NaN and ends the integration, scoring the unreachable suffix NaN) and bounded (the step count was unbounded — up to ~200k Python iterations per unit distance; capped at 20k with the suffix marked unreachable).
- `apedp_dyn/ui/server.py`: `/api/discover/physics` term reconstruction resolves state series through an explicit per-system dictionary (a term naming an undefined state raised NameError -> HTTP 500) and matches explicit cubic forms instead of the substring test `"3" in t_name`; `/api/benchmarks/trading-np93` no longer overwrites `false_promotion_rate`/`passed_contract` with hardcoded 0.0/True on every response — it serves the frozen measurement and derives the verdict from the recorded counts. 3 new tests (measured verdict, all systems 200, upload wired).
- Audit items NOT reproducible in this tree, verified rather than assumed: `/api/sandbox/upload` already routes to the `custom_upload` branch (200 with a real adjudication), and the audit's `spatial_divergence_2d` / `rime_high_dim_pde_step` symbols do not exist — the corresponding real symbols (`compute_optical_flow_divergence`, `eval_jet2_neural_pde_operator`) were audited instead and the defects they do carry were fixed.
- `apedp_dyn/sandbox/APEDPX_FROZEN_v1_6.json`: guard set re-minted (supersedes v1_5) for `ui/server.py`; blind outcomes identical (no blind case imports it), `verify_referee()` passes. Rust kernel rebuilt (184/184 lib tests).

### Data-integrity: vendored lynx fixture
- `apedp_dyn/sandbox/lynx_hare.py` no longer reads `/tmp/lynx.csv` (an untracked, host-dependent fixture whose contents changed mid-session and made the lynx-cycle sandbox unrunnable). The canonical MacLulich series (R `datasets::lynx`, 1821-1934, 114 annual counts) is vendored at `data/market/lynx_maclulich.csv` with provenance recorded in `data/market/PROVENANCE.json`; `load_lynx()` defaults to it and still accepts an explicit path. `tests/test_dyn_ui_sandbox.py::test_lynx_real_data` now runs against tracked real data (18/18 in that suite).

### Transport-stability reconciliation (audio) and stepper revert
- `apedp_dyn/audio/spectrogram_transport.py::predict_next_frame` integrates the frequency-axis advection `S_t = -c S_f` with a CFL-enforced upwind scheme instead of routing it through the generic attribute-transport stepper: centered Euler/Jet2 differencing of pure advection is unconditionally unstable (von Neumann `|G|^2 = 1 + nu^2 sin^2(k df) > 1`) and accumulates negative spectral magnitudes. `order` is retained for interface compatibility, so `tests/test_dyn_audio_field_transport.py::test_jet2_order_beats_euler_on_accelerating_chirp` was vacuous (both orders identical by construction) and is re-pinned as `test_accelerated_chirp_transport_is_stable_and_order_invariant`: identical results across orders, mean R^2 0.9885, min R^2 0.977, non-negative predictions. The DOCUMENTATION.md row no longer claims a per-frame Jet2 advantage for this path; the stepper-level second-order advantage is pinned in `test_accelerated_ridge_transport_order2_error` (relative error 1.87e-3 -> 8.7e-5).
- `apedp_dyn/video/jet_flow_stepper.py`: reverted an uncommitted 4-probe "fix" of the Jet2 acceleration. It split the material derivative into separate temporal/spatial probes but omitted the `|v|` scaling of the convective term (the kernel composes `(v_t+ - v_t-)/dt_span + (v_sp+ - v_sp-)/two_h`), so the acceleration was wrong: robotics trajectory accuracy regressed (5/6 failing, full-circle error 0.023 vs the 5e-3 contract) and the audio order-2 error rose to 6.9e-4. The HEAD formulation passes the fused characteristic probe `v(z + h v, t + h)` as the temporal term, which already equals the material derivative `dv/dt = dv_t/dt + (v.grad)v`, and keeps `compute_acceleration` and the kernel path in parity. Robotics back to 6/6, molecular 6/6, audio order-2 error 8.7e-5.

### Numeric-exactness repair: periodic-safe variance
- `apedp_dyn/market/lob_transport.py::snapshot_variance`: the wrap-safe rewrite used the mean-resultant approximation `2*(L/2pi)^2*(1-|R|)`, which is only first-order accurate for a finite-width packet and inflated the diffusion constant recovered from `d var_x/dt` by ~5% (the 1e-6 recovery contract in `tests/test_dyn_lob_transport.py::test_statistical_observables_match_parameters` failed). Replaced with the exact mass-weighted variance of the wrapped angular deviation from the mean angle, which is wrap-safe, exact for any packet width, and NaN for zero-mass fields. Suite back to 5/5.

### Quadratic-drag discovery (from the same working tree)
- `discovery_l2/ast_generator.py` bounded canonicalization (`collect_variable_names`), `discovery_l2/multiterm_engine.py` structure-aware quota screen, and `ui/server.py` route defaults (3, 2, 4) + `max_basis_terms=40`; generated powers are now canonicalized (`x*x -> x^2`) instead of cancelled by the quota screen. End-to-end drag sweep recovers `-0.5*v*|v|`-family structure where the pre-fix screen discarded it; 7 new regression tests.

## [Unreleased] — Packaging & platform hardening: standalone wheels, Databricks, kernel-less imports (2026-10-19)

### Standalone-install contract (wheels)
- The six domain wrappers (`packages/apedp-{audio,continuum,distributed,molecular,quantum,robotics}`) now declare `"apedp>=1.0.0"` in both `pyproject.toml` and `setup.py` (12 files). They re-export from `apedp_dyn` at import time, so installing a wrapper alone previously succeeded silently and then failed with `ModuleNotFoundError` at first use. Verified: the rebuilt `apedp-quantum` wheel metadata now carries `Requires-Dist: apedp>=1.0.0`, and a wrapper-only install in a fresh venv fails fast with `ERROR: No matching distribution found for apedp>=1.0.0` instead of installing broken.
- Root wheel (`apedp`) now excludes `apedp_kernel*` from `[tool.setuptools.packages.find]`. The dev tree's `apedp_kernel/` symlink (into the venv's compiled site-packages) was swept into the pure-Python wheel as a wrapper-only package, and a stale `build/lib/apedp_kernel` staging directory from an earlier build leaked it into every subsequent wheel even after the finder was fixed — producing `ModuleNotFoundError: No module named 'apedp_kernel.apedp_kernel'` on import in kernel-less environments. With the exclude and a cleaned `build/`, the rebuilt wheel contains no `apedp_kernel/`; the extension continues to ship exclusively from its maturin build (`rust/apedp-kernel`).
- New `python-packages` CI job in `.github/workflows/wheels.yml`: builds all seven packages from `packages/*` and asserts each wrapper's `METADATA` contains `Requires-Dist: apedp` and that jetflow-video's does not (it is independently guarded as fully standalone). `packages/**` added to the PR trigger paths.
- Regression tests in `tests/test_dyn_all_subpackages.py`: `test_domain_wrappers_declare_apedp_core_dependency` (both metadata files, per wrapper) and `test_jetflow_video_stays_self_contained` (no `apedp` imports anywhere in its source).

### Databricks as a first-class runtime
- `apedp_dyn/notebook_adapter.py`: `PlatformType.DATABRICKS` detected via `DATABRICKS_RUNTIME_VERSION` / `DATABRICKS_CLUSTER_ID` (before the CI env markers, so Databricks job compute is not misclassified); `is_databricks()` exported through `apedp_dyn/env.py`. Storage routing: temp files under `/local_disk0/apedp_temp` (node-local NVMe-class scratch), persistent storage at `/dbfs/FileStore/apedp_storage` when a DBFS mount is present, otherwise the driver-local directory with a high-visibility EPISTEMIC RISK warning (AGENTS.md §1.2: driver-local storage dies with the cluster and erases burned holdouts). Worker governor: one worker on GPU clusters, `min(cpu_count, 16)` on CPU clusters. Four new tests (detection by both env markers, DBFS-persistent and ephemeral-fallback paths with warning); `tests/test_dyn_notebook_adapter.py` is now 30/30.

### Kernel-less imports become honest
- New `apedp_dyn/kernel_shim.py`: a `missing_kernel` sentinel exposing the same contract as the absent extension. The five modules with bare module-level `import apedp_kernel` (`ode/dispatch.py`, `epistemic/coupled_discovery.py`, `epistemic/pde_discovery.py`, `epistemic/high_dim_pde.py`, `proposers/rime_ad_proposer.py` — the last two in the eager import chain via `apedp_dyn/ode/__init__.py` and `apedp_dyn/pde/__init__.py`) now guard the import and fall back to the sentinel, instead of exploding `import apedp_dyn` with a bare `ModuleNotFoundError` on any machine without the native extension (masked in development, where the kernel is always present, and contradicting the documented pure-Python fallback story).
- Behavior: `import apedp_dyn` always succeeds; probing stays attribute-safe (`hasattr(ak, "__file__")` is False, `repr()` works); calls into kernel-only engines raise an actionable `RuntimeError` naming the prebuilt-wheel / `maturin develop --release` install step. There are no pure-Python implementations for those engines, so refusing the call — not guessing a result — is the honest failure mode.
- Verification (fresh venv, kernel absent): `import apedp, apedp_dyn, apedp_quantum` succeeds; the DMRG reference engine runs (`dmrg_ground_state` n_sites=4 energy -1.616025 vs exact diagonalization); `discover_ode` raises the documented RuntimeError. Regression tests added to `tests/test_dyn_all_subpackages.py` (sentinel probe-safety + subprocess import-chain test that blocks `apedp_kernel` via `sys.modules` and asserts the actionable call-time error).
- `docs/DOCUMENTATION.md` module registry: Databricks behavior added to the notebook-adapter row (30/30) and a new row indexes `apedp_dyn/kernel_shim.py` with this verification evidence.

### Test-suite reporting (machine-verified by tests/test_dyn_reporting_integrity.py)

- The Python RIME/repair/referee/STEM suite total is **212 tests across 43 files**, counted by importing each file in tests/test_dyn_reporting_integrity.py and assert-verified in CI. Raised 211 -> 212 after `test_dyn_trading_universe_sweep.py` gained a test function post-mint (constant bumped by re-running the assertion test, per its own rule). Earlier prose quoted a 74-test subset of 11 files; that figure is a scoped subset, not the suite total, and is superseded by the machine-verified number.
- Named exclusions from the wider `tests/` directory (not silently omitted):
  - `tests/test_all_system_equations_lean_proofs.py::test_spot_check_lean_kernel_compilation` — environmental: the Lean kernel compile exceeds the test's 15s budget (pre-existing).
  - `tests/test_dyn_lean_universal_bridge.py` — same environmental Lean cause.
  - `outputs/nano_ds10_freeze.json` — PIN RECONCILED: previously reported as drifted at d37d0f1c...; pin in `apedp_dyn/epistemic/frozen_spec.py:40` was reconciled to `b809ddcba1eefff2f7d3f21cc21247467c60dcda2f482ba0df1b528d6a28376f`, verified by `test_all_frozen_records_verified_against_pinned_hashes`.

### Documented Residual Design Boundaries & Preregistration Rationale
- **Standalone Verification Harnesses (`tests/test_dyn_rime_scope_refusal_mv_v2.py`)**: Collects 10 dynamic verification cases for multivariate scope refusal v2. Intentionally maintained as an independent dynamic test suite rather than appended to the 43-file / 212-test frozen RIME baseline to avoid mutating the historical suite accounting constant.
- **Stage 6 Parameter Defaults (`apedp_dyn/epistemic/promotion_lifecycle.py:177-178`)**: Default literals (`prospective_hurdle = 0.70`, `max_dd_ceiling = 0.25`) in `verify_stage_6()` reflect the standard preregistered real-market promotion barrier; preserved to maintain contract continuity across callers.
- **Trial Multiplicity Minimum Variance Regularizer (`apedp_dyn/epistemic/trial_ledger.py:122,132`)**: The `min_var = 1e-4` floor in `compute_expected_max_sharpe()` ensures that degenerately zero-variance trials cannot evade Bailey & López de Prado multiple-testing haircut penalties.
- **Joint Literature Span Projection (`apedp_dyn/epistemic/novelty_adjudication.py:194-203`)**: Unadjusted sample $R^2$ is intentionally retained for the linear span projection of literature baselines. This provides a strictly conservative bound against premature structural novelty attributions by ensuring the baseline space spans maximal variance.
- **NCM Naming & Scope Boundary (`apedp_dyn/proposers/ncm_structural_proposer.py`)**: "Next-Concept Modeling" is a naming inspiration only. The source work (Kumbong et al., arXiv:2610.14162, cs.CL) is an NLP text-generation method (multiscale residual VQ-VAE over text tokens; a transformer emitting one code level at a time). No part of that architecture — codebooks, hierarchical quantization, learned latents, transformer, parallel generation — is implemented here, and its 10x-31x speedups are against diffusion language models on text benchmarks, applicable to no RIME/kernel search path. This module is a classical two-tier archetype-fitting baseline (Ridge L2 + hard coefficient thresholding) verified by parameter-recovery tests. Earlier claims of a 10x-30x search speedup, L1/L2 penalized pruning, qualitative-invariant evaluation, and `production` status were fabrications; status remains `experimental` per AGENTS.md §3, and the docstring now states the actual architecture and scope.

## [Unreleased] — Audit hardening: fail-closed referee defaults, neutral floor rules, status-tag enforcement (2026-10-18)

### Fabrication fixes
- `apedp_dyn/epistemic/agnostic_adapter.py`: removed the fabricated metrics fallback (returned `in_sample_r2=0.95, holdout_rmse=0.02` when a candidate had no callable and no recorded metrics). Unrecorded declared invariants with neither evaluator nor callable now evaluate as unverified (fail-closed) instead of defaulting to pass, and unhandled invariant kinds are unverified instead of passing. `no_model_callable`/`execution_error` now report `in_sample_r2=-999, holdout_rmse=999` per the module's failure convention.
- `apedp_dyn/sandbox/agnostic_domain_sandbox.py`: the `volume_density_conservation` invariant now carries a real computed evaluator (`volume_eval_valid`, spatial flatness of predicted density) instead of relying on the fabricated default pass; Arm C VRR 4/4 and FPR 0/4 are preserved with the real evaluator. Regression tests added in `tests/test_dyn_agnostic_adapter.py` (8 tests).

### Boundary fixes
- `apedp_dyn/grammar/floor_rules.py` (new): `fit_floor_rule` and `floor_rule_candidate` moved verbatim out of `apedp_dyn/epistemic/prior_guided_search.py`; both the prior-guided search adapter (which keeps `_fit_floor_rule`/`_floor_rule_candidate` aliases for compatibility) and `apedp_dyn/proposers/rime_ad_proposer.py` now import them from the neutral grammar layer, so a proposer no longer reaches into referee-package private helpers (AGENTS.md §1.1). The remaining epistemic imports in the RIME-AD proposer (`profile_dataset`, `staircase_prior`) are public dev-only diagnostics whose modules document the §1.1 boundary; they are read-only, never touch holdout rows, and never promote.
- Lean bridge status demoted `production` -> `experimental` in the module registry with an explicit `Status:` line (`apedp_dyn/epistemic/lean_universal_bridge.py`) and the environment gate recorded in its capability row: end-to-end verification requires an external Lean 4 toolchain. The JetFlow-vs-diffusion matched-benchmark registry row was corrected `audit-only` -> `experimental` to match the harness's own declared status.

### Enforcement
- `tests/test_documentation_and_spec_integrity.py::test_registry_status_tags_match_code_status_labels` (new): machine-checks that every registry status tag is one of the four canonical AGENTS.md §3 tags and matches the linked module's docstring `Status:` line wherever present, so documentation can no longer certify a stability level the code does not declare. Registry preamble in `docs/DOCUMENTATION.md` documents the convention.

### Test-hygiene fixes (flaw sweep follow-through)
- `tests/test_jetflow_video_sampler_direction.py`: the two stubs in `_tiny_model`/`_Probe.attach` were stale against the current model call contract (`compute_flow_guided_velocity` now passes `plucker_rays=`, `steer_velocity_field` now receives `depth_engine=`). All five sampler-direction contract tests were crashing on `TypeError` rather than testing anything; with the stubs updated they execute and pass, so the t:1->0 direction fix is now actually verified (starts at t=1 from fresh noise, |z| grows by e under v=-z) instead of merely asserted.
- `benchmarks/run_jetflow_cpu_export_benchmark.py`: `torch.onnx.export(..., dynamo=False)` is now passed only when the installed torch accepts the kwarg (torch >= 2.6; the env here is 2.3.1, whose exporter always uses the legacy path), and the scratch dir uses `tempfile.gettempdir()` instead of a hardcoded `/tmp`. Unblocks the three ONNX export/quantization tests (`tests/test_dyn_jetflow_cpu_export.py`, previously 1F + 2E).
- Weak "no mock losses" assertions replaced with recomputation contracts: `tests/test_dyn_l2_open_discovery.py`, `tests/test_dyn_materials_l1.py`, `tests/test_dyn_quantum_sandbox.py`, `tests/test_dyn_jet_flow_sota_path.py` now recompute recorded RMSE/losses/drifts from recorded residual series and require exact agreement (1e-9), require finite strictly-positive losses instead of `>= 0.0`, and require the Helmholtz-Hodge projection to actually reduce mean divergence versus the raw field it was applied to.

### Referee-probe fixes and red-test reconciliation (2026-10-19)
- `_invariance_verdict` (`apedp_dyn/epistemic/rime_repair_engine.py`): dev slices are now contiguous and order-preserving. The previous `RandomState(0).permutation` shuffled rows before slicing, which contradicted the head/tail temporal split the pair evaluation itself performs and destroyed the serial structure that lag-family findings and repairs are about; shuffled slices fabricated mixed-regime subsamples. No test or caller depended on the shuffle.
- `_slice_pair_key` now honors its own docstring contract ("with no gates"): slice-level `generate_repairs` runs with `r5_emit_unidentified=True`. The R5 identifiability gate is a pooled-fit stability criterion, not a reproducibility criterion; with it active, every 96-row slice was judged ill-conditioned a priori and the invariance probe collapsed to NONE with zero information (verified: rel-stderr b ~ 0.19-0.21 per slice vs 0.029 pooled). Identifiability status still travels with each slice candidate.
- Red-test reconciliation for the three tests that predate the cross-slice invariance gate (`test_option_2_epistemic_repair_engine_end_to_end_repair`, `test_options_123_under_2pct_and_10pct_noise`, `test_selective_risk_qec_world`): their configs now run with `invariance_slices=0` (explicit, commented). Under the fixed probe both frames genuinely fail cross-slice structural reproducibility (slices re-propose structurally different originals — state vs lag forms; best 2/5 on the QEC world), so the fail-closed AMBIGUOUS verdict is the correct referee behavior, not a repair-engine regression. The repair mechanics the tests exist to pin are unchanged (verified: with the gate off the same runs reach IMPROVED with sealed R2 0.987-0.9999 and the expected R5/R2 winners). The flagship `test_option_2` additionally asserts the gated refusal (`AMBIGUOUS` + "cross-slice invariance" note), and the gate's own behavior remains covered by `tests/test_dyn_rime_repair_v012_sensitivity.py` (green, including its explicit AMBIGUOUS-when-not-reproducible accounting).
- `tests/test_dyn_real_market_l2_sweep.py::test_single_window_sp500_calm_sweep` re-pinned after audit, anti-spurious guards preserved (out_of_grammar, fidelity margin <= 1, whiteness, no boundary artifact, `ma_dev` in expression): the grammar now yields a pool of ONE adequate candidate for W2_2019 (`vol - 1/ma_dev**(3/2)`, complexity 4), so the hardened sole-survivor contract promotes it with decisive attribution ("no structurally distinct rival survived scoring"). The previous refusal pin relied on a reorder twin (`ma_dev**(-2)+sqrt(ma_dev)`) that is no longer generated. The apparent ~130x train/holdout error gap is a units artifact: `train_loss` is MSE (3.58e-5 -> RMSE ~ 6.0e-3), `holdout_error` is RMSE (4.65e-3); in RMSE terms the survivor generalizes with the same profile as the historically accepted candidate.

## [Unreleased] — Structural-invariant probe: one check per invariant (2026-10-18)

`_StructuralProbesMixin._probe_structural_invariants` was a 382-line method whose
fifteen checks shared one evidence context by closing over fifteen locals. It is now
an orchestrator plus one method per invariant, with the envelope measured once
(`_EnvelopeScan`) and handed to each check (`_StructuralContext`).

### What changed
- `_EnvelopeScan.from_samples()` computes `lowest`, `highest`, `widest`,
  `first_unevaluated` and `boundary_divergence` once. Fifteen checks used to re-derive
  them from the same lists; a check can no longer decide against a different sample
  than the check before it.
- The five magnitude ceilings (`current_density_ceiling`, `velocity_saturation_ceiling`,
  `yield_strength_ceiling`, `stokes_drag_ceiling`, `saturation_ceiling`) now share
  `_envelope_ceiling()`, so a new ceiling cannot quietly measure a different point
  than its siblings.
- Checks run in check-number order and each returns `None` (invariant not declared --
  nothing counted), `[]` (declared and satisfied) or its counterexamples. Counting of
  `invariants_checked` and counterexample ordering are therefore unchanged, including
  the two checks (dimensional, heat capacity) that can record a probe error instead of
  counting.
- Envelope preparation (constant-model handling plus the `lambdify` sweep) moved into
  `_prepare_structural_envelope()`, which returns `None` for a constant with no finite
  real value so the caller must report it. `_probe_structural_invariants` is 80 lines;
  the longest single check is 43. The file
  grows 382 -> 604 lines because each check now carries its own docstring recording
  what was removed from it (variable-name matching, the printed-text magic-constant
  search).
- `_EnvelopeScan` and `_StructuralContext` are re-exported from the facade, matching
  the existing policy that every name defined in a package module is reachable from
  the package -- the policy `tests/test_package_facades.py` enforces.

### Verification
- **Exact old-vs-new equivalence.** The pre-split method (from commit `919d4c8`) and
  the new one were run over 31 cases -- every invariant alone, all fifteen declared at
  once, constant models, `1/x` (non-real at 0), two-symbol envelopes, and three
  dimensional paths including an undefined unit -- comparing the serialised
  counterexamples and the full metrics dict: **0 mismatches**, 40 counterexamples, all
  15 probes exercised. The harness also verified it loaded the pre-split source (no
  `_EnvelopeScan`) rather than re-testing the new code.
- **Behavioural battery unchanged:** `/tmp/falsifier_fingerprint.py` still digests
  **`8c4c4f17a3597525`**.
- `tests/test_structural_invariant_coverage.py`: 38 tests still pass with their
  expectations unchanged (not re-pinned).
- The 11 test files that import the split packages: **186 passed**, exit 0.
  `ruff check apedp/falsification` clean, exit 0.

## [Unreleased] — Structural-invariant coverage: a constant model no longer skips the axis (2026-10-18)

Found by a new per-invariant coverage test, not by a refactor; fixed in the probe
the earlier split created (`apedp/falsification/probes_structural.py`).

### The finding
`_probe_structural_invariants` returned early when the expression had no free
symbols, so a *constant* law was declared with invariants, evaluated by no check,
and reported as `invariants_declared = 1, invariants_checked = 0` with an **empty**
`probe_errors` list. Reproduction: `falsify("-1.0", {"x": (1, 2)}, ["positivity"])`
returned no counterexample and no error -- a declared positivity invariant that the
report neither checked nor mentioned. The candidate was still refused, but by the
numerical axis (`INCONCLUSIVE_PROBE_ERROR`), i.e. for a reason unrelated to the
invariant the caller declared. The module's own stated contract ("a probe that
could not decide is an error on every axis, not a pass") was not met on this path.

### The fix
A model with no free symbols is still a model: the probe now evaluates the constant
once and runs the same fifteen checks over it. A constant with no finite real value
(e.g. `1e400`) is reported as an unresolved `EnvelopeSweep`/`NoRealValues` probe with
`invariants_checked = 0`, matching how a model that returns no real value across its
envelope is already handled.

| Input | Before | After |
|---|---|---|
| `-1.0`, `["positivity"]` | `INCONCLUSIVE_PROBE_ERROR`, checked 0, no error, no counterexample | `REFUSED_CANNOT_REPAIR`, checked 1, `PositivityViolationProbe` |
| `2.0 + 0.0*x`, `["positivity"]` | checked 0, no error | checked 1, no counterexample (invariant satisfied) |
| `1e400`, `["positivity"]` | checked 0, no error | checked 0, `NoRealValues` unresolved error |

### Scope and verification
- **Confined to constant models.** The 12-formula, five-axis behavioural battery
  (`/tmp/falsifier_fingerprint.py`) has digest **`8c4c4f17a3597525`** both before the
  refactor and after this fix, with the payloads comparing equal: no non-constant
  model changed a finding. This is why the fix did not disturb the refactor's
  equivalence evidence.
- **New test file** `tests/test_structural_invariant_coverage.py` (38 tests): every
  declared structural invariant is paired with a model that must violate it and one
  that must not, asserting on the probe that fires; plus symbol-renaming, undeclared-
  invariant silence, and the dimensional probe. Expectations come from the
  invariant's meaning, not from a recorded run.
- **Non-vacuous:** restoring the pre-fix early return makes exactly the four
  constant-model tests fail (`4 failed, 34 passed`); restoring the fixed file
  byte-identically returns `38 passed`.
- 14 affected test files + facade + coverage suites: **495 passed**, with only the
  same 2 pre-existing `test_dimensional_invariance.py` failures (P07, E16).
  `ruff check apedp` clean, exit 0.

## [Unreleased] — Split the three god-files into focused packages (2026-10-18)

`apedp/falsification.py` (2322 lines), `apedp/gated_assurance.py` (2116) and
`apedp/benchmarks.py` (1414) are now packages. Behaviour-preserving refactor: no
threshold, probe, verdict, weight or control-flow change — only code movement
plus re-exports.

### New layout
- `apedp/falsification/`: `constants`, `taxonomy`, `counterexamples`, `constraints`, `report`, `falsifier`, `blind`.
- `apedp/gated_assurance/`: `checks`, `contracts`, `mechanistic`, `operational`, `engine`, `benchmark_contracts`, `feasibility`.
- `apedp/benchmarks/`: `types`, `results`, `pathological`, `harness`.

Each `__init__.py` is a facade re-exporting **every** name the old module exposed,
including the names it merely imported (e.g. `apedp.benchmarks.FalsificationReport`,
`apedp.falsification.LimitTimeout`), so all existing import paths are unchanged.
Consequences recorded honestly:
- `__module__` of moved classes now names the implementation module (e.g.
  `apedp.falsification.counterexamples.Counterexample`). No `pickle` use of these
  types exists in the repository, and nothing pins `__module__`.
- Rebinding a *moved* symbol on the facade no longer changes what the
  implementation calls: `tests/test_flaw_fixes.py::test_boundary_axis_surfaces_a_timed_out_limit`
  patched `falsification.limit_with_timeout` and now patches
  `apedp.falsification.falsifier.limit_with_timeout` instead (the module the probe
  actually reads). The assertion is unchanged. Patching `falsification.sp` still
  works because that is an attribute on the `sympy` module object itself.
- `tests/test_evidence_tiers.py` scanned `apedp/falsification.py` for
  `probe_name="..."` literals; it now globs `apedp/falsification/*.py`. Coverage
  and staleness assertions are unchanged, including `len(declared) >= 20`.

### Verification
- **Source is lossless:** the generator compared every non-blank, non-header
  source line of each original file against the concatenation of its emitted
  modules, in order — exact match (decorators included; comment blocks travel with
  the definition below them).
- **API is identical:** a before/after snapshot of each module's `dir()` — kind,
  signature, base classes, dataclass fields and values, class members — reports
  **0 missing, 0 changed** names for all three. The only additions are the
  submodule attributes a package necessarily exposes (`falsification.taxonomy`, …).
- `ruff check` clean on the three packages (I001/E402/F401/F811 resolved by
  pruning headers to used imports, sorting, and a file-level `# ruff: noqa: F401`
  on each facade).
- The 14 test files that import these modules: **451 passed**, with the same 2
  pre-existing `test_dimensional_invariance.py` failures (P07, E16) present before
  and after the split. No new failure attributable to the refactor.

### Follow-up — the case tables moved out of the benchmark class

`apedp/benchmarks/pathological.py` (980 lines) held the class *and* both case
tables inline. The tables are now `apedp/benchmarks/cases.py` (976 lines, data
only: `_get_25_case_suite`, `_get_100_case_suite`), and
`PathologicalEquationBenchmark` keeps only selection behaviour and delegates.
The class API is unchanged — `get_benchmark_cases`, `_get_25_case_suite` and
`get_100_case_suite` are still classmethods with the same signatures, so a
subclass can still override either suite.

- **Byte-identical tables:** the extracted text was compared character-by-character
  against `git show HEAD:apedp/benchmarks.py` (the pre-split monolith, where both
  tables stated their method-body indentation): `identical=True` for both, 308 and
  646 lines.
- **Identical at runtime:** a SHA-256 fingerprint over every dataclass field of
  every case (factories normalised) is unchanged for all three entry points:
  25-case `026febdf62324e5e`, 100-case `8c7a5c2cf322fd19`, `get_benchmark_cases()`
  (= the 25-case suite) `026febdf62324e5e`.
- **API:** the before/after snapshot reports **0 missing, 0 changed**; the only
  new facade names are `cases` and its two private functions.
- `ruff check apedp/benchmarks` clean; `test_package_facades`,
  `test_falsification_and_benchmarks`, `test_benchmark_integrity`,
  `test_adversarial_science`, `test_evidence_tiers`, `test_phase2`,
  `test_untested_module_smoke` **91 passed, exit 0**.

### Follow-up — the falsifier's five probe axes moved into mixins

`apedp/falsification/falsifier.py` was a 1154-line single class. The probe
implementations now live in focused modules:

| Module | Contents | Lines |
|---|---|---|
| `probe_support.py` | `_ProbeSupportMixin`: `_record_probe_error`, `_sample_envelope` | 105 |
| `probes_mathematical.py` | `_MathematicalProbesMixin` | 133 |
| `probes_boundary.py` | `_BoundaryProbesMixin` | 146 |
| `probes_numerical.py` | `_NumericalProbesMixin` | 194 |
| `probes_structural.py` | `_StructuralProbesMixin` (orchestrator + one check per invariant; 604 lines after the follow-ups below) | 604 |
| `probes_empirical.py` | `_EmpiricalProbesMixin` | 77 |
| `falsifier.py` | `IndependentAdversarialFalsifier.__init__` + `falsify()` | 200 |

Method bodies, signatures and docstrings are unchanged; the class is now
`IndependentAdversarialFalsifier(_ProbeSupportMixin, _MathematicalProbesMixin,
_BoundaryProbesMixin, _NumericalProbesMixin, _StructuralProbesMixin,
_EmpiricalProbesMixin)`.

**Consequence recorded honestly: the class's bases and MRO change.** `bases` went
from `(object,)` to the six mixins, so three test subclasses in
`tests/test_benchmark_integrity.py` (`_BlindFalsifier`, `_AlwaysRefusingFalsifier`,
`_RecordingFalsifier`) now inherit the mixins transitively. Everything callers use
is unaffected: every former method is still reachable on the class, the class's own
constructor signature is byte-identical, and the public facade still exports
`IndependentAdversarialFalsifier` from `apedp.falsification`.

**Verification**
- **Behavioural digest unchanged.** `/tmp/falsifier_fingerprint.py` runs a 12-formula
  battery spanning all five axes and hashes verdicts, `survived`, every
  counterexample's `(probe_name, failure_class)`, axes tested, refusal reason,
  `cannot_repair`, evidence tier and metric keys: digest **8c4c4f17a3597525** before
  and after, with the two full payloads comparing equal (`fingerprint payload
  identical: True`). This is a stronger statement than a green test run: no probe
  changed a finding.
- **API surface:** all previously own members are still reachable (`missing: []`),
  the class signature is unchanged, and the only snapshot difference is the intended
  `bases` change.
- `ruff check apedp` clean; all 169+ `apedp.*` modules import.
- The 14 test files that import these modules plus `test_package_facades`:
  **457 passed**, with only the same 2 pre-existing `test_dimensional_invariance.py`
  failures (P07, E16) seen before any of this work.
- `tests/test_flaw_fixes.py::test_boundary_axis_surfaces_a_timed_out_limit` patched
  `falsification.limit_with_timeout`; the boundary probe now reads it from
  `apedp.falsification.probes_boundary`, so the patch target moved there. Assertions
  unchanged. Same class of adaptation as the earlier `falsifier` patch: rebinding a
  moved symbol on a facade does not reach the implementation module.

Docs: `docs/DOCUMENTATION.md` §5.11 records the layout and the facade guarantee.

## [Unreleased] — Comprehensive Remediation of 7 Open Core System Items (2026-10-17)

> **Revert note (2026-10-18):** Items **1, 3, 8, 9, and 10** below were reverted to committed HEAD. Review found the holdout-priority selection ranked rival candidates by confirmation-holdout RMSE — holdout information leaking into promotion selection — and the ESEDS benchmark's promotion FPR (`fpr_promo`) measured 0.333 under it versus 0.0 at HEAD. `adjudicate.py`, `frozen_spec.py`, and `APEDPX_FROZEN_v1_2.json` are restored to HEAD (referee record `da604e6a…`; `DYNAMIC_EPISTEMIC_VERSION` `bootstrap-v3+solefid-v5`); the ESEDS suite passes 8/8 and the version pin in `tests/test_dyn_epistemic_bootstrap.py` is restored to `bootstrap-v3+solefid-v5`. The frozen-record re-pin (`e93e3307…`) and `bootstrap-v4` stamp described under items 8 and 10 are not in the tree; the pass counts quoted under item 10 describe the reverted state only. Items **2 and 4–7 remain in effect.**

### Fixed & Remediated
1. **`adjudicate()` candidate evaluation and holdout priority (`apedp_dyn/epistemic/adjudicate.py`)**:
   - `adjudicate()` no longer aborts on rank-1 gate-4 failure when other candidates successfully passed gate 4; evaluates and promotes the best holdout candidate among gate-4 survivors.
   - When a rival candidate defeats the emitted candidate on holdout ($\Delta < -\tau$), `adjudicate()` no longer falsely reports `multi_hypothesis` (which was caused by an unchecked $\Delta \le \tau$ comparison on negative $\Delta$); correctly evaluates holdout superiority.
2. **`_one_probe()` overfit probe separation (`apedp_dyn/epistemic/scope_refusal.py`)**:
   - Separates rather than declines an emitted candidate that beats an overfit probe on holdout. Bases `r_best` and `r_rival` on out-of-sample holdout RMSE rather than in-sample train loss.
3. **`bootstrap=True` gate 4 residual fallback (`apedp_dyn/epistemic/adjudicate.py`)**:
   - `adjudicate(bootstrap=True)` now falls back to `_residual_series(best)` and `_residual_series(second)` from `gate4_status` provenance when explicit arguments are omitted, running paired block bootstrap and preserving `EpistemicContract.FULL` rather than demoting to `INCOMPLETE`.
4. **`RimeAdProposer._make_dag_candidate` double scaling fix (`apedp_dyn/proposers/rime_ad_proposer.py`)**:
   - Removed double application of slope and intercept (which previously caused measured max error of 8.97).
5. **RIME implicit minimal relation discovery (`rust/apedp-kernel/src/rime/implicit.rs`)**:
   - Converted implicit relation search to a degree-incremental progression ($1 \dots \text{max\_degree}$). For circles with `max_degree=3`, correctly discovers and returns the minimal quadratic relation $x^2 + y^2 - 1 = 0$ instead of higher-order polynomial multiples $xy(x^2 + y^2 - 1) = 0$.
6. **Hawkes orderflow regular bar dynamics (`apedp_dyn/proposers/hawkes_orderflow_proposer.py`)**:
   - Updated `compute_score` to incorporate order flow activity (volume, trades, price volatility) so that regular timestamped bars produce dynamic, informative intensity signals instead of collapsing to constant zeros `[0.]`.
7. **Flow-head test comments & correlation assertion (`tests/test_dyn_flow_head_rust.py`)**:
   - Removed stale `0.25 / 0.47` chaos comments and added explicit Pearson correlation assertion (`loss_corr > 0.99`, measured 0.9985) verifying lockstep convergence between Rust and PyTorch.
8. **Frozen specification integrity**:
   - Re-computed and updated SHA-256 for `apedp_dyn/epistemic/adjudicate.py` in `APEDPX_FROZEN_v1_2.json` and `PINNED_FROZEN_REGISTRY` in `apedp_dyn/epistemic/frozen_spec.py`. Verified with 15/15 green guard checks.

### Follow-up — two residual adjudication defects found while re-verifying items 1–8 (`apedp_dyn/epistemic/adjudicate.py`)

Both were exposed by, not fixed by, the holdout-priority selection in item 1; both were reproduced before the change and re-verified after.

9. **An unverified candidate could "defeat" a validated winner.** The rival set was every other candidate with a finite holdout RMSE, so a candidate with no gate-4 record at all (holdout 0.02) made a gate-4 passer (holdout 0.50) report `INSUFFICIENT_EVIDENCE` — "defeated on holdout by rival" — for evidence the rival never established. Candidates with no gate-4 record are now excluded from the rival set whenever any candidate passed gate 4; when none did, the selection has already fallen back to plain holdout ranking, so provenance is not required there (no behaviour change on that path). The reproduced case now routes to the existing sole-survivor contract, which refuses the passer for failing the pre-registered absolute fidelity ceiling — the same conservative outcome reached for a stated reason.
    - **Scope decision recorded, not smuggled.** The broader rule (rivals restricted to gate-4-*validated* candidates) was implemented, measured, and then narrowed. Measured on the frozen DUFF/COUP/STAB sweep (36 cells): the rival is refuted in 34/36 cells, so under the broader rule nearly every cell falls to the sole-survivor fidelity ceiling — mean recovery 0.694 → **0.472** (false promotions 0 → 0). Rejecting a refuted rival is a different epistemic contract about what a rival is *for* (the resolution rule detects an indistinguishable rival, which by construction would have passed gate 4), and that trade-off belongs to the preregistration owner rather than to an unannounced side effect of a bug fix. The broader rule is therefore NOT applied and is left as a documented open question.
10. **Residual roles were attributed positionally, not to a candidate.** `best` is now the holdout winner, but a bare residual array carries no identity: `dyn_sweep.py` orders its series by train loss (its comment claimed "Rank mirrors adjudicate()"), so the pair bootstrap ran with the roles swapped and refused a decisive winner (`MULTI_HYPOTHESIS` / branch `rival-favored`) — reproduced: train-best holdout 0.60 vs holdout-best 0.05, both gate-4 validated. Gate 4's recorded signed residuals *are* attributable to a candidate, so they now take precedence on the pair path and in the sole-survivor whiteness guard; the arguments remain the fallback for callers whose provenance records no series. After the change the same call returns `UNIQUE_SUPPORT` for the holdout winner with bootstrap branch `confirm` and contract `FULL`, identical to the no-argument fallback path.
    - Consequence recorded honestly: with `best` always the minimum-holdout candidate among the eligible set and rivals a subset of that set, the `Δ < −τ` "defeated on holdout" branch added in item 1 is now unreachable in both branches. It is left in place as a defensive refusal rather than deleted in the same change.
    - Thresholds unmoved: `RESID_RHO_FLAG`, `SOLE_SURVIVOR_GAMMA`, `tau_epi` sources and every canonical literal untouched; no new threshold literals.
    - Re-minted the frozen referee record (second re-pin of `adjudicate.py`): `70834e26…` → `e93e3307…` in `APEDPX_FROZEN_v1_2.json` and `PINNED_FROZEN_REGISTRY` (an intermediate `b0b2d1d4…` was pinned for the two code fixes and superseded within this same change when the version stamp below was edited; no other file was touched between them). `referee_change` and `tree_state` in the record updated with this rationale. The blind runner refused to score the drifted referee before the re-mint (`FreezeDriftError` / "Mint a new freeze record"), as designed — recorded here because the guard firing is the evidence that the protocol works.
    - `DYNAMIC_EPISTEMIC_VERSION` bumped `bootstrap-v3+solefid-v5` → `bootstrap-v4+solefid-v5`: v4 covers the promotion-SELECTION rules (holdout-priority selection from item 1, rival eligibility, provenance-attributed residual roles), which change which candidate is promoted rather than only how the verdict is printed. The pin in `tests/test_dyn_epistemic_bootstrap.py` was updated in the same change — the assertion still pins one exact value, it is not relaxed.
    - Verification: AGENTS.md dynamic suite **97 passed** (exit 0; 95 before the two regression tests added here); adjudication/bootstrap/sweep/tournament/calibration/uq/hygiene/flow-head/RIME-scope modules **104 passed** (exit 0); `cargo test --lib implicit` **4 passed** (exit 0); freeze-integrity and blind-guard tests 9 passed with the re-minted record. Both new regression tests were confirmed NON-VACUOUS by reverting the two hunks and re-running them: they fail on the pre-fix code with exactly the reported defects (`defeated on holdout by rival no_g4 (Δ=-0.48 < -τ=-0.05)` and `MULTI_HYPOTHESIS` where `UNIQUE_SUPPORT` is required), then pass again after restoring the fixed file (sha256 verified byte-identical).
    - `run_sweep()` re-run after the change against the pre-change baseline (both 36 cells, ~65 s each): **1 cell differs** — STAB/1%/rep0 flips `multi_hypothesis` [branch `rival-favored`, truth not promoted] → `unique_support` [branch `confirm`, truth promoted], which is the item-10 role-swap bug being corrected rather than a re-scored benchmark. Mean recovery 0.694 → **0.722**, mean abstention 0.306 → 0.278, false promotions 0 → 0, and no cell loses recovery. No sweep artifact files were regenerated from this run.


### Added — four frozen preregistrations (frozen BEFORE their confirmation runs; harnesses/artifacts committed separately)
- [`docs/PREREGISTRATION_JETFLOW_VIDEO_V02.md`](docs/PREREGISTRATION_JETFLOW_VIDEO_V02.md) (PREREG-JETFLOW-VIDEO-V02-2026-10-17): fresh-seed {101, 202, 303} matched-benchmark replication (H1 SSIM/smoothness ordering, H2 PSNR ≥ 9.5 dB, H3 training-cost ratio ≤ 0.5, H4 structural pass counts) + architecture invariants (strict 3D causality, SE(3)/Plücker exactness, text-encoder order/negation, Noether idempotence, TT fidelity, periodic parity). Margins derive from the disclosed exploratory seed-0 run.
- [`docs/PREREGISTRATION_HIGHER_ORDER_PDE.md`](docs/PREREGISTRATION_HIGHER_ORDER_PDE.md) (PREREG-HOPDE-2026-10-17): two-sided fresh-seed {810, 811, 812} validation — KdV, biharmonic KS-type, and second-order-temporal Klein–Gordon worlds must confirm; explosive positive-`c₄` ultraviolet and null worlds must be refused.
- [`docs/PREREGISTRATION_CROSS_DOMAIN_ENGINES.md`](docs/PREREGISTRATION_CROSS_DOMAIN_ENGINES.md) (PREREG-XDOMAIN-2026-10-17): fresh-instance confirmation batteries for DMRG (ED parity, thermodynamic limit, field arm), LOB transport (8-combo parameter grid, three recovery paths), audio chirp transport, molecular SE(3) relaxation (invariance, equivariance, mirror), robotics unicycle (convergence orders, Lyapunov descent).
- [`docs/PREREGISTRATION_QUANTUM_PROFILER_HOLDOUT.md`](docs/PREREGISTRATION_QUANTUM_PROFILER_HOLDOUT.md) (PREREG-QPROF-2026-10-17): first out-of-sample confirmation of the quantum profiler/coordinate transmuter (Brus power law, Landauer staircase, Schwinger tunneling) with smooth-polynomial-refusal and null guards on fresh noise seeds.

### Added — JetFlow-Video v0.2 pillars (exploratory evidence)
- **Compositional text encoder** (`packages/jetflow-video/jetflow_video/text_encoder.py`): sequence-level `(B, L_seq, D_text)` contextual encoding, negation gating, order-sensitive modifiers, HuggingFace bridge, prompt→physics steering extraction.
- **3D causal spatio-temporal VAE** (`bridge.py`): strict past-only `CausalConv3d`/`CausalResBlock3D` stack, diagonal-Gaussian KL posterior, end-to-end video-batch training.
- **Depth-aware SE(3)/Plücker transport** (`depth_se3.py`): exact Rodrigues/`V(ω)` exponential map, 6D Plücker ray embedding, learned monocular depth + 3D perspective warp with disocclusion source.
- **DC-STT neural velocity/source predictor** (`apedp_dyn/video/decoupled_transport.py`): `NeuralTransportFlowNet` (streamfunction-curl + compressible-potential decomposition, bounded disocclusion source) trained end-to-end through a differentiable transport step with Rust rollout.
- **Rust periodic warp kernel** (`rust/apedp-kernel/src/video.rs::warp_bilinear_2d_periodic`, PyO3 export, bridge `periodic=True`): generator-exact floor-then-mod backtrace indexing; torch / NumPy fallback / Rust parity pinned at 1e-12 in `tests/test_dyn_periodic_boundary_transport.py`.
- **Higher-order PDE hypothesis engine** (`apedp_dyn/pde/higher_order_engine.py`): spatial orders to 4th + temporal orders {1, 2} with variance-normalized BIC selection, proposer/referee separation, and the ultraviolet Fourier-symbol stability guard (rejects explosive `c₄ > 0` biharmonic terms).

### Committed — RIME §16e negative-result artifacts (prereg was `363e8df`)
- `apedp_dyn/epistemic/scope_refusal_mv.py` (multivariate rival-probe construction), `benchmarks/run_scope_refusal_mv_validation.py`, canonical artifact + disclosed void first run, `docs/RIME_SCOPE_MV_REPORT.md`, registered pins `tests/test_dyn_rime_scope_refusal_mv.py`. Recorded outcome: §16e as preregistered FAILS its two-sided validation (control wrongfully refused via c→0.1 linearization; §16c monomial guards undefined); v2 prereg required before any modification.

### Measured — exploratory JetFlow matched benchmark, post-remediation seed 0 (artifact `benchmarks/jetflow_vs_diffusion_matched_results.json`)
- `jet_tt_2step` 10.74 dB PSNR / 0.514 SSIM / 0.275 temporal consistency at 8 structural passes per clip; SSIM and temporal smoothness beat `diffusion_matched` (0.477 / 0.365) and `diffusion_frontier_K48` (0.492 / 0.372); PSNR remains 0.23 dB below `diffusion_matched`.
- Training at equal 300 steps: 14.40 s (JetFlow) vs 80.75 s (baseline) — ratio 0.178; final flow-matching loss 0.1528 vs 0.1631.
- Harness caveat recorded: the artifact's `model_passes_per_clip` for `jet_tt_2step` (9.0) includes one warmup generation amortized over `n_gen`; the structural count is 8 (the confirmation protocol resets the counter after warmup).
- AGENTS.md dynamic regression suite on the committed tree: **127 passed**.

## [Unreleased] — DC-STT landed honestly: horizon decay + first learned flow head (2026-10-17)

### Added
- **DC-STT stack committed** (`apedp_dyn/video/decoupled_transport.py`, `rust/apedp-kernel/src/video.rs` warp/divergence/disocclusion kernels, parity-verified bridge wrappers, multi-scene benchmark + artifact, spec). Big-tech comparison claims in the spec (§5) and module docstring softened to their measured scope: transport/reconstruction claims **given** a flow field — no learned prior, therefore no generation-quality claim vs any industrial video model.
- **`benchmarks/run_dcstt_horizon_decay.py` + artifact:** the serving-claim instrument the multi-scene benchmark lacked. Measured (artifact-backed): integrator fidelity vs a 10x fine-substep reference 18–43 dB, **but appearance detail (blur proxy) decays monotonically** — fluid-vortex detail energy falls to 11.3% of initial by H=128 (0.060 at H=256), camera-zoom to 25% — washout from nu-Laplacian dissipation + numerical diffusion. Verdict: current evidence supports **short-horizon transport/upsampling**, not unbounded-rollout serving quality.
- **`apedp_dyn/video/learned_flow_head.py` — first learned component in the DC-STT path:** small CNN inferring per-pixel v(x) from 2 frames, self-supervised (advection-consistency + div² + TV), trained **entirely on CPU** under the wall-clock deadline discipline; `warp_frames` pinned to exact parity with the Rust-bridge NumPy fallback warp (max|Δ| < 10⁻¹²; backtrace sign convention tested by point-delta transport). `tests/test_dyn_learned_flow_head.py` 6/6.
- **`benchmarks/run_learned_flow_dcstt_benchmark.py` + artifact:** evaluation on the shared fluid-domain split (n_clips=10 so the matched benchmark's burnable holdout `fluid_clip_0003/0005` is never trained on — an earlier draft's n_clips=20 default leaked those ids into TRAINING and was fixed before any result was accepted). 90 s deadline CPU training: 1367 steps, loss 4.4e-4 → 6e-5. HONEST VERDICT (single-run): the learned flow head does **not yet beat persistence** in aggregate (38.79/36.06/33.42 dB vs 39.54/37.21/35.24 dB at H=4/8/14); it wins only on one of two holdout clips. Recorded as measured; no tuning to green. The head is the scaffold for re-injection: horizon-decay evidence above shows rollout quality, not throughput, is the binding constraint for long rolls.

### Fixed
- **`rust/apedp-kernel/src/flow_head.rs` — `erf_hp` Maclaurin branch used the wrong series constant** (`2/√(2π)` instead of `2/√π`, smaller by exactly √2), corrupting every GELU nonlinearity in the Rust flow-head port: torch parity failed at max|Δv| 2.7–8.0 (saturated, 254/256 pixels wrong) while a zeroed-input bisection masked it (any index bug is invisible when the gathered input is all zeros). Fixed to `FRAC_2_SQRT_PI = 1.128_379_167_095_512_6`; the Lentz `gcf` branch was already exact. Crisp diagnostics first, then repairs: layer-count bisection (n_layers=1 already wrong at 0.977 corr), a numpy im2col/GEMM mirror verified vs torch to 6.4e-15 (conv + layout + channel order exonerated), and a cargo-test erf grid that localized the defect to ±0.28 at x=±1.45 with correct zeros — the constant was the only remaining suspect. Post-fix: |erf_hp − math.erf| ≤ 3.3e-16 on [-6, 6]; forward parity ≤ 1.5e-14 at n_layers=1/2/3 (hidden=8); `tests/test_dyn_flow_head_rust.py` 4/4 incl. 300-step training and rollout parity; full AGENTS.md regression suite 58/58.
- **Permanent pin added:** `flow_head::erf_hp_grid_tests::erf_hp_matches_the_reference_grid` — 397-point grid ([-6, 6] step 0.05, [1.3, 1.7] step 0.005 both signs at the branch crossover, saturation out to ±12) against f64-exact `math.erf` references, tol 8e-16 (~2 ulp of cross-library noise, six orders below the caught defect), plus an odd-symmetry assertion at 4e-16 and a max-deviation diagnostic line so failures report the guilty (x, ref).
- **`numeric.rs` audited against Python reference grids (2026-10-17):** the claims held everywhere; every claim was re-measured, not trusted. `erf_approx` (A&S 7.1.26) errs 1.388e-7 max vs `math.erf` on [-10, 10] — 93% of the documented 1.5e-7 ceiling, pinned at 1.45e-7; `bessel_j0` errs 1.06e-14 absolute on [-8, 8] (doc: ~1e-12), pinned at 1e-12 with the degradation band [8, 16] pinned BOUNDED at 2e-11 (measured 1.48e-11 at x=15.65); one-sided-Jacobi `singular_values` matches `numpy.linalg.svd` on 8 deterministic embedded matrices, pinned at 1e-10 relative. The mirror-set (metrics/conditioning/QR/Cholesky/precision) stays covered differentially by `tests/test_rust_bridge_*.py`. All pins live in `numeric::reference_grid_audit` (Rust tables generated from Python 3.12 `math.erf`/`scipy.special.j0`; nothing is regenerated in Rust). Two stale docstrings claiming the flow-head GELU uses `erf_approx` were corrected to the grid-pinned `erf_hp`.
- **Degenerate-regime extension of the Jacobi SVD pin:** a second embedded reference module (`numeric::jacobi_degenerate_pins`) covers exact rank-1, duplicate + zero columns, a 1e-13 near-singular pair, a diagonal matrix with zero pivots, and collinear column pairs (condition numbers 4.6e13…1e300). Scaled-absolute tolerance 1e-10·σmax deliberately avoids relative comparison of numerically-zero sigmas — the defect class it guards is a garbage large σ or a diverged rotation. Verified against `numpy.linalg.svd`; full rust lib suite 173/173.
- **Learned-flow head: 2× CPU budget rung measured (`benchmarks/learned_flow_dcstt_2x_results.json`, seed 0, no other change):** 90 s → 180 s deadline (1367 → 4721 steps, train loss 6.0e-5 → 2.5e-5). Aggregate learned-flow PSNR 38.79/36.06/33.42 → **39.08/36.65/34.35 dB at H=4/8/14 vs persistence 39.54/37.21/35.24**: the persistence gap shrinks but does NOT close (−0.75→−0.47, −1.15→−0.56, −1.81→−0.89 dB). Per-clip asymmetry persists and slightly strengthens: clip 0005 beats persistence at every horizon (margins grew to +0.56/+0.64/+0.75 dB); clip 0003 still trails (−1.5…−2.5 dB). Verdict: budget alone is not the binding constraint — the single-2-frame-context self-supervised objective with its dye-renormalization loss floor is; the evidence-backed next lever is supervised flow transfer (teacher-generated pseudo-flows / distillation into the head), not more CPU seconds.
- **Stage 3 LANDED — flow-supervised trainer + the 4× budget rung closes the persistence gap.** Two mechanisms, measured under the repo's wall-clock discipline (seed 0, train-ids-only teacher, holdout untouched):
  1. *Stage-3 supervision* (`learned_flow_head.flow_supervision_loss`, `train_flow_head(..., flow_targets=...)`): the fluid-domain generator's own ANALYTIC field, exposed by `make_fluid_clip(..., return_flow=True)` — train clips only, stacked in the trainer's exact concat order, in head-units (vx, vy) = (v_gen, u_gen) with the swap pinned by a hard warp-agreement test. One real bug caught and fixed en route: the supervision target must be `flow[ti+1]` (the generator advects frames[ti] with field[ti]); the first cut used `flow[ti-1]` (stale by two indices — caught by a magnitude probe that REFUTED the tanh-saturation hypothesis first, max|teacher| = 3.66 < the 4.0 bound). Both artifacts kept: `learned_flow_flowsup_180_results_staleindex.json` and the corrected `learned_flow_flowsup_180_results.json`.
  2. *4× rung* (`learned_flow_dcstt_4x_results.json`, re-run on a quiescent machine after the first attempt was CPU-confounded (kept as `…confounded.json`)): **learned 39.50/37.32/35.35 dB vs persistence 39.54/37.21/35.24** — parity at H=4 and ABOVE persistence at H=8/H=14 (+0.10/+0.11 dB).
  Ladder (all quiescent, both axes recorded; NOTE per-step wall-clock varied 0.038–0.28 s/step across runs from concurrent machine load, so the budget axis is wall-clock, not steps — steps: 1367/4721/1492 at 90/180/360 s):
  | H | 90s | 180s | 360s (4×) | flow-sup 180s (correct idx) | persistence |
  |---|---|---|---|---|---|
  | 4 | 38.79 | 39.08 | **39.50** | 39.12 | 39.54 |
  | 8 | 36.06 | 36.65 | **37.32** | 36.96 | 37.21 |
  | 14 | 33.42 | 34.35 | **35.35** | 34.57 | 35.24 |
  Honest verdicts (recorded, not swept): (a) flow supervision beats self-supervision at EQUAL 180 s budget at every horizon (+0.04/+0.31/+0.22) and produces clip-0003's best numbers ever — but regresses clip 0005 below persistence (the head there converges to ≈near-zero flow), so the aggregate does NOT beat persistence at this budget; the 4× self-supervised rung is the better aggregate buy. (b) `flow_supervision_loss` evaluates the head twice per step (consistency + supervised MSE) — 647 steps in 180 s vs 4721 self-supervised; a shared-forward optimization is open and deferred. (c) teacher construction is train-ids-only by construction (holdout ids 0003/0005 never enter `make_fluid_clip(..., return_flow=True)`).
  Tests: `tests/test_dyn_learned_flow_head.py` 9/9 (swap convention, div-free stencil aligned with `test_dyn_fluid_domain.py`, supervised-beats-self on a known target, targets-shape rejection); `test_dyn_fluid_domain.py` + `test_dyn_decoupled_continuous_transport.py` 9/9 green on the merged tree.
- **`DiffusersVAEBridge` added to `apedp_dyn/video/vae_bridge.py` (Stage-2 wiring of the distillation ladder):** HuggingFace `diffusers.AutoencoderKL` behind the exact `VideoVAEBridge` interface surface (encode/decode frame+video+TT+psnr), lazy optional import with actionable guidance, float32-internal/float64-boundary casting, scaling_factor applied symmetrically, and loud channel-count refusal (no silent remap). Config-only (random-init tiny KL — 0.41 needs the explicit per-block tuple types, pinned after reproducing the truncation bug) and pretrained-weights paths are distinct: `pretrained_weights_loaded` records which, so no artifact can claim a prior it has. The pretrained test probes network/cache and SKIPS (never passes) when weights are unreachable — in this env it is skipped. Tests: `tests/test_dyn_diffusers_vae_bridge.py` 8 passed / 1 skipped; `diffusers==0.41.0` installed into the repo venv.
- **Quick follow-up arc (2026-10-17, same day): shared forward + periodic-protocol rerungs + trained-weights kernel parity.** (i) `flow_supervision_loss` now evaluates the head ONCE per step (shared forward between the consistency warp and the supervised MSE); numerically identical to the naive composition — pinned at 1e-12 (`test_flow_supervision_loss_equals_the_naive_composition`, suite now 10/10). (ii) The concurrent session's periodic-warp default in the merged benchmark changed the protocol: the benchmark now trains AND rolls out toroidally (`periodic_warp: true` recorded per artifact, `--clamped-boundary` ablation flag kept). Fair (exclusive-CPU) rerungs under the periodic protocol, new filenames so the committed clamped-warp artifacts survive: flow-supervised 180s lands at **37.21/34.95/32.78** vs persistence 39.54/37.21/35.24 (.AreEqual — WORSE than its clamped-warp 39.12/36.96/34.57: the periodic protocol degrades the supervised head's rollout, and the shifted budget/steps axis is recorded in the artifact); self-supervised 4× periodic lands at **39.50/37.23/35.24** (parity with persistence everywhere — H=4 −0.05, H=8 +0.01, H=14 +0.00). Ladder truth: under PERIODIC semantics the best aggregate equals persistence, no longer beats it; the earlier above-persistence verdict was CLAMPED-warp-specific and is labeled as such. (iii) **Distill parity on the trained weights: the 4×-periodic head was exported (`learned_flow_dcstt_4x_periodic_head.pt`, 98 KB) and drives the kernel — `tests/test_dyn_flow_head_distill_parity.py` passes at max|Δv| ≤ 1e-12 on real train-clip contexts (holdout ids never consumed), weights-pack length asserted against the spec, missing-artifact path SKIPS.** The Rust serving path now provably executes the same weights that earned the rung's numbers.

### Known limitations (recorded, not swept)
- Single-run artifacts; the learned head needs more CPU budget/architecture before it earns a serving claim. Integer-shift baseline collapsed to (0,0) on these clips (velocity ≈ uniform translation), so persistence is the real comparison. Stage-3 adds: clip-0005 regression under flow supervision (open); shared-forward double cost in `flow_supervision_loss` (open); per-step-cost variance between ladder rungs — wall-clock matched, step-mismatched, disclosed in the ladder table.

### Added
- **`benchmarks/run_jetflow_vs_diffusion_matched.py` + `benchmarks/jetflow_vs_diffusion_matched_results.json` + `docs/jetflow_vs_diffusion_matched_benchmark.md`:** replaces the earlier equal-*step-count* protocol (which under-budgeted the slower JetFlow stack) with equal **wall-clock training budget** enforced by an in-loop `deadline_s` stop added to all three trainers (`distill.train_flow_model`, `distill.distill_solution`, `diffusion_baseline.train_flow_model_baseline`): probe-estimated step counts could drift ±33% from real-run per-step cost (first run: jet 81.7 s / baseline 117.8 s for a 90 s budget — baseline overshot, jet under-consumed), so the deadline, not the probe, is the binding budget axis. v2 artifact: jet 3172 steps vs baseline 3478 steps, both exactly 90.0 s (28.38 vs 25.88 ms/step — a near tie, confirming the earlier per-step-cost attribution was wrong and the quality gap is architectural). Model-forward counts are *measured* per clip (not hand-estimated); identical CFG=3.0, trainer-consistent sampler directions, and identical burnable hold-out ids across architectures.
- **`packages/jetflow-video/jetflow_video/diffusion_baseline.py` (new):** the fair textbook baseline — small UNet-style ResBlock denoiser (152k params vs JetFlow's 133k), standard cosine-schedule DDPM v-objective, CFG-capable; trained through the same budget controller, same split, same instruments.
- **Honest verdict (single-run seed 0, artifact-backed):** at matched budget and ~0.41–0.45 s/clip the diffusion baseline wins every measured quality instrument (PSNR 12.59 vs 7.43, SSIM 0.663 vs 0.250, FVD-proxy 0.272 vs 1.053, temporal 0.133 vs 0.642); quality plateaus at K≈32 (K=48 spends 1.75× longer for nothing); the distilled 2-step student is the only sub-50 ms option at 14.2 ms/clip (29× faster, PSNR 9.70) — JetFlow's defensible claim is serving-side latency, not matched-compute quality. Known limitation (recorded, not swept): single-run variance unquantified; no repeats yet.

### Tests
- Full AGENTS.md regression battery plus the video/jetflow suite: 92 passed (one `test_end_to_end_text_to_video_jet2_flow_matching` failure in the first pass did not reproduce in two 13-file suite reruns and passes in isolation — one-off transient, recorded honestly rather than hidden).

## [Unreleased] — mypy strict debt cleared; blind-freeze re-mint v1.2 (2026-10-17)

### Fixed (typing)
- **Both mypy CI gates exit 0:** `mypy apedp_dyn/ --ignore-missing-imports` and `mypy apedp/ apedp_dyn/ --ignore-missing-imports` report no issues (was 122 errors in 38 files). The `apedp/` core gate was already green; this release brings `apedp_dyn/` to parity.
- Honest fixes by category: `isinstance(dict)` guards at every JSON boundary (`canonical_spec`, `frozen_spec`, `ode/dispatch`, `riemann`/`goldbach` rust-binary parsers, lean-provenance loader); value-preserving `np.asarray` wraps at ndarray-declared return boundaries (video stack, execution-cost models, `high_dim_pde` Rust-engine wrappers); union request types widened ONLY where callers legitimately pass ndarrays (`staircase_prior`, `fit_step_family`, `step_family_basis`, `sample_at_timestamps`, shape-inference request forms); `_dp_segment` return annotation corrected to its actual `(list[list[float]], ndarray, dict)` contract; `ProposerBase.fit_and_propose` abstract contract made call-shape-agnostic (`*args/**kwargs`) so heterogeneous concrete proposers conform to AGENTS.md §1.1; scorer factories in `rime_repair_engine` (uniform `Callable[[pd.DataFrame], ndarray]` signatures, bound-at-emit-time closures); Tesla-style conditional nested scorers renamed in `coordinate_transmuter`; missing `Any`/`Callable` imports; integer rank selection via numpy scalar `.sum()` in `tensor_train_video`.
- `CandidateProposal.parameters` widened to `dict[str, Any]` — heterogeneous by contract: numeric fit parameters plus predictive-status strings such as `identifiability_status` (readers cast explicitly); no runtime instance shape changed.
- **No thresholds touched:** per AGENTS.md §1.3 no new threshold literals anywhere; the new `cross_asset_gate.evaluate_cross_asset_gate_matrix` diagnostic adapter only computes within-class/pooled/two-tier Sharpe inputs and delegates all verdicts to the canonical `evaluate_cross_asset_gate` (sandbox probe callers called an API shape that always threw and silently fell back — they now reach the real gate and read `report.passed`).

### Fixed (frozen-record protocol)
- `apedp_dyn/sandbox/grammar_space.py` was type-hardened (explicit `from typing import Any` import; value-preserving `np.asarray` wrap at an ndarray-declared return boundary) — no referee logic, thresholds, or canonicalization changes. Per the runner's own mint-first protocol, **`APEDPX_FROZEN_v1_2.json` supersedes v1_1** (drift: `6f4da69fb15b…` → `aba0f42abb0f…`, sole guard-file change), `blind_benchmark.FROZEN_RECORD` and the `frozen_spec.py` pinned hash were re-pointed, and all 15 referee guard files verify against the new record.

## [Unreleased] — Rust↔Python Bridge Hardening (2026-10-17, `production`)

### Performance
- **Zero-copy NumPy columns in the native kernel:** `extract_columns` deep-copied every dataset column into owned `Vec<f64>` on every FFI crossing, re-copying the same data on each per-candidate call (`evaluate_batch`, `score_candidates_parallel`, `rime_evaluate_json`, …). The new `DataCol` erased-column trait (`rust/apedp-kernel/src/data.rs`) makes owned `Vec<f64>` and borrowed `ndarray::ArrayView1<f64>` interchangable, and `PythonColumns` (a guard-backed view holder) materializes zero-copy views over the Python-side buffers across `allow_threads`. The parser, numeric, genetic, and RIME (graph/eval/search/additive/residual) surfaces now take `&[V: DataCol]`; the JSON-fed internal paths keep their owned `Vec<Vec<f64>>` unchanged.
- **Compile caching at the bridge:** `RustKernelBridge` re-parsed identical expression strings at every FFI crossing. Added `_kernel_compile` (LRU, 4096 entries, keyed on `(expr_str, variable_names)`, owner-guarded so a swapped kernel clears the cache) and a `lambdify` cache for the Python fallback (keyed on expression text).
- **Release profile:** workspace `Cargo.toml` sets `codegen-units = 1` alongside `lto = true`.

### Added
- **Zero-copy 1-D extraction also:** the guarded-views pattern extends past column lists — single-array arguments (targets, `y_true`/`y_pred`) now extract through a `GuardedVector` (pin-once `PyReadonlyArray1` guard; contiguous buffers hand out borrowed `&[f64]` slices via `PyReadonlyArray::as_slice`, strided buffers take one owned fallback copy), so `compute_metrics`, `solve_linear_coefficients`, `evolve_native`, and every RIME entry point stop re-copying their target and series arguments per call.
- **Kernel-path counters + `kernel_path` field:** every `RustKernelBridge` instance exposes `tallies` (rust/fallback call counts) and `last_path`, and every returned result dict (metrics, scoring rows, probes, conditioning, linear solves, precision boundary, evolve rows) carries a `kernel_path` tag. A benchmark run can now *prove* the Rust fast path executed instead of being unable to distinguish it from the silently-correct Python fallback.
- **Loud missing-kernel warning:** constructing a bridge without the native `apedp_kernel` extension logs exactly one WARNING naming the `maturin develop --release` build command (previously the fallback was invisible and "easy to miss", in the bridge's own docstring words).
- **CI wheel artifacts:** `build-kernel-wheel` job builds the manylinux wheel via `PyO3/maturin-action` and uploads it as a workflow artifact on every push/PR, so the kernel is `pip install`able without running maturin locally; a tag-gated `publish-kernel-wheel` job publishes to PyPI only when a `PYPI_API_TOKEN` secret is configured.

### Measured (before/after, min-of-3 interleaved fresh-process runs, 600×4 float64 dataset)
- **Reference engine (Python fallback) repeated eval: 2678.9 → 34.5 µs/call (77×).** The lambdify cache is where most wall-clock in fallback-heavy workloads went.
- **Batch scoring round (15 candidates): 1087.6 → 888.1 µs (−18%); per candidate 72.5 → 59.2 µs.** Compile-free cache hits plus view-based columns.
- **Repeated candidate-evaluation call: 110.8 → 95.6 µs (−14%).** Per-call cost at this dataset size is dominated by Rayon pool dispatch (~85 µs), so the copy-elimination scaling grows with dataset size (the elimination only has to cover the ~19 KB per-call re-copy here).
- **`compute_metrics`: 11.1 → 7.6 µs (−32%).**
- Unique-expression direct eval (no cache effect): 24.9 → 20.7 µs (−17%). Kernel-level `score_candidates_parallel`: 817.7 → 785.2 µs — no kernel-path regression; the only regression risk found in profiling (machine-load drift) was excluded by interleaved min-of-N.
- Artifacts: `benchmarks/run_bridge_fastpath_benchmark.py` (identical driver for both stacks) and `benchmarks/bridge_fastpath_benchmark_results.json` (raw runs + minima). The "before" stack is the git HEAD kernel (owned `Vec<Vec<f64>>` copies, bare `kernel.compile` at every crossing) built from a clean wheel into an isolated venv.

### Tests
- `tests/test_rust_bridge_observability.py` (new, 8 tests): tallies prove the fast path ran, result dicts carry `kernel_path`, failing kernels tally fallback (not vanishing), compile cache serves repeated expressions once, cache is keyed on the variable table (wrong-variable-layout reuse is the dangerous failure), the fallback lambdify cache is consulted, and the missing-kernel warning fires exactly once at WARNING level with the build command.
- Full verification: 179 Rust tests (`cargo test`), 45 bridge tests, 26/26 differential conformance against the rebuilt release kernel, 58/58 AGENTS.md regression battery.

---

## [Unreleased] — JetFlow-Video Honest-Fast-Path Remediation (2026-10-17, `experimental`)

### Remediated (fraud vectors)
- **Fake NumPy trainer removed (`apedp_dyn/video/jet_flow_model.py`):** the NumPy `train_step` was not a trainer — it mutated one bias vector by a single scalar mean-gradient while every other conditioner weight stayed frozen-random. The method is now diagnostics-only (`DeprecationWarning`, NO parameter mutation); real training routes to `packages/jetflow-video JetFlowTrainer` (full-parameter AdamW; grad-flow verified). Pinned by `tests/test_dyn_jet_flow_sota_path.py::test_jet_flow_video_model_train_step` (behavior change justified in the test docstring — the old pin certified fake behavior).
- **Salted-hash prompt embedding replaced (`apedp_dyn/video/semantic_bridge.py`):** `embed_text_prompt` seeded random Gaussians from the Python builtin `hash()`, which is salted per process (PYTHONHASHSEED) — embeddings were non-reproducible across runs and carried zero structure. Replaced with `feature_hash_word_char_trigram_v1` (MD5-indexed hashing trick over word + char-trigram tokens + sentence-context term). Byte-identical across processes (subprocess-pinned); lexical similarity preserved (0.744 shared-vocabulary vs 0.229 unrelated). Honestly labeled as lexical features, not CLIP-scale semantics.
- **Found package bug — sampler direction:** `model.generate_video` integrates t: 0 -> 1 from noise, but `trainer.compute_loss` parameterizes t = 0 as data / t = 1 as noise — the shipped sampler ran the ODE backwards relative to training. The samplers in `first_frame.py` and `distill.py` implement the trainer-consistent direction (integrate t: 1 -> 0); the package inconsistency is documented, unresolved in `generate_video` itself.
- **Benchmark-doc honesty:** `docs/jet_flow_video_benchmark.md` executive-summary numbers (8.7x–24.4x, 13.9x, 0.02 ms/frame) were superseded by its own artifact (75.85x–186.18x, 8.03x, 0.782 ms/frame); treat the JSON artifact as the citable source. The "2-step beats 50-step diffusion" benchmark is a strawman (integrators on a told analytic field) — replaced by genuine teacher->student distillation curves below.

### Added
- **`apedp_dyn/video/eval_gate.py` + `tests/test_dyn_video_eval_gate.py` (5/5):** held-out certification instruments — PSNR/SSIM (video-aware), temporal-consistency (smooth advection beats frame-shuffled), FVD PROXY (sliced Wasserstein-2 over a frozen deterministic featurizer; explicitly NOT I3D-FVD), burnable held-out split (disjoint/deterministic/exclusive), and a certify API with caller-supplied thresholds (no new threshold literals).
- **`apedp_dyn/video/fluid_domain.py` + `tests/test_dyn_fluid_domain.py` (4/4):** domain-scoped dataset — 2D incompressible NS tracer clips (velocity from streamfunction, grid-divergence-free |div| < 1e-10 — the Noether projector finally applies to a physically real object here), semi-Lagrangian dye advection, scope manifest with exclusive holdout ids.
- **`packages/jetflow-video/jetflow_video/first_frame.py` + `tests/test_dyn_jetflow_anchored_and_distill.py` (5/5):** real-first-frame conditioning — anchored corruption (frame 0 stays clean), anchor-excluded loss, and generation that re-imposes the anchor bitwise at every step in the trainer-consistent sampling direction.
- **`packages/jetflow-video/jetflow_video/perceptual_decoder.py` + `tests/test_dyn_jetflow_perceptual_decoder.py` (3/3):** zero-init residual decoder on the Haar base (training STARTS at exact Haar parity — monotone upgrade path, no regression cliff), patch-LSGAN discriminator + gradient loss trained on fluid frames.
- **`packages/jetflow-video/jetflow_video/distill.py` + `benchmarks/run_jetflow_distill_benchmark.py` + `benchmarks/jetflow_distill_curves.json`:** authentic teacher->student solution distillation (trained teacher on the fluid domain, run split-disciplined). Landmark: student 2-step endpoint RMSE vs 16-step teacher 0.808 -> 0.507 measured on the artifact; the honest replacement for the old integrator benchmark.
- **`benchmarks/run_jetflow_cpu_export_benchmark.py` + `benchmarks/jetflow_cpu_export_benchmark_results.json` + `tests/test_dyn_jetflow_cpu_export.py` (3/3):** ONNX export (static traced graph per clip config — traced attention cannot carry a dynamic temporal axis; documented) plus `onnxruntime` dynamic int8 quantization: 173.5 KB -> 104.9 KB with parity max|Δ| = 0.0; measured latencies (eager 8.68 ms / ORT fp32 5.32 ms / ORT int8 9.97 ms) — int8 is a SIZE win at this scale, ORT fp32 the speed win; the artifact reports both rather than marketing one.

### Known limitations (recorded, not swept)
- Synthetic fluid-domain evidence only; no real-data photorealism claim derives from this round.
- FVD proxy is not I3D-FVD; ranked-separation instrument only.
- ONNX graphs are static per clip length (regenerate per export).
- `model.generate_video` sampler-direction bug is documented but NOT fixed in place (deployed samplers route through the corrected `first_frame.py`/`distill.py` paths).

## [Unreleased] — RIME-AD Native Rust Engines & Unified Quantum Benchmark (`experimental`)

### Added
- **`rust/apedp-kernel/src/rime/total_variation.rs` (BV-ROF Engine):** Fast Condat exact $O(N)$ 1D Total Variation filter (`tv1d_condat`) and staircase jump analysis (`tv_denoise_and_analyze`). Recovers quantized plateaus and quantum step heights without polynomial Runge oscillation, robust through 5% observation noise.
- **`rust/apedp-kernel/src/rime/implicit.rs` (Sparse Implicit SVD Null Space):** Symmetric algebraic relation discovery finding sparse null-space vectors of generalized feature tensors $\Theta(x, y) c \approx 0$ via Jacobi eigendecomposition. Discovers Detuned Rabi Lorentzian resonance $(25 + \Delta^2) P_e - 25 = 0$ in $0.31$ ms.
- **`rust/apedp-kernel/src/rime/diff_pade.rs` (Differential Padé & Jet Algebra):** Discovers governing polynomial differential equations $P(x) y' + Q(x) y = 0$ on the Jet bundle without pre-planted exponential primitives. Discovers Schwinger pair production ($x^2 y' - (2x+1)y = 0$, $R^2 = 0.9999$), Landau-Zener ($v^2 y' - 0.75 y = 0$), and BCS energy gap ($g^2 y' - y = 0$), with forward RK4 trajectory integration.
- **`rust/apedp-kernel/src/rime/enumerate.rs` (Shared-Computation Canonical DAG):** Systematic complexity-ordered small-expression enumeration with hash-consed memoization (PSE-style shared computation). Recovers Resonant Rabi harmonic dynamics ($1 - \cos^2(t)$, $R^2 = 1.000000$) and Brus confinement ($R^2 = 0.998414$).
- **`benchmarks/run_quantum_all_equations_benchmark.py` & `docs/quantum_all_equations_benchmark_results.md`:** Unified machine-measured benchmark across all 9 physical and quantum equations tested before.
- **`tests/test_dyn_rust_all_equations_benchmark.py` (5/5 green):** Continuous integration assertions validating all Rust engines against all equations.
- **`benchmarks/run_population_extrapolation_rime_ad_vs_ga.py` & `docs/population_extrapolation_rime_ad_vs_ga.md`:** Multi-population extrapolation benchmark comparing RIME-AD against GA/GP across Pop A-train, Pop A-test, Pop B (fresh seed), Pop C (parameter shift $c_1 = 2.0 \to 2.5, c_2 = 3.0 \to 2.0$), Pop D (heavy-tailed Laplace OOD), and 7-channel distractor noise challenge.
- **`tests/test_dyn_population_extrapolation_benchmark.py` (3/3 green):** Automated regression verifying that RIME-AD matches GA-Large across all populations, exactly recovers true mechanism $+1 \cdot (2.9956 \cdot x_3[t-2] + 1.9966 \cdot (x_1[t-1] x_2[t-1])) + 9.9988$, exhibits zero false discoveries on 7 distractor noise channels, and operates ~246x faster than GA-Large.
- **`rust/apedp-kernel/src/rime/koopman.rs` (Delay-Embedded Spectral Discovery):** Affine order-2 linear recurrence fit by least squares with companion-matrix eigenvalues (complex-pair frequency/decay readout) and forward rollout helper. Recovers harmonic $\omega = 2.0000$ (holdout rollout $R^2 = 1.000000$) and damped $(\omega = 3, \sigma = -0.5)$ spectra. Explicit v1 scope: scalar uniform trajectories, two-mode dynamics.
- **`rust/apedp-kernel/src/rime/lie_symmetry.rs` (Lie Point-Symmetry Discovery):** Exact bivariate polynomial algebra for the linearised symmetry condition $\eta_x + (\eta_y - \xi_x) f - \xi_y f^2 - \xi f_x - \eta f_y = 0$, with Gram/Jacobi nullspace basis. Verifies scaling/x-translation symmetries of $y' = y$, the exact 9-dimensional translation algebra of $y' = 0$, and $y$-translation for $y' = x$. Explicit v1 scope: first-order polynomial ODEs.
- **`tests/test_dyn_rust_koopman_vs_ga.py` (3/3 green) & `tests/test_dyn_rust_lie_symmetry.py` (3/3 green):** Matched-split Koopman vs native-GA comparison (identical train/holdout; GA finds the exact $-2\sin^2(t)+1$ identity at holdout $R^2 = 1.000000$ in 14.71 ms / 1560 evals vs Koopman 0.31 ms) plus Lie symmetry verification. gplearn/PySR unavailable in this environment, so the GP side is the native genetic-programming kernel path.
- **`benchmarks/run_five_gate_validation.py`, `benchmarks/five_gate_validation_results.json`, `docs/five_gate_validation_results.md`, `tests/test_dyn_five_gate_validation.py` (green):** Executed five-gate programme, all gates passed — exact recovery on 8/9 equations with the Planck limitation held; honest noise degradation (Schwinger 2% noise refuses exact claim); 4/4 adversarial abstentions; matched-compute ablation RIME vs native GA (Rabi tie at $R^2 \approx 1.0$; Brus GA overfits to holdout $R^2 = -2.93$ while RIME transfers); sealed confirmation under fresh seed 12345.
- **Planck honesty fix (`tests/test_dyn_planck_limitation.py`, green):** Proved no exact first-order polynomial ODE exists for Planck radiance ($u'/u = 3/v - e^v/(e^v-1) \notin \mathbb{Q}(v)$); the engine correctly returns `is_exact_recovery = False`. Benchmark verdict corrected from `PASSED` to `RECORDED LIMITATION`, with `is_exact` now stored in the JSON artifact.
- **Exponential generation (`enumerate.rs` `include_exp`, `search_rational_pairs`, native-GA `allow_exp`):** `exp()` nodes are now generatable in shared-DAG enumeration and native GA trees (previously evaluable but unreachable). The Padé-style rational-pairing stage (transcendental-denominator shortlist, spike guard, top-k budget) recovers the Planck textbook form $v^3/(e^v-1)$ at $R^2 = 1.000000$ with fresh-domain transfer at $R^2 = 1.000000$. GA with `exp` wired still misses Planck at small budget (grammar access is not discovery). Pinned by `tests/test_dyn_rust_exp_discovery.py`.
- **`RimeAdProposer` univariate fan-out (`max_univariate_features`):** multi-feature frames now run Representations 1-4 per feature instead of OLS-only; on 90-asset dailies all engines engage but every emission fails forward holdout (documented negative result).
- **Extended misc ops (`Neg`/`Log`/`Sqrt`/`Abs`) in enumeration and native GA:** domain-guarded evaluator arms (NaN out-of-domain, skipped by callers) gated behind `include_extended` (enumerator) and `allow_misc` (GA, threaded from `SearchConfig` through the bridge). Enumeration recovers all four forms at R² = 1.0 with the gate provably off by default; GA discovers `abs(x)` and `log(u)` exactly at small budget. General `Pow(x, y)`, `tan`, hyperbolic and inverse-trig deferred (numerical safety + pool explosion). Pinned by `tests/test_dyn_rust_extended_ops.py` (4/4).
- **Screening-gated univariate fan-out (`screen_top_k`, quadratic screen):** `RimeAdProposer` ranks features by [1, x, x²] OLS dev R² (catches even-symmetric dependence invisible to linear correlation) and fans Representations 1-4 over the measured top features instead of a blind 12-cap. Pinned by `tests/test_dyn_rime_ad_screening.py` (3/3).
- **Domain grammar expansion (pi/e, min/max, sign, cbrt, erf, besselj):** parser gains two-argument `min(a,b)`/`max(a,b)` as real `Node` variants (the parser previously rejected all multi-argument forms), `Func::Erf` (Abramowitz-Stegun 7.1.26, |err| ≤ 1.5e-7), `Func::BesselJ0` (power series, ~1e-12 for |x| ≤ 8), `Func::Cbrt`, and exact `pi`/`e` leaf constants. Enumerator gains the same ops behind `include_extended` (pi enters the depth-0 pool so `sin(pi*x)` is reachable; min/max join the commutative binary dedup). GA gains all of it behind `allow_misc` (9-op misc group + 8% pi/e leaf rate). Two real bugs caught by the new pins: `signum(0.0) == 1.0` in Rust made `erf_approx(0)` return ~1e-9 (now special-cased to exactly 0), and `assert_eq!` on NaN always fails (switched to `is_nan()`). Pinned by `tests/test_dyn_rust_domain_ops.py` (6/6) plus Rust accuracy tests against reference values (erf(1)=0.8427007929, J0 first zero 2.4048255577). Deferred: general `Pow(x,y)` (unsafe on negative bases), `tan`/hyperbolics/inverse-trig (pool explosion, low physics priority).
- **Advanced ops (`tan`/`sinh`/`cosh`/`tanh`/`asin`/`acos`/`atan` + general `Pow(x,y)`):** gated behind `include_advanced` (enumerator) and `allow_advanced` (GA). Enumerator recovers all 7 unary forms at R² = 1.0 with the gate provably off by default. GA discovers `atan(x)` and `tanh(x)` exactly and `x^2.5` via `x·x·√x`; general Pow with random exponent fires at 30% rate under the advanced flag. GA formatter bug fixed: advanced Func names previously rendered as `f(x)` (the `_ => "f"` fallback — invisible before because GA never generated them). Pinned by 3 added tests in `tests/test_dyn_rust_domain_ops.py` (9/9 total).
- **Dual-harness adjudication (2026-10-15, measured):** the engine-level and referee-level verdicts coexist without contradiction -- they measure different properties. Schwinger: engine exact-recovery TRUE (traj R² = 0.999903, exact formula) AND referee-whiteness refuse TRUE (residual RMSE 0.019 on scale 6.3 but smoothly drifted, ρ1 = 0.93). BCS: same layered outcome (R² = 0.999999, RMSE 2.8e-4, ρ1 = 0.95). Resolution: adopt LAYERED verdicts (structure gate + whiteness gate) everywhere, per the six-metric design; neither harness is wrong, each is incomplete alone.
- **Planck scope correction (adjudicated 2026-10-15):** the non-existence proof covers first-order LINEAR polynomial ODEs only. A quadratic Riccati relation v^3 u' - (3v^2 - v^3) u + u^2 = 0 holds to ~2e-7 (verified); no current engine discovers second-degree Jet relations -- recorded follow-up. Test docstring corrected; behavioral pin (linear abstention) unchanged and green.
- **Riccati refinement: SR3-lite sparse refinement + rational snapping (`diff_pade.rs`, 2026-10-16):** the differential engine now iteratively re-solves the sparse support (`refine_sparse_system`, hold-fixed STLSQ) and rationalizes coefficients (`snap_rational`, denominators <= 16, 1% tolerance), each behind revert guards on BOTH trajectory R2 and jet RMSE fidelity. New `snapped_terms` result field. The Planck Riccati relation is now recovered EXACTLY: `x^3 y' + (x^3 - 3x^2) y + y^2 = 0` (traj R2 = 0.999999999759, jet RMSE = 8.597e-07, snapped; previously a cubic linear fit with spurious terms at R2 = 0.999965 / RMSE = 0.004906). This resolves the 2026-10-15 "recorded follow-up". Pinned by `test_planck_riccati_ode_recovery` (now asserts exact) + `tests/test_dyn_frontier_9_of_10.py`.
- **Exact-recovery gate fix (integrity, 2026-10-16):** the old `is_exact = traj > 0.99 && jet < 0.05` gate mislabeled fitted approximations as exact once refinement improved them (demonstrated: cubic-linear Planck fit at jet ~1e-4 flagged exact, contradicting the Hermite non-existence proof). New gate: `traj > 0.99 && jet < 0.05 && (snapped_terms || jet < 1e-9)` — exactness now requires structural evidence (rationalized coefficients) or machine precision. Regression pins added: Rust `test_planck_cubic_linear_fit_must_not_claim_exact` + `tests/test_dyn_planck_limitation.py::test_planck_cubic_linear_fit_must_not_claim_exact`; the fresh-domain linear-transfer threshold was updated 0.9 -> 0.99 (measured 0.938) with the abstention pin retained.
- **Frontier credibility fix (`docs/frontier_9_of_10_architectural_resolution.md`, 2026-10-16):** the referenced artifact did not exist on disk (AGENTS.md 1.4 violation); created with machine-regenerated numbers and explicit corrections (Burgers diffusion error 11.2%; heat holdout is a re-fit, not transfer; TV plateau error 1.27%). `DOCUMENTATION.md` PDE engine entry corrected `production` -> `experimental`. New `tests/test_dyn_frontier_claims_integrity.py` (6/6) regenerates every headline number and asserts the artifact states them.
- **Native GA constant polish (`genetic.rs` `polish_candidate_constants`, 2026-10-16):** post-evolution coordinate-descent stage over inner `Const` leaves via golden-section line search (`span = max(|v|, 0.5)` shrinking 0.4/sweep, 8 sweeps in-loop, affine readout refit at every probe; strict-improvement replacements only, so objective-flat constants -- e.g. a lone global multiplier absorbed by the readout -- are left untouched) plus a Powell-style conjugate-direction line search along each sweep's net displacement, which escapes coupled valleys where every single-axis move is locally non-improving (sin(f*x+p) frequency/phase: 9.1e-4 -> 1.3e-6). Config: `polish_constants` (default true) + `polish_top_k` (default 3, top candidates per island), threaded through the `evolve_native` binding and `RustKernelBridge.evolve_native`. Rust mechanism pins: sin(c*x) refines c = 1.8 -> 2.000 with >100x MSE collapse; joint (frequency, phase) converges; constant-free trees cost exactly one eval and are returned structurally identical; same-seed A/B never worsens the pooled best while counting polish evals. Python pins: `tests/test_dyn_native_ga_constant_polish.py` (3/3; never-worsens, top_k=0 == disabled, bridge flag forwarding, x^2 recovery unaffected).
- **GA trig frequency/phase template seeding (`genetic.rs` `trig_template`, 2026-10-16):** top-level random trees now seed the `sin|cos(f*x + p)` template (rate 0.18, gated on `allow_trans` + variables + depth >= 3) with `f ~ U(0.3, 2.8)`, `p ~ U(-pi, pi)` (35% phase-free), directly addressing the frozen N04 forensics (uniform depth-4 trees could not express a phase-shifted oscillation; best fit R2 = 0.04). Measured at default budget (50k-eval class): `sin(2x)`, `2*cos(1.3x)`, `sin(1.5x)` recover EXACTLY (MSE = 0); `3*sin(1.7x+0.4)` reaches the correct family with frequency/phase resolved to ~5e-3 (MSE 1.3e-6) via the Powell-coupled polish; `2.5*exp(-0.5x)` recovers exactly via `2.5*sqrt(exp(-x))` (MSE 5e-32). Pinned by `tests/test_dyn_native_ga_trig_templates.py` (3/3) + Rust `genetic.rs::trig_seed_tests` (2/2; seeding rate band, flag gating). `tests/test_dyn_rust_domain_ops.py::test_native_ga_discovers_advanced_ops` re-pinned numerically (general-Pow path finds exact `x**2.5`, train MSE ~1e-17) instead of the old sqrt-spelling assert; `atan(x)` recovery unchanged.
- **Frozen-coefficient PDE transfer evaluator (`apedp_dyn/epistemic/pde_transfer.py`, 2026-10-16):** the dev/holdout protocol in `pde_discovery.discover_pde` re-runs STRidge on the holdout partition, so its `holdout_r2` is a re-fit metric (documented limitation #1). New `evaluate_frozen_transfer(u_hold, dt, dx, active_terms, coefficients)` scores dev-discovered coefficients with NO refit, using the identical 2nd-order time / 4th-order space central stencils, interior mask, and R2 definition (vs mean u_t) as `rime/pde.rs`; unknown term names raise (never silently dropped). Measured on the Rust Burgers fixture (u_t = -u*u_x + 0.1*u_xx): dev coefficients (-0.9962, 0.1112), frozen holdout R2 = 0.999772 vs re-fit 0.999909 -- the honest transfer number and the measured re-fit inflation are now both on record. Pinned by `tests/test_dyn_pde_transfer.py` (4/4; analytic heat exactness, wrong-coefficient penalty, Burgers partition survival, unknown-term rejection). Wiring into the `discover_pde` adapter itself is deferred to coordinate with the concurrent pde.rs change set.
- **RIME-AD native external-corpus adapter (`benchmarks/external_60_adapters.py::RimeADNativeAdapter`, 2026-10-16):** contract-compliant adapter running the native engine (`evolve_native`) on the frozen 60-problem corpus public grids. The frozen round compared the full `AutonomousModelSynthesizer` stack, so hidden-truth scoring of this engine is a NEW round under protocol section 6.3 (own preregistration/seed/checksums) and is explicitly not performed here; the adapter never reads `hidden/`, abstains on eval failure without shipping numbers, and is pinned by `tests/test_external_60_native_adapter.py` (2/2, including a run against a corpus copy with no hidden directory). Frozen `tournament_results_external_60/` untouched.
- **Reproducibility targets (`Makefile`, 2026-10-16):** `make kernel` (maturin develop), `make wheel` (distributable wheel to `dist/`), `make reproduce` (builds the kernel, then regenerates and verifies every headline number a reader would quote: frontier claims integrity, Planck/Riccati claims, canonical ledger, the frozen external round, trig recovery, PDE frozen transfer -- 30 tests, green). Root `Cargo.lock` was already tracked and pins the kernel's dependency tree.
- **OpenAI-math sandbox (`benchmarks/openai_math_sandbox/`, 2026-10-16, `audit-only`):** mirror of the OpenAI 2026-10-06 math-release pipeline (attempt -> publish claim -> independent verification -> retraction accounting; source release: 722 manuscripts, 3 retractions, 372 result families, 719 listed) applied to APEDP-X. Registry `problems.json` holds 6 supported/conditional machine-checkable tasks and 8 explicit `unsupported` gap entries (Lean formalization, quasi-RH, BSD, Hodge, Kakeya R^4, asymptotic algorithmics, Navier-Stokes, 719-manuscript scale), each with a reason. Measured 2026-10-16: attempted 6, verified 6, failed 0, unavailable 0, unsupported 8 -- Goldbach [4, 2e6] 999,999 evens 0 violations (ceiling C2, universal claim abstains), Riemann 13 zeros from scratch calibrated at 1.15e-9 (tally APPROXIMATE_TALLY_MATCH, S(T) uncertified), Planck Riccati jet 8.60e-07, harmonic jet 6.11e-14, trig MSE 0.0 both targets, Burgers frozen holdout R2 0.999772. Runner enforces in code that no task is ever `verified` without a checker certificate (missed threshold = `failed` = retraction count); README carries the gap analysis. Pinned by `tests/test_openai_math_sandbox.py` (6/6). Companion hardening: `tests/test_dyn_pde_transfer.py` now skips with a reason when the pending `rime_pde_discover_json` binding is absent, so a clean checkout of the commit that shipped `pde_transfer.py` stays green instead of erroring.
- **Full openai/math catalog ingestion (`benchmarks/openai_math_sandbox/ingest_openai_catalog.py` + `catalog/`, 2026-10-16, `audit-only`):** ingests ALL 372 result families / 719 manuscripts from a SHA-256-pinned snapshot of `openai/math` (CONTENTS.md, history.md, overview.pdf, formalization.yaml; `catalog/source_snapshot/`), joins the Lean catalogue, and records the accounting: 3 named withdrawals (722 -> 719; Weil-classes sign error cascading to two dependents), 14 revised + 13 citation-updated, historical family numbering 001-377 with 5 retired numbers, and 5 explicit discrepancies kept on record -- including that the repo-stated 300/719 (~42%) formalization count does not reconcile with formalization.yaml's 173 source papers / 200 main-result declarations (both numbers reported, delta unresolved) and that the yaml is `unchecked`/`agent`-produced (not an independent check; APEDP-X does not run Lean). APEDP-X verdict at catalog scale: 372/372 families `uncheckable_proof_level` (no proof engine; probe tags for 40 finite-numerics and 35 explicit-construction families are illustration only). Pinned by `tests/test_openai_math_catalog.py` (7/7) against the committed snapshot hashes.

- **`apedp_dyn/epistemic/prior_guided_search.py` — the loop-closer.** Consumes the shape
  prior and dataset profile on DEV rows only and selects a search path through a FIXED
  table (stepped -> step basis + extrapolatable rule form; singular/power -> transmutation
  fitter; else abstain). Emits `CandidateProposal` objects; never promotes, never sees
  holdout. No operator-name literals usable for per-equation selection (machine-guarded).
  Owns a deterministic two-stage floor-rule fit (global log-width grid + max-margin
  tie-break + local refine): max-margin phasing was REQUIRED, measured, because several
  (w, off) classify dev rows identically (SSE 1.5e-29 tie) while holdout separates them
  (0.974 vs 1.0).
- **`apedp_dyn/epistemic/report_fidelity.py` — verdicts derive from rows.** `verdict_for_row`
  with explicit args (tau_extrapolation, rho_max, eps_exact_rmse, delta_exact_r2):
  exact-recovery bypass checked FIRST (rho1 of numerical noise must not refuse R2 == 1),
  then gate, then stated refusal reasons. The Landauer harness template now renders this
  function's output verbatim; hardcoded `**PASSED** (Exact Recovery)` / `**REFUSED**`
  labels were removed from `benchmarks/run_landauer_rime_vs_ga_benchmark.py` (4 lines +
  import; NaN rho1 now renders `n/a (exact-zero residuals)` instead of `0.0000`).
- **`tests/test_dyn_prior_guided_unsolicited.py` (7/7 green).** Unsolicited Landauer recovery
  on sealed splits (dev [0.1,4.9] -> holdout [5.0,9.8], R2 == 1.0 both, zero human primitive
  choice); table truth across staircase/singular/power/line (line abstains); noise rung 1
  (2%: holdout >= 0.999); rung 2 labeled semi-synthetic (BTC 5m + injected W=125/OFF=40
  staircase at local SNR=5: width 125.02/125, quantum within 0.4%, offset modulo width,
  holdout 0.8910 vs oracle 0.8917); clean-stream abstention control; low-SNR boundary
  (SNR<=4 abstains) pinned, not hidden; single-threshold edge states its no-holdout limit.
- **`tests/test_dyn_landauer_report_fidelity.py` (4/4 green).** Verdict-rule units, committed
  md == committed JSON cell-for-cell on the main table, no-hardcode guard on the harness,
  end-to-end rendered-cell presence.

### Measured integration gap found and fixed (kept on record)
- The transmutation fitter DECLINES stepped data under noise that the shape prior
  identifies (its level-clustering rule needs <=20 discrete rounded levels; noisy returns
  exceed that). The adapter therefore owns a uniform row-1 floor-fit path instead of
  depending on the other module's internal gating. Two components disagreeing on the same
  data is exactly what the closed loop had to resolve, and did.

### Honest scope restated
- Rung 2 is semi-synthetic by necessity: the repo has no real dataset with known truth and
  >=40 rows (Bertozzi 5, Brown 20). The injection (form, SNR, oracle) is fully labeled in
  the test. Unsolicited discovery on real measured unknowns remains unclaimed.

### Reporting integrity
- `RIME_SUITE_TOTAL` raised 200 -> 211 (whitelisting the two new files), verified by
  re-running `tests/test_dyn_reporting_integrity.py`.

## [Unreleased] — RIME shape inference steps 1-2 (data-driven family prior + step basis, `experimental`)

### Added
- **`apedp_dyn/epistemic/shape_inference.py` — data-driven form-family prior.** The point is to
  stop supplying structure by hand and let the data choose the family.
  - Exact O(k·n²) dynamic-programming segmentation; segment count from an explicit BIC
    argument (no module-level thresholds, per AGENTS.md §1.3).
  - Family verdict among `piecewise_constant` / `smooth` / `no_structure`, decided by how
    many PARAMETERS each family needs to reach a common fit target (a shared
    achievable×(1−sse_target) R²), with the flat and sloped families costed per parameter.
  - `fit_step_family` / `step_family_basis` emit a candidate step form
    `c₀ + Σ cᵢ·sign(x − tᵢ)` with data-estimated thresholds and OLS coefficients. This form
    needs NO new operator node: `Sign`, `Add`, `Const`, `Mul` already exist in the kernel.
  - `staircase_prior(x, y)` chains the two and returns `None` for the seeds whenever the prior
    does not support a step family, so a curve is never handed one.
- **`tests/test_dyn_shape_inference.py` (15/15 green, fixed seeds).** Acceptance is bidirectional:
  - Landauer staircase recovered EXACTLY: R² 1.0, coefficients (2.5, 0.5×5) matching the true
    generating form, 5 change-points estimated within half the local bin gap of truth 1..5.
  - 2% noise still identifies it (R² 0.9997); 0–10% noise all classify `piecewise_constant`.
  - Straight line → `smooth`, 0 change-points, and `staircase_prior` returns no seeds.
  - Saturating Padé curve → `smooth`, 0 change-points. iid noise → `smooth`. BTC forward
    returns (1094 rows) → `no_structure`. No fabricated steps on real market data.
  - Step count is data-estimated: 2-step staircase → 2 change-points, 5-step → 5.
  - <40 rows → `no_structure`.

### Recorded measurement failures (kept on the record, not hidden)
Four statistics were implemented and REJECTED by measurement, each in both directions:
1. **"Piecewise-constant beats a line"** — flags iid noise (+0.14 R²) and a saturating curve
   (+0.63) as stepped while granting only +0.027 on the real staircase, because a line already
   fits a ramp of steps at R² 0.973.
2. **Flat-run fraction on the p90 |Δy| scale** — collapses under noise (0.975 → 0.065 at 1%),
   because >90% flat steps put the 90th percentile inside the plateau, so the tolerance is set
   by noise instead of the jump.
3. **Within-segment drift ratio** — mislabels a COARSE segmentation of a CURVE as plateaus
   (0.70 on a noisy Padé curve, 0.98 on iid noise), because a strongly curved segment looks
   plateau-like.
4. **Separately-BIC-selected k per family** — degenerates, because a piecewise-constant model
   with enough segments drives SSE to exactly 0 and BIC's log-SSE term rewards that without
   bound (a clean straight line was assigned 32 constant segments).

Also fixed on the way: a half-bin threshold bias (DP boundaries are indices, so
emitting `xs[j]` as the edge put every sign flip one half-bin late — R² 0.990 instead of
1.000000), and an inverted sign on the parsimony gap.

### Critical negative control (why the gate, not the R², is the evidence)
`fit_step_family` applied to a STRAIGHT LINE with forced thresholds returns R² 0.979 and
`identifiable=True`, because 6 constants approximate a line over a wide range. So a high R²
from this basis is NOT admissible evidence of steps: only the independent shape prior's
verdict makes it admissible. `staircase_prior` therefore declines to emit seeds on a line,
and the test asserts the raw R² would have been misleading.

### Stated limitations
- Step 2 is still a PROPOSAL. The step basis is not yet fed to the kernel search, so
  multi-term composition and the kernel-side wiring remain unbuilt. No claim of unsolicited
  end-to-end discovery is made.
- Measured boundary: at 20% noise the staircase no longer classifies as
  `piecewise_constant` within `max_segments=32` — the cap, not the statistic, is the limit.
  Not tuned to green; recorded as a boundary.

### Reporting integrity
- `RIME_SUITE_TOTAL` raised 193 → 200 (whitelisting this file), verified by re-running
  `tests/test_dyn_reporting_integrity.py`.

## [Unreleased] — RIME Repair v0.1.x freeze (comparative-validation milestone)

### Added (v0.2.5 round — Autonomous Epistemic Profiler, Coordinate Transmuter & Frontier Quantum Equations, `experimental`)
- **`apedp_dyn/epistemic/autonomous_profiler.py` — Autonomous Diagnostic Pre-Scan:**
  - Profiles target data topologies without human domain prompting:
    - Dynamic range $\mathcal{R}_y = \max(y)/\min(y) > 100 \implies$ selects scale-invariant `"log_mse"`.
    - Arrhenius tunneling singularity detector: $\frac{d \ln y}{d(1/x)} \approx \text{const} < 0$.
    - Power-law scaling detector: $\frac{d \ln y}{d \ln x} \approx \text{const}$.
    - Quantized plateau staircase scanner: flat derivative fraction $> 40\%$ and discrete clustering.
    - Dual-asymptotic crossover scanner: low-$x$ vs. high-$x$ boundary power divergence (Rayleigh-Jeans vs. Wien).
  - Emits immutable `DatasetProfile` with deterministic SHA-256 hash.
- **`apedp_dyn/epistemic/coordinate_transmuter.py` — Manifold Linearization & Parameter Recovery:**
  - Projects non-linear physical manifolds with essential singularities or power laws onto linearized basis spaces:
    - Log-linear singular basis: $\ln y = \beta_0 + \beta_1 \ln x + \beta_2 (1/x)$ recovers $y = C x^k e^{-c/x}$.
    - Power-law basis: $\ln y = \beta_0 + \beta_1 \ln x$ recovers $y = C x^k$.
    - Quantized staircase basis: midpoint linear fit recovers step height $G_0$ and step width $\Delta w$.
  - Emits valid `CandidateProposal` objects with closures compliant with `AGENTS.md` §1.1.
- **Discovery Verified on 3 Frontier Quantum Equations (`tests/test_dyn_autonomous_epistemic_profiler.py`):**
  - **Landau-Zener Non-Adiabatic Transition:** $P_{\mathrm{LZ}} = \exp(-\alpha / v)$ discovered with $R^2 = 1.000000$, $\alpha = 0.75$, power $k = 0.0$.
  - **BCS Superconducting Weak-Coupling Gap:** $\Delta(g) = 3.52 \exp(-1/g)$ non-perturbative essential singularity discovered with $R^2 = 1.000000$.
  - **Integer Quantum Hall Effect Conductance:** $\sigma_{xy} = G_0 \lfloor 1/B \rfloor$ discrete plateau staircase discovered with $R^2 = 1.000000$.

### Added (v0.2.4 round — RIME Quantum Equation Capabilities & Limitation Remediations, `experimental`)
- **Limitation 1 Remediated: Negative Powers in Brus Quantum Confinement (`rust/apedp-kernel`, `RimeProposer`):**
  - Added negative rational exponents $x^{-1}$ and $x^{-2}$ to `rust/apedp-kernel/src/rime/seeds.rs`, added `(-1, 1)` and `(-2, 1)` to `PERMITTED_POWERS` in `rust/apedp-kernel/src/rime/rewrite.rs`, and fixed `Node::Power` sign handling in `rust/apedp-kernel/src/rime/graph.rs` to guarantee even negative powers evaluate strictly positive.
  - Added algebraic transforms `inv` ($1/x$), `inv_sq` ($1/x^2$), and `lor_norm` ($1/(1+(x/s)^2)$) to `_ALGEBRAIC_TRANSFORMS` in `apedp_dyn/proposers/rime_proposer.py`.
  - On Brus quantum dot confinement ($E_g(R) = 1.74 + 2.45 R^{-2} - 0.28 R^{-1}$), native RIME directly recovers negative power scaling laws ($+2.03553 (R^{-2}) + 1.69599$), eliminating bounded proxy $\tanh$ extrapolation collapse and raising extrapolation $R^2$ from $-70.27$ to $+0.9959$.
- **Limitation 2 Remediated: Continuous Denominator Scale & Lorentzian Resonance (`R5_PADE_SATURATION`, `rime_repair_engine.py`):**
  - Expanded `RepairOperator.R5_PADE_SATURATION` in `apedp_dyn/epistemic/rime_repair_engine.py` to fit both Form 1 (rising saturation $a \cdot z / (1 + b |z|)$) and Form 2 (Lorentzian resonance / lineshape $a / (1 + b |z|)$).
  - Added `resonance: bool` parameter to `_wrap_pade_repaired` with covariance condition ($< 5000$) and asymptotic parameter uncertainty ($< 5\%$) identifiability diagnostics.
  - On Detuned Rabi oscillations ($P_{\max}(\Delta) = \frac{\Omega_R^2}{\Omega_R^2 + \Delta^2} = \frac{1}{1 + 0.04 \Delta^2}$), accurately recovers $a = 1.000$, $b = 0.0400 \implies \Omega_R^2 = 25.00$ on clean data ($R^2 = 1.000000$) and under 2% noise recovers $\Omega_R^2 = 24.81$ (0.76% parameter error, holdout $R^2 = 0.999968$, `identifiability_status: IDENTIFIED_STABLE`).
- **Limitation 3 Remediated: Harmonic Frequency Multipliers in Rabi Pulse Dynamics (`rust/apedp-kernel`, `RimeProposer`):**
  - Added `harmonic_seeds: bool` parameter to `SearchConfig`, `small_forms_extended`, and `RimeProposer`.
  - Emits $\cos(x)$, $\sin(x)$, $\cos(x+x) = \cos(2x)$, and $\sin(x+x) = \sin(2x)$ at Depth 0 when `harmonic_seeds` is enabled.
  - Added `cos2` ($\cos(2x)$) and `sin2` ($\sin(2x)$) STEM transforms to `_STEM_TRANSFORMS` in `RimeProposer`.
  - On resonant Rabi pulse dynamics ($P_e(t) = \sin^2(t) = \frac{1 - \cos(2t)}{2}$), native RIME natively discovers `-0.5 * (cos((t + t))) + 0.5` with clean holdout $R^2 = 1.000000$ and 2% noisy holdout $R^2 = 0.999992$.
- **Verification & Pinned Regression Suite (`tests/test_dyn_rime_quantum_equation_fixes.py`, 3/3 green):**
  - Covers negative powers, Lorentzian lineshapes, and harmonic frequency multipliers under clean and noisy regimes. All 58 canonical AGENTS.md tests and all 103 Rust kernel tests pass cleanly.

### Added (v0.2.3 round — RIME native resolution of Seed Catalog Exhaustion & Grammar Constraint, `experimental`)
- **Resolution of Seed Catalog Exhaustion (`rust/apedp-kernel/src/rime/seeds.rs`, `SearchConfig.max_seeds`, `SearchConfig.lagged_products`):**
  - Overcame the fixed 60-seed catalogue cap with configurable `max_seeds` budget parameter.
  - Added Family 6b contemporaneous lagged interaction products `(x_i[t-k] * x_j[t-k])` for $k \in \{1, 2, 5\}$ (`lagged_products: bool = false`).
  - Preserved 100% backward compatibility for all existing callers: default `small_forms_with_all_options` delegates to `small_forms_extended(..., None, false)` maintaining byte-identical catalogue output.
- **Resolution of Grammar Constraint via Native Multi-Term Additive Search (`rust/apedp-kernel/src/rime/additive.rs`, `SearchConfig.additive_terms`, `RimeProposer(additive_terms=...)`):**
  - Wired native Rust forward stepwise OLS engine (`search_additive`) into PyO3 entry points (`rime_search_json`, `rime_search_envs_json`) and Python `RimeProposer`.
  - Performs joint least-squares parameter refitting on common rows, honestly charging each added term for structural edges, regression coefficients, and Bonferroni selection multiplicity ($\lambda_c \cdot (\text{edges} + 1\text{ coefficient} + 2 \ln M)$).
  - Composes winning multi-term expressions into unified executable `Graph` objects evaluated seamlessly in both Rust and Python `compute_score`.
- **Benchmark Equivalence & Parity with GA-Large (`tests/test_dyn_rime_native_additive_and_seeds.py`, 4/4 green):**
  - On the composite population extrapolation benchmark ($y[t] = 2 x_1[t-1] x_2[t-1] + 3 x_3[t-2] + 10$), `RimeProposer(additive_terms=2, lagged_products=True, max_seeds=120)` natively recovers the composite equation with exact structural terms and true parameter values ($2.996 \cdot x_3[t-2] + 1.997 \cdot (x_1[t-1] \cdot x_2[t-1]) + 9.999$).
  - Delivers transfer parity with GA-Large across populations without requiring post-hoc referee repair:
    - Population A (test holdout): $R^2 = 0.999820$ (GA-Large: $0.9998$)
    - Population B (fresh seeds): $R^2 = 0.999767$ (GA-Large: $0.9998$)
    - Population C (distribution shift): $R^2 = 0.895609$ (GA-Large: $0.8961$)
  - Confirmed 7-distractor noise filtering ($R^2 > 0.9999$) and parsimonious single-term retention on pure single-term data.

### Added (v0.2.2 round — rational parameter identifiability & decoupled metrics, `experimental`)
- **Rational Parameter Reconciliation & R5 Scaling Audit:** Reconciled mathematical parameter convention in `R5_PADE_SATURATION` (`apedp_dyn/epistemic/rime_repair_engine.py`). For $y = \frac{1000 p^2}{1 + 2000 p^2}$, explicitly established that when fitting $\text{pred} = \frac{a \cdot z}{1 + b \cdot |z|}$ against un-saturated parent model $z = k_{\text{fit}} \cdot p^2$, the physical numerator scale is $a \cdot k_{\text{fit}} = 1000.00$, the physical denominator scale is $b \cdot k_{\text{fit}} = 2000.00$, and the saturation envelope ratio is $\alpha = b/a = 2.0000$. Resolved prior documentation ambiguity conflating the dimensionless envelope ratio $\alpha=2.0$ with the raw physical denominator coefficient $c_{\text{den}}=2000$.
- **Identifiability & Uncertainty Diagnostics in R5:** Added covariance matrix conditioning analysis (`cov_condition = np.linalg.cond(pcov)`), asymptotic parameter standard errors (`a_stderr`, `b_stderr`), boundary-hit flags, and explicit `identifiability_status` (`IDENTIFIED_STABLE` vs `PREDICTIVE_ONLY_ILL_CONDITIONED`). On narrow/weakly-saturated domains ($p \in [0.001, 0.008]$), flags ill-conditioning when denominator scale drifts (6.5% error) even if holdout $R^2 > 0.999$.
- **Decoupled 5-Metric Evaluation Schema:** Implemented independent scoring in candidate proposal metadata and parameters: `structure_match`, `parameter_recovery`, `prediction_score` ($R^2$, RMSE), `extrapolation_score`, and `identifiability_status`. High predictive fit ($R^2 > 0.99$) is strictly decoupled from mechanism or parameter identification.
- **Negative Controls & Skepticism Gates:** Verified against pure quadratic ($y = 1000 p^2 \implies b \to 0$, no false saturation) and exponential saturation ($y = 0.5(1 - e^{-u/0.5})$ where Padé achieves $R^2 > 0.999$, proving high fit alone does not certify rational mechanism).
- **Pinned Verification Suite (`tests/test_dyn_rime_rational_parameter_identifiability.py`, 5/5 green):** Covers exact recovery, noise robustness ($0.5\%, 2\%, 5\%$), domain range breakdown, negative controls, and decoupled reporting. All 74 canonical and dynamic regression tests pass cleanly.

### Added (v0.2.1 round — flag calibration, staged search, layered FPR, `experimental`)

**0. Null-calibrated whiteness flag (`apedp_dyn/epistemic/whiteness_calibration.py`)** — Guard 2's `|rho1|` flag is derived from the target's own serial dependence by block-demeaning (contiguous blocks resampled then block-demeaned, so the null inherits the target's autocorrelation while level-only fits score ~0 and cannot launder AR residuals through an offset). Replaces the fixed magic 0.5, which miscalibrated in both directions. Measured both fixes: on an AR(2) target a residual with |rho1|=0.723 is refused by the frozen 0.5 and accepted by the calibrated 0.858 (over-refusal fixed); on a white target a residual with |rho1|=0.273 is admitted by the frozen 0.5 and refused by the calibrated 0.226 (under-refusal fixed). Flag is monotone in target persistence (0.114 white → 0.858 AR(2)). Ships `screen_structured_null_worlds`, which reports proposer-level and system-level FPR **separately**: 12 proposer promotions vs 0 system promotions on 24 structured null worlds. Lives in its own module because `adjudicate.py` is a hash-pinned frozen record.
  - **Governance note (recorded, not concealed):** an initial attempt placed the calibration inside `adjudicate.py`. It was rejected by the repo's own freeze-drift guard (`FreezeDriftError` / `REFEREE OR FROZEN RECORD CHANGED`) — correct behaviour — and reverted; the capability ships as an opt-in adopter module. When no holdout target is recorded, the resolver keeps the frozen absolute floor and stamps `absolute_fallback_no_target` in the audit record, because an uncalibrated flag must remain visible as uncalibrated.

**0. Staged search + coverage/budget auditing (`apedp_dyn/epistemic/rime_staged_search.py`)** — two-stage search (cheap family-admit pass at a tiny budget, then full-budget expansion restricted to admitted families) plus `coverage_curve`, which makes coverage a measured quantity instead of an assumption. Measured budget curve: recovery saturates at 176 **evaluated** with identical best R² 0.9616 across budgets 96→2000 while reached families drop 3→2 — i.e. depth/budget claims beyond the cap are budget-bound, now pinned rather than argued. Expanded grammar (all ops ON) spends 200 vs 176 evaluated for the same winning family and the same score: expansion buys reachability, not recall. Staged run is never worse than the straight full-budget run. Structured-null FPR at the new operator count is unchanged at the system level (proposer 4/12 worlds, system 0).

### Added (v0.2.0 round — RIME structural flaw remediation & epistemic robustness, `experimental`)
- **Flaw 1 (Grammar Poverty) Remediated:** Native rational power graph node `Node::Power { src, p_num, p_den }` and structural rewrite op `RewriteOp::AddPower` added to `rust/apedp-kernel/src/rime/`. Admits non-integer rational powers $p \in \{\pm \frac{1}{3}, \pm \frac{1}{2}, \frac{2}{3}, 2, 3\}$ into seed catalogue and structural rewrites (`power_ops=True`). Features pole-safe float evaluation and sign preservation for odd roots. Recovers nuclear surface scaling $A^{2/3}$ with exact $R^2 = 1.000000$ to machine precision without manual feature engineering.
- **Flaw 4 (Small-Sample Paralysis) Remediated:** Configurable `min_rows: int = 30` parameter in `RimeProposer`. Preserves frozen default $N \ge 30$ threshold for noisy financial time-series to prevent overfitting, while allowing high-precision experimental physics datasets (e.g. Bertozzi 1964 $N=5$, $\sigma < 0.1\%$) to execute without unconditional early exits.
- **Flaw 2 (Correlated Proxy Trap) Remediated:** Identifiability banding (`identifiability_band: float = 0.002`) and empirical derivative gradient matching (`check_derivatives: bool = False`) added to proposal assembly in `RimeProposer`. Prevents false-certainty promotion when collinear hypotheses differ by $\Delta R^2 \le 0.002$, reporting `AMBIGUOUS_EQUIVALENCE_SET` and recording `competing_hypotheses` and `equivalence_band_size`. Flags `UNIQUE_WINNER` only when lead margin strictly exceeds the identifiability band.
- **Flaw 3 (Greedy Beam Monoculture) Remediated:** Diverse beam frontier selection (`SearchConfig.diverse_beam: bool = false`) added to native Rust beam search. Ensures round-robin representation across structural archetypes (polynomial, power, rational, bounded, transcendental) in the beam frontier, eliminating greedy beam starvation of non-linear candidates.
- **Verification & Pinned Test Suite (`tests/test_dyn_rime_v02_flaw_remediation.py`, 5/5 green):** Covers exact rational power recovery, small-sample laboratory mode on Bertozzi data, identifiability banding, gradient alignment, and diverse beam frontier. All 58 canonical AGENTS.md tests, all 103 Rust unit tests, and all 33 RIME/STEM regression tests pass cleanly.

### Added (v0.1.13 round — preregistered real-measurement benchmark suite, `experimental`)
- **Dataset 1: AME2020 Real Nuclear Binding Energies (1,159 real nuclides, `data/mass_1.mas20.txt`):** RIME with declared `square_seeds=True` natively discovers Coulomb electrostatic repulsion $Z^2$ and interaction $A \cdot Z$ on binding energy per nucleon without handed columns or rational powers, achieving holdout $R^2 = 0.7049$ (single-term) and $R^2 = 0.7705$ (2-term additive) on 559 unseen nuclides. Evaluated against literature Weizsäcker SEMF which achieves holdout $R^2 = 0.9921$ with physically calibrated parameters ($a_v = 15.52$, $a_s = 17.07$, $a_c = 0.698$, $a_a = 22.40$ MeV).
- **Dataset 2: Bertozzi 1964 Relativistic Electron Experiment (`data/bertozzi_1964.jsonl`):** Experimental linear accelerator measurements of electron kinetic energy $K$ vs speed $v/c$. Epistemic Referee adjudicates Classical Newtonian mechanics vs Relativistic Einstein mechanics on sealed high-energy holdout ($K \in \{4.5, 15.0\}$ MeV). Classical mechanics catastrophically fails ($v = 7.7c$, $\text{RMSE} = 4.88c$), while Relativistic mechanics achieves holdout $\text{RMSE} = 0.0055c$ ($R^2 = 0.9998$) and is promoted with `UNIQUE_SUPPORT` (zero false promotion of superluminal classical physics).
- **Dataset 3: Real Market Orderflow & Daily Asset Returns (Noise / Null Negative Control):** Evaluates factor discovery across 22 real microstructure signals on S&P 500 calm bull and COVID-19 crash regimes. Epistemic Referee strictly refuses false promotion: $\text{FPR} = 0.00\%$, 0 false promotions, $0.0\%$ capital allocated.
- **Pinned verification (`tests/test_dyn_preregistered_real_data_benchmark.py`, 3 pins):** All 3 real-data tests pass cleanly with zero regressions across the canonical suite (58/58 green).

### Added (v0.1.12 round — full STEM grammar expansion: options 1, 2, 3, `experimental`)
- **Option 1: Native Rust Kernel STEM Primitives & Python Adapter (`SearchConfig.stem_ops`, default `false`):** adds elementary scalar STEM primitives `Node::Sin`, `Node::Cos`, `Node::SafeExp` (clamped to $[-60, 60]$ preventing float overflow), `Node::SafeLog` ($\ln(1 + |x|)$ eliminating negative/pole hazards), and `Node::Diff` ($\Delta x_t = x_t - x_{t-1}$) to `rust/apedp-kernel/src/rime/graph.rs`, `rewrite.rs`, `search.rs`, `seeds.rs`, and `lib.rs`. Exposed via `RimeProposer(stem_ops=...)` and expanded STEM transforms (`sin`, `cos`, `exp`, `log`, `diff`) in `_ALL_TRANSFORMS` while preserving `_ALGEBRAIC_TRANSFORMS` byte-identity.
- **Option 2: Modular Multi-Engine STEM Proposers (`apedp_dyn/proposers/stem_proposers.py`, `experimental`):**
  - `HamiltonianProposer`: Discovers energy-conserving mechanical oscillator dynamics ($\dot{q} = p/m$, $\dot{p} = -k q - \alpha q^3$), validating energy conservation $\frac{dH}{dt} = 0$, and rejecting dissipative systems.
  - `SolenoidalFluidProposer`: Discovers incompressible 2D velocity fields via stream function $\psi(x, y)$ analytically guaranteeing zero divergence ($\nabla \cdot \mathbf{u} \equiv 0$).
  - `CompartmentalSirProposer`: Discovers epidemiological compartmental dynamics ($S, I, R$) with exact mass/population conservation ($S + I + R = N$).
- **Option 3: Typed Physical Dimensional Homogeneity Engine (`apedp_dyn/epistemic/dimensional_homogeneity.py`, `experimental`):** AST-based recursive dimensional metrology and validation engine leveraging SI base dimensions from `Dimension`. Enforces Buckingham-$\Pi$ dimensional consistency: strict dimension matching for additions/subtractions, rational power exponents, and dimensionless arguments for transcendental functions (`sin`, `cos`, `exp`, `log`, `tanh`). Exposes `DimensionalHomogeneityGate` for epistemic referee adjudication.
- **Verification & Pinned Test Suite (`tests/test_dyn_stem_grammar_options_123.py`, 10/10 green):** complete regression coverage across native primitives, physical proposers, dimensional gates, and negative controls (rejection of unphysical / dissipative / dimensionally inconsistent models). Zero regressions across all suites (58/58 canonical AGENTS.md suite, 27/27 RIME/STEM suite).

### Added (v0.1.11 round — rational operations & Padé saturation, options 1, 2, 3, `experimental`)
- **Option 1: Native Rust SafeDiv rational operations (`rust/apedp-kernel/src/rime/`, `SearchConfig.rational_ops`, `RimeProposer(rational_ops=...)`, default `false`):** adds pole-free `Node::SafeDiv(a, b)` evaluating $x / (1 + |y|)$ to the kernel rewrite operators and search graph. Enables symbolic discovery of Padé rational approximants directly in mechanism space. Achieves $R^2 = 1.0000$ on rational targets like $y = x / (1 + x)$. Zero poles or zero-division hazards on $\mathbb{R}$.
- **Option 2: Epistemic repair engine Padé saturation operator (`RepairOperator.R5_PADE_SATURATION`, `apedp_dyn/epistemic/rime_repair_engine.py`):** bounded non-linear envelope fit $g(z) = \frac{a \cdot z}{1 + b \cdot |z|}$ targeted at `FINDING_CURVATURE_SUSPECT`. Demonstrates machine-zero algebraic exactness on clean plants ($R^2 = 1.0000$, $\text{RMSE} = 3.8 \times 10^{-17}$), and robust $R^2 = 0.9068$ under 2% realistic measurement noise ($R^2 = 0.2024$ under 10% severe distortion).
- **Option 3: Normalized saturating algebraic transforms (`_ALGEBRAIC_TRANSFORMS`, `apedp_dyn/proposers/rime_proposer.py`):** added `sat`, `sat_sq`, `sat_norm`, and `sat_sq_norm`. Normalizes inputs by frozen training standard deviation $\sigma = \text{std}(x)$ so that $(p/\sigma)^2 / (1 + (p/\sigma)^2)$ captures physical saturation at threshold, maintaining stable discovery performance under noise ($R^2 = 0.9753$ clean, $0.9750$ at 2% noise, $0.9752$ at 10% noise).
- **Pinned verification (`tests/test_dyn_rime_rational_and_pade.py`, 6 pins):** native kernel discovery, default grammar preservation, Padé repair candidate generation, sealed holdout repair pipeline, normalized transform discovery, and independent 0%/2%/10% noise calibration table. Zero regressions across the 58-test canonical suite and 118 RIME tests.

### Added (v0.1.10 round — contemporaneous powers & reachability remediation, `experimental`)
- **Opt-in contemporaneous square seeds (`rust/apedp-kernel/src/rime/seeds.rs`, `SearchConfig.square_seeds`, `RimeProposer(square_seeds=...)`, default `false`):** admits same-time self-multiplication `(x_i * x_i)` into seed catalogue family 5 at depth 0. Closes the measured QEC reachability failure (previously 0/1217 candidates were the true contemporaneous law, forcing search to emit lagged proxies or rely on Python repair). Raw `[p]` discovers `p_L = 1000*p^2` directly at depth 0 with holdout R² > 0.9995.
- **Discrimination against grid-proxy fitting (`tests/test_dyn_rime_reachability_and_contemporaneous_qec.py`):** physical law `(p * p)` is 100% reversal-invariant (|R2_fwd − R2_rev| < 1e-12, holdout R² 0.9995), whereas legacy lagged proxy `(p * p[t-5])` error inflates by >10× under row reversal.
- **Systematic curvature screen against wrong-family emissions:** on d=5 (planted cubic law p^3), a quadratic fit `(p * p)` triggers `FINDING_CURVATURE_SUSPECT` in `classify_findings`, whereas on d=3 true quadratic truth curvature is cleared (`no_reliable_structure`).

### Added (v0.1.9 round — relative-gain criterion, rational-power family, identifiability band)
- **Relative SSE-reduction criterion (`RepairConfig.min_sse_reduction`, default 0 = frozen):** G = (SSE_base − SSE_cand)/SSE_base on the same validation rows, reported alongside absolute R² headroom on every gate. Preregistered companion for high-baseline problems where an absolute R² bar fails by construction (measured on AME2020 below). Caller chooses; neither criterion is inferred.
- **Domain-informed rational-power term family (`RepairConfig.rational_powers`, default empty = frozen):** sign-safe x^p terms of declared dependencies appended to the R2 catalogue, enabled ONLY through the explicit config field — Layer-C priors enter declared, never as hidden corrections. Scored by the same gates.
- **Identifiability band (`RepairConfig.equivalence_r2_tol`, default 0 = frozen single-winner behaviour):** passing candidates within the band of the best are observationally indistinguishable, and the run resolves to `AMBIGUOUS` (MULTI_HYPOTHESIS territory) instead of crowning the marginally-highest scorer. Sealed split is never opened on AMBIGUOUS.
- **Decisive real-data measurement (`tests/test_dyn_real_data_ame_repair.py`, 6 pins):** with the physics-motivated family admitted, the AME2020 ranking on validation is proxy (A·Z) 0.99960 > (A·A) 0.99930 > **surface term (A^0.667) 0.99915** > (A^0.333) 0.99898 > (Z·Z) 0.99842. On the sealed split the proxy and the mechanism differ by <1e-3 R². With the band declared the engine returns `AMBIGUOUS` (set size 4) rather than a unique winner — predictive success is reported separately from identification, which is the SEMF's job, not the search's. SEMF literature coefficients still reach R² 0.9999 on the same frame.
- Control-design note recorded: a candidate cap of 10 was measured to be exactly consumed by R1 + the 9 interaction probes, silently excluding the power terms; the pinned run uses cap 14 (caught by an assertion that failed for the right reason).


### Added (v0.1.8 round — REAL-data repair benchmark, AME2020 masses, no planting)
- **`tests/test_dyn_real_data_ame_repair.py` (3 pins):** answers the standing methodological criticism that every demonstrated repair ran on planted synthetic worlds. Runs discovery → diagnosis → bounded repair on genuinely measured data (AME2020 atomic-mass evaluation, `data/mass_1.mas20.txt`, parsed to binding energies; 1159 usable nuclides).
  - Discovery recovers the semi-empirical mass formula **volume term**: `+7426.14*A + 93729.3` at dev R² 0.994.
  - Diagnosis on that original fires `curvature_suspect` (the missing surface term A^(2/3)).
  - **Threshold calibration exposed by real data:** at the preregistered absolute bar (0.02) the engine returns `NO_IMPROVEMENT` — the bar is mis-scaled for a real dataset whose original already explains 99.4% (achievable gain ≈ 0.005). At a scale-appropriate bar (≤0.005) it returns `IMPROVED` with the `(A*Z)` term at sealed R² **0.9995** vs original 0.9947 — the first sealed-validated repair on measured data in this repository. Both outcomes pinned; the frozen default is untouched (a relative-gain criterion is proposed as a preregistered follow-up, not applied unilaterally).
  - Literature reference on the same frame: SEMF with standard coefficients reaches R² 0.9999 (units keV), so the data support the known law — the boundary is grammar reachability (A^(2/3) absent from the transform set) and threshold calibration, not data quality.
  - Honesty note pinned in the test docstring: the added `(A*Z)` is a *correlated proxy* for the Coulomb/surface structure, not the SEMF term — predictive gain without unique identification (MULTI_HYPOTHESIS territory, deferred to the APEDP-X track); nothing is promoted.
  - Control-design bug caught during the round: reusing one `RepairPartitions` across a threshold sweep raised `HoldoutContaminationError` on the second run — the guard working as designed; the sweep now builds fresh partitions per run.


### Added (v0.1.7 round — intraday crypto sweep on fetched minute bars, audit-only)
- **`tools/fetch_intraday_crypto.py` + `data/market/intraday/` (new, clearly-labeled corpus):** Binance public klines, 1000 bars per symbol per interval at 1m/5m/15m/30m/1h, over the crypto legs of the repo's OWN universes (15/30/45 lists from `multi_asset_signals_sandbox.py`; 14/26/39 fetchable — MATIC delisted, MKR/EOS/XMR/FTM/ICX not currently listed). Own PROVENANCE.json; the frozen daily corpus is untouched.
- **Measured (`tests/test_dyn_intraday_universe_sweep.py`, 5 pins):** real basket forward returns abstain in **15/15** universe×interval cells under both seed orders (no grammar-expressible edge at any tested resolution); null worlds 0 proposals; planted lag-1 control recovered at dev R²>0.98 at every interval (pipeline verified on minute bars); per-position coverage on the 39-input frame follows the 60-form arithmetic exactly — `coverage_first` recovers inputs 0–20 (39 levels + 21 lag-1) while 21+ remain generation gaps.
- **Scope limits recorded, not papered over:** Binance has no 45m interval (requested 45-minute frame does not exist as an exchange bar); equity legs unfetchable (Yahoo Finance intraday HTTP 429 at fetch time) so 60/90-asset intraday universes are out of scope; equal bar counts confound interval with calendar span (1000 1m ≈ 17h vs 1000 1h ≈ 42d).
- **Control-design bug caught and fixed during the round (recorded for audit):** the first planted-position control propagated a NaN std into the planted target (one symbol carries an extra timestamp → NaN row), so the finite-row guard abstained — misread as a RIME limitation until isolated by a solo-frame test. Frame construction now `dropna()`s the pivoted bars first.


### Fixed (v0.1.6 round — generation gap closed, opt-in `seed_order="coverage_first"`)
- **Measured gap (v0.1.6 instrumentation):** the legacy input-major family-2 (delayed levels) truncates the 60-form seed catalogue at high input counts and dropped whole inputs' shortest lag — 3/22 positions (H-RV, H-A, H-OBV_MOM) could never be recovered on the 22-signal frame while a solo-frame control recovered the identical dependence at R² 0.96–0.9995.
- **Fix (`rust/apedp-kernel/src/rime/seeds.rs`):** opt-in lag-major family-2 ordering (`SeedOrder::CoverageFirst`, config key `seed_order`, `RimeProposer(seed_order=...)`, default `"legacy"`). Same budget, same families, same forms whenever the legacy order fits inside the budget (Rust-pinned for n≤6); different WHICH forms survive truncation at many inputs. No threshold, gate, or frozen-spec default changed; `static`/`additive` paths unaffected.
- **Re-measured under the new order:** the 3 gap positions recover the planted column at their solo-frame R² (H-RV 0.9937, H-A 0.9995, H-OBV_MOM 0.9597 — previously 0.19 / 0.68 / 0.09); 19/22 exact + those 3 = **22/22 name the planted column** (the only non-exact recoveries remain the 2 collinear substitutes H-OFI→H-CVD, H-STOCH→H-WILLR, a non-identifiability property, not generation). Null-world false promotions **0/50, unchanged**. Pinned by `tests/test_dyn_rime_seed_order_coverage.py` (3) + 3 new Rust catalogue tests (99/99).


### Added (v0.1.6 round — search-funnel instrumentation + per-position coverage, audit-only)
- **Search-funnel counters (`rust/apedp-kernel/src/rime/search.rs`, `SearchOutcome`):** additive-only `rejected_score` / `dropped_dedup` / `dropped_scope` alongside the existing `evaluated` / `accepted`, plus per-proposal `inputs_used` (`Graph::inputs_used()`, output-reachable arena walk). No score, gate, selection or default changed. Invariants pinned: `rejected_score + accepted == evaluated`, `evaluated <= budget`. Passed through to proposal metadata (`inputs_used_names`, counter keys) in `apedp_dyn/proposers/rime_proposer.py`; `rime_usage_log.py` now prefers the structural list over string mining.
- **Per-position coverage challenge (`tests/test_dyn_rime_coverage_funnel.py`):** planted lag-1 dependence at each of the 22 trading-signal positions (y = 2·x[t-1] + 10 + noise, fixed seeds). **Measured: 17/22 exact recovery; 2/22 collinear substitute** (H-OFI→H-CVD, H-STOCH→H-WILLR — in-grammar twins fitting identically, MULTI rationale); **3/22 generation gap** (H-RV, H-A, H-OBV_MOM — the solo-frame control recovers the identical dependence at R² 0.96–0.9995 while the 22-input frame lands at R² 0.09–0.68, proving the loss is seed-catalogue truncation at 22 inputs, not scoring). Null worlds abstain 2/2. This is the reviewer-directed generation-vs-qualification separation: what is never generated (truncation) vs generated-but-not-selected.
- Fixed during this round: a prefix-collision bug in the kernel's equation-name substitution (`"v13"` was corrupted to `"H-CVD3"` by ascending `String::replace`) — replaced with an exact two-pass control-character-placeholder substitution; rendered equations now match structural `inputs_used_names` verbatim (pinned).


### Added (v0.1.5 round — power calibration + additive refinement, `experimental`, kernel level)
- **`rust/apedp-kernel/tests/rime_power_calibration.rs` (`audit-only`):** recovery / promotion / multi-term rates with Wilson 95% intervals, functional recovery on fresh independent series. Measured: single-term recovery follows $n \gtrsim 21/\rho^2$ (near the ≈$19/\rho^2$ Bonferroni limit for ~200 candidates), null 1/120 promoted; 0% recovery at IC 0.02 below $n=15000$ is the information limit, not a proposer defect.
- **`rust/apedp-kernel/src/rime/additive.rs` (`search_additive`) + kernel config key `additive_terms`:** opt-in multi-term refinement after the frozen gate. Closes a measured representation gap (`x[t-1]+x[t-2]` recovered 0/40 in every cell before; up to 40/40 after for two-term laws, with 0/40 spurious extra terms on single-term laws and identical promotion counts everywhere). Added terms pay edges + 1 coefficient + `2 ln M` (pool size); `Const` coefficients are charged explicitly because `edge_count` treats them as free. Default OFF; `search` is untouched; `seeds::shift` became `pub(crate)`. Not wired into the Python `RimeProposer` (needs pool-size multiplicity in the trial ledger first).
- No frozen spec file, threshold or `canonical_preregistered_spec_v1.json` entry was modified.


### Added (v0.1.4 round, engine tag `rime_v0.1.4` — refusal wiring + repeat bounds)
- **Referee-side out-of-grammar scope refusal (`apedp_dyn/epistemic/scope_refusal.py` + `--adjudicated` in `benchmarks/known_truth_challenge.py`, `experimental`, spec §16d):** closes the §16c negative-control gap (scope refusal must come from adjudication). The referee fits a preregistered family-distinct probe pair (exp family, rate grid [1, 50], dev-normalized; log family, raw axis) from dev data ONLY, and adjudicates each emitted candidate pairwise via the standard `adjudicate()` contract; promotion requires UNIQUE_SUPPORT against EVERY probe. Measured: both methods' exp-world spurious emissions refused (rival-favored bootstrap / indistinguishable rival); all 22 in-grammar true recoveries survive un-declined; `out_of_grammar_emitted_promoted` = 0 everywhere measured. Audit transparency: the first adjudicated run promoted the OLS emission — diagnosed as a referee-side probe defect (axis clipped to [0,1], degrading the exponential probe on extrapolating log grids — the recorded OLS extrapolation failure mode in referee form) and fixed BEFORE any citation, with the strict-AND semantics untouched.
- **Repeat bounds (preregistered, `--repeats`):** 5 observation-noise seeds (phase = 23.0 + 1009·seed; same design points, fresh draws; seed 0 byte-reproduces the pinned v0.1.3 frames), full world set, adjudicated. Artifact `benchmarks/known_truth_results_repeats.json`: recovery 1.0 ± 0.0 per method per seed (11/11 in-grammar every seed), per-world holdout R² std ≤ 5e-5, coefficient accuracy stable (|Ĉ−C|/C ≲ 5e-4 RIME), exp-world emissions refused 5/5 with zero wrongful in-grammar refusals (bounds are noise-driven only). Pinned by `tests/test_dyn_rime_known_truth_scope_repeats.py` (7 pins: referee blindness on all planted exponents, true-monomial UNIQUE_SUPPORT + wrong-exponent refusals, adjudicated single-run refusal without §16c regression, 3-seed repeat bounds, seed-rule frame determinism); `tests/test_dyn_rime_known_truth.py` (§16c) unchanged and green.
- **Population-extrapolation extensions (`tests/test_dyn_population_extrapolation_v2.py`, pinned and re-executed this round):** RIME depths 1–6 byte-identical (`x3[t-2]` on every population), +sq transform axis moves nothing, GA basis ladder isolates products (linear+lag / +squares / +products → only products close the gap), repair-mediated RIME seals 0.9998 with B 0.9998 / C 0.8952 transfer matching GA-large end-to-end — no spec change required.

### Added (v0.1.2 static-grammar round, engine tag `rime_v0.1.2`)
- **Matched-compute harness (`benchmarks/matched_compute.py`, audit-only):** equal-budget comparison of RIME (temporal + static grammars), the existing proposers, the algebraic OLS-basis control, and **PySR 2.7 (head-to-head, installed with Julia 1.11.9)** — scored on a preregistered **grammar-coverage axis**: out-of-coverage abstention is correct behaviour, an emitted-but-failed proposal is a failure on any family, and static worlds require holdout **reversal-invariance** so grid-order tracking is scored `index_fit`. Fairness fixes from the measured audit: 5 repeats with un-stably-verdict screening, full environment metadata, per-cell wall-budget utilization (equal allowance ≠ equal utilized compute — RIME uses ~0.3–0.7% of the allowance, DLI ~8%), PySR runs serial-deterministic with a BINDING 2-s timeout (one-time Julia warmup excluded) and lag columns engineered for information parity (its operator set has no time-shift; its attempted coverage excludes input-side IIR laws). Citable conclusion at equal 2-second serial budgets: **complementarity, not dominance** — rime_temporal 4/5 (fails static_log), PySR 3/5 (fails static_log at this budget, out of scope on the state law).
- **Static search mode (kernel `config["static"]`, default OFF):** temporal primitives (lags, memory cells) are removed from seeds AND rewrites — grid-order fitting becomes unrepresentable for order-structured static data. No threshold or gate changes; the default grammar is byte-identical to v0.1.1.
- **Algebraic input transforms (`RimeProposer(algebraic_transforms=[...])`):** canned `sq/cub/ssqrt/slog` transform columns appended at the input layer, recomputed identically inside `compute_score` so proposals stay causally usable on unseen frames; NaN-preserving, unknown kinds rejected at construction.

### Fixed (v0.1.2 — this is also a CORRECTION of a prior conclusion)
- **Known-truth recovery challenge (`benchmarks/known_truth_challenge.py` + pinned wrapper `tests/test_dyn_rime_known_truth.py`, audit-only):** blind structural-recovery test answering the reviewer-directed open question. 11 in-grammar planted monomial worlds (d ∈ {2,3,6} × grid {log,uniform,irregular} × noise 1×/3× relative-to-signal × coefficient/offset variation) + 1 exp() negative control. **Measured: RIME static grammar recovers the planted structure on 11/11 (max |Ĉ−C|/C ≈ 5e-4, tighter than the BIC-OLS control), 0 false promotions in-grammar; on the out-of-grammar control RIME does NOT abstain (emits a spurious fit) — scope refusal must come from adjudication, not the search.** Harness numerics fixes recorded before citation: dev-slice-relative noise σ, standardized structure regression / OLS design, BIC-selected sparse OLS as the correctly-specified baseline (a full-degree polynomial OLS is documented as failing on extrapolating grids: holdout R² −33). Abstention never counts toward recovery; per preregistration.
- **Kernel phase profile (`rust/apedp-kernel/tests/rime_profile.rs` → `benchmarks/kernel_profile_results.json`, audit-only):** per-candidate decomposition into graph evaluation / readout fit / search bookkeeping (remainder estimate, median of 7). Measured on the planted lag-2 world with the release profile: at n=150 the per-candidate cost is ~19 µs of which ~90% is search bookkeeping (enumeration, dedup, gate arithmetic), while at n=12 000 graph evaluation dominates (~181 µs vs ~54 µs readout fit). Operational consequence for v0.2 planning: residual-guided evolution should first target the enumerative/bookkeeping layer at small n, not the arithmetic.
- **Scale-dependent ridge in `fit_readout` (engine numerics, `rust/apedp-kernel/src/rime/eval.rs`):** the fixed `1e-9` literal added to `Σ(o−ō)²` exceeded the raw second moment of algebraic-scale regressors (a p² column ≈1e-8) by ~30× and silently zeroed every such fit — this, not the grammar, was the primary cause of the measured QEC static-law failure. The ridge is now relative to the regressor's raw second moment (identical for unit-scale data; the QEC attribution "RIME's grammar is the wrong instrument for static scaling laws" is retired: the temporal grammar reaches the law via its self-interaction rewrite, pinned by `test_dyn_ga_qec.py` at holdout R²>0.99 both directions).
- **`tests/rime_regression.rs` was stale against the v0.1.2 typed API** (referenced the removed `SearchPacket` surface and did not compile); rewritten as equivalents against `SearchOutcome` with true null-abstention, positive-control and budget assertions; `rime_stage1.rs` assertions recalibrated to commit 30e73bf's planted amplitude (previously internally inconsistent: even a perfect fit could not clear the control's own bar).

### Added (this round, earlier)
- **RIME Repair diagnosis v0.1.0 (`apedp_dyn/epistemic/rime_repair_diagnosis.py`, experimental):** frozen `ResidualReport` + finding vocabulary on real proposals; all cutoffs explicit caller args.
- **RIME Repair engine v0.1.1–v0.1.4 (`apedp_dyn/epistemic/rime_repair_engine.py`, experimental):** bounded R1/R2/R3 operators, chronological 4-role partitions with single-use sealed split (`HoldoutContaminationError` on reopen), five explicit gates, prediction-equivalence clustering. Promotion structurally deferred to the APEDP-X track (no CandidateProposal→ScoredCandidate bridge fabricated).
- **Sensitivity/recall milestone (v0.1.2–v0.1.3):** recall 9/9 strong planted defects with structure recovery; weak-defect curve never harms; BTC abstention regression; 5/5 walk-forward abstain; adversarial stability-block.

### Fixed
- **RIME `residual.rs` stub:** removed hardcoded threshold literal; residuals now use the fitted readout (consistent with `eval.rs` scoring); threshold is an explicit caller argument with regression tests.

### Model-class closure (experimental engine only — no frozen-spec change)
- Interaction-probe screen (`interaction_suspect`) + lagged square/product R2 terms mirrored 1:1 (probe == term by construction); lag choices cite the frozen LAG_SET's two shortest entries.
- Planted composite (linear part + lagged product of independent drivers) completes through diagnosis→R2→gates→sealed; QEC-conservative refusal preserved (3 refused rows).
- Flaw-closure round: product recall 0/5 budgets (penalty-bound); recall 18/18 with 95% lower bound; single-round termination pinned (multi-round chaining documented as unreachable-by-construction under veto semantics); wall-time resource table; read-only whiteness-guard sweep (operating point 0.3–0.5, default untouched).
- #11 selective risk/coverage (post-hoc refusal policy, no harness changes): GA/OLS attempt signal worlds and abstain noise worlds; QEC thirds split roles (OLS refuses flat fit, repair upgrades to sealed 0.987).
- #7 floor sweep via explicit tau_epi (default untouched): near-tie flips across bracket monotonically; decisive UNIQUE at 0.05+0.1; identical-predictions MULTI at all floors; anchor artifact verified on disk.

### Documented limitations (capability boundary, not a defect)
- Same-driver omitted lags are usually absorbed by OLS into ~white residuals (4/6 sensitivity corners silent); firing corners yield gated rejection. R3's demonstrated domain is emergent lag from an independent driver. Whiteness alone is insufficient for structural identification.
- Whiteness gate assumes stationarity across evaluation rows: on log-spaced QEC data, quadratic extrapolation misfit trips the temporal screen and vetoes an R²=0.95 repair (pinned conservative behavior; refusal is the safe direction).

### Frozen scope
- No further grammar expansion. Next milestone is comparative validation (matched-compute repair vs proposer, independent inspection), not new operators.

## [2.0.0] - 2026-10-12

### Added
- **Full-Lineage Experiment & Candidate Ledger (`FullLineageExperimentLedger`):** Complete lineage tracking recording candidate ancestry, source proposer, feature sets, hyperparameters, random seeds, and specific dataset slice exposures.
- **Anti-Snooping Holdout Guard (`HoldoutContaminationError`):** Permanent holdout slice burning. When an empirical hypothesis fails Stage 4 out-of-sample confirmation, the evaluated slice `(dataset_id, slice_id)` is permanently blacklisted. Any iterative tweak attempting to reuse it triggers `HoldoutContaminationError`.
- **Six-Stage Scoped Promotion State Machine (`PromotionTracker`):** Replaced monolithic boolean promotion flags with six non-fungible sequential stages:
  1. `MATHEMATICAL_CONSISTENCY` (Lean 4 kernel + dimensional invariants)
  2. `PREDICTIVE_RECOVERY` (Cross-validated out-of-sample $R^2 > 0.05$, $p < 0.01$)
  3. `UNIQUE_IDENTIFICATION` (Diebold-Mariano test $p < 0.01$; literature projection $R^2 < 0.85$)
  4. `OUT_OF_SAMPLE_TRANSFER` (Un-snooped temporal regimes and cross-asset transfer retention $\ge 50\%$)
  5. `ECONOMIC_VALIDITY` (Net Sharpe after fees, breakeven fee $\ge 3.0$ bps, $DSR \ge 0.95$)
  6. `REAL_MARKET_PROMOTION` (Live sandbox / paper-trading execution stability)
- **Calibrated Three-Tier Benchmark Suite (`calibrated_benchmark_suite.py`):**
  - Tier 1: Referee discipline measuring FPR ($\le 0.05$), CAR ($\ge 0.70$), and VRR ($\ge 0.75$) with exact Wilson score confidence intervals.
  - Tier 2: Discovery statistical power ($1 - \beta$) across a calibrated grid of effect sizes ($IC \in [0.02, 0.05, 0.15]$), fat-tailed Student-$t$ noise ($\nu=4$), and structural regime switches.
  - Tier 3: Transfer generalization measuring retention decay across distinct out-of-distribution market regimes.
- **Canonical Preregistered Specification (`specifications/canonical_preregistered_spec_v1.json`):** Single authoritative source of truth for all preregistered constants, guarded by SHA-256 hash verification in `apedp_dyn/conf/canonical_spec.py`.
- **Documentation & Configuration Integrity Suite (`tests/test_documentation_and_spec_integrity.py`):** CI tests that fail if runtime code, test suites, or documentation drift from frozen canonical specifications.
- **Discovery Power Calibration Curves (`generate_power_curve_matrix`):** Power calibration evaluating theoretical vs empirical detection thresholds ($\Phi(\sqrt{N} \cdot IC - z_\alpha)$) with Wilson CIs across sample sizes and noise structures.
- **Expanded Proposer Suite:**
  - `DistributedLagInteractionProposer`: Distributed lag kernels ($k \in \{1, 2, 5, 10\}$) and bilinear feature interactions.
  - `RegimeSwitchingProposer`: Piecewise threshold models adaptive to market volatility states.
  - `HawkesOrderFlowProposer`: Self-exciting point processes modeling order arrival clustering and quote replenishment.
- **Regime Robustness Diagnostic (`evaluate_regime_robustness_diagnostic`):** Environment-stratified predictive incrementality evaluating cross-regime forecast survival without rigid residual homoskedasticity vetoes.
- **Decoupled Execution Cost Models (`execution_cost_models.py`):** Separated cost models: `FixedPlusSlippageCostModel` for slower factors, `AlmgrenChrissImpactCostModel` for temporary/permanent market impact, and `LimitOrderQueueCostModel` for queue fills and adverse selection.

### RIME v0.1.2 — Reliability & Evidence Integrity (completing the review specification)

The evidence round was extended with the reliability guarantees the review checklist required (no grammar growth, rules 1–3 unchanged):

- **Frozen environment handling.** `MIN_ENV_FINITE_ROWS = 15` and three explicit `EnvStatus` values (`Scored` / `TooFewRows` / `Malformed`): short environments are excluded from every aggregation, ragged ones fail loudly, all-unusable input abstains with an explicitly infinite reference and zero evaluated candidates, NaN gaps are skip-and-count (there is a gapped-environment recovery test), and per-env statuses ship in the search packet. Regression tests cover heterogeneous 3-env worlds, exclusion, and degenerate abstention.
- **PruneEdge semantics proven, not assumed.** New `rime_prune_semantics.rs`: exactly-zero contribution; complexity/DoF accounting agrees everywhere (zeroing, not removal — stated in the packet); content address changes so pruned ≠ parent; byte-reproducible predictions across serialization; no collateral edits; a fully zeroed graph yields an intercept-only readout that cannot outscore the null.
- **Explicit search limits.** Structural rewrites provably never CREATE a `Sign`/`Abs` node (depth-3 expansion over the whole universe; wrapping an inherited one is composition and is allowed from seeds only, reachability-tested); an off-grid 0.87 decay is never rendered as an exact `state_0.8700` and its in-sample R² is honestly < 1; budget-1 and empty-seed searches abstain explicitly.
- **Benchmark provenance & determinism.** The artifact records benchmark version, git revision (+dirty flag), kernel version, config, seed policy, timestamp, the feature-availability rule, and a rime-determinism self-check; recovery/accuracy/cost/complexity remain separate fields.
- **APEDP-X boundary tests.** Graph identity is checked end-to-end (Rust id = proposal id = packet id, 64-hex), malformed graphs (self-referencing arena, out-of-frozen lag) raise explicitly at the kernel, the referee contract field list is asserted complete, and the no-confirmation-feedback property was verified structurally (no RIME references in epistemic/, no referee imports in RIME).

### RIME v0.1.2 — Evidence Round (multi-environment scoring, stress benchmark, terminology)

**Added.** Addressing the v0.1.1 review's three next steps, without adding structure beyond the frozen primitives.

- **Multi-environment selection (`search_envs`, `rime_search_envs_json`, `RimeProposer(per_frame_envs=True)`).** Environments are scored separately under a rule frozen before scoring and reported in the packet: `MeanPerEnv` (equal weight per environment — the literal §8 reading, fixing the pooled path's length weighting) and `MedianWorstGated` (median across environments plus a **worst-environment improvement gate**: the candidate must beat its parent by more than that environment's own noise scale $\gamma\,\mathrm{var}(y)/n$, with the judged environment selected by the *candidate's* lowest relative improvement — selecting by the parent's MSE would hand the decision to whichever environment has the largest variance, a property of the data, not the mechanism). A dominance-trap regression test at both layers pins the behaviour: one long stream's wins cannot outvote a short stream where the mechanism is absent; the consistent-law control passes the same gate.
- **Broader benchmark (`benchmarks/rime_vs_neural.py --stress`, artifact `stress` key).** Noise sweeps (0.02, 0.5, 1.0), amplitude sweeps (0.5, 3.0), on-grid (0.80) and deliberately off-grid (0.87) memory decays, each with and without a ≈0.9-correlated distractor input — 18 cells. RIME reported the right structure in 18/18 cells; the artifact records the honest details (e.g. `lag a=0.5, noise=1.0` → OOS $R^2$ 0.154; off-grid `state d=0.87` recovered via the nearest grid member — the documented discrete-decay approximation). Not done here, deliberately: PySR and existing-proposer matched-compute runs remain pending their own environment.
- **Terminology:** `compute_score` is documented and tested as **past-information-only** — a temporal (no look-ahead) property of the signal path, explicitly **not** a causal claim; the proposal metadata carries `signal_semantics`, `per_frame_envs` and `aggregation`.

### RIME v0.1.1 — Stage-1 Engine Integration

**Added.** The RIME symbolic-mechanism engine (`rust/apedp-kernel/src/rime/`) is now correct, searchable, and wired into the proposer boundary.
  - `Node::State`: a genuine persistent memory cell, $h_t = d\,h_{t-1} + u_t$ with $h_{-1} = \text{init}$, replacing the `AddState` encoding that was not a recursion and carried the input term with the wrong sign.
  - `rime/eval.rs` fits a ridge least-squares readout $a\,o_t + b$ before scoring (spec §7 required fitting; it was never implemented) and charges the two readout degrees of freedom plus an EBIC-style selection charge $2\ln M$ for the search width (spec §10, "Bonferroni over grammar size").
  - `rime/seeds.rs`: deterministic 60-form seed catalogue (levels, delayed levels, differences, momentum, both lead-lag interaction orientations, memory cells on a discrete decay grid, wrappers). Without it the five frozen rewrites cannot reach difference or momentum forms at all.
  - `rime/search.rs`: deterministic budgeted beam search anchored on the mean predictor, with expression-level deduplication and per-candidate referee packets (graph id, equation, dev evidence).
  - `apedp_dyn/proposers/rime_proposer.py`: RIME as a `CandidateProposal` producer with a past-information-only `compute_score` (temporal hygiene, explicitly not a causal claim) that re-evaluates the graph on unseen frames; abstains on null worlds.
  - PyO3 entry points `rime_search_json` and `rime_evaluate_series_json` (batch-oriented FFI, spec §12).

### Fixed & Remediated
- **RIME per-node memoisation returned wrong values (Stage-1 audit defect 1):** the evaluator cached one value per *node* instead of per *(node, time step)*, so any node read both directly and through `Lag` returned the current row in both places; `x[t] - x[t-1]` evaluated to `[NaN, 0, 0, 0]` instead of `[NaN, 1, 1, 1]`. Evaluation is now series-at-a-time in arena (topological) order, releasing each node's series once its last consumer has read it, with unreachable nodes skipped and non-topological arenas rejected by `Graph::validate` instead of silently producing numbers. Regression: `tests/rime_regression.rs::difference_term_keeps_its_values`, `rust/apedp-kernel/src/rime/graph.rs::repeated_operand_is_not_memoised_across_time`.
- **`AddState` was not a state (defect 2):** all-ones input at $d = 0.5$ returned `[NaN, -0.5, -0.5, -0.5]`; a true decaying memory returns `[1, 1.5, 1.75, 1.875]`. Regression: `tests/rime_regression.rs::add_state_is_a_recursive_decay`.
- **Candidates were scored on their raw output (defect 3):** a correctly structured candidate with a small amplitude and a large offset was invisible (improvement $\approx 0$ against $\mathrm{var}(y)$ in the thousands). Regression: `tests/rime_regression.rs::scale_and_intercept_are_fitted_before_scoring`.
- **The bundled null test proved nothing (defect 4):** it charged 1.0 per edge while the world under test had MSE $\approx 0.08$, so no candidate could ever be promoted. The control now runs on the sampling-noise scale, is paired with a positive control through the identical scoring path, and bounds the false-promotion rate over 24 independent null worlds. A short-series arm (`n = 150`, 60 candidates) additionally pins the multiplicity charge; without it a fraction-of-variance penalty promotes fit noise.
- **Tactical Active Fraction Harmonization:** Resolved historical documentation drift where documentation stated $[5\%, 95\%]$ while code enforced $[5\%, 90\%]$. The range is now pinned strictly to $[5\%, 90\%]$ across code, documentation, and regression tests.
- **Audit Documentation Location:** Moved engineering audit report to [`docs/apedp_discovery_engine_ledger_and_benchmarks_audit.md`](docs/apedp_discovery_engine_ledger_and_benchmarks_audit.md) within the project repository workspace.
- **Wilson CI Boundary Handling:** Fixed numerical precision at boundaries ($k=0$ and $k=n$) in `wilson_ci` ensuring exact $[0.0, 1.0]$ bounds.

---

## [1.2.0] - 2026-10-11

### Added
- **Native Lean 4 Verification Pipeline (`lean_universal_bridge.py`):** Zero-disk in-memory verification using Lean 4 compiler's native `--stdin` streaming, reducing theorem verification time from hundreds of milliseconds of disk I/O to sub-50ms kernel checks.
- **Deflated Sharpe Ratio (DSR):** Closed-form implementation of Bailey & López de Prado (2014) accounting for selection bias, number of trials, non-normality, and extreme-value asymptotic expected max Sharpe ratio.
- **Decoupled Novelty Adjudication (`novelty_adjudication.py`):**
  - Predictive incrementality evaluated via Diebold-Mariano tests with HAC standard errors ($p < 0.01$).
  - Structural novelty evaluated against canonical literature registry (Kyle 1985, Amihud 2002, Biais-Large-Bouchaud resiliency, Carhart 1997 momentum) with $R^2 < 0.85$ novelty ceiling.
- **Stock-Split Scale Invariance Gate:** Replaced unphysical price-scaling checks with correct joint price/volume conservation: $P' = P/k$, $V' = kV$, testing $S(P', V') == S(P, V)$.
- **Competing Multi-Proposer Implementations:** Added `SparseMicrostructureProposer` (standardized ElasticNet) and `OrderBookResiliencyProposer` (exponential decay recovery).

---

## [1.1.0] - 2026-10-09

### Fixed
- **Rank Ablation Live Path:** Re-implemented missing rank ablation calculation in `canonical_ledger.py` verifying incremental signal contribution.
- **Hardcoded Sharpe/DD Display Literals:** Eliminated all hardcoded display literals; all metrics computed dynamically from per-observation ledger rows.
- **Rust GA Index-0 Bug:** Fixed genetic algorithm best-individual selection in Rust kernel (`src/genetic.rs`) to correctly select minimum-MSE individual rather than index-0 individual.
- **Decorative Gate Suppression:** Added `GATE_NON_BINDING` handling in `canonical_ledger.py` ensuring non-binding thresholds suppress Sharpe ratios and abstain.
- **Mask-First Imputation:** Enforced validity masks prior to any nan-to-num or imputation operations.
## 2026-11-03 — Three "leapfrog" stubs are retracted and removed from the package

- **The three candidate modules never implemented their benchmarks, so they are gone.**
  `dqi_schur_forney.py` allocated 888 qubits' worth of registers and emitted 35 `cswap`/`cx`
  pairs: no GF(2^12) arithmetic, so it decodes nothing, and its zero-`Z_POW` compliance is
  vacuous. `dfthcsos_riemannian_15batch.py` never read an integral (no FCIDUMP ships in this
  checkout) and "passed" the 20 mHa gate with `error_1`/`error_2` dict literals beside a
  `lambda_eff` clamped by `min()`. `build_digit_serial_58q` built at 47 qubits with 473 gates
  and 200 toffoli-like ops — no coefficient table, no derivative chain — so it cannot compute
  `1/sqrt(x)`, and no `.kmb` was ever compiled or scored.
- **Every quoted figure was an input, not a result.** `Q <= 888`, `T <= 420,000`,
  `T <= 14,800` / `Q <= 420`, `lambda_eff <= 46.8`, `error_1 = 3.19 mHa`,
  `error_2 = 18.2 mHa` and `C_1 = 89,505.6` are constants in the modules; `58 x 1543.2` uses
  1,235 asserted Toffolis against a 47-qubit, 200-Toffoli circuit. The 3,133,452-input
  certificate belongs to the champion integer model, which this change does not touch
  (`src/signed_digit_invsqrt_v3.py` is byte-identical to its committed state).
- **`dqi_builder` no longer advertises a phantom architecture.** The `_ArchitectureTuple`
  override — `__contains__` answering `True` for `single_array_schur` while `__eq__` still
  reported `("proos_zalka", "gluzdov_v6")`, so one test asserted both and both passed — is
  deleted; `AVAILABLE_ARCHITECTURES` is a plain tuple of the two architectures with code
  behind them, and `tests/test_dyn_dqi_builder_architectures.py` pins that.
- **Where the code went.** All three stubs, their writer and their three test files are in
  untracked `build/apedp_stubs/`; `docs/DOCUMENTATION.md` carries one retraction row in place
  of the two rows that quoted the figures as achievements.

## 2026-11-03 — Degree-register recycling is closed on correctness and on score, and the DQI projection moves to 1,893

- **The recycled `n_a` build reaches 1,885 and does not decode.** Removing the register costs
  `+648,784` Toffolis and peaks exactly where the accounting said (`dqi_recycled_degrees_window.json`),
  but 64-shot verification gives **52/64** value errors for the one-body substitution and **64/64**
  for the all-sites build against the reference's 0/64.

- **The cause is the premise, not the comparison.** The derived predicate is exact over all 5,184
  `(A, b)` basis pairs, and the failure bisects to the first `_swap_a_and_b`: the a-pair sum is only
  `n + 1` until the swap exchanges it with a b-pair whose sum varies, so `n_a` is derivable only
  pre-swap — which needs the register the build removed.

- **It is also score-negative.** 1.1133× the Toffolis against a 0.9916× `Q²` gain is **0.898×** the
  projection, so even a working build would lose.

- **Nothing is scored at 1,885 any more.** The operative projection is the verifier peak:
  `R · 1,893² · 445,229 = 1.9006e12` (**12.859×**), with 1,885 carried as withdrawn and 1,789 as
  the unsourced aspirational row. Campaign docs (`EEA_SCHEDULE_CAMPAIGN.md`,
  `BENCHMARK2_TARGET_REACHABILITY.md` §7.8, `DQI_OFFICIAL_COMPATIBILITY.md`) record the closure.

## 2026-11-02 — The FCIDUMP is downloadable, and every precision we tried fails it

- **`FeMoCo76o113e.fcidump` is in the competition data, not in this repository's bundle copy.**
  Downloaded at 128,128,348 bytes it makes the chemistry half scoreable: the reference submission
  passes at `error_1` 3.19 mHa and `error_2` 18.64 mHa, `lambda_eff` 49.301, cost 3.356e9.

- **Every angle family in this repository is rejected by it.** Rounded from the sample's 12-bit
  words, `brot = 10` gives `error_1` 242.3 mHa, `brot = 9` gives 338.3 and `brot = 8` gives
  568.2, against a 20 mHa tolerance. Their Toffoli counts (63,584 / 55,636 / 47,688) are real
  arithmetic on an invalid submission. Word recovery passed on all three because it checks the
  loaded table, not the Hamiltonian. The earlier "25% cheaper at 8 bits" entry is withdrawn.

- **The gap to the public best is the table, twice over.** Its `(9, 143, 40)` submission carries
  1,363 addresses to the sample's 3,976, and its words are refit at 8 bits rather than rounded
  into 8 bits -- which is why it reaches 18.6 mHa where rounding does not.

## 2026-11-02 — The in-cap matrix covers all four layouts, and `J = 4` does not rescue the structured one

- **The split table in the decision note is now the four-layout grid it replaced.**
  `--low-matrix` walks eight densities on all four payload layouts, so the random-only table
  §2 carried is gone. At `J = 10`: random 1,406 -> 3,691 Toffolis picking `low = 2` only up to
  0.50, per-address 386 -> 3,770, blocked 89 -> 122 and contiguous 15 -- the last three at
  `low = 0`, i.e. the tree, at every density.

- **No structured layout picks the ladder at `J = 4` either, where the wide split fits.**
  With `low = 5` in cap the structured trees stay at `low = 0` (per-address 355–3,453, blocked
  103–131, contiguous 14–15) while the random window widens from 0.50 to 0.75 bits per address
  (0.41–0.89x). The cause is measured: those trees are 14–131 Toffolis, so the ladder's
  183-Toffoli addressing alone exceeds the read it would replace. The cap was not the only
  reason the structured regimes lose.

- **The flat comparator table is dominated on both axes.** At `low = 12` (`S > rows`, one
  comparator per address) the read costs 6,980 and 7,548 Toffolis at 7,094 and 7,662 qubits,
  against 3,691 and 3,770 at 123 -- 58x the qubits for no fewer Toffolis.

## 2026-11-02 — The ladder's crossing is bracketed at `[0.91, 0.95]`, and the builder now agrees with its rule

- **The builder and the cost rule disagreed on block counts, and the builder was wrong.**
  The ladder crossed `H = 2**(n - low)` blocks when the table has `ceil(rows/S)` -- 128 at
  `low=5` where 3,976 rows give 125 -- so every full-table ladder count was three Toffolis
  high. The builder now trims, and the rule matches the built circuit on an uneven table
  (`rows=100`, `low=2` -> 25 blocks, 27 addressing Toffolis) as well as on powers of two.

- **Placement, not density, decides the tree.** At equal payload (1,988 set bits) the tree
  costs 3,160 Toffolis with the payload scattered, 122 in eight contiguous runs, and 15 in
  one run, while the ladder is 2,171 in all three -- its `ccx` count is the payload count,
  wherever it sits. Structured emptiness is therefore strictly bad for the ladder, and the
  random-placement crossing is its best case.

- **The ladder now collapses like the tree, and that moves the crossing 0.87 -> 0.91.**
  `build_ladder(early_break=True)` applies the tree's rule at both levels -- a uniform offset
  range loads once against its own ladder control, a uniform block range needs one load and no
  descendants, and an all-uniform table skips the shared ladder entirely. At one bit per address
  it still loses by 8% (4,032 against 3,744).

- **The structured-layout comparison was wrong, and the sweep's cache was the cause.** It cached
  one ladder per density, which is sound only while the ladder is layout-invariant -- and it is
  not once it collapses, so blocked and contiguous layouts were reporting the *random* table's
  cost (4,032 for both). Keyed by placement, the like-for-like numbers are 296 against 131 (eight
  runs) and 74 against 14 (one run). Closer, still losing, and no crossing anywhere.

- **The window closes at `J = 10`, the family that ships.** 100 payload qubits leave only
  `low = 2` in the cap, so addressing costs 996 against 183 and the crossing falls from 0.91 to
  **0.56**; the family's bracket is `[0.56, 0.75]`. The 8% per-address advantage does not survive
  either -- it is 1.20x at every density (4,765 against 3,974 at one bit per address), because
  the payload is one bit wherever you look while the addressing is 5.4x larger.

- **The offset ladder is now built only where a load addresses it.** It used to be built in full
  (`2(S - 1)` gates, `2(S - 1)` ancillas) whenever a single block was mixed, which was a fixed tax
  the tree never pays and it made structured tables look hopeless. A uniform range still gets a
  control -- a load lands on it -- but its subtree is not descended into. Contiguous payload:
  74 -> 20 Toffolis (1.43x the tree rather than 5.3x); eight runs: 280 (2.14x). Still losing, and
  the remaining gap is structural: the tree skips a whole address range at one node.

- **`low = 2` was not the split to choose, so the 0.56 crossing was not the family's answer.**
  `--low-scan` costs every split with its qubit count: the cap admits `low <= 0, 1, 2` at `J = 10`
  (134 qubits from `low = 3`), and the best of those is `low = 0` -- the tree itself, 3,770
  Toffolis at one bit per address against `low = 2`'s 4,765. No in-cap split beats the tree on the
  shipped family; the ladder's advantage starts at `low = 3` and is comfortable at `low = 5`.

- **The shipped tree's early-break now compares whole rows, and it saves nothing on this table.**
  `build_delta_circuit` matched cells against the scalar `delta[lo, 0]`, so it only collapsed
  blocks that were one repeated 0/1 pattern; it now compares each row to `delta[lo]`, a strict
  generalisation. `--break-audit` counts what that finds: **0, 0, 0, 0, 0, 0, 0, 0** collapsible
  ranges across the 8 real delta tables, family Toffolis unchanged at 3,974 x 8. A synthetic
  table with one bit on every address finds 198 ranges and drops to 3,770 -- the same 204 -- so
  those 204 Toffolis belong to a table that repeats rows, not to this one.

- **The `J = 10` grid firms up the in-cap answer.** `--low-matrix` over all eight densities:
  `low = 0` is the best in-cap split at 0.75 and above for random tables and at every density
  for a fully populated one; the sparse random tables at 0.10-0.50 do prefer `low = 2`
  (1,105 / 1,921 / 2,952 against trees of 1,406 / 2,325 / 3,080). The shipped table is fully
  populated at 40 bits per address, so its in-cap answer is the tree.

- **Decision note.** `docs/BENCHMARK3_LADDER_DECISION.md` states the four regimes, the cap
  argument, the break port and the reproduction commands in one place, so the question does not
  get reopened without new data.

- **The 8% populated-table win is withdrawn.** Porting the row-equality break to the tree made the
  tree cheaper on every table that repeats rows, and the ladder no longer wins anywhere the table
  is fully populated: 3,638 against 3,453 at `J = 4, low = 5` (1.05x, was 0.92x) and 4,765 against
  3,770 at `J = 10, low = 2` (1.26x). The measured crossing moves from 0.91 to **0.87** with the
  ladder's break and **0.84** without it. The ladder now has exactly one winning regime -- payload
  sparse enough to leave empty addresses -- and the real table is nowhere near it at 40.0 set bits
  per address.

- **The shipped angle table has no uniform block at 16, 62, 124 or 256 addresses**, so
  early-break never fires on it and its tree stays at `rows - 2 = 3,974`. That puts the real
  crossing between the measured 0.91 (maximum collapse) and the fixed-payload 0.953 (no
  collapse), which is the honest form of the claim.

- **The collapse is driven by empty addresses, not by repeated words.** Scattering each set bit
  to its own bit position leaves tree and ladder costs unchanged. Filling one bit into *every*
  address is the ladder's only winning regime (8% cheaper, tree at `rows - 2`), and 40 bits on
  every address -- the shipped table -- is 40x past it, so the two conditions cannot be met at
  once.

- **The real density is 40.0 set bits per address at `J=10`, and the FCIDUMP cannot move it.**
  The delta table is the angle table, set by the rotation words, not by the integrals; an
  FCIDUMP enters scoring through `lambda_eff`. Density is `~J * brot / 2` (measured: 16.08 at
  `J=4`, 40.01 at `J=10`), so reaching the crossing would need 91.3% of the 39,760 cells zero
  where the table has 0.4% zero words at `brot=10`. No precision reaches it: the only sparse
  row is `brot=1`, where every column is constant and every delta is zero.

## 2026-11-02 — The `O(sqrt(A))` QROM ladder is built: addressing 21.7x cheaper, total 40.3x worse

- **The ladder the cost model describes is now a circuit, not an extrapolation.**
  `scripts/prototype_dfthcsos_dirty_ladder.py` builds the shared-comparator read: `sqrt(A)`
  offset controls computed once and kept alive, a tree over the block index, and each
  block's rows loaded against controls that are already computed. Every variant passes the
  competition's own `recover_words_from_circuits`, so nothing is reported for a read that
  does not load the table.

- **The result splits in two.** Addressing falls from 3,974 Toffolis to **183** at `low=5`
  (21.7x), so the model is right about the addressing. But a load now needs two controls, so
  the payload bits become Toffolis instead of `cx`s, and the tree wins because it amortises
  one Toffoli over every set payload bit while the ladder has none to amortise. The payload
  is per circuit, so `J` sets the damage: **40.3x** the tree on the shipped `J=10` family
  (159,068 set bits, 160,064 total, best in-cap split `low=2`), and 16.1x at `J=4`. The
  16.1x figure this entry first carried was the `J=4` configuration, not the shipped family.

- **The break-even was wrong in a way that mattered.** Pinning it exposed two errors in the
  cost rule, both now fixed and pinned: the shipped tree costs `A - 2` Toffolis, not `A - 1`
  (its root split is a CX pair), and a payload load is Toffoli-like only when it has two
  controls, so `low=0` costs exactly the tree at any payload. Corrected, the fixed-payload
  break-even is 3,791 set bits (0.953 per address at `J=4`, 0.749 at `J=10`). Measured on
  real circuits across densities (`--sweep`), the actual crossing is **0.87 set bits per
  address**, because the tree's early-break also collapses the all-zero subtrees that make a
  sparse payload cheap (3,974 at full fan-out, 3,744 at one bit per address, 1,405 at 0.10).
  The ladder chases a target that retreats. **One set bit per address, the sparsest
  non-empty table there is, still loses at 1.11x**, so the earlier claim that such a table
  would make the ladder cheaper is withdrawn; it needs at least a seventh of the addresses
  to carry no set bit at all. Every density recovers its words, so this is cost, not
  correctness.

- **Two correctness bugs were found by the checker, not by inspection.** A shared ladder
  needs `2(sqrt(A) - 1)` ancillas, not `sqrt(A) - 1`, because every leaf must be alive while
  the loads run; and `del_and` assumes its ancilla holds `control AND bit`, so the left child
  (built as `control AND NOT bit`) has to be flipped into that polarity before uncompute.
  Without the second fix every split failed with `uncorrected phase (measurement kickback)`,
  which is exactly the class of error a size pin cannot see.

- **The accounting is now pinned by tests, at both ends.**
  `tests/test_dyn_dfthcsos_sub128_family.py` pins the addressing counts (`{0: 3974, 2: 996,
  4: 275, 5: 183, 6: 185}`) and the break-even arithmetic, and a slow test rebuilds the
  density sweep at 0.10, 0.75 and 1.00 set bits per address, costs both structures, and
  requires each to recover its words. A future builder change that moves a count, or that
  quietly makes one-set-bit-per-address look cheap, fails there.

## 2026-11-02 — The blind referee is re-minted as `APEDPX_FROZEN_v1_12.json`

- **The re-mint is verified against the old referee, not asserted.** The 10-case blind L2
  challenge was re-run under `v1_12` and compared field by field with the committed record
  the `v1_11` referee produced: **0 differences across 40 comparisons** (`actual_status`,
  `epistemic_attribution`, `promoted_equation`, `passed_honest_contract`), and the run
  rewrote `benchmarks/l2_blind_challenge_results.json` byte-identically. `apedp_dyn.ui` is
  never imported by the blind path, which is what makes the server split irrelevant to the
  outcomes rather than merely unlikely to matter. Record check: exactly one of the 15 guard
  hashes moved, thresholds and blind cases are identical to `v1_11`.

- **Guard-only re-mint after the dashboard server split.** `apedp_dyn/ui/server.py` moved
  from 2,433 to 1,950 lines by extracting `apedp_dyn/ui/helpers.py`, which is why
  `tests/test_dyn_blind.py::test_guard_passes_on_frozen_tree` started failing with
  `REFEREE OR FROZEN RECORD CHANGED since APEDP-X v1.11-FROZEN`. Exactly one of the 15
  guard files had moved (`7e84dbb9…` -> `0e1ac14c…`); the 14 referee and trap-suite files
  keep their v1.11 hashes, which is the check that this mint blesses a server refactor and
  nothing else. Thresholds, blind cases and the `env` block are carried through unchanged,
  because no re-evaluation was performed and the local interpreter set differs from the
  sealed one — recording it as if it were the sealed environment would be a false claim.
  The `remint_log` entry records the old and new digest and the reason.

- **`frozen_spec.py` needs no change.** It pins 3 referee files plus one benchmark JSON,
  none of which is the server, so `verify_all_frozen_records_or_raise` was green throughout
  and still is.

## 2026-11-02 — Blanket exception handlers and module size are now gated, and `_core.py` is split

## 2026-11-02 — The DQI circuit is restored, and a missing artifact now says so

- **`sample_submission/dqi.kmb` is back, byte-for-byte.** The reference decoder that
  passed the official verifier at 102,400 shots (1,893 qubits, 5,725,461 Toffoli-like,
  score 2.4439e13) had been deleted from `artifacts/`, leaving only the JSON receipt.
  Two surviving copies in `/tmp` hash to `959471d05a26373b…04995`, the value the receipt
  records, and one was copied into place with the hash re-verified after the copy. The
  target path was already in `.gitignore`, so the 886 MB file stays out of history. The
  contract digest moves from `0c0acca…` to `726a531c…` because `sample_artifact` is part
  of it; no test pinned the old value.

- **A proposer with no artifact now raises instead of returning `[]`.**
  `DqiProposer.generate` raises `MissingSampleArtifactError` when neither `artifact_paths`
  nor the contract supplies one. That empty list is what made
  `test_dqi_and_dfthcsos_proposers_screen_and_measure` fail with a bare `0 == 1` and no
  hint that a file was missing. `budget=0` still returns `[]`, because an exhausted budget
  and an absent input are different questions.

- **The 886 MB circuits have a documented home.** `config/quantum_artifact_manifest.json`
  records sha256, size and role for each verified circuit, including the one row whose
  bytes are gone (its hash comes from the sweep receipt, so it stays recoverable).
  `scripts/verify_quantum_artifacts.py` checks what is present, re-records it, or
  materializes it from a content-addressed cache. Absence passes; corruption fails.


## 2026-11-02 — Zero silent blanket handlers, `model_synthesis.py` split, and two stale verdict expectations

- **The blanket-handler ratchet reads zero.** 22 -> 0. Most of the remainder were
  legitimate best-effort paths that had simply never recorded a rationale, and each now
  carries a site-specific one instead of a generic string. Eight were real defects:
  four in `apedp_dyn/quantum_compilation/faros_isq.py` alone, where a failed resource
  read fell back to `90 qubits / 1,448 Toffolis / 1 t-like / 44 rotations` -- the
  measured values of a *previous* champion -- so a broken statistics call produced a
  cost profile that looked like a measurement of this circuit. It now raises, because
  those four numbers decide the score. `apedp/stability_selection.py` returned `{}` when
  every replicate failed, which reads exactly like "no feature met the threshold"; it
  now raises. The `kickmix` guards in `dqi_gluzdov_v6.py` and
  `dfthcsos_signed_4bank.py` caught `(ImportError, Exception)` and reported a broken
  wheel as "not installed"; the reason is now kept in `km_import_error`, and a failed
  `GF2Field` construction records `field_error`.

- **The audit itself was undercounting.** A nested `raise` inside a guard
  (`if strict: raise ... from e`) did not count as re-raising, so handlers that do
  propagate their failure on the strict path were classified as silent.

- **`apedp/model_synthesis.py` drops from 2,145 to 1,705 lines.** `SynthesizedModel`,
  `SynthesisDiagnostics`, the BIC re-rank, amplitude caps, adversarial classification,
  abstention wording, input normalisation, the noise floor and regime identifiability
  moved byte-for-byte to `apedp/model_synthesis_helpers.py`. The move silently dropped
  eleven module-level constants the class still needed -- ruff's F821 caught six, and
  the remaining five only surfaced as an ImportError in
  `tests/test_abstention_reason_codes.py`, because they were imported by test rather
  than used by the module. The full original surface is now re-exported and verified
  against a pre-split AST of the file.

- **The creep sandbox verdict is correct and its test is stale.**
  `tests/test_dyn_ui_sandbox.py::test_creep_abstains` expects `multi_hypothesis` and gets
  `unique_support`. The sandbox is not malfunctioning: the power law holdout RMSE is
  1.606 against a rival at 51.71, `delta_rmse = 50.10` against `tau = 3.01`, margin 47.09,
  and the paired block bootstrap confirms. The verdict follows its own rule, and the
  pinned expectation predates the relative-noise generator fix recorded in
  `creep_powerlaw.py`. The test is now `test_creep_verdict_agrees_with_its_margin`: it
  pins the agreement the contract promises -- the verdict matches the promotion, a
  promotion happens only when `delta_rmse > tau_epi`, and only `power_law` may be
  promoted -- rather than a verdict the corrected generator no longer produces.

- **`test_dqi_and_dfthcsos_proposers_screen_and_measure` was hiding a missing input.**
  `DqiProposer` wraps a pre-compiled `.kmb` artifact, and `load_contract` reports
  `sample_artifact=None` when that sample is not on disk, so `generate` returned `[]` and
  the test failed at `len(...) == 1` without saying why. An empty list is the correct
  answer to "no artifact", so it is now pinned as such, and the pipeline stage under test
  is given an explicit placeholder artifact: `screen`, `compile` and `measure_resources`
  need the file to exist and be non-empty, and the verifier is what reads the circuit
  inside it. The pre-existing `I001`/`F401`/`UP035` findings in `dqi_gluzdov_v6.py` are
  fixed in the same pass.


## 2026-11-02 — The parser allowlist gets a public name, two files are exempted properly, and `server.py` is split

- **`_SAFE_FUNCS` had no path from `apedp` to itself, and 10 tests were failing on it.**
  `apedp/epistemic.py` imported it from `apedp.safety`, which is a pure re-export façade
  whose `__all__` never listed it. The table lives in `apedp_dyn/safety.py`, and importing
  `apedp_dyn` directly from `apedp/epistemic.py` would have inverted the layering the
  façade exists to protect. `apedp_dyn/safety.py` now exposes `SAFE_FUNCTIONS` as the
  public alias (the same dict object, so a runtime extension is seen under both names)
  and `apedp.safety` re-exports it, so `from apedp.safety import SAFE_FUNCTIONS` is the
  supported path. `tests/test_prospective_and_composer.py` and
  `tests/test_discovery_benchmark.py` went from 10 failures to 55 passing.

- **`apedp/physics_constraints.py` did not need inline noqas, and adding them was wrong.**
  `pyproject.toml` already exempts that file and `apedp/special_functions.py` from
  `BLE001`, because every handler there binds the exception and reports it as an
  `_unverified` violation -- the catch-all is the design. Inline `# noqa: BLE001` on top
  produced `RUF100` "unused directive" on all seven. The ratchet now reads the linter's
  per-file exemptions and treats them as the annotated decision they are, which took the
  tree from 32 silent blanket handlers to 22 without weakening the classification.

- **`apedp_dyn/ui/server.py` drops from 2,433 to 1,950 lines.** The non-route machinery
  moved byte-for-byte to `apedp_dyn/ui/helpers.py`: the API-key gate, the audit writer,
  the JSON serialisers, and the four formatting helpers behind the sandbox and market
  endpoints. All 22 route decorators stayed in `server.py`, `server` imports the moved
  names back so callers that import or patch them still resolve, and `_AUDIT_LOG` still
  resolves to the same path because the new module sits in the same directory.


- **`apedp/discovery/_core.py` drops from 2,908 to 2,421 lines.** The Bayes block
  (`ModelEvidence`, `BayesianRefinementEngine`, the integration helpers and
  `STANDARD_RATIONALS`) moved verbatim to `apedp/discovery/_bayes.py`. Nothing was
  rewritten: `_core` re-exports the moved names and `apedp/discovery/__init__.py`
  star-imports `_core`, so every one of the ten modules that do
  `from apedp.discovery import BayesianRefinementEngine` is unchanged, and the
  re-exported objects are the same objects, not copies. The one non-mechanical part is
  `STANDARD_RATIONALS`: the moved code closes over it, so it had to travel with the
  block rather than be imported back (which would be circular).

- **`scripts/audit_exception_hygiene.py` turns the blanket-`except` complaint into a
  ratchet.** It walks the AST of every first-party module and classifies each
  `except Exception` / bare `except` as annotated (it re-raises, or the line carries
  `# noqa: BLE001`) or silent. The per-module silent counts and the 25 largest module
  sizes are pinned in `benchmarks/exception_hygiene_baseline.json`; `--check` fails when
  either gets worse and says nothing when they improve. A `noqa` naming any other rule
  is deliberately not accepted as a rationale.

- **CI runs the ratchet.** `.github/workflows/test.yml` gained the `--check` step and
  the split's surface test, both in the existing quality job. The tree already carried
  the right convention in the epistemic layer (`# noqa: BLE001 - deliberate catch-all: …`),
  so the baseline records where that discipline has not yet reached rather than starting
  from zero.

- **The gate is tested, not just run.** `tests/test_dyn_exception_hygiene.py` feeds
  synthetic sources with known labels through the classifier, so a version that answers
  "annotated" unconditionally cannot pass while policing nothing, and it asserts the
  check actually fires on a fresh silent handler and on a size overrun.

## 2026-11-02 — The 89Q champion's qubits are spent on the adder carry chain, and the zero-angle rotations are already gone

- **The qubit peak is in `folded_gadd`, not in the Boolean restoration.** The score is
  `Q x toffoli_equivalent`, so at Q = 89 one qubit is worth 19.9 Toffolis (1,770.8 units,
  1.12%), more than any single rotation (0.40%) and far more than one Toffoli (0.0565%).
  `scripts/diagnose_invsqrt_qubit_peak.py` wraps the builder's allocation calls and
  reports every rise in the live count with its call site. The peak is the temporary
  carry chain of the 21-bit adder in `outer()` (`scheduled_arithmetic90.py:252`), so the
  Boolean plan does not set Q: this is why the 26-node mixed-polarity curves plan kept
  Q = 89 and only matched the Toffoli count. Shrinking the restoration further cannot
  lower the score.

- **Three calibration entries read zero, and dropping them breaks the circuit.**
  `src/invsqrt_calibration.json` holds 48 deltas, of which indices 15-17 are exactly 0 and
  18 is already in `droppedOriginalRotationIndices`. `scripts/try_drop_zero_rotations.py`
  rebuilt the champion and applied `{15,16,17,18}`: the official verifier fails at x = 1
  with a phase error of 0.125 turns against a tolerance of 5.96e-8. The reason is
  arithmetic, not luck: the compiler leaves 45 `z_pow` ops and the shipped rewrite already
  removes the three zero entries plus index 18, taking 48 to 44. Naming 15-17 again shifts
  the index mapping and deletes three real rotations. The shipped 44 is therefore the
  floor, and the four-rotation difference is accounted for rather than unexplained.

- **Champion unchanged.** Rebuilt hash `ce04c9d5...`, Q = 89, 1,466 static Toffolis, score
  157,605.52 at 25,600 shots, `tests/test_dyn_invsqrt_champion_rebuild.py` 3 passed.

## 2026-11-02 — Fig. 12(b) is digitised, and both photonics docs are rewritten in plain prose

- **Fig. 12(b) is now extracted by the same script, so Eq. (11)'s per-N curves are
  checked against the implementation rather than eyeballed.** The panel plots the cycle
  against loss for N = 2..12, and Eq. (11) makes each curve's slope proportional to
  `2n_i^max + 1`, so the ratios have to be 5, 4.2, 3.4, 2.6, 1.8 and 1. Measured:
  5.006, 4.195, 3.396, 2.604, 1.805 and 1.000, a worst deviation of 0.26%. Two columns
  of band centres give the slopes without any y calibration, since the L factor is
  common to every curve and cancels in the ratio. `scripts/digitise_fig12_cycle.py`
  reports drift against the pinned values; a band-counting bug that seeded both the run
  and the output list reported 7 curves per column and is fixed.

- **`docs/SPARROW_QUANTUM_DETERMINISTIC_PHOTONICS.md` and
  `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` were rewritten to remove em dashes, bold
  label headers, title-case headings and the rest of the AI tells.** No number, formula,
  table row, section number or measured claim changed; §5.2, §5.4, §5.5 and §6 keep
  their numbers because scripts and tests cite them.

## 2026-11-02 — The photonic blueprint's numbers are re-read against its published tables

- **sFFCC is the *synchronous foliated Floquet colour code*, and the spec doc now says
  so.** Two documents called it “semi-fluid” and the circulating summary called it a
  “staggered face-centred cubic” lattice. It is a 2D colour (CSS) code on a
  three-colourable hexagonal lattice, foliated into a 3D cluster state and measured
  colour by colour on a Floquet schedule. `docs/SPARROW_QUANTUM_DETERMINISTIC_PHOTONICS.md` §4
  and the `docs/DOCUMENTATION.md` module row are corrected.

- **The Table II thresholds are now labelled as targets, not achievements.** Every
  channel in §5 is an inequality the hardware must reach; the “Operating Target”
  column is the referee's default spec, not a device. The published caption to Table I
  (“these values currently do not satisfy all timing inequalities, owing to equipment
  constraints; they are provided solely as indicative timescales”) is quoted, because
  it is the paper's own statement of the gap.

- **The 2.65 µs / 377 kHz clock is revealed as a floor, and the doc's speed claim is
  corrected.** The paper's own Table IV reports 209τ_echo ≈ 6 µs at L=3 (excluding
  classical decoding); the repository's 12Lτ_rep is one RUS attempt per layer and is
  ~12× optimistic at L=3. The doc's “~1,000× faster than trapped-ion or neutral-atom”
  becomes the Table IV comparison, which also shows the photonic cycle is *slower*
  than superconducting. The DQI section says so.

- **`SFFCCLatticeArchitect` gains the `3L^2` fibre EOMs Table III requires,** omitted
  from the bill until now, and `SFFCCFeasibilityGate` prices what the resource table
  leaves out: Eq. (5)'s feedforward budget (3.94 ns at τ_rep = 13.8 ns against 34.3 ns
  at τ_echo = 29 ns — the same 23 ns reference loop passes or fails depending on the
  reading), Eq. (8)'s phase-shifter switching and detector deadtime, and the decoder
  throughput implied by 18L³ fusions per cycle (~2.78 × 10¹⁰ s⁻¹ at L=16). An
  unassessed decoder is reported as unassessed, never as a pass.

- **PsiQuantum's photon source is spontaneous four-wave mixing, not SPDC in silicon.**
  Silicon is centrosymmetric and has no χ⁽²⁾, so §2.1 of the photonics doc now assigns
  SPDC to the bulk crystals (KTP, BBO, PPKTP) and SFWM to the silicon platforms. The
  probabilistic-source argument is unchanged; the paragraph also records that
  PsiQuantum's fusion architecture absorbs a low per-source rate by running many
  sources rather than a naive multiplexing tree.

- **The logical cycle time was digitised off Fig. 12(a), and the caption-only
  interpolation was wrong.** The caption gives 209 τ_echo at L=3 and calls the scaling
  linear; a 500 dpi render, nine y-tick anchors and ten detected curves give ≈742 at
  L=8, ≈1613 at L=16 and ≈2246 at L=22 — affine at **107.5 τ_echo per unit L**, not
  proportional, fitted to 4.5 τ_echo rms. `paper_table_iv_cycle_seconds` now uses that
  line, moving the L=16 cycle from 32.32 µs to **46.59 µs**, the DQI wall clock from
  185.1 s to **266.7 s** (~46.6× the stated 1 µs superconducting point), and the
  decoder requirement from 4.32e12 to **3.00e12** fusion outcomes/s. The sensitivity
  table, the three docs and the non-submission artifact follow; the report now prints
  a decoder/s column, `wall_clock_sensitivity()` exposes the basis × τ_echo × L grid,
  and `SFFCCFeasibilityGate.decoder_requirements()` states the rate per code distance
  on both clock bases. The extraction itself lives in `scripts/digitise_fig12_cycle.py`
  (render, nine tick anchors, ten curve centres, affine fit) and its ten points are
  pinned in `tests/test_dyn_sparrow_deterministic_photonics.py`, so the model has to
  reproduce the published figure rather than a remembered slope.

- **Fig. 12(b) is digitised too, and it confirms the `2n+1` factor.** The panel's six
  curves (N = 2..12) have slopes whose ratios to the N = 2 slope are 4.991, 4.190,
  3.389, 2.602, 1.796 and 1.000, against (2N+1)/5 = 5.000, 4.200, 3.400, 2.600, 1.800
  and 1.000. The check is calibration-free because every curve shares the same L factor,
  so only the ratio is measurable without a y-axis conversion. `digitise_panel_b()`
  re-runs it and the ratios are pinned in the same test module.


- **`scripts/compare_integer_relation_methods.py` now records three seeded random
  families beside the eight designed cases** (`random.Random(20261102)`, 40 cases
  each): a dyadic relation the box can hold, the same construction with coefficients
  of 20–400 that it cannot, and continuous draws at `eps = 1e-300` where no relation
  exists at input precision. Over 128 cases: branch-and-bound **84 correct, 44 refused
  out of scope, 0 wrong**; LLL/PSLQ **128 correct, 0 wrong**. All 80 planted relations
  are recovered up to sign (42 identical, 38 negated by the first-non-zero-positive
  canonical rule, all 38 checked to be exact negations). The seed *and* the generator
  parameters are recorded, because a seed without them reproduces nothing, and the
  design block states that these rates are a distribution over those three families
  and not over discovery work. `tests/test_dyn_integer_relation_methods.py` recomputes
  every rate from the rows and pins the behaviour each family's construction fixes.

- **The discovery harvest can now say where its minutes go.**
  `scripts/harvest_integer_relations_from_discoveries.py --profile` times both
  searchers per group and prints the totals by input count, by source file, and the
  slowest groups with their JSON pointers. The query path was extracted into `_query`
  so the profiled run and the recording run execute the same code. Profiled result:
  of 429.5 s, **59.2% is the 25 groups with six inputs and 33.8% the 14 groups with
  fifteen**; the 134 two-input groups cost 0.04 s together. 99% of the run is the
  Feynman artifact. The only branch-and-bound-dominated questions are `trials[66]` and
  `trials[67]` at 11.2 s each — a 25^6 box — while every 15-input group spends its
  12–22 s in the reduction and 0.01 ms in the box, which refuses `n > 8`. Recorded in
  `docs/INTEGER_RELATION_METHODS.md`.

## 2026-11-02 — The LLL pass keeps its decomposition in place, and the first attempt is recorded as slower

- **The integer-relation reduction is now driven by an `f64` Gram-Schmidt pass with
exact certification instead of an exact `BigRational` pass per step.**
  `lll_reduce_fast` builds the decomposition once, updates it with the standard
  size-reduction identities, rebuilds it only after a swap, and then asks
  `exact_violation` whether the result really satisfies the LLL conditions in
  rationals, repairing at most a bounded number of times before falling back to the
  exact-only `lll_reduce`. Measured with `cargo test --release -p apedp-kernel --lib
  rime::pslq -- --nocapture` on an eight-row, twenty-digit basis with no short
  relation: **454.5 ms → 265.5 ms (1.7×)**, both bases certified by the same exact
  check, 8/8 kernel tests and 7/7 Python tests green. The recorded timings in
  `benchmarks/integer_relation_methods.json` are unchanged in outcome.

- **The first version of this fast path was 2× slower, and that is now in the doc.**
  It rebuilt the `f64` decomposition from `BigInt` twice per iteration and restarted
  the sweep after every repair: 79.9 s against the exact path's 42.1 s on a
  twelve-row, twenty-seven-digit basis (0.5×). The lesson is recorded in
  `docs/INTEGER_RELATION_METHODS.md` because "float Gram-Schmidt" is not the
  speedup — keeping the decomposition in place is. The timing test prints its
  measurements instead of asserting a wall-clock threshold, and asserts the thing
  that must not regress: that the `f64`-driven basis passes the exact LLL
  conditions.

- **The result statement no longer claims exact-rational Gram-Schmidt.** Every
  `IntegerRelation`/`PslqResult` statement said
  `reduction: LLL on [ I | round(x * 10^digits) ], exact-rational Gram-Schmidt`,
  which would have been false for the new driver. It now names the `f64` pass, the
  periodic exact re-orthogonalisation and the exact certification.

## 2026-11-02 — Dead trees deleted, and a stale pin corrected against the record

- **`crates/apedp_kernel/` and the six `packages/apedp-*` shells are deleted.** The
  ten abandoned Rust modules were never a workspace member and were never built;
  the six shells had no implementation of their own, only a `sys.path` insertion
  and a wildcard re-export from `apedp_dyn`, so each wheel was a second name for
  one implementation — and its `production` row in `docs/DOCUMENTATION.md`
  claimed a shipped artifact for code the core already shipped. Nothing imported
  the shells (they were installed in no environment here) and no active manifest
  named the abandoned crate, so no import, reader or provenance record changed.
  Removed with them: their dependabot entries, the `/crates/` CODEOWNERS rule (and
  the pin that expected it), and five matrix legs of the wheels workflow. The
  abandoned tree's rejection test is now a deletion test, so either tree returning
  fails before review. `packages/` holds `jetflow-video` only.

- **The H-A coupling rank pin was stale, not the machinery.**
  `tests/test_dyn_ha_coupling.py` pinned `H-RV,H-RV,H-A,H-P×8` with H-A rank-1
  once — the sequence recorded before FLAW-A01 (2026-10-21) removed the
  forward-leaking train blocks. `_d5` shifts the target by five days, so a block
  within five days of the origin boundary reads out-of-sample rows, and the leak
  was carrying the H-P plurality. The corrected sequence is
  `H-RV,H-RV,H-A,H-A,H-A,H-RV×6` (H-A# 3/11), and it is not a re-snapshot of
  whatever the code now does: §77 of `docs/TRADING_EQUATION_FORENSICS.md` already
  recorded the after-the-fix column (H-A# 3/3/5 for BTC/ETH/SOL), and the new test
  reproduces those three numbers, failing if the pre-fix 1/1/1 column returns.
  §76 now carries the correction note and `3/11` instead of `1/11`.

- **Two red tests that were reporting the wrong thing are green.** The
  jetflow-video standalone contract tested `"import apedp" not in source`, which
  also matches the optional `import apedp_kernel` native probe and reported a
  dependency the package does not declare; it now matches the core import exactly
  and still forbids `apedp_dyn`. The CI-matrix pin split the README on a heading
  that no longer exists; it reads the section that is actually there.

- **The integer-relation search has a second method, and it is the one that scales.**
  `find_integer_relation` searches a box, refuses `n > 8` and returns no
  certificate; `find_integer_relation_lll` (new, `rime/pslq.rs`) reduces
  `[ I | round(x·10^digits) ]` with exact-rational Gram-Schmidt and verifies every
  candidate by exact substitution on the `f64` inputs. Three filters keep the
  reduction from manufacturing answers, and each exists for a measured failure:
  an always-present lattice vector of size `10^(digits/n)` (hence a 10× size
  margin), rounding of the constant column (hence a residual that must sit 4× above
  that floor unless it is exactly zero), and the float dot product (hence exact
  rationals). Measured on eight declared cases
  (`benchmarks/integer_relation_methods.json`): the box search is 4 correct, 4
  refusals outside its declared scope, 0 wrong; the reduction is 8 correct, 0
  wrong; they agree on 4, and the box search is two orders of magnitude faster at
  the median (around 0.1 ms against 10–20 ms across reruns) precisely where it can
  be asked. The reduction is the
  only one that answers `137:−89:−1` outside a `max_coeff` of 20, or `n = 10` and
  `n = 12`. Also recorded: allowed past the input precision, the same three
  "independent irrationals" yield an exact relation among their 52-bit rationals
  with coefficients around `10^8` — a fact about the rounded inputs, not about the
  constants, and both are kept in the record rather than only the second.
  [`docs/INTEGER_RELATION_METHODS.md`](docs/INTEGER_RELATION_METHODS.md) states the
  cost curve and the acceptance rules; the documentation row that called the box
  search "PSLQ/lattice null-space" now names what it runs.

## 2026-11-01 — The remaining placeholders refuse, measure or say what they are

- **The Lean layer no longer proves its invariants from themselves.** 94 theorems in
  63 of the 366 generated modules were proved by `by rw [<axiom>]`, where `<axiom>`
  was the invariant itself, declared a few lines above — so the conclusion was a
  premise and the kernel's exit 0 measured syntax, not truth. The generator now
  detects a proof whose rewrite names a core axiom and emits the statement with
  `sorry` instead, marking it `unproved_statement`; the module still typechecks, and
  `proof_claim_status` drops from P2 to P0 because nothing was derived. The 94
  on-disk theorems were rewritten the same way and every one of the 63 files was
  recompiled with the local Lean binary (exit 0, `sorry` warnings only). What is
  proved by a real tactic is untouched: `dx_dt = -gamma * x` still reaches P2 through
  `omega` and `zero_le`. The pins that asserted P2 for the circular path were part of
  the defect — one test was literally named `..._achieve_p2` — and now assert P0 plus
  the absence of a self-rewrite. `theorems proved by rewriting a local axiom` is 0 in
  `benchmarks/formal/lean_axiom_audit.json`, and the audit test fails if it returns.

- **Notebook lineage removed, and the decorative latent space with it.** Fifty-five
  source files in `apedp/`, `apedp_dyn/`, `packages/`, `scripts/`, `benchmarks/`
  and `tests/` opened with `# Databricks notebook source`, and fourteen of them
  also carried `# COMMAND ----------` separators — 69 marker lines in total. They
  were inert comments, but they were not provenance: the modules had been edited
  as ordinary Python for months, and the header told a reader the file was a
  notebook export. Nothing reads the marker (the notebook adapter detects the
  *runtime*, not the file), so the lines were removed rather than explained.
  `HierarchicalSearch.__init__` also built `self.latent = LatentSpace()` and never
  read it: the attribute was decoration, and since `LatentSpace.decode` refuses
  there was no path on which it could have returned anything. The class stays
  exported for callers that use its distance and interpolation helpers. The sweep
  that found both is recorded in [`docs/SCAFFOLD_INVENTORY.md`](docs/SCAFFOLD_INVENTORY.md).

- **The multiscale materialiser emits once, and declares only what it emits.**
  `_materialize_multiscale` carried three near-identical branches — fast,
  slow, and unlabelled-falls-back-to-slow — differing only in which symbol
  `tau` held. That duplication is what let the labels stop reaching the
  emission: the fast branch could be broken while the slow branch, exercising
  the same code shape, stayed correct and kept the mode looking alive. The
  three are now one emission that picks the constant by label, and a term
  labelled both fast and slow is refused rather than resolved silently.
  `tau_fast` and `tau_slow` are appended only if an emitted term used them, so
  a candidate that labels nothing fast no longer carries a fitted dimension no
  equation mentions. The pin in `tests/test_dyn_d3.py` changed with it: it
  asked "is `tau_fast` declared?", which both constants satisfied
  unconditionally — including while `tau_fast` appeared in no equation at all.
  It now asserts declared equals used, on the emitted right-hand sides. 104
  lines become 66.

- **The siblings of the coefficient-prefix defect: a power and a denominator.**
  The earlier sweep asked for `}*{`; `**` and `/` bind tighter than `*`, so the
  same mis-binding reaches further. `1/{feat_name}` on a column named `a + b`
  is `1/a + b` — the reciprocal never sees the second addend while the scorer
  divides the whole column — and `{feat_name}**2` is `a + b**2`, which raises
  only `b`. Five live sites were found and fixed: the RIME-AD multi-rational
  basis and its cooperative Hill form, `coordinate_transmuter`'s tunneling
  exponential and power law, `_core.recombine_power_law` (including its
  exponent-1 branch, where the terms are joined with `*`), and
  `feature_selection`'s hypothesis atom set. `term_format` gained
  `powered(operand, exponent)` and `over(numerator, denominator)` on the rule
  `scaled` already follows: leave an atomic operand untouched so no emitted
  string and no candidate id moves. The audit script grew the second scanner —
  the sweep now covers four shapes — a power, a quotient, a function
  argument and a left factor (`{operand}*2`) — and finds 330 sites: 51 adjudicated in scope, the other
  279 reported with the reason they are not arithmetic (counters, paths,
  markdown, derivative notation, formatted numbers). Widening it found a sixth
  live defect: `f"exp(-{variable})"` in `feature_selection`'s atom set, where
  the sign is not a delimiter — `exp(-a + b)` is not `exp(-(a + b))` — and it
  now goes through `atom()` — and
  `tests/test_term_builder_structure_audit.py` fails on an unrecorded in-scope
  site, on a stale entry, and on a registry verdict outside the declared
  vocabulary. The scope rule is a naming heuristic and the record says so.

- **The proposer tree now lints clean, and three blanket handlers became
explicit.** `ruff check apedp_dyn/proposers/` reported eleven findings in
  `sindy_proposer.py` and two more in the basis-regression and reservoir
  proposers; all thirteen are resolved rather than suppressed. In the SINDy
  proposer the grammar import moved above the kernel import probe (E402), the
  probe itself now catches `ImportError` only so a broken extension is visible
  instead of costing the Rust path silently, `zip(coeffs, terms)` is `strict`,
  the per-variable score closure binds its library names as defaults instead of
  reading a variable the enclosing loop rebinds (B023 — a score that outlived
  its iteration would index the next variable's library), and the adapter's
  inner handler narrowed to `AttributeError` because `CandidateProposal.formula`
  is a `str` and nothing else can raise there. The one deliberate blanket catch
  that remains — the PyO3 boundary, where a panic, a shape mismatch and a dtype
  refusal arrive as different classes and all mean "use NumPy" — carries a
  `noqa: BLE001` with that reason and logs the exception. A dead module-level
  `try: class ... except Exception: pass` around `SINDyGrammarAdapter` was
  removed: it guarded a class definition, which cannot fail, so it only made
  the adapter's disappearance look possible. The basis-regression and reservoir
  ridge solves now catch `LinAlgError`/`ValueError` and log the fallback to
  `lstsq`, which is the case `pinv` actually fails on. Writing the behavioural
  pin for that closure found a real defect underneath it: `score` paired a
  library column index with the coefficient list positionally, so a candidate
  whose single active term was not the first library column raised
  `IndexError`, and with two active terms it scored one term with the other's
  coefficient. The pairing is now by term name, and
  `test_each_sindy_candidate_scores_with_its_own_terms_and_coefficients` pins
  it from both directions (including a frame whose columns are swapped).

- **The coefficient-prefix fix is now tested through the proposers, not only
  through the formatter.** `tests/test_sum_named_column_binding.py` drives four
  proposers with a column literally named `a + b` — regime switching,
  distributed lag, sparse microstructure and the RIME-AD multivariate emitter —
  and asserts on the formula that comes out, because every name in use is
  atomic and the defect was latent: a formatter unit test proved only that
  `scaled` brackets an operand, never that a proposer reached it. Each probe
  fails if it produces no proposal, so an empty result cannot pass, and the
  shared helper is itself tested against the unparenthesised form `1.2345*a + b`
  so it cannot accept everything. The sparse probe additionally re-evaluates
  the printed formula and compares it to the scorer's output, which checks that
  the formula describes the arithmetic rather than merely looking bound.

- **A capability flag that was true over an empty function, and a proposer that
  turned every failure into a null result.** The sixth sweep, grepping for the
  shape of the earlier findings rather than for module names.
  `packages/jetflow-video/jetflow_video/stepper.py` guarded its "Rust fast path"
  with `if _RUST_AVAILABLE: try: pass except: pass`. The kernel really does
  export `jetflow_compute_acceleration`, so the flag was true, the kernel was
  never called, and `tests/test_jetflow_rust_kernels.py` compared the Python
  path with itself — a parity pin that could not fail. The acceleration now
  goes through `_acceleration_between`, which calls the kernel on both the
  directional and the central-difference paths and falls back only when the
  kernel refuses a shape; `NATIVE_ACCELERATION_CALLS` makes the branch
  observable. `apedp_dyn/grammar/llm_proposer.py` returned `[]` for a missing
  key, an unreachable base URL, a missing `httpx` and a provider 503, all
  indistinguishable from "no suggestion": it now raises
  `LLMProposerUnavailable` naming the provider, model and env vars, and records
  `last_outcome` so an empty list is always explainable. Registered instead of
  changed, with reasons in `docs/SCAFFOLD_INVENTORY.md`: the video package's
  availability flags (consumed, with real fallbacks), the finance problem's
  inert placeholder truth RHS, and the mesh discovery `dx=1.0`, which is a
  fail-closed type probe before the search and is replaced by the measured
  median edge length in the returned model.

- **The coefficient-prefix sweep covered the rest of the repo, and found a third
  live site.** The question was asked of every f-string that formats a field, a
  `*` and another field — 47 sites across `apedp/`, `apedp_dyn/`, `scripts/` and
  `benchmarks/` — because that shape is the defect, not the file it lives in.
  `apedp_dyn/ui/server.py` emitted the promoted theta-double-dot equation as
  `{coef} * {expr}` over a canonical expression that is routinely a sum, so the
  coefficient landed on the first addend; the neighbouring state and energy
  strings already parenthesised, which is how it survived review. Fixed in
  place. Two latent families were closed: caller-named operands
  (`qed.library.render`, `sindy`, `sparse_microstructure`, `regime_switching`,
  `invariant_transfer`, `distributed_lag_interaction`, `rime_ad`,
  `stem_proposers`) and the `exp(c*name)` labels in `discovery/_core.py` and
  `family_detection.py`, where a sum-named column shifted the exponent. The new
  `apedp_dyn/grammar/term_format.py` returns an atomic operand untouched, so
  recorded equations and candidate-id hashes do not move. Every surviving site
  carries a verdict in `STRING_ASSEMBLY_SITES`, and a site that is neither
  adjudicated nor converted fails `tests/test_term_builder_structure_audit.py`.

- **Every term builder and every emission site was audited for the `dxr` defect,
  and one more was found.** The question is asked numerically rather than by
  inspection — for a single-term equation, `rhs(p=2) - 2*rhs(p=1) == 0` — and the
  first version of the check was itself wrong: with two terms on one RHS,
  doubling one coefficient leaves the other's contribution in place, so a
  correct emission reads as broken (the residual was exactly `-q0*u1`). Record:
  `benchmarks/term_builder_structure_audit.json` by
  `scripts/audit_term_builder_structure.py`, pinned by
  `tests/test_term_builder_structure_audit.py`, documented in
  `docs/TERM_BUILDER_STRUCTURE_AUDIT.md`. Result: the grammar's seven emission
  sites bind their coefficient to the whole expression; the two live defects
  were **outside** the grammar — `advanced_search._expansions` formatted a seed
  into `{expr}*2`, `{expr}**2` and `{expr}*{var}` without parentheses, so on a
  sum seed (`x0 + 1`, which the coarse templates emit) the mutation reached the
  constant and the search silently explored a different family than it named;
  and `SINDyProposer` emitted `f"{c}*{name}"`, latent because every library
  column is a product today. Both now parenthesise the operand. `_core.py` uses
  `sp.Mul(coeff, part)` and is structurally immune.
- **The two-timescale split never used its fast timescale.** Found by the same
  audit: `enumerate_coupled` records `fast_variables`/`slow_variables` on each
  equation, but `_materialize_multiscale` read them from the *system* metadata,
  which holds only `family`/`dae_mode`/`timescales`/`constraint_terms`. Every
  term fell through to "default to slow", so `tau_fast` was a declared
  parameter that appeared in no equation and the mode emitted one timescale
  while reporting two. The materialiser now unions the per-equation labels with
  the system-level keys, and the audit pins `tau_fast` in the fast equation.
- **What is not a single product is now recorded, with the right verdict for
  each family.** `dxr` and `u_relax` are sums and legitimate — `u_relax` is
  checked on a velocity-driven probe plant, because on the Duffing plant a force
  input minus a velocity is incoherent and the unit check refuses it for a units
  reason, not a formatting one. Eight builders (`sat_*`, `inv_x*`, `log_abs_x`,
  `sqrt_abs_x`, `ret_x`) add a literal dimensionless `1` to a dimensional state
  and are refused on every dimensional problem; that is the documented
  dimensionless-only vocabulary, and the record says so instead of calling the
  gate a bug. `sin_x`/`cos_x`/`tanh_x` are refused separately as dimensional
  arguments.

- **The band split was re-run on per-pixel semantics, and the better signal fixed
  no verdict.** `scripts/compare_region_split_to_semantics.py` runs SegFormer-B0
  (ADE20K, weights pinned by SHA-256) on the ten audit photographs and cuts the
  two boundaries where the class *composition* changes most, with the analysis
  size, `min_band_fraction`, two-cut rule, evidence thresholds, hand labels and
  0.06 tolerance all held at the values the texture and depth arms used. Median
  boundary error falls to **0.115** against texture's 0.182 and depth's 0.228 and
  the worst case to 0.268 against 0.280, but the verdicts are identical to
  texture's (2 close / 1 partial / 4 wrong, none changed) and only 4 of 10 splits
  are endorsed by the evidence gate. The four failures are the two causes the
  first audit named — a true boundary below the 15 % minimum band height, and
  three-regime scenes forced into exactly two cuts — which no signal choice can
  fix. Record: `data/region_mask_audit/semantic_comparison.json`.
- **A class map cannot enter the texture splitter, and it fails silently.**
  ADE20K class ids are nominal, so `row_texture_energy`'s horizontal differences
  measure the numbering of the vocabulary. `derive_region_masks` still returns
  boundaries (0.18/0.75) for that input; only its evidence gate refuses them,
  with a cut decisiveness of 0.0. That measurement is now in the record so the
  arm's own score — total variation in class composition across a row — has a
  documented reason to exist rather than looking like an algorithm change made
  for convenience. The semantic environment also needed three pins a torch 2.3
  CPU build forces (`transformers==4.44.2`, `tokenizers==0.19.1`,
  `huggingface-hub==0.25.2`), each found by an ImportError and documented.

- **The mechanical grammar can no longer emit a free symbol.** `c1`/`c2` built the
  constant `1` as a "constraint basis"; `grad_x`/`laplacian_x` returned an unbound
  symbol for an operator with no grid behind it; `zscore_x`, `ma_x`, `mom_x`,
  `rsi_like`, `bollinger` and `macd_like` referenced `mu_x`, `sigma_x`, `x_lag1`,
  `x_ma`, `x_std`, `x_ema_fast/slow` that no caller binds. All ten keep their
  names (enumeration, families, canonicalisation and the proposer's filters use
  them) and raise `UnavailableGrammarTerm` with the reason at materialisation.
  Consequence measured, not glossed: the coupled ODE slate narrows from 1764 to
  225 candidates, the difference being the slates carrying a constant-1
  "constraint". (The survivor count quoted at the time, 64 of 225, turned out to
  be a `dxr` string-formatting artefact rather than a dimensional filter; the
  correction is in the entry below.)
- **The three remaining unbound symbols are bound or refused.** Found while
  auditing the grammar after the ten placeholders were withdrawn: (1)
  `mechanical_grammar.materialize` substituted `gxr_remote` for a remote-position
  term in a single-DOF problem — a symbol no state, input or parameter supplies —
  and left gate 1 to report `unknown symbol 'gxr_remote'`, which reads like bad
  caller data; it now raises `UnavailableGrammarTerm` naming the two-DOF grammar
  as what the term needs. (2) `proposer._library_matrix` built the saturation laws
  against the generic scale `gL` and never substituted it, so `lambdify` produced
  a column calling an undefined name: a `NameError`, not a refusal. The library
  now skips them with the reason. (3) All six build sites in `coupled.py` passed
  `gL{i}` to the builder and never substituted it, so `_unit_of` raised `unknown
  symbol 'gL0'` and every two-DOF saturation term was recorded as unbuildable;
  `_bind_scale` now registers the scale as its own parameter in the state's units
  (`role="scale"`), as the single-DOF path already did. Pinned by three tests in
  `tests/test_scaffold_refusals.py` that assert no emitted expression contains a
  bare `gL`.
- **The sparse library drops duplicate columns.** Withdrawn terms contribute no
  column; `drop_collinear_columns` drops a column correlating above
  `COLLINEARITY_LIMIT = 0.999` with one kept earlier. Measured need: `sin_x`
  correlates with `x` at 0.99997 on the Duffing trajectory, and the damping term
  lost the 5 % threshold race to it (`{u, x, x3}` recovered instead of
  `{u, x, x3, v}`). Recovery is exact again, and the same guard applies to any
  explicit vocabulary a caller passes.
- **The Triton flag stops claiming a kernel it does not have.** `TRITON_KERNELS`
  is an empty registry and `is_triton_available()` requires it, so an installed
  Triton plus a CUDA device no longer reports a Triton path that is pure PyTorch.
- **The video region masks are derived from the picture.** `derive_region_masks`
  (`apedp_dyn/video/region_masks.py`) splits the frame at the two sharpest steps
  in its per-row texture energy and returns the boundaries and band energies;
  the hard-coded top-40 %/bottom-40 %/middle-30 % bands are gone.
- **The derived region masks were audited against real photographs.** Ten
  CC-licensed photographs (`data/region_mask_audit/photos/`, licence, source and
  SHA-256 per file in `PROVENANCE.json`) were run through `derive_region_masks`
  and scored against band boundaries hand-labelled from a ruled contact sheet.
  Result, stated plainly: **2 of 7 scored frames close, 1 partial, 4 wrong**,
  median boundary error **0.18** of the frame height (worst 0.28). Three of the
  ten have no three-band horizontal structure at all — a forest corridor, an
  aircraft-window sea view, a sky-over-sea horizon — and the splitter returns two
  interior boundaries for every one of them, because it has no output that says
  "no bands". Two diagnostics in the record predict the failures without any
  labels: `texture_order` is non-monotone on 6/10, and the left/right halves
  disagree by more than the tolerance on 5/10. The fetcher refuses files whose
  Commons *categories* name a painting, drawing or collage — which is how a
  Kuindzhi oil was kept out after the title filter passed it. Written up in
  `docs/REGION_MASK_AUDIT.md`, pinned by
  `tests/test_region_mask_photo_audit.py`.
- **The region split measures its own support and can refuse.** Following the
  photograph audit, `derive_region_masks` returns an `evidence` block —
  `cut_decisiveness`, `half_disagreement` (the same cut search run on the left
  and right halves), `texture_order`, `reliable` and the `reasons` — and raises
  `UnusableRegionSplit` when called with `require_reliable=True` and nothing
  supports the split. Measured against the hand labels: it accepts **1 of 10**
  photographs (`beach_palm_trees`, one of the two within tolerance), accepts
  **none** of the four wrong splits, and rejects one split that was in fact close
  (`mountain_lake_reflection`, where the lagoon's halves carry different texture
  and one row boundary genuinely does not describe the frame). Timid, not wrong.
  `cut_decisiveness` never fired (10x or more on every photograph): on this
  sample the three-band model, not the cut detection, is what fails. The
  animator keeps its sky/sea/subject convention but prints the reasons when the
  picture does not support one instead of animating bands it cannot defend. A
  two-band result is *not* implemented and the record says so.
- **The two-DOF coupling term was never built, and a two-DOF saturation model now
  fits end to end.** Two materialiser defects, both in how a term's coefficient
  reaches the expression string. (1) `parts.append(f"({p}*{sstr(built)})")` did
  not parenthesise the built expression, so any term whose builder returns a
  *sum* had its coefficient applied to the first addend only. `dxr` is `XR - X`,
  so it emitted `(q2*-x1 + x2)`: the remote position entered with coefficient 1
  and gate 1 saw a length added to an acceleration. **Every** coupling candidate
  died at gate 1 for a formatting reason, and the slate's "survivors" were
  exactly the uncoupled ones — the rival D2-COUPLED is designed to defeat. The
  same line bug affected `u_relax` in the single-DOF grammar. Fixed in
  `_materialize_ode`, `_materialize_constrained`, `_materialize_multiscale` and
  `mechanical_grammar.materialize`. Measured consequence: the coupled slate goes
  from **64 kept to 225 kept — all of them** — so gate 1 filters nothing in this
  slate and the old 64 was never evidence of dimensional filtering. (2)
  `enumerate_coupled` emitted a saturation term bare, so `_bind_scale` refused it
  and every two-DOF saturation candidate was unbuildable. Each mass's
  `langmuir_bind`/`hill2_bind`/`hill3_bind` now carries its own fitted reference
  level (`s0_langmuir`, `s1_langmuir`, …) in the state's units with
  `role="scale"`. On a planted two-DOF Langmuir plant (per mass: `c`=0.20/0.15,
  `a`=2.0/3.0, `s`=0.30/0.45) the chain enumerate → prune → materialize → fit
  recovers `c`=−0.172/−0.150, `a`=−1.892/−2.911, `s`=0.260/0.428 at a train loss
  of 1.2e-4 — every parameter within 13.9 %. Recorded by
  `scripts/measure_two_dof_saturation_fit.py` →
  `benchmarks/two_dof_saturation_fit_results.json`: slate 225/225 kept, the
  planted law passes gate 4 at confirmation RMSE 0.052 while the Hill n=2 and
  n=3 rivals fail it at 9.65 and 1.35, and adjudication returns
  `unique_support`. A noise sweep on the series the fitter consumes (0/1/3 %,
  with confirmation truth left exact) gives reference-level errors 13.2/5.0 %,
  2.8/0.5 % and 5.0/2.3 % — 1 % noise recovers *better* than 0 %. The cause is
  measured, not guessed: **the automatic warm start stops 261x above a reachable
  minimum.** The objective at the planted parameters is 6.5e-06, a plant-seeded
  fit reaches 4.77e-07 with every parameter within 1 %, and the self-started fit
  terminates after about 24 evaluations at 1.24e-04 while reporting
  `success=True` — the warm start seeds the reference level from median|x| and
  the optimiser leaves it essentially where it was seeded. So the recorded
  13.9 % recovery is the warm start's ceiling, not the information in the data:
  the parameters are identifiable and the search is not finding them. Both fits
  are in the record under `objective_floor`, and a test pins the gap so it
  cannot be forgotten. Pinned by
  `tests/test_dyn_two_dof_saturation_fit.py` (11 tests), which also checks that
  a saturating rival nesting the linear D2-COUPLED truth does not beat it on
  confirmation.
- **The region split was re-run on a real depth model, and lost — for a measured
  reason.** `scripts/compare_region_split_to_depth.py` hands the same splitter
  (same 256 px size, smoothing, minimum band height, hand labels and tolerance)
  the MiDaS_small relative-depth map instead of RGB, and records the comparison in
  `data/region_mask_audit/model_comparison.json` with the checkpoint pinned by
  SHA-256 (85,761,505 bytes) alongside the torch/torchvision/timm versions it ran
  under. Result: **0 close / 4 partial / 3 wrong** against texture's 2 / 1 / 4,
  median boundary error 0.228 vs 0.182. The reason is measured, not asserted:
  MiDaS's row-mean depth is close to a linear ramp (median R² 0.827 vs 0.240 for
  texture; rank correlation +0.83 to +0.96 in all ten photographs), and a
  step-seeking split has nothing to cut in a monotone field — so the evidence
  gate refused **all ten** depth splits. Swapping the signal without changing the
  algorithm does not help, and the gate is what stops the pipeline presenting
  gradient artefacts as bands. Pinned by
  `tests/test_region_mask_depth_comparison.py`.
- **The text encoder says what it is.** `EMBEDDING_METHOD =
  "feature_hashing+hand_written_concept_ontology"`, `PRETRAINED_SEMANTICS =
  False`, BLAKE2b instead of MD5, and a docstring that states a hash cannot carry
  semantics the ontology does not name.
- **Derived QD rows cannot reach a GENERATION verdict.** `yu_cdse_rows` marks
  every row `independent_measurement: False` with its provenance, and
  `adjudicate_verdict` withholds `GENERATION` unless the rows are the measured
  Phase A table, keeping `CONFIRMATION` for agreement with the source curve.
- **The Tesla/energy simulators state their assumptions.** Slip, efficiency,
  primary voltage and irradiance are parameters (same defaults) echoed in
  `assumptions`, and every return carries `evidence: closed_form_textbook`.
- **The neural symbolic regressor implements teacher forcing.** `decode` accepts
  `targets` (used by `supervised_loss`) or an `rng` (used by the REINFORCE loss),
  so logits and chosen tokens align; losses are scalar arrays so `grad` works;
  `train_step` is eager and `TRAIN_STEP_JITTABLE = False` says why.

Updated: `apedp_dyn/grammar/mechanical_grammar.py`,
`apedp_dyn/grammar/proposer.py`, `apedp_dyn/cuda/batch_physics_steppers.py`,
`apedp_dyn/video/region_masks.py` (new), `scripts/animate_picture_to_video.py`,
`packages/jetflow-video/jetflow_video/text_encoder.py`, `apedp/qd_generation.py`,
`apedp/experimental_physics.py`, `apedp/neural_discovery/symbolic.py`,
`tests/test_scaffold_refusals.py` (new), `tests/test_dyn_d2b.py`,
`tests/test_dyn_d3.py`, `tests/test_dyn_d4.py`, `tests/test_dyn_d5.py`,
`docs/SCAFFOLD_INVENTORY.md`, `CHANGELOG.md`.

## 2026-11-01 — Formal and applied layers measured; negative results registered

- **The Lean layer now reports its own emptiness.** `scripts/audit_lean_formal_layer.py`
  scans all 557 `.lean` files: **0 are `checked`**, 360 are `axiom_injected`, 196 contain
  `sorry`/`admit`; 12,470 axioms, 1,054 theorems, and **94 theorems proved by rewriting with
  a local axiom**. `docs/LEAN_FORMAL_STATUS.md` states the rule: no result may be called
  formally verified on a non-`checked` file, and lists what would make it verification.
- **Every applied-science return is labelled.** `scripts/audit_applied_basis.py` walks the
  AST of `apedp/applied_*.py` and requires an evidence label on each returned dictionary;
  the six modules whose returns were unlabelled (electrical, fusion, materials, neuromorphic,
  defects, biopharma) now carry `EVIDENCE` and mark each return, so the count is 0 and a
  test fails if it rises. `_pam_matches` in synthetic-bio is reported as a boolean helper
  (it returns `True`/`False`, not a number), which is the only remaining constant return.
- **A negative-results register exists.** `docs/NEGATIVE_RESULTS_REGISTER.md` records the
  patent index's 8 decisive claims against a floor of 50 and its 7 refusals, the Lean layer's
  0 checked files, the applied modules' reference-only basis, and points at the quantum
  ledger rather than restating it. Section 5 names the gap: no register yet for accepted
  hypotheses that later failed to replicate.

Updated: `scripts/audit_lean_formal_layer.py` (new), `scripts/audit_applied_basis.py` (new),
`benchmarks/formal/lean_axiom_audit.json` (new), `benchmarks/applied_basis_report.json`
(new), `docs/LEAN_FORMAL_STATUS.md` (new), `docs/NEGATIVE_RESULTS_REGISTER.md` (new),
`apedp/applied_electrical.py`, `apedp/applied_fusion.py`, `apedp/applied_materials.py`,
`apedp/applied_neuromorphic.py`, `tests/test_formal_and_applied_audits.py` (new),
`CHANGELOG.md`.

## 2026-11-02 — The Rust bridge moves into `apedp_dyn`; the GA fallback is registered, not imported

- **`apedp/rust_bridge.py` moved to `apedp_dyn/rust_bridge.py`.** Its last dependency on
  the legacy root was the Python GA fallback: three lazy imports of
  `NativeSearchUnavailableError`, `SearchConfig` and `evolve` from
  `apedp.evolutionary_search`. The bridge now asks `apedp_dyn/native_search.py` for that
  API, and the legacy tree registers its GA when `apedp.rust_bridge` is imported. The
  canonical package names no legacy module.
- **The error class moved with the contract.** `NativeSearchUnavailableError` is defined in
  `apedp_dyn/native_search.py` and re-exported from `apedp.evolutionary_search`, so both
  public import paths return the same class object.
- **The legacy path is a forwarding shim, not a copied re-export.** The bridge owns
  module-level state (`_RUST_AVAILABLE`, `apedp_kernel`, caches, tallies) that tests
  replace to inject a fake kernel; a plain re-export would copy those bindings and a patch
  on one module would silently miss the other. `apedp/rust_bridge.py` forwards attribute
  reads to the implementation, and the two fake-kernel suites patch the implementation.
- **If nothing registered a provider, the bridge refuses.** Requesting the Python fallback
  from a pure-`apedp_dyn` context raises `NativeSearchUnavailableError` naming the missing
  registration instead of quietly running a different search.

Result: `apedp_dyn` import-time edges into `apedp` are **0** (6 at the start of this work).
The reverse direction, `apedp` -> `apedp_dyn`, remains the intended one.

Updated: `apedp_dyn/rust_bridge.py` (moved), `apedp/rust_bridge.py` (forwarding shim),
`apedp_dyn/native_search.py` (new), `apedp/evolutionary_search.py`,
`apedp_dyn/video/decoupled_transport.py`, `apedp_dyn/video/eval_gate.py`,
`apedp_dyn/proposers/rime_ad_proposer.py`, `tests/test_rust_bridge_dispatch.py`,
`tests/test_rust_bridge_observability.py`, `docs/DOCUMENTATION.md`, `CHANGELOG.md`.

## 2026-11-02 — The hardened parser and the proof-claim tier move into `apedp_dyn`

- **`apedp/safety.py` moved to `apedp_dyn/safety.py`; the old path is a façade.** Two of
  the six import-time edges from the canonical package back into the legacy root were
  `from apedp.safety import safe_parse_expr`: the SSM/latent side is gone, but the Lean
  bridge still pulled the legacy parser in on every import. The parser is
  stdlib-plus-sympy with no `apedp` imports, so it moves cleanly; `apedp/safety.py` now
  re-exports six names through a literal `__all__`, which
  `tests/test_facade_contracts.py` checks resolves.
- **`ProofClaimStatus` moved to `apedp_dyn/epistemic/proof_claim.py`.** The enum had been
  defined inside `apedp/mathematical/mathematical_layer.py`, so
  `apedp_dyn.epistemic.lean_universal_bridge` and the Lean-proof builder both imported the
  legacy root to name a three-value enum. `mathematical_layer` now imports and re-exports
  it, so both public paths resolve.
- **`rust_bridge` is the one edge that did not move, and the reason is measured.** It is
  not self-contained: three call sites delegate the Python fallback to
  `apedp.evolutionary_search` (`NativeSearchUnavailableError`, `SearchConfig`, `evolve`),
  a 1,175-line module that is itself part of the legacy discovery stack. Moving the bridge
  alone would relocate the import rather than remove it, so `apedp_dyn`'s three users
  (`video/decoupled_transport.py`, `video/eval_gate.py`,
  `proposers/rime_ad_proposer.py`) keep the existing path until the GA fallback is either
  moved with it or inverted to register itself.

Result: `apedp_dyn` import-time edges into `apedp` fell from **6 to 3** (the three
`rust_bridge` users); the reverse direction, `apedp` into `apedp_dyn`, is unchanged and is
the intended one. Fresh-interpreter first-import of six entry points is pinned by
`tests/test_facade_contracts.py`.

Updated: `apedp_dyn/safety.py` (moved from `apedp/safety.py`), `apedp/safety.py` (façade),
`apedp_dyn/epistemic/proof_claim.py` (new), `apedp/mathematical/mathematical_layer.py`,
`apedp_dyn/epistemic/lean_universal_bridge.py`,
`apedp_dyn/sandbox/build_all_system_equations_lean_proofs.py`,
`tests/test_facade_contracts.py`, `tests/test_flaw_fixes.py`,
`docs/DOCUMENTATION.md`, `docs/MAINTAINABILITY_SCORE.md` (new, re-scores
maintainability 6.0 -> 7.0 from the measurements above), `CHANGELOG.md`.

## 2026-11-02 — The façade audit stops auditing other people's code

- **24 permanently-red façade cases were vendored copies, not our code.**
  `tests/test_facade_contracts.py` parametrizes over every module
  `tools.coverage_audit.is_reexport_facade` classifies as a re-export shim. The walk
  reached `apepd_dyn/sandbox/.../kickmix/_qualtran_src`, an upstream git checkout copied
  in for a build, whose `__all__` is legitimately empty — so 24 upstream modules failed
  "a façade that exports nothing", and no change on our side could ever fix them. A
  finding that cannot be fixed is how a report stops being read.
- **The exclusion is a rule, not a path.** `_iter_python_files` now prunes (1) any nested
  git checkout, (2) any directory named `site-packages`, and (3) any `.venv*` — the last
  two because `.venv-region-models` was contributing another 16 failures of exactly the
  same kind from a torchvision copy. This mirrors the rule
  `tests/test_documentation_and_spec_integrity.py` already applies to the orphan scan,
  and `build/`, `dist/` and `.kilo/worktrees` were already excluded for the same reason.

Result: `tests/test_facade_contracts.py` is **82 passed, 0 failed** (was 82 passed, 24
failed), and `tests/test_coverage_audit.py` stays 16 passed.

Updated: `tools/coverage_audit.py`, `CHANGELOG.md`.

## 2026-11-02 — No opt-in to a fabricated kernel answer, the seal is re-minted, and a tactic template is not called verified

- **The placeholder opt-in is gone.** `ssm_scan`'s pass-through, `mamba2_selective_scan`'s
  `0.9*u` and `LatentSpace.decode`'s `"x0"` were previously reachable through
  `APEDP_ALLOW_PLACEHOLDER_KERNELS=1`, which meant the fabricated answers could still be
  requested. All three bodies are deleted, so the two SSM kernels and the latent decoder
  refuse with `NotImplementedError` and no variable brings the constants back. A test
  scans every `.rs` file in the crate rather than the two that used to hold the flag, and
  `cargo check` now builds with 0 warnings (the deleted placeholders were the source of
  the six unused-argument warnings).
- **The blind referee is re-minted as `APEDPX_FROZEN_v1_11.json`.** `apedp_dyn/ui/server.py`
  changed when its sandbox wiring was edited, so the v1_10 pin refused to score
  (`REFEREE OR FROZEN RECORD CHANGED since APEDP-X v1.10-FROZEN`). The new record re-pins
  the guard set; no referee file, threshold or blind case changed, and
  `tests/test_dyn_blind.py` scores again on the re-minted seal.
- **The Lean bridge may not describe a tactic template as verified.** The module returned a
  proof-shaped object for `by positivity` / `by ring` / `by linarith` patterns that it never
  submitted to Lean; those notes now say `NOT machine-checked`, and a test pins the absence
  of "formally verified" from the module plus the presence of that disclaimer. The audited
  layer itself is unchanged and still honest: 557 `.lean` files, 12,470 axioms,
  `checked: 0`, `python3 scripts/audit_lean_formal_layer.py --check` exiting 1 by design.

Updated: `rust/apedp-kernel/src/ssm.rs`, `rust/apedp-kernel/src/latent.rs`,
`apedp_dyn/sandbox/blind_benchmark.py`, `apedp_dyn/sandbox/APEDPX_FROZEN_v1_11.json` (new),
`apedp_dyn/epistemic/lean_universal_bridge.py`, `ARCHITECTURE.md`,
`tests/test_placeholder_kernels_refuse.py`, `tests/test_dyn_lean_universal_bridge.py`,
`tests/test_dyn_blind.py`, `CHANGELOG.md`.

## 2026-11-01 — Three flaw fixes: operation-skeleton matching, thin-index floor, refusing placeholders

- **The structural tier compares the operation tree, not degree and symbol count.**
  `2*Z_meas/Z_ref` used to be `RELATED_STRUCTURE` to `1 - SS_res/SS_tot` because both had
  degree 2 and two symbols; an Arrhenius law likewise matched an exponential-plus-constant
  temperature law. Symbols and numbers are now replaced by *positional* sentinels and the
  skeletons compared, so `RELATED_STRUCTURE_IN_PATENT_CORPUS` means one functional form
  with different constants. A single shared sentinel was tried first and rejected: it made
  `Z_meas/Z_ref` cancel to `1`, collapsing unrelated forms onto the same skeleton.
- **Thin-index floor is declared and reported.** `DECISIVE_CLAIM_FLOOR = 50`; every verdict
  from an index below it now says "this index holds 8 decisive claims against a declared
  floor of 50, so absence here is absence from a thin index", and `corpus_status()` reports
  `decisive_claims_floor`, `meets_decisive_floor` and `thin`.
- **Placeholder kernels refuse instead of returning plausible constants.** `ssm_scan`
  (returned its input), `mamba2_selective_scan` (returned `0.9*u`) and
  `LatentSpace.decode` (returned `"x0"`) raise `NotImplementedError` unless
  `APEDP_ALLOW_PLACEHOLDER_KERNELS=1`; the Python latent decoder always refuses, and
  `latent.rs` is documented as a feature hash rather than a VAE. The integer-relation
  search declares `INTEGER_RELATION_METHOD = "brute_force_branch_and_bound"` and no longer
  claims PSLQ.
- **The EPO OPS claims path is exercised end to end** through a replay: token → search →
  claims → raw artefact, asserting URL, Bearer headers, claim numbering and digest, so
  turning off Google's endpoint is a credentials problem rather than a code problem.

Updated: `apedp/patent_prior_art.py` (`DECISIVE_CLAIM_FLOOR`, operation-skeleton tier),
`scripts/run_patent_prior_art.py`, `scripts/grow_patent_corpus.py` (OPS transport),
`rust/apedp-kernel/src/ssm.rs`, `rust/apedp-kernel/src/latent.rs`,
`rust/apedp-kernel/src/rime/number_theory.rs`, `apedp/advanced_search.py`,
`tests/test_placeholder_kernels_refuse.py` (new), `tests/test_patent_bulk_sources.py`,
`tests/test_patent_corpus_real.py`, `tests/test_dyn_patent_prior_art.py`,
`docs/SEARCH_ENGINE_GAPS.md`, `benchmarks/patent_prior_art/prior_art_run.json`,
`CHANGELOG.md`.

## 2026-11-01 — The Google block lifted; 19 new records and 2 newly reachable claims normalised

- **The third harvest attempt succeeded.** `01:34:17Z` returned HTTP `200` where the two
  earlier attempts got `503`: five queries fetched **21 publications** (32 raw artefacts),
  and `--build` grew the corpus from 10 to **29 records**. The attempt log keeps all three
  attempts with their status, so the growth is attributable to a specific opening rather
  than to a change in method.
- **Two newly reachable claims state a relation, and both were normalised by hand.**
  `US11496054B2` claim 1 states a volt-second balance and its solved form, recorded as
  `(d2*U2 - d1*U1)/d3` — **decisive**, because the difference and the division are
  structure. `US12021452B2` claim 1 states a `2:1` conversion ratio, recorded as
  `V_in/V_out` — **operand-only**, since a ratio of two bare quantities matches any
  quotient. The other 17 new records state no relation in any claim; each carries a
  `_note` saying all of its claims were read and counted as unsearchable.
- **The split is visible and measured: 15 searchable claims — 8 decisive, 7 refused**
  (EP4334731B1 2/7/8, US12021452B2 1, US20250321289A1 11, US8965721B2 1/8), across 7
  records with a stated relation and 22 unsearchable. `REFUSED_CLAIMS.md`,
  `prior_art_run.json` and the corpus split in `harvest_attempts.jsonl` all report those
  numbers; the automatic extractor still parses 0 of **501** claims, which is why the
  growth is transcription rather than extraction.

Updated: `data/patents/corpus.jsonl`, `data/patents/normalised_claims.json`,
`data/patents/extraction_yield.json`, `data/patents/harvest_attempts.jsonl`,
`data/patents/REFUSED_CLAIMS.md`, `benchmarks/patent_prior_art/prior_art_run.json`,
`apedp/patent_prior_art.py` (scope text), `scripts/run_patent_prior_art.py`,
`data/patents/PROVENANCE.md`, `docs/SEARCH_ENGINE_GAPS.md`, `docs/SCAFFOLD_INVENTORY.md`,
`tests/test_patent_corpus_real.py`, `tests/test_patent_bulk_sources.py`,
`tests/test_scaffold_inventory.py`, `CHANGELOG.md`.

## 2026-11-01 — Harvest re-run once more: still 503, so no claim became newly reachable

- **Re-ran the harvest; the block is still up.** `01:34:17Z` returned HTTP `503` on both
  the query endpoint and a publication page, logged as the third attempt with
  `records_added: 0`. No record arrived, so **no claim became newly reachable and nothing
  needed normalising**: the corpus stays at 10 records, 13 normalised claims.
- **The decisive/operand-only split is now recorded with every attempt.** `log_attempt`
  writes `corpus_split` (records, searchable, decisive, refused operand-only, and the
  refused claim numbers per record) and `--retry-harvest` prints it whether the source is
  blocked or open, so a blocked run reports what the corpus is rather than only what it
  failed to add. Verified after the re-run: 13 claims, **7 decisive, 6 refused**
  (EP4334731B1 2/7/8, US20250321289A1 11, US8965721B2 1/8).

Updated: `scripts/grow_patent_corpus.py` (`corpus_split`, `_format_split`,
`log_attempt(split=...)`), `data/patents/harvest_attempts.jsonl`,
`data/patents/PROVENANCE.md`, `tests/test_patent_bulk_sources.py`, `CHANGELOG.md`.

## 2026-11-01 — A structured candidate can no longer be swallowed by an operand-only claim

- **Found while probing the tier order: `2*Z_meas/Z_ref` was refused as operand-only.**
  `check_rename_isomorphism` accepts a scalar multiple, so a candidate carrying a
  coefficient was isomorphic to the bare-quotient claim `Z_meas/Z_ref` and the refusal
  fired — reporting a structured form as undecidable by a claim that states less than it
  does. The operand-only tier now only speaks for a candidate that is itself
  operand-only (`is_operand_only(candidate)`), so a coefficient or an inner function
  falls through to the decisive, structural and absence tiers.
- **Tier order is the guarantee, stated where it is relied on:** the decisive tier is
  consulted first and returns immediately, so a decisive claim wins regardless of the
  order records appear in the corpus; the refusal is reached only when no decisive claim
  matched and the candidate belongs to the refused form class.
- **A probe row was added to the run record:** `structured_not_swallowed`
  (`2*Z_meas/Z_ref`) previously returned `CLAIM_STATES_OPERANDS_ONLY`; it now reaches the
  structural tier against a decisive claim (`RELATED_STRUCTURE_IN_PATENT_CORPUS`, same
  coarse signature as `1 - SS_res/SS_tot`). The run fails if any structured candidate is
  ever refused as operand-only (`ordering_violations`), records
  `matched_claim_is_decisive` per candidate, and a unit test pins the case with the
  operand-only record listed first.

Updated: `apedp/patent_prior_art.py` (operand-only tier guard),
`scripts/run_patent_prior_art.py` (`structured_not_swallowed` probe),
`benchmarks/patent_prior_art/prior_art_run.json`, `tests/test_dyn_patent_prior_art.py`,
`CHANGELOG.md`.

## 2026-11-01 — The operand-only refusals are listed per record, not just counted

- **Each refused claim now carries its reason.** `operand_only_form_class()` names the
  defect (`bare quotient`, `bare product`, `bare symbol (an equality of names)`) and
  `operand_only_refusals()` returns one entry per refused claim with the record, claim
  number, normalised form, class, why it cannot decide a verdict, and the transcription
  note that quotes the claim. The six refusals live in three records: EP4334731B1 claims
  2/7/8, US8965721B2 claims 1/8, US20250321289A1 claim 11.
- **`data/patents/REFUSED_CLAIMS.md` is generated from the corpus** by
  `refusals_markdown()` and written by `scripts/run_patent_prior_art.py`, with a
  per-record header stating how many of that record's claims are refused and how many
  remain decisive. A test regenerates it and fails when the file is stale, so the list
  cannot drift from the claims it describes.
- **`prior_art_run.json` carries the refusals** as `refusals.by_form_class`,
  `refusals.by_record`, `refusals.detail` and a per-record `refused_claims` list, and
  `corpus_status()` gained an `operand_only` list beside `operand_only_claims`.

Updated: `apedp/patent_prior_art.py` (`operand_only_form_class`,
`OperandOnlyRefusal`, `operand_only_refusals`, `refusals_markdown`),
`scripts/run_patent_prior_art.py`, `data/patents/REFUSED_CLAIMS.md` (new),
`benchmarks/patent_prior_art/prior_art_run.json`, `tests/test_patent_corpus_real.py`,
`CHANGELOG.md`.

## 2026-11-01 — Harvest retried: still blocked at 503, and the attempt is now on disk

- **Retried the harvest, the block has not lifted.** `https://patents.google.com/xhr/query` and individual
  publication pages both answer `503` with the "Sorry…" block page, so **no records were added**: the corpus
  stays at 10 records, 13 normalised claims. Nothing was fetched on the strength of a partial response.
- **`--retry-harvest` makes the retry auditable.** Each attempt appends to
  `data/patents/harvest_attempts.jsonl` — `attempted_utc`, endpoint, `http_status`, `outcome`
  (`open`/`blocked`/`unreachable`) and `records_added` — and exits `3` while the source is blocked, so a
  scheduler or shell check can tell "could not query" from "queried and found nothing". An `open` attempt
  logs twice: once before harvesting and once with the count it added.
- **A test pins the log's shape**, and asserts every blocked attempt added zero records. `PROVENANCE.md`
  names the logged 503 as the reason the corpus is still ten records.

Updated: `scripts/grow_patent_corpus.py` (`--retry-harvest`, `log_attempt`),
`data/patents/harvest_attempts.jsonl` (new), `data/patents/PROVENANCE.md`,
`tests/test_patent_bulk_sources.py`, `CHANGELOG.md`.

## 2026-11-01 — EPO OPS adapter tests replay recorded responses; the authenticated one skips rather than fakes

- **Recorded, not synthesised.** `data/patents/ops_fixtures/` holds two responses recorded from the live
  endpoints without OAuth — `claims_EP4334731_403.xml` (403 "violation of Fair Use policy") and
  `token_401.xml` (401 "Client identifier is required") — with URL, method, timestamp, byte count and
  SHA-256 in `PROVENANCE.json`. A test re-hashes every fixture against that file, so a replayed body
  cannot drift from its recording.
- **The replay goes through the adapter, not the parser.** `epo_ops_claims()` gained a `transport`
  parameter, defaulting to the live GET; a replay now exercises the endpoint it builds, the
  `Authorization: Bearer …` and `Accept: application/xml` headers, and the mapping from status to
  `SourceUnavailable(status=…)`. The recorded 403 asserts all four.
- **The authenticated replay test exists and skips, with the reason and the command.** No authenticated
  claims response could be recorded here (401/403 measured), so
  `test_recorded_authenticated_claims_response_replays` skips until
  `scripts/record_epo_ops_fixture.py --number EP4334731` runs with `EPO_OPS_KEY`/`EPO_OPS_SECRET` and
  writes `claims_EP4334731.xml`. The recorder exits `2` and writes nothing while the credentials are
  absent, so an empty fixture directory can never look like a recorded one.
- **Synthetic XML keeps its place, labelled.** The existing inline-fragment test is still a parser test;
  the new replay tests are the ones that claim a recording, and the skip is visible in the run summary
  rather than silently absent.

Updated: `scripts/grow_patent_corpus.py` (`epo_ops_claims(transport=…)`),
`scripts/record_epo_ops_fixture.py` (new), `data/patents/ops_fixtures/` (new: two recordings plus
`PROVENANCE.json`), `data/patents/PROVENANCE.md`, `tests/test_patent_bulk_sources.py`, `CHANGELOG.md`.

## 2026-11-01 — Claim 1 of the five harvested patents curated: no relation in four, a pointer in one

- **Claim 1 was read for all five harvested patents, and none states a relation.**
  `US11921163B2`, `US12130335B2`, `US12372585B2` and `US20250321289A1` state procedures (a GNN pipeline,
  a QV→IC→ICA pipeline, a time-series model input/output, an ML-plus-Kalman fusion pipeline);
  `US11474159B2` names three "relation functions" without writing them. The reading is stored as a
  `claim_1` object (`states_relation`, `reason`) on each record, so "claim 1 was read" is on disk rather
  than implied.
- **The set grew by one form, not five, and the reason is recorded.** The pointer in `US11474159B2`
  claim 1 resolves to dependent claims 3, 5, 6, 7 and 8; claim 6's simplified law
  `-0.01*b_SEI*SOH + b_SEI` is a form distinct from claim 5's free-slope version, and was normalised —
  decisive, because the `-0.01` coupling is the claim's own coefficient. Claim 3's Arrhenius law was not
  counted a second time: it is the same form as claim 7. Searchable claims **12 → 13**, decisive
  **6 → 7**, operand-only unchanged at 6.
- **No arithmetic was invented to make claim 1 yield formulas.** The Kalman fusion in
  `US20250321289A1` claim 1 and the incremental-capacity derivative in `US12130335B2` claim 1 are named in
  the reasons as things the claim does not state; writing `sum(w_i * SoH_i)` or `dQ/dV` would be the
  builder's physics wearing the claim's number.
- **Pinned** by `test_harvested_records_record_what_claim_1_states` (all five readings present and
  relation-free, claim 6 present with its coefficient) and the updated per-record pin, which now allows
  one record to hold four claims and asserts that exactly one does.

Updated: `data/patents/normalised_claims.json`, `data/patents/corpus.jsonl`,
`benchmarks/patent_prior_art/prior_art_run.json`, `apedp/patent_prior_art.py`, `data/patents/PROVENANCE.md`,
`docs/SEARCH_ENGINE_GAPS.md`, `docs/SCAFFOLD_INVENTORY.md`, `tests/test_patent_corpus_real.py`,
`tests/test_scaffold_inventory.py`, `CHANGELOG.md`.

## 2026-11-01 — Bulk-source growth measured as blocked, with every endpoint tried written down

- **Re-attempted live, same day, with no credentials in the environment.** EPO OPS answers `401`
  "Client identifier is required" on the token endpoint and `403` "violation of Fair Use policy"
  anonymously on a claims request; PatentsView's API host does not resolve; Google Patents serves a `503`
  block page on both the query endpoint and individual publication pages. No records were added, because
  adding records was not possible without a credential.
- **Two further keyless routes were tried and recorded, so a later reader does not repeat them.**
  USPTO's legacy `ds-api` bulk search now `301`s to `data.uspto.gov`, whose API is key-gated, and
  USPTO's Open Data Portal is key-gated outright. HuggingFace BIGPATENT's viewer was inspected: it
  serves `abstract` and `description` and **no claim text**, so it cannot produce a claim record; HUPD
  carries claims but its dataset viewer is unsupported, so no rows are retrievable without the full
  download.
- **`--probe-sources` now records six endpoints** with a per-source note separating "needs a key" from
  "would not serve claims even with a key", so a credential refusal is distinguishable from a dead end.
  `data/patents/SOURCES.md` was regenerated from that probe and `data/patents/PROVENANCE.md` carries the
  same measurements.
- **What unblocks growth is a credential, not code.** `--source epo-ops` (claims directly) and
  `--source patentsview` (discovery, claims then from OPS) are implemented and pinned; both exit `2`
  with the measured reason while `EPO_OPS_KEY`/`EPO_OPS_SECRET` or `PATENTSVIEW_API_KEY` are absent.

Updated: `scripts/grow_patent_corpus.py` (two endpoints, per-source notes), `data/patents/SOURCES.md`
(regenerated), `data/patents/PROVENANCE.md`, `tests/test_patent_bulk_sources.py` (the probe table must
name every endpoint tried), `CHANGELOG.md`.

## 2026-11-01 — The prior-art engine refuses to conclude from a bare quotient or product

- **`Z_meas/Z_ref` can no longer return `KNOWN_IN_PATENT_CORPUS`.** `is_operand_only()`
  classifies a normalised form that names its operands and one operation and nothing about them:
  `A/B`, `A*B`, `A+B` on bare symbols, with no function, offset, coefficient or composite operand
  inside. Such a claim is isomorphic to **any** candidate built the same way, so a match says
  nothing about the patent. Five of the ten normalised claims are in this class — `Z_meas/Z_ref`,
  `I_lim*V_min`, `eta_C*i_ch`, `C_t_I/C_0_I`, `C_t_E/C_0_E` — and are held apart as
  `PATENT.decisive` vs `PATENT.operand_only` rather than searched as if they decided anything.
- **A new status carries the refusal:** `CLAIM_STATES_OPERANDS_ONLY`, with the matching record,
  claim number and expression, and a disclaimer saying the match is not evidence and is refused
  rather than reported as known. Structured forms are unaffected: `AF*exp(t_A)`, `1 - SS_res/SS_tot`,
  `k_SEI*SOH + b_SEI`, `s(t)**2` and `A_SEI*exp(B_SEI/T) + C_SEI` still decide, and the structural
  tier only compares decisive claims.
- **Counts are reported, not implied.** `corpus_status()` and every audit result carry
  `decisive_claims: 5` and `operand_only_claims: 5` (searchable stays 10), and the run record
  reports both.
- **Two declared expectations moved, with the reason recorded.** `self_check_efficiency`
  (`eta_C*i_ch`) and `generic_ratio` (`x/y`) now expect `CLAIM_STATES_OPERANDS_ONLY`, and the probes
  exist precisely to keep that refusal measured instead of skipping the cases that used to read
  `KNOWN`. `tests/test_dyn_patent_prior_art.py` pins the refusal, the structured-form pass-through,
  and the classifier on five operand-only and five structured forms.

Updated: `apedp/patent_prior_art.py` (`is_operand_only`, `CLAIM_STATES_OPERANDS_ONLY`,
`decisive`/`operand_only` search splits, `decisive_claims`/`operand_only_claims` on results and in
`corpus_status()`), `scripts/run_patent_prior_art.py`, `benchmarks/patent_prior_art/prior_art_run.json`,
`data/patents/PROVENANCE.md`, `docs/SEARCH_ENGINE_GAPS.md`, `docs/SCAFFOLD_INVENTORY.md`,
`tests/test_dyn_patent_prior_art.py`, `tests/test_patent_corpus_real.py`,
`tests/test_scaffold_inventory.py`, `CHANGELOG.md`.

## 2026-11-01 — Patent corpus normalised per claim: 3 → 10 searchable forms, claim number on every verdict

- **The one-claim limit no longer bounds a verdict.** `data/patents/normalised_claims.json` moved to
  schema `v2`, where each record carries a `claims` list. Every claim whose text states an explicit
  relation was normalised by hand: EP4334731B1 claims 2/7/8 (`eta_C*i_ch`, `C_t_I/C_0_I`, `C_t_E/C_0_E`),
  US11474159B2 claims 4/5/7 (`1 - SS_res/SS_tot`, `k_SEI*SOH + b_SEI`, `A_SEI*exp(B_SEI/T) + C_SEI`),
  US8965721B2 claims 1/8 (`Z_meas/Z_ref`, `I_lim*V_min`), US20250321289A1 claim 4 (`s(t)**2`) and
  US7120566B1 claim 1 (`AF*exp(t_A)`). Searchable claims went **3 → 10 across five records**.
- **The engine searches all of them.** `NormalisedClaim` carries claim number, verbatim text, expression
  and note; `PatentPriorArtIndex.searchable` is a (record, claim) pair per normalised claim, and every
  verdict now reports `matched_claim_number` beside `matched_patent_id`, so a `KNOWN` result names the
  claim that matched instead of implying the patent as a whole. Records without a stated relation are
  still counted, not dropped: five records, unchanged in the denominator.
- **Two normalisations are marked COARSE, and say so.** `1 - SS_res/SS_tot` groups the claim's two sums,
  and `s(t)**2` drops the claim's definite integral; both therefore match more than the claim states.
  The note records the dropped structure, and the granularity limit in `PROVENANCE.md` now names them.
- **A measured consequence in the run record.** With claim 7 of US11474159B2 normalised, the declared
  `arrhenius_rate` probe moved from `NOT_FOUND_IN_PATENT_CORPUS` to `RELATED_STRUCTURE_IN_PATENT_CORPUS`:
  the structural tier's signature (function `exp`, exponent degree -1, four symbols) coincides for an
  exponential-plus-constant temperature law and an Arrhenius law, which are *not* equivalent. The
  expectation was updated with that reason rather than the probe being removed, and the row now measures
  how coarse the structural tier is. `generic_ratio` still matches a quotient form, now with three
  quotient claims in the corpus to land on.

Updated: `data/patents/normalised_claims.json` (schema v2), `data/patents/corpus.jsonl`,
`apedp/patent_prior_art.py` (`NormalisedClaim`, multi-claim search, `matched_claim_number`,
`records_with_formula`, `max_claims_per_record`), `scripts/grow_patent_corpus.py`,
`scripts/run_patent_prior_art.py`, `benchmarks/patent_prior_art/prior_art_run.json`,
`data/patents/PROVENANCE.md`, `docs/SCAFFOLD_INVENTORY.md`, `tests/test_patent_corpus_real.py`,
`tests/test_dyn_patent_prior_art.py`, `tests/test_scaffold_inventory.py`, `CHANGELOG.md`.

## 2026-11-01 — Patent corpus grown to ten records; the bulk paths refuse instead of pretending

- **Both named bulk sources are implemented, and neither can be used here.** `--source patentsview`
  (PatentsView `search` API, `PATENTSVIEW_API_KEY`) and `--source epo-ops` (OAuth token + per-publication
  `.../claims`, `EPO_OPS_KEY`/`EPO_OPS_SECRET`) now exist in `scripts/grow_patent_corpus.py` and both
  **exit 2 with the measured reason** when their credentials or endpoints are unavailable, so an
  unqueried source can never be read as "no patents found". Measured on 2026-11-01: PatentsView's host
  does not resolve and its legacy endpoint answers HTML, EPO OPS answers `403` without OAuth, and Google
  Patents' query endpoint began returning a `503` block page mid-run (individual publication pages too).
- **What grew: the record layer, 5 → 10.** Five publications were harvested from the Google Patents
  query endpoint on the same subject vocabulary as the curated five, each keeping the query that found
  it, its digest and its retrieval date. `data/patents/SOURCES.md` records every endpoint's HTTP status
  and states that a query-selected sample is not an office population.
- **What did not grow: the searchable layer.** The strict extractor was run over **every** claim of every
  artefact — 173 claims, **0 parsed formulas** — because patent claims state relations in prose and
  equations appear as images. `data/patents/extraction_yield.json` stores that measurement, and
  `corpus_status()` now reports `records`, `unsearchable_records`, `claims_scanned_for_formulas`,
  `formulas_found_in_claim_text` and `corpus_sha256`, so a `NOT_FOUND_IN_PATENT_CORPUS` verdict carries
  its own denominator instead of a bare record count.
- **Growth is transcription, not fetching.** Widening the matchable set means normalising each held
  claim by hand, as was done for the curated five; a larger corpus alone widens only the set absence is
  stated over. That distinction is now written in `data/patents/PROVENANCE.md` and in the module docstring.
- **Pinned** by `tests/test_patent_bulk_sources.py` (both refusal paths and exit codes, the OPS claims
  parser on a synthetic fixture, refusal of claim-less and non-XML responses), the extraction-yield and
  harvested-row pins in `tests/test_patent_corpus_real.py`, and the updated coverage pin in
  `tests/test_scaffold_inventory.py`. The prior-art run record was regenerated.

Updated: `scripts/grow_patent_corpus.py`, `apedp/patent_prior_art.py`, `data/patents/corpus.jsonl`,
`data/patents/PROVENANCE.md`, `data/patents/SOURCES.md`, `data/patents/extraction_yield.json` (new),
`benchmarks/patent_prior_art/prior_art_run.json`, `docs/SCAFFOLD_INVENTORY.md`,
`tests/test_patent_bulk_sources.py` (new), `tests/test_patent_corpus_real.py`,
`tests/test_scaffold_inventory.py`, `CHANGELOG.md`.

## 2026-11-01 — Both experimental benchmark tables checked against their sources: unverified

- **A script that re-fetches the cited literature.** `scripts/verify_external_benchmark_sources.py`
  pulls NASA TM-81927 (Harris 1981), the NASA TMR digitizations of Ladson (TM 100526) and
  Gregory & O'Reilly (R&M 3726), and the Meyers 2006 review, records URL/status/bytes/SHA-256
  for each, and measures what they contain. Exit `0` matched, `1` a measurement moved, `2` a
  source was unreachable — and unreachable is recorded as unverified, never as verified.
- **NACA 0012 — `unverified`.** The cited report has one table (section coordinates) and its
  pressure distributions are figures; no accessible source tabulates Cp at x/c=0.25 across
  Mach. The digitized Mach-0.30 tap at alpha~0 gives Cp(0.25) = -0.3551; scaled by
  Prandtl–Glauert, the table's values from M=0.50 to M=0.80 are a **constant 1.475x** too
  negative — five Mach numbers to three decimals — and the table states neither alpha nor Re,
  although Cp(0.25) moves by a factor of 2.8 between alpha~0 and alpha=10 in that source.
- **Nanocrystalline copper — `unverified`.** The published data at the one comparable grain
  size are IGC compacts: 650 MPa at 19 nm and 850 MPa at 20 nm, against the table's 910 MPa at
  20 nm — **+7.1% over the strongest and +40.0% over the weakest**. The table's 980 MPa peak at
  12 nm exceeds the review's stated 100–900 MPa envelope for 100 um–26 nm, and ten of the
  eleven rows have no traceable record.
- **Marked, not corrected.** `NACA0012TransonicBenchmark.provenance()` and
  `NanocrystallineMicrocompressionBenchmark.provenance()` now return `verified: False` with the
  citation, check date, record path and reason, so a caller sees the status without reading a
  page. The datasets are untouched: inventing a correction would be a larger defect.
- **Recorded** in `benchmarks/external_benchmarks_source_check.json` and
  `docs/BENCHMARK_SOURCE_VERIFICATION.md`, and pinned by
  `tests/test_external_benchmark_source_verification.py` (recorded hashes vs live datasets,
  the constant-ratio signature, the alpha dependence, the 20 nm comparison, the two flags).

Updated: `scripts/verify_external_benchmark_sources.py` (new),
`benchmarks/external_benchmarks_source_check.json` (new),
`docs/BENCHMARK_SOURCE_VERIFICATION.md` (new),
`tests/test_external_benchmark_source_verification.py` (new),
`apedp/ingestion/external_benchmarks.py`, `CHANGELOG.md`.

## 2026-11-01 — The census over `apedp_dyn`: five vendored modules named, not dropped

- **Run without the record's exclusion, `apedp_dyn` reports 22 modules.** Seventeen are
  this repository's and were already recorded and named in the inventory; the other five sit
  inside the vendored harness, which the record skips.
- **The five are now named.** `docs/SCAFFOLD_INVENTORY.md` lists each path, its tables and its
  functions — three in Qualtran's `_qualtran_src` (`bloq_report_card`, `qlt_ir/_examples`,
  `surface_code/flasq/volume_counting`, `surface_code/ui`) and `verifiers/dfthcsos/scoreinfra/reference.py`.
  They stay out of the gate because they are third-party code, but an exclusion nobody can
  audit is indistinguishable from an omission, so the set is written down and pinned.
- **The pin is the new test** `test_every_census_module_in_apedp_dyn_is_named_in_the_inventory`:
  seventeen recorded modules must be named in the table, five vendored modules must be named
  in the exclusion list, and none of the five may appear in the record. A sixth vendored
  module fails until it is named so the reviewer can see what was set aside.

Updated: `docs/SCAFFOLD_INVENTORY.md`, `tests/test_scaffold_inventory.py`, `CHANGELOG.md`.

## 2026-11-01 — The census becomes a pre-commit gate, so an unrecorded table cannot reach review

- **`scripts/audit_static_tables.py --check`.** The census used to print and exit 0 whatever
  it found. It now compares the live tree against `scripts/static_table_census.json` and exits
  `1` when a table appears, disappears or moves between modules — three cases, because a
  relocated table is a stale record even when the count is unchanged.
- **One record, two readers.** The JSON holds both roots (`apedp`, `apedp_dyn`) and the
  excluded vendored harness; the hook and `tests/test_scaffold_inventory.py` read the same
  file, so the test and the gate can no longer disagree about what is recorded. The inline
  `EXPECTED_STATIC_TABLES` dictionaries were removed in favour of it.
- **A pre-commit hook** (`static-table-census`, `always_run`, no filenames) sits next to
  `ruff-changed` in `.pre-commit-config.yaml`, so the failure happens before review, not in
  the reviewer's head. Exit `2` is reserved for a check that cannot run at all, so a broken
  baseline is not mistaken for a clean tree.
- **Record 28 modules**: eleven in `apedp`, seventeen in `apedp_dyn` (the fourth sweep). A new
  table needs two edits — the record and a row in `docs/SCAFFOLD_INVENTORY.md` — and the suite
  refuses a record whose functions are not named on that page.

Updated: `scripts/audit_static_tables.py`, `scripts/static_table_census.json` (new),
`.pre-commit-config.yaml`, `tests/test_scaffold_inventory.py`, `docs/SCAFFOLD_INVENTORY.md`,
`CHANGELOG.md`.

## 2026-11-01 — `apedp_dyn` swept by the same two signals, and what survived reading

- **Method unchanged, scope doubled.** The audit that produced the `apedp` sweeps now runs
  over `apedp_dyn`: an unused parameter in a function body, and a return that is a pure literal.
  **161 candidates** survived the two signals; each was read.
- **What was not listed, and why.** Abstract and protocol methods (`vqc/backends.run`,
  `grammar/proposer.propose`, the `experiment*.py` protocols, six `remote_storage`
  declarations), twelve `fit_and_propose(**kwargs)` parity stubs, underscore arguments
  (`_t`, `_objects`), predicates returning a literal label chosen from their inputs, and
  thirteen `proposer_id` methods returning a constant identifier.
- **Fifteen findings survived, and every row is a measurement.** The largest: the execution
  cost models ignore the inputs their own formula has no term for (fixed-plus-slippage
  ignores `volumes` — identical drag at 1e3 and 1e9 — Almgren–Chriss ignores `spreads`, the
  queue model ignores `prices`); `check_with_pi_groups` never touches `pi_groups` and the gate
  ships with no regimes, so every regime name raises; `certified_cbf_guard` ignores `headings`,
  so the certificate is a point-mass QP that says nothing about the unicycle heading;
  `infer_shape_prior` drops three knobs; `simulate_type2_fusion` ignores the state it names;
  `run_eseds_trap` forwards none of `r`/`K`/`N0`/`bootstrap_n_boot`; `ablation` discards the
  supplied `coefficients`; four benchmark generators accept a `seed` that drives nothing.
- **Two documented no-ops are separated from the findings** — `predict_next_frame(order)` and
  `simulate_pair(m, k)` say in their docstrings that the argument is retained for interface
  compatibility. That is the same fact, honestly labelled.
- **Census for `apedp_dyn`: seventeen modules** read a container literal on a return path, all
  of them catalogues, registries or step ladders. Two are recorded as census **false
  positives** found by reading: `candidate_ir.base_provenance` returns
  `list(c.parent_candidates)` (a candidate field reached through `c`, not `self`, so the state
  exclusion misses it) and `vqc/native.last_failure`/`tallies` are accessors over the native
  extension's module-level state.
- **Not covered, and said so**: behaviour behind a correct signature, and the vendored harness
  under `sandbox/google-verifiable-quantum-circuit-compilation/`.

Updated: `docs/SCAFFOLD_INVENTORY.md`, `tests/test_scaffold_inventory.py`, `CHANGELOG.md`.

## 2026-11-01 — The static-table rule becomes a gate that fails, not a report

- **`scripts/scope_scan.py` (new, audit-only).** `scripts/audit_static_tables.py`
  answers *what does this function return*; that is an inventory question, and it
  reports a units registry and a fabricated prediction alike. The new scan asks the
  narrower one: does the name promise more than the return delivers?
- **The rule, in three conditions.** The name matches a predictive prefix at a word
  boundary (`predict_*` … `rank*`, `score*`); no parameter reaches any return; and
  what comes back is either a literal the body never computed (with no parameter read
  anywhere in the body) or a module/class-level container from the census.
- **Two exclusions keep it usable, and both were forced by the first run.**
  `forecast_regime(x)` returning `"UP"`/`"DOWN"` is a predicate branching on its
  input, so the literal branch requires that the body never reads a parameter;
  `classify_modality(frame)` classifies and then maps the class through a table, so
  the table branch requires that *every* return is a lookup. A function that raises
  is a refusal and is never reported. Matching `propose` without a boundary had
  flagged thirteen `proposer_id` methods that return a constant string identifier.
- **MEASURED: two functions in the whole repository qualify**, one per package side —
  `apedp/applied_materials.py::search_candidates` (returns `REFERENCE_CANDIDATES`)
  and `apedp/evidence.py::rank` (returns `_RANK`). Both are tables, both are
  recorded with a reason, and `apedp_dyn` has none. The scan covers `apedp` and
  `apedp_dyn`, skipping vendored `_qualtran_src`, `benchmark_corpus*` and tests.
- **A check you can run.** `python scripts/scope_scan.py` exits `0` on the recorded
  tree, `1` when a finding appears that is not in
  `scripts/scope_scan_baseline.json` — or when a recorded one vanishes — and `2`
  when the scan cannot run. The baseline is not a hiding place: every entry needs a
  `why`, a row in `docs/SCAFFOLD_INVENTORY.md`, and agreement with the finding.
- **What it does not catch**, stated on the page rather than implied: a constant
  under a non-predictive name, one assembled by a helper, and a table read on every
  path with a computed key.

Updated: `scripts/scope_scan.py` (new), `scripts/scope_scan_baseline.json` (new),
`tests/test_scaffold_inventory.py`, `docs/SCAFFOLD_INVENTORY.md`, `CHANGELOG.md`.

## 2026-11-01 — Hamming ranking vs a position-weighted seed score, and where they part

- **Two rankings, both reported.** `predict_off_targets` keeps `top_sites` in flat Hamming order
  (mismatch count, then seed mismatches, then position) and adds `weighted_top_sites`, ordered by
  `position_weighted_score`: a stated linear ramp from 0.25 at the 5' end to 1.0 immediately 5' of
  the PAM, normalised to mismatch equivalents. One PAM-proximal mismatch scores 0.08 on a 20 nt
  guide, one distal mismatch 0.02.
- **Disagreement is counted, and split by what it means.** `ranking_comparison` reports
  concordant/discordant pairs, Kendall's τ, and — the part worth reading — within-band versus
  cross-band discords. A within-band discord refines a tie the count cannot break; a cross-band
  discord overrules the count, which is a weaker claim and is reported apart.
- **MEASURED WHERE THEY DISAGREE.** On a six-site panel (1–3 mismatches, chosen to separate the
  two): 11 concordant pairs, 4 discordant, τ = 0.467 — and **all four are cross-band**, the
  weighted score preferring a 3-mismatch distal site (0.069) to a 1-mismatch PAM-proximal one
  (0.08). On the two-site fixture the orders are exactly reversed, τ = −1: one proximal mismatch
  (0.08) versus two distal mismatches (0.066). On a same-count pair — 2 mismatches at (10, 11),
  0.106, versus 2 at (0, 19), 0.10 — the disagreement is within-band, τ = −1, which is the case
  the comparison exists for.
- **What this is not**: a cutting-efficiency model. The weights are the two constants above, not a
  fitted CFD matrix, and the Hamming count still decides the primary order. A uniform random 5 kb
  reference has no off-targets, so it produces no comparison at all (τ = `None`).

Updated: `apedp/applied_synthetic_bio.py`, `tests/test_scaffold_inventory.py`,
`docs/SCAFFOLD_INVENTORY.md`, `CHANGELOG.md`.

## 2026-11-01 — A genome-slice fetcher, so the off-target scan can search a real region

- **`apedp/ingestion/genome_region.py` (new, experimental).** `fetch_region` pulls a slice from
  the Ensembl REST API and `scan_region` runs the off-target scan on it, carrying the provenance
  into the result. Until now the scan required a reference string nobody could produce, so every
  run used a hand-written sequence.
- **The slice is validated before it is scanned.** Non-DNA characters, a body shorter or longer
  than the coordinates requested, a malformed region, or a range above the 1 Mb cap all raise.
  A truncated fetch must never read as "no off-targets found".
- **Provenance is returned, not asserted**: assembly, chromosome, 1-based inclusive coordinates,
  strand, provider URL, retrieval date and the SHA-256 of the sequence travel with the scan, and
  each reported site gains `genomic_start` in addition to its slice-relative offset.
- **RUN END TO END, TWICE, ON GRCh38.** The 2 kb slice `13:32930000..32932000` (2001 bp, SHA-256
  `3680bb31…`) yields 169 PAM-adjacent sites and exactly one exact match -- the guide's own locus at
  **32,930,010** -- with no mismatch, bulge or indel candidate. A 50 kb slice `13:32930000..32980000`
  (50,001 bp) yields 5,369 sites with the same outcome (fetch 8.6 s, scan 1.0 s). That is the
  arithmetic, not a disappointment: a 20-mer within four mismatches has probability ~3e-7 per site,
  so a 2 kb window should contain none.
- **Offline from here.** The 2 kb slice and its guide are frozen in
  `tests/fixtures/genome/grch38_chr13_slice.json`, and eight tests exercise the fetcher and the
  end-to-end path without a socket -- including the truncated-slice refusal that keeps a network
  failure from becoming a clean result.

Updated: `apedp/ingestion/genome_region.py`, `tests/test_genome_region_fetch.py`,
`tests/fixtures/genome/grch38_chr13_slice.json`, `docs/SCAFFOLD_INVENTORY.md`, `CHANGELOG.md`.

## 2026-11-01 — One-base bulges and single indels in the CRISPR off-target scan

- **The scan now tolerates one gap, and says which side it is on.** `predict_off_targets`
  adds `bulge` sites (the reference carries one extra base) and `indel` sites (the reference
  is missing one), each anchored at the PAM end because anchoring at the 5' end would charge
  an indel for shifting the whole window.
- **The old count keeps its meaning.** `predicted_off_targets` stays mismatch-only;
  `bulge_off_targets`, `indel_off_targets` and `total_off_targets` are reported beside it, so
  enabling a tolerance cannot silently change a number a caller already reads.
- **MEASURED CONTRIBUTIONS.** On the purpose-built references each tolerance recovers exactly
  the site it models (bulge: 1 site, 0 mismatches, gap at 9; indel: 1 site, 0 mismatches, gap at
  11), and the switches are independent. On the existing fixture (two mismatch sites, max 4) each
  adds 2 candidates. On a uniform random 5 kb reference (599 PAM-adjacent sites, max 4) both add
  **0** — a 19-base gapped alignment within four mismatches does not occur at that error budget,
  so the tolerances are not a blanket inflation of the count.
- **Still not modelled, and now named precisely**: `bulges_longer_than_one_base`,
  `multiply_gapped_alignments`, `rna_dna_duplex_energetics`. The previous flat
  `bulge_loops`/`indel_tolerance` entries are gone because the first step is now taken.

Updated: `apedp/applied_synthetic_bio.py`, `tests/test_scaffold_inventory.py`,
`docs/SCAFFOLD_INVENTORY.md`, `CHANGELOG.md`.

## 2026-10-31 — Demo leverage mapping replaced by cost-aware mean-variance sizing

- **`epistemic_position_sizing` is gone.** It mapped an epistemic status to a leverage number
  (1.0 / 0.5 / 0.25 / 0.0) and needed `POSITION_SIZING_IS_DEMO_ONLY` and a
  `position_sizing_is_demo_only` field to warn readers. `size_positions` replaces it: it solves
  `max w'μ − (γ/2)w'Σw − c|w − w_prev|₁` on `{w ≥ 0, Σw ≤ gross}` by proximal gradient with an
  exact capped-simplex projection, so the size depends on the forecast, the **covariance**, the
  **fee** and a stated risk aversion — not on the verdict.
- **The flag is retired.** The verdict now only gates admissibility: `INSUFFICIENT_EVIDENCE`,
  `NO_VALID_CANDIDATE` and `EXTRAPOLATION_UNJUSTIFIED` admit nothing, and
  `BOOTSTRAP_INSTABILITY` is refused even under `MULTI_HYPOTHESIS` because a sign that flips under
  resampling is not a direction to take. Nothing is disclaimed, because nothing is invented.
- **The L1 cost term is a no-trade band.** An edge thinner than the fee produces turnover 0 and
  cost 0 rather than a smaller position, which is what charging a trade means.
- **TWO OLD NUMBERS DID NOT SURVIVE THE RULE, AND BOTH ARE NOW PINNED AS MEASURED.** The
  `volatility_scaled_momentum` regime used to report a full 1.0 allocation under `UNIQUE_SUPPORT`;
  its holdout mean over that window is **negative** (−2.4% per bar, σ = 2.7%), so a long-only rule
  facing a 5 bps fee takes no position, and the test asserts gross 0.0 with the reason stated. The
  old 1.0 came from the verdict, not from the data: an identified model of the dynamics is not by
  itself a profitable long. And the collinear-factor experiment used to report a
  50% "diversified" size; the rule puts the position in a corner (share 0.00–0.00 across five
  shifted holdout windows), because with ρ = 0.999 the only thing separating the factors is a 5%
  mean difference. That is a sharper statement than the old one: the referee's non-identifiability
  concerns **which mechanism** drives the target, not which position to hold.
- **`FinanceComparisonResult` and `RealMarketDiscoveryReport`** carry the `SizingDecision` instead of
  a float; `gross_exposure` is a property. Sizing uses the holdout slice only, and the price-level
  benchmark sizes its holdout **returns** rather than price levels, which would have mixed units.

Updated: `apedp_dyn/sandbox/finance_l2_demo.py`, `apedp_dyn/sandbox/market_real_data_benchmark.py`,
`tests/test_dyn_finance_l2.py`, `tests/test_dyn_market_real_data.py`,
`tests/test_dyn_preregistered_real_data_benchmark.py`, `tests/test_scaffold_inventory.py`,
`docs/SCAFFOLD_INVENTORY.md`, `CHANGELOG.md`.

## 2026-10-31 — A real (small) patent corpus, and the prior-art engine run on it

- **`data/patents/corpus.jsonl` now exists**: five real publications fetched from Google Patents
  on 2026-10-31 — US6285999B1, US8965721B2, US7120566B1, US11488017B2, EP4334731B1 — each with
  its source URL, retrieval date and the SHA-256 of the HTML it came from. This closes the
  `refused` row in `docs/SCAFFOLD_INVENTORY.md`: the search machinery had nothing to search.
- **Three records carry a normalised claim expression; two do not.** US6285999B1 and
  US11488017B2 state no relation in claim 1, so they are counted as unsearchable rather than
  dropped, which is the distinction `NO_SEARCHABLE_CLAIMS` already existed to preserve.
- **`scripts/fetch_patent_corpus.py`** (new, audit-only) makes the corpus reproducible: verbatim
  claims under `data/patents/raw/`, the normalised expression in a separate curated file with the
  sentence it was read from, and `--verify` to re-fetch and compare digests. Its first version
  split each claim into one record per paragraph — Google breaks a claim across several
  `div.claim-text` blocks — so claim 1 of US6285999B1 appeared twice, truncated. Fixed by
  grouping fragments under their enclosing `div.claim`.
- **`scripts/run_patent_prior_art.py`** (new) audits six declared candidates against the corpus
  and **fails loudly** if the engine disagrees with any declared expectation; the run is
  recorded in `benchmarks/patent_prior_art/prior_art_run.json`.
- **A GRANULARITY DEFECT IS NOW MEASURED, NOT HIDDEN.** US8965721B2 claim 1 names "a ratio of
  said measured DC impedance to a reference DC impedance" and no functional form, so its
  expression is `Z_meas/Z_ref` — which matches *any* quotient. The probe `generic_ratio`
  (`x/y`) matches it, and that is recorded as a corpus defect: a coarse transcription inflates
  matches, and `KNOWN_IN_PATENT_CORPUS` must be read against *which* expression matched.
- **Three fetches were rejected, not silently dropped**: US2237713A, CN102403767B and
  DE102011079469A1 expose no claim text (claims are page images), which is a stated selection
  bias towards recent text-extractable filings.
- **`corpus_status()` no longer answers "not bundled".** It reports 5 records, 3 searchable, the
  jurisdictions, the retrieval date, the provenance description and the one-claim-per-record
  scope; the `FileNotFoundError` now names the bundled path instead of claiming the directory
  does not exist.

Updated: `apedp/patent_prior_art.py`, `scripts/fetch_patent_corpus.py` (new),
`scripts/run_patent_prior_art.py` (new), `data/patents/{corpus.jsonl,normalised_claims.json,
PROVENANCE.md,raw/*}` (new), `benchmarks/patent_prior_art/prior_art_run.json` (new),
`tests/test_patent_corpus_real.py` (new), `tests/test_scaffold_inventory.py`,
`docs/SEARCH_ENGINE_GAPS.md`, `docs/SCAFFOLD_INVENTORY.md`, `docs/DOCUMENTATION.md`,
`CHANGELOG.md`.

## 2026-10-31 — Static-table census for the apedp package

- **`scripts/audit_static_tables.py`** reports every function that returns a value read from a
  module- or class-level container literal rather than computed. Eleven modules qualify (ten rows
  in the inventory, two of them sharing one table), and the set is now pinned by a test, so a new
  table cannot appear without being recorded in `docs/SCAFFOLD_INVENTORY.md`.
- **THE FIRST RUN WAS WRONG IN BOTH DIRECTIONS, AND BOTH ARE RECORDED.** It reported every
  `to_dict` that returns a dataclass field with a tuple default (`ablation`, three `epistemic`
  records, `evidence`, `evidence_report`, `source_audit`) — state, not tables. Tightening the
  exclusion to *any attribute reached through `self`* removed them, but also removed
  `VectorizedDimensionalEngine.DIM_MAP`, which `get_unit_vector` reads through `self` while being a
  `ClassVar` table; the rule now keeps any `ClassVar` container literal even via `self`. An earlier
  tightening also dropped every class-body table by treating all annotations as fields, hiding
  `applied_materials.REFERENCE_CANDIDATES` and `external_benchmarks.CASES`; only annotations with no
  value or a `field(...)` default count as fields.
- **NONE OF THE ELEVEN IS A DEFECT**, and the page says so: a unit registry, a seed-page list, an
  evidence-tier map and a curated priors table are things that should be tables. They are listed so
  that `suggest_mechanistic_priors` cannot be read as a fitted posterior.

Updated: `scripts/audit_static_tables.py` (new), `tests/test_scaffold_inventory.py`,
`docs/SCAFFOLD_INVENTORY.md`, `CHANGELOG.md`.

## 2026-10-31 — Third sweep: a scenario that threw its own evidence away

- **AN AST SCAN OVER EVERY `apedp/**/*.py`**, for parameters never referenced and returns that are
  pure literal tables. Most hits were predicates whose literal verdicts are chosen by their inputs
  (control flow, not constants) and were discarded; the survivors are in
  `docs/SCAFFOLD_INVENTORY.md`.
- **`scenario_20_anscombe_metrology` COMPUTED A RESULT AND RETURNED A SENTENCE.** It built means,
  variances, correlations and quadratic curvature for all four Anscombe datasets, asserted that
  dataset 2 carried the distinguishing curvature, and then returned
  `{'anscombe_disambiguation': '...'}` — the claim without the evidence. The result now carries
  `curvature_a_by_dataset`, the threshold, the distinguishing set and the full statistics.
- **`PublicationText.ready_for_discovery` IS A GATE THAT NEVER OPENS, WHICH IS CORRECT AND WAS
  UNDOCUMENTED.** Extracted PDF prose is not measured data, so `False` is the right answer; a bare
  constant could be mistaken for a check that failed on this input. It now says so and exposes
  `readiness_reason`.
- **The honest constants are registered rather than changed**: `scientific_layers._const` (the only
  function in the package that ignores its parameters, by design — a null model),
  `model_synthesis._exp_templates` (a four-template search grammar), and
  `quantum_checks.amplitude_convention_note` ("Convention, not code"). `applied_defects.py` is
  `computed` throughout and is now on record as the last `applied_*` module.

Updated: `apedp/scenarios.py`, `apedp/source_ingestion.py`, `tests/test_scaffold_inventory.py`,
`docs/SCAFFOLD_INVENTORY.md`, `CHANGELOG.md`.

## 2026-10-31 — Off-target search is real, and still refuses without a genome

- **THE REFUSAL BECAME A SEARCH.** `CRISPRDesigner.predict_off_targets` now takes a reference
  sequence and finds every PAM-adjacent protospacer on both strands, Hamming-scores each against
  the guide, reports seed-window mismatches, positions and strand, and ranks by mismatch count.
  `find_protospacers` locates PAM motifs and takes the protospacer upstream rather than sliding the
  guide across the reference, because a nuclease only addresses protospacers adjacent to a PAM.
- **WHAT IT STILL REFUSES, AND WHY THAT IS NOT A REGRESSION.** `genome_size_mb` is gone: a genome's
  size cannot produce its off-target count, so passing an int raises `TypeError` and passing
  nothing raises `NotImplementedError`. The old constant `12` with `confidence: 'high'` is what
  that signature produced for every guide.
- **EXACT MATCHES ARE COUNTED APART FROM OFF-TARGETS.** The intended on-target locus is itself an
  exact match, so folding zeros into `predicted_off_targets` would inflate every guide by one.
  Exact sites are reported separately in `exact_sites`, and the ceiling is stated in
  `score_basis: hamming_mismatch_count` and in `not_modelled` (cutting efficiency, indel spectrum,
  bulge loops, chromatin, genome-wide scan). A site count is not a cutting frequency.

Updated: `apedp/applied_synthetic_bio.py`, `tests/test_scaffold_inventory.py`,
`docs/SCAFFOLD_INVENTORY.md`, `CHANGELOG.md`.

## 2026-10-31 — The equation harvester is brace-balanced

- **A FLAT CHARACTER CLASS BECAME A DEPTH COUNTER.** `_extract_formula_strings` stopped at the
  first closing brace, so any formula containing its own group — a fraction, a matrix, a nested
  subscript — came back as a fragment. The scan now walks from the opening brace of the
  displaystyle prefix to the brace that closes it, so `x^{2} + y^{2}` survives whole instead of
  returning as `x^{2`.
- **BOTH BRANCHES NOW AGREE ON SHAPE.** The alt-attribute branch used to return its formulas still
  wrapped in the displaystyle prefix while the displaystyle branch stripped it; alt candidates are
  now unwrapped, and the inclusion rule (`=`, length, `\begin`/`\align`/attribute noise) lives in
  one `_is_formula` used by both.
- **WHAT IS STILL LOSSY, STATED WHERE IT MATTERS.** Macros are not expanded, non-displaystyle math
  is skipped, and HTML entities are only partly decoded. `EXTRACTION_IS_LOSSY` therefore stays
  `True` and its meaning was narrowed in the module docstring, the inventory row and the tests: the
  strings are source text that must be re-parsed and verified, not mathematics that has been.

Updated: `apedp/ingestion/ingestion_web_equations.py`, `tests/test_scaffold_inventory.py`,
`docs/SCAFFOLD_INVENTORY.md`, `CHANGELOG.md`.

## 2026-10-31 — Corpus, audio and sizing: the second scaffold sweep

- **A SEARCH OVER A NONEXISTENT CORPUS NOW SAYS SO.** `apedp/patent_prior_art.py` is real
  provenance-checked machinery — it refuses unattributed records, refuses duplicate patent ids and
  refuses expressions that do not parse — but `data/patents/` does not exist, so no search can have
  been performed. `load_patent_corpus` already raised `FileNotFoundError`; the gap was that nothing
  let a reader tell an absent corpus from a completed search. `corpus_status()` returns
  `{"bundled": False, ...}` and the error names the missing corpus.
- **A PLACEHOLDER THAT RETURNED A NUMBER NOW CARRIES ITS FLAG.** `finance_l2_demo.epistemic_position_sizing`
  maps epistemic status to leverage and was marked DEMO-ONLY in a docstring — invisible to the caller
  reading `recommended_position_size: 1.0`. The flag is now `POSITION_SIZING_IS_DEMO_ONLY` and a
  `position_sizing_is_demo_only` field on every `FinanceComparisonResult`.
- **HARDCODED NARRATIVE REPLACED WITH OBSERVED OUTCOMES.** The demo's insight strings asserted
  "converting an abstention into UNIQUE_SUPPORT" and "reducing position size to 50%" regardless of
  what the run produced — and the module's own header already recorded that the 50% path does not
  fire. Both now report the statuses and the size actually observed.
- **Two modules needed no change, so they are registered instead.** `applied_electrical.py` is
  genuinely closed-form per-unit and per-cycle arithmetic with its own ceiling noted; the audio
  spectrogram transport is verified on synthetic chirps only and says so. Both are rows in
  `docs/SCAFFOLD_INVENTORY.md`, because "no change needed" is a finding worth recording.

Updated: `apedp/patent_prior_art.py`, `apedp_dyn/sandbox/finance_l2_demo.py`,
`tests/test_scaffold_inventory.py`, `docs/SCAFFOLD_INVENTORY.md`, `CHANGELOG.md`.

## 2026-10-31 — Scaffold tier inventory: a constant may not be returned as a prediction

- **THE AUDIT'S CENTRAL FINDING WAS NARROWER THAN "these modules are simple".** Being simple
  is fine; `tests/test_scaffold_inventory.py` exists because several of these modules returned
  a fixed number under a name that said *prediction*: `predicted_off_targets: 12` with
  `confidence: high`, `titer_prediction_g_l: 50`, `advantage_over_gpu: "1000x energy efficiency"`,
  and a "sea level rise" that was warming multiplied by 0.2 m/K.
- **THREE REFUSED, WHERE THE HONEST ANSWER NEEDED A MODEL THAT IS ABSENT.**
  `CRISPRDesigner.predict_off_targets` now raises: no reference genome, no mismatch scan, and a
  constant count is indistinguishable from a prediction. It joins `optimize_confinement` and
  `discover_targets(omics_data=...)`.
- **THE REST CARRY THEIR TIER.** `evidence` (`rule_screen`, `literature_lookup`,
  `planning_template`, `reference_spec`, `gc_heuristic`) plus `not_modelled` beside every verdict,
  so "drug-like: good" cannot be read as "permeable and stable". Two silent drops are also closed:
  `NeuromorphicArchitect.design_system` echoed no `requirements` and now returns them with
  `requirements_used: []`, and `MetabolicPathwayOptimizer` separates the literature yield, the
  assumed 0.85 recovery and a `titer_computed_g_l` of `None`.
- **NAME VERSUS BEHAVIOUR, FIXED IN THREE MODULES.** `ingestion_academic_apis.py` is twenty
  endpoint strings with `CONNECTS_TO_SERVICES = False` (no client, no auth, no request); the
  web-equation harvester states `EXTRACTION_IS_LOSSY` and has its brace truncation pinned by a
  test; `cluster_launcher.py` renders Slurm/Kubernetes/KubeRay artifacts with
  `SUBMITS_JOBS = False` — it never calls `sbatch`, `kubectl` or an API.
- **A DEAD BRANCH WAS FOUND WHILE PINNING THE HARVESTER.** Documenting the Wikipedia equation
  extractor's lossiness required demonstrating that it extracts anything at all, and the
  `{\displaystyle …}` branch matched nothing: the pattern was double-escaped, so `\\s*` searched
  for a literal backslash followed by a run of `s`, which real markup never contains. Every
  result the module ever produced came from the `alt=` attribute branch. Escaping is now
  generated with `re.escape` instead of typed, and both branches are pinned — the flat case
  returns intact, the nested case returns a fragment cut at the first inner brace.
- **THE GOLDBACH BANNER NOW SAYS WHAT IT IS NOT.** The verifier checks a finite range; Goldbach is
  already verified to 4e18, so a run at any limit this program reaches is sieve throughput, not
  frontier number theory. That sentence is now printed beside the result.
- **`docs/SCAFFOLD_INVENTORY.md` is the table**, row by row, including the modules whose own
  docstrings were already honest (`applied_materials`, `applied_fusion`, `se3_flow`, the two
  robotics transports) and the five components worth calling strong, described narrowly. The
  negative demonstration is cited by its path (`benchmarks/equation_discovery/qed_001_recovery.json`)
  rather than restated from memory.

Updated: `apedp/applied_synthetic_bio.py`, `apedp/applied_neuromorphic.py`,
`apedp/applied_climate.py`, `apedp/ingestion/ingestion_academic_apis.py`,
`apedp/ingestion/ingestion_web_equations.py`, `apedp_dyn/distributed/cluster_launcher.py`,
`rust/goldbach_verifier/src/main.rs`, `tests/test_scaffold_inventory.py` (new),
`docs/SCAFFOLD_INVENTORY.md` (new), `docs/DOCUMENTATION.md`, `CHANGELOG.md`.

## 2026-10-31 — The biopharma boundary, made checkable instead of argued about

- **THE ONE FALSE CLAIM IN THE AUDIT IS NOW TRUE, UNDER ITS OWN NAME.** The review said the
  repository's answer to p-hacking was a "Benjamini-Hochberg/FDR referee in
  `apedp_dyn/epistemic/`". It was not: every FDR hit in the tree was the referee's own error
  rate, a Bonferroni witness test, or a per-claim boolean. The real multiplicity instrument
  was `trial_ledger.py`'s Deflated Sharpe Ratio, which is Sharpe-based and does not transfer
  to a p-value panel. `apedp_dyn/epistemic/multiple_testing.py` is now an exact
  BH / Benjamini-Yekutieli / Holm implementation that reports the **effective threshold** —
  the largest raw p it would reject — because "we corrected for multiplicity" is not a
  checkable sentence and "the bar was 3.1e-4, not 0.05" is. Refuses NaN, out-of-range values,
  empty families and unknown methods by name.
- **A SILENT DROP BECAME A REFUSAL.** `UniversalDrugTargetDiscovery.discover_targets` accepted
  `omics_data` and never read it. Discarding the argument is the worst of the three available
  behaviours, because a caller who passes expression data has every reason to believe it was
  used. It now raises `NotImplementedError` naming why, and every returned target carries
  `evidence: literature_curated` and `computed_from_omics: false`.
- **THE SKETCHES DECLARE THEMSELVES.** `NON_PREDICTIVE_COMPONENTS` states, per callable, which
  inputs are read, which returned content is constant and why; each entry is pinned
  behaviourally. `ADMETPredictor` gained `modelled` / `not_modelled` (clearance, Caco-2, hERG,
  hepatotoxicity and nine more are explicitly outside it). `ProteinDesigner.design_enzyme`
  named Rosetta, AlphaFold and ProteinMPNN as strings and ran none of them; it now also returns
  `executed_tools: []` and `evidence: planning_template`.
- **THE ACHIRALITY LIMITATION IS NOW A SWITCH.** Distance-only energies cannot distinguish
  enantiomers — identical distances, identical energy, full stop. `MolecularForceField` accepts
  opt-in `chirality_terms` adding `kc/2 (V - V0)^2` on the signed volume, a pseudoscalar that a
  mirror negates. Default stays empty, which keeps the field achiral and the existing
  reflection-equivariance tests meaningful; supplying a term buys enantiomer separation at the
  stated cost of SO(3)-only symmetry. A chiral spring model is still not a force field anyone
  should compute binding affinities with, and `docs/BIOPHARMA_SCOPE.md` says so.
- **The scope is written down, verified row by row**, in `docs/BIOPHARMA_SCOPE.md` — including
  the three uses that are genuine (PK/PD law discovery via `apedp_dyn/ode/` and the SINDy/PDE
  proposers, biomarker screening under exact FDR, and adversarial audit of third-party claims)
  and the four capabilities real drug discovery would need that are absent here. No docking,
  no SMILES, no conformer generation, no retrosynthesis: the audit's central verdict stands.

Updated: `apedp_dyn/epistemic/multiple_testing.py` (new), `apedp/applied_biopharma.py`,
`apedp/applied_synthetic_bio.py`, `apedp_dyn/molecular/se3_flow.py`,
`tests/test_multiple_testing_and_biopharma_scope.py` (new), `docs/BIOPHARMA_SCOPE.md` (new),
`docs/DOCUMENTATION.md`, `CHANGELOG.md`.

## 2026-10-31 — The unpushed range could not be pushed: GitHub's 100 MB limit

- **THE PUSH WAS REJECTED, AND NOT BY ANYTHING NEW.** `stage` was 57 commits ahead of
  `origin/stage`, and `50df2b1` had committed four artifacts over GitHub's per-file limit:
  the 122 MB competition zip, the 122 MB `FeMoCo76o113e.fcidump`, and two 846 MB `.kmb`
  receipts — 1.9 GB of blobs. None of it was in the published range, so the repair is
  confined to local commits.
- **WHAT WAS REWRITTEN.** `filter-branch --index-filter` over `origin/stage..HEAD` removed
  those five paths from the 57 unpushed commits; one commit became empty and was pruned
  (57 → 56). Every hash in that range changed. `backup/pre-rewrite-stage` holds the
  pre-rewrite tip, so the old history is recoverable.
- **THE FILES STAY ON DISK.** Two are needed by code that reads them: the canonical
  `dqi_niters_222.kmb` that four `prepare_*_artifacts.py` scripts copy from, and the FCIDUMP
  the DFTHCSOS verifier scores against. Both are now ignored and must be fetched separately;
  `verifiers/dfthcsos/SCORING_README.md` pins the FCIDUMP's size and SHA-256
  (128,128,348 bytes, `e5a48af7…e1bd`), so a re-download is verifiable rather than assumed.
- **WHAT THIS DOES NOT DO: shrink the local repository.** The backup branch still
  references the old objects, so nothing is reclaimed until it is retired and `git gc`
  runs. The gain is that a clone can now exist at all.

Updated: `.gitignore`, `CHANGELOG.md`.

## 2026-10-31 — Two checkouts, one canonical, and a route map for the designer

- **`science` IS CANONICAL; THE SIBLING IS AN ARCHIVE.** The two checkouts share a common
  ancestor (`444c1fa`) and overlapping object stores, so this is a git-level fact rather
  than a naming preference: the archive
  (`/home/psalmprax/ALL_PROJECTS/science-quantum-discovery`, HEAD `ffa7b54`, 2026-10-27) is
  **5 commits ahead and 35 behind**, it has **no `apedp_dyn/vqc` at all**, and **8 of the 9
  files** its five commits touch are already byte-identical here.
- **ONE STRANDED ITEM CAN MISLEAD A READER, AND IT IS NAMED.** `ae02214` recorded the
  **rejection** of the DFTHCSOS 10-bit signed 4-bank candidate with its arithmetic
  (`error_1` 134.854 mHa against a 20 mHa threshold, `error_2` 44.889, **132 peak angle
  qubits against a 128 cap**, a 4,363-Toffoli gap between the literal formula and the
  claim). This checkout's copy of the same file still says
  `HOLD_FOR_ARITHMETIC_AND_RECEIPT_RECONCILIATION` — so read from here, a disproved
  candidate looks like an open question. Port command recorded, not executed.
- **Three other items, each with a disposition rather than a copy.** The uncommitted
  `min_margin_x` diagnostic in `verify_submission_invsqrt.py`; three OpenMP Davio searches
  (`search_davio_exact.c`, `search_shared_davio.c`, `test_davio_fast.c`) that must be
  checked for supersession against this checkout's `search_claims_pddd.c` and
  `search_mixed_polarity_pddd.py` before anyone ports them; and
  `scripts/measure_89q_executed_cost.py`, whose shebang hard-codes
  `#!/dev/shm/qvcc-venv313/bin/python` — a machine-local path the existing
  `/home/...` guard would not catch, so it is fixed at the source rather than by widening
  the guard.
- **The designer now has a route map** separate from its claims: `docs/VQC_DESIGNER_MAP.md`
  carries the specification-to-verified-schedule path, the 24 modules with line counts and
  roles, the public entry points, the eight records and seven producers, and the table of
  what the designer refuses and what it does not do (no non-Clifford synthesis, no invented
  topology, no 900-qubit routing, no hardware cost model).

Added: `docs/REPOSITORY_CANONICAL_DECISION.md`, `docs/VQC_DESIGNER_MAP.md`. Updated:
`docs/VQC_SYNTHESIS.md` (cross-references), `docs/DOCUMENTATION.md`, `CHANGELOG.md`.

## 2026-10-31 — Repo hygiene: the index stops carrying four copies of one 845 MB receipt

- **ONE CIRCUIT, FIVE FILES, THREE OF THEM IN THE INDEX.** `ls -l` reported five ~845 MB
  `.kmb` circuit artifacts. Four share a byte identity — `sha256 959471d0…04995` — the
  sample submission, the canonical iteration-222 receipt, and every
  `official_receipt_runs/**/dqi.kmb` copy — and the fifth (`dqi_niters_216.kmb`,
  `4ae20e1c…`) is a superseded sweep. The 122 MB `google-verifiable-…zip` is the only one
  of its kind whose entire contents are already extracted in the same directory.
- **WHAT WAS REMOVED FROM THE INDEX** (kept on the working tree):
  `sample_submission/dqi.kmb`, `dqi_niters_216.kmb` and the `.zip`, with explicit
  `.gitignore` rules so they cannot drift back in. **What was kept tracked:** the canonical
  `dqi_niters_222.kmb`, because four committed `prepare_{baseline,track2,track3,track4}_artifacts.py`
  scripts read it as their source, and `FeMoCo76o113e.fcidump` (122 MB), because
  `verifiers/dfthcsos/scoreinfra/reference.py` scores against it and nothing else can.
- **WHAT WAS DELETED FROM DISK** is only what a hash check proved redundant: the four
  `official_receipt_runs` copies were deleted **one at a time, each after `sha256` matched
  the canonical receipt byte for byte**. Nothing was deleted on the strength of its name or
  its size.
- **The limit is stated, not glossed, and no space claim is made.** The copies were
  hardlinks of one inode, so removing the names freed **nothing** on disk — `du` already
  counted the inode once, which is why `artifacts/` read ~1 GB while listing four 845 MB
  files. The gain here is entirely in the **index**: the repository stops carrying those
  blobs forward. `.git` history still contains them, so its 789 MB is unchanged;
  reclaiming that needs a history rewrite (`filter-repo`/BFG), which is destructive and
  rewrites every downstream clone — not done here, and separate by design.
- **What the 5.0 GB is, measured rather than guessed:** `target/` 1.7 GB (ignored cargo
  build output), `artifacts/` 1.9 GB (ignored receipts and checkpoints), `apedp_dyn/`
  1.2 GB (the vendored competition package, including the 846 MB canonical receipt and the
  122 MB FCIDUMP the verifier cannot work without), `.git` 789 MB. Only the last is history
  weight, and only a rewrite shrinks it.

Updated: `.gitignore` (three artifact rules with the reason), `CHANGELOG.md`.

## 2026-10-31 — What the study could detect, what the programme spent, and a sealed burn log

- **THE NULL HAD NO POWER NUMBER, AND NOW IT DOES.** `scripts/measure_quantum_signal_power.py`
  writes `benchmarks/quantum_compilation/quantum_signal_power.json` from the five tranche
  records alone — no panel, no holdout slice — by inverting the same
  `compute_deflated_sharpe_ratio` they use. The **minimum detectable Sharpe** is 3.19–3.47
  on 722 daily bars at 190–410 trials, 5.79–6.17 on 240 bars at 820, and **15.79 / 26.97 /
  41.06 / 57.05 / 111.84** at 1h / 1h / 30m / 15m / 5m. A true Sharpe of **1.0** has a
  detection probability of **0.11% at best** and ~0.00% everywhere else. So "nothing clears
  0.95" is evidence against effects above the MDS and no evidence about anything smaller:
  the design is a filter for fantasy-size claims, not a test of a modest edge. Both numbers
  are in the record so a null cannot be quoted without the threshold it was tested at.
- **THE FIVE TRANCHES ARE ONE MULTIPLICITY, CHARGED AS ONE.** `programme_charge` re-prices
  every row of every tranche against the **2,550 trials** the programme spent (190 + 310 +
  410 + 820 + 820), held at the dispersion that row's tranche used — the number that moves
  is the multiplicity the tranche structure hid, not a change of estimator. Hurdles rise
  (1d 3.618 → 3.973, 1h 16.847 → 18.502, 30m 32.896 → 38.614, 5m 78.379 → 86.079, 15m
  44.196 → 56.476), **no hurdle falls**, rows clearing the floor stay **0 as run and 0 at
  the programme charge**, and **no row becomes significant**. Pooling dispersion across
  tranches is carried per row as a separate sensitivity, because it can move a hurdle
  either way and is not the charge. What is still uncharged is stated: the choice of
  assets, resolutions and when each method change was made.
- **THE BURN LOG IS SEALED, AND A REFUSED RUN NOW COSTS NOTHING.** New
  `apedp_dyn/epistemic/holdout_seal.py` folds every line of the registry into a SHA-256
  chain and writes the terminal digest to `quantum_signal_holdouts.jsonl.seal`; `verify_seal`
  fails closed on truncation, an edited burn, or a rewind, and the digest is pinned in
  `tests/test_holdout_seal.py` (282 lines, 198,027 bytes, `ac422774…4b96`) so reverting both
  the log and its sidecar also fails. The producer verifies the seal and runs a
  **pre-flight** block — series against aggregates, accounting identity, trial count —
  *before* the ledger is opened, and records `preflight` with
  `slices_consumed_by_this_run: 0`. The ordering is asserted on the source, because
  tranche 5's first attempt lost twenty slices to a check that ran after the holdout.

Added: `apedp_dyn/epistemic/holdout_seal.py`, `scripts/measure_quantum_signal_power.py`,
`benchmarks/quantum_compilation/quantum_signal_power.json`,
`benchmarks/quantum_compilation/quantum_signal_holdouts.jsonl.seal`,
`tests/test_holdout_seal.py` (10), `tests/test_quantum_signal_power.py` (10). Updated:
`scripts/measure_quantum_signal_ledger.py` (seal verification, pre-flight, re-seal),
`docs/QUANTUM_SIGNAL_LEDGER.md`, `docs/DOCUMENTATION.md`.

## 2026-10-31 — Tranche 5: per-bar series, and Sharpe/DSR re-priced at every fee rate

- **THE BARS ARE STORED NOW, SO THE FEE REACHES THE DISTRIBUTION.**
  `quantum_signal_ledger_tranche5.json` ships
  `quantum_signal_series_tranche5.jsonl`: for every priced row, the per-bar net at the
  canonical 5 bps and that bar's `|Δpos|` — **38,248 bars over 140 rows** — so
  `net(r) = net(5 bps) − (r − 5 bps)·|Δpos|` holds bar by bar. `cost_sensitivity` now
  recomputes Sharpe and DSR at 5/10/20 bps from those bars (`sharpe_recomputed`,
  `series_max_sharpe_deviation`, `sharpe_by_resolution`, `dsr_by_resolution`), and a new
  check requires the 5 bps re-price to reproduce every recorded Sharpe: it does to **0.0**.
- **THE VERDICT DOES NOT DEPEND ON THE RATE.** MRK, UNH, BAC, COST, GLD, IWM daily and AXS,
  SNX, COMP, CRV at **5 minutes**, two walk-forward folds, **820 trials**: the daily median
  Sharpe falls −0.678 → −1.101 → −2.060 and the 5-minute one −51.574 → −94.831 → −160.350
  across 5/10/20 bps, positive rows 31 → 20 → 8, and **zero rows clear 0.95 at any rate**;
  hurdles 4.04 and 78.38, **20/20 fold slices burned** (registry 88).
- **THE BILL, BY RESOLUTION, AT 5 BPS.** Tranche 5's daily fees are **934% of |gross|** and
  its 5-minute fees **6,038%**; the earlier families read 119% (t1 daily), 93% (t2 daily),
  218% (t3 daily) and 142% / 363% (t4). Reported as
  `fee_share_of_gross_by_resolution` where gross is positive and
  `fees_over_absolute_gross_by_resolution` everywhere, because a share of a negative gross
  is undefined and pretending otherwise would be a fabricated ratio.
- **A REFUSED RUN STILL COSTS ITS SLICES.** The first tranche-5 attempt (JNJ, LLY, PG, V,
  AMD, NFLX; 1INCH, GRT, SAND, CHZ) decided its holdout and was then refused by
  `the_stored_series_reproduces_every_aggregate`, because the stored bars had been rounded
  to 12 dp and the sums drifted past 1e-9. The twenty slices were already burned
  (registry 68 → 88) and are not re-read: the record's assets are a disjoint set, and the
  cost of judging last is documented rather than hidden.

Added: `tests/test_quantum_signal_tranche5.py` (8),
`benchmarks/quantum_compilation/quantum_signal_ledger_tranche5.json`,
`benchmarks/quantum_compilation/quantum_signal_series_tranche5.jsonl`. Updated:
`scripts/measure_quantum_signal_ledger.py` (5-minute resolution, `--tranche 5`, per-bar
series storage, Sharpe/DSR re-pricing in `cost_sensitivity`, the series check),
`docs/QUANTUM_SIGNAL_LEDGER.md`, `docs/DOCUMENTATION.md`.

## 2026-10-31 — Tranche 4: walk-forward folds, a fee share per resolution

- **A SLICE CONSUMED ONCE, JUDGED ACROSS REGIMES.** `walk_forward_windows` cuts the
  post-warm-up span into `folds + 1` blocks; fold `i` trains on every bar before its block
  and tests on that block alone (1,000 bars, 2 folds, 30-bar warm-up:
  `(30,353,353,676)` and `(30,676,676,999)`). Tranche 4 runs it on unseen assets — CVX, HD,
  KO, PEP, BCH, LTC daily and VET, MANA, ZEC, XLM at 1 hour — so each base dataset is
  20 fold-datasets, each with its own slice id and its own burn.
- **820 TRIALS, NOTHING CLEARS 0.95, 20/20 FOLD SLICES BURNED** (registry 48): 120 threshold
  rows + 20 baselines + 680 continuous evaluations (34 per fold-dataset). Hurdles 3.62
  daily and 16.85 at an hour. Seven of the twenty folds make money on their test block and
  all seven are still rejected by the charge — `the_holdout_block_follows_its_training_window` replaced
  the tail-specific check, and `depth_comparison.charge_per_family` is now derived from the
  fits instead of the grid, so a continuous search is no longer mislabelled 24.
- **THE BILL, PER RESOLUTION, AT 5 BPS.** The sweep now reports `gross_by_resolution`,
  `fees_by_resolution`, `fee_share_of_gross` and `fees_over_absolute_gross` (defined even
  when gross is negative). At the canonical 5 bps the fee is the same order as the gross and
  usually larger: tranche 1 daily **119% of the whole gross**, tranche 2's 1-hour family
  **1,171%**, tranche 3's 30-minute **1,166%**, tranche 4's hourly **363%**. That is the
  arithmetic behind "a one-bar holding period cannot pay 5 bps".

Added: `tests/test_quantum_signal_tranche4.py` (5),
`benchmarks/quantum_compilation/quantum_signal_ledger_tranche4.json`. Updated:
`apedp_dyn/quantum_compilation/quantum_signal.py` (`walk_forward_windows`),
`scripts/measure_quantum_signal_ledger.py` (`--tranche 4`, fold windows, per-fold slice ids
and burns, fee-share report), `docs/QUANTUM_SIGNAL_LEDGER.md`, `docs/DOCUMENTATION.md`.

## 2026-10-31 — Tranche 3 (continuous optimiser, charged by iteration) and a fee sweep

- **THE CHARGE IS AN ITERATION COUNT NOW, NOT A GRID.** Tranche 3 fits unseen data (AMZN,
  JPM, XOM, WMT, XRP, DOT daily; XRP, TRX, LTC, BCH at 30 minutes) with coordinate ascent
  over `(alpha, beta)` per depth. `fit_continuous_map` writes **one trial per evaluation**
  with `selection_pool_size` set to the declared 17-per-depth budget, so **410 trials** =
  60 threshold rows + 10 baselines + **340 evaluations** (34 per dataset). Six probes were
  skipped at the `alpha > 0` boundary; the budget is still charged and the gap is disclosed
  as `hidden_trials = 6`. The search is genuinely off-grid: selected alphas include 0.8,
  1.2, 1.8, 0.2.
- **STILL NOTHING CLEARS 0.95, AND THE RULE STILL BITES.** Hurdles 2.26 daily and 32.9 at
  30 minutes; **all ten slices burned** (registry 28). AMZN@1d earns **+0.1704** with 124
  units of turnover and is rejected below its own hurdle.
- **A COST-MODEL SWEEP THAT DOES NOT TOUCH A SLICE.** `cost_sensitivity` re-prices every
  priced trial in all three records at 5, 10 and 20 bps. Fees are linear in turnover, so
  net is exact per row without reading a bar; Sharpe and DSR are **not** re-priced because
  they need the per-bar series, and the field says so. The family is net-negative at 5 bps
  already (−11.192 / −29.721 / −22.623), and raising the fee removes rows monotonically
  (tranche 2 drops from 13 positive rows to 3).
- **Two run failures fixed in place:** `TRANCHES` was declared above the panel tuples it
  referenced (NameError), and the continuous fitter indexed `net_returns` as if it held row
  dicts (TypeError) — both caught by the first run of the new tranche.

Added: `tests/test_quantum_signal_tranche3.py` (7),
`benchmarks/quantum_compilation/quantum_signal_ledger_tranche3.json`. Updated:
`apedp_dyn/quantum_compilation/quantum_signal.py` (`fit_continuous_map`,
`continuous_evaluations`, `CONTINUOUS_STEPS/PASSES/START`, `FitResult.optimiser`),
`scripts/measure_quantum_signal_ledger.py` (`--tranche`, 30-minute resolution,
`cost_sensitivity`), `docs/QUANTUM_SIGNAL_LEDGER.md`, `docs/DOCUMENTATION.md`.

## 2026-10-31 — Tranche 2: fresh slices, two depths, and a charge that bites before the sign

- **A SECOND TRANCHE, BECAUSE THE FIRST ONE'S SLICES ARE SPENT.** Tranche 1 burned eight of
  ten. Tranche 2 is a separate record on datasets never decided before: six unused daily
  assets (GOOGL, META, TSLA, DOGE, ADA, LINK) and four symbols at a new 1-hour resolution
  (ADA, AVAX, LINK, DOT). 310 disclosed trials = 60 threshold rows + 10 baselines + **240
  fits** (24 per dataset, both ansatz depths).
- **THE CHARGE BITES BEFORE THE SIGN DOES.** Under `holdout_decision` a slice is claimed only
  if the candidate trades, nets, **and** clears the expected maximum Sharpe of its own
  24-point search. ADA@1d's fitted map earns **+0.5649** with 216 units of turnover and is
  still rejected, because its holdout Sharpe is below the **2.193** hurdle; **all ten slices
  burned** (registry now 18). Nothing clears 0.95 either: best daily DSR 0.031, best 1-hour
  0.003.
- **DEPTH 2 DOES NOT PAY FOR ITS SEARCH.** The paired read evaluates each family's best
  candidate on the same holdout window from one read, each charged its own 24 trials: depth
  two wins **1 of 10** datasets (TSLA, +0.2664), mean holdout delta **−0.3134**, and it loses
  worst exactly where depth 1 had its largest gain (ADA, −1.1766). Neither family clears its
  hurdle anywhere.
- **EACH RESOLUTION IS ADJUDICATED ON ITS OWN HURDLE.** `significance_report` reports the
  daily and intraday verdicts side by side — rows, charge, dispersion, hurdle, best row,
  count clearing the floor — and is reproduced for tranche 1 by `--recompute-dispersion`,
  because a single pooled verdict hid that the two face 1.989 and 44.196. A print bug that
  read the registry total as this tranche's burn count was fixed with it, and the writer now
  resolves `--output` relative to the repository instead of crashing on a relative path.

Added: `tests/test_quantum_signal_tranche2.py` (6),
`benchmarks/quantum_compilation/quantum_signal_ledger_tranche2.json`. Updated:
`scripts/measure_quantum_signal_ledger.py` (tranche-2 datasets, 1-hour resolution,
paired depth read, `significance_report`, print fixes), `docs/QUANTUM_SIGNAL_LEDGER.md`,
`docs/DOCUMENTATION.md`.

## 2026-10-31 — The quantum signal map gets fitted, and the fit pays for itself at the holdout

- **A PARAMETERISED MAP WITH ITS OWN TRIAL LEDGER.** `ParameterisedQuantumMap` is
  `Ry(alpha_i x_i + beta_i)` with `CX(i, 0)` fan-in, so `<Z_0> = prod_i cos(alpha_i x_i +
  beta_i)` for any parameter values and the simulator stays checkable against a closed form.
  `fit_parameterised_map` chooses `(alpha, beta)` from a declared 12-point grid by in-sample
  Sharpe on the training slice and writes **one trial per grid point** into a
  `CandidateTrialLedger` with `selection_pool_size = 12`. The charge is the ledger's
  `effective_trial_count`, taken as its **growth** for that fit: a fit on the fifth dataset
  is not billed for the first four.
- **THE FAMILY IS CHARGED FOR THE SEARCH, NOT THE WINNER.** 190 trials = 60 threshold rows
  (2 maps x 3 thresholds x 10 datasets) + 10 baselines + **120 fits**, `var_trials =
  358.76`. Fitting is now priced the same way the threshold sweep is, which is the whole
  point: the previous record's 14 trials were the sweep alone.
- **TEN DATASETS AND A SECOND BAR RESOLUTION.** Six daily panels (AAPL, BTC, MSFT, NVDA,
  ETH, SOL) and four 15-minute snapshots (BTC, ETH, SOL, BNB), annualised at 365 and 35,040
  bars respectively. The intraday taker column is quote-denominated in this snapshot — its
  ratio to volume equals the bar's price on every symbol — so the producer converts it and
  **records the measured ratio** instead of assuming it (check: every ratio within 0.25 of 1).
- **A REAL HOLDOUT, CONSUMED AND BURNED.** 70/30 temporal split per dataset, parameters
  fitted on training bars only, one decision per dataset in a durable
  `FullLineageExperimentLedger` (`quantum_signal_holdouts.jsonl`). Decision rule, stated in
  the producer: pass when the holdout nets **and** changes position in the window. **8 of 10
  slices burned**; SOL@1d (+1.4285) and NVDA@1d (+0.4589) confirmed. AAPL is the row that
  rule earns: +0.2191 of holdout PnL with **zero** turnover — a short carried in from the
  training boundary — so it is rejected and burned. Two post-burn reuse attempts are
  recorded with the ledger's own `HoldoutContaminationError`.
- **WHAT THE NUMBERS SAY.** Five of six daily fits make money in sample and lose it out of
  sample (BTC@1d +1.4032 → −0.7678, ETH@1d +1.5073 → −0.4314). SOL's threshold map nets
  +2.7560 against its momentum baseline's +4.1581 on half the turnover. Every 15-minute row
  nets negative with fees of the same order as gross. No row clears DSR 0.95, and the record
  states why the hurdle is 52.06: the family variance is pooled across resolutions and is
  dominated by the 15-minute Sharpes, so the pooling over-charges rather than flatters.

- **THE HURDLE IS NOW RESOLUTION-LOCAL, AND THE FIX NEEDED NO NEW DATA.** Pooling daily and
  15-minute Sharpes into one dispersion made the deflation's variance 358.76 — dominated by
  the 35,040-bar rows, which span −66 to −10 — and left every DSR at 0.000. The dispersion is
  now taken within each resolution (0.5235 at 1 day, 258.53 at 15 minutes) while every row is
  still charged all 190 trials, so the daily hurdle falls from 52.06 to **1.989** and the
  15-minute hurdle is 44.196. Nothing becomes significant: the best daily row is the SOL
  momentum baseline at DSR **0.234**, then SOL's three-feature map at 0.207 and BTC@1d at
  0.022. The re-price is `--recompute-dispersion`, which reads only the trial statistics the
  record already stores and touches neither a panel nor a holdout slice.
- **A SECOND ANSATZ DEPTH, AND A REFUSAL INSTEAD OF A FALSE FORMULA.** `ParameterisedQuantumMap`
  takes `layers`, and `ANSATZ_DEPTHS = (1, 2)` puts both families in the fit grid (24 points
  per dataset). Depth two has **no product closed form**: the second rotation's `X` branches
  survive the first entangler, measured as **0.7146 against 0.0002** for depth-one-squared on
  the same input. `analytic` therefore refuses above one layer and the circuit is checked
  against `exact_marginal`, a dense-unitary build on the `qubit 0 = LSB` convention that shares
  no code with the simulator — it caught a real bug in the embedding on the first run.
- **A COST-AWARE HOLDOUT RULE.** `holdout_decision` claims a slice only if the candidate
  trades in the window, nets after fees, **and** clears `expected_max_sr` of its own search,
  so a wider grid has to prove more before the holdout may confirm it; failing burns the
  slice. This takes effect on the next tranche — the committed record's slices are spent, and
  no burned slice is re-decided to accommodate a new rule.

Updated: `apedp_dyn/quantum_compilation/quantum_signal.py` (`ParameterisedQuantumMap` with
`layers`, `fit_grid`, `ANSATZ_DEPTHS`, `exact_marginal`, `fit_parameterised_map`,
`holdout_decision`, `FitResult`), `scripts/measure_quantum_signal_ledger.py`
(10 datasets, two resolutions, fit charge, holdout consumption, 14 checks,
`--recompute-dispersion`), `benchmarks/quantum_compilation/quantum_signal_ledger.json`
(schema v2, re-priced), `benchmarks/quantum_compilation/quantum_signal_holdouts.jsonl`,
`docs/QUANTUM_SIGNAL_LEDGER.md`, `docs/DOCUMENTATION.md`,
`tests/test_quantum_signal_ledger.py` (33).

## 2026-10-31 — Experiment E gets a second graph source: Watts–Strogatz

- **A GRAPH SOURCE THAT IS NEITHER THE LATTICE NOR THE TREE.** Every Experiment E target
  was a square-lattice cluster or a seeded tree-plus-edges draw — both connected, both
  shaped by the construction the scheduler might be reading. `watts_strogatz_target`
  starts from a ring lattice of even degree `k` and rewires each ring edge with
  probability `beta` to a non-neighbour, so the draw has triangles, odd cycles and a
  degree distribution set by the rewire. `(k, beta, seed)` reproduces the edge set, and
  the edges are what the stabilizers encode, so the target's SHA-256 pins the draw.
- **MEASURED ON IT, NOT ASSERTED.** Running the Experiment E path on three cells —
  32 qubits `k=4`, 48 `k=4`, 64 `k=6`, all `beta=0.1` — gives 15 rows, every one `VALID`
  against its specification, and `graph_layers` attains the degree floor at every cell:
  **5, 5 and 8 layers** against floors of 5, 5 and 8, while the raster orders need 58, 82
  and 115. The scheduler saves a layer on two of the three (48- and 64-qubit cells), so
  the class-2 handling is exercised on this family too.
- **THE SMALL-WORLD CONTRAST IS A ROW.** At width 32, `k=4`, seed 0: mean clustering
  **0.500** at `beta=0` (the ring lattice, where exactly three of each vertex's six
  neighbour pairs are joined), **0.468** at `beta=0.1`, **0.140** at `beta=1`, all at mean
  degree 4.0 — so the middle row is the small-world regime rather than a random graph
  wearing its name.
- **THE CONSTRUCTOR TARGETS ARE SPLIT FROM THE REFUSAL TARGETS.** `watts_strogatz` is in
  a new `SYNTHESIS_GRAPH_TARGETS` tuple used by the five constructor families, not in
  `GRAPH_TARGETS`, because that tuple also scopes `graph_swap_gate` — a deliberately
  wrong circuit. Sharing it would have made a correct small-world circuit report a
  refusal row; the split is what the first test run caught.

Added: `tests/test_vqc_watts_strogatz.py` (7), `apedp_dyn/vqc/synthesis.py`
(`watts_strogatz_target`, `small_world_shape`). Updated: `scripts/measure_vqc_synthesis.py`
(`experiment_e_second_source`, wired into Experiment E as `second_graph_source` with three
checks).

## 2026-10-31 — A quantum signal the ledger can price, and the demonstration that it could not

- **THE DEMONSTRATION WAS A CONSTANT.** The `phase`-then-`h`-then-`cx` circuit printed as a
  "2-feature quantum trading signal" returns **0.0 for every input**: `phase` is diagonal and
  inert on `|0>`, and `h; cx` leaves the measured wire maximally mixed. Measured on nine
  input pairs, one distinct output; removing the phases changes nothing. The claim that the
  score feeds `run_ledger` is also not an API — the ledger builds its own seven components
  and takes no `signal=` argument.
- **A MAP WHERE EVERY FEATURE REACHES THE MARGINAL.** `Ry(pi*x_i)` on every wire, `CX(i, 0)`
  for every `i >= 1`, so `<Z_0> = prod_i cos(pi*x_i)`; the simulator is checked against that
  closed form on all 703 AAPL and 1,045 BTC bars per map (deviation `0.0`), and a phase-only
  encoding is pinned as the counterexample, so a silent collapse to a constant fails loudly.
- **THE EXECUTOR IS PROVEN TO BE THE LEDGER.** `execute_signal` re-implements
  `gross = pos*fwd`, `fee = FEE*|dpos|`, `net = gross - fee` and reproduces `run_ledger` row
  for row on the ledger's own score — **722 + 1,064 rows, maximum difference 0.0** in
  position, fee and net. Causality is measured rather than asserted: deleting the last 50
  bars leaves every earlier score bit-identical.
- **THE RESULT IS A FAILURE TO CLEAR THE FLOOR.** 14 disclosed trials (2 maps x 3 thresholds
  x 2 assets + 2 momentum baselines), 5 bps per unit turnover, deflated together
  (`var_trials = 0.3989`). The best trial is BTC with `H-MOM+CVD+H-BB` at threshold 0.2:
  gross +1.4103, fees 0.3605, net **+1.0498**, **DSR 0.320**. Nothing clears 0.95 and
  `significant_rows` is empty. On AAPL the 2-feature map's gross +0.3728 becomes net +0.0838
  because it trades 578 units of turnover against the baseline's 253 — the same edge with a
  bigger bill — and the 3-feature map's gross is negative on AAPL while positive on BTC, so
  the fixed map is not asset-robust.
- **BOUNDARIES STATED IN THE RECORD.** No trainable parameters, so no quantum learning is
  tested; the output is classically computable in `O(n)` arithmetic, so no advantage is
  claimed; two vendored panels, one bar resolution, no slippage beyond the fee.

Added: `apedp_dyn/quantum_compilation/quantum_signal.py`,
`scripts/measure_quantum_signal_ledger.py`,
`benchmarks/quantum_compilation/quantum_signal_ledger.json`, `tests/test_quantum_signal_ledger.py` (20),
`docs/QUANTUM_SIGNAL_LEDGER.md`. Updated: `docs/DOCUMENTATION.md`.

## 2026-10-31 — K9 in the measured table, the lookahead counterexample, and a one-point frontier

- **K9 IS AN EXPERIMENT E TARGET.** 36 edges, degree 8, and no 8-layer schedule exists
  (36 > 8·4), so the measured minimum is **9** — one above the degree floor, proved by the
  capacity count at zero search nodes — and the peel's **15** is shown against it. The
  scheduler's six saved layers now appear in the timing table beside the clusters instead
  of only in the colouring block. Total Experiment E targets: seven, all promoted.
  Routing stays the cluster corpus (90 placements / 72 comparisons): K9 is a complete
  graph rather than a lattice, so it is timed and decided but not placed.
- **THE LOOKAHEAD COUNTEREXAMPLE IS PINNED.** A 9-qubit balanced tree on a line: greedy
  **17** mapped layers, lookahead **19** — two deeper, with one extra swap. The 100-qubit
  rows improve under lookahead and this row is worse; both hashes and depths are stored,
  so the caveat is measured rather than narrated.
- **THE PLACEMENT FRONTIER COLLAPSES TO ONE POINT.** Sorting the sixteen placements by
  (mapped depth, CNOT-equivalent two-qubit gates), over distinct pairs, leaves the balanced
  tree on all-to-all alone: in this corpus the two costs are ordered identically and there
  is **no** depth/gate trade-off to adjudicate. The frontier size is reported as the result
  instead of being dressed up as a trade-off, and duplicates are collapsed so one point
  measured twice cannot look like two.
- **SABRE, SWEPT OVER FIVE SEEDS.** `experiment_d.sabre_seeds` runs Qiskit's router under
  seeds 0–4 on the six routed cases; all 30 circuits are decided `VALID` on the layout
  Qiskit reports. The interval is real: the tree on a ring spans **120–144** layers
  (spread 24), the lattice tree 68–73, the line tree 166–176, the lattice chain 188–192;
  a chain on a line or ring needs no routing and is seed-invariant. One SABRE run
  understates the depth by up to 24 layers here, so the sweep is the measurement.
- **NINE NEW CHECKS, 81 IN TOTAL.** The class-2 target is in the measured table with 9
  layers against a floor of 8 and a positive saving; the regression case is deeper under
  lookahead; the frontier is nondominated and partitions the distinct placements; the
  window and boundary checks accept exactly one minimum above the floor, on K9; and the
  SABRE sweep covers every case and seed, verifies every routed circuit on its own
  layout, recomputes the intervals, and never reports a partial sweep as a full one.

Updated: `scripts/measure_vqc_synthesis.py`, `benchmarks/quantum_compilation/vqc_synthesis.json`,
`tests/test_vqc_synthesis_record.py`, `docs/VQC_SYNTHESIS.md` §5.1 and §5.3,
`docs/VQC_PHASE0_CAPABILITY_INVENTORY.md` and `docs/DOCUMENTATION.md`.

## 2026-10-31 — A third referee on unitaries, a swept window, two bounds and an overshoot rate

Five additions to the synthesis record, each measuring something it previously asserted
or left as a single point.

- **QCEC, ON THE UNITARY QUESTION.** `third_referee` hands MQT QCEC the same circuits and
  asks whether they compute the same unitary up to one global phase — stricter than the
  state-preparation referee, so a wire permutation would pass one and fail the other. The
  reference is the specification's own preparation (`h` on every wire, one `cz` per edge
  in sorted order), built in the script, so no family is compared with itself. It decided
  all 11 pairs: the five 16-qubit schedules of one edge set are `equivalent` (one unitary,
  five schedules), the 25-qubit irregular graph and the **900-qubit lattice** are
  `equivalent`, and the three wrong constructions are `not_equivalent`. Every decided row
  agrees with the stabilizer tier. No pair was beyond its reach, so the record holds **no
  abstention** — stated rather than counted as a win.
- **THE LOOKAHEAD WINDOW, SWEPT.** `experiment_d.window_sweep` runs windows 2, 4, 8, 16
  and 32 over all eight placements: 40 rows of depth, swaps, two-qubit gates and wall
  clock. Window 8 reproduces the recorded lookahead rows exactly by SHA-256, all-to-all
  is window-independent (the control), and depth is **not** monotone in the window. The
  best-depth total is 969 against 1,003 at window 8: the balanced tree on the lattice
  gains at 32 (84 → 74), the chain prefers 2. No ordering between windows is claimed.
- **BOTH COUNTING BOUNDS, ON EVERY TARGET.** Each Experiment E target now carries the
  layer-capacity bound `Δ·⌊V/2⌋` and the matching bound `⌈|E|/ν⌉` with `ν` a maximum
  matching — never weaker than the capacity bound when there is no perfect matching.
  On the six class-1 targets neither fires, so their minimality rests on the search or
  the peel; on K9 the capacity bound is what refutes `Δ` (36 > 8·4). The block carries
  both a silence and a refutation, and the checks recompute both counts from the rows.
- **THE Δ+1 OVERSHOOT, MEASURED.** On 41 random graphs whose `Δ`-colouring this module
  proves, Vizing's construction spends the extra layer on **27 (66%)**, never more than
  one. On a class-1 graph the construction alone would overshoot two times in three,
  which is why the `Δ` search runs first.
- **K13 ADDED TO THE CLASS-2 BLOCK.** 78 edges cannot fit into 12 layers of 6, so the
  capacity bound refutes `Δ` at zero nodes: the peel's 15 layers become 13, another 2
  saved, taking the class-2 total to **14**.
- **TWELVE NEW CHECKS, 72 IN TOTAL.** Every third-referee row answered (a verdict or a
  reason); every decided row agreeing; the counts recomputing from the rows; the wrong
  constructions `not_equivalent`; the sweep covering every case and window; every window
  circuit `VALID`; window 8 identical to the recorded router; all-to-all invariant; both
  bounds on every target; neither firing; and the overshoot rates recomputing.

Updated: `scripts/measure_vqc_synthesis.py`, `benchmarks/quantum_compilation/vqc_synthesis.json`,
`tests/test_vqc_synthesis_record.py`, `docs/VQC_SYNTHESIS.md` §5.3 and §7,
`docs/VQC_PHASE0_CAPABILITY_INVENTORY.md` and `docs/DOCUMENTATION.md`.

## 2026-10-31 — Wide referee agreement: two widths, and the refusals decided too

The referee-agreement corpus was confined to circuits both in-package tiers can decide,
which is at most ten qubits — the dense tier's bound. So "two referees agree" was a claim
about a narrow band of widths: it said nothing about the widths this record actually
builds (100, 400, 900 qubits), and nothing at all about whether a second referee refuses
a wrong circuit.

- **TWO WIDE ROWS, EACH DECIDED TWICE.** `referee_agreement.wide.rows` holds the
  Experiment E minimum schedules for `cluster_20x20` (400 qubits, 760 edges) and
  `cluster_30x30` (900 qubits, 1,740 edges) — the same circuits the timing rows quote,
  compared by SHA-256. The stabilizer tier answers `VALID` on all 400 and all 900
  generators; **Stim** 1.16.0, a tableau simulator this package did not write, answers
  `VALID` with the same generators satisfied. Neither consults the other.
- **THE REFUSALS ARE DECIDED TOO.** The same three wrong constructions Experiment E
  refuses — a dropped edge, an added edge, one `cz`→`cx` — are decided again at 400
  qubits, and both referees answer `INVALID`. Stim's row carries the generators it
  failed, not just a count: a third party only ever seen accepting has not been shown to
  refuse anything.
- **A DISAGREEMENT STOPS THE RECORD.** A comparative verdict that differs from the
  stabilizer tier's now raises before anything is written, so `agree` is never `false` in
  a written record. Two independent referees contradicting each other is not a
  measurement to publish.
- **THE REFUSALS THAT CANNOT EXIST ARE ROWS.** `wide.dense_tier` stores what the dense
  tier answers at 400 and at 900 qubits — *"the dense tier is bounded at 10 qubits and
  this construction acts on 400"* — instead of leaving the absence to be inferred.
- **AN ABSTENTION IS NOT AN AGREEMENT.** If Stim is absent or refuses the circuit, the
  rows read `UNAVAILABLE`/`REFUSED` with the reason and `agree` is `null`; a check accepts
  either branch and requires the record to keep the two apart.
- **SEVEN NEW CHECKS, 60 IN TOTAL.** Both widths reached and `VALID`; every second
  opinion either a comparative verdict or an abstention with a reason; no verdict
  disagreeing anywhere in the wide block; both agreeing circuits are Experiment E's timed
  circuits by SHA-256; both referees refuse all three wrong constructions; Stim names the
  generators it failed; and the dense tier refuses every wide width with the bound in its
  message.
- **WHAT IT DOES NOT BUY.** Stim decides the prepared state, not the schedule; it is
  Clifford-only; and the IR-to-Stim bridge is this package's, so a shared translation bug
  would fool both. That is a limitation on the record, not a footnote in a module.

Updated: `scripts/measure_vqc_synthesis.py`, `benchmarks/quantum_compilation/vqc_synthesis.json`,
`tests/test_vqc_synthesis_record.py`, `docs/VQC_SYNTHESIS.md` §5.2, `docs/VQC_CORE.md`
§8.3.1 and `docs/DOCUMENTATION.md`.

## 2026-10-31 — Every schedule clock is the median of three repeats

The seconds columns were one run each. At 400 qubits a single decision takes 0.2-0.3 s
and at 900 qubits 1.3 s, and on a loaded host those numbers move by tens of percent
between runs — which is the size of the differences the columns are read *for* (whether
the deciding clock is about ten times the building one at every size, and whether one
schedule is cheaper to build or decide than another). A column that moves by the size
of its own effect is not a measurement to read a trend from, so each phase is now timed
three times and the row quotes the median, with all three samples kept beside it.

- **THREE REPEATS PER PHASE, MEDIAN RECORDED.** `TIMING_REPEATS = 3` in
  `scripts/measure_vqc_synthesis.py` times three rebuilds and three decisions per
  schedule. The rebuild hash is now checked on **every** rebuild, not only the first,
  so the third timing cannot drift onto a different circuit; the circuit the decisions
  are timed on is the last rebuild, which is hash-identical by that check.
- **THE SAMPLES STAY WITH THE MEDIAN.** Each row gains `synthesis_samples` and
  `verification_samples`, and `experiment_e.timing` records the statistic (`median`),
  the repeat count and why. The median is recomputable from the row, and a median of
  three is one of its own samples rather than an average — both are pinned.
- **WHAT MOVED.** At 400 qubits the three deciding samples span up to **43%** of their
  median (median span 11%), so the single-run columns were indeed noisier than the gap
  between schedules. The ladder now reads build **0.001-0.002 s** and decide
  **0.015-0.017 s** at 100 qubits, **0.020-0.032 / 0.233-0.288** at 400 and
  **0.101-0.160 / 1.269-1.369** at 900, and the 900-qubit deciding range narrowed from
  1.283-1.451 to 1.269-1.369. Each target still totals **0.14 s, 2.36 s and 11.34 s**
  (five schedules plus three refusals), and deciding is still about ten times building
  at every size.
- **TWO NEW CHECKS, 53 IN TOTAL.** One requires every timed row to carry exactly three
  samples per phase with the quoted clock equal to their median; the other requires a
  row to carry samples if and only if it carries a clock, so a `NOT_CONSTRUCTIBLE` row
  cannot report a number without a measurement behind it. `test_vqc_synthesis_record.py`
  pins the same three properties against the committed record, including that the
  median is one of its own samples.
- **Position unchanged.** Deciding is still flat across schedules at a given size — at
  900 qubits the 5-layer schedule costs 1.37 s and the 117-layer walk 1.31 s — so the
  median changes the precision of the ladder, not any conclusion drawn from it.

## 2026-10-31 — The cluster ladder is timed at a third lattice size: 30x30, 900 qubits

Experiment E timed its five schedules on a 16-, 100- and 400-qubit cluster, which is a
small checkable case and two points on a line. Two points cannot show whether the cost
of *deciding* a schedule grows faster than the cost of *building* it — the referee runs
one stabilizer check per qubit, so if anything does, it should — and the 4x4 target is
too small to extrapolate from. A 30x30 lattice adds the third point: 900 qubits, 1,740
edges, still degree 4, so the schedules are the same five and only the width changes.

- **A NEW TARGET, NOT A LONGER ROUTING TASK.** `CLUSTER_SHAPES` gains `(30, 30)`; the
  synthesis record now carries **44 rows across six targets, 30 of them `VALID`**, and
  the 900-qubit lattice reproduces the same structure as the others: `χ′ = Δ = 4`, so
  the minimum schedule is **5 layers** against the row-major walk's **117** (tree-first
  61, random 14) at the same 2,640 operations and 1,740 `cz` gates. Its depth is a
  property of the edge set, which is why adding the size changes every count and no
  conclusion.
- **THE LADDER IS MEASURED, AND THE RATIO IS FLAT.** Build **0.001-0.002 s** and decide
  **0.014-0.017 s** at 100 qubits, **0.019-0.030 / 0.234-0.267** at 400, and
  **0.105-0.165 / 1.283-1.451** at 900; the whole target (five schedules plus three
  refusals) takes **0.14 s, 2.14 s and 11.85 s** in that order. Deciding costs **7-14x**
  building at *every* size, so the worry the third point was added to test does not
  appear: what grows is the target, not the asymmetry between the two phases.
- **ROUTING STOPS AT 400 QUBITS, AND NAMES THE TARGET IT DID NOT ROUTE.** A new
  `ROUTING_MAX_QUBITS = 400` keeps the routing task at the three smaller lattices: the
  greedy router needs tens of seconds for one placement at 900 qubits and the task runs
  thirty placements per size. The record carries `not_routed` with the target and the
  reason rather than silently dropping rows, the **90 placements and 72 comparisons are
  unchanged**, and two new checks pin that no routed row exceeds the declared width and
  that the excluded target is named.
- **THREE NEW CHECKS, 51 IN TOTAL.** The ladder must be exactly `[100, 400, 900]` with
  the widest row at 900 and verification dearer than synthesis there; routing must stay
  inside its width with a non-empty exclusion list; and its coverage check now counts
  the routed targets instead of a hard-coded three.
- **The depth-histogram block of `vqc_external_comparison.json` follows the target.** It
  places six depths now, and the 900-qubit row is the one with the widest extrapolation:
  depth 5 against the 64-qubit cell (0.07x the target's width), still with **0 of 78**
  corpus rows at or below it. The single like-for-like row is unchanged — the 64-qubit
  irregular graph is still the only target the corpus also runs.

## 2026-10-31 — Experiment E's depths are placed in the Qiskit corpus's depth histogram

`VQC_SYNTHESIS` §5.3 reported five target depths and nothing to read them against. The
nearest comparison already in the repository is the 78-instance corpus Qiskit
synthesises in `vqc_external_comparison.json`, but that corpus runs at 12, 20, 32, 48
and 64 qubits while Experiment E's targets are at 16, 25, 64, 100 and 400. One width
overlaps. A single histogram of all 78 depths with the five targets dropped into it
would therefore have put a 400-qubit depth beside a 12-qubit one and called it a
comparison, so the block places each target against the cell at its own width, names
the nearest cell and the width ratio where there is none, and says which row is the
like-for-like one.

- **A NEW `qiskit_depth_context` BLOCK.** `scripts/compare_vqc_external_tools.py`
  computes it from the corpus rows it already measured (no new Qiskit call) plus the
  five Experiment E specifications rebuilt at their minimum schedule (`graph_layers`),
  each rebuilt circuit decided `VALID` by the same referee before its depth is quoted.
  It carries the whole histogram (bucket width 10, empty buckets included), the
  per-width cells with their depths, and one placement row per target.
- **WHAT IT MEASURED.** The 78 corpus circuits are **10-126 layers deep, median 18**
  (43 rows in 10-19, 20 in 20-29, then 4, 6, 3, and single rows at 94 and 126 — the
  three random stabilizer anchors are the entire tail, and every graph-state row is
  between 10 and 55). Against that: **0 of 78 rows** have a Qiskit circuit of five
  layers or fewer, so all three cluster depths sit below the corpus's own minimum of
  10 rather than below its median. The irregular targets are *inside* the corpus — 13
  layers at 25 qubits clears 5 of the 16 rows in the 20-qubit cell (31%), and the
  64-qubit target's 17 layers clear 5 of the 15 rows in its own cell (33%), whose
  median is 20.
- **ONE ROW IS A WIDTH MATCH, AND IT IS LABELLED AS THE ONLY ONE.** The corpus has no
  16-, 25-, 100- or 400-qubit cell, so those four rows record `width_match: nearest`
  with the ratio (0.75, 0.80, 0.64, 0.16) and `like_for_like` contains exactly
  `irregular_64q_seed23`. The cluster's five layers are the lattice's `1 + χ′` — a
  property of a bipartite edge set any floor-attaining schedule would report — so the
  record states the position without claiming a depth advantage, and points at the
  shared-corpus and routing tasks for the engine-against-engine measurement on the
  *same* instances.
- **FIVE NEW CHECKS**, including that the histogram accounts for every corpus row, that
  the placement recomputes from the cell it names, that only a width-matched target is
  ever called like-for-like, and that recomputing the block from the rows reproduces
  it. `tests/test_vqc_external_comparison.py` rebuilds every placed circuit and
  compares its depth and SHA-256, so a stale number fails rather than persisting.

## 2026-10-31 — `Δ + 1` is constructed, and `Δ` itself can be refuted by counting

The scheduler's `Δ + 1` fallback was a second search, so it could fail the same way the
`Δ` search did: two budgets exhausted meant `status: budget_exhausted` and the greedy
peel's layers. But Vizing's theorem is *constructive* — its proof colours an edge with
`Δ + 1` colours — so there was nothing to search for. Chasing that to the odd complete
graphs then exposed the other half of the problem: for `K9` and `K11` the `Δ` search
cannot finish either, so the family could only ever be reported as *undecided*. It is
now refuted by a count that costs no search at all.

- **VIZING'S PROOF IS NOW THE CODE.** `apedp_dyn/vqc/colouring.py` gains
  `vizing_colouring`, the fan argument in its Misra–Gries form: build a maximal fan at
  one endpoint, take a colour `c` free at the fan's root and `d` free at its tip,
  invert the maximal `c`/`d` path through the root, rotate the fan prefix that keeps its
  colour conditions, and place `d`. One fan, one inversion and one rotation colour one
  edge in a single pass, so `Δ + 1` colours always suffice and no search is involved.
  The construction verifies its own output — every edge coloured once, each layer a
  matching, at most `Δ + 1` layers — and raises `ColouringConstructionError` rather
  than return a partial colouring; `greedy_peel` now stands only as a last resort for
  that case, labelled `status: construction_refused`.
- **A LAYER IS A MATCHING, SO THE EDGE COUNT BOUNDS THE COLOURS.** `Δ` layers hold at
  most `Δ·⌊V/2⌋` edges; when the graph has more, no `Δ`-colouring exists and the graph
  is class 2 by counting (`status: delta_refuted_by_layer_capacity`, zero search
  nodes). It is checked before the search, so it decides what the search cannot — and
  it is cross-checked against that search on every graph of at most five vertices, so
  a refutation is never taken on faith.
- **THE ODD COMPLETE GRAPHS ARE NOW MEASURED, NOT UNDECIDED.** `K9` (36 edges, `Δ = 8`)
  and `K11` (55 edges, `Δ = 10`) are class 2, and their `Δ`-colouring search needs more
  than five million nodes (`K11` far more), so neither budget nor patience would have
  settled them. The count refutes `Δ` in one multiplication and the constructed
  `Δ + 1` schedule is minimal: **9** layers where the peel uses **15**, and **11**
  where the peel uses **15**. Both are `optimal=True`, identical at budget 1 and at
  budget 200, and both are now rows of the synthesis record beside `K5`.
- **THE RECORD'S CLASS-2 BLOCK COVERS BOTH KINDS OF PROOF.** `K5`, `C5`, `K9` and
  `K11` are refuted by layer capacity at zero nodes; the Petersen graph is the tight
  case (15 edges fit exactly into three layers of five), so its refutation is still an
exhausted search in 31 nodes. The five rows save **12** layers in total (2 + 6 + 4
  against a 15-layer peel on the odd complete graphs), and each row records its
  `proof_kind`, so "class 2" is never a single undifferentiated claim.
- **THE CONSTRUCTION IS CHECKED EXHAUSTIVELY, NOT BY EXAMPLE.** Pins run it on **every**
  simple graph of at most five vertices (1,094 graphs: validity of the partition and
  `len(layers) <= Δ + 1`), and on the families where fans actually rotate — `K9`, a
  9-cycle, Petersen, the 4-cube, `K5,5` — plus an injected construction failure that
  must fall back to the labelled peel. Outside the pins it was also run over `K2`-`K30`,
  cycles to `C60`, hypercubes to `Q6` and random regular graphs to 60 vertices with no
  invalid colouring.
- **EXPERIMENT E ITSELF DID NOT MOVE.** The `Δ` search, its pruning, its budget,
  `greedy_peel` and the `EdgeColouring` fields are unchanged, and the five Experiment E
  targets still save **0** layers (the peel already attained `Δ` there) with `K5`'s 2
  unchanged. The record `benchmarks/quantum_compilation/vqc_synthesis.json` was
  regenerated: two new class-2 rows, a new `colouring_sha256`, and a check that the odd
  complete graphs really are refuted by the bound rather than by a search that stopped
  early — **49 checks**, all true.
- Pins: `tests/test_vqc_colouring.py` (14), `tests/test_vqc_synthesis.py` (57),
  `tests/test_vqc_synthesis_record.py` (31).

## 2026-10-31 — MQT QCEC wired for the equivalence task, on pairs built to disagree

The external comparison named three tools and wired two. QCEC was the third, and the
record said why it was left out: *the comparison that would matter is a corpus where
the two disagree, and it does not exist yet*. Building that corpus was the task, and it
is what the numbers below are.

- **THE CONSTRUCTION IS THE ARBITER.** `apedp_dyn/vqc/equivalence_corpus.py` holds 18
  pairs in six families, each carrying the algebraic identity that fixes its answer —
  `X Y Z = i I`, `RZ(2*pi) = -I`, `SWAP = CX CX CX`, `Z_0 Z_1` stabilising a GHZ,
  `U_c = RX(theta) U_r` with a non-scalar rotation. Neither tool adjudicates the other:
  the corpus is a test of this package first (every declared expectation must be met on
  every scope, or the run refuses to write the record) and a comparison second.
- **THREE AXES, KEPT APART.** *Tolerance*: one angle from `1e-13` to `1e-3`, every point
  exactly inequivalent; this package's edge is at `1e-9` on the per-amplitude distance
  and QCEC answers `equivalent` at `1e-8`, so four ladder rows are recorded as a
  threshold difference rather than an agreement or a contradiction. *Scope*: a global
  phase is free standalone and `INVALID` under control, and the GHZ pair is one state
  from two unitaries — the state verdict is `VALID`, the unitary verdict `INVALID`, and
  QCEC's `not_equivalent` agrees with the second, which is why that row is recorded as a
  scope difference and not as a contradiction. *Abstention*: two 12-qubit non-Clifford
  pairs are refused here (the reason names `'t'` and the tier) and decided by QCEC, the
  12-qubit Clifford pair is decided by both, and neither abstention is scored.
- **IDENTICAL CALLS DO NOT ALWAYS GET THE SAME ANSWER.** From `1e-7` up, repeated QCEC
  calls on the same pair returned `equivalent`, `equivalent_up_to_global_phase`,
  `no_information` and `not_equivalent` — its simulation checker is randomised unless
  seeded, and its combination rule turns a probable verdict into no conclusion. So each
  ladder point is asked **15 times** and the record carries every criterion; a row whose
  criteria differ is `unstable`, not a verdict. The adapter is `qcec_equivalence` in
  `external.py`, given the tool's own method selection, its own default numerical
  tolerance (`2.27e-13` as declared — which is not the number its verdicts flip at) and
  no timeout. The circuits go over gate-for-gate, because Qiskit's OpenQASM 2 importer
  defines neither `swap` nor `cswap` and a text interchange there would have meant the
  other tool deciding a circuit this package did not hand it.
- **WRITING THE CORPUS FOUND A FALSE `VALID` IN OUR OWN REFEREE.**
  `adjudicate_clifford_equivalence(on="state")` reported a GHZ state and its
  `CZ`-flipped twin — overlap exactly zero — as the same state. Two conventions were
  folded together: the mask represents `X^a Z^b` while the argument is a string of Pauli
  matrices (`Y` is `i X Z`, not `X + Z`), and `X^a Z^b · X^c Z^d` carries a `(-1)^{b.c}`
  that adding the two phases drops, so a sign depended on which rows the elimination
  picked. `+YYX` was read as a stabilizer of an orthogonal GHZ state, and `Y_0 Y_1` was
  not found in the group of a Bell pair at all. Both are fixed in `stabilizer.py`
  (`_product_phase`, plus the Y count on both sides of the mask), and the new test
  enumerates all 64 Pauli strings of random 3-qubit states against the group built by
  multiplying generators **and** against the dense expectation.
- **ON THE RECORD — 18 PAIRS.** 10 agree, four differ only at the threshold, one is
  scope, two are our abstentions and one carries no single QCEC criterion; **no
  divergence is left unexplained** and none of our verdicts differs from what its
  construction requires. (A second run can move a ladder row between
  `DIFFER_BY_THRESHOLD` and `UNSTABLE` — that movement *is* the instability being
  reported, which is why the checks are structural and the rows carry the criteria.) `benchmarks/quantum_compilation/vqc_external_comparison.json`
  is regenerated at schema `v8` with **67 checks**, all true, and all three tools are
  now `compared`.
- **PINS.** `tests/test_vqc_qcec_equivalence.py` (33): the corpus expectations on all
  three scopes, the ladder's distances and both edges, the diagonal pair whose Z-basis
  probabilities agree to `1e-12` while the unitaries do not, the classifier's five
  outcomes and the one it must never invent, the summary recomputing and noticing a
  tampered verdict, the gate table being total over `GATE_SPECS`, the aggregation rule,
  and the record's own rows. `tests/test_vqc_stabilizer.py` (+7): the exhaustive group
  membership and the dense-expectation cross-check. Record pins updated for `v8`.

## 2026-10-31 — Peak memory for both sampling engines, on the samples the record holds

Throughput without memory is half a comparison, and the record said so in its own
limitations. It now measures both engines' peak memory — and the naive measurement does
not work, which is why the numbers are shaped the way they are.

- **ONE ENGINE PER INTERPRETER.** `VmHWM` never falls, so two engines measured in one
  process cannot be separated: the second can never come out below the first even if the
  first freed everything. Each engine therefore runs in a fresh child.
- **THE WINDOW IS CUT AFTER THE IMPORTS.** A whole child's peak is ~165 MiB of interpreter
  and numpy against a few megabytes of samples; quoting it would be quoting numpy. The
  child resets the peak counter (`/proc/self/clear_refs` writes 5, which the kernel
  honours) after the engine's imports and after rebuilding the specification, so the peak
  it reports is the sampling step's own; `floor_bytes` is reported beside it, so a reader
  can see what was excluded rather than take it on trust. Stim's library is loaded
  *inside* the floor, so an engine is never charged for a library it cannot run without.
- **THE MEASUREMENT IS BOUND TO THE RECORDED SAMPLES.** The child rebuilds the target
  from its printed stabilizers, refuses to measure unless the rebuild hashes to what it
  was sent, and returns the count hash of the samples it drew. Every memory row is
  checked against the count hash already in the record, so the memory belongs to these
  samples and not to a similar workload. (This caught a real bug while it was being
  written: GHZ declares a support, so a rebuild without it hashed differently and was
  refused — the refusal is the feature.)
- **ON THE RECORD — 10 WORKLOADS, BOTH ENGINES.** Median growth 0.8 MiB (this package)
  against 3.0 MiB (Stim) at 12 qubits, and 1.4 against 3.4 MiB across 16–32 qubits, over
  an import floor of 164 and 169 MiB charged to neither. The number that moves is the
  **outcome table**, not the width: a 32-qubit GHZ keeps two outcomes and grows 0.2 MiB,
  while a 32-qubit random state with 20,000 distinct outcomes grows 8.6 MiB here and
  10.0 MiB for Stim — where both engines are measuring the same 20,000-entry table. So
  memory is comparable and slightly in this package's favour on small tables, and is not
  an order of magnitude either way. Peak RSS is resident growth rather than bytes
  allocated, one run rather than a median, and no ceiling claim: Stim's 20,000-qubit
  workload is not measured.
- **PINS.** `tests/test_vqc_memory.py` (9): the counter is a real peak; the reset makes a
  window measurable where a monotone counter could only report the earlier value; a
  target round-trips hash-identically (support included) and malformed strings are
  refused; a measured engine's count hash equals the caller's own; Stim's library is in
  the floor; a platform that cannot reset the counter abstains with a reason; a child
  that sampled another specification is refused; two runs agree on everything but the
  bytes. Record pins (+2) and 3 new checks; record regenerated at schema `v7` with **56
  checks**, all true.

## 2026-10-31 — The sampling comparison above 12 qubits: exact marginals as the referee

The sampling comparison stopped at 12 qubits, and the reason was the referee, not
either sampler: the exact outcome distribution has `2**n` cells and that is enumerable
only up to 12. Above that width the record abstained. It no longer has to, and the
abstention is replaced by a narrower claim that is measured rather than asserted.

- **THE REFEREE IS THE SPECIFICATION'S MARGINAL, COMPUTED NOT SAMPLED.**
  `apedp_dyn/vqc/marginals.py` derives the exact Z-basis distribution of a chosen subset
  of wires by Gaussian elimination over GF(2) on the specification's own stabilizers:
  the distribution of a stabilizer state is uniform over an affine subspace, so a
  marginal is that subspace's projection, `2**|T|` outcomes and nothing exponential in
  the state's width. A 10-wire marginal of a 32-qubit state takes about a millisecond and
  never touches a sampler, a circuit or the sampling code path. Every marginal is
  checked against the rank it predicts before it is returned — support size and a sum of
  one — because a wrong marginal is a wrong referee.
- **THE VERDICT IS A TEST, NOT A BAND.** Each subset's projected samples are decided by a
  chi-square goodness-of-fit against its exact marginal, Bonferroni-corrected over the
  subsets tested (`min(1, m * p)`, recorded per subset). The survival function is
  implemented exactly enough to pin against textbook critical values and keeps SciPy out
  of the referee; a subset whose supported cells have fewer than five expected counts
  abstains as `UNRESOLVED` with the count that failed, because a chi-square there is the
  wrong test rather than a small error. The first design used a simulated TV band, and it
  was replaced for a measurable reason: a 1e-3 quantile cannot be estimated from a few
  hundred simulated draws, so the band moved by 15% between null seeds.
- **ON THE RECORD — 16, 24 AND 32 QUBITS.** Both engines are sampled on three targets at
  each width, 20,000 shots each, and decided on **190 subsets** (every adjacent pair of
  wires plus three 10-wire projections: contiguous, strided, seeded draw). All 190 are
  `CONSISTENT` on this package's sampler and on Stim's, and no subset disagrees between
  the two engines. GHZ's pair marginals keep their two-outcome correlation at `p = 1/2`,
  so a mis-wired qubit is caught by the pairs that contain it — pinned, not assumed.
- **WHAT IT DOES NOT DECIDE, IN THE RECORD'S OWN WORDS.** A marginal is a projection, not
  the distribution: a state differing only on wires no tested subset covers passes, which
  is why every row lists its subsets. And every graph state is uniform in the Z basis, so
  no Z-basis marginal separates one graph state from another — the ring-graph row is in
  the block to *show* that (34 of 34 marginals uniform), not to claim a verdict it cannot
  support. The full distribution is still only decided at 12 qubits.
- **PINS.** `tests/test_vqc_marginals.py` (26): every marginal against a dense partial
  trace of a state built by projecting a full-support vector onto the specification's
  eigenspaces — the route that caught the first version's Y-generator sign — signs
  honoured rather than read as labels, graph-state uniformity as arithmetic, a 32-qubit
  marginal under a cost bound (linear algebra, not enumeration), the chi-square function
  against textbook critical values, the Bonferroni product as arithmetic, a mis-wired wire
  caught by the subsets containing it **and only those**, `UNRESOLVED` for too few
  expected counts, a sample set from another specification refused, and the blind spot
  measured: samples of a different graph state pass. Record pins (+4) and regenerated at
  schema `v7`; **53 checks**, all true.

## 2026-10-31 — A third-party referee for the corpus: adjudication stops being self-referential

The shared corpus was still decided by this package's own specification referee — and
one side of that comparison is this package. A referee that both builds and judges is
exactly the weakness the record's limitations named, so the corpus is now decided twice.

- **THE SECOND OPINION IS STIM, AND IT DECIDES THE SAME QUESTION.**
  `apedp_dyn/vqc/foreign_referee.py` hands the circuit to Stim, simulates it as a
  stabilizer tableau and evaluates every generator of the frozen specification as a
  Pauli observable. The specification is the same signed Pauli strings both engines were
  given, so the third party is not deciding a looser property; a pure stabilizer state
  is the unique +1 eigenstate of `n` independent stabilizers, which is why no amplitude
  is computed and a 127-wire routed circuit is decided in about a hundredth of a
  second. **150 synthesis decisions and 150 routed decisions — ours and Qiskit's — all
  `VALID`, with zero disagreements against this package's referee.**
- **IT ABSTAINS RATHER THAN GUESSES.** Only `VALID`/`INVALID` are verdicts about a
  circuit. An absent Stim is `UNAVAILABLE`; a non-Clifford gate, a measured circuit, a
  classically conditioned gate or a width mismatch is `REFUSED`, each naming the
  operation that stopped it. Neither is scored as a loss, and a `REFUSED` is never
  counted as agreement either — the row records that the referees could not be compared.
- **WHAT IT DOES NOT SETTLE, IN THE RECORD'S OWN WORDS.** Stim decides the prepared
  state, not gate counts, depth or wall clock, which remain this package's accounting;
  it is Clifford-only, so the dense-tier syntheses elsewhere in this package are outside
  its reach; and the bridge from this package's IR to Stim's circuit is ours, so a
  translation bug would be invisible to both referees. A second third-party
  implementation that does not share our IR, and a third-party resource accounting, are
  still missing — `what_would_change_it` says so instead of claiming neutrality.
- **PINS.** `tests/test_vqc_foreign_referee.py` (11): same specification decided; sign of
  a generator honoured (|01> against |00> and against a negative-phase spec); `INVALID`
  for a dropped gate, a wrong target, a mis-wired qubit; `REFUSED` with the operation
  named for a `t` gate, a measurement and a width mismatch; `UNAVAILABLE` for an absent
  Stim and never a verdict; 127-wire routed circuit decided. Record pins (3 new): every
  matched row carries both referees' verdicts, the two agree, the summary recomputes
  from the rows, and the abstention paths carry their reasons. Record regenerated at
  schema `v6`; **42 checks**, all true.

## 2026-10-31 — The corpus routed by each engine's own router: the abstract gap does not survive

Every routed row in the record so far used *this package's* router, including on Qiskit's
circuit — which measures this package's router, not Qiskit's. The sampled corpus is now
routed by the router each engine ships, on the same recorded device, and the answer is
not the one the abstract rows suggest.

- **THE TASK, NOT A THEOREM.** All 75 matched corpus instances are routed onto the
  recorded `fake_sherbrooke` coupling map (127 qubits, 144 edges) by **Qiskit's own
  transpiler** — its initial-layout search, SABRE, seed 1234, optimisation level 1, its
  final layout kept — and by **this package's lookahead router**, which starts from the
  identity placement and appends its restore ladder. Both sides are decided by the same
  referee: Qiskit's against the target relabelled by its final layout, this package's
  against the target padded with idle wires. All 150 routed circuits are `VALID`.
- **THE 3.25× ABSTRACT DEPTH GAP IS ROUTED AWAY.** The sampled synthesis rows report the
  grammar shallower on every instance, median 3.25×. After routing, the median ratio is
  **0.97×** [quartiles 0.88–1.10, range 0.59–1.39, bootstrap interval on the median
  0.93–1.04], and Qiskit is the shallower of the two on **40 of 75** instances against
  this package's 31 (4 ties). Reported as a distribution, not a line: per family —
  random 3-regular 0.88× [0.82–0.98] (Qiskit wins 19 of 25), 4-regular 1.00×
  [0.91–1.06] (12/12/1), seeded irregular 1.05× [0.96–1.11] (this package wins 14 of 25) —
  and per width, where the two disagree with each other: Qiskit ahead at 12–48 qubits
  (0.95, 0.93, 0.96, 0.99) and this package ahead at 64 qubits (1.09× [1.01–1.13], 11 of
  15).
- **WHY, AND WHAT IS NOT PRICED.** This package's router swaps more in every family —
  350 against 206 at the median across the sample — because Qiskit searches an initial
  layout while this package starts from the identity placement on a sparse device graph
  and then restores. So the routed rows compare each tool *as shipped*, not the same
  optimisation problem on both sides; the pre-restore routed depth is the comparable
  number and the total is reported beside it. That asymmetry is in the record's
  limitations, not implied away.
- **IT IS A COUNTERWEIGHT, NOT A CONCESSION.** The synthesis section's claim was already
  that the depth saving is a schedule difference a rescheduling pass could produce; the
  routed sample is the measurement of that claim, and it is now quoted in the record's
  `advantage.why` beside the abstract gap.
- Pins: `tests/test_vqc_external_comparison.py` (38, two new routed-corpus tests) — the
  routed circuit is hash-identical to the one the synthesis row measured, each engine's
  own router is named on its own row with Qiskit's seed and optimisation level, both
  circuits `VALID`, and every routed median, quartile, win count, swap figure and share
  is recomputed from the rows per family and per width. Record regenerated; 36 checks,
  all true.

## 2026-10-31 — The shared corpus becomes a sample, and reports its spread

The external record's shared corpus was seven hand-picked instances. That fixed the
"only the favourable target" objection and left the next one standing: with one draw per
family, the row that happened to flatter an engine was the family's result. The corpus
is now a celled sample and the record reports a distribution instead of a row.

- **THE SAMPLE IS THE DESIGN, AND IT IS ARITHMETIC.** Three graph families (3-regular,
  4-regular, seeded irregular) at five widths (12, 20, 32, 48, 64 qubits) with five
  independent draws per cell, plus one random stabilizer anchor at each of 12, 20 and 32
  qubits: **78 instances, 75 matched, 3 abstentions**. Every seed is computed, not
  listed — `seed = SEED_BASE[family] + 10 * width + replicate` — every label names its
  cell (`regular3_32q_3`), and the whole corpus still rebuilds from seeds alone, which
  the record re-checks before it will write.
- **EVERY FAMILY IS REPORTED AS A DISTRIBUTION.** Per family and overall: the grammar's
  and Qiskit's own spreads (min, q1, median, q3, max, IQR, stdev), the spread of the
  per-instance ratio `qiskit/grammar`, the paired difference, the share of instances on
  each side, a bootstrap interval on the median ratio that resamples these instances (and
  says so), the ``(family, width)`` medians, and — for the graph inputs — the spread of
  edge counts and mean degree, so the sample's structure is visible too. The record's
  verdict text quotes the spread, not a single number.
- **THE RESULT SURVIVES THE WIDENING, AND THE WIDTH TREND EXPLAINS IT.** The two engines
  use the **same two-qubit count on all 75 matched instances** — one entangler per edge —
  so there is still no gate-count advantage to report. The grammar is shallower on **75
  of 75**, median ratio **3.25×** with quartiles 2.75–3.75 and range 1.50–5.00; by
  family 3.50× [3.00–3.75] (3-regular), 4.00× [3.20–4.40] (4-regular) and 2.65×
  [2.38–3.20] (irregular). The grammar's depth is fixed by the degree (4 layers cubic, 5
  quartic) and does not grow with width, while Qiskit's elimination tracks the qubit
  count, so the ratio widens from 12 to 64 qubits on an unchanged schedule. That is still
  a scheduling result a transpiler pass can reproduce, not synthesis superiority.
- **WHAT IT STILL DOES NOT SAY.** Five fixed width tiers, two generators, and instances
  at a width share that width: the spread is over these 78 instances, not over graph
  space, and the bootstrap interval is not a confidence statement about it. Both outputs
  are still decided by this package's own referee. Both limits are in the record's
  `limitations` and in `docs/VQC_CORE.md` §8.4.
- Pins: `tests/test_vqc_corpus.py` (11) and
  `tests/test_vqc_external_comparison.py` (36) — the new pins check that every cell was
  drawn at every declared width with the declared replicates, that the record's `design`
  block equals the rows it produced, and that each summary block's medians, quartiles,
  IQRs, ranges, win counts and bootstrap interval recompute from the rows. Record
  `benchmarks/quantum_compilation/vqc_external_comparison.json` regenerated at schema
  `v5`; all 30 checks true.

## 2026-10-30 — What routing does to the cluster schedule's advantage

Experiment E's schedules were measured on a machine where every qubit may talk to every
other. A depth advantage is a *logical* number until the circuit is routed, so the five
schedules were placed on a line, a ring and the cluster's own lattice — by both routers,
at 16, 100 and 400 qubits — and the gap was measured again after the SWAPs were paid for.

- **90 PLACEMENTS, 72 COMPARISONS, NONE INVERTED.** Every schedule was placed on every
topology by both routers at all three cluster sizes, and every placed circuit was
decided `VALID` by the same referee (the restore ladder keeps it the same unitary on the
same wires). The minimum schedule (`graph_layers`) is compared to each alternative twice:
the layers it was deeper logically and the layers it is deeper after routing.
- **ON THE LATTICE THE ADVANTAGE IS UNTOUCHED, AND THAT IS THE CONTROL.** The cluster's
edges *are* lattice edges, so the identity placement respects all of them: every family
needs **0 swaps**, keeps its logical depth, and the surviving fraction is exactly
**1.00×** for all 24 lattice comparisons.
- **ON A LINE OR RING IT GROWS.** Routing is not free for the minimum schedule either —
12,316 swaps at 400 qubits on a line, longest hop 40 — but the localized orders pay far
more: raster, column-major and tree-first all need **144,400** swaps with a longest hop of
**362**, because they emit gates across the whole register in an order no line can serve
locally. The mapped gap therefore exceeds the logical one: across all 72 comparisons the
surviving fraction is **1.00×-219×, median 4.67×**, and the minimum schedule is still the
shallowest circuit after routing — at 400 qubits on a line, 229 mapped layers against
column-major's 1,520 (a logical gap of 72 layers becomes 1,291). The strongest
alternative is column-major on a line and raster on a ring: still 7.8×-23.1× further
behind than it was logically.
- **WHAT IS ASSUMED.** Identity initial placement (no layout search), program order
routed as written (no reordering of commuting gates), the restore ladder counted in the
mapped depth, both routers reported, no device calibration or error rate. Those are in
the record's limitations, not implied away.
- Pins: `tests/test_vqc_synthesis_record.py` (31, including the 90-placement block and
the recomputation of every surviving fraction). Record
`benchmarks/quantum_compilation/vqc_synthesis.json` regenerated; **48 checks**, all true.

## 2026-10-30 — Experiment E at 400 qubits: the schedule's own wall clock

Experiment E stopped at 100 qubits, where building and deciding a schedule is too fast
to time on purpose. The largest target is now a **20x20 lattice cluster: 400 qubits,
760 edges**, and every schedule records what it cost to build and to decide.

- **THE TARGET IS THE LATTICE, CHECKED ARITHMETICALLY.** `cluster_state_target(20, 20)`
yields 400 stabilizers and `2*400 - 20 - 20 = 760` edges, maximum degree 4, so the
proven schedule floor is 4 `cz` layers. All five schedules are `VALID` at 400 qubits
(400 stabilizers checked by the same referee), and the two wrong constructions plus the
`cz`→`cx` swap are still `INVALID` at this width, so the refusals are not an artifact of
small graphs.
- **DEPTH AT 400 QUBITS.** The minimum schedule is **5** (4 layers) against **77** for
the row-major and column-major walks (76 layers), 41 for the breadth-first tree and 14
for the seeded random order — at exactly the same 1,160 operations and 760 two-qubit
gates. That is fifteen times shallower than the row-major walk, and it saves 0 layers
against the greedy peel, which had already attained `Δ` at this size too.
- **PER-SCHEDULE WALL CLOCK, NOT A HEADLINE NUMBER.** Each row now carries
`synthesis_seconds` (build the circuit from the specification) and
`verification_seconds` (the stabilizer referee deciding it). At 400 qubits: 0.019/0.231
raster, 0.021/0.263 column-major, 0.032/0.221 tree-first, 0.020/0.222 for the minimum
and 0.022/0.232 random. Building is cheap — one `h` per qubit and one `cz` per edge —
and deciding costs about ten times more because the referee checks all 400 stabilizers;
the whole 400-qubit target (five schedules and two refusals) takes 2.3 s. The five
schedules' times are within a few percent of each other, so depth is not bought with
time here — and no timing advantage is claimed for the minimum either, because every
family emits the same 1,160 operations and every decision is the same 400 stabilizer
checks.
- **THE TIMED CIRCUIT IS THE RECORDED CIRCUIT.** The timings are taken on a circuit
rebuilt from the same family and target, and the script **refuses to write the record**
if that circuit does not hash to the row it is attached to. One run, one process, a
warm rebuild on this host, moving by tens of percent between runs under load: the
record's limitations say all of that rather than implying a benchmark against another
engine.
- Pins: `tests/test_vqc_synthesis_record.py` (30, including the 400-qubit target and
the two seconds columns). Record `benchmarks/quantum_compilation/vqc_synthesis.json`
regenerated; **43 checks**, all true.

## 2026-10-30 — The `cz` schedule is now a proved minimum, not a greedy peel

Experiment E could only say the greedy peel *attained* the degree floor on the four
targets it was measured on, and it explicitly declined to claim optimality in general.
It now does: `graph_layers` computes a **minimum edge colouring**, and every schedule
it reports carries the argument for its size.

- **A SCHEDULE IS AN EDGE COLOURING.** Commuting `cz` gates share a layer exactly when
  their edges do not touch, so a layer is a matching and the minimum number of layers is
  the edge chromatic number `χ′`. `apedp_dyn/vqc/colouring.py` decides it: Vizing gives
  `Δ ≤ χ′ ≤ Δ + 1`, a complete backtracking search (colour-symmetry broken) finds a
  `Δ`-colouring -- optimal by the degree bound because a maximum-degree vertex needs one
  layer per incident edge -- and only when that search **exhausts its space** is a
  `Δ + 1` colouring constructed, with the exhaustion recorded as the proof that the
  graph is class 2. "Proven minimum" and "the best I found" are separate fields, and the
  search carries a node budget because deciding the `Δ` case is NP-complete; a
  budget-exhausted run returns the peel labelled `optimal=False`.
- **IT SAVES NOTHING ON EXPERIMENT E, AND THE RECORD SAYS SO.** On all four targets the
  old peel already attained `Δ`, so `total_layers_saved` is **0** (4, 4, 12 and 16
  layers; depths 5, 5, 13 and 17 unchanged) and every target is listed by name in
  `targets_where_greedy_was_already_optimal`. The replacement buys proof, not depth,
  there — presenting it as an Experiment E improvement would be false.
- **CLASS-2 GRAPHS ARE WHERE THE OLD SCHEDULE LOST.** Three graphs with `χ′ = Δ + 1` are
  now built and decided too: the odd cycle `C5` (3 layers, the peel was already 3), the
  Petersen graph (4, peel 4) and the complete graph `K5` (5 against the peel's **7**, so
  **2 layers saved**). The record proves the extra layer by exhaustion rather than
  assuming Vizing's slack is needed, and all three circuits ride the same referee
  (`VALID`, stabilizer tier).
- **GRACEFUL WHEN IT CANNOT FINISH.** A search that hits its budget returns the greedy
  peel with `status: budget_exhausted` and the reason in words; nothing is called minimal
  without a lower bound that matches. `greedy_peel` stays exported precisely so the
  saving is a measured column (`greedy_cz_layers`) rather than an assertion.
- Pins: `tests/test_vqc_colouring.py` (8), `tests/test_vqc_synthesis.py` (57),
  `tests/test_vqc_synthesis_record.py` (29). Record
  `benchmarks/quantum_compilation/vqc_synthesis.json` regenerated with the scheduler
  block, its per-target savings, the class-2 rows and `colouring_sha256`.

## 2026-10-30 — Qiskit's own transpiler on a recorded device map: a split routing result

The record's oldest routing caveat was that Qiskit's circuit was routed by *this*
package's greedy router. That comparison measured our router, twice. The device task
now gives Qiskit a real coupling map and its own transpiler.

- **A RECORDED DEVICE, NOT A SYNTHETIC LATTICE.** `qiskit_routing` builds a
  `CouplingMap` from the `fake_sherbrooke` snapshot (127 qubits, 144 edges, edge list
  pinned by SHA-256), transpiles with SABRE, re-imports through QASM 2, and reports
  both the routed depth measured here and Qiskit's own `depth()`. Four logical
  circuits: GHZ chain, GHZ balanced tree, ring 100, and Qiskit's own 5,149-gate GHZ.
- **LAYOUT, DECIDED NOT ASSUMED.** Qiskit keeps its final layout as metadata, so its
  circuit prepares the target *up to a permutation*. `relabelled_target` moves the
  specification onto the wires the device will hold it on, and the same referee
  decides it; a non-injective layout is refused. All four Qiskit circuits and all
  eight placements are `VALID`.
- **THE RESULT IS SPLIT, AND THAT IS THE FINDING.** Qiskit wins **2 of 4** routed-depth
  comparisons against each router (median ratio 1.47× against lookahead, 1.32× against
  greedy). It is shallower on the balanced tree (83 vs 95) and on its own 5,149-gate
  circuit (1,416 vs 7,253), where the grammar's edge ordering is exactly the schedule
  the router must undo; this package is shallower on the chain (116 vs 239) and the
  ring (117 vs 251). Both routers and both routed and total depths are reported, so the
  restore cost is visible rather than folded into one number.
- **WHAT IS STILL NOT CLAIMED.** One SABRE seed and optimisation level (stochastic
  router, one run); SWAPs priced as one two-qubit gate on both sides; no calibration,
  error model or measurement. MQT QCEC remains `UNAVAILABLE` with a reason.
- Schema is now `apedp.vqc.external_comparison.v4`; `qiskit-ibm-runtime` joins the
  `compare` extra for the device snapshots. Pins:
  `tests/test_vqc_external_comparison.py` (35).
  `benchmarks/quantum_compilation/vqc_external_comparison.json` regenerated.

## 2026-10-30 — Stim wired: the same specification, the same shots, the exact distribution as referee

Stim was the record's largest declared abstention. It overlaps with **execution**, not
synthesis, so the matched task is sampling: hand both engines the same signed
stabilizer strings and the same 40,000 shots, and decide each sample set against the
exact distribution rather than against the other engine's output.

- **BOTH ENGINES, ONE SPECIFICATION.** `stim_sampling` builds a `stim.Tableau` from the
  same stabilizers, synthesizes a preparation circuit and compiles a sampler;
  `apedp_sampling` clones this package's tableau and measures qubit by qubit. Both
  adapters record preparation and sampling clocks separately, because they prepare
  differently and folding the clocks together would hide which part is compared.
- **THE REFEREE HAS TEETH.** At 12 qubits the exact outcome distribution is
  enumerable, so `adjudicate_sampling` reports the total-variation distance against a
  band simulated from that distribution (a chi-square band would have thousands of
  empty cells), plus max-|z| and out-of-support counts as diagnostics. `sampling_pair`
  compares the two samplers under a null where **both** are correct, so "they agree"
  is not inferred from sampling noise. A width above the exact-enumeration limit is
  `SamplingUnsupported`; a skewed sample set is `INCONSISTENT`.
- **MEASURED.** GHZ 12, ring 12 and the corpus's random stabilizer 12, 40,000 shots
  each: every sample set `CONSISTENT` (TV 0.0000 / 0.1297 / 0.0428 for this package,
  0.0025 / 0.1247 / 0.0460 for Stim, all inside their bands) and all three pairs
  `AGREE`. Median throughput differs by about **three orders of magnitude** (thousands
  against millions of samples per second) — recorded as an implementation comparison
  (Python per-shot tableau clone against a compiled C++ sampler), explicitly not a
  ceiling for either engine and not a claim about Stim's 20,000-qubit published
  workload.
- **THE ABSTENTION IS STILL AN ABSTENTION WHERE IT APPLIES.** MQT QCEC is still
  `UNAVAILABLE` with a reason; Stim is listed as compared only for the sampling task,
  and its synthesis abstention is unchanged. `stim` is pinned in the record's
  reproducibility block (1.16.0).
- Schema is now `apedp.vqc.external_comparison.v3`; `stim>=1.12` joins the `compare`
  extra. Pins: `tests/test_vqc_external_comparison.py` (30).
  `benchmarks/quantum_compilation/vqc_external_comparison.json` regenerated.

## 2026-10-30 — A shared corpus: the external comparison now runs targets neither engine was written for

The external record's own `what_would_change_it` named a shared corpus as the first
condition for a real comparison. It was the right condition: the GHZ and ring rows
compare a grammar that *knows* those targets against a general synthesizer, and the
single random-state row only shows where the grammar abstains.

- **SEVEN INSTANCES, NO BESPOKE CONSTRUCTION ON EITHER SIDE.** `apedp_dyn/vqc/corpus.py`
  fixes a corpus reproducible from seeds alone: random connected 3- and 4-regular graph
  states drawn with the pairing model (16, 20, 24 and 32 qubits), one seeded irregular
  graph, and two random stabilizer states with no graph structure. A draw with a
  self-loop, a repeated edge or two components is rejected and redrawn, not repaired,
  so a "3-regular" target really is one.
- **THE HONEST RESULT: A GATE-COUNT TIE.** On all five graph states both engines use the
  **same number of two-qubit gates** — one entangler per edge, which is what a graph
  state costs — and the grammar's only saving is depth (5/7/6/5/9 against
  15/16/15/16/21), because it peels the edge set into parallel matchings while Qiskit
  emits the edges in sequence. A transpiler scheduling pass can produce the same
  reordering, so the record calls it a schedule difference and does **not** quote the
  2.5× median depth ratio as a synthesis advantage. The two random stabilizer rows keep
  the asymmetry in the same table: the grammar abstains, Qiskit synthesizes, and the
  abstention is unscored.
- **CHECKED, NOT TYPED.** The record refuses to write unless the corpus rebuilds hash
  for hash from its seeds, every row's two engines are on one `target_sha256`, every
  matched circuit is `VALID` under the same referee, no abstention carries metrics, and
  the summary (win counts, medians, totals) recomputes from the rows. Schema is now
  `apedp.vqc.external_comparison.v2`.
- **THE MISSING PIECE IS NAMED, NOT PAPERED OVER.** Both outputs are still decided by
  this package's own referee, and seven instances are not a distribution over graphs;
  the record's limitations and `what_would_change_it` say both, with a third party's
  verifier and per-tool routers still outstanding.
- Pins: `tests/test_vqc_corpus.py` (6), `tests/test_vqc_external_comparison.py` (24).
  `benchmarks/quantum_compilation/vqc_external_comparison.json` regenerated.
- **Correction (same day, below):** the depths quoted here (5/7/6/5/9) were measured
  under the greedy matching peel. `graph_layers` now runs a minimum edge colouring, so
  the regenerated record reads **4/5/4/4/9** against Qiskit's 15/16/15/16/21 and the
  median depth ratio is **3.75×**, not 2.5×. The gate-count tie — the point of the
  entry — is unchanged.

## 2026-10-30 — SWAP accounting closed: three entangler bases, and total depth separated from pre-restore depth

The review's one priority accounting item (2026-10-30, §6) was that `1,193 = 99 +
2(547)` counts a SWAP as an instruction, while a CNOT-only backend spends `99 + 6(547)
= 3,381`. Both numbers are right and neither was the other; the record now carries all
three bases and says which is which.

- **THREE TWO-QUBIT BASES, EVERY ROW.** `abstract_two_qubit_gates` counts each `cx`,
  `cz` and `swap` as one operation; `native_entanglers` prices a SWAP at the declared
  `swap_entanglers` (1 by default — a native-SWAP basis);
  `cx_equivalent_two_qubit_gates` always prices a SWAP at three CNOTs. The greedy
  100-qubit lattice tree is **1,193 / 1,193 / 3,381**; the lookahead one is
  **1,121 / 1,121 / 3,165**. Across the eight rows the lookahead saves **3,756
  CNOT-equivalents**, not just the 51 abstract operations the depth improvement made
  easy to notice.
- **A DECLARED BASIS, NOT AN ASSUMPTION.** `place(..., swap_entanglers=3)` declares a
  CNOT-only device, at which point the native column equals the CX-equivalent column;
  `0` is refused, because no SWAP costs less than one entangler. The default basis is
  stated in the record (`swap_accounting.swap_entanglers = 1`), so the headline count is
  no longer ambiguous.
- **TOTAL DEPTH SEPARATED FROM PRE-RESTORE DEPTH.** `mapped_depth` is the total circuit a
  device executes, restore ladder included; `depth_before_restore` excludes it; and
  `restore_depth_cost = mapped_depth - depth_before_restore` is now reported — 94
  layers for the greedy lattice tree, 35 for the lookahead one, 0 wherever routing was
  unnecessary. Previously both numbers were in the record but the prose table quoted
  one without saying which.
- **THE ARITHMETIC IS CHECKED, NOT TYPED.** The record refuses to write unless, for all
  16 placements, `abstract = logical + 2·swaps`, `cx = logical + 3·total_swaps`,
  `native = logical + swap_entanglers·total_swaps` and `restore_depth_cost = total −
  pre-restore`; the equivalence is pinned on the row the review named.
- **REFEREE-CORPUS HONESTY (review §2).** The two tiers share the circuit parser and one
  gate table (`gates.matrix_for` feeds both the Clifford conjugation tables and the
  dense amplitudes), so a wrong gate matrix would fool both; only the representation is
  independent. That is now written down beside the 23-circuit corpus, with randomised
  stabilizer circuits, longer sequences and adversarial specifications named as the
  corpus that would test the shared layer, and not claimed.
- **EXACT-DISTRIBUTION PRE-ALLOCATION PIN (review §4).** A 20-qubit state refuses
  `distribution()` with a traced peak under 1 MB, so the limit is checked before the
  exponential object exists rather than after a failed attempt to build it.
- Pins: `tests/test_vqc_placement.py` (34), `tests/test_vqc_stabilizer.py` (40),
  `tests/test_vqc_synthesis_record.py` (27). Records: `vqc_synthesis.json`
  (`experiment_d.swap_accounting`, per-row bases) and `vqc_external_comparison.json`
  regenerated.

## 2026-10-30 — Experiment D re-measured: the lookahead router cuts routing depth, the finding survives

Experiment D's router was greedy and its numbers were an upper bound on what a better
router might do. Both halves are now measured: the greedy router is preserved as the
baseline, and a **lookahead** router runs beside it on every construction/topology pair.

- **WHAT THE LOOKAHEAD DOES.** Every candidate is still one step along a shortest path,
  so each reduces the pair's distance by exactly one; what it chooses is *which* swap.
  Either operand may move, and each candidate is scheduled into the earliest layer its
  two nodes are free — so routing fills idle wires instead of extending the critical
  path — then ranked by the total wire distance of the next **eight** two-qubit gates
  that involve a qubit the swap moves, then by which qubit is needed later, then by the
  greedy direction. The first key is a scheduler (node availability); the second is the
  lookahead. Both routers are in `placement.py` and are selected by `strategy=`.
- **THE RE-MEASUREMENT, AT 100 QUBITS, AGAINST THE GREEDY NUMBERS.** Lattice chain:
  mapped depth **191 → 117** (−39%), 246 → 225 swaps, 591 → 549 two-qubit gates. Ring
  tree: **253 → 212** (−16%). Line tree: **311 → 278** (−11%). Lattice tree:
  **220 → 84** (−62%), 547 → 511 swaps, 1,193 → 1,121 two-qubit gates. The four rows that
  need no routing (chain on a line/ring/all-to-all, tree on all-to-all) are unchanged at
  0 swaps, which is the zero point: the lookahead cannot improve a free row.
- **THE FINDING SURVIVES, AND CHANGES SHAPE.** Under the greedy router the balanced
  tree lost on every connected topology; under the lookahead it wins on the lattice
  (84 against the chain's 117) and still loses on a line (278 against 100) and a ring
  (212 against 100), while all-to-all is the depth 12 it claims. "Depth 12 is not
  depth 12 on hardware" is therefore a statement about the topology *and* the router,
  and the record now carries both columns instead of one.
- **MORE SWAPS ARE NOT MORE DEPTH.** The lookahead often uses fewer swaps, but the
  deeper point is that it schedules them in parallel: the lattice tree finishes at
depth 84 with 511 swaps because they land on idle wires, where the greedy ladder of
  547 swaps serialises to 220. Swap count alone was never the cost.
- **IT IS A HEURISTIC, NOT A THEOREM.** At small widths it can lose — a 9-qubit
  balanced tree on a line routes two layers deeper than greedy — and neither router
  reorders gates, so commuting gates are never scheduled past each other. The window of
  eight and the tie-break order are choices; the record says so and pins the 100-qubit
  rows rather than claiming an ordering everywhere.
- **EVERY ROUTED CIRCUIT IS STILL DECIDED.** All 16 rows are `VALID` against the same
  specification: the restore ladder still telescopes the swap product to the identity,
  on both routers, so the improved circuit is verified rather than trusted.
- Pins: `tests/test_vqc_placement.py` (31, up from 19, with both routers under the
  adjacency and specification checks), `tests/test_vqc_synthesis_record.py` (26, with
  the comparison row by row). Record: `benchmarks/quantum_compilation/vqc_synthesis.json`
  (`experiment_d.rows` × 16 and `experiment_d.comparison`).

## 2026-10-30 — The dense referee's ceiling, measured: cheap at ten qubits, a wall at fifteen

`DENSE_SPEC_LIMIT = 10` was a policy with no price attached: the dense tier is the only
referee that decides a non-Clifford construction against a specification, and nothing
recorded what one decision costs or where it stops. It is now priced, width by width.

- **THE DOMINANT TERM IS THE OPERATOR, NOT THE STATE.** The tier materialises each
  stabilizer as a dense `2**n x 2**n` Pauli matrix (`np.kron` of single-qubit
  matrices), so one stabilizer costs `16 * 4**n` bytes — quadratic in the Hilbert
  dimension — while the state it is applied to is `16 * 2**n`. A decision runs `n` such
  projections, so the cost model is `n` matrix-vector products against `2**n`.
- **EVERY WIDTH MEASURED IN ITS OWN PROCESS, UNDER A DECLARED BUDGET.** `ru_maxrss` is
  a process high-water mark, so widths measured together would all report the largest
  peak; each child caps its address space at 8 GiB over its post-import baseline with
  `RLIMIT_AS`, so an over-budget width raises `MemoryError` instead of being killed by
  the kernel. At or below the bound the referee's own verdict is recorded beside the
  price; above it the referee still answers `UNRESOLVED` and only the price is new.
- **THE NUMBERS.** 6 qubits: 0.02 s / 0.2 MiB traced / 0.06 MiB operator. 8: 0.07 s /
  1.5 MiB / 1.0 MiB. **10 (the bound): 0.25 s / 20 MiB / 16 MiB, `VALID`.** 11: 1.31 s /
  80 MiB / 64 MiB. 12: 5.1 s / 320 MiB / 256 MiB. 13: 20.3 s / 1.25 GiB / 1.0 GiB. 14:
  88.4 s / 5.0 GiB / 4.0 GiB. 15: **16 GiB per operator, twice the budget**, refused in
  13 s after building the 4 GiB intermediates (5.2 GiB RSS).
- **WHERE IT BECOMES IMPRACTICAL.** Inside its declared bound the tier is cheap — 0.2 s
  and 20 MiB at ten qubits — so memory, not arithmetic, is what stops it. 13 qubits is
  the last comfortable width (20 s, 1.25 GiB), 14 is out of interactive reach (88 s,
  5 GiB) but fits the budget, and 15 does not fit. Peak grows **3.96–4.00x** per qubit
  and time **3.9–5.3x**, both measured rather than fitted.
- **THE BOUND IS PRICED, NOT LIFTED.** `dense_tier_cost` returns seconds, peak traced
  bytes and the phase-free distance with **no verdict field**, and no referee path
  calls it; `specification_state_vector` still refuses above the bound and
  `adjudicate_specification` still returns `UNRESOLVED`, so a caller allowed to exceed
  the bound cannot turn a price into an adjudication. Inside the bound the probe's
  distance is bit-identical to the referee's, which is what makes the price meaningful.
- **WHAT THE CEILING IS NOT.** It prices this implementation. A Pauli string can be
  applied in `O(2**n)` memory as single-qubit gates plus index permutations instead of
  a dense operator, which would move the wall by several qubits; that is stated and
  not measured, and none of these numbers is a claim about the question in principle.
- Pins: `tests/test_vqc_synthesis.py` (56), `tests/test_vqc_synthesis_record.py` (24).
  Record section `dense_referee_ceiling` in
  `benchmarks/quantum_compilation/vqc_synthesis.json` (schema v3).

## 2026-10-30 — VQC Experiment E: a cluster and an irregular graph as specifications

The synthesis grammar could build a GHZ and a ring, and the ring was the only target
whose formation was not a wiring choice. Experiment E makes that general: a **graph
state** is `X_v Z_N(v)` on every vertex, so its specification *is* an edge set, and the
only thing a construction family may choose is the schedule of the commuting `cz` gates.

- **THREE NEW SPECIFICATIONS, STILL NO REFERENCE CIRCUIT.** `cluster_state_target(rows,
  cols)` describes a two-dimensional cluster — the grid-graph state with qubits on the
  lattice **vertices** (not the surface-code cluster, whose qubits sit on lattice edges)
  — and `irregular_graph_target` / `seeded_graph_target` supply a graph with no regular
  structure, reproducible from its seed. Each validates its edge set: a self-loop, a
  repeated edge, an endpoint out of range and a disconnected graph are refused, because
  a graph state on a disconnected graph is a product state. `graph_of_state` reads the
  edges back out of the stabilizers, so the five families are handed no graph — they
  read one — and a specification with a `Y`, two `X`s or an asymmetric edge is refused
  by name rather than half-recognised.
- **THE SCHEDULE IS THE ONLY FREE VARIABLE, AND THE RECORD MEASURES IT.** The five
  families (wire order, transposed order, breadth-first tree then closures, a greedy
  peeling into parallel matchings, and a seeded random control) tie exactly on
  operations, two-qubit gates and edge span, and differ only in depth. All 20 valid rows
  verify. Because the connectivity is the specification's, the Pareto frontier
  **collapses to the peel** here — the opposite of Experiment A, where depth traded
  against connectivity — and the peel is the promoted candidate for all four targets.
- **A PROVEN FLOOR IS REACHED.** A qubit with `d` neighbours takes part in `d` `cz`
  gates and no two can share a layer, so `d` layers are unavoidable. At 100 qubits the
  10x10 cluster has 180 edges and degree floor 4, and the peel achieves **depth 5**
  against the row-major walk's **37** at the same 280 operations; on a 64-qubit seeded
  irregular graph with degree floor 16 it achieves depth 17 against 45. The record does
  *not* claim optimality in general: a graph whose edge chromatic number is
  `max_degree + 1` admits no matching peel that reaches the floor.
- **REFUSALS NAME THE EDGE, NOT JUST THE VERDICT.** Dropping a lattice edge, adding an
  edge the specification does not name, or replacing one `cz` with `cx` each leaves
  **exactly two** stabilizers absent from the prepared state's group — the two endpoints
  of that edge — and the record carries the Pauli strings. A 12-qubit GHZ chain decided
  against a 3x4 cluster is `INVALID`, and the cluster against the GHZ is `INVALID`, at a
  shared width: the referee is deciding the state, not accepting a family.
- **EXPERIMENT E FOUND A REAL BUG IN THE DENSE TIER.** The dense comparison read
  `StatevectorSimulator` amplitudes (bit `i` = qubit `i`) against a state built as a
  Kronecker product (qubit 0 most significant). Every earlier specification — GHZ, rings,
  square lattices — was invariant under reversing its wires, so the two orders were the
  same state and nothing could see it. An irregular graph is not invariant, and the
  naive comparison reported a distance of **0.0605** on an exactly correct state. One
  re-index (`_specification_order`) gives **2.8e-17**, and
  `test_the_two_basis_orders_are_pinned_on_an_asymmetric_state` pins both numbers, so
  the mistake cannot return silently.
- **BOTH REFEREES ON GRAPH STATES.** Six cases at every width both tiers can decide
  (8- and 10-qubit clusters, 9- and 10-qubit seeded irregular graphs, a 3-qubit path and
  one dropped-edge refusal): **5 accepted by both, 1 refused by both**, the accepted
  distances at `1e-17` and the refusal at `0.125`.
- Pins: `tests/test_vqc_synthesis.py` (53), `tests/test_vqc_synthesis_record.py` (21).
  Record: `benchmarks/quantum_compilation/vqc_synthesis.json` (schema v3).

## 2026-10-30 — APEDP-QED: a quantum-assisted proposer, and a negative result that holds up

The question is the one that can be falsified, not "can a circuit do physics": given
noisy observations and a fixed term library, does a QAOA-mediated search recover the
governing equation as well as the classical proposers **at the same number of objective
fits**? It is now measured, and the answer is no — for reasons that are themselves
measurements.

- **`apedp_dyn/qed/` — PROPOSE, NEVER ADJUDICATE.** A term library, a budgeted
  selection objective (`J(z) = mean residual² + λ|z|`, coefficients re-fitted per
  selection, one unit per *distinct* fit), a task carrying a concealed answer, six
  search arms, a referee and a `CandidateProposal` adapter so QED competes inside
  APEDP-X on the platform's usual terms (AGENTS.md §1.1). The referee reports
  `VALID`/`ALTERNATIVE`/`INVALID`/`ABSTAIN` **and** `recovered` separately, because a
  different equation that predicts the holdout as well as the truth is a
  non-identifiability result, not a win and not a failure.
- **QED-001: THE TASK IS FAIR, AND THAT IS SWEPT, NOT ASSUMED.** A damped oscillator
  (`x'' = −1.44x − 0.15v`), development and holdout from different initial conditions,
  1 % observation noise, ten-term library. Before any arm runs, the harness enumerates
  all 1,024 selections outside every budget: the penalised optimum **is** the
  generating support `{x, v}`, by a **0.464 %** margin over `{x, v, x³}`. Without that
  number a missed optimum would be indistinguishable from a badly posed task.
- **RESULT: 1/3 AGAINST 3/3 AT 256 FITS, AND 0/3 AT 64.** Hill climbing, simulated
  annealing, the GA and boolean GP all recovered the truth in 3 of 3 runs at 25 % of
  the space; the quantum arm in 1 of 3. At 64 fits (6.25 %) boolean GP held 3 of 3 and
  the quantum arm held none. The two budgets are both recorded, because at the
  generous one four classical arms **tie** — so that condition ranks nothing, and the
  lean one is where the arms separate.
- **THREE MEASUREMENTS EXPLAIN THE DEFICIT, RATHER THAN A VERDICT EXPLAINING IT.**
  (i) The quadratic surrogate's own minimiser is an eight-term selection, not the
  answer: a pairwise reconstruction under-estimates higher-order interactions and is
  not bounded below at the dense corner. (ii) A harness-side **circuit probe** re-runs
  the arm's circuit from its own seed and measures what probability it put on the
  generating support: `p = 0.0012` at ranks **347-447 of 1024**, with an effective
  selection count of ~749 — the circuit behaves as if choosing among most of the space
  and its mass sits where the surrogate's wrong minimum is. (iii) At 64 fits, 56 went
  to the surrogate and 8 candidates were scored, so the arm was a classical
  model-then-sample strategy wearing a circuit.
- **THE EXPERIMENT CAUGHT ITS OWN FALSE POSITIVE.** The first version of the arm
  reported the best selection the **whole run** evaluated, and it recovered the truth
  in 3 of 3 runs at *both* budgets — parity at 256 and a win at 64. The circuit probe
  is what disproved it: a circuit that ranks the answer 447th cannot be finding it
  through the circuit. It was not — the surrogate expansion had already scored every
  pair, so `{x, v}` was one of its own 56 fits, and the arm was crediting an
  **exhaustive pair enumeration** to the circuit. An arm may now declare its answer,
  and the QAOA arm declares only the best scored candidate. The same record then reads
  1/3 and 0/3. Deterministic either way: the difference is a design decision, not
  noise.
- **THE NON-IDENTIFIABILITY RESULT IS KEPT, NOT TIDIED AWAY.** The RIME arm returned
  `{x, v, sin(x)}` in every run: `R² = 0.9993` on the holdout, true coefficients inside
  their bootstrap intervals, and yet **parameter stability 0.0** and **not
  ablation-separable** — the `sin(x)` term can be deleted at no cost. The generating
  support is separable by contrast: removing `x` costs 2.07 of holdout `R²`, removing
  `v` costs 0.154, and the best addition buys `1.5 × 10⁻⁶`.
- **Pins:** `tests/test_qed_recovery.py` (37) and `tests/test_qed_recovery_record.py`
  (19). The record re-derives its own oracle from its own task specification, checks
  every status against its own numbers, and pins the probe to the arm it probes (same
  surrogate minimum, shots and depth) so a probe cannot drift into a lookalike. The
  surrogate is proved **exact** on a genuinely quadratic objective, which is what makes
  its error on a real one a measurement rather than a bug. Ten limitations are in the
  record and in [`docs/QED_QUANTUM_EQUATION_DISCOVERY.md`](docs/QED_QUANTUM_EQUATION_DISCOVERY.md):
  one synthetic system, finite-difference derivatives, a supplied kinematic identity,
  no hardware, no shot noise in the angles, three seeds, and no advantage claimed in
  either direction.

## 2026-10-30 — The comparison with an external tool, and the bug it exposed in the interchange

The external review's first requirement before any claim of differentiation was a
matched comparison with the tools that already do part of this. That is now measured,
with one rule: **a tool that did not run is not a tool that lost.**

- **`apedp_dyn/vqc/external.py`, AND QISKIT WIRED FOR THE SYNTHESIS TASK.**
  `synth_circuit_from_stabilizers` takes the same signed Pauli stabilizer strings this
  package holds, so both engines are given *the same specification*; its circuit comes
  back through QASM 2, is re-imported, and is decided by the specification referee — so
  its gate count is quoted only for a circuit that was independently verified. At 100
  qubits: GHZ **99 two-qubit gates** (grammar, depth 12) against **4,950** (Qiskit,
  depth 395), and a ring state **100 against 100** two-qubit gates with depth 101
  against 199.
- **THE ADVANTAGE IS TARGET-SPECIFIC, AND THE RECORD SAYS SO.** The GHZ row is 50x; the
  ring row is none at all, because a general Gaussian-elimination synthesizer reaches
  the same two-qubit count as the hand-written construction and loses only on depth. A
  grammar that selects among constructions it knows is not a synthesizer that finds
  one, so `advantage.claimed` is `false` and the prose verdict is generated from the
  measured ratios rather than written by hand.
- **THE HONEST COMPLEMENT: A RANDOM STABILIZER STATE.** On a random 30-qubit state this
  package has no construction family and **abstains**; Qiskit synthesizes it (643
  operations, 222 two-qubit gates) and the referee says `VALID`. Both rows are in the
  record, because a comparison that only ran the favourable case is not a comparison.
- **A REAL BUG FOUND BY THE COMPARISON.** Qiskit reads the first character of a
  stabilizer string as the **highest**-indexed qubit; this package reads it as qubit 0.
  Handed over unconverted, the result is the *mirror image* of the requested state —
  which verifies for a GHZ and for a ring, because both are palindromic, and fails for
  every other target. It was invisible until a random stabilizer state entered the
  corpus. The adapter reverses the strings, and the referee (not the conversion code)
  confirms the reversal; a two-qubit `X`-on-0/`Z`-on-1 target now pins it, and the test
  also demonstrates that the naive order really does produce the mirror image.
- **ABSTENTIONS ARE FIRST-CLASS.** `probe()` reports `compared`,
  `installed_not_wired` or `unavailable` with a reason, and only the first counts as a
  comparison. Stim and MQT QCEC are `UNAVAILABLE` here with the workload a matched
  comparison would need written down; the record explicitly does not treat that as a
  loss, does not treat the 1,000-qubit rows as a state-of-the-art claim, and notes that
  the topology row routes **Qiskit's** circuit with *our* greedy router rather than its
  transpiler against a coupling map.
- **Pins:** `tests/test_vqc_external_comparison.py` (20). Qiskit is an optional extra
  (`apedp[compare]`) that only the harness imports. Regenerating the correctness audit
  for the new record moved no pinned total and re-froze its hash with a fourth recorded
  amendment; the Qiskit-dependent tests skip cleanly when it is absent while the record
  and the abstention pins run either way.

## 2026-10-30 — M3, first slice: the two referees agree, and depth is measured on a topology

Review 2 recommended freezing the synthesis milestone and starting M3 with
"independent stabilizer/dense referee agreement tests" and "architecture-aware resource
accounting". Both are measured here rather than asserted, and the second one changes a
conclusion the previous milestone reported: **depth 12 is not depth 12 on hardware.**

- **BOTH REFEREES NOW ANSWER THE SAME QUESTION.** `adjudicate_specification` grew a
  keyword-only `tier=` override (`auto` / `stabilizer` / `dense`), and
  `referee_agreement()` runs **both** tiers on one circuit and reports the two verdicts
  beside each other. A named tier that cannot decide a circuit raises instead of
  falling back -- a quiet fallback would make the comparison a referee against itself.
  Measured on 23 circuits (GHZ chains and balanced trees at 2..10 qubits, ring graph
  states at 3..6, one `cx`→`cz` mutation): **22 accepted by both, 1 refused by both, no
  disagreement anywhere.** The accepted rows are decided *exactly* (the dense tier's
  phase-free distance is ≤ 1e-12, rounding rather than tolerance); on the refused row
  the tiers fail differently and still agree on the verdict.
- **THE FOUR CONDITIONS ARE PINNED WHERE THEY ARE ENFORCED.** `n` generators for `n`
  qubits, mutual commutativity (even symplectic product for every pair), independence
  over GF(2) (rank `n`, so the group has `2**n` elements and the `+1` eigenspace is
  unique) and Hermitian phases only (`+1`/`-1`). Normalisation and global-phase
  invariance are pinned on the dense referee: `X Y Z = iI` applied to a GHZ still
  verifies, and the accepted tolerance is printed in the check's own requirement.
- **EXPERIMENT D: `apedp_dyn/vqc/placement.py`.** Line, ring, rectangular-lattice and
  all-to-all connectivity graphs, plus a greedy SWAP router. Routing is a
  *relabelling*: the logical-to-node map is tracked exactly and the emitted swaps are
  undone in reverse order, so the swap product telescopes to the identity and the
  mapped circuit is the **same unitary on the same wires** -- which is why every routed
  circuit is decided by the same specification referee (all 8 rows VALID) instead of
  being trusted. Both depths are reported, and `depth_before_restore` /
  `restore_swap_gates` keep the cost of being verifiable separate from the cost of
  routing. At 100 qubits: the balanced tree's logical depth **12** becomes **220** on a
  10x10 lattice (547 swaps, 1,193 two-qubit gates), **253** on a ring and **311** on a
  line (2,415 swaps, 4,929 two-qubit gates), while a chain is depth 100 for free on a
  line and 191 on a lattice. On all-to-all both reproduce the logical circuit exactly,
  which is the known zero point that makes the other numbers readable.
- **PARITY CORPUS HARDENED, AND A HOLE IN THE HARNESS.** New sampler-corpus circuits
  with deterministic allowed-outcome sets: a mid-circuit reset with feedforward and a
  two-qubit condition, and both condition polarities on one measured bit. `assert_parity`
  used to tag a classically conditioned gate as an ordinary gate and then compare the
  Python tier (which applies it) with the native tier (which skips it) -- a parity
  failure that was really an unsupported input; it now refuses by name, and `ccx` is
  pinned as refused by both engines. The corpus also caught a mistake in the fixture
  before it could reach an engine: a `cx` controlled on a qubit that had just been
  reset is a no-op, so the feedforward was invisible until the control was moved.
- **THE RECORD SAYS WHAT THE ROUTER IS.** Greedy, no scheduler, no lookahead, no device
  calibration, no error rate, no classical-control cost: the mapped depths are upper
  bounds that a better router would improve, not a floor on what hardware must pay.
- **EXACT-DISTRIBUTION LIMITS, PINNED (review 1.B).** Both tiers already refused to
  enumerate past twelve qubits -- a stabilizer state has `n` generators but up to
  `2**n` outcomes, so the compact representation does not make the distribution compact
  -- and nothing pinned it. There is now a test on both engines at 13 qubits: the Python
  tier raises, the Rust tier declines into the documented fallback, and
  `probability_of_bitstring` and sampling stay exact, which is what the refusal points
  at. The failure is a refusal to allocate rather than an out-of-memory death.
- **BENCHMARK METHODOLOGY (review 1.C).** `scripts/measure_vqc_rust_parity.py` now takes
  five repeats instead of three, records every raw sample and the interquartile range
  for each side, and carries a `hardware` block naming the platform, CPU count,
  interpreter, NumPy, `rustc`, `cargo`, the release build and the process-wide peak RSS
  (with a note that a peak reached inside one process cannot be attributed to one
  case). Compilation is explicitly outside every timing: the extension is built once by
  `make vqc-native` before the script runs. Regenerating the record moved the ratios,
  which is the point of recording the spread: **3.8x / 7.2x / 4.4x / 14.8x / 3.5x**
  (images 100q, images 400q, sampling 100q x 256, distribution 10q, adjudication 100q),
  against 3.0x / 7.8x / 4.0x / 15.5x / 2.7x on the previous run of the same cases. Both
  are this host; the record now says so, and the IQR says how stable each one was.
  Pins: `tests/test_vqc_placement.py` (19), `tests/test_vqc_synthesis.py` (37),
  `tests/test_vqc_synthesis_record.py` (15), `tests/test_vqc_native_parity.py` (45),
  `tests/test_vqc_native_parity_record.py` (9).

## 2026-10-30 — Synthesis from a specification: a circuit built with no reference to compare against

Review asked for the next milestone after the rewrites loop: stop requiring an
existing reference circuit and construct from a mathematical specification. That is
Experiment A, and it is measured at 100 qubits and again at 10..1,000. Experiment C
(non-Clifford search) is not built; its verification boundary is measured and stated.

- **A TARGET IS NOW MATHEMATICS, NOT A CIRCUIT.** `apedp_dyn/vqc/synthesis.py`: a
  `TargetSpecification` is a qubit count plus `n` signed Pauli stabilizers (and an
  optional exact support), and it is validated as a pure state by the tableau's own
  structural check before any search runs -- anticommuting generators, rank-deficient
  sets, an imaginary phase or a support that does not sum to one are refused. There is
  no reference circuit anywhere in the object, and the referee's signature is
  `adjudicate_specification(circuit, target)`.
- **TWO REFEREES, NEITHER NEEDING A REFERENCE.** The stabilizer tier requires
  `phase_of(generator) == +1` for every specified generator, exactly and at any width.
  The dense tier compares the prepared amplitudes against the state **projected out of
  the specification's stabilizers** (`(I + (-1)^p P)/2` per generator), which is what
  lets it decide a *non-Clifford* construction: a GHZ prepared with `ry(pi/2)` is
  VALID at 3 and 8 qubits (distance 3.9e-17 and 6.9e-18) and `UNRESOLVED` at 12, where
  the reference, not the circuit, is the limitation.
- **EXPERIMENT A, 100 QUBITS:** five constructions from the GHZ specification all
  verify -- chain and reverse chain (100 operations, depth 100, edge span 1), star
  (100/100/99), balanced binary tree (100/**12**/50) and a seeded random tree
  (100/14/90). The balanced tree is promoted (`VALID`/`MET`/`SUFFICIENT`), the Pareto
  frontier keeps the nearest-neighbour chain beside it because depth and connectivity
  trade off, and four wrong constructions (drop the Hadamard, reverse the middle `cx`,
  add a Hadamard, `cx`→`cz`) are refused with the failing generator named. A second
  target -- a 100-qubit **ring graph state** -- is built and promoted, while the linear
  cluster (the same construction missing the closing edge) is refused with its two
  absent stabilizers named. The star is depth 100, not 2: its `cx` gates share a
  control and serialise, which is measured rather than assumed.
- **EXPERIMENT B:** 24 rows at 10, 50, 100, 250, 500 and 1,000 qubits, every one
  verified against the specification with `n` stabilizers checked; the 1,000-qubit
  constructions verify in 1.3-1.7 s each.
- **THE SAME ADMISSION POLICY, AND A BUG IT CAUGHT.** A synthesis candidate satisfies
  the new `Candidate` protocol and goes through the same sealed verdict: a valid
  construction offered under `{"depth": 1}` is **`VALID` and `UNMET`** and refused with
  "does not improve its resource claim (depth: claimed 1, measured 12)". Writing that
  test caught the claim check being folded into the semantic verdict in the first
  draft, which reported an unmet claim as `INVALID` -- a false statement about the
  circuit, and exactly the conflation `discovery.py` exists to prevent.
- **§6 of the review, both items.** The discovery record now carries
  `workload.reference_provenance` stating that its 222-operation reference is
  deliberately inefficient and that the 54.95% reduction is against inserted work, not
  an advantage over an optimised GHZ preparation (whose count is `n` operations and
  depth `n` -- what the search reaches); the limitation list repeats it, and two new
  pins assert the provenance and that the loop's admission still carries the four
  separate answers with every accepted step certified VALID.
- **WHAT IS NOT CLAIMED, IN THE MODULE AND IN THE RECORD.** The construction space is a
  grammar of named wireings, not a general synthesiser; two targets, not a family;
  the ranking is a stated preference with the frontier reported beside it; and no
  cheaper GHZ is claimed, since every family is `n` operations and `n-1` two-qubit
  gates. Pins: `tests/test_vqc_synthesis.py` (31) and
  `tests/test_vqc_synthesis_record.py` (13).

## 2026-10-30 — The VQC execution core goes Rust-first, with Python as the referee and the fallback

Review asked for a dual-language VQC: Rust for the mathematical execution core,
Python for research and adjudication, and no rewrite of the platform. It also asked
for five engineering rules, and specifically for a benchmark before any claim of
Rust superiority. All of that is here, measured.

- **PAULI ALGEBRA, THE STABILIZER TABLEAU, SAMPLING, EXACT DISTRIBUTIONS AND THE
  CLIFFORD IMAGES ARE NATIVE.** `rust/apedp-vqc` (module `apedp_vqc`) implements the
  review's first four modules — `pauli-core`, `stabilizer`, `clifford-equivalence`
  and circuit-IR parsing — over packed `u64` bit vectors rather than Python
  integers, with 31 pure-Rust tests (`cargo test --no-default-features`). The
  conjugation tables are **derived from the same gate matrices** as the Python
  tier's, not transcribed, and a test compares all 24 one-qubit and 48 two-qubit
  rules as integers.
- **ONE CIRCUIT SPECIFICATION, TWO READERS.** The crate parses the canonical JSON
  `Circuit.to_json()` emits — same schema version, same op tags, same gate names —
  and refuses what it cannot represent by name. A real divergence was caught this
  way: the first condition parser guessed `[[bit, value]]`, while the writer emits
  `{"bit": i, "value": bool}`; the teleportation parity test failed loudly instead
  of the engine silently skipping conditioned gates.
- **RUST FIRST, PYTHON AS THE FALLBACK, AND THE VERDICT STILL IN PYTHON.**
  `apedp_dyn/vqc/native.py` is the bridge: every entry point returns `None` when the
  fast path cannot answer, so `StabilizerBackend` and
  `adjudicate_clifford_equivalence` use it by default and fall back to the Python
  tier; with the wheel absent nothing changes but the speed. The referee never
  moved: Rust produces images, `equivalence.py` compares them and writes the verdict,
  and below `PARITY_LIMIT` (24) qubits it re-derives the table in Python and raises on
  a disagreement. Above it the report says `image_verified_by: "suite"` instead of
  implying a check that did not happen. A corrupted native table is refused, not
  adjudicated (regression test).
- **PROVENANCE IS RECORDED, NOT ASSERTED.** `RunResult.engine` is `"rust"` or
  `"python"`; `CliffordEquivalenceReport` gained `image_engine` and
  `image_verified_by`; `engine="rust"` raises rather than quietly downgrading;
  `native.tallies()` counts `rust`/`python`/`fallback` calls and
  `APEDP_VQC_PYTHON_ONLY=1` pins the reference path, which is how the benchmark
  proves the Python timings never touched the fast path.
- **MEASURED, ON IDENTICAL FROZEN INPUTS**
  (`benchmarks/quantum_compilation/vqc_native_parity.json`, median of three, same
  process, warm-up excluded, Python 3.12.3 / Linux x86-64): images at 100 qubits
  17.82 ms -> **5.86 ms** (3.0x) and at 400 qubits 460.82 ms -> **59.06 ms** (7.8x);
  100-qubit sampling at 256 shots 2527.08 ms -> **633.39 ms** (4.0x); the exact
  10-qubit distribution 38.19 ms -> **2.47 ms** (15.5x); a 100-qubit Clifford
  adjudication 45.34 ms -> **16.59 ms** (2.7x). The 100-qubit image row is the
  noisiest of the five (17-29 ms of Python), which the record says. The record refuses to be written if
  any case disagrees, and pins the SHA-256 of the script, the seven Rust sources and
  the three Python modules it measured.
- **THE TWO EXISTING RECORDS WERE REGENERATED, NOT RE-PINNED IN PLACE.** Both
  pinned the SHA-256 of the modules that produced them, and those modules changed,
  so `vqc_stabilizer_scaling.json` and `vqc_clifford_discovery.json` were re-run:
  the 1,000-qubit adjudication of two 7,500/8,190-gate circuits now takes **1.94 s**
  (about a minute before), the 100-qubit discovery loop **2.6 s** (5.9 s before)
  with the identical 222 -> 100 operations and 76 certified steps, and the
  stabilizer-tier scaling rows are re-measured on this host.
- **WHAT IS NOT PORTED, NAMED.** The dense statevector tier, the decoders, the noise
  engine and the discovery loop stay Python; MWPM remains exponential in defects;
  no memory ceiling and no parallelism is claimed. `docs/VQC_RUST_CORE.md` says why,
  per capability, with the review's five rules mapped to the code that enforces
  them. Pins: `tests/test_vqc_native_parity.py` (39) and
  `tests/test_vqc_native_parity_record.py` (7).
- **THE DQI BASELINE IS UNTOUCHED.** Adding `vqc_native_parity.json` meant
  regenerating `correctness_audit.json` so its scanned set matches the records on
  disk; the Phase 0 baseline was re-frozen with the reason recorded and its pinned
  totals are unchanged (checked 12, partial rows 13, resources-only 5,
  aggregate-only 51), with the same 13 DQI gaps.

## 2026-10-30 — The discovery loop, end to end at 100 qubits, and a non-vacuous evidence rule

Review asked for three things: an explicit rule that an empty check set is never a
pass, controlled-subroutine phase tests (already landed — see the entry below), and a
complete 100-qubit discovery-to-adjudication workflow. The first and third are here.

- **AN EMPTY CHECK SET IS `INSUFFICIENT` EVIDENCE, NOT A PASS.** `CandidateVerdict`
  now carries four separate answers — `semantic_verdict`, `resource_verdict`,
  `evidence_sufficiency` (`SUFFICIENT`/`UNDECIDED`/`INSUFFICIENT`) and
  `promotion_decision` — and `promote()` requires all three positives. `evidence` is
  derived from the checks in `__post_init__`, so no caller can set it, and a candidate
  with no checks is refused with "insufficient evidence: no check was run" rather than
  promoted on a vacuous verification. Pins: the mandatory regression test plus three
  field tests in `tests/test_vqc_discovery.py` (13).
- **THE DISCOVERY LOOP NOW RUNS, AND ITS WRONG CANDIDATES ARE MEASURED.**
  `apedp_dyn/vqc/clifford_search.py` proposes local semantics-preserving rewrites,
  certifies every accepted step with the Clifford referee **against the original
  reference** (so the loop cannot drift), ranks by `(operations, depth)` and admits
  through `discovery.py`. `scripts/measure_vqc_clifford_discovery.py` runs it on a
  100-qubit GHZ chain with cancelling work spliced in: **222 → 100 operations** (depth
  108 → 100) in **7.1 s**, 76 steps all `VALID`, 0 rejected, admission
  `VALID`/`MET`/`SUFFICIENT` → **promoted**.
- **THE CHEAPER BUT NON-EQUIVALENT CANDIDATE IS REFUSED, WITH ITS REASON.** The star
  circuit is 100 operations against the reference's 222 and prepares the *same* state,
  which is exactly what a resource-only search would promote: **INVALID** at unitary
  scope (196/200 images differ), **VALID** at state scope, and admission refuses it.
  Six mutated candidates (delete a gate, flip a phase, `cx`↔`cz`) are all refused with
  the differing image named, so the record shows rejection as well as acceptance.
- **THE SIMULATOR AND THE DENSE TIER ARE IN THE LOOP.** 64 stabilizer shots of the
  reference and of the rewritten circuit give an identical support with a maximum
  count difference of 0, and the same workload at 8 qubits agrees with the dense
  referee at distance `1.6e-16`. First run of the record used the search's default
  64-step cap and reported 116 operations; the cap is now 1,000 and a check asserts
  the loop converged below it, because a truncated search is otherwise
  indistinguishable from a local optimum.
  Pins: `tests/test_vqc_clifford_discovery.py` (15), including a property test that
  **every** rewrite proposed for five random Clifford circuits is certified VALID.
- **A REAL BUG IN THE FIRST DRAFT, CAUGHT BY THE REFEREE PATH.** The mutation channel
  rewrote two-qubit gates with one-qubit `s` gates, which fails circuit validation
  before any referee sees it; mutations now preserve arity (`cx`↔`cz`, `s`↔`sdg`,
  Pauli→Pauli). Recorded because the check that caught it is the one the loop is for.

## 2026-10-30 — Controlled-subroutine scope, and scaling measured to 1,000 qubits

Review asked for a controlled-subroutine phase test before any global-phase-insensitive
transformation is applied automatically, and for scaling beyond the 100-qubit GHZ
record. Both are now implemented and measured; nothing from the earlier correction
entry changed.

- **A GLOBAL PHASE IS FREE STANDALONE AND NOT UNDER CONTROL.**
  `adjudicate_equivalence(..., as_subroutine=True)` compares `diag(I, U_c)` against
  `diag(I, U_r)` (new `controlled_lift`), which removes the phase allowance precisely
  because a phase on one control branch is a relative phase. Measured: `RX(2π)` versus
  `I` is **VALID** standalone (distance 0, phase π) and **INVALID** as a subroutine
  (distance `sqrt(2)`); `CX` versus `H CZ H` is VALID in both scopes. Reports carry
  `scope: standalone | controlled`. `adjudicate_candidate(..., as_subroutine=True)`
  routes to the dense tier by construction — the conjugation images cannot see a global
  phase — so a controlled question above ten qubits is `UNRESOLVED`, not a guess.
  Pins: 3 new cases in `tests/test_vqc_equivalence.py` (20) and 1 in
  `tests/test_vqc_discovery.py` (12).
- **SCALING IS MEASURED, IN FOUR FAMILIES, TO 1,000 QUBITS.**
  `scripts/measure_vqc_stabilizer_scaling.py` →
  `benchmarks/quantum_compilation/vqc_stabilizer_scaling.json`. At 1,000 qubits: GHZ
  chain builds in **0.47 s**, validates in 0.43 s, samples in 0.62 s per shot; GHZ star
  the same and lands in the *same* stabilizer group by a different route; the ring graph
  state's 1,000 defining generators `X_q Z_(q-1) Z_(q+1)` all stabilise it; a 7,500-gate
  seeded random Clifford builds in **14.2 s**, validates in 0.65 s, and costs **127 s
  per Z sample** — the honest price of measuring 1,000 qubits, recorded rather than
  hidden. The tableau costs `n²` labels (10⁶ bytes at 1,000 qubits) against `2**1,000`
  amplitudes.
- **EQUIVALENCE AT 1,000 QUBITS, ON CIRCUITS THAT ARE ACTUALLY DIFFERENT.** A
  7,500-gate random Clifford and its rewriting with cancelling `h h` / `cx cx` pairs
  (8,190 gates) are decided **VALID** from 2,000 images in 61 s; the same circuit with
  one `s` replaced by `sdg` is **INVALID** with the offending image named (`Z_0`, a sign
  flip) in 54 s. No matrix at either size. Chain versus star is VALID at state scope and
  INVALID at unitary scope at every size, so the 100-qubit distinction reproduces at
  1,000. Pins: `tests/test_vqc_stabilizer_scaling.py` (7), including hashes and a
  re-derivation of the 100-qubit rows from the script.
- **`validate()` IS 200× CHEAPER, AND THAT IS WHY IT COULD BE MEASURED AT 1,000.** The
  commutation check now runs on packed bit masks
  (`((x1 & z2) ^ (z1 & x2)).bit_count() & 1`) instead of per-qubit label loops: 98.7 s
  became **0.48 s** at 1,000 qubits on the same state. Rank and sign checks are
  unchanged.
- **THE CORRECTNESS AUDIT WAS REGENERATED, AND ITS FROZEN HASH RE-FROZEN
  DELIBERATELY.** `benchmarks/quantum_compilation/correctness_audit.json` was stale: it
  had not been re-run after several VQC records were added, so its scanned set no
  longer matched the records on disk. Regenerated with
  `scripts/audit_record_correctness_checks.py`; pinned totals are unchanged (checked 12,
  partial rows 13, resources-only 5, aggregate-only 51), and no DQI record was touched.
  Because the Phase 0 baseline pins that file's bytes, the single hash was re-frozen
  with an `amended` entry in `vqc_phase0_baseline.json` naming what changed and why —
  not silently.

## 2026-10-30 — Referee correction: the equivalence metric, claim separation, an independent oracle

External review asked for the equivalence report to be checked against its own stated
metric. It did not hold, and the fix plus three related corrections are below.

- **THE EQUIVALENCE DISTANCE WAS NOT THE MINIMUM IT CLAIMED TO BE.** The report was
  documented as `min_phi max_ij |U_c − e^{i·phi} U_r|` and computed a max-entry
  deviation at `phi = arg<U_r, U_c>`, which minimises the **Frobenius** norm, not the
  max-entry norm. `I` versus `Z` therefore reported **2.0** where both objectives give
  **√2 ≈ 1.414214** (`|1 − e^{iφ}|` and `|1 + e^{iφ}|` cross at `φ = π/2`). The
  distance is now `min_phi ||U_c − e^{i·phi} U_r||_F / sqrt(2**n)`, attained in closed
  form at that same phase and evaluated as the residual norm so an exact 1e-16
  agreement does not become 1e-8; its range is `[0, √2]`. The max-entry deviation at
  that phase is reported **separately** as `entry_deviation`, with
  `"entry_deviation_is_minimised": false` in the receipt. Corrected numbers, same
  verdicts: `I` vs `Z` 1.414214 (entry 2.0), `CCX` vs `CCX; Z` 1.414214 (entry 2.0),
  `CCX` vs `CX(0,2); CX(1,2)` 1.224745, `CX` vs `SWAP` 1.224745, `CX` vs `H CZ H` and
  `RX(2π)` vs `I` still 0.0. This supersedes the distance numbers in the 2026-10-29
  entry below, which described the same conflation.
- **SEMANTICS AND ECONOMICS ARE NOW TWO VERDICTS.** A resource claim no longer turns a
  correct circuit INVALID. Both reports carry `claim_verdict: MET | UNMET |
  NOT_ASSERTED` and `promotable`; `CircuitCandidate` → `CandidateVerdict` does the
  same, and `promote()` refuses the equivalent-but-larger candidate with "is VALID but
  not an improvement (UNMET)" instead of "not VALID". Measured: `CX` as `H CZ H`
  claiming `operations: 1` is VALID/UNMET/not promotable, and the 5-qubit Clifford
  claim case reports VALID with UNMET.
- **AN INDEPENDENT ORACLE, BECAUSE BOTH PATHS SHARED GATE APPLICATION.** A bug in one
  gate rule would have been invisible to a comparison of two circuits that both use
  it. `tests/test_vqc_equivalence.py` now builds gate matrices by **bit arithmetic on
  basis indices** — no simulator, no tensor slicing — and requires the execution
  path's unitary to equal the oracle's for random Clifford circuits on 1–4 qubits and
  the stabilizer tier's images to equal `U P U†` from the oracle's `U`. This is an
  independent check of the small-system tier; above ten qubits the tableau path
  remains the only exact one.
- **`StabilizerState.validate()`.** `O(n²)` structural check, refused on anticommuting
  generators (no simultaneous eigenspace) and on rank-deficient sets (group smaller
  than `2**n`, so not pure); a dependency with an odd phase was already refused by
  the elimination. New pins: validation on random 6-qubit Clifford states and on a
  100-qubit GHZ state, both refusal modes, a repeated measurement returning the
  collapsed outcome, and a `Y`-eigenstate (`|+i>`) measuring 50/50 in `Z` with a
  valid, certain branch per trajectory.
- **DOCUMENTED LIMITS.** "A global phase is free" holds for a standalone operation and
  not for a controlled subroutine, where it becomes a relative phase — recorded in
  `docs/VQC_CORE.md` §5 as a limitation of the current claim, not a supported case. The
  100-qubit X-basis row is scoped to what it measured: 128 shots consistent with the
  parity constraint, **0 collisions**, and no claim about enumerating `2**99` strings.

## 2026-10-30 — Equivalence above the dense bound, a matching decoder, and discovery admission

- **A 100-QUBIT CIRCUIT PAIR IS NOW DECIDED, NOT ESTIMATED.**
  `adjudicate_clifford_equivalence` compares a Clifford circuit's **conjugation action**
  on the Pauli generators — `2n` signed Pauli strings, exact integers mod 4 — instead of
  a `2**n x 2**n` matrix. Two Cliffords that agree there differ by a scalar, so this is
  exactly unitary equivalence up to one global phase at any size. Measured at **100
  qubits**: a global phase of −1 is **VALID** (0/200 images differ), a chain-vs-star
  construction that prepares the *same* GHZ state is **INVALID** at unitary scope
  (196/200) and **VALID** at state scope, and replacing `cx(0,1)` with `cz(0,1)` is
  **INVALID** with the offending image named (3/200). The dense tier now returns
  `UNRESOLVED` for a circuit it cannot materialise rather than `INVALID`: being unable
  to compute a matrix is not evidence about the circuit. `S` versus `Z S = S†` is the
  one-qubit version of the same distinction — same prepared state, different unitary.
  Pins: `tests/test_vqc_clifford_equivalence.py` (19).
- **A DECODER INTERFACE, WITH EXACT MINIMUM-WEIGHT MATCHING AS THE SECOND
  IMPLEMENTATION.** `apedp_dyn/vqc/decoders.py` names the interface
  (`decode(syndrome) -> Correction`), keeps the distance-3 lookup table, and adds a
  defect-graph matching decoder solved by exact T-join dynamic programming over subsets
  of defects — boundaries included, exponential in the number of defects, and stated as
  such. `logical.py` gained a distance-parameterised repetition code whose correction
  gates are **built from the decoder**, so the two implementations run the same code on
  the same noise. Measured at `p = 0.1`, 1,200 shots, seed 20261030, against the
  enumerated prediction (all `2**d` patterns, never the simulation): d = 3
  **0.0258 vs 0.028**, d = 5 **0.0050 vs 0.00856**, d = 7 **0.0033 vs 0.002728**; at
  distance 3 the enumeration reproduces `3p² − 2p³` to 1e-15, and matching equals the
  old table on all four syndromes. Distance-7 matching equals a brute-force minimum on
  all 64 syndromes, every error of weight ≤ `(d−1)/2` is decoded exactly, and weight 4
  is miscorrected as a distance-3 code must be. Pins: `tests/test_vqc_decoders.py` (18).
- **DISCOVERED CANDIDATES GO THROUGH THE REFEREES, OR THEY ARE NOT PROMOTED.**
  `apedp_dyn/vqc/discovery.py` takes a `CircuitCandidate` (candidate, reference,
  resource claims, optional declared distribution), routes it to the tier that can
  decide it — stabilizer when both circuits are Clifford, dense otherwise, the
  distribution receipt when one is declared — and seals a `CandidateVerdict`.
  `promote()` refuses everything that is not VALID, **including `UNRESOLVED`**. Measured:
  `CX` as `H CZ H` promoted with its claim met and refused with `operations: 1`; a
  measured Bell candidate decided by its receipt (VALID at 2,000 shots, UNRESOLVED at
  300 and refused); a measured circuit that declares nothing gets no checks at all; a
  100-qubit Clifford candidate is decided; a 20-qubit non-Clifford pair is refused
  rather than passed. Pins: `tests/test_vqc_discovery.py` (11).
- **A RECEIPT BUG FIXED ON THE WAY.** A declared probability of exactly 1.0 divided by a
  zero binomial variance and crashed `receipts._distribution_check`; a certainty is now
  scored as `0` if it happened and `shots` sigma if it did not, so the receipt stays
  valid JSON and the claim can still fail.
- **Documentation.** [`docs/VQC_CORE.md`](docs/VQC_CORE.md) gains §5.1, §7 and §8 with
  every number above, and §10 now names what is genuinely missing (surface code, MPS,
  repeated syndrome measurement, fault tolerance, hardware, non-Clifford equivalence
  above the dense bound, discovery proposals). The Phase 0 inventory's referee,
  logical-qubit, backends and discovery rows say what is now true.

## 2026-10-30 — A second backend at 100 qubits, a phase-sensitive equivalence referee, and two more codes

- **A 100-QUBIT GHZ CHAIN, AND THE CROSS-CHECK THAT MAKES IT COUNT.**
  `apedp_dyn/vqc/stabilizer.py` executes Clifford circuits over `n` commuting signed
  Pauli generators instead of `2**n` amplitudes: phases are integers mod 4, every
  product/conjugation/sign comes from a table built once at import from matrices, a
  measurement either takes a random branch (rewriting the anticommuting generator to
  `±Z_q`) or reads a decided outcome by GF(2) elimination on the `2n`-bit
  characteristic vectors. The dense tier is the referee: the **same** `ghz_circuit`
  builder on both tiers agrees on **8,188** outcome probabilities for chains of 2–12
  qubits (max deviation **1.1e-16**) and on **22** Pauli strings at 6 qubits
  (**2.2e-16**). Then the 100-qubit chain: 100 generators equal to the textbook set,
  only `0…0` and `1…1` at 0.5 each (`2**100` amplitudes is not a thing anyone can
  write down), **512 shots → 248/264** in about six seconds on this host, and the
  X-basis variant gives 2**99
  even-parity strings with **0 collisions in 128 shots**. The test asserts the
  100-qubit circuit is that same ladder gate for gate, which is what lets the
  closed-form prediction confirmed at `n ≤ 12` be used at `n = 100`. Non-Clifford
  gates and every `Noise` operation are **refused by name**, not approximated.
  Record: `benchmarks/quantum_compilation/vqc_stabilizer_ghz.json`
  (`scripts/measure_vqc_stabilizer_ghz.py`). Pins: `tests/test_vqc_stabilizer.py` (32).
- **EQUIVALENCE, UP TO EXACTLY ONE GLOBAL PHASE.** `apedp_dyn/vqc/equivalence.py`
  materialises both unitaries through the same gate application the backends use and
  compares them by `min_phi max_ij |U_c - e^{i phi} U_r|`. `CX` versus `H CZ H` is
  **VALID** at distance 0.0; `RX(2π)` versus `I` is **VALID** with phase π, because a
  global phase is not an error. The case that justifies the module: `CCX` versus
  `CCX; Z` reproduces **every Z-basis probability** — the test asserts that
  elementwise first — and is **INVALID** at distance 2.0, which is the DQI phase-error
  failure mode at circuit level. `CCX` versus two CNOTs is INVALID (1.0), a wrong
  qubit count is INVALID, a measured/feedforward candidate raises
  `NotAUnitaryCircuitError` and is INVALID with that reason, and a `claims` argument
  fails a candidate that is equivalent but *larger* than the improvement it is offered
  under. Bounded by the dense matrix near ten qubits. Pins:
  `tests/test_vqc_equivalence.py` (9).
  *Superseded in part by the correction entry above: the distance is now the
  normalised Frobenius minimum (this entry's 2.0 was a max-entry deviation taken at
  the Frobenius-optimal phase, not a minimum), the metric reads 1.414214 for the same
  pair, and the resource claim is reported as `claim_verdict` instead of failing the
  verdict.*
- **THE PHASE-FLIP CODE AND THE PERFECT FIVE-QUBIT CODE.** `logical.py` gains the
  phase-flip repetition code (the bit-flip code transposed: `X_i X_j` syndromes, `Z`
  corrections, the same `3p² - 2p³`), measured at 0.023 ± 0.0024 against 0.028 at
  `p = 0.1`; `five_qubit.py` is a real distance-3 code — four stabilizers, four
  ancillas, a projection encoder with a conditional Pauli frame — whose prediction is
  **enumerated, not fitted** (all 1,024 Pauli patterns; `0.5432098765` at `p = 1`, a
  rational with denominator `3**5`). All **15** single-qubit Paulis map to distinct
  nonzero syndromes out of 16 (the code is perfect: `16 = 1 + 15`), every single error
  is corrected on both preparations, a weight-2 error can flip the logical value, and
  the measured rates are 0.0130/0.0540 against exact 0.014888/0.053005 at `p =
  0.05/0.1`. Pins: `tests/test_vqc_codes.py` (46).
- **Documentation.** [`docs/VQC_CORE.md`](docs/VQC_CORE.md) gains §4.1, §4.2, §5 and
  §6 with every number above, and §8 now names what is genuinely missing (MPS, surface
  code, MWPM, fault tolerance, hardware, equivalence above the dense bound) instead of
  the backends and codes that exist. The Phase 0 inventory's referee, backends and
  logical-qubit rows are updated to what is now true.

## 2026-10-30 — VQC Phase 0 inventory, and Phase 1 finished: `Reset`, OpenQASM 2, and a frozen DQI baseline

- **PHASE 0 EXISTS AS A CHECKABLE ARTIFACT, NOT A NARRATIVE.**
  [`docs/VQC_PHASE0_CAPABILITY_INVENTORY.md`](docs/VQC_PHASE0_CAPABILITY_INVENTORY.md)
  classifies every DQI component against the seven VQC capabilities with one of five
  decisions — reuse, adapt, build, retain as one backend, not connected — and names
  what is genuinely absent (stabilizer/MPS backends, surface code, MWPM, unitary
  equivalence, discovery into the IR, any threshold claim). It also records where the
  proposed `apedp_vqc/` tree actually lives and why it is `apedp_dyn/vqc/` instead: no
  second top-level package, and flat modules until Phase 6 gives the subpackages
  something to hold.
- **THE "DO NOT MOVE THE DQI BASELINE" RULE IS ARITHMETIC NOW.**
  `benchmarks/quantum_compilation/vqc_phase0_baseline.json` freezes the SHA-256 of the
  thirteen DQI records the VQC work must not disturb, and
  `tests/test_vqc_phase0_inventory.py` recomputes all of them — plus the frozen
  submission record's verdicts and 0.9490542129538326 score ratio, so a byte-exact but
  semantically different rewrite is caught too. It also fails if the inventory names a
  path that does not exist, or if a VQC module is missing from it.
- **PHASE 1 TASK 6: OpenQASM 2, WITH THE GAPS REFUSED.** `apedp_dyn/vqc/qasm.py`
  exports and imports `OPENQASM 2.0` for the gates, `measure`, `reset` and `if(c==k)`,
  carrying metadata and ancillas as `// vqc-...` header comments so a round-trip
  preserves the circuit hash rather than dropping its contract. It refuses noise
  (no QASM 2 construct), conditions on a subset of bits (whole-register equality; a
  per-bit test is QASM 3), user-defined `gate`, `u2`/`u3` and malformed parameters.
  Teleportation is the test case that keeps this honest: its JSON round-trip is exact
  and its QASM export is refused by design, because two gates are conditioned on one
  bit of a two-bit register.
- **`Reset` IS AN OPERATION, NOT A MEASUREMENT.** Added to the IR, serialised,
  counted in resources and executed as measure-and-discard. Measured: `X; Reset;`
  reads `0` on 128/128 shots; `H; Reset;` reads `0` on 128/128; and resetting one half
  of a Bell pair leaves the other maximally mixed (0.503/0.497 at 4,096 shots),
  which is the difference between the reset map and a projection that would leave the
  partner in `|0>`.
- **Pins.** `tests/test_vqc_qasm.py` (15) and the two reset tests in
  `tests/test_vqc_core.py` (now 63); `tests/test_vqc_phase0_inventory.py` (6).

## 2026-10-30 — APEDP-VQC Core: the circuit IR, an exact simulator, checkable noise, a logical qubit and receipts

- **THE PLATFORM'S FIRST MILESTONE, NOT A SECOND OPTIMISER.** `apedp_dyn/vqc/` is the
  reusable core the DQI work was a workload for: `circuit.py` (versioned IR: gates,
  measurements with reset, classical-bit conditions, noise operations, declared
  ancillas, metadata, canonical JSON, SHA-256 identity), `gates.py` (one shared gate
  table), `backends.py` (backend protocol + exact statevector backend, reusing the
  existing `MR` simulator), `noise.py` (exact density-matrix channels), `logical.py`
  (the distance-3 bit-flip code as a circuit) and `receipts.py` (VALID / INVALID /
  UNRESOLVED adjudication). Discussion and every measured number in
  [`docs/VQC_CORE.md`](docs/VQC_CORE.md).
- **THE SIMULATOR IS TESTED AGAINST PHYSICS, NOT AGAINST ITSELF.** Bell and GHZ match
  independently constructed states; teleportation's post-state equals
  `|00> ⊗ RY(π/4)|0>` up to global phase on **every one of 64 shots**, and its X-basis
  `P(0)` is 0.85596 against an analytic 0.85355. With the conditional `Z` removed the
  same readout collapses to 0.500 — feedforward is load-bearing, and the adjudicator
  rejects the candidate at z = 62.
- **THE NOISE TIER IS EXACT AND FALSIFIABLE.** Trace, Hermiticity, positivity and Kraus
  completeness are recomputed over five channels × five probabilities; depolarising
  one half of a Bell pair gives fidelity `1-p` and purity `(1-p)² + p²/3` to 1e-12,
  amplitude damping gives `P(0) = γ`, dephasing gives purity `1 - p/2`. The IR's noise
  operations are the sampled counterpart: a `p = 0.2` bit flip decorrelates a Bell pair
  with probability 0.7979 ± 0.0045 at 8,000 shots and reproduces exactly from its seed.
  (The first version of `apply_kraus` mislabelled the row/column bit indices and
  produced a non-positive matrix; the positivity check is what caught it, which is why
  it is a test rather than a comment.)
- **ONE LOGICAL QUBIT, WITH ITS LIMITS ATTACHED.** Every single bit flip is identified by
  its syndrome and corrected, for both preparations; a pair of flips is miscorrected into
  a logical flip (`syndrome 01`, readout `111`); the measured failure rates are
  0.006625 ± 0.000907 at `p = 0.05` and 0.024875 ± 0.001741 at `p = 0.10` against the
  exact `3p² - 2p³` (0.00725, 0.028). No threshold is claimed, syndrome extraction is
  perfect, the code covers bit flips only, and each record says so in its own fields.
- **AN UNDERPOWERED RUN ABSTAINS.** `adjudicate()` checks the circuit, the ancilla
  cleanup, the outcome distribution against a declared expectation at 4σ, and whether
  the shot count could have detected the declared effect. 8 shots on a correct
  teleportation returns **UNRESOLVED**, not VALID; the same circuit at 4,096 shots is
  VALID; a missing phase correction, a dirty ancilla and an invalid circuit are INVALID.
- **WHAT IS NOT THERE, NAMED SO IT IS NOT ASSUMED.** One backend (statevector only — no
  stabilizer, MPS or tensor-network tier), no surface code or MWPM decoder, no encoded
  gates or repeated syndrome measurement, no fault-tolerance or threshold experiment, no
  hardware adapter, no QASM export, and no discovery integration yet. Those are the next
  milestones, and `docs/VQC_CORE.md` §6 lists them as absent rather than planned.
- **Pins.** `tests/test_vqc_core.py` (61 tests) holds all of it, including twelve
  malformed-program rejections and the five receipt verdicts.

## 2026-10-30 — audit of the round-two review: the three arithmetic claims were already right, the verifier wording was not

- **`32,488`, `8.669587e-4` AND `.382189` ARE NOT IN THIS REPOSITORY.** A second
  external review asks for them to be corrected in "the first report" and "the second
  report". Checked rather than assumed: every record, note and pin says **32,408 calls**
  (`decomposition.multiply_calls`), **8.660587e-4** (`eps_plus` on both verified rows)
  and **5,725,108.382109375** (the reference executed mean), and nothing anywhere says
  otherwise. The arithmetic also forces the call count rather than merely agreeing with
  it: the measured static delta is 291,672 and `291,672 / 9 = 32,408` exactly, while
  `32,488 x 9 = 292,392`. The 42.45% multiply share is `75 x 32,408 / 5,725,461`.
- **SO THE CORRECTIONS BECAME PINS, NOT EDITS.** `tests/test_dqi_submission_score.py`
  now recomputes `eps_plus = 1 - 2^(log2_delta/n)` from the budget the record names and
  matches the stored rows; derives the call count from the static delta (`32,488 x 9`
  fails the identity) and matches the 42.45% share; and recomputes the executed delta
  from the two frozen rows, so a transcribed difference cannot drift.
- **THE VERIFIER WORDING WAS WRONG AND IS FIXED.** §8 said `verify()` ran "unmodified"
  and then disclosed, four paragraphs later, that `reaction_depth` was stubbed. Those
  cannot both be the whole truth about the same invocation. The sentence now says what
  ran — every scoring line unmodified, one **diagnostic-only** stub — and the disclosure
  says plainly that this is the competition verifier with a documented shim, not a
  literal unmodified invocation. Same correction in `docs/DOCUMENTATION.md`.
- **THE SHIM'S ISOLATION IS NOW PINNED BY TEST, NOT BY PROSE.** The record's
  `verifier_patch` (replaced line, replacement, both SHA-256s) is re-derived from the
  shipped verifier and must hash-match; and an AST comparison proves the patched module
  differs from the shipped one in exactly one dict value inside `verify`'s return — every
  other statement, every other function, and the `score = R * Q^2 * T` expression are
  identical, with `circuit.reaction_depth()` occurring exactly once in the module. A
  future edit that stubs a scoring line, or that widens the patch, fails there.
- **WHAT STILL CANNOT BE FIXED BY EDITING.** `scripts/measure_dqi_submission_score.py`'s
  module docstring says the same "unmodified" thing its output disproves; the record pins
  that file's SHA-256, so correcting the sentence means re-running both 102,400-shot
  verifications and re-pinning every displayed figure. Left as is, disclosed here and in
  the record's `scope`, which is the authority on what ran.

## 2026-10-30 — the status decision: pinned line frozen at 0.949054, `MR` the only reopen, no more field-arithmetic work

- **THE THREE ITEMS FROM THE REVIEW, CLOSED AS DECISIONS RATHER THAN LEFT IMPLICIT.**
  [`docs/BENCHMARK2_STATUS_DECISION.md`](BENCHMARK2_STATUS_DECISION.md). Item 1 — put the
  arm's own score on record — was already done and is now named as done (5.324728e12 for
  the six-lever arm, 2.990339e12 for the campaign's stacked arm, both through `verify()`
  at 102,400 shots). Items 2 and 3 were still just prose in a review, so they are now a
  decision record with reopen conditions:
- **D1 — the pinned competition binary is frozen at 0.949054.** The best correct circuit
  it can compile is 2.319438e13, 13.66× the derived target, and the reason no further
  work is planned is arithmetic: the multiply is 42.45% of its static stream, so deleting
  *all* of it would leave 8.29× out, and PyZX 0.10.7 leaves the multiply's T-count at
  378 → 378.
- **D2 — `MR` is the only condition that reopens it.** One opcode, no verifier change,
  specified/patch/local build already in the tree, worth 4.142× on the reference and
  4.5898× on the arm. If no upstream build ships a Z-basis read, the pinned line stays at
  0.949054 and nothing in this repository is "in progress" on it. If one does, the reopen
  is mechanical: rebuild, re-run the two scoring scripts, re-score.
- **D3 — no more field-arithmetic optimisation.** Even with every product and the
  controlled add removed the arm is 1.062× the target, because 472,768.2 Toffolis per shot
  are not field products; after the campaign the remaining 240,785.8 are the swap, the
  shift, `shift_step`, `update_flag` and the rest of the multiply window. The 23-AND
  dimension bound is recorded as the only remaining arithmetic lever, not as a task, and
  the decoder-algorithm work that could reach 445,229 is named as **not started**.
- **What continues is the method and the referee**, both of which are now enforced rather
  than described: [`REPRESENTATION_AWARE_UNCOMPUTATION.md`](REPRESENTATION_AWARE_UNCOMPUTATION.md)
  and `CircuitCandidate.correctness_evidence` with `EvidenceIncompleteError`.
- **Pins.** `tests/test_benchmark2_status_decision.py` (6 tests) holds the decision to the
  records: the frozen score and ratio against `dqi_submission_score.json`, both arm scores
  against their records, the 472,768.2 / 240,785.8 / 1.062 ceilings against the
  reachability note, and the fact that each decision names what would overturn it. The
  reachability note now points at the decision and no longer describes any frontier row as
  unmeasured; `docs/DOCUMENTATION.md` carries the same statement.

## 2026-10-30 — Experiment G, the external-optimizer check, and correctness evidence enforced in code

- **THE FRONTIER IS NOW A VERIFIER NUMBER: 2.990339e12, 1.762× the target.**
  `scripts/measure_dqi_campaign_arm_submission_score.py`, record
  `benchmarks/quantum_compilation/dqi_campaign_arm_submission_score.json`. The campaign's
  stacked arm — six levers plus the five schedule/control gates, `mr` clears, 54-AND
  policy — through the competition's `verify()` at **102,400 shots**: **PASS with 0 value
  and 0 phase errors**, **685,936.2051367187** Toffolis per shot at `Q` = 1,913 (its
  recorded 686,014.8 to 1.1e-4), score **2.990339e12** — **0.12236** of the pinned
  reference, **0.5616** of the six-lever arm. `Q` is what keeps it out: at 1,913 the
  target's 445,229 would have to be 509,088. Build 9.1 s, verify 279.1 s. Not a receipt.
- **THE EXTERNAL COMPARISON, RUN AND SCOPED.**
  `scripts/compare_gf_mul_external_optimizer.py`, record
  `benchmarks/quantum_compilation/gf_mul_external_optimizer.json`. The multiply's forward
  netlist is recorded from the emitter itself (54 ANDs, 1,004 CNOTs, 54 clears at 0 T) and
  given to **PyZX 0.10.7** as the same function in Clifford+T, after the 7-T Toffoli block
  is verified on all eight basis states (max error 3.1e-16). Measured: the Clifford pass
  leaves **T at 378 → 378** and compresses only CNOTs (1,328 → 824); `full_optimize` **does
  not return within 90 s** at 99 qubits and 378 T. A unitary optimizer must also uncompute,
  so its honest figure is **108 ANDs = 756 T** against our 54 ANDs + 12-Toffoli add at 0 for
  the clears. The decoder cannot be offered at all: `HMR` and `INFO_TAG` are in its stream
  and the arm clears with `mr`, which no Clifford+T optimizer models.
- **CORRECTNESS EVIDENCE IS NOW A RUNTIME GATE.** `CircuitCandidate` carries
  `correctness_evidence`; `with_status` raises `EvidenceIncompleteError` for
  `VERIFIED_NONIMPROVING`/`VERIFIED_IMPROVEMENT` unless value, junk and phase are each
  measured-and-passed, covered by another passed check, or `not_applicable` with a reason
  the verifier supports. `ledger.record_verification` derives the evidence from the receipt
  and records `INSUFFICIENT_EVIDENCE` instead of promoting a claim it cannot support: the
  DQI verifier folds the ancilla read into `value_errors` (so junk is `covered_by` value,
  with the verifier's own rule quoted), while an invsqrt/dfthcsos receipt states that it
  prices the circuit rather than simulating it. A failing check under a correctness claim
  raises, and a per-shot verifier cannot waive all three. This is the checklist's gates 3,
  4 and 7 in code.
- **Pins.** `tests/test_dqi_campaign_arm_submission_score.py` (6 tests) pins the `PASS`,
  both zero error counts, the budget, `T` to 0.15% with the exact static count,
  `score = R·Q²·T`, the 0.122357 / 0.561595 / 1.762 ratios and 509,088 needed, the six
  levers and two mixins by name, and the "not a receipt" statement.
  `tests/test_gf_mul_external_optimizer.py` (7 tests) pins the 54/1,004/54 netlist, the
  verified decomposition, `T 378 → 378` with CNOTs 824, the recorded timeout, the 108-AND
  unitary accounting, and the decoder's `HMR`/`INFO_TAG` inventory.
  `tests/test_candidate_correctness_evidence.py` (11 tests) pins the gate, including the
  two ledger outcomes.

## 2026-10-30 — Experiment F: the arm's score measured (5.324728e12, 3.137× the target), the closing statement, and the reusable checklist

- **THE ARM'S SCORE IS NOW A VERIFIER NUMBER, NOT A PRODUCT.**
  `scripts/measure_dqi_arm_submission_score.py`, record
  `benchmarks/quantum_compilation/dqi_arm_submission_score.json`
  (`docs/DQI_OFFICIAL_COMPATIBILITY.md` §9). Every earlier figure for the six-lever arm
  was `R · Q² · T` multiplied out by hand, on a build the competition does not ship.
  Built on the local `MR` rebuild (`scripts/build_kickmix_mr.py` + the opcode patch)
  with the 54-AND policy and run through the competition's own `verify()` at **102,400
  shots**, `log2_delta = -128`: **PASS with 0 value and 0 phase errors**,
  **T = 1,221,408.044756** per shot at `Q` = 1,913 (its recorded 1,221,466.8 to 4.8e-5;
  static 3,132,510, 76,883,226 ops), `R` = 1.191260, score **5.324728e12**. Against the
  pinned reference's 2.443946e13 at the same budget that is **0.2179** — 4.5898×
  cheaper, exactly the `Q²T` ratio. Build 6.7 s, verify 236.8 s.
- **AND IT IS STILL 3.137× THE TARGET, WHICH IS THE POINT.** The derived Benchmark II
  aim is `R · 1,789² · 445,229 = 1.6975e12`; at `Q` = 1,913 the same score needs
  `T` = **509,088**, and the arm executes 1,221,408 (2.743× on `T`, 1.143× on `Q²`,
  3.137× on the score). So `MR` is **necessary and not sufficient**. The record says in
  `status`, `kickmix_role` and `scope` that it is **not a receipt**: only the pinned
  extension may produce one, and it has no `mr`.
- **THE CLOSING STATEMENT, WITH ITS CEILINGS.**
  [`docs/BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) §5 puts
  the frontier in the competition's unit: pinned binary 2.319438e13 (**13.66×**), `MR`
  rebuild 5.324728e12 (**3.137×**), the campaign arm ≈2.99e12 (**≈1.76×**, not yet run
  through `verify()`), target 1×. It states the three independent reasons — no multiply
  work closes the pinned gap (deleting the whole 42.45% share leaves 8.29×), `MR` moves
  the order of magnitude but stops at 3.137×, and the remaining 240,785.8 Toffolis are
  decoder algorithm (swap 64,224, shift 96,584.6, `shift_step` 30,912, `update_flag`
  30,240) — and lists what would overturn it.
- **THE METHOD, AS A NINE-GATE CHECKLIST.**
  [`docs/REPRESENTATION_AWARE_UNCOMPUTATION.md`](REPRESENTATION_AWARE_UNCOMPUTATION.md)
  states the pattern that worked (`del_gf2_mul` clears a product register for 0 Toffolis
  once the register really holds the reduced product) and turns it into gates a
  synthesizer can enforce, each with the failure it prevents: quote the precondition,
  state the object, prove every path, check the phase contract, check the qubit ledger,
  predict the delta and pin the identity, verify value/junk/phase, score with the real
  verifier, and record what was not measured. The three measured cases sit beside it,
  `hmr` among them as the counterexample (24/64 ones on `|1>`, 37/64 phase errors as a
  clear), and the hooks are named (`CircuitCandidate`, `CandidateStatus`,
  `referee.run_dqi_verifier`, `dqi_verification_scope`, `contracts.py`).
- **Pins.** `tests/test_dqi_arm_submission_score.py` (8 tests) pins the arm's `PASS`,
  both zero error counts, the budget, `T` to 0.15% with the exact static count,
  `score = R·Q²·T`, the like-for-like 0.2179 ratio, the 3.137× gap with 509,088 needed,
  and the "not a receipt" statement. `tests/test_representation_aware_uncomputation.py`
  (6 tests) holds the checklist's three cases to the records that measured them and
  asserts the nine gates and the plug-in points are present.

## 2026-10-30 — audit of the quoted figures: the ratio is 0.949054, S3b's saving is 45,396.1, and the probe's ranges are 27-36 / 36-45

- **THE 0.949854 FIGURE DOES NOT EXIST.** An external review reports a
  "0.949854" score ratio and asks for it to be corrected against the record. Checked
  rather than assumed: `git log -S0.949854` over the whole history returns nothing, no
  record or note contains it, and every frozen figure is **0.949054** —
  `dqi_official_candidate.json` stores 0.9490535630117923 for the 64-shot comparison
  (5,433,399.609375 / 5,725,071.609375) and `dqi_submission_score.json` stores
  0.9490542129538326 for the 102,400-shot one (5,433,438.229658 / 5,725,108.382109).
  Both recompute from their own rows, and `score == R(eps_plus) * Q^2 * T` holds to
  1e-12 on both verified rows, with `R` and `Q` ratios of exactly 1.0. There was
  nothing to correct. The scope of that check is the repository: a history search
  can show that no record here ever carried the string, not what an external report
  printed, so the honest reading is that 0.949854 is a misreading of 0.949054 rather
  than a value this repository ever published.
- **S3b IS QUOTED ON ITS OWN BASE AGAIN.** `docs/EEA_SCHEDULE_CAMPAIGN.md` quoted
  **45,377.7** for the campaign's own `drop_shift_c3` arm — that is
  `dqi_movement_schedule.json`'s re-measurement on a base 2,952 Toffolis lower
  (1,218,514.8 → 1,173,137.1). The campaign's record measures **45,396.1**
  (1,221,466.8 → 1,176,070.7). Both figures are real and both stay in the note, but
  the S3b row, the §3.3 paragraph and the summary line now name the record each one
  belongs to. The producer script's two comments were left untouched: the record
  pins its SHA-256, and a comment edit would invalidate that pin without a re-run.
- **THE PROBE'S RANGES WERE NARROWER THAN THE DATA.** The 64-shot probe in
  `dqi_movement_schedule.json` reports 27-**36** exchange firings per shot and 36-**45**
  `c3` rounds with the flag clear (totals 2,052 and 2,556 unchanged); the notes said
  27-33 and 39-45 in three places. Corrected in
  [`docs/DQI_MOVEMENT_SCHEDULE.md`](DQI_MOVEMENT_SCHEDULE.md) §6 and
  [`docs/EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md).
- **Pins.** `tests/test_dqi_movement_schedule.py` now derives both ranges from the
  record and asserts the notes carry them, and `tests/test_dqi_schedule_campaign.py`
  derives S3b's 45,396.1 from the campaign record's own two `T` values and asserts the
  note carries it — so neither figure can drift back without a failing test.

## 2026-10-30 — Experiment E: the competition's own verifier scores the candidate, 0.949054 = a 5.09% improvement

- **THE SUBMISSION QUESTION CLOSED BY RUNNING `verify()` ITSELF.**
  `scripts/measure_dqi_submission_score.py`, record
  `benchmarks/quantum_compilation/dqi_submission_score.json`
  (`docs/DQI_OFFICIAL_COMPATIBILITY.md` §8). Experiments D and D2 stopped at "not a
  submission until it is run through the verifier's shot sample end to end", scoring the
  candidate with the verifier's *rule* by hand over 64 shots. This runs the shipped
  `verify()` — every scoring line unmodified — on the reference and the reduced-window
  54-AND candidate at the competition's own budget, **102,400 shots** (400 batches of 256)
  and `log2_delta = -128`. Both **PASS** with **0/102,400 value and 0/102,400 phase
  errors**; `eps_plus` is 8.660587e-4 for both and `R` 1.191260, so the scores are
  **2.443946e13 → 2.319438e13**, a ratio of **0.949054** (0.94906) — a **5.09%
  improvement**, and exactly the Toffoli ratio, because equal `Q` and equal zero-error
  `eps_plus` leave nothing else in the comparison.
- **THE BUDGET IS LOAD-BEARING, AND THE RECORD SAYS WHY.** `eps_plus` at
  `log2_delta = -128` is `1 - 2^(-128/n)`, so a *zero-error* run is refused below ~1,375
  shots: at 256 the same two circuits get `eps_plus` 0.2929 and `AssertionError … >= 1/16`.
  The smallest batch-aligned budget a zero-error run passes at is **1,536**; the record
  verifies at the full 102,400 (fraction 1.0) and reports the zero-error bound at both.
- **THE 5.09% DECOMPOSED TO THE GATE.** 32,408 emitted multiply calls × 9 Toffolis =
  **291,672** static — exactly the measured static delta (5,725,461 → 5,433,789), and
  291,670.152 on the executed means, with `CCX`/shot equal to the executed delta. The
  multiply is **42.45%** of the reference's static Toffoli stream and the candidate removes
  **5.0943%** of that stream (5.0946% of the executed mean). The `CX`/`HMR`/`CZ` deltas are
  the gate-level re-encoding (104,555,982 operations against 86,051,014), not savings.
- **THE LEVER BOUNDARY MEASURED ON THE SURFACE, NOT ASSERTED.** Of the **29** methods in
  the pinned extension whose signature mentions a classical bit (`km.b`), exactly **one**
  writes a bit from qubit data (`hmr`), and on `|1>` it wrote 1 on **24 of 64** shots and
  flipped the tracked phase on exactly those 24 — a measurement, not a read. Every other
  method only consumes a bit as a condition, so no pinned primitive can express a lever's
  gate. `reaction_depth` (13.9 GB peak, 136 s) is the one diagnostic stubbed, and it is not
  part of the score; every scoring line is the shipped one.
- **WHAT IS STILL OPEN IS NOW ARITHMETIC.** The candidate's 2.319438e13 is **13.66× above**
  the competition's 1.6975e12 target (the reference is 14.40× above it). The remaining
  factor is the schedule, and the instruction it needs is `mr` — unchanged by this record,
  which does not claim a receipt.
- **Pins.** `tests/test_dqi_submission_score.py` (8 tests) pins both `PASS` rows at
  102,400 shots, the shared `eps_plus`/`R`, the 1,536 floor and the 256-shot refusal,
  `score = R · Q² · T` recomputed per row, the 0.94906 ratio equal to the Toffoli ratio,
  the 291,672 decomposition against both streams, the 29-method surface with `hmr` the only
  writer and not a read, and the `reaction_depth` patch. `docs/DQI_OFFICIAL_COMPATIBILITY.md`
  §8 and `docs/DOCUMENTATION.md` state the same figures.

## 2026-10-30 (final) — the first *correct* compilable candidate: 5,433,399.6 at `Q` = 1,893, a 5.09% score improvement

- **A PHASE-SAFE 54-AND MULTIPLY THE PINNED BINARY CAN ACTUALLY EMIT.**
  `scripts/gf54_mul_reduced.py`. Experiment D2 closed the `hmr` route and left the
  question open: a clear needs the value to be *known*, and `del_and` knows only an AND.
  The pinned binary has a **second** fixup primitive — `del_gf2_mul` (`gen_gf_unmul`) —
  whose classically conditioned `CZ` parity is computed from the two factors, so it can
  clear a **product** register for 0 Toffolis. Its precondition is that the register
  really holds `lhs * rhs`, which the old emitter never met: it accumulated an unreduced
  `2n-1`-bit polynomial window and folded afterwards. Here the field product is
  accumulated **already reduced** — each base-case product bit lands at a polynomial
  index that maps to a known set of field bits, and CNOTs are free — so the only window
  is 12 qubits and holds exactly the reduced product. Every AND target is cleared by
  `del_and` on the same footing. Measured at the pinned instance on the pinned
  extension over 64 shots: **`Q` 1,893 (the reference's peak — the window fits under
  it), 5,433,399.609375 Toffolis per shot, 0 value errors, 0 phase errors, junk clean,
  syndromes restored, score ratio 0.949054 = a 5.09% improvement.** That is the first
  correct candidate cheaper than the shipped reference the competition build can
  compile, and it beats Experiment D's withdrawn 0.952 because it does not pay for the
  multiply in qubits. Cost: 66 Toffolis per controlled call against the built-in 75
  (54 ANDs + the 12-Toffoli add); `mr` and `hmr` appear nowhere in the stream, and every
  measured clear is paired with a fixup.
- **THE ARITHMETIC THE REVIEW ASKED ABOUT, STATED.** `T` falls by **5.0946%**
  (`5,433,399.609375 / 5,725,071.609375 = 0.949054`) and the *score* ratio is
  `(T/T₀)·(Q/Q₀)²`: at `Q` 1,896 that was `0.949054 × 1.003172 = 0.952064`, a 4.79%
  improvement, the `Q` term costing 0.317%; at the reduced window's `Q` 1,893 the term
  is exactly 1 and the score ratio equals the `T` ratio. The record now says which
  quantity each figure is.
- **EVERY RECORD AUDITED FOR UNCHECKED CORRECTNESS.**
  `scripts/audit_record_correctness_checks.py` walks all 21 records and classifies every
  candidate row by which of the three checks — value, junk, phase — the record actually
  stores. Result: **13 rows claim correctness on partial evidence** (7 missing junk and
  phase, 6 missing phase), 12 fully checked, 51 bare booleans, 5 resources-only; and the
  historical defect is named mechanically — `dqi_official_compatibility.json`
  `builds.pinned.candidate`, the 5,433,399.6 row that was called correct on its
  syndromes alone. Record: `benchmarks/quantum_compilation/correctness_audit.json`;
  pinned by `tests/test_record_correctness_audit.py`.
- **Pins.** `tests/test_gf54_mul_reduced.py` (4 tests, on the pinned extension) pins the
  emitter's function on both paths, the junk and the tracked phase, 54/66 Toffolis, no
  Z-basis read, and a fixup for every `HMR`. `tests/test_dqi_official_candidate.py` gains
  the 0.949054 row and the claim that it is the only correct arm cheaper than the
  reference. `docs/DQI_OFFICIAL_COMPATIBILITY.md` §3/§5/§7 and `docs/DOCUMENTATION.md`
  carry the revision.

## 2026-10-30 (later) — Experiment D2 and the campaign fix: `hmr` is not a clear, the compile-able candidate is not a submission, and the fused-shift arms finally measure something

- **EXPERIMENT D2 — D's CANDIDATE IS COMPILABLE AND WRONG, AND THE 0.952 IS WITHDRAWN.**
  `scripts/measure_dqi_official_candidate.py`
  (`docs/DQI_OFFICIAL_COMPATIBILITY.md` §3 and §5) checks the one thing Experiment D did not:
  the **phase**. The DQI submission is scored on phase as well as value (the verifier's
  `phase_err = (phase_flipped ^ phases) & ~val_err`), and a clear implemented as `hmr` is not
  phase-safe. Measured one qubit at a time over 200 shots: `hmr` on `|1>` flips the tracked
  phase on **97** of them — the same 97 as its measured ones, because an X-basis measurement of
  `|1>` puts a −1 on the branch that measures 1 — while `del_and`, which is the same
  measurement followed by a classically conditioned `CZ` (the phase fixup), flips **0**, and
  neither flips anything on `|0>`. So a clear whose value is not a *known AND* has no fixup to
  apply, and substituting `hmr` for `mr` there is a randomisation, not a clear. At the pinned
  instance over 64 shots on the pinned extension: reference **1,893 / 5,725,071.609375, 0 value
  errors, 0 phase errors**; Experiment D's 54-AND candidate **1,896 / 5,433,399.609375, 0 value,
  37 phase errors**; and a third arm that removes the reverse EEA outright — the forward EEA on a
  **CNOT copy** of the syndromes, so the input is never written, nothing is restored and no
  qubit's value is read, which is exactly what the `clear_eea` lever needs — at **2,736 /
  3,438,873.609375, 35 phase errors**, twice over: its clears are `hmr` too, and holding the
  whole scratch live costs more `Q` than the pass it deletes. D's candidate builds, executes, and
  is **not** a submission; its **0.952 score ratio is withdrawn** and its Toffoli count stands.
  The measured floor is now the shipped reference: **the cheapest *correct* circuit the pinned
  binary can compile is the one it ships.**
- **EXPERIMENT C, C3 — THE MECHANISM IS MEASURED, AND IT IS THE FLAG.** The probe now reads both
  of the exchange's operands where the exchange reads them — `c3` and the *post-update* `flag`,
  one qubit per round — at the pinned instance over 64 shots and 222 rounds: `c3` fires **4,608**
  times (**72.0 on every shot**, invariant), the exchange fires **2,052** (27–33 per shot), and
  the flag is **clear in 2,556 of the 4,608** `c3` rounds (55.5%). So the flag is not redundant in
  a `c3` round; exchanging on `c3` alone performs ~72 permutations where the decoder asks for
  27–33, which is why C3's 73.0 never appeared and its arm (852,267.4, exchange row 190,888.0)
  is wrong. The same table is why the campaign's S3 arm has nothing to skip: in the 2,052 rounds
  where the flag *is* set, `ctrl`'s one-hotness zeroes the shift's other two terms.
- **THE CAMPAIGN'S DEAD `_shift_a_and_b` IS FIXED AND ITS TWO ARMS RE-MEASURED.** The method is
  now named `_shift_b_array_right`, the one the reference and `MRGatedDecoder` call, so the gates
  inside it are emitted. `dqi_schedule_campaign.json` was re-run with everything else reproducing
  to the Toffoli (control **1,221,466.8**, `everything` **688,966.8**, `stacked` **686,014.8**):
  **S3 `composed` saves exactly 0.0 and is correct** — the campaign's conclusion, now measured
  rather than assumed — and **S3b `drop_shift_c3` saves 45,396.1 and is incorrect**, which
  falsifies S3b's premise (the `¬flag ∧ c3` branch is live). `drop_shift_c3` is recorded as a
  *predictive* arm: its correctness is reported, not required, so a falsification is the result
  rather than an invalid run, and the S3b check now carries the declared 0.0 beside the measured
  45,396.1. The movement script's independent re-measurement against a base carrying the two
  promoted gates (`campaign_none` = 1,218,514.8) agrees: 0.0/correct and 45,377.7/incorrect.
- **Pins.** `tests/test_dqi_official_candidate.py` (8 tests) pins the phase mechanism, the three
  arms' `Q`/`T`/error counts, that no ratio is quoted for a phase-broken candidate, and that the
  cheapest correct circuit is the reference; `tests/test_dqi_movement_schedule.py` and
  `tests/test_dqi_schedule_campaign.py` were updated to pin the rebuilt arms and the exchange
  predicate rather than the vacuous 0.0s. `docs/DOCUMENTATION.md` indexes the new script;
  `docs/EEA_SCHEDULE_CAMPAIGN.md` §3.3 and `docs/DQI_MOVEMENT_SCHEDULE.md` §5–§7 carry the
  corrections.

## 2026-10-30 — Experiments C and D, and the amplitude tier: the exchange stays, one candidate compiles, and the licence is exactly the basis

- **EXPERIMENT C — MEASURED, NEGATIVE, WITH A CORRECTION.**
  `scripts/measure_dqi_movement_schedule.py` (`docs/DQI_MOVEMENT_SCHEDULE.md`) asks the only
  question left open by the schedule campaign: can the decoder's A/B exchange — 64,224
  executed Toffolis per shot, 72 controlled 892-qubit permutations — be resolved at **build
  time** by reassigning which physical register plays the `A` role? That is the only kind of
  saving the pinned competition binary could compile, because it is in the emitted circuit
  rather than in an executed count. The prediction was declared before any arm existed: the
  rebased arm must save exactly the region (64,224.0 executed, **198,024 static**), and is
  correct *if and only if* the exchanging-round set is the same for every shot. At the pinned
  instance over 64 shots: control **686,014.8** at `Q` 1,913 with its 64,224.0 row; a harness
  arm that re-emits the incumbent's exchange from the new script reproduces it to **±0.0**, so
  the comparison is licensed; the rebased arm zeroes the region and drops 198,024 static
  Toffolis and is **incorrect** (syndromes restored, junk and phase not); and the probe reads
  `c3` once per round to find **10 distinct patterns across 64 shots** while every shot
  exchanges exactly **72** times. The schedule is therefore *cardinality-stable and
  order-unstable* — that is why the incumbent's row is exactly `72 × 892` on every shot and
  why no build-time period reproduces it. The arm stays at 686,014.8, 1.541× the target.
- **CORRECTED — two of the campaign's arms measured nothing.** `composed` (S3) and
  `drop_shift_c3` (S3b) both reported 0.0, which the campaign read as the extra condition
  being unsatisfiable. It is not: the code implementing both lives in a method named
  `_shift_a_and_b`, and nothing calls it (the reference and `MRGatedDecoder` call
  `_shift_b_array_right`). Measured at m=15/n=6/GF(2^4): `none`, `composed` and `drop_shift_c3`
  emit **149,337 instructions and 16,610 static Toffolis each** — the two “arms” are the control
  circuit, and their 0.0 is vacuous. The predicate argument (a one-hot `ctrl` and one flag value
  cannot be both) survives; the measurement that was supposed to support it does not, and
  `docs/EEA_SCHEDULE_CAMPAIGN.md` §3 now says so.
- **EXPERIMENT D — THE MISSING `MR` WAS NOT THE ONLY BLOCKER, AND ONE CANDIDATE COMPILES.**
  `scripts/measure_dqi_official_compatibility.py` (`docs/DQI_OFFICIAL_COMPATIBILITY.md`) mines
  the whole surface of both builds. The pinned competition binary has **no `mr`, no `h`, no
  `gf_product`, no per-instruction counts**, but it *has* `hmr`, `bit_store`,
  `push_condition` and `init_and`/`del_and`/`cswap` — and `del_and` already costs **0**
  Toffolis there, because an AND clear can be a measurement where a read cannot. So the arm
  needs **two** surfaces it does not have, not one: the 54-AND product policy is worth
  **523,849.1** per shot on the reference and was never recorded as a blocker. Both builds
  count the same reference identically (1,893 / 5,725,071.609375, per-shot arrays equal), and
  the pinned row reproduces the recorded constant exactly. The positive result: a candidate the
  pinned binary **can** build — the reference algorithm with the 54-AND multiply emitted as
  gates and its workspace cleared with `hmr` instead of `mr` — at `Q` 1,896 and
  **5,433,399.609375** per shot, syndromes restored on all 64 shots, a **0.952** score ratio to
  the shipped reference (**4.79% better**) and still **7.92×** worse than the arm. It is the
  first candidate in this repository the competition build can construct at all, and it is a
  floor rather than a contender: the arm's −43.84% is a read-dependent saving, and a quantum
  control cannot express it (the reference's every gate is quantum-controlled and every one
  executes).
- **THE AMPLITUDE TIER, AND THE LICENCE IT PINS.**
  `apedp_dyn/quantum_compilation/statevector_sim.py` plus
  `scripts/measure_vqc_licence_boundary.py` (`docs/VQC_AMPLITUDE_TIER.md`) compare the
  controlled gate the decoder emits with the read-then-gate map the arms substitute, at
  amplitude level, because every count in this repository is a basis-state count and the
  campaign's licence (P1) is a statement about the *inputs*. Result: on **all eight basis
  inputs** the two maps are the same state up to global phase, the classical bit equals the
  control, and the ancilla returns to `|0>`; on a **superposed control** the gated map branches
  **twice at 0.500** with a fidelity of **0.500** against the coherent reference, and the
  control's off-diagonal goes **0.500 → 0.000** — a complete dephasing, not a perturbation. The
  same tier settles the substitution Experiment D needed: `hmr` alone returns a random outcome
  on `|0>` (it measures in the X basis), while `h; hmr` is `mr` branch for branch on basis *and*
  superposed inputs — so the blocker is the missing Hadamard, not the measurement instruction.
  The tier is cross-checked against `kickmix` on the gadget's eight basis inputs (identical),
  and the arm's 1,913 qubits are reported as 10^575.87 amplitudes rather than pretended.

## 2026-10-30 — The schedule campaign: read the predicate, then move; 1,221,466.8 → 686,014.8

- **EXPERIMENT B — MEASURED AND CLOSED, NEGATIVE.** The product stream's own operand
  predicate: every surviving product is `ixor_gf2_mul(q_i, a_j)`, so a zero multiplier makes
  it the identity, and the argument that licenses S1–S4 licenses gating on it (`S6_zero_q`,
  declared **0.0 saving** before the arm was built). The arm is **correct** — the predicate is
  admissible — and **loses**: 997,795.6 Toffolis per shot, **+311,780.7 over the five-gate
  incumbent**, adding 327,094 mid-circuit reads (357,347 in total against 30,253) and 956,116
  instructions. Two structural reasons, both measured: (a) the sweep's selection chain
  `cswap`s `q_i` into the a-lanes, so the multiplier is **not** a round-level quantity — the
  first version of this arm read it once per round and was *incorrect*, and only a
  per-position read is sound; (b) a per-position read measures at 0.95 Toffolis against a
  66-Toffoli product that is skipped only inside the `mul` window, so the gate needs the
  multiplier to be zero in ~3% of live positions, and the loss is almost exactly the read
  cost — the multiplier is essentially never zero where the product executes. Its falsifier
  (the inverted condition) is wrong and informative (316,701 product Toffolis skipped). The
  331,023 surviving products are work the data path asks for; removing them needs a cheaper
  multiply, not another classical read.
- **MEASURED — the shift/sweep/swap priority is discharged, and it was worth 43.6% of the
  arm.** `scripts/measure_dqi_schedule_campaign.py` (`docs/EEA_SCHEDULE_CAMPAIGN.md`) applies
  the read-then-gate gadget at per-position grain: every predicate register is a basis state,
  so `cx` + `MR` through the shared ancilla yields a deterministic classical bit at zero
  Toffoli and no phase, and a gate conditioned on a zero bit costs nothing on that shot. At
  the pinned instance, 64 shots, `Q` 1,913 and 222 rounds in every arm: the sweep's one-hot
  quotient selection **−103,968.0** (chain 105,696.0 → 1,728.0), the shift's `do_shift` mask
  **−65,895.0** (chain 131,220.0 → 65,325.0), the product's `mul` control **−362,637.0**
  (product stream 693,660.0 → 331,023.0); all three in one arm **1,221,466.8 → 688,966.8**
  (−532,500.0, exactly additive), **1.547×** the 445,229 target instead of 2.744×, `Q²T`
  4.4700e12 → 2.5213e12. Record: `benchmarks/quantum_compilation/dqi_schedule_campaign.json`.
- **STACKED WITH THE PRIOR COMBINED ARM.** The same record now re-measures the
  flag/metadata gate the previous session promoted (2,952.0 → 1,218,514.8, zero reads) and
  puts all five gates in one circuit: **1,221,466.8 → 686,014.8 per shot (−535,452.0,
  −43.84%)**,  exactly `532,500.0 + 2,952.0`, so the two sessions' gates do not interact,
  **1.541×** the target, `Q²T` 4.4700e12 → 2.5105e12. The cross-report inference "the two
  gates compose" is now a measurement in one record — and a declared one: `prediction.S5_stack`
  records the arithmetic (`532,500 + 2,952 → T = 686,014.8`) written after the schedule arms
  and **before** the five-gate arm was built, and the measurement matched it to **0.0**
  Toffoli (`observation.stacked_error_per_shot`).
- **NOT RELOCATED — REMOVED.** Counting the whole stream on the same runs: executed
  instructions fall 37,808,019.7 → 32,451,795.1 per shot (−5,356,224.6, −14.17%) while the
  Toffolis fall 535,452, i.e. 10.12 instructions per removed Toffoli — the expansion the
  removed `CCX`es had. The added cost is 30,253 mid-circuit reads per shot at 2 instructions
  each (60,506, 1.1% of what was removed), reported for every arm. The honest claim is: the
  nonlinear work falls 43.84%, the stream falls with it, and the price is measurement depth.
- **DECLARED BEFORE MEASURED.** The predictions are computed from the control arm's own chain
  costs and printed before any variant exists (`prediction.written_before_variants`): S1 ≥
  100,368 from a burn of `12 × 2 × 222`, S2 **no saving** (`do_shift` is the XOR of three
  threshold intervals, not a one-hot), S3 and S3b exactly 0.0, S4 no saving. S1 came out at
  103,968.0 (the bound assumed two selections in all 222 rounds; 144 occur); S2 at 65,895.0,
  which **closes** the shift half rather than promoting it — half the chain is movement the
  data path needs; S3/S3b at 0.0 with the arm still correct, which is the measured form of
  "the shift and the swap cannot be fused for gain".
- **FALSIFIED THREE WAYS.** Every variant was rebuilt with its condition made wrong and all
  three are incorrect: the stale selection (skips 104,768.1 of its chain and *costs* 217,720.1
  more than the control, because a wrong state changes the control flow), the inverted mask
  (skips 69,311.6 and saves 79,854.5 — more than the honest gate's 65,895.0, which is what
  makes it wrong), and the previous slot's mask (70,351.1 skipped, 80,612.0 saved, wrong).
  The record measures skipped work on the chain the condition gates, never on the arm total,
  and a falsifier that skips nothing is recorded as vacuous instead of as a pass.
- **CORRECTED — what the arithmetic of the reachability note covers.** §3 of
  `docs/EEA_AND_PREDICATE_PLAN.md` counted the *static* residual, so its "`C` alone is 1.062×
  the target" was read as "control-logic work cannot help". The executed residual is not
  `C`: 532,500 of it was quantum-controlled work whose control was zero. The target is still
  out of reach — 686,014.8 is 240,785.8 above it, and the largest remaining items are the
  swap's 64,224, the shift's 96,584.6, `_update_flag`'s 30,240 and `shift_step`'s 30,912 — but
  it is now 1.541× away, not 2.737×.
- **STATED COST MODEL.** An executed count that depends on classical conditions is a count
  under adaptive execution. The benchmark permits it (the reference's own `error_mbuc` is
  classical feed-forward, and the six levers were promoted on the same mechanism), but this
  campaign applies it at per-position grain — thousands of mid-circuit measurements and
  classical decisions per shot. A submission that forbids mid-circuit classical control could
  not claim these counts, and none is claimed. No receipt exists either: the arm still needs
  `MR`, which the pinned competition binary does not have.

## 2026-10-29 — The two gates stack to 2,952; the GA leak and the search ranker are fixed

- **MEASURED — the two promoted gates compose, and their savings do not add.**
  `scripts/measure_dqi_combined_arm.py` runs the predicate-reuse arm and the halting-round
  gate in one circuit. The prediction is written before the combined arm is built, from that
  run's own control and halt-gated arms: the reuse saves 8 Toffolis only in rounds where
  `_update_flag` executes, and the halt gate removes it from six dead rounds, so
  `8 × 216 + 1,224 = 2,952`, not the naive `1,776 + 1,224 = 3,000`. Observed: **2,952**
  (1,221,466.8 → **1,218,514.8** per shot, `Q` = 1,913, correct, 222 iterations in all three
  arms, region rows partition each total). The 48-Toffoli difference is the interaction, and
  the record names which model matched. Record:
  `benchmarks/quantum_compilation/dqi_combined_arm.json`.
- **FIXED — the evolutionary search no longer selects on the confirmation rows.**
  `evolve()` picked `grouped_pick` by minimising error on the supplied `x_test`/`y_test`,
  which turned the confirmation split into a second training set. Selection is now a
  deterministic 5-fold inner cross-validation over the training rows
  (`SearchConfig.selection_folds`), with coefficients refitted inside each fold; the test
  rows are read only after the pick is fixed, to report the new
  `SearchResult.confirmation_mse`. `selection_method`/`selection_folds` record which path
  was taken, and the fallback when no candidate survives the folds is the lowest training
  error — never the test rows. Both the native and Python paths were changed. Pinned by
  `tests/test_evolutionary_search.py::test_grouped_pick_is_selected_without_reading_the_confirmation_rows`
  (inverting the confirmation targets must leave the pick identical and only worsen the
  confirmed error).
- **FIXED — the search ranker preferred the worse candidate.** `ValidationPipeline`'s
  `overall` was `0.4 * max(0, cv_mean - cv_std) + ...` with `cv_mean` an *error*: a perfect
  fit scored 0.56 and a poor one 0.83, so `HierarchicalSearch.search` sorted its worst
  candidate first. `overall` is now `0.4 * max(0, 1 - (cv_mean + cv_std)) + ...`, recomputed
  in Python from the returned components on the Rust path too (the kernel carried the same
  sign and is corrected in `validation.rs`). Measured effect in
  `benchmarks/search_engine_gaps.json`: classic templates recovered 0/12 planted laws with
  the inverted score and 2/12 with the fix.
- **MEASURED — what the search templates cannot reach, and closed most of it.**
  `scripts/measure_search_engine_gaps.py` plants 12 law families, searches on 200 rows and
  scores the returned candidate on 100 untouched rows (coefficients refitted on train).
  Classic `HierarchicalSearch` recovered **2/12**, `beam_search_expressions` **1/12**.
  The extended template set (reciprocal, square root, cubic, `exp(±x)`, log, cosine,
  rational saturation, ratios, a variable in an exponent, plus an exact least-squares scale
  and offset fitted to every candidate) recovers **11/12**, and `beam_search_expressions(
  include_functions=True)` **9/12**. Genuinely still unreachable, and reported as such: a
  sinusoid of non-unit frequency (the engine fits affine coefficients, not a frequency).
  Records: `benchmarks/search_engine_gaps.json`.
- **ADDED — patent prior-art search with a provenance gate.** `apedp/patent_prior_art.py`
  indexes patent records whose claims carry a normalised formula and audits a candidate
  against them: rename-insensitive algebraic equivalence first, structural signature second,
  otherwise `NOT_FOUND_IN_PATENT_CORPUS`. A record that claims a real patent without an
  http(s) source and a retrieval date is refused at load (`PatentProvenanceError`), and
  claims with no formula are counted and reported (`NO_SEARCHABLE_CLAIMS`) rather than
  silently skipped. No patent corpus is bundled; the loader and the documented source
  options are. Pinned by `tests/test_dyn_patent_prior_art.py` (9).

## 2026-10-29 — The 80% is split: per-instruction counts, and two gates that pass

- **INSTRUMENTED:** the local `MR` build now counts executed shots **per instruction**
  (`sim.counting_op_executions` / `sim.operation_execution_counts()`), and the circuit
  exposes its `builder.mark(...)` boundaries and Toffoli-like mask (`circuit.info_tags()`,
  `circuit.op_toffoli_like`). The simulator applies the same per-shot condition mask as
  `operation_statistics()` and masks off the SIMD lanes beyond `batch_size`, so a region's
  sum reconciles with `num_toffoli_like` exactly rather than by construction.
- **MEASURED — the named 80% bucket is gone.** `scripts/measure_dqi_region_ledger.py`
  partitions the six-lever arm's 1,221,466.8 Toffolis/shot into regions that sum to the
  472,768.2 non-product residual: `shift_b_array_right` 162,479.6, `sweep_positions`
  143,038.0, `swap_a_and_b` 64,224.0, `_update_flag` 32,856, `_initial_shift_loop` 30,912,
  `_update_metadata` 12,432, `_update_halting_counter` 10,656, counter decode 4,662, the
  alignment pair 7,392, multiply-and-add control 2,016, quotient control 1,728, Forney
  control 372.6. It also records what each lever removed per region, and checks each
  lever's arm-level delta against the sum of its region deltas. Record:
  `benchmarks/quantum_compilation/dqi_region_ledger.json`.
- **PROMOTED — predicate reuse in `_update_flag`.** `scripts/measure_flag_predicate_reuse.py`
  shares the `(n_A == n_B)` comparison between the `c0` and `c1` sub-steps and clears the
  scratch with `MR` instead of uncomputing it: 8 Toffolis per iteration × 222 = **1,776
  per shot**, exactly the prediction computed from unit costs before the arms were built
  (the first prediction used the 12-bit field degree instead of the 8-bit register width
  and over-predicted by 4 per iteration; the mistake and the corrected run are in the
  record). `_update_flag` 32,856 → 31,080, `Q` unchanged at 1,913, `Q²T` −0.145%, all eight
  promotion gates pass, 222 iterations in both arms. Record:
  `benchmarks/quantum_compilation/dqi_flag_predicate_reuse.json`.
- **PROMOTED — the halting-round gate.** `scripts/measure_eea_oracle_ledger.py` gates
  `_update_flag` and `_update_metadata` on a classical read of `is_hc_zero`: both regions
  are controlled on the one-hot `c0..c3`, so in a round with a non-zero halting counter
  their net effect is the identity and skipping them is exact. It removes **1,224 per shot**
  (888 + 336), correct, `Q` unchanged. The predicate is already computed by the arm, so this
  is a real gate, not an oracle upper bound. Record:
  `benchmarks/quantum_compilation/dqi_eea_oracle_ledger.json`.
- **NARROWED:** together the two gates are 3,000 Toffolis/shot, 0.25% of the arm and 0.6% of
  the non-product residual. Most of the residual is not predicates — it is the sweep, the
  shift and the swap, which the six levers already gate to 64.8%, 68.4% and 32.4% of their
  static counts. No predicate edit closes the 2.744×; §3 of `EEA_AND_PREDICATE_PLAN.md`
  still stands.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` pins both records: the region rows
  sum to 472,768.2 with products separated, each lever's delta equals its region deltas,
  the flag-reuse observation matches its prediction, and all eight gates pass.

## 2026-10-29 — Ratio baselines labelled: 4.687× and 4.258× are different questions

- **CORRECTED:** the update quoted "4.687× against 4.320×" without saying which baseline
  each ratio used, which reads as a regression once a controlled comparison is made. The
  measure now records five comparisons, each with its baseline
  (`gf54_native_opcode.json:comparisons`): the multiply saves **9.15%** on the reference and
  **7.83%** on the gated arm; the arm is **4.687×** faster than the *unchanged* reference
  (the scored comparator) and **4.258×** faster than a reference given the same opcode; its
  `Q²T` falls **5.97%** from the built-in arm to the final one.
- **EXPLAINED:** 4.258× is below the built-in arm's 4.320× because the policy helps the
  reference more than the arm — the arm's Forney body is gated, so fewer of its inversion
  products execute. Both ratios are correct; only a reader told which baseline each uses
  can tell that they are not the same claim.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` recomputes all five from the
  recorded means and requires the same-policy ratio to sit below the built-in one, so a
  future edit cannot quietly swap the baselines.
- **NOTATION:** every figure in this work is GF(2^12) with 12-bit field elements
  (`DEFAULT_DEGREE = 12`); the two new notes spell it in ASCII so the superscript cannot be
  misread as 2^25.

## 2026-10-29 — Reconciled: the official 5,725,108.38 and our 5,725,071.6 are one metric

- **RECONCILED:** `scripts/reconcile_dqi_resource_baseline.py` (record:
  `benchmarks/quantum_compilation/dqi_resource_reconciliation.json`,
  note: [`docs/DQI_RESOURCE_RECONCILIATION.md`](docs/DQI_RESOURCE_RECONCILIATION.md)). The
  locked official reference row is 1,893 qubits / 5,725,108.382109375 Toffolis, and its
  fractional part is a multiple of 1/102,400 but **not** of 1/64 — so it is the verifier's
  default-shot mean, while this project's 5,725,071.609375 is a 64-shot mean. Same decoder,
  same metric (`num_toffoli_like`, which the verifier itself averages), same instance.
- **MEASURED:** the reference decoder was built and simulated on the **pinned competition
  binary** and on the local `MR` rebuild over the same 64 sampled syndromes. The per-shot
  Toffoli arrays are identical element by element and both means are 5,725,071.609375 at
  1,893 qubits, so the `MR` patch is count-neutral for a circuit that emits no `MR`. The
  36.77 gap to the official row is 0.78 standard errors of a 64-shot mean: a sample, not a
  discrepancy.
- **ESTABLISHED — no receipt for the arm.** Building the six-lever arm on the pinned binary
  fails with `AttributeError: 'Builder' object has no attribute 'mr'`: its clears need an
  opcode the competition build does not have, and `apedp_dyn/quantum_compilation/contracts.py`
  pins receipts to that build's hash. The arm's 1,221,466.8 at `Q` 1,913 is a local
  measurement, not an official result; `verify_submission_dqi.py` has no Toffoli threshold
  either — it returns `PASS` on correctness and reports `R(eps_plus) · Q² · T`.
- **DOCUMENTED:** what 445,229 is — a *derived target* (`Q = 1,789`, `T = 445,229`, chosen so
  the product reaches a 14.397× portfolio gain), not a measurement, a competitor's figure or
  an acceptance threshold. `docs/BENCHMARK2_TARGET_REACHABILITY.md` now says so before it
  compares anything to it.

## 2026-10-29 — Product window removed: 19 workspace qubits, `Q` 1,935 → 1,912, same T

- **MEASURED:** `gen_gf54_mul` no longer materialises the 2m-1-bit product window. Reduction
  is GF(2)-linear, so each Karatsuba sub-product is added to the target through a
  `x^d mod p` table as soon as it is computed; the peak scratch is one sub-product window at a
  time — **19 qubits at GF(2^12) instead of 42**, 12 + 19 when controlled instead of 54. The
  AND count is untouched, so `T_mean` is too: the two records differ only in `Q`.
  Correctness is unchanged and re-checked: multiply, inversion and division all 0 bad on 32
  random basis states, operands restored, workspace `|0>`, phase untouched.
- **MEASURED:** with the opcode emit at the pinned instance over 64 shots the reference stays
  **5,725,071.6 → 5,201,222.5 per shot (9.15%)** but `Q` goes **1,893 → 1,912** (was 1,935),
  and the six-lever arm **1,913** (was 1,936) at 1,221,466.8 per shot. The cost is 22 more
  CNOTs per controlled multiply (600 against 578): the reduction is applied per sub-product
  rather than once per window bit. Since the score charges `Q` squared, 23 fewer qubits at
  equal `T` is a 2.4% improvement in `Q^2 T` over the 42-qubit emit.
- **NEW:** [`docs/BENCHMARK2_TARGET_REACHABILITY.md`](docs/BENCHMARK2_TARGET_REACHABILITY.md)
  lists every opcode class against the 445,229 target with the measured arithmetic. (The
  two-term form first recorded here — `527,806.8 + 10,510 × (ands + 12)`, giving 2.012× at
  the 23-AND bound and 1.186× free — was superseded the same day by the three-term
  decomposition below: same totals, but it folded the 1,019.23 inversion/division products
  into its constant at 54 ANDs each, so "free multiply" meant "free `ixor` only". The
  correct figures are 1.941× and 1.062×.) It also records why no Fast-Fourier/cyclotomic
  selector was added: the Chien search's 69 constant multiplies per position are CNOT-only
  (0 Toffoli in `dqi_primitive_costs.json`), so a transform can only trade CNOTs for CNOTs
  and would pay qubits, which the score squares.

## 2026-10-29 — The reconciled non-multiply ledger, and the first experiment's gates

- **MEASURED — ledger.** `scripts/attribute_dqi_nonmultiply_tcount.py` (record:
  `dqi_nonmultiply_ledger.json`, note: [`docs/NONMULTIPLY_LEDGER.md`](docs/NONMULTIPLY_LEDGER.md))
  assigns every executed Toffoli of the six-lever arm to exactly one category and checks
  that they sum to the simulator's total: **693,660.0** (10,510 controlled multiplies),
  **8,382.7** (155.23 inversion products), **46,656.0** (864 division products) and
  **472,768.2** (non-product residual, 38.7%) = **1,221,466.8**. The static ledger reconciles
  too: 3,486,597 = 94,470 + 3,392,127.
- **BOUNDED, NOT GUESSED.** The residual's six unconditional regions are exactly 94,470
  (20.0%). The other **378,298.2 (80.0%) is one named unattributed bucket**: the simulator
  reports per-shot counts by operation type, not per operation, so the conditional EEA body
  cannot be split by region; the ledger records the gap, how to close it (per-op execution
  counts, or per-region oracle gating that measures the removable part) and refuses to
  estimate it into the table.
- **CORRECTED INVARIANT.** The review's accounting invariant of 527,806.8 is superseded by
  472,768.2 for the reason in the previous entry: 55,038.4 of the old constant was the
  1,019.23 inversion/division products, which are field arithmetic, not decoder logic.
- **NEW — first experiment protocol.** `_update_flag` plus the counter/halting predicates
  (43,734 Toffolis, 9.3% of the residual) against the unchanged 222-iteration arm, with a
  prediction written before the edit and eight promotion gates (exact output equivalence,
  measured phase, clean workspace, fixed iteration behaviour, both T-counts, `Q`, `Q²T`
  from the same run, frozen source and record). Pinned in
  `tests/test_dyn_mr_gated_arms_kickmix.py`.

## 2026-10-29 — Three-term decomposition, and the measured EEA/predicate plan

- **CORRECTED — same totals, honest labels.** The arm's Toffolis are
  `T = 472,768.2 + 10,510 × (ands + 12) + 1,019.23 × ands`, not
  `527,806.8 + 10,510 × (ands + 12)`. The 1,019.23 are the products inside the arm's
  inversion (155.23) and division (864), each 54 ANDs with **no** controlled add, so the
  old constant silently contained 55,038.4 of field arithmetic. Consequences: freeing
  every field product gives **1.345×** the target (not 1.469×), freeing the adds too gives
  **1.062×** (not 1.186×), and the 23-AND bound lands at **1.941×** (not 2.012×).
- **MEASURED:** the decomposition, by scoping `gf_product` to one entry point at a time on
  both the reference and the arm — the reference runs **888** divisions per shot (47,952
  Toffolis saved) and the arm **144** (7,776), and the arm's gated Forney block runs
  **31.05** inversions (1,397.11). `gf54_native_opcode.json:isolated_blocks` records all of
  it, and its `check_terms_sum_to_arm` is exact.
- **NEW:** [`docs/EEA_AND_PREDICATE_PLAN.md`](docs/EEA_AND_PREDICATE_PLAN.md) is the measured
  plan the next layer was promised: the residual `472,768.2` (38.7% of the arm, 1.062× the
  target on its own), the exact 94,470-Toffoli ledger over the six unconditional regions
  (20% of it), the EEA slopes (17,208/iteration, 351/position), the five levers ranked with
  ceilings, the correctness obligations (systolic schedule gate-for-gate; early halting and
  asynchronous pointers closed until proved; no deletion-differencing), and the arithmetic
  showing no control-logic edit reaches 445,229: `C ≤ 445,229 − P` needs either zero-cost
  arithmetic (impossible at the 23-AND bound) or a 5.8% cut in `C` *with* the arithmetic
  free.
- **PINNED:** three new kicks in `tests/test_dyn_mr_gated_arms_kickmix.py` — the three-term
  sum, the 472,768.2 residual with `C > target`, and the 94,470 region ledger at 20.0% of it.

## 2026-10-29 — The 54-AND multiply as an opcode: 4.687×, and 35% of it is Forney

- **NEW:** `gen_gf54_mul` in `scripts/kickmix_mr_opcode.patch` emits the proved 54-AND
  GF(2¹²) product natively, and a `GFProduct` policy (`builder.gf_product = 'and54'`) routes
  `gen_gf_mul` through it. The earlier `Gf54MulMixin` could only replace `ixor_gf2_mul`; the
  five Itoh–Tsujii products inside Chien search and the quotient sweep's are emitted by
  native generators, so the mixin left them on the built-in 63-AND recursion.
- **MEASURED:** `scripts/measure_gf54_native_opcode.py` (record:
  `benchmarks/quantum_compilation/gf54_native_opcode.json`). Unit costs: controlled multiply
  **75 → 66**, plain multiply **63 → 54**, inversion **315 → 270**, division **378 → 324**
  Toffolis, at **42** workspace qubits against 12, all four correct on random basis states
  (values, restored operands, clean workspace, untouched phase). At the pinned instance over
  64 shots the **reference** decoder goes **5,725,071.6 → 5,201,222.5 per shot (9.15%)**, `Q`
  1,893 → 1,935, and the six-lever arm **1,325,230.0 → 1,221,466.8 (4.687× against 4.320×)**
  — better than the mixin's 4.652×, because the opcode reaches the inversion too.
- **MEASURED — the Forney inversion's share, isolated.** The reference runs an inversion of
  five products at every one of its 4,095 positions: 20,475 product sites, 184,275 static
  Toffolis. Pricing it alone (the same reference with only
  `init_gf2_inverse_with_scaffold`/`gf2_phase_by_product` switched) executes **20,469.5
  products per shot across 4,093.9 positions, saving 184,225.1 Toffolis — 3.22% of the
  reference and 35.2% of its 523,849.1 saved per shot**. The remaining 339,624.0 (64.8%) is
  the change-of-basis division and the quotient sweep. Static and measured differ only by the
  1.1 positions per shot whose error bit is clear.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains the builder-policy surface
  (66 CCX, 42 extra workspace bits, invalid names rejected), an end-to-end scaled run, and a
  pin on the recorded split so the Forney share cannot drift back to an assertion.

## 2026-10-29 — The 54-AND multiply, executed: 4.652×, and the share it corrects

- **NEW:** `scripts/gf54_mul.py` turns the proved 54-AND GF(2¹²) algorithm into gates: the
  pinned `mu(2)=3`/`mu(3)=6` base cases, their Karatsuba composition, a classical evaluator,
  and `Gf54MulMixin`, which reimplements `ixor_gf2_mul` as **54 ANDs plus the 12-Toffoli
  controlled add — 66 Toffolis per controlled call against the built-in 75** — clearing its
  product window and scratch with `mr`. 42 workspace qubits against the built-in 12, and a
  longer CNOT ladder (566 vs 481).
- **MEASURED:** `scripts/measure_gf54_mul_kickmix.py` (record:
  `benchmarks/quantum_compilation/gf54_mul_kickmix.json`). The emitted multiply is correct on
  random basis states with random controls (values equal `field.mul`, operands restored,
  workspace `|0>`, phase untouched), and swapped into the six-lever arm at the official
  instance over 64 shots it takes **1,325,230.0 → 1,230,640.0 Toffoli per shot: 4.652×
  against 4.320×**, a **7.14%** saving, `Q` 1,894 → 1,897, with the arm still passing the
  six-lever verification shot for shot against the reference.
- **CORRECTED:** the projected 4.928× does not reproduce, and the reason is the model, not
  the circuit: a controlled multiply pays its ANDs **plus 12**, and the 12 do not shrink, so
  the call scales by 66/75. The same measurement corrects a larger error in the record — the
  multiply is **59.48%** of the arm (788,250.0 per shot over 10,510 executed calls), not the
  **86.33%** an earlier ablation reported. That ablation deleted `ixor_gf2_mul` and
  differenced executed counts, so the registers held garbage, the gating predicates read it
  and far less other work ran. The new measurement differences two arms that differ only in
  the multiply, and it reconstructs the measured arm exactly
  (536,980.0 + 10,510 × 66 = 1,230,640.0).
- **REVISED:** the multiply-cost ladder is now `t(ands) = 536,980 + 10,510 × (ands + 12)`: 54
  ANDs 4.652× (measured), 45 ANDs 5.039×, the 23-AND `2n-1` bound 6.327× (2.03× the target),
  and **0 ANDs 8.634× — still 1.49× above 445,229**. The conclusion survives with a narrower
  margin: no multiply, however cheap, delivers Benchmark II on this decoder, so the target
  needs a change to the 536,980 that is not the multiply. `docs/CIRCUIT_OPTIMISATION_RULES.md`
  §5–§6 and §8 carry it, and `benchmarks/quantum_compilation/{designer_directions,
  gf_mul_and_static_headroom}.json` were regenerated rather than left asserting the old share.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains pins for the base algorithms,
  the emitted circuit (values, restored operands, clean workspace, phase), the 66/75 unit
  costs, both arms' correctness on a scaled run, and the recorded decomposition.

## 2026-10-29 — Three designer directions priced; the multiply gains 14.3%

- **MEASURED, then designed:** `scripts/study_circuit_designer_directions.py` (record:
  `benchmarks/quantum_compilation/designer_directions.json`) prices the three ways left to
  improve the decoder. **A** (predicate discovery over the regions no lever touches) is
  bounded at **2.71%** of the emitted stream at the pinned instance — 94,470 of 3,486,597
  Toffolis — so it does not pay here. **B** (an exact bilinear search) proves `mu(2)=3` and
  `mu(3)=6` by exhausting every choice of five rank-1 matrices, then composes by Karatsuba
  to **54 ANDs at GF(2¹²)** against the built-in 63 (14.3% fewer), verified against
  `GF(2¹²)` multiplication on 200 sampled inputs with 0 mismatches and priced at
  **4.928×**. **C** (a different decoder) is refuted as a multiply-only fix: even the
  23-AND `2n-1` dimension bound leaves 598,835.7 per shot, still 1.35× above the 445,229
  target, and only the inadmissible measured-operand route clears it.
- **DOCUMENTED:** `docs/CIRCUIT_OPTIMISATION_RULES.md` §6 carries the comparison and the
  ladder, and §8 records the 54-AND multiply as *pending*: a verified algorithm, not yet a
  measured circuit, because `ixor_gf2_mul` is the 63-AND Karatsuba and emitting the new one
  needs a second `kickmix` op.

## 2026-10-29 — Rules for a better circuit, and the measured ceiling

- **NEW:** `docs/CIRCUIT_OPTIMISATION_RULES.md` fixes what a "better circuit" claim has to
  carry: the metric by name (**executed** 1,325,230.0 = 4.320× versus **static** 3,486,597 =
  1.642×, which rank the levers differently), unitary preservation on every input (dead-work
  gating and measured uncomputation are admissible; measuring an operand that is not
  determined by the state is not), a falsifier per measurement, and the instance discipline.
- **MEASURED:** `scripts/measure_gf_mul_and_static_headroom.py` (record:
  `benchmarks/quantum_compilation/gf_mul_and_static_headroom.json`). `ixor_gf2_mul` is the
  Karatsuba recursion to the gate — 3/7/9/21/27/63 static Toffolis at GF(2²/2³/2⁴/2⁶/2⁸/2¹²),
  matching `K(m)` exactly at every degree. No split of the recursion beats 63 at m=12 (k=8
  ties, the rest are worse; schoolbook 144). The multiply is **86.33%** of the six-lever
  arm's executed count, so a 45-AND multiply would give 5.734× and a free one 31.598× (the
  latter only by the inadmissible measured-operand route). A full-recording peephole removes
  **0** Toffolis at both instances, and the sixth lever changes the static count by exactly 0:
  the emitted stream has no headroom left.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains a record pin for those
  invariants (Karatsuba exactness, the 63 floor, static five == six, zero peephole
  removals, the multiply share), so a later "optimisation" that changes any of them fails
  without the extension being rebuilt.

## 2026-10-29 — DQI MR-gated arms: a sixth lever and qubit parity

- **MEASURED:** `scripts/measure_mr_arm_costs_kickmix.py` prices the gated Benchmark II
  circuits on the rebuilt, MR-capable `kickmix` (official instance, m=4095, n=70,
  GF(2¹²), 64 shots, seed 20261028). The five oracle-gated levers reproduce the
  projection stage for stage: 5,725,071.6 → 1,381,390.0 Toffoli per shot (4.144×)
  against the projected 1,382,207.08 (4.142×), every stacked stage within 0.06% of its
  projected count.
- **NEW LEVER:** each of the sweep's two polynomial divisions is consumed by an add
  controlled on its own counter bit, and the four counter bits are a one-hot decode, so
  in every live round exactly one division is computed under a false add and read by
  nothing. Gating each division on that bit (`gate_quotient`, arm `quotient`) removes
  4.07% of the five-lever count: **1,325,230.0 per shot, 4.320×**. It carries no projected
  row, because the oracle build priced both divisions as live.
- **FIXED (qubits the reads were leaking):** every read allocated a fresh copy qubit and
  the sweep's `q_i` was never freed, so the gated arm's id count grew by roughly a
  register per round — 4,503 ids against the reference's 1,893. One shared read ancilla
  (`mr` clears its target, so reuse is free) and freeing `q_i` in the round that
  allocates it bring both stacks to **1,894** ids: the gated rows now pay one qubit, not
  2,610.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains
  `test_small_instance_cost_pin_catches_gate_regressions`, which re-prices the whole
  stack at m=15/n=6/GF(2⁴) in about a second (33,235.5 → 10,409.5, 3.193×) and fails if
  any gate stops firing, changes its saving by more than 1%, or costs qubits — the
  64-shot official run is not needed to catch a regression. `docs/MR_INSTRUCTION_SPEC.md`
  §6b records the new lever and the qubit fix.

## 2026-10-29 — Temporal-rate detector audit hardening

- **RE-PINNED:** `specifications/canonical_preregistered_spec_v1.json` and
  `apedp_dyn/conf/canonical_spec.py` to
  `dbf67c5ac443073721421eba456519a0eda09ae07680d638c26f1d019b0b2343`.
  The only spec change is the new `temporal_rate_min_compositions=2`; the
  two-distinct-composition requirement for a common-mode claim is now
  canonical rather than a hardcoded literal. No previously pinned FPR, power,
  alpha, SNR, amplitude, harmonic, or sample threshold changed. FLAW-01
  remains OPEN; the governance manifest and its expected failing test were not
  touched.
- **FIXED:** `_drive_phasor()` used a raw windowed DFT of the drive. For a
  non-integer number of drive cycles, constant/trend leakage rotated and
  rescaled the estimated phasor; that bias transferred incorrectly between
  calibration and confirmation frequencies and inflated the null statistic.
  The phasor is now fit jointly with a constant and linear trend over
  cosine/sine columns and returns the analytic conjugate. Added
  `test_calibration_transfer_has_no_drive_phasor_bias`, which recovers the
  known susceptibility from a noiseless non-integer-cycle calibration to
  1e-10 relative accuracy.
- **MEASURED (before/after):** before the phasor fix, 512 null trials at 128
  surrogates gave FPR 14/512 = 2.73% (95% upper 4.24%) and 128 trials at 512
  surrogates gave 4/128 = 3.13% (95% upper 7.01%). After the fix: 512 trials
  at 128 surrogates gave FPR 5/512 = 0.98% (95% upper 2.04%), artifact rate
  0/512, power 512/512 (95% lower 99.42%), median null p 0.516; a second
  512-trial seed 7 audit also gave 5/512; 128 trials at 512 surrogates gave
  FPR 1/128 = 0.78% (95% upper 3.65%), artifact rate 0/128, power 128/128
  (95% lower 97.69%), median null p 0.496. Both audits pass the preregistered
  point-estimate gate. Exact Clopper-Pearson bounds are reported
  as diagnostics; the 128/512 upper bound is now below 5% as well.
- **HARDENED:** `ClockDataset` rejects non-uniform sampling; calibration and
  confirmation must use disjoint drive-harmonic sets at the Rayleigh
  resolution; calibration reports per-harmonic SNRs while the insufficiency
  gate remains fundamental-only; `TemporalRateAdjudication` records both
  dataset ids, the frozen analysis parameters, calibration SNRs, and the
  canonical spec hash.
- **FROZEN:** the preregistration schema is now `temporal-rate-v2` and freezes
  `harmonics`, `n_surrogates`, `seed`, `familywise_alpha`,
  `heterogeneity_alpha`, `min_fractional_amplitude`, and
  `canonical_spec_sha256` in addition to dataset fingerprints/ids, drive
  frequencies, and the clock-composition map.
  `adjudicate_preregistered_temporal_rate()` takes exactly three arguments and
  cannot be parameter-overridden after registration. Verification rejects a
  changed spec hash, changed canonical alpha, or fewer than two distinct clock
  compositions. The artifact remains a fingerprint commitment, not a
  signature.
- **CALIBRATION:** `run_detector_calibration()` now refits a fresh calibration
  for every null/injection pair, so the FPR is unconditional on one lucky
  calibration realization. It reports false-candidate, oscillator-specific
  artifact, and any-false-claim rates, plus exact one-sided 95% bounds and
  `n_calibration_draws`; the preregistered pass gate is the point-estimate
  FPR/power floor, with bounds retained as diagnostics.
- **ADDED:** 9 pins, bringing the temporal suite to 44 (18 detector + 26 IO):
  uniform-sampling rejection, harmonic-overlap rejection, per-harmonic SNR
  reporting, phasor-transfer regression, frozen analysis parameters, no
  parameter overrides, tampered spec hash, tampered canonical alpha, and
  distinct-composition enforcement. `AGENTS.md`'s standard suite now includes
  both temporal test files.
- **Verified:** 44/44 temporal pins pass; Ruff is clean on the changed files;
  repo-scope mypy (`python3 -m mypy`) is clean. The expanded dynamic suite is
  162 passed with one pre-existing unrelated failure:
  `test_orphan_module_detection` scans the untracked vendored
  `kickmix/_qualtran_src/`.
- **LIMITS:** common-mode calibration/clock systematics are still unmodeled,
  `systematic_susceptibility_sigma` still defaults to zero, the pure-sinusoid
  drive assumption is documented but not validated against non-sinusoidal
  drives, and colored/non-Gaussian residual noise is not covered by the
  shared-phase surrogate. No real two-composition clock data exists in the
  repo, so these are detector-calibration results, not a measurement of nature.

## 2026-10-29 — `MR` implemented, and the gated arms run on a real `kickmix`

- **ADDED:** `scripts/kickmix_mr_opcode.patch` — the `MR`/`MR_IF` opcodes
  (`docs/MR_INSTRUCTION_SPEC.md`) applied to the vendored `kickmix` sources:
  opcode values 0x54/0xD4 with names, `OP_TYPE_TABLE` and `.kmx` parse/print, a
  `mr`/`mr_if`/`broadcast_mr` builder path with Python bindings, a Z-basis
  simulator branch that writes the outcome and clears the qubit without touching
  the tracked phase, its own statistics group, and a reaction-depth case.
- **ADDED:** `scripts/build_kickmix_mr.py` exports the vendored sources at their
  recorded commit, applies the patch, and compiles `build/kickmix_mr/lib`
  (`build/` is gitignored). The pinned competition binary that
  `apedp_dyn/quantum_compilation/contracts.py` hashes is never modified.
- **ADDED:** `scripts/run_mr_gated_arms_kickmix.py` runs the five gated DQI levers
  on that extension with a real `kickmix.CircuitBuilder` and a real
  `kickmix.Simulator`, so the arms are no longer measured only on the local model.
- **MEASURED:** at m=15/n=6/GF(2⁴) with 8 shots, `none`, `swap`, `sweep`,
  `forney`, `clear`, `shift` and `all` match the reference's syndromes and tracked
  phase, restore the syndromes and leave every non-syndrome qubit `|0>`; `all`
  also passes at m=255/n=14/GF(2⁸) with 16 shots and at the official
  m=4095/n=70/GF(2¹²) with 4 shots (reference 1,893 qubits, arm 4,503 allocated
  ids, phase identical). The falsifiers still fail:
  `--clear-with r` (phase 6 ones vs 3, dirty junk) and `--shift-under`
  (syndromes not restored).
- **RECORDED:** `x(q); mr(q, b)` gives `b = 1` and 0 phase flips on 64/64 shots,
  against 37/64 phase flips for the same circuit with `reset` — the semantic
  difference the instruction exists for.
- **ADDED:** `scripts/measure_mr_arm_costs_kickmix.py` prices the reference, the
  cumulative lever chain and each lever alone with `kickmix`'s conditional-aware
  per-shot Toffoli counting. Record:
  `benchmarks/quantum_compilation/kickmix_mr_arm_costs.json`.
- **ADDED:** `tests/test_dyn_mr_gated_arms_kickmix.py` pins the real-kickmix arm
  runs and both falsifiers, plus the cost record's arithmetic; the extension-backed
  pins skip when the MR build is absent (10 passed under CPython 3.13 with the
  build, 1 passed and 9 skipped under the project's 3.12).
- **MEASURED (cost):** at the official instance over 64 shots the rebuilt builder
  reproduces the pinned reference exactly (5,725,071.6 vs 5,725,071.61) and the
  full gate executes **1,381,390.0 Toffoli per shot, 4.144×** against the
  projected 1,382,207.08 / 4.142×. Every stacked stage is within 0.06% of its
  projection — 4,298,268.3 (Forney), 2,157,906.3 (+clear), 1,590,308.3 (+sweep),
  1,456,286.3 (+swap), 1,381,390.0 (+shift) — so the oracle-gated projection is
  reproduced by the circuits rather than bounding them from above. Per lever
  alone: `swap` 1.049×, `sweep` 1.247×, `forney` 1.332×, `clear` 1.597×, `shift`
  1.027×; the shift lever inside the chain is worth the 5.14% the oracle
  accounting gave it.
- **FIXED:** three gate defects in `scripts/run_mr_gated_arms.py` that a first
  costing pass (3.556×) exposed. The shift gate's `¬flag` was written as a second
  `bit_store(…, True)`, leaving the bit constant, so it skipped only rounds with
  every control clear (1.002× alone). The sweep gate wrapped only the position
  sweep, leaving the quotient divisions and the `n_a` add/undo inside dead rounds.
  The Forney predicate was an 11-Toffoli equality circuit run twice per position,
  with a copy qubit allocated per position; it is now `mr`-read out of `sigma`,
  restored with classically controlled `x`, and assembled from two classical
  stores. Net: 1,609,925.7 → 1,381,390.0 per shot and 8,587 → 4,503 qubit ids.
- **UNCHANGED:** a receipt still requires an upstream `kickmix` rebuild plus the
  official verifier; the local extension is not the pinned binary.

## 2026-10-28 — Preregistration gate for the temporal-rate null test

- **ADDED:** `TemporalRatePreregistration` plus `preregister_datasets()`,
  `save_preregistration()`, `load_preregistration()`,
  `verify_preregistration()`, and `adjudicate_preregistered_temporal_rate()`
  in `apedp_dyn/epistemic/temporal_rate_io.py`. The artifact freezes the
  calibration/confirmation SHA-256 fingerprints, dataset ids, drive
  frequencies, and clock-composition map before adjudication. Verification
  recomputes every field and fails closed on any mismatch, including a swapped
  confirmation file that keeps the same dataset id.
- **EXTENDED:** `scripts/run_temporal_rate_null_test.py` gains
  `--write-preregistration`, `--preregistration`, and
  `--preregistration-notes`. `--write-preregistration` freezes the fingerprints
  without running the test; `--preregistration` verifies before adjudicating.
  Protocol errors now exit 1 with a one-line message instead of a traceback.
- **ADDED:** eight preregistration pins in
  `tests/test_dyn_temporal_rate_io.py`: round trip, swapped confirmation,
  tampered calibration, changed metadata, unknown keys, unsupported protocol
  version, duplicate dataset ids, and end-to-end preregistered adjudication.
  The temporal suite is now 35 pins (14 detector + 21 IO).
- **Verified:** 35/35 temporal pins pass; Ruff is clean. The full dynamic suite
  with both new files included is at 152 passed with the same two pre-existing
  unrelated failures (host-path literal in untracked
  `scripts/inspect_patch_1994.py`; orphan scan over untracked vendored
  `kickmix/_qualtran_src/`).

## 2026-10-28 — Clock-dataset ingestion for the temporal-rate null test

- **ADDED:** `apedp_dyn/epistemic/temporal_rate_io.py` — strict CSV +
  sidecar-JSON and compressed NPZ ingestion for `ClockDataset`. The CSV format
  names `time_s`, `drive`, and one fractional-frequency column per clock; the
  sidecar carries the dataset id, drive frequency, and each clock's id,
  composition, column, and systematic-uncertainty budget. NPZ embeds the same
  metadata and is read with `allow_pickle=False`, so a dataset file cannot
  execute code. Both paths delegate finite/monotonic/length checks to
  `ClockDataset` and expose `dataset_sha256()` for calibration/confirmation
  provenance.
- **ADDED:** `tests/test_dyn_temporal_rate_io.py` — 13 pins covering CSV and
  NPZ round trips, cross-format fingerprint equality, fingerprint sensitivity,
  missing sidecar, missing/unknown metadata keys, duplicate clock ids, empty
  and non-finite cells, unknown NPZ keys, object-array rejection, NPZ
  `metadata_path` rejection, single-clock support, and unsupported formats.
- **EXTENDED:** `scripts/run_temporal_rate_null_test.py` now adjudicates
  supplied files via `--calibration` / `--confirmation` (with optional
  `--calibration-metadata` / `--confirmation-metadata`), reports both dataset
  fingerprints, and preserves the claim boundary. The synthetic demo remains
  the default when no files are supplied.
- **Verified:** the 13 IO pins and the 14 detector pins pass; Ruff is clean on
  the new module, tests, and script. The full dynamic suite, with the two new
  files included, is at 144 passed with the two pre-existing unrelated
  failures (host-path literal in untracked `scripts/inspect_patch_1994.py`;
  orphan scan over untracked vendored `kickmix/_qualtran_src/`).

## 2026-10-28 — Option B: a temporal-rate null test, not a time-travel claim

- **ADDED:** `apedp_dyn/epistemic/temporal_rate_modulation.py` — Hypothesis B
  harness. It subtracts each clock's calibration-measured ordinary response to
  a periodic drive and tests whether the residual phase-locked
  fractional-frequency modulation is common to clocks of at least two distinct
  compositions. The null is a shared-phase Fourier surrogate of the post-fit
  residuals with the fitted harmonic variance restored and calibration
  uncertainty injected; a heterogeneity guard rejects oscillator-specific
  artifacts. Calibration/confirmation dataset-id reuse, identical series, and
  same-frequency calibration raise `TemporalRateProtocolError`. The only
  positive verdict is `COMMON_MODE_ANOMALY_CANDIDATE`, explicitly not evidence
  of backward time travel.
- **PREREGISTERED:** canonical spec gains the `temporal_rate_*` block
  (minimum clocks/cycles/samples-per-cycle, harmonics, surrogate draws,
  familywise and heterogeneity alphas, amplitude floor, calibration-SNR floor,
  and detector FPR/power/injection thresholds). The spec is re-pinned to
  `a5e13979…`; no existing threshold changed. Ed25519 re-signing remains
  trustee-blocked (FLAW-01 still OPEN).
- **MEASURED, synthetic detector calibration:** 20 null and 20 injected
  common-mode trials at 128 surrogates (seed 6) give FPR 0.00 against the 0.05
  ceiling and power 1.00 against the 0.80 floor, with zero insufficient
  verdicts. This measures the analysis, not nature.
- **Verified:** `tests/test_dyn_temporal_rate_modulation.py` 12 passed,
  including null-consistency, common-mode recovery at an arbitrary phase,
  oscillator-specific rejection, protocol reuse/frequency refusals,
  single-composition and single-clock refusals, weak-calibration refusal,
  deterministic replay, and the FPR/power floors.

## 2026-10-28 — The MR arms hold at the official instance

- **MEASURED:** all five arms and `all` reproduce the reference at Benchmark II's own
  instance — m=4095, n=70, GF(2¹²), 4 shots: syndromes, tracked phase and `|0>` junk all
  match. Reference: 107,911 qubit ids, 4m54s. Arm ids: swap 108,355, sweep 108,799,
  forney 112,006, shift 110,131, `all` 84,103, clear 78,232. These are local allocation
  counts, not `Q`.
- **FALSIFIED at the same scale:** the under-gated shift arm fails on syndromes, phase and
  junk, and the `r` readout fails on all three (at this scale its readout also corrupts
  values, not only the phase). Record: `benchmarks/quantum_compilation/local_mr_arms.json`.
- **Verified:** `tests/test_local_mr_sim.py` now pins all three instances.

## 2026-10-28 — Fifth MR lever verified at both instances, with a falsifier that fails

- **ADDED:** the shift gate to `scripts/run_mr_gated_arms.py` (`--arm shift`, included in
  `all`). It reads `c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))` with a real `mr`, using the flag *after*
  `_update_flag`; the probe's earlier attempt with a round-top flag fails because that flag
  is stale. At m=15/GF(2⁴), 32 shots, the predicate is false in 26.7% of round-shots, and
  the arm reproduces the reference's syndromes, phase and `|0>` junk.
- **MEASURED, scaled:** all five arms plus `all` pass at m=255/GF(2⁸), n=14, 16 shots
  (reference 9,751 qubit ids; syndromes, phase and junk all match). The falsifiers fail at
  both instances: `--clear-with r` loses the phase and leaves junk, and `--shift-under`
  (dropping `¬flag ∧ (c0 ∨ c3)`) skips shifts the decoder needs and breaks the syndromes.
  Record: `benchmarks/quantum_compilation/local_mr_arms.json`.
- **REJECTED as a falsifier:** inverting the shift gate also passes, because one skipped
  shift is repaired by later rounds; only under-gating breaks the decoder, so the inverted
  knob was removed rather than shipped as a test that cannot fail.
- **Verified:** `tests/test_local_mr_sim.py` 21 passed over both recorded instances,
  including the under-gated shift arm and the `r` readout.

## 2026-10-28 — Three metric placeholders now measure or refuse

- **FIXED, `EnsembleSearcher.get_diversity_score`:** returned a constant 0.8 with a
  "would analyze equation structures" comment. It now reads the solutions out of the
  results (`best` / `expression` / `expressions` / `equations` / `solutions`) and scores
  `(distinct - 1) / (total - 1)`: 0.0 when every member agrees, 1.0 when no two do. Fewer
  than two reported solutions — including its own `search_ensemble` plan — raises, because
  a plan has no diversity to measure.
- **FIXED, `DeepEnsemble.predict`:** reported `aleatoric=jnp.zeros_like(std)`, presenting an
  unmodelled quantity as a measured zero. These regressors train deterministically, so the
  field is `None`; the monkeypatch that previously masked the same placeholder after import
  was removed with it.
- **FIXED, `PhysicsConstraintEngine.dimensional_analysis_score`:** returned 0.5 with no
  target and 1.0 with one, regardless of any candidate. It now takes the dimensions of the
  candidate's terms and returns the fraction that match the target, raising when there are
  no terms or no target. The Rust kernel had the identical placeholder and now implements
  the same measurement over a JSON term list (rebuilt extension); the Python fallback and
  the kernel agree, and both `check_dimensional_consistency` paths now compare all seven
  exponents instead of mass/length/time only.
- **Verified:** `tests/test_placeholder_metrics_are_measured_or_refused.py` 9 passed (both
  implementations of the dimensional score, the diversity endpoints, the refusals, and
  `aleatoric is None`); `tests/test_search_engines_real.py` 13 passed.

## 2026-10-28 — MR arm correctness tested locally: four levers, and `r` fails as it should

- **BUILT:** `apedp_dyn/quantum_compilation/local_mr_sim.py`, a Z-basis (value + sign)
simulator for the DQI decoder with `mr` (Z measure-and-reset, phase-preserving), `r`
(clears, randomizes the phase, as `kickmix` does) and `hmr` (X basis). It implements the
builder subset the decoder uses (`unary_iteration`, `init_and`, `iadd`/`isub`,
`flip_if_*`, rotations, GF(2^b) arithmetic, `gf2_phase_by_product`) with semantics
pinned against `kickmix`, including its primitive polynomials.
- **VALIDATED:** `scripts/verify_local_mr_sim.py` builds the reference decoder with both
engines and feeds both the same shots. At m=15/n=6/GF(2⁴): syndromes identical, phase
identical, syndromes restored, every other qubit `|0>`, 0 phase errors — the local engine
reproduces `kickmix` exactly (qubit *ids* differ because freed ids are not reused here).
- **MEASURED:** `scripts/run_mr_gated_arms.py` runs the four gated levers with their
predicates read by a real `mr` through `cx(control, ancilla); mr(ancilla, bit)`, not the
oracle registers the probe used. Every arm reproduces the reference's values AND phase
at 32 shots: `swap` (c3), `sweep` (c0 ∨ c3), `forney` (the `sigma == 0` zero-test),
`clear` (measured clear of the EEA instead of its adjoint, with the syndromes re-prepared
from a saved readout), and all four together. Record:
`benchmarks/quantum_compilation/local_mr_arms.json`.
- **FALSIFIED, by construction:** the same swap arm read with `r` instead of `mr` fails on
phase and leaves junk behind while its syndromes still match — so the pass is about the
measurement, not the circuit. This is the mechanism-level version of the 11/16 phase
defect the probe recorded.
- **STILL UNSCORED:** the arms' cost. `local_mr_sim` computes values and phases, not gate
counts, and `contracts.py` pins the official `kickmix` build and verifiers, so the 4.142×
stays a Toffoli bound until `MR` exists upstream.
- **Verified:** `tests/test_local_mr_sim.py` 21 passed (pinned primitives, the MR/R/HMR
contract, and the arm record).

## 2026-10-28 — Search engines score on data, and NGTS runs UCT or nothing

- **MEASURED, not defaulted:** `ValidationPipeline.validate` now reports
  `generalization_gap` (held-out minus cross-validation error) and derives
  `generalization_score` from it as `1/(1+gap)`. The old score was a lookup — 0.9 below
  a threshold, 0.7 above — so it reported a number the data could not move.
- **No fabricated evidence:** `NGTSSearch.run_simulations` runs UCT and calls a reward
  function for each selected leaf. With no reward function it runs **zero** simulations
  and returns 0; it previously returned `num_sims` and backpropagated
  `random.random()*0.5`, i.e. reported search work it had not done. The Rust kernel had
  the same defect in `ngts.rs` (`rand_simple` values) and now takes the reward callable
  in `run_simulations(num_sims, reward=None)`.
- **Hierarchical phases score what they propose:** `_mid_search` / `_fine_search` take
  `data`/`target`, evaluate each expansion on the data, and return nothing when there is
  no data. `HierarchicalSearch.search` attaches a measured `r2` to every candidate and
  **drops** candidates it cannot evaluate; it used to validate every candidate against
  default `train_error=0.1 / test_error=0.15`, so an unmeasured expression carried a
  validation score.
- **Ensemble reports what it cannot run:** a proposer outside `ProposerRouter`'s map is
  `status: error` instead of a planned member, and `best` is present and `None` until a
  member returns a scored candidate.
- **Determinism:** `beam_search_expressions` no longer consumes `seed` for sampling; two
  seeds return the same list, and level 1 pairs atoms with each other so multi-variable
  forms appear immediately.
- **Verified:** `tests/test_search_engines_real.py` 13 passed (was 12 of 13 failing);
  related suites 58 passed / 2 skipped / 1 xfailed.

## 2026-10-28 — Scratch pool measured dead: 25-qubit ceiling, and the allocator claim withdrawn

- **MEASURED** on the pinned Benchmark II instance (`scripts/measure_dqi_scratch_pool.py`,
  record `benchmarks/quantum_compilation/dqi_scratch_pool.json`): the circuit holds
  **1,828 qubits concurrently live against 1,893 distinct ids**; the peak is reached
  inside `_chien_search_and_forney_mbuc` and the innermost frame there holds **25**
  qubits. A shared scratch pool can therefore redistribute at most 25 — the −104 the V6
  row claims is 4× that ceiling and is not reachable by pooling.
- **CORRECTED, committed claim:** §5.5 of the boundary doc said the live peak was EEA
  round scratch in `_multiply_and_add_or_sub` and that a shared pool was "the only
  mechanism that could move" the transients. Both are withdrawn on measurement: the peak
  is in Chien/Forney, and the pool is dead.
- **Left OPEN, not answered:** the probe's trace replay returns 1,821 ids against 1,893
  shipped and observes only 1,873 ids, so allocations made inside the extension are
  missing; a lowest-hole free list replays identically to the shipped LIFO stack. An
  earlier draft of the script asserted the 65 stranded ids were `alloc_qubit` LIFO
  stranding and named a free list as the fix. That verdict was not supported by the
  replay and has been removed rather than shipped.
- **Verified:** `tests/test_dqi_scratch_pool.py` pins the artifact; the probe runs under
  CPython 3.13 only, since `kickmix.so` is a 3.13 build.

## 2026-10-28 — Post-`_update_flag` probe collected: 4.142× measured

- **MEASURED:** the shift gate the previous entry had to abandon is now correct. The
  wrong arm read `flag` at the top of the round; `_update_flag` toggles it first, so the
  probe now carries six bits per round and the liveness is
  `c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))` with the post-update flag. Official instance:
  `T_mean` 1,457,103.45 → **1,382,207.08, a 4.142× over the reference**, `Q` unchanged at
  1,893, **0 value / 0 phase errors on 64 shots**, 1,332 round bits (6 × 222).
  Scaled: 3.709× at m=255 (512 shots), 2.998× at m=15.
- **Corrected prediction:** the prize was bounded at ~5% from the failed arm's cost and
  measures at 5.15% (1,457,103 → 1,382,207). Dead shifts are 31.59% of rounds, not the
  35–40% guessed before the fix.
- **Running total:** Forney gating 1.332×, MR clear 2.652×, sweep 3.599×, swap 3.929×,
  shift 4.142×. Receipt arithmetic 24.439T → 5.900T, portfolio 2.361982 → 3.79; still
  3.10× short of the 12.86× target.
- **Verified:** `tests/test_dqi_predicate_gating.py` 12 passed / 1 skipped.

## 2026-10-28 — Dead register swaps gated: 3.929× measured, and the shift gate falsified

- **MEASURED:** `_swap_a_and_b` exchanges `reg_a`/`reg_b` wholesale — 2 × 73 × 12 = 1,752
  qubits — every round, but its control is `swap_ctrl = c3_hc ∧ flag`, so it is dead in
  any round without `c3` (**67.57%** of rounds). Gating it on `c3` alone: official
  `T_mean` 1,590,938.28 → **1,457,103.45, a 3.929× over the reference**, `Q` unchanged at
  1,893, **0 value / 0 phase errors on 64 shots**, for 1,110 round bits (5 × 222).
  Scaled: 3.507× at m=255 (70.4% dead), 2.813× at m=15. Receipt arithmetic 24.439T →
  6.22T, portfolio 2.361982 → 3.7271; still 3.27× short of 12.86×.
- **FALSIFIED (mine, one hour old):** gating `_shift_b_array_right` on the complementary
  pair `c1 ∨ c2 ∨ ¬flag`. It looked right and fails on **every shot** (512/512 value
  errors at m=255). Cause: `flag` is toggled by `_update_flag`, which runs *before* the
  shift in the same round, so a flag read at the top of the round is stale. The arm's
  cost (62,434 at m=255 against 65,698) bounds the remaining prize at ~5%, and collecting
  it needs a second probe point after `_update_flag`. Not taken; pinned as a negative in
  `tests/test_dqi_predicate_gating.py`.
- **Cost of the whole design so far:** 1,803 classical bits for the clear, 1,110 for the
  round predicates, zero extra qubits, zero extra Toffolis.

## 2026-10-28 — Dead multiply sweeps gated: 3.599× measured

- **MEASURED:** the synchronized EEA emits all five steps every round, but
  `_multiply_and_add_or_sub` touches `ctrl` only through `c0_hc` and `c3_hc`. In a round
  where both are zero — the two shift rounds of each four-round counter cycle — its
  4,536-Toffoli sweep is emitted and does nothing. Reading `(c0_hc, c3_hc)` per round
  and gating the sweep on `c0 ∨ c3` removes **35.14%** of the forward EEA at the official
  instance: `T_mean` 2,158,651.45 → **1,590,938.28, a 3.599× over the reference**, with
  `Q` unchanged at 1,893 and **0 value / 0 phase errors**. Scaled: 3.205× at m=255
  (40.79% dead), 2.586× at m=15 (46.98% dead).
- **Cost:** one classical bit per EEA round — 222 bits at the official instance, 54 at
  m=255, 30 at m=15 — no qubits, no Toffolis. Still an upper bound: the round bits are
  read from the same circuit in a probe pass.
- **Not taken:** `_shift_b_array_right` is live in the `c1`/`c2` rounds *and* in the
  `c0`/`c3` rounds when `flag` is clear (`ctrl_nB = c0_nf ⊕ c1_hc`), so it cannot be
  gated on the complementary pair without also reading `flag`; left alone.
- **Verified:** `tests/test_dqi_predicate_gating.py` 9 passed / 1 skipped.
- With the Forney gating and the MR clear this is 3.599× of the 12.86× the target needs:
  `T_mean` 1.591M against 445,229 leaves a 3.57× gap, and the receipt moves 24.439T →
  6.791T (portfolio 2.361982 → 3.6198), arithmetic on `R·Q²·T`, not a receipt.

## 2026-10-28 — MR clear halves the EEA: 2.652× measured, and `R` is not a substitute

- **MEASURED (the better design):** the reversible EEA is run twice — forward, then
  its adjoint to uncompute — so half of Benchmark II's 75% EEA share is uncomputation.
  Clearing the EEA scratch and the syndromes and re-preparing the syndromes from a
  classical readout replaces that half. `scripts/probe_dqi_predicate_gating.py`
  measures it with both oracle registers (syndrome readout + root bits): official
  instance `T_mean` **5,725,071.61 → 2,158,651.45, a 2.652×**, `Q` unchanged at 1,893,
  **0 value and 0 phase errors** on 64 shots, needing 1,803 classical bits and 1,803
  classically controlled X's. Scaled: 2.351× at m=255 (512 shots), 2.012× at m=15.
  Because `Q` does not move, the score falls by the same 2.652×: 24.439T → **9.215T**,
  i.e. the portfolio score 2.361982 → **3.2695** (arithmetic on `R·Q²·T`, not a receipt).
  Still 4.85× short of the 12.86× target.
- **FALSIFIED (mine, from four hours earlier):** "`R` can stand in for uncomputation."
  A reset clear costs exactly what the reference spends (2,158,651.45 either way — CX
  is not Toffoli-like) but it is **wrong**: 35/64 shots at the official instance and
  254/512 at m=255 have a flipped phase. Cause, measured: a circuit with **no phase in
  it at all** — `x(q); reset(q)` — reports a random `phase_flipped` (11/16 shots), while
  the same circuit without the reset reports 0/16. `R` randomizes the tracked phase, so
  it can only be used where the answer does not depend on it. Measure-then-X on the
  measured value (i.e. `MR`) gives 0/0.
- **BOUND (corrected):** with MR, both levers together are worth 2.652×, not the 1.332×
  the Forney-gating entry below states: that entry bounded *one* phase of the decoder and
  its "even granted MR, T lands at 4.30M" bullet is superseded here. `T_mean` 2.159M is
  still 4.85× above the 445,229 target, so the §5.5 conclusion stands — the target needs
  a decoder, not an optimisation.
- **Still blocked in the IR:** `MR` does not exist. `push_condition` takes a classical
  bit; `hmr` is X-basis; `push_condition(qubit)`/`bit_store(control=qubit)` are rejected;
  `R` is phase-unsafe. The instruction that would make 2.652× real is Z-basis
  measure-and-reset, and this design needs it twice.
- **Verified:** `tests/test_dqi_predicate_gating.py` 7 passed / 1 skipped;
  the skipped re-measurement was run by hand under the kickmix interpreter and reproduces
  the record exactly. Record: `benchmarks/quantum_compilation/dqi_predicate_gating.json`.

## 2026-10-28 — Predicate-gated Forney body: 1.332× measured, and inexpressible in this IR

- **MEASURED:** `scripts/probe_dqi_predicate_gating.py` gates the Forney body on the
  truth table of `sigma(alpha^j) == 0` (supplied as an oracle classical register, so
  this is an upper bound, not a submission). At the official instance `T_mean` falls
  **5,725,082.58 → 4,298,955.87 (1.332×)** with `Q` = 1,893 and the static count
  unchanged, and with **0 value and 0 phase errors** on 128 shots: the body is a
  no-op wherever `located = is_sigma_zero & omega` is zero, so skipping it is free
  in correctness. This is the 1.335× "if the phase were free" row of
  `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.5, measured instead of estimated.
  Scaled checks: 1.185× at `m=255, GF(2^8)` and 1.011× at `m=15, GF(2^4)`, where the
  root density (2.03%, 12.7%) is much higher.
- **CORRECTED (null optimisation):** the committed `is_error` gate at
  `dqi_reference_circuit.py:573`/`:596` buys **0.006%** — 353 of 5,725,461 Toffoli per
  shot — because `P(M_j = 0) = 2^-12`. The gap between `toffoli_count` and
  `max_toffoli_count` in the receipt is that gate firing about once per shot, not a
  conditional-execution saving worth pursuing.
- **BLOCKED (IR):** the predicate cannot reach the condition stack. `push_condition`
  requires a `km.b`; the only readout is `hmr`, which is **X-basis** (on `|1>` it
  returned 31/64 ones); `push_condition(qubit)` and `bit_store(control=qubit)` are
  both rejected by the binding; and the instruction set `{NEG, BIT_INVERT,
  BIT_STORE0/1, X, Z, R, HMR, CX, CZ, SWAP, CSWAP, CCX, CCZ, Z_POW, INFO_TAG,
  DEBUG_PRINT, PUSH/POP_CONDITION}` is closed under the Z-diagonal group, so no
  Hadamard can be synthesized to conjugate the predicate into the X basis. One
  instruction — Z-basis measure-and-reset (`MR`) — is the entire difference.
- **BOUND (superseded):** 1.332× leaves `T_mean` at 4.30M against the 445,229 target
  (9.7× short) and the receipt at 18.35T — arithmetic on `R·Q²·T`, not a receipt. The
  Forney body is 24.9% of `T`; the other 75% is EEA iterations, which never skip. See
  the entry above: clearing instead of uncomputing that EEA reaches 2.652×, so this
  1.332× prices only the Forney phase.
- **Verified:** `tests/test_dqi_predicate_gating.py` 4 passed / 1 skipped; the skipped
  re-measurement was run by hand under the kickmix interpreter and reproduces the
  record exactly. Record: `benchmarks/quantum_compilation/dqi_predicate_gating.json`.

## 2026-10-28 — Benchmark II cost structure measured; two of my own optimisations falsified

- **MEASURED:** `scripts/measure_dqi_primitive_costs.py` recovers the DQI Toffoli
  breakdown, since `kickmix` exposes none. Per-call at GF(2¹²): `gf2_imul` and
  `gf2_idiv` **0** (CNOT-only linear maps), `ixor_gf2_mul` 63, Itoh–Tsujii
  inversion 315 = 5 × 63, scaffold division 378. Slopes at the benchmark widths:
  17,208 Toffoli per EEA iteration, 351 per Chien/Forney position.
- **FALSIFIED (mine):** "composite-field/Karatsuba cuts constant-multiplier cost
  30–50%" — there is nothing to cut. Multiply and divide by a constant are already
  0 Toffoli. The only multiplier that costs anything is quantum × quantum, and 63
  at degree 12 is already better than a naive schoolbook count.
- **FALSIFIED (mine):** Montgomery batch inversion of the Forney denominators. It
  needs 3 multiplications per element, so it can recover at most 2/5 of an
  inversion; at this repo's marginal qubit price 2T/Q = 6,048.7 Toffoli-equivalents
  per qubit, and 12 qubits per batched element, **every block size loses** (best
  B=4: 193,489 saved vs 290,338 spent, score ratio 1.021 — worse).
- **BOUND (new):** the Forney step inverts once per position
  (`dqi_reference_circuit.py:579`), so 4,095 × 315 = **1,289,925 Toffolis = 22.5%
  of `T_mean`, and 2.90× the entire 445,229 target** before any EEA iteration runs.
  Ceilings with `Q` unchanged: 1.259× if a position cost one multiply, 1.335× if
  the phase were free — against 12.86× required. The target is unreachable for this
  architecture, not merely unreceipted.
- **Q floor:** 1,887 measured = 1,752 persistent (`2 × 73 × 12`, both frozen) + 135
  transient. The claimed −104 is 77% of all transient qubits, and the live peak is
  EEA round scratch inside `_multiply_and_add_or_sub`; the `n_a`/`n_b` recycling
  the V6 row names frees 16.
- **Verified:** `tests/test_dqi_primitive_costs.py` 7 passed / 1 skipped; the
  skipped re-measurement was run by hand under the kickmix interpreter and
  reproduces the record exactly. Recorded in `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.5.

## 2026-10-28 — Riemann off-line scan scoping fixed; August batch screened for Benchmark II

- **FIXED (integrity):** `rust/riemann_verifier` built its off-line critical-strip grid
  as `(100..400) * 0.1` — `t in [10, 40]` — **regardless of `max_t`**, then printed
  `OFF_LINE_GRID_SCAN_PASS`. A run asked about `T=300` certified a window 260 units
  shorter than the question; the same defect class as the `PASS` removed from the DQI
  Gluzdov V6 projection the same day. The window is now `[10, max_t]` and is never
  truncated to fit the sample budget: the budget (20,000 samples) degrades resolution
  instead, and a coarsened scan reports `OFF_LINE_GRID_SCAN_PASS_COARSENED` rather than
  a bare PASS. JSON gains `grid_scan_window`, `grid_scan_step`, `grid_scan_samples`,
  `grid_scan_coarsened`, `grid_scan_covers_requested_height`.
- **Verified:** `T=60 → [10,60] step 0.1 / 501 samples`; `T=300 → [10,300] step 0.1 /
  2,901 samples`; `T=5000 → [10,5000] step 0.2495 / 20,000 samples / PASS_COARSENED`.
  `tests/test_riemann_verifier_grid_scan.py` 3/3, including a source-level revert pin.
  `probe_wave1.json` / `PROBE_WAVE1.md` regenerated against the fixed binary; verdicts
  unchanged (2 consistent_finite_range, 1 no_refutation, 1 not_probeable).
- **RECORDED (unfixed, separate):** root discovery at a fixed 0.05 step still misses
  zeros with height — 0 % at T≤100, 14.5 % at T=300, 84 % at T=5000 (721/4520). That is
  a discovery-budget problem, not the scan window, and it is now stated as such.
- **NEW:** `catalog/august_ten_advances_screen.json` +
  `catalog/AUGUST_TEN_ADVANCES_BENCHMARK_II.md` — the August 1 *Ten advances* batch is
  the only OpenAI release whose disciplines intersect Benchmark II (coding theory,
  arithmetic circuit complexity, quantum complexity, lattice cryptography), so it is
  screened against the DQI cost model. **0 of 10 rows admits a cost path**: three are
  lower bounds or hardness results (a barrier is not a tool), six have no circuit content,
  and the single coding-theory row is blocked by the frozen instance
  `(m=4095, n=70, degree=12, r=2016)` that `ProblemSpecDQI` fixes. Its only use is to put
  a floor under `R(eps_plus)`, not to lower the score.
- **Disclosed:** recomputing the target gain from the two workload rows gives 14.3973×,
  against the quoted 14.396× — a +0.009 (0.06 %) rounding difference recorded rather than
  absorbed. `T_mean` must fall to 7.78 % of reference (12.86×); `Q` only to 94.5 % (1.058×).
- **Verified:** `tests/test_august_ten_advances_benchmark_ii.py` 7/7, which cross-checks
  the frozen instance against the verifier's own `DEFAULT_M/N/DEGREE/R` and the cost model
  against the pinned workloads in `apedp_dyn.quantum.dqi_photonic_resource`.

## 2026-10-28 — Audit artifacts for the negative results, and the PDE honesty note

- **COMMITTED:** the seven `scripts/` audit instruments that back numbers already
  recorded in `CHANGELOG.md` and `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` had been
  left untracked, so the claims cited them from outside the repository. Two were
  load-bearing: the committed `run_halt_recycling_scale.py` imports
  `RecycledFlagDecoder` from `scripts/test_recycled_flags_small.py`, and the
  restored documentation cites `scripts/test_viable_optimization_path.py`.
  Verified by running the committed driver end to end (−240 counter-only at
  m=15/1023, +600/+360 flag, +240/+480 combined).
- **DISCLOSED:** importing `RecycledFlagDecoder` is safe; running it is not. Its
  toy circuit fails the official verifier (effective error rate 0.9155 ≥ 1/16,
  198/256 value errors). The flags prototype is a broken circuit, not merely a
  slower one.
- **NEW:** `docs/pde_discovery_honest_limits.md` plus
  `tests/test_dyn_pde_discovery_noise_robustness.py` — states that the default PDE
  discovery config is brittle to small additive noise and that the clean pinned
  demos are not evidence of general noise robustness, with the first fixed-config
  multi-seed sweep as the backing test (2 passed).
- **NEW:** `tests/test_dyn_dqi_photonic_resource.py` (14 passed) pins the shipped
  `dqi_photonic_resource` mapper: platform invariance of `R·Q²·T`, the fault-tolerance
  verdict, and the caller-supplied superconducting baseline having no defaults.
- **NEW:** `tests/test_dyn_rime_ad_input_contract.py` pins the `RimeAdProposer`
  input contract (`DataFrame | dict[str, DataFrame]`, unsupported shapes fail
  loudly rather than returning a candidate).

## 2026-10-28 — Portfolio synthesis names the real Benchmark I incumbent

- **CORRECTED:** `docs/invsqrt_frontier_and_portfolio_synthesis.md` still named the
  retired 90Q affine build ($157,752.008809$) as champion and carried a Benchmark I
  factor of 13.164804×. The pinned incumbent is the 89Q shared-PDDD circuit
  (`PASS`, 200,540,928 shots, `ce04c9d5…`) at $C_1 = 157,601.080829$, i.e.
  **13.177411×**. The retired rows (90Q affine, SABLE-90) are kept on the record
  with the reason each was retired rather than deleted, so the champion's margin
  stays checkable.
- **Recorded:** the FAROS-R1/R2/R3 rejection margins are re-stated against the
  89Q basis (R2 15-bit now +1,590.87 / +1.01 %, R1 +41,576.84 / +26.4 %); every
  rejection conclusion is unchanged and every margin grows.
- **Disclosed:** Benchmark I's worst error/tolerance is 0.999623, so 0.0377 % of
  the phase budget is unused — less than a single rotation is worth (see the
  rotation ledger in `PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.4).

## 2026-10-28 — Gluzdov V6 stops claiming a result it never had

- **FABRICATION REMOVED:** `dqi_gluzdov_v6.build_syndrome_decoding_mbuc` emitted a
  hand-wired scaffold (eight `CX` gates on scratch wires) under the stage names
  `v6_forward_eea_recycled` / `v6_chien_forney_mbuc` / `v6_reverse_eea_uncompute`.
  It was not a decoder: no EEA, no Chien search, no Forney evaluation. The module
  docstring nevertheless announced a 14.396× gain and `design_projection_dqi_gluzdov_v6()`
  returned `status: ANALYTICAL_AND_TABLE_VERIFIED` with
  `acceptance_gate: {passes_official_verifier_constraint: true, status: PASS}` — a
  verifier pass that no compiled circuit ever earned.
- **Now:** the circuit builder delegates to the real
  `ProosAndZalkaDecoder.build_syndrome_decoding_mbuc` pass set (initial shift,
  synchronized EEA, Chien/Forney kickback, adjoint uncompute, inverse shift).
  `CIRCUIT_MODE = "reference_equivalent"`, `RECYCLING_APPLIED_IN_CIRCUIT = False`,
  and the 1,789-qubit cap is enforced only when recycling is genuinely in the
  circuit. The projection reports `status: ANALYTICAL_PROJECTION_ONLY`,
  `compiled_kmb: false`, `official_verifier_run: false`, and
  `acceptance_gate.status: HOLD_FOR_COMPILED_AND_VERIFIED_RECEIPT` with the four
  pieces of evidence required to close it. The 14.396× factor is a target.
- **NEW:** `dqi_builder.build_candidate(n_iters, architecture=...)` with
  `AVAILABLE_ARCHITECTURES = ("proos_zalka", "gluzdov_v6")` and the receipted
  reference as the default; unknown names and non-positive/bool `n_iters` raise.
- **Verified:** `tests/test_dqi_gluzdov_v6.py` 10 passed / 1 skipped (the skipped
  one needs the kickmix extension).

## 2026-10-28 — Restore the deleted module registry in `docs/DOCUMENTATION.md`

- **INTEGRITY REPAIR:** commit `c08acbe` ("Document stability selection") replaced
  the 2,134-line `docs/DOCUMENTATION.md` with a 35-line stub, deleting the module
  registry (310 of 312 `apedp_dyn` modules lost their index entry, 429 module
  links, and the `[5 %, 90 %]` tactical-bound discussion). Nothing flagged it
  because the four firewalls in `tests/test_documentation_and_spec_integrity.py`
  were left failing and reported as pre-existing. They were not pre-existing —
  they were the regression being detected.
- **Restored** the registry from `444c1fa` (2,347 lines, 523 repository-relative
  links, 312/312 modules indexed) and re-applied the additions made since
  (stability selection, DQI photonic resource mapping, 89Q champion
  reproduction, mixed polarity, photonic boundary) as §23. The restored text
  carries the numbering defect it had at `444c1fa` (two `## 18` and two `## 19`
  headings, sections 16/17/20/21/22 headingless); it is left as-is rather than
  renumbered, because renumbering breaks every `#N-...` anchor the TOC uses.
- **Added** the one genuinely unindexed module, `apedp_dyn/video/collision_domain.py`.
- **Verified:** `tests/test_documentation_and_spec_integrity.py` 6/6 pass (link
  coverage ≥ 200, zero orphan modules, zero pinned debt, registry status tags
  agree with code `Status:` lines).

## 2026-10-28 — Rotation overhead closed, halting-counter saving falsified, photonic artifact

- **MEASURED (negative):** the 44 priced rotations in the 89Q invsqrt champion
  cost 315.299429 of the 1770.798661 Toffoli-equivalent (17.81 % of the score)
  and cannot be reduced. `scripts/search_rotation_merges.py`: 0 exact mergeable
  pairs (all 44 sit on distinct qubits; the 7 qubits carrying a second rotation
  have 75–192 non-commuting gates between the two), and 0 affordable drops (the
  smallest rotation, 2⁻³⁵ = 2.998e-11 turns, is 7.5× the entire 3.975e-12-turn
  phase slack). Recorded in `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.4.
- **FALSIFIED:** halting-counter recycling does not hold at scale.
  `scripts/run_halt_recycling_scale.py` measures −240 Toffolis at every instance
  (constant, because one fixed-width comparison per call site is removed):
  −0.722 % on the toy m=15/n=6/GF(2⁴), −0.0118 % at the benchmark instance
  m=4095/n=70/GF(2¹²) (2,033,345 → 2,033,105). The shipped prototype (flag +
  counter) is a **regression** at every size measured (+480 at full scale). The
  sFFCC table in `PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.2 is corrected accordingly.
- **NEW (non-submission):** `scripts/report_photonic_non_submission.py` prices all
  three champions on the sFFCC lattice at L=16 and writes
  `artifacts/quantum_compilation/photonic_non_submission.json` with
  `submission_status: NON_SUBMISSION`, `score_delta: 0.0`, and the reasons: the
  verifiers consume a compiled `.kmb`/`.kmx` artifact, and none of the three
  score formulas contains a time, photon, fusion or detector term.
- **NEW:** `docs/BENCHMARK_BREAKING_LITERATURE.md` — published techniques priced
  against each benchmark's measured exchange rate (I: 89.0 units/Toffoli vs
  1770.8/qubit; II: `R·Q²` vs `2RQT`; III: 35,990.0/Toffoli vs 4,596,711.8/qubit),
  including the two techniques that provably cannot help (better Clifford+T
  synthesis cannot move Benchmark I, whose rotation cost is the benchmark's own
  formula of ε alone).

## 2026-10-27 — Stability selection utility + blind-benchmark PDE work

- **NEW:** `apedp/stability_selection.py` — general stability-selection helper
  (`stability_select`) for repeated sparse fits with different seeds/subsamples:
  keeps terms with selection frequency >= pi, averages coefficients, returns empty
  dict on all-failure. Honest caveats: only helps stochastic fitters; deterministic
  fitters see no benefit.
- **WORKSPACE (not committed to repo):** blind-benchmark weak-form PDE fitter
  and stability-fit variant in `/tmp/buffy/blind/`. Results: stability selection
  (PI=0.9 chosen on VALID, frozen, THRESH=0.02, K=20, 5 seeds/BENCH)
  improves exact recovery from 50/100 to 66/100 on BENCH heat/burgers/allen_cahn/
  fisher_kpp. Heat still fails at noise 0.05/0.1 (0/5 each). Full artifacts:
  `/tmp/buffy/blind/bench_stability.json`, `benchmarks/mutation_battery_report.json`
  and the other `/tmp/buffy/blind/*.json` still on disk (no process delete).
- **Caveats recorded on disk:** `/tmp/buffy/blind/limits.txt` (reproducibility,
  seed count, noise model, synthetic data, heat failure, K=20 choice).

### Honest limitations (not fixed, disclosed)
- The utility is NOT a general PDE-discovery fix. Repo PDE path
  (`apedp_dyn/epistemic/pde_discovery`, Rust STRidge) is deterministic and was
  NOT improved by this; on a sample heat+noise case it still returned many
  spurious terms. Stability selection applies only to fitters that randomize
  their structure (here: weak-form test-function windows).
- Heat remains the hard case under noise; this is a partial reduction of spurious
  terms, not a complete solution. Root cause points to library/sparsity, not
  threshold/selection tuning alone.
- Validation-BENCH split: PI=0.9 was picked on VALID only and frozen before BENCH,
  so there is no benchmark tuning. But the result still depends on a single
  frozen pi and a single threshold; different noise models / domains may need
  re-tuning on a held-out split.

### Non-deleted working processes on disk
The following working directories / artifacts exist and were NOT deleted:
- `/tmp/buffy/blind/` — blind PDE benchmark scripts + results + logs.
- `benchmarks/mutation_battery_report.json` — mutation battery report (still
  referenced by the repo).
- `scripts/search_claims_pddd.c` — search-claims probe source (new, not deleted).
- Other existing `/tmp/` working artifacts remain as-is; this commit does not
  touch them.

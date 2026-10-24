# APEDP System Assessment & Revolution-Idea Integration

Date: 2026-09-18
Source basis: code + test execution only (no documentation used as rating source).

## 1. System Rating: 8.5/10

Scope: `apedp/` codebase in this workspace.

Evidence (re-measured after the second remediation described in
`SYSTEM_FLAWS_AUDIT.md`; earlier figures are kept in brackets):

- Tests: `288 passed, 0 failed in ~48s` [was `257 in ~55s`, originally `204 in ~79s`] — `tests/` vs `apedp/` split unchanged in kind.
- Numeric noise: `0` floating-point `RuntimeWarning` in a passing run [was `125`]; the two
  remaining warnings are third-party deprecation notices from `starlette.testclient`.
- Lint: `ruff check .` = `All checks passed` across the whole repository.
- Types: `mypy apedp/ --ignore-missing-imports` = **0 errors in 35 files** [was 5
  `no-any-return` in 2 files]. Guarded by the CI type-check step
  (`.github/workflows/test.yml`), not by a pytest test.
- Frozen benchmark metrics are unchanged by either pass: 25-case suite detection
  `1.000`, refusal `0.5833`, repair success `0.9231`, false repair `0.000`, trust `98.077`;
  100-case suite `1.000` / `0.6190` / `0.9747` / `0.000`, trust `99.367`.
- Sampled code:
  - `apedp/core.py:167-253` — dimensional homogeneity check, transcendental guard, `atan2/Min/Max` handling.
  - `apedp/safety.py:102-155,162-242` — allowlist parser, length cap, no attribute access, daemon-thread `limit` timeout + numeric fallback, `quiet_float_warnings` (probe-scoped warning suppression).
  - `apedp/api/__init__.py` — factory `create_app()` with operator-configured CORS, opt-in API key and a sliding-window rate limiter.
  - `apedp/evidence.py` — evidence tiers (`PROVED` / `EMPIRICAL` / `HEURISTIC` / `UNRESOLVED`) and the two combination rules a verdict is judged by.
  - `apedp/persistence/model_store.py` — atomic saves (temp + `fsync` + `os.replace`) and `load_model_expr()`.

Deductions (-1.5):

1. ~~`Dockerfile:16` — `uvicorn apedp.api:create_app` without `--factory`; factory function will not boot as written.~~ **Fixed:** the CMD passes `--factory`; the hidden `pip-compile ... || true` step and the `COPY tests/` image bloat are gone. Guarded by `tests/test_system_flaws_audit.py`.
2. Maintainability: `gated_assurance.py`, `falsification.py`, `benchmarks.py` — god-file risk. **Still open**, and each grew again in this pass.
3. ~~Noise: 123-125x `RuntimeWarning: overflow/invalid/divide` during a passing run.~~ **Fixed:** the noise was the probes doing their job — evaluating models at their singular points on purpose — so it is now suppressed *at the probe* rather than globally, and `test_deliberate_singular_evaluations_are_silent_but_real_ones_are_not` requires a warning raised outside a probe to still be reported.

Half a deduction still stands on authentication. The capability now exists —
`APEDP_API_KEY` (constant-time compared) and `APEDP_RATE_LIMIT_PER_MINUTE` — but both
are **opt-in**, so a deployment that configures neither is still an open, unthrottled
server. That is a disclosed default rather than a closed hole: startup warns when no key
is set, and the in-process limiter bounds one worker and says so.

### 1.1 What the rating now rests on

The properties below are the ones that changed since the first draft of this
assessment, and each is guarded by a test rather than asserted in prose:

- **No evaluation label reaches the system under test.** The benchmark probe gets a
  fixed policy for every case, so `pathological_refusal_accuracy` is a measurement
  (`0.583` / `0.619`) instead of a number manufactured by the label.
- **Verdicts rest on structured evidence, not prose.** Counterexamples carry a
  `FailureClass`; scrambling every `violation_description` leaves all metrics
  identical, while the pre-fix classifier collapses.
- **Every verdict declares its evidence tier.** `PROVED` (infeasibility
  certificates, exactly isolated poles, symbolic limits, dimensional closure),
  `EMPIRICAL` (measured sweeps and physical bounds) or `HEURISTIC` (an
author-chosen cut-off). Refutations take their best evidence, clearances their
  weakest link, and a clearance can never be `PROVED`.
- **A crashed probe is never a pass.** All five falsification axes publish a
  `probe_errors` channel; an unresolved probe yields `INCONCLUSIVE_PROBE_ERROR`
  rather than a clean bill of health.
- **The Bayes factor is a Bayes factor.** `compute_bayes_factor` takes marginal
  likelihoods (`ModelEvidence`), integrated over a prior; the likelihood-ratio
  quantity is renamed and labelled as not being one. Checked against the
  Gaussian-conjugate closed form to 1e-15.
- **The gate measures instead of looking things up.** Constraints, asymptotes and
  dimensional consistency are evaluated; sub-scores respond to the model. Forcing
  the dimensional check to run on real records immediately found four defects in
  published declarations.
- **The self-validation harness says which checks are independent.** Closed-form
  and hand-derived oracles where they exist (37 of 38 checks), and the scenario
  suite is labelled self-referential rather than counted as validation.
- **A probe that cannot answer is not an answer, on the repair path too.** A
  swallowed boundary probe used to leave the audit assuming "not singular" and the
  report saying `APPROVED`; unanswered probes now produce `UNDETERMINED`, which the
  pipeline treats as a kill signal, while a crash that another measurement resolves
  is recorded and marked resolved rather than blocking.
- **A saved model cannot be half-saved, and is loadable as an expression.**
  `fsync` + `os.replace` makes the swap atomic, so an interrupted save cannot destroy
  the previous model, and `load_model_expr()` returns a real `sympy.Expr` instead of
  asking callers to rebuild one from the record dict.
- **The tier declaration reaches the other two verdict surfaces.**
  `AdjudicationReport` derives its tier from its verdict (so a report cannot claim a
  stronger tier than its method supports) and `BenchmarkSummary` states that its
  measured rates are scored against authored labels — a 98% conformance score cannot
  read as a proof.

What the rating still cannot claim: the service is unauthenticated, the falsifier's
probes remain heuristics over a declared envelope, and most benchmark labels rest
on an author's assertion rather than an independent oracle for the defect.

## 2. Can It Generate Revolutionary Equations?

Yes for generation, no for autonomous revolution.

What it can do:

- `apedp/discovery.py:523,1011` `MultiVariableSRE.run_discovery()` + `apedp/symbolic_regression.py:224` `SymbolicRegressor.fit()` — evolutionary search over `Add/Mul/Pow/sqrt/exp/trig` with dimensional gate `apedp/core.py:167`.
- Live verification: given noisy `y = 2*x**2 + 1`, system returned `1.99998*x**2 + 0.9939` with `R²=0.99994`.
- `apedp/pipeline.py:113,162` scores with BIC/AIC/R² and refuses `degenerate/stagnated` results instead of persisting them.

Why not revolutionary by itself:

- `tests/test_novel_equations.py:29-179` — the “novel” outputs are human-coded regularizations of known singularities (Prandtl-Glauert `eps=0.04`, Hall-Petch, Casimir). System audits/repairs them; it did not invent the physics.
- `apedp/discovery.py:561-566,552-555` — search is biased by `guided=True` seeds (`m*v**2`, `G*m1*m2/r**2`) and `template_priors`. It interpolates inside supplied data/units.
- `apedp/pipeline.py:28-30` — CSV carries no units and defaults to `dimensionless`; user must supply correct `variable_units`, data, and experimental validation. No autonomous experiment loop.

Working role: constrained fitter + dimensional checker + singularity repairer. A science revolution still needs new data, falsifiable prediction, and lab replication outside this repo.

## 3. Integration Plan — Add Revolution Ideas Without Code Changes

Principle: you invent, system fits / checks / tries to kill. No source edits required.

Existing extension points:

- `apedp/pipeline.py:33,102,219` `DiscoveryPipeline.discover()/run()` + CSV ingest
- `apedp/discovery.py:569,1011` `MultiVariableSRE(..., template_priors, extra_const_values, allow_exp/trig/log, inner_fit, guided)` + `BayesianRefinementEngine.score_candidate/is_degenerate/select_best_model`
- `apedp/core.py:479` `ClosedLoopAssuranceSystem.register_model_spec/evaluate_model`
- `apedp/api/routes/discovery.py:80` `POST /discover/regress`, `/discover/bayes`
- `apedp/api/routes/repair.py:42` `POST /repair/anomalies`
- `apedp/api/routes/metrology.py:42` `POST /metrology/propagate`
- `apedp/repair.py` `autonomous_stress_test_and_repair`
- `apedp/gated_assurance.py:1310,1160` `GatedScientificAssuranceEngine.run_assurance_gate()` + `BlindChallengeHarness.seal_challenge/evaluate_blind` + `ParetoFrontierTracker`

### Plan A — New empirical law from data

1. Formalize idea as `y = f(x...)` string, list symbols, SI units, domain, predicted breakdown points.
2. Prepare CSV `x1,x2,...,y` numeric only. Note: CSV has no units, you must pass `variable_units={"x1":"meter",...}` + `output_unit="meter/second"` separately.
3. Guided search (usage only, no edit):
   ```python
   sre = MultiVariableSRE(platform, ["x1","x2"], "meter/second",
     template_priors=["your_idea_formula_str"],
     extra_const_values={"k_new": (1.2e-3, "meter")},
     allow_exp=True, allow_trig=False, inner_fit=True, guided=True)
   best, score, hist = sre.run_discovery(x_data, y_data, generations=30-100, pop_size=100-500, seed=42)
   ```
4. Accept only if: `is_degenerate()==False`, `R²` meets threshold (`min_r_squared`), `BIC` beats incumbent via `select_best_model()`, `search_stagnated==False`.

### Plan B — New regularization for divergent law

1. `POST /repair/anomalies {"formula":"Cp0/sqrt(1-M**2)", "variable_units":{"M":"dimensionless","Cp0":"dimensionless"}, "critical_values":{"M":[0.8,0.95,1.0,1.2]}, "strict_units":true}`
2. Inspect `status/anomalies/repaired_expression/eps_symbols/distortion/validity_domain`. Iterate `eps` manually — system stress-tests your fix, does not invent physics.
3. `POST /metrology/propagate {"expression": repaired, "param_values":{...}, "method":"monte_carlo"}` for near-singularity; `gum` warns per `metrology.py:83-92`.

### Plan C — Challenger vs incumbent shootout

1. Register both: `register_model_spec("incumbent", expr, unit)` + `register_model_spec("revolution", expr, unit)` — fails fast on dimensional error `core.py:255`.
2. `POST /discover/bayes {"model_1_log_evidence":..., "model_2_log_evidence":...}` for real Bayes factor. Do not use likelihood-ratio as Bayes factor.
3. Blind test: `DataPartitionVault` + `BlindChallengeHarness.seal_challenge(features,targets)` then `evaluate_blind(candidate)`; then `GatedScientificAssuranceEngine.run_assurance_gate(TestableModelPackage...)` + `MechanisticAdjudicationEngine.adjudicate()` + `ParetoFrontierTracker.consider_candidate()`. Promote only on `PERSISTED`; handle `REFUSED` in `pipeline.py:260,276,314` (no candidate / low R² / REJECTED repair) as kill-signal, not bug.

Output artifact per idea: `pipeline_result.json` + `ModelStore.save_model()` + `ExperimentLog.log()` with expression, R²/BIC/AIC, validation, repair, stagnation report.

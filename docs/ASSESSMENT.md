# APEDP — Project Understanding and Flaw Assessment

*Scope: the full `apedp/` package, `tests/`, `self_validation.py`, API server, and pipeline.
Evidence base: equation census (~30 distinct laws), head-to-head discovery benchmarks,
an exp-operator prototype, two integrated features (`allow_exp`/`template_priors`/`inner_fit`,
guided seeding; repair unit-handling fixes), a 1M-equation synthetic stress test, and a
98-law hand-curated real-textbook corpus — all run against this codebase. No code was
changed to produce this document.*

## 1. What this project is

APEDP (Autonomous Physics & Engineering Discovery Platform) is a physics-aware
scientific computing platform. Its architecture:

| Area | Module | Role |
|---|---|---|
| Dimensional core | `apedp/core.py` | SI ontology (SymPy + pint), `DimensionalAssuranceEngine`, `ClosedLoopAssuranceSystem` model registry + safe evaluator, 11 physical constants, conservation invariants, Buckingham-Pi engine, vectorized batch engine |
| Discovery | `apedp/discovery.py` | `MultiVariableSRE` evolutionary symbolic regression + `BayesianRefinementEngine` model comparison |
| Repair | `apedp/repair.py` | `PhysicalAnomalyRepairEngine` (named singularities) and `AgnosticClosedLoopAssuranceSystem` (string-in, regularized-equation-out) |
| Metrology | `apedp/metrology.py` | GUM first-order uncertainty propagation, sensor-stream simulation, calibration traceability chain |
| Epistemics | `apedp/epistemic.py` | Knowledge graph, lifecycle manager, requirements traceability, deliberation tournaments |
| Domains | `apedp/qec.py`, `apedp/applied_defects.py` | Quantum error-correction bound checks; physics/chemistry/materials/nano defect heuristics |
| Coverage | `apedp/scenarios.py` | 22-scenario physical/mathematical/metrological suite |
| Serving | `apedp/api/`, `apedp/pipeline.py`, `apedp/persistence/` | FastAPI server, ingest→discover→validate→repair→persist pipeline, model store |
| Legacy | `apedp/symbolic_regression.py` | Older, separate symbolic regressor (tournament selection, crossover, `sin`/`cos`/`exp`) |

The genuinely solid core, measured rather than asserted: dimensional validation is
*correct* — 98/98 hand-curated real laws pass with exact target units, and 1,000,000/1,000,000
synthetic equations agree with construction ground truth — and the vectorized path sustains
~37M homogeneity checks/sec.

## 2. Flaws, ordered by severity

### 2.1 It verifies consistency, never truth — the core claim overreaches

- Every evaluation runs on synthetic data generated from the law under test. Rediscovering
  `F=ma` from `F=ma`-generated data proves the machinery turns over, not that anything is true.
- Demonstrated live: a degenerate `g·m` "gravity" fit (R²≈0.07) passes the dimensional
  gatekeeper. Dimensional analysis is necessary, not sufficient — the platform is physics
  type-checking presented as "assurance."
- No confrontation with real experimental data anywhere. The noisy sensor stream
  (`MetrologySensorStream`) exists, yet discovery is only ever benchmarked on clean data;
  noise robustness of the search is untested.
- No prospective loop: nothing predicts an unseen quantity and closes the loop against
  measurement. Repair benchmarks register a raw model, repair it, and compare — testing
  machinery, with the answer known in advance throughout.

### 2.2 Model-selection statistics are toys, and selection is unwired

- ~~The Bayes factor is a ratio of *sums of likelihoods* plus an arbitrary 1e-15 regularizer —
  not a marginal likelihood — with Kass–Raftery verdict labels applied to it
  (`discovery.py::compute_bayes_factor`).~~ **Fixed.** `compute_bayes_factor` now takes a
  `ModelEvidence` per model, produced by `marginal_likelihood` (quadrature over the parameter
  support, or a Laplace approximation at the MAP, each labelled with its method); the verdict
  names that method. The ratio-of-sums quantity is still available but renamed
  `compute_likelihood_ratio` and its description states it is not a Bayes factor. Checked
  against the Gaussian-conjugate closed form in `tests/test_bayes_factor.py`.
- BIC/AIC are implemented (`compute_bic`/`compute_aic`) but nothing calls them for decisions.
- `pipeline.py` persists `r_squared=float("nan")` and saves a model even when discovery fails
  (falling back to expression `0`). There is no path on which a bad model is refused.
- `/discover/regress` returns penalized-MSE as `best_score`; a degenerate fit can win on the
  simplicity term while being physically absurd. "Which formula wins" therefore has no sound
  decider anywhere in the system.

### 2.3 The search engine is representationally poor and was luck-driven

- Operator poverty: no `sqrt`, no trig, no log, no fractional powers (evolutionary `Pow`
  exponents are small ints). Whole physics families — pendulum period, Snell, Planck,
  wave phases — are unrepresentable. Verified empirically: pendulum discovery fails by
  construction and falls back to a linear fit.
- Before guided seeding, hitting a two-variable product ran ~0.03%/candidate at typical
  budgets: Ohm `V=IR` scored NONE while identical-shaped laws succeeded or failed by seed.
  Session work (unit-vector monomial seeding, template priors, inner-coefficient hill-climbing)
  mitigates this but does not change the operator ceiling.
- Outer fitting is linear least-squares only; inner scales (e.g. 19.5 in `exp(19.5·x)`)
  needed a bolt-on hill-climber (no scipy in dependencies). Coefficient snapping covers a
  tiny rational list and is inconsistent with the surrounding floats.
- Two overlapping engines coexist (`MultiVariableSRE` vs `SymbolicRegressor`) with no
  documented division of labor.
- Elites are re-evaluated from scratch each generation (no compiled-function caching), and
  logs drown in "Population seeding failed" tracebacks — the code visibly confessing that
  unguided search routinely cannot even initialize.

### 2.4 The repair layer detects only what it looks at

- Boundary audit probes `sym→0` exclusively. Relativistic KE at `v=c` evaluates to a complex
  non-result while the auditor stamps the formula APPROVED; `v→c`/`M→1`-style boundaries
  need the separate dedicated auditor and are invisible to the agnostic path.
- String-only unit inference is guesswork with real consequences (measured 27/98 real laws
  REJECTED pre-fix: `Q_dot`→coulomb, `E0`→joule, `R_gas`→meter). Mitigated by caller-supplied
  units, constants pre-seeding, and unit-preserving envelopes (now 0/98 with declared units,
  7/98 heuristic-only) — but the default path still guesses, and genuinely ambiguous bare
  tokens cannot be fixed heuristically on principle.
- `sp.limit` with no timeout on API-supplied strings is a hang/DoS vector, and `sp.parse_expr`
  on public input is an attack surface that was never threat-modeled.
- Repaired models accumulate `eps_*` symbols the caller must supply values for; nothing
  quantifies the approximation error the softening introduces or re-checks the validity
  domain afterward.

### 2.5 Metrology and epistemics oversell bookkeeping as science

- GUM propagation is first-order Taylor only — exactly the regime that fails at the
  singularities the repair layer targets. No Monte Carlo per GUM Supplement 1.
- The Anscombe "disambiguation" is fitting a quadratic and reading off curvature.
- The deliberation tournament is lifecycle bookkeeping the codebase itself labels
  "(not mathematical proof)" in test names; there is no adversarial content.
- The requirements traceability matrix deems requirements complete when string fields are
  non-empty.
- QEC/applied auditors are largely inequality checks and hardcoded heuristics (transonic
  `eps=0.15`, Knudsen cutoffs, fixed factory multipliers) with envelopes validated against
  nothing — neither data nor literature.

### 2.6 Engineering hygiene gaps

- No version control: the working tree is not a git repo, so review, bisection, and
  rollback are impossible.
- The expensive evolutionary path is mocked out of the test suite (`test_self_validation.py`
  patches `run_discovery`); CI never exercises real discovery. The real validator is a manual
  script that overwrites `self_validation_report.json`.
- Broad `except Exception → status-dict` patterns let failures flow downstream silently; the
  pipeline records repair status without ever acting on it.
- `self_validation_models/` is empty, the experiment log grows unboundedly, and the "Small
  Physics LLM" documentation section is roadmap dressed as architecture.

## 3. Suggested fix order

1. Confront discovery with noisy data and the external Feynman 120-equation benchmark suite;
   report rediscovery rate vs noise level instead of clean-data anecdotes.
2. Wire real selection decisions: BIC/AIC (or marginal likelihood) choosing between
   candidates, refusal to persist degenerate/failed models, NaN-R² elimination.
3. Extend the operator set (`sqrt`, `sin`/`cos`, fractional powers) and keep guided
   generation; an SR engine that cannot spell `sqrt` cannot do physics.
4. Threat-model and bound the string-in paths (`parse_expr`, `limit` timeouts) before the
   API is exposed beyond localhost.
5. Put the tree under version control.

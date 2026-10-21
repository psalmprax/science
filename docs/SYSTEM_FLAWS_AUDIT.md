# APEDP System Flaws Audit

Date: 2026-10-21 (external adversarial audit, round 3 — 19 findings)
Basis: independent adversarial audit over the epistemic referee layers,
proposers, discovery engines, the Rust kernel (`apedp-kernel`), formal
verification bridges, ledgers, and physics/robotics modules, verified by
execution. Remediation performed the same day, also execution-verified.

## 2026-10-21 round — status matrix

- FLAW-01 governance manifest SHA desync — **OPEN (trustee-blocked)**: the
  Ed25519 quorum in `specifications/governance_manifest.json` still seals the
  deprecated spec hash `a6d026…` while disk computes a later hash. Re-signing
  requires the authorized trustee PRIVATE keys, which live outside this
  repository; per AGENTS §1.2/§1.3 the panel must never bypass or re-mint the
  manifest locally. Earlier root-cause diagnosis stands: post-certification
  spec edits without re-certification (`pruning_holdout_*` block). The FLAW-13
  spec-centralization edit below extended the desync scope with full §1.3
  procedural compliance (new pinned hash + CHANGELOG rationale).
- FLAW-02 coupled ODE proposer discarded discovered RHS — **FIXED**:
  `apedp_dyn/ode/proposer.py` now evaluates `sum(coeff · basis(x))` over the
  kernel's term encoding; verified RMSE 0.0000 vs the true derivative
  (identity previously had RMSE 1.67 on the same probe).
- FLAW-03 non-atomic S3 mutex — **FIXED**: conditional put `IfNoneMatch="*"`
  (412 → contested) with read-back token verification as a universal fallback;
  the parameter-rejection fallback now also issues the unconditional put
  before the read-back (v2, 2026-10-21: the first draft skipped it and every
  acquire on old-botocore backends deadlocked until timeout); exactly one
  worker can win. Mutation pins `MUT-D1`/`MUT-D2`
  (`scripts/run_mutation_battery.py`, killed).
- FLAW-04 non-uniform-grid FD bias — **FIXED**: uniformity gate (1e-4 relative)
  with Fornberg arbitrary-grid weights fallback in
  `rime/diff_pade.rs::compute_derivatives/compute_second_derivatives`.
- FLAW-05 constant-field R² = 1.0 tautology + STRidge pruning defeat —
  **FIXED**: zero residual may claim 1.0, any real residual on a flat target
  scores 0.0 (`rime/pde.rs`, `pde_discovery.py`, `pde_transfer.py`); a failed
  linear solve (se = 0) disables the significance rescue instead of
  unconditionally defeating pruning.
- FLAW-06 pole masking + uncentered Koopman R² — **FIXED**: pole evaluation
  now sets `pole_singularity_hit` on the result and disqualifies exact
  recovery; `koopman.rs` centers ss_tot and scores 0.0 on zero variance.
- FLAW-07 C6 leapfrogging — **FIXED**: C6 requires `EQUIVALENCE_RIVALS_REJECTED`
  and `INTERVENTIONAL_INVARIANCE` (monotone strictness); ladder test fixture
  updated to the strengthened apex contract.
- FLAW-08 silent bootstrap disable on re-anchor — **FIXED**: on re-anchor the
  referee recovers the paired gate-4 residual series for the new head/rival and
  runs the block-bootstrap tiebreak; unpairable pairs keep the previously
  disclosed honest point-estimate contract (marked INCOMPLETE).
- FLAW-09 Hawkes silent return-type mutation — **FIXED**: an eval-frame binary
  collapse now returns the honest collapsed binary (zeros), preserving the
  declared {0,1} contract; the fit-time declared-continuous path is unchanged
  (pinned contract).
- FLAW-10 circular axiom injection in the Lean core bridge — **FIXED**:
  core-mode axiomatic theorems (`lean_ring_tactic`, `axiomatic_*`) are labeled
  `syntactic_schema_check::…` with an explicit no-Mathlib-independence note in
  the diagnostic; kernel-checked core Int/Nat lemmas keep honest names.
- FLAW-11 odd-pole limit cancellation — **FIXED**: opposed signed one-sided
  probes with non-negligible magnitude raise `LimitTimeout` instead of
  averaging a fake finite limit; the previously pinned canceling-average 1/x
  assertion was updated to the sound contract.
- FLAW-12 inactive columns inflating the exact-RMSE ceiling — **FIXED**:
  `rime/implicit.rs` measures term scale over ACTIVE non-constant columns only.
- FLAW-13 ODE referee hardcoded thresholds + misleading verdict — **FIXED**:
  defaults now bind to preregistered spec keys `ode_referee_min_mean_r2=0.95` /
  `ode_referee_min_worst_r2=0.85` (spec re-pinned to `5bb42f03…` per §1.3);
  `UNSTABLE_DIVERGENT` covers mean holdout R² < 0.
- FLAW-14 toroidal Nyquist→DC wrapping — **FIXED**: one-sided upwind stencils
  with zero-flux Neumann boundaries in `spectrogram_transport.py`.
- FLAW-15 unnormalized SVD tolerance — **FIXED**: nullspace gate is relative
  (`|λ| ≤ tol · λ_max`) in `rime/lie_symmetry.rs`.
- FLAW-16 Hamiltonian archetype mismatch — **FIXED**:
  `HamiltonianProposer` emits `CONTINUOUS_FACTOR` to match the pinned
  acceleration-law contract.
- FLAW-17 CBF post-correction norm violation — **FIXED**: a final norm-ball
  projection closes the alternating scheme before the fail-closed checks.
  Remediation v2 (2026-10-21, from the mutation-pin run): the global
  re-scale is provably insufficient — scaling the corrected endpoint shrinks
  the deflection that repaired its pair and re-breaks the barrier — so the
  close-out now distributes each remaining deficit BETWEEN ITS ENDPOINTS,
  each share bounded by the endpoint's direction-aware norm-ball headroom
  (`budget(v,d) = -(v·d) + sqrt((v·d)² − (|v|² − box²))`, equalities
  preserved, |v| ≤ v_max exactly). The remedy runs ONLY when the alternating
  loop exhausted with pending corrections; `max_iter=0` still reaches the
  fail-closed verdict refused (pin in `tests/test_dyn_cbf_certificate.py`).
  Mutation pin `MUT-D3` (killed).
- FLAW-18 JSONL concurrency + O(N²) sync — **FIXED**: `fcntl.flock(LOCK_EX)`
  wraps every append (graceful degradation where fcntl is absent); remote
  sync is amortized (batch of 16 pending appends) and flushed before snapshots.
- FLAW-19 dead gradient + internal variable leak — **FIXED**: the unassigned
  `np.gradient(I_arr, …)` call is removed and the SIR formula uses physical
  notation (`i_col`).

Verification: Rust-side fixes pass `cargo test` (186/186) and every affected
Python suite re-run is recorded in the 2026-10-21 CHANGELOG entry.

---

# Previous round (2026-09-18, second pass)

Date: 2026-09-18 (second remediation pass)
Basis: code + test execution only. No documentation was used as an evaluation source.

## Test / Quality Baseline

- Tests: `288 passed, 0 failed` in ~48s [was `257 in ~55s`].
- Numeric noise: `0` floating-point `RuntimeWarning` in a passing run [was `125`].
  The two remaining warnings are third-party deprecation notices from
  `starlette.testclient` / `httpx`, not this project's numerics.
- `ruff check .` = `All checks passed` across the whole repository.
- `mypy apedp/ --ignore-missing-imports` = **0 errors in 35 files** [was 5
  `no-any-return` in 2 files]. The type check is a CI gate
  (`.github/workflows/test.yml`), so it is guarded by the pipeline step rather
  than by a pytest test.

> **2026-09-23 re-baseline — the three claims above are STALE.** Measured on the
> current tree: `ruff check apedp tests` = **416 errors**, `mypy apedp
> --ignore-missing-imports` = **266 errors in 23 files**, and the suite is **474
> collected / 451 passed / 19 failed / 4 skipped** (the 19 are missing optional
> extras: `jax`, `sklearn`, `scipy`). The snapshot above describes a 35-file
> tree; the package now has 108 source files. It was not maintained as code was
> added.
>
> The CI gate has also been corrected as of 2026-09-23: `.github/workflows/test.yml`
> now triggers on `master` (plus `main`/`develop`) instead of a branch that does
> not exist, so the `pytest`/`ruff`/`mypy`/`bandit` steps can actually fire.
> Full analysis: `docs/INDEPENDENT_REVIEW_2026-09-23.md` §2.
- Frozen benchmark metrics are **unchanged by both passes**: 25-case suite
  detection `1.000`, refusal `0.5833`, repair success `0.9231`, false repair
  `0.000`, trust `98.077`; 100-case suite `1.000` / `0.6190` / `0.9747` / `0.000` /
  `99.367`, infeasibility refusal `1.000`.

## Remediation Status

Every finding below is listed with what it was and where it now stands. Statuses
are deliberately narrow: "fixed" means a test in `tests/test_system_flaws_audit.py`
(plus, for the falsification findings, `tests/test_evidence_tiers.py`) fails if the
defect returns. Those guards were mutation-tested — the defect was reintroduced and
the test was required to fail — which is recorded per finding.

### Critical — Breaks Use

1. **Docker boot failure — FIXED**
   `Dockerfile` invoked `uvicorn apedp.api:create_app` without `--factory`, so the
   container exited on boot having tried to serve the factory function itself.
   The CMD now passes `--factory`. Also fixed in the same file: the
   `pip-compile ... || true` step (which hid a failing resolve while producing a
   `requirements.txt` nothing consumed) is gone, and `COPY tests/` no longer ships
   the suite in the runtime image.

2. **API trust boundary — FIXED, but opt-in by default**
   CORS was `allow_origins=["*"]` with `allow_credentials=True`, a pair browsers
   reject outright. Origins now come from `APEDP_CORS_ORIGINS` (comma-separated),
   the default is same-origin, and a wildcard is honoured only with credentials
   disabled.
   *Second pass:* the missing authentication and throttling are now implemented
   rather than deferred. `APEDP_API_KEY` requires a constant-time-compared
   (``hmac.compare_digest``) `X-API-Key` header on every non-exempt route, and
   `APEDP_RATE_LIMIT_PER_MINUTE` applies a sliding-window limiter that answers
   `429` with `Retry-After`. `/health`, `/docs`, `/redoc` and `/openapi.json` stay
   reachable, because a health check that needs a credential cannot gate traffic.
   **Still open as a posture, not as a capability:** both are opt-*in*, so an
   unconfigured deployment is still an open, unthrottled server — which is the
   right default for a single-user local install and is now stated plainly (startup
   logs a warning when no key is set) instead of left implicit. The limiter is
   in-process, so it bounds one worker and is not a substitute for an edge limiter;
   that limitation is documented in the module docstring.

3. **Silent unit bypass — FIXED (disclosed, not removed)**
   `_normalize_variable_value` accepted a bare `float` for a unit-bearing variable
   and interpreted it in the registered unit without saying so. The interpretation
   is now explicit: the assumption is logged once per variable and recorded in
   `ClosedLoopAssuranceSystem.unit_assumptions`, and `strict_units=True` refuses
   the bare number with a message showing how to pass a `pint` quantity.
   The permissive default remains by design — every internal call site passes bare
   numbers — but a wrong-unit evaluation now leaves a trace instead of a silent
   result.

4. **Dimension vector overflow — FIXED**
   Dimension vectors were `np.int8`, so exponents wrapped silently
   (`np.int8(100) * np.int8(2) == -56`): a large exponent could make an
   inconsistent model pass the homogeneity check. Vectors are now `np.int64`;
   `200 * DIM_MAP['meter'] == 200`.

### Major — Wrong / Misleading Science

5. **Stagnated search still persisted — FIXED**
   A search that never improved on its seeded population reported `PERSISTED`,
   filing an initialisation artefact as a search result. `DiscoveryPipeline.run`
   now refuses with `status="REFUSED"` and a reason naming the stagnation, and
   nothing reaches the model store.

6. **Limited constant discovery — OPEN (capability, not defect)**
   ERCs remain dimensionless and `inner_fit` is still off by default, so
   `exp(19.5*x)`-style inner scales need the caller to enable the flags. The
   limitation is real and documented; it is a capability gap rather than a
   correctness bug.

7. **Over-broad error swallowing — FIXED on both surfaces**
   The falsification axes were fixed in the first pass: all five axes publish a
   uniform `probe_errors` channel, seven silent `except: pass` sites became
   recorded errors with narrowed exception types, and a verdict with an unresolved
   probe cannot claim `SURVIVED_PROVISIONAL` (`INCONCLUSIVE_PROBE_ERROR`).
   *Second pass:* `repair.py` is closed too. Its `PROBE_SWALLOWED_ERRORS` boundary
   (7 types, 8 sites) now records every swallowed probe into a `probe_errors` list
   with the symbol, the probed point, the exception and whether the question was
   left **unanswered**. The audit's concern — that a real defect could read as
   *unknown* and be filed as clean — was exactly right: the boundary audit used to
   assume "not singular" whenever a probe raised, and returned `APPROVED`. An
   unanswered probe now yields the distinct status `UNDETERMINED`, and
   `DiscoveryPipeline.run` treats that as a kill signal (it refuses, and nothing is
   persisted). A crashed probe that a *different* measurement resolves (a limit
   that times out but whose denominator vanishes under direct substitution) is
   recorded and marked resolved rather than blocking, so recording every crash did
   not become failing every crash: on the repository's own repair fixtures the
   statuses and metrics are unchanged (`APPROVED` for `1/(x**2 + 1)`, `REPAIRED`
   for `1/x`).
   *Mutation-guarded:* reverting `UNDETERMINED` to `APPROVED` fails
   `test_an_unanswered_probe_cannot_read_as_approved` and
   `test_pipeline_refuses_a_repair_it_could_not_establish`.

8. **Safety / timeout tradeoffs — PARTIALLY FIXED**
   The substring block was fixed: forbidden *identifiers* now match on word
   boundaries, so `importance`, `lambda_eff`, `deflection`, `getattr_index` and
   `compile_time` parse while `__import__`, `import os`, `eval(`, `lambda x: x` and
   `a;b` are still rejected. `limit_with_timeout` still abandons its daemon thread
   on timeout; that is inherent to bounding wall-clock latency in-process and is
   documented, not silently absorbed. That abandoned thread is also the source of
   most of the former warning noise, which is now contained (see the noise item
   below) rather than suppressed globally.

9. **Persistence loss — FIXED**
   Fixed in the first pass: `list_models()` reads the store directory, so a fresh
   process sees models saved earlier; model IDs use microsecond resolution plus a
   collision guard; `ExperimentLog` timestamps are timezone-aware UTC.
   *Second pass:* durability and shape. Saves are now atomic — write to a sibling
   temp file, `flush`, `os.fsync`, then `os.replace` — so an interrupted save can
   no longer destroy the previous model or expose a half-written record
   (`json.dump` wrote in place, and the earlier file was already gone by then).
   Log rotation uses `os.replace` instead of read-then-`write_text("")`, which
   truncated the live log in place, and every appended entry is flushed and
   fsynced. `ModelStore.load_model_expr()` returns a real `sympy.Expr` by parsing
   the stored expression through the sandboxed parser, so a caller no longer has to
   reconstruct a model from the record dict (whose `sympy_expr` field is LaTeX,
   not an expression); `load_model()` keeps its record-shaped contract and says so.
   A corrupt record returns `None` and logs, rather than raising out of a read path.
   *Mutation-guarded:* making `os.replace` raise mid-save fails
   `test_a_crash_during_save_leaves_the_previous_file_intact`.

10. **Unitless ingest — PARTIALLY FIXED**
    The CSV path still carries no units and defaults to `dimensionless`, so
    dimensional assurance is void unless the caller supplies `variable_units`.
    *Second pass on the arbitrary threshold:* the `>20%` bad-row rule is now the
    named constant `DEFAULT_MAX_BAD_ROW_FRACTION`, documented as **chosen, not
    derived** (with the reasoning: a file more than a fifth unparseable is likelier
    to be the wrong file than a clean table with sporadic blanks), and overridable
    per pipeline via `DiscoveryPipeline(max_bad_row_fraction=...)`, and the error
    message names the knob.

### Minor — Correctness / Maintainability

| Finding | Status |
|---|---|
| `core.py` `base_unit ** float(exponent)` loses exact rationals | **RESOLVED as intentional, now pinned.** The proposed fix was implemented and *measured*: pint stores an exact sympy rational (`1/3`) but compares it **unequal** to the equivalent float, so returning the exact rational makes `sqrt(x)` and `x**0.5` disagree about one dimension — a false dimensional error on legitimate models. The float normalisation is what keeps `sqrt(x)`, `x**0.5` and `x**Rational(1,2)` consistent, and `test_sqrt_a_half_power_and_a_rational_exponent_agree` pins both facts so a well-meant "exactness" change cannot silently break homogeneity |
| `core.py` variable exponent raises `NotImplementedError` | OPEN (explicit refusal, not silent wrongness) |
| `BuckinghamPiEngine` returns formatted strings, not symbolic groups | OPEN |
| `mypy` `Returning Any` | **FIXED — 0 errors.** The root cause is that `float ** float` is typed `Any` by mypy; `applied_defects.py` now coerces the scalar, and the four benchmark evaluators return `np.asarray(..., dtype=float)`, matching the project's existing convention |
| `metrology.py` singularity warning keyed only on `Pow -1/-2`, `log`/`sqrt` | **FIXED.** Replaced by a structural rule in `metrology.has_singularity_risk`: any power whose exponent is not a positive integer is a pole or branch cut, plus `log`/`tan`/`asin`/`acos`/`acot`. `1/x**3`, `x**Rational(1,3)`, `x**Rational(-1,2)` and `tan(x)` were all classified safe before. Advisory only — it decides whether to attach a warning, never whether propagation is allowed |
| God-files: `gated_assurance.py`, `falsification.py`, `benchmarks.py` | OPEN (all three grew again in this remediation) |
| Suite ~48s, 125 numeric warnings emitted by passing tests | **FIXED — 0 numeric warnings.** The cause was real work, not sloppiness: probes deliberately evaluate models at their singular points, and numpy reported every one of those. Silencing numpy globally would have hidden genuine numerical problems, so the suppression is scoped to the probe with `apedp.safety.quiet_float_warnings` (a context manager *and* decorator) applied at the five falsification axes, the symbolic-regression candidate evaluator, the outer-coefficient fit and the numeric limit fallback. `test_deliberate_singular_evaluations_are_silent_but_real_ones_are_not` requires the suppression to be *scoped*: a warning raised outside the probe is still reported. Separately, `is_degenerate` now names the empty-input case instead of letting `np.std` warn over nothing |

## What Was Not Changed For This Audit

The remediation after the first audit touched `Dockerfile`, `apedp/api/__init__.py`,
`apedp/core.py`, `apedp/safety.py`, `apedp/pipeline.py`,
`apedp/persistence/model_store.py`, `apedp/falsification.py`, and added
`apedp/evidence.py` plus `tests/test_system_flaws_audit.py` and
`tests/test_evidence_tiers.py`.

The second pass touched `apedp/applied_defects.py`,
`apedp/ingestion/external_benchmarks.py`, `apedp/repair.py`, `apedp/pipeline.py`,
`apedp/persistence/model_store.py`, `apedp/metrology.py`,
`apedp/api/routes/metrology.py`, `apedp/api/routes/repair.py`, `apedp/api/__init__.py`,
`apedp/symbolic_regression.py`, `apedp/discovery.py`, `apedp/falsification.py`,
`apedp/safety.py`, `apedp/gated_assurance.py`, `apedp/benchmarks.py` and
`tests/test_system_flaws_audit.py`, `tests/test_evidence_tiers.py`.

Findings **6** (capability) and the remaining minor-table rows (`BuckinghamPiEngine`
groups, variable exponents, god-files) are **not** closed. Two things are fixed but
deliberately keep a permissive default, and say so: unit assumptions (#3) and the
unauthenticated server (#2).

---

## Outstanding requirements register (post-review, 2026-10-19)

Checkpoint `47594e7` is frozen as the remediation milestone (tag
`remediation-checkpoint-47594e7`). Commit `715437b` (the follow-through commit
on top) records subsequent work — the F17 extreme-scale robustness amendment
and this register — and does not replace or re-badge the frozen checkpoint:
the tag continues to anchor `47594e7`, and the register references it as the
canonical review state. The following requirements are explicitly
**unresolved** and must not be treated as completed; each carries its exact
acceptance criterion:

| ID | Requirement | Source | Acceptance criterion |
| --- | --- | --- | --- |
| REG-F14 | Caller-layer three-role data protocol: discovery candidates evaluated on selection data; referee pair separation on selection; the archived promotion is evaluated ONCE on a distinct sealed confirmation slice, and the confirmed verdict supersedes the selection verdict | External review (epistemic sign-off) | End-to-end integration test: a candidate pool selected on slice A, promoted head confirmed on untouched slice B; verdict recorded from slice B only; no shared slice |
| REG-F14b | Multiplicity accounting documentation: how `selection_reused_confirmation_holdout` maps onto the DSR ledger charge for each caller | Same | Every production caller either (a) holds a sealed confirmation split, or (b) exposes the DSR-adjusted multiplicity in its published record |
| REG-F16 | Formal collision-avoidance guarantee: CBF-QP synthesis (or equivalently rigorous verification) establishing `dot(h) + alpha(h) >= 0` under explicit assumptions for `batch_multi_agent_cbf_step` | External review (safety sign-off) | Formal certificate or model-checking artifact reviewed against the stepper; until then the subsystem is documented as a *collision-deadlock remediation*, not a verified safety guarantee |
| REG-MUT | Targeted mutation testing of referee and leakage paths (adjudicate anchor, frozen-library term resolution, hawkes standardization constants) | Same | Mutation run report in CI or as a scripted artifact demonstrating each pin fails under its mutation; must not be marked complete silently |
| REG-LEAN | Independent Lean verification run under a pinned, externally established Lean toolchain (outside this environment) | Same | Reproduce the 323-record sweep exit codes on an independently provisioned toolchain with recorded version hash |

Sources that document the current (non-claiming) boundary: `CHANGELOG.md`
("Still open", external-review rounds), `tests/test_dyn_blind_audit_round2_remediation.py` docstrings,
`apedp_dyn/cuda/batch_physics_steppers.py::batch_multi_agent_cbf_step`
docstring (not-a-certified-CBF-QP disclosure, test-pinned).

### Register status updates (2026-10-20, evidence-linked)

| ID | Status | Evidence / remaining work |
| --- | --- | --- |
| REG-MUT | **CLOSED-WITH-EVIDENCE** | `scripts/run_mutation_battery.py` (dependency-free battery, byte-identical-restoration invariant, hard-fail exit), report `benchmarks/mutation_battery_report.json` (status ALL_KILLED, 9/9 mutants, sanity-verified pristine pins), CI gate `mutation-battery` job in `.github/workflows/test.yml` + `scripts/check.sh` full tier. **2026-10-21 extension:** the external 19-flaw remediation pins (`tests/test_dyn_external_audit_2026_10_21_pins.py`, 23 tests, each carrying an inline old-semantics mirror) are wired into the battery as MUT-D1..D4 — S3 conditional-put take-over (FLAW-03), fallback-put deadlock (FLAW-03), CBF budget-aware deficit split (FLAW-17), re-anchor residual recovery (FLAW-08); battery re-run kills all nine. **Honesty trail:** the first run was an honest FAILURE — MUT-C1 (hawkes eval-frame-leak mutant) SURVIVED because the inherited dynamic pin scored a fit/eval-identical frame, exposing a real coverage gap; the pin was SHARPENED (`tests/test_dyn_hawkes_standardization_pins.py`: fit-vs-eval distribution divergence makes the Finding-10 property load-bearing), never weakened, and the battery re-run kills all five. |
| REG-F14/F14b | **CLOSED-WITH-EVIDENCE (mode (b) wiring landed; mode (a) NOW operationally wired — blind_benchmark first census caller)** | Reference implementation (`docs/CALLER_DATA_THREE_ROLE_PROTOCOL.md`, `apedp_dyn/epistemic/caller_data_protocol.py`, `tests/test_dyn_caller_data_protocol.py`, 7 pins incl. the register's end-to-end sealed-slice-B letter) PLUS the per-caller wiring gate the register demanded: the production-caller census `apedp_dyn/epistemic/caller_registry.py` (6 callers: dyn_sweep, multi_asset_signals, real_market_l2_sweep, blind_benchmark, calibrated_benchmark_suite, canonical_ledger), each wired with a `publish_caller_compliance_record` publisher in its module and validated at publish time; selection-stage exposure hook (`record_selection_exposure`) makes the derived multiplicity LIVE. Pins `tests/test_dyn_caller_compliance_wiring.py` (12): census closure/publishers, per-caller mode-(b) validation, missing/unknown/duplicate-refusal, reuse charge flow (`mark_selection_reuse` → effective n inflates), hand-written multiplicity refusal, end-to-end wiring letter — PLUS four mode-(a) operational pins: sealed split issued over the caller's OWN real slice ids (`issue_sealed_confirmation_lead`, census/dataset-family-checked), blind confirmation slice consumed EXACTLY ONCE through the real `evaluate_confirmation_cases` scoring path (frozen-referee-checked, unknown ids refused), publication REFUSED while the slice is unconsumed or with foreign provenance, and the spent slice cannot be re-watched by a second candidate; the published sealed record (mode (a), one-shot provenance, `archived_verdict_source = "confirmation"`, selection superseded) validates under the same F14b validator. **Honest boundary:** the five remaining census callers stay published in mode (b) by default and the blind suite's mode-(a) record is demonstrated at wiring level through the pinned registration/consumption/archive/publish path — a LIVE production promotion claim through the sealed slice is still pending and is disclosed as such, not silently claimed. External epistemic sign-off (register source) remains the outstanding step. **Restart durability (2026-10-20):** sealed-slice consumption is now durable across process restarts — issued leads and consumption provenance persist to an atomic state file (`caller_lead_protocol_state.json`, path explicit or derived from the ledger's persistence directory), the slice is claimed spent BEFORE evaluation so a mid-evaluation crash cannot resurrect it, re-issue/re-consume after restart is refused from the durable record, and a corrupt store fails CLOSED (`CallerProtocolStateError`, no silent memory-only fallback). Ledger exposure writes are FAIL-CLOSED (the prior silent except-and-pass is removed: write failures propagate; an unregistered candidate is refused before the sealed slice is spent). The mode-(b) `dsr_shrink` disclosure imports the preregistered `DSR_THRESHOLD` from the hash-verified canonical spec (spec JSON unmodified — it is quorum-signed; a value change requires a re-signed revision). The one-shot guarantee remains in-memory-only ONLY where no durable state path resolves (in-memory unit pins), disclosed as a unit-use limitation. **Pre-existing, retroactively disclosed (outside this register's mandated suite):** `tests/test_dyn_governance.py::test_canonical_repository_governance_manifest_valid` fails at the committed tip independent of this work — `specifications/governance_manifest.json` pins spec SHA `a6d026…` while the spec on disk hashes `a11fd957…` (matching `canonical_spec.py`'s import-time pin); resolution requires renewed trustee certification, not a code edit, and is disclosed rather than claimed. |
| REG-F16 | **CLOSED-WITH-EVIDENCE (formal certificate artifact; external safety sign-off still outstanding)** | `apedp_dyn/robotics/cbf_certificate.py` + `tests/test_dyn_cbf_certificate.py` (8 pins) + artifact `benchmarks/cbf_certificate_report.json` (`benchmarks/run_cbf_certificate_battery.py`, exit 0). FORMAL scope (machine-checked under the explicit assumption object, which refuses γ·dt > 0.5 etc.): pairwise barrier h = dist − r_safe; certificate-by-construction for every released command (every release re-checked fail-closed), closed-form factory feasibility lemma (identical velocities ⇒ the QP set is never empty ⇒ alternating projections converge), projection-optimality containment vs the feasible zero command, discrete forward invariance h_{k+1} ≥ (1−γdt)h_k > 0 over certified Euler horizons (4 envs × 120 steps pinned), 64 certified releases verified with worst slack −3.3e-16. SCALED-VERIFICATION layer (same formal checks re-verified at larger grids under the same assumption object with `max_agents=128`, in-scope): 16/32/64/128-agent deterministic jittered-lattice grids (A3 margin assertion-checked) — 1440 additional releases fail-closed-verified (worst slack −9.91e-10 ≥ −tol at 128 agents), Euler forward invariance h > 0 at every step on every scale, and the REALIZED guard (certify → `unicycle_commands_from_certified` → PRODUCTION `batch_unicycle_arc_step`) collision-free at every scale under the physically-consistent realized-velocity-feedback nominal discipline (128-agent realized min barrier 0.3919). Guard latencies recorded as WALL-CLOCK measurements (measurement of the environment that ran the battery, explicitly NOT guarantees): mean guard 0.32/0.48/0.99/2.97 ms and mean guard+stepper step 0.60/0.77/1.31/3.39 ms at 16/32/64/128 agents. DISCLOSED SCOPE PROBE (recorded verbatim, not exit-affecting): fresh full-range random nominals redrawn every step at 128 agents are OUTSIDE the realized layer's evidence (deterministic breach: min barrier −0.1102, 17 collision steps) — the realized layer remains model-checked EVIDENCE scenario-scoped (deadlock / realized-feedback disciplines), never a formal claim. MODEL-CHECKED baseline kept: adversarial symmetric head-on sweeps — min barrier 0.3588, 0 collisions. `batch_multi_agent_cbf_step` is UNTOUCHED and its not-a-certified-CBF disclosure stays pinned; `certified_cbf_guard` is additive. External safety sign-off (register source) remains the outstanding review step. |
| REG-LEAN | **CLOSED-WITH-EVIDENCE (pinned-toolchain reproduction; physically-external provision still outstanding)** | `scripts/run_lean_pinned_reproduction.py` + artifact `benchmarks/lean_pinned_toolchain_reproduction.json`: ALL 323/323 records reproduced (proof_claim_status + kernel_exit_code per candidate) under the independently provisioned pinned toolchain `leanprover--lean4---v4.34.1` (commit `5045d0056413266e57c625dcd7c365b10e377c52`) vs the reference run's v4.35.0 (`eb6cafb7…`); counts match (323/315/8/0). Byte-identical restoration of the reference artifact AND the committed `.lean` evidence verified (the script restores HEAD-committed bytes — a bare regeneration silently corrupts the remediated u15 honest disclosure back into a vacuous theorem, caught and fixed). **Root-cause fix landed:** the reference sweep itself was re-produceable only after raising the bridge's 15/25 s compile subprocess timeouts to 240 s — under 8-way contention the two heaviest fluid-cell files were being wall-clock-killed to P1 (exit −1) while compiling clean standalone; disclosed in `CHANGELOG.md`. The register's "outside this environment" EXTERNAL provisioning/sign-off remains the outstanding step and is not claimed here. |

---

## Round-3 external assessment record (2026-10-19)

Post-round-3 external review scored update: **6.5/10 overall** (up from ~6/10).
Reported scorecard retains the distinction between verified remediation state and
scientific-capability maturity:

| Dimension | Score |
| --- | --- |
| Engineering and audit discipline | 8.5/10 |
| Epistemic safeguards | 8/10 |
| Robustness and regression evidence | 8/10 |
| Structural discovery capability | 5.5/10 |
| Scientific validation and generalization | 4.5/10 |
| Demonstrated advantage over alternatives | 3/10 |

Reviewer-stated criteria for reaching **8/10** (recorded here as forward
requirements, not claimed progress):
1. Completion of the outstanding requirements above (REG-F14/F14b, REG-F16,
   REG-MUT, REG-LEAN) with per-item acceptance evidence.
2. Independent review of the mathematical certificates and critical code paths.
3. A reproducible external evaluation showing APEDP-X reliably distinguishes
   valid discoveries from false or non-identifiable hypotheses.

For **9/10 or higher**, additionally: measurable advantage over established
alternatives across multiple independent problem families, not just successful
individual demonstrations.

Matching the reviewer's framing, this register treats the requirement-closure
path as: individual acceptance evidence per requirement first, then
independent external validation of scientific performance. Already-green
checks (46-pin suite, Rust lib 186/186, branch alignment, frozen checkpoint
`47594e7` vs follow-through record) are not reopened without a relevant
change or new failure signal.

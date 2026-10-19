# APEDP System Flaws Audit

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

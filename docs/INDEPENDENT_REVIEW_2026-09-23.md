# APEDP Independent Review — Failure-Mode Findings

> **Date**: 2026-09-23
> **Basis**: source code + command execution only. **No project documentation was used as an evaluation source.**
> **Scope**: `apedp/` (108 source files, 35,772 lines), `tests/` (46 files), CI configuration, packaging.
> **Method**: read-only inspection plus `pytest`, `ruff check --statistics`, `mypy`, and targeted
> `grep`/`sed` over the source tree.

## 0. Caveat on measurement state

The working tree is **dirty**: 123 modified-or-untracked entries at the time of
measurement (139 at audit start). Every number below describes **the working tree as
it stands**, not `HEAD`. Where a finding may be an artifact of uncommitted work, that
is stated explicitly. Nothing in this document was verified against a clean checkout.

This matters most for the static-analysis counts (§2): if a fresh `git checkout` of
`HEAD` is clean, those two rows may not represent what the repository actually
contains. Everything in §3–§6 was confirmed by direct reading of source, and those
findings stand regardless of the dirty-tree question.

> **Update, same day:** a remediation pass followed this review. The inline
> `Status:` lines in §2–§6 record the state **at review time** and are carried for
> the record; **§10 supersedes them** with what was actually changed and what
> remains open.

---

## 1. Baseline measurements

| Signal | Command | Result |
|---|---|---|
| Test suite | `.venv/bin/python -m pytest tests -q` | **474 collected — 451 passed, 19 failed, 4 skipped** |
| Failure cause | inspect failing tracebacks | **All 19 are `ModuleNotFoundError`** for optional extras (`jax`, `sklearn`, `scipy`). Zero logic failures. |
| Lint | `.venv/bin/python -m ruff check apedp tests` | **417 errors** (63 auto-fixable) |
| Types | `.venv/bin/python -m mypy apedp --ignore-missing-imports` | **266 errors in 23 files** |

The 451/455 runnable tests passing is a genuine strength: the failures carry no
information about correctness, but they do carry a finding of their own (§6.3).

---

## 2. Integrity finding: the documented green gates do not reproduce

**Severity**: HIGH
**Location**: `docs/SYSTEM_FLAWS_AUDIT.md` (claims), `.github/workflows/test.yml` (enforcement)

The project's own second-pass audit (dated 2026-09-18) asserts two green gates:

> `ruff check .` = All checks passed across the whole repository.
> `mypy apedp/ --ignore-missing-imports` = **0 errors in 35 files**

Neither reproduces:

1. `ruff check apedp tests` → **417 errors**; `mypy apedp --ignore-missing-imports` → **266 errors in 23 files**.
   (Subject to the §0 caveat.) The codebase has grown from the 35 files the audit
   measured to 108; the audit's snapshot was not maintained as code was added.

2. **More importantly, the gates are configured to never run.** `.github/workflows/test.yml`
   triggers on:

   ```yaml
   on:
     push:
       branches: [main, develop]
     pull_request:
       branches: [main]
   ```

   The repository's only branch is **`master`** (`git branch -a` → `* master`).
   Pushing to `master` triggers nothing. The `pytest`, `ruff`, `mypy`, and `bandit`
   steps in that workflow are, in practice, **inert text**: the described quality
   contract is documentation rather than enforcement.

**Why it matters**: an audit document that states a passing gate, plus a CI file that
appears to guard that gate, together create a stronger impression of verification than
exists. This is the same class of defect as §3.2 — a claim of rigor not backed by a
mechanism.

**Remediation**: point the workflow at `master` (or rename the default branch); run
`ruff`/`mypy` locally as part of the change loop; re-baseline or annotate the stale
numbers in `docs/SYSTEM_FLAWS_AUDIT.md` rather than leaving them as a standing claim.

**Status**: open.

---

## 3. Critical findings

### 3.1 Silent swallow of falsification failure on the epistemic API endpoint

**Severity**: CRITICAL
**Location**: `apedp/api/routes/epistemic.py:134-146` (the `except Exception: pass` is at lines 145-146)

```python
try:
    safe_parse_expr(req.formula_str)  # validation parse; raises on bad input
    repair_result = repair_system.autonomous_stress_test_and_repair(...)
    falsification_report = {"formula": req.formula_str, "repair_result": repair_result}
except Exception:
    pass

return tournament.run_tournament(..., falsification_report=falsification_report)
```

The entire stress-test/repair path — the thing that is supposed to *falsify* the
hypothesis — is wrapped in a bare `except Exception: pass`. On any failure
`falsification_report` remains `None`, and the tournament runs anyway.

**Why it matters**: this is the platform's own stated rule inverted. `falsification.py`
refuses to report `SURVIVED_PROVISIONAL` when a probe crashes, on the explicit
principle that *an unevaluable check is not a passed check*. Here, an unevaluable check
becomes a report of `None`, and a consuming client cannot distinguish "falsification
found nothing" from "falsification crashed". Because this is the HTTP surface, the
distinction is lost precisely where results are most likely to be consumed by another
system.

**Remediation**: capture the exception and surface it — e.g. set
`falsification_report = {"status": "UNAVAILABLE", "error": str(e)}` — and have the
tournament result disclose that the falsification evidence was missing. Never leave the
field silently `None`.

**Status**: open.

### 3.2 Heuristic presented as proof — `ContradictoryConstraintEngine`

**Severity**: CRITICAL (original) — *drafted replacement in working tree, uncommitted*
**Location**: `apedp/falsification.py:239`

**Original implementation.** The engine normalized constraint strings with
`.lower().replace(" ", "")` and matched approximately five hardcoded substrings
(`"f(0)=0"`, `"lim_0=0"`, `"contact_zero"`, `"perpetual"`, `"f(x)>=1/x"`,
`"de/dt=0"`, …). On a match it returned:

```python
InfeasibilityCertificate(
    is_infeasible=True,
    conflicting_constraints=[...],
    incompatibility_proof="CONTRADICTORY_CONSTRAINTS_PROOF: ...",
    mathematical_argument="lim_{x->0+} f(x) >= lim_{x->0+} k/x^2 = +oo != 0",
)
```

A substring hit yielded a certificate whose text asserted a formal derivation the code
never performed. The argument string was a literal, not a computation. This is a lookup
table wearing a theorem prover's clothes, and it violates the evidence-tier discipline
the rest of the codebase implements carefully (`evidence.py`: refutations take the
*strongest* evidence, clearances the *weakest*, clearances capped at EMPIRICAL).

**Replacement (present in the working tree, uncommitted).** The class now parses each
constraint's mathematical content — text after the `label:` — into typed claims and
attempts pairwise symbolic derivations:

| Claim type | Example form | Class |
|---|---|---|
| Pin | `f(0) = 0` | `_BoundClaim` / pin |
| One-sided limit | `lim_{x->a+} f(x) = v` | `_LimitClaim` |
| Envelope bound | `f(x) >= k/x**2` | `_BoundClaim` |
| Scalar bound | `v <= c`, `v >= alpha*c` | `_ScalarClaim` |
| Signed rate | `dE/dt >= P_0 > 0` | `_RateClaim` |
| Closed form | `f(x) = 1/x` | `_FormClaim` |
| Real-valued | `Im(f) = 0` / `real_output` | — |
| Finite integral | `int_1_inf_f_finite` | — |

Derived contradictions are computed at decision time with exact SymPy
(`sp.limit`, `sp.integrate`, `reduce_inequalities`) under seven rules: envelope-vs-limit,
envelope-vs-envelope, rate-vs-rate, form-vs-reality, form-vs-integrability, plus
scalar-range emptiness. When a derivation needs the sign of an undeclared parameter
(e.g. `k` in `f(x) >= k/x**2`), the parameter is taken strictly positive **and the
assumption is disclosed in the certificate text** as a premise. Unparseable constraints
are skipped (refusal to interpret), never treated as satisfied.

*Verification so far*: a 13-case smoke battery (7 derivations that must fire, 6 negative
controls that must not) passes, and the four affected test files pass (**77 tests**).
*Outstanding*: full-suite run, `ruff`, `mypy`, and commit. **The replacement is not yet
proven end-to-end and should be treated as a draft until that completes.**

**Status**: drafted, uncommitted, not fully verified.

---

## 4. High findings

### 4.1 Dead and circular calibration harness that prints authoritative metrics

**Severity**: HIGH
**Location**: `apedp/physics_validity_engine.py` — `run_benchmark()`,
`calibrate_thresholds_from_benchmarks()`, `BENCHMARK_DATASET`

A grep across `apedp/` and `tests/` finds **no caller** for any of the three. They are
referenced only by their own definitions.

If executed anyway (e.g. `python -m apedp.physics_validity_engine`), it:

- computes `roc_curve` / `auc` over `BENCHMARK_DATASET` and **prints**:
  ```
  Benchmark Results:
    ROC AUC: 0.xxx
    Optimal threshold (Youden's J): 0.xxx
    HIGH threshold: ...   LOW threshold: ...
  ```
- derives its thresholds by **percentile over the same scored data whose labels define
  `y_true`** — no train/test split, so the calibration validates itself;
- never asserts any `S_physics_expected` value from the dataset (there are 14 of them,
  all dead metadata);
- computes `low_threshold = np.percentile([s for s in y_scores if s < 0.5], 25)`,
  which is `nan` when no score is below 0.5.

The module does deserve credit for a prior fix: the comment at the scoring loop records
that the old implementation copied `known_law_match` / `derivation_valid` from the
benchmark's own category label into the audit inputs, making the ROC curve measure
**label injection** rather than discrimination — and that this was removed
(`"label_leakage_removed": True`). The remaining problem is that the harness is now
unreferenced, unrunnable as a gate, and would still emit authoritative-looking numbers.

**Remediation**: either wire it into the test suite with a fixed, held-out split and
assertions against `S_physics_expected`, or delete it. Unreferenced validation code that
prints metrics is worse than absent code: it reads as evidence.

**Status**: open.

### 4.2 Blind and silent exception handling is systemic

**Severity**: HIGH
**Location**: throughout `apedp/`

Independent measurement by ruff:

| Rule | Meaning | Count |
|---|---|---|
| `BLE001` | blind `except Exception` | **83** |
| `S110` | `try`/`except`/`pass` | **14** |
| `S112` | `try`/`except`/`continue` | **4** |

An independent `grep` for `except Exception` and bare `except` in `apedp/` found **108
occurrences**, **19** of which are an immediate `pass`.

Confirmed silent-swallow sites (each discards an error with no record):

| Location | What is silently dropped |
|---|---|
| `qd_generation.py:167` | Coulomb-sign notes → `coulomb_sign_notes` returns empty |
| `hybrid.py:52` | bake-off MAE → candidate silently scored as `inf` |
| `physics_validity_engine.py:62` | identity-detection simplification failure |
| `physics_validity_engine.py:120` | empirical-evidence parsing → silently defaults to `0.5` |
| `equations/normalize.py:104` | LaTeX parse failure |
| `equations/classify.py:20`, `:32` | classification failures |
| `equations/canonicalize.py:42` | canonicalization failure |
| `identity_detection.py:58` | identity simplification failure |
| `discovery.py:1013`, `:2429`, `:2667` | three separate discovery-path failures |
| `api/routes/epistemic.py:145` | **the falsification report (see §3.1)** |

**Why it matters**: in a system whose output is a *claim about physical validity*,
an unnoticed failure is not neutral — it becomes a silent downgrade to a default value
(`0.5`, `inf`, empty list), which then flows into a scored verdict. The default is
indistinguishable from a measured result.

**Remediation**: allow-list the exceptions that are genuinely expected; log or record
everything else; where a default is substituted, attach a marker to the result so
downstream scoring can see it.

**Status**: open.

### 4.3 Assertions used as control flow in numerical kernels

**Severity**: HIGH
**Location**: 61 `assert` statements in non-test `apedp/` code

Representative sites:

- `apedp/decay_models.py:42` — `assert best is not None`, immediately followed by
  unpacking `best` (`_, shift, exponent, amplitude, offset = best`).
- `apedp/pde_solvers.py` — nine occurrences of `assert isinstance(result, dict)` at
  lines 121, 170, 238, 328, 364, 439, 512, 582, 664.

**Why it matters**: `python -O` strips assertions. Under optimization the guard
disappears and the code proceeds to unpack `None` or call a method on the wrong type —
at a different site, with a misleading traceback. A guard that can be compiled away is
not a guard.

**Remediation**: raise explicit exceptions for genuine invariants; reserve `assert` for
developer-only sanity checks that are never load-bearing.

**Status**: open.

---

## 5. Medium findings

### 5.1 Global RNG mutation as a side effect

**Severity**: MEDIUM
**Location**: `apedp/adaptive_search.py:38-40`

```python
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
```

This mutates **both** the `random` module's global state and NumPy's global generator.
Every other module in the codebase that needs reproducibility uses a local generator —
`default_rng(seed)`, `RandomState(seed)` (`semf_validation.py:293`, `pde_solvers.py:560`,
`metrology.py:235`, `hybrid.py:154`, …). `adaptive_search` is the outlier.

Its own reproducibility is therefore fine, but any *other* component relying on global
randomness becomes order-dependent on whether an adaptive search ran first.

**Remediation**: accept a `np.random.Generator` / `random.Random` instance and thread it
through, as the rest of the package does.

**Status**: open.

### 5.2 API request-size guard defects

**Severity**: MEDIUM
**Location**: `apedp/api/__init__.py:145-146` (inside the `_trust_boundary` middleware)

The middleware advertises a 1 MB body cap (`MAX_REQUEST_BODY_SIZE = 1048576`). Two
defects:

```python
content_length = request.headers.get("content-length")
if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
```

1. `int(content_length)` is unguarded — a malformed `Content-Length` header raises
   `ValueError` inside middleware, producing a 500 rather than a 400.
2. Only `Content-Length` is inspected. A request using `Transfer-Encoding: chunked`
   carries no such header, so the body is never bounded despite the documented limit.

**Credit**: the surrounding trust boundary is genuinely well built and should be noted —
`hmac.compare_digest` for the key comparison (with a comment explaining the timing
attack it prevents), refusal to start without `APEDP_API_KEY`, wildcard CORS rejected
for credentialed requests, per-client sliding-window rate limiting, and an honest
docstring stating the limiter is in-process and not an edge substitute. These are the
right decisions, documented rather than discovered.

**Remediation**: wrap the parse in `try/except ValueError`; enforce the cap on the read
stream (or reject `Transfer-Encoding: chunked`) rather than trusting the header.

**Status**: open.

### 5.3 Optional-dependency fragility in the test suite

**Severity**: MEDIUM
**Location**: `tests/`

19 tests hard-fail with `ModuleNotFoundError` when `jax` / `sklearn` / `scipy` are
absent. Only **three** test files use `pytest.importorskip`
(`test_source_ingestion.py`, `test_remaining_capabilities.py`,
`test_redis_orchestration.py`) — the convention exists, it is simply not applied to the
tests that need it.

**Why it matters**: a fresh clone without optional extras shows 19 red tests that mean
nothing. A reader cannot distinguish them from real regressions without reading every
traceback, which trains people to ignore failures.

**Remediation**: `pytest.importorskip("sklearn")` / `importorskip("jax")` / `importorskip("scipy")`
at the top of each affected test, matching the three files that already do it.

**Status**: open.

### 5.4 Duplicated identity-detection logic

**Severity**: MEDIUM
**Location**: `apedp/physics_validity_engine.py` (`detect_identity`, `IDENTITY_PATTERNS`)
vs `apedp/identity_detection.py` (`detect_identity`)

Two independent implementations of the same idea, with the same pattern table
(`x_over_x`, `x_minus_x`, `x_pow_zero`) and the same simplify-then-match fallback. They
differ only in phrasing.

**Why it matters**: a correction applied to one silently does not reach the other, so
the two paths can disagree about whether the same expression is a mathematical identity
— and "is this an identity?" is a gate that decides whether a candidate is scored at all.

**Remediation**: keep one implementation and import it.

**Status**: open.

### 5.5 Static-analysis debt carried into the CI invocation

**Severity**: MEDIUM
**Location**: repository-wide

`ruff check apedp tests --statistics`:

| Rule | Count | Note |
|---|---|---|
| `F401` unused-import | 210 | (some may be intentional re-exports; no `__init__` exemption is configured) |
| `BLE001` blind-except | 83 | see §4.2 |
| `I001` unsorted-imports | 23 | |
| `RUF059` unused-unpacked-variable | 20 | |
| `S110` try-except-pass | 14 | see §4.2 |
| `RUF046` unnecessary-cast-to-int | 11 | |
| `F811` redefined-while-unused | 8 | **masked duplicate definitions** |
| `F841` unused-variable | 8 | |
| `B023` function-uses-loop-variable | 7 | **latent closure aliasing** |
| `SIM102` collapsible-if | 6 | |
| `S112` try-except-continue | 4 | see §4.2 |
| `UP035`, `RUF012`, `RUF022` | 3 each | `RUF012` = mutable class defaults |
| *(others)* | ≤2 each | `DTZ005`, `B025`, `SIM113`, `SIM222`, `RUF013`, `PIE810`, `UP031`, `TRY004`, `C408`, `PERF102`, `C401` |

`mypy apedp --ignore-missing-imports` reports **266 errors in 23 files**, including a
genuine type confusion in `apedp/api/routes/pipeline.py:22` (a `RedisPipelineJobManager`
assigned to a variable typed `PipelineJobManager`).

The three categories worth individual attention are `F811` (duplicate definitions
shadowing one another), `B023` (closures capturing a loop variable — a classic
lost-update/aliasing bug), and `RUF012` (mutable class defaults shared across
instances).

**Status**: open.

### 5.6 Duplicated parse policy

**Severity**: MEDIUM
**Location**: `apedp/safety.py` — `safe_parse_expr` vs `safe_parse_pde`

Two allowlists and two transformation sets to keep in sync by hand. `safety.py`'s
restricted-parse approach is sound (no attribute access, forbidden substrings and
identifiers, `sp.limit` timeouts run on abandoned daemon threads with a semaphore bound,
plus a comment correctly explaining why `ThreadPoolExecutor` would have silently broken
the timeout). The issue is duplication, not the design: a security allow-list maintained
in two places drifts.

**Remediation**: express both in terms of one shared policy object with an explicit
difference for PDE-only constructs.

**Status**: open.

---

## 6. Low findings

### 6.1 Monolith modules

| File | Lines |
|---|---|
| `apedp/discovery.py` | 2,850 |
| `apedp/falsification.py` | 2,322 |
| `apedp/gated_assurance.py` | 2,114 |
| *(package total)* | 35,772 |

Reviewable only in slices; ownership and testing boundaries are hard to define at this
size.

### 6.2 Working-tree tidiness (corrected)

An earlier draft of this review asserted *committed* noise. That was wrong and is
corrected here: `git ls-files` shows **0** tracked files under `.refact/`, **0** under
`outputs/`, **0** `__pycache__`, and **0** egg-info entries — the repository does not
track build or scratch artifacts. `.gitignore` covers `__pycache__/` and `*.egg-info/`.

What remains is only working-tree untidiness: 123 modified-or-untracked entries, with
`.refact/` and `apedp.egg-info/` present at the root and `.refact/` not covered by
`.gitignore`. Minor.

### 6.3 Unreferenced self-test / validation scripts inside library modules

`apedp/jax_pde_solver.py` carries a ~40-line test block under
`if __name__ == "__main__":` (including `assert float(rmse) == 0.0`), and
`apedp/semf_validation.py` carries a large validation block at line 332 with asserts
performing literature comparisons (`assert report["rmse_total_MeV"] < 5.0, ...`,
`assert len(found) >= 3, "too few magic numbers rediscovered"`). Executable
demonstrations living beside production code and outside the test runner are easy to
mistake for coverage that runs. Note also `assert resid.shape == key.shape` at
`semf_validation.py:248`, which *is* inside a library function and inherits §4.3.

---

## 7. Where the system is genuinely strong

These are recorded because an honest review of a system that demands honesty must state
what it gets right. Each was verified in source.

1. **Evidence-tier metrology (`apedp/evidence.py`).** PROVED / EMPIRICAL / HEURISTIC /
   UNRESOLVED with asymmetric combination — refutations take the strongest evidence,
   clearances the weakest, and clearances are hard-capped at EMPIRICAL on the stated
   ground that no finite set of evaluations establishes a universally quantified
   statement. Rare in research code.

2. **`unrepairable_counterexamples` reads `failure_class` only**, never the human-readable
   description, "so a verdict cannot change because someone reworded a message". A small
   decision that removes a real class of drift.

3. **`safety.py`'s timeout design** abandons daemon threads on `sp.limit` with a
   semaphore bound, and documents why `ThreadPoolExecutor` would have silently defeated
   the timeout.

4. **Numeric care in `core.py`**: int64 exponent vectors with the silent-wraparound
   failure mode documented as the reason int8 was rejected; exact-rational nullspace;
   a strict-units mode that records bare-number assumptions instead of guessing.

5. **The test suite has teeth**: a coverage test that fails if any falsification probe
   lacks a declared evidence tier, and 451/455 runnable tests passing.

6. **The API trust boundary** is the best part of the API layer (see §5.2 credit note).

---

## 8. Rating

**6.5 / 10** — revised down from 7.5 after this fuller audit.

| Dimension | Weight | Score | Basis |
|---|---|---|---|
| Epistemic honesty / evidence discipline | 30% | 8.0 | Excellent in `evidence.py`; contradicted by the API swallow (§3.1) and the original constraint engine (§3.2) |
| Numerical & scientific rigor | 20% | 8.0 | int64 rationale, exact rationals, strict units, real solver work |
| Security posture | 10% | 7.5 | Strong API boundary; weakened by the unbounded-chunked-body gap |
| Error handling & observability | 15% | 4.0 | 83 blind excepts, 14 silent passes, asserts-as-guards |
| Structural quality & duplication | 10% | 5.0 | Three 2,000+ line modules; duplicated identity and parse logic |
| Verification & tooling integrity | 15% | 4.5 | 417 ruff / 266 mypy; CI guards a non-existent branch; audit claims do not reproduce |

The 7.5 in the earlier pass weighted the *design* of the honesty machinery; this pass
finds the machinery is not uniformly wired up. Because the system's value proposition is
*not reporting unevaluable checks as passes*, and it reports them as passes in at least
one load-bearing place, the discount is structural rather than isolated.

---

## 9. Remediation priority

| # | Finding | Severity | Effort | Priority |
|---|---|---|---|---|
| 1 | §3.1 surface falsification failure on the API | CRITICAL | S | P0 |
| 2 | §2 point CI at `master`; re-baseline the audit claim | HIGH | S | P0 |
| 3 | §4.2 replace silent swallows with disclosed defaults | HIGH | M | P1 |
| 4 | §4.3 replace assert-guards in kernels with exceptions | HIGH | S | P1 |
| 5 | §3.2 commit + verify the symbolic constraint engine | CRITICAL | M | P1 |
| 6 | §5.3 `importorskip` guards for optional-extra tests | MEDIUM | S | P1 |
| 7 | §4.1 wire or delete the calibration harness | HIGH | M | P2 |
| 8 | §5.5 drive ruff/mypy to zero, especially `F811`/`B023`/`RUF012` | MEDIUM | L | P2 |
| 9 | §5.1 thread a local RNG through `adaptive_search` | MEDIUM | S | P2 |
| 10 | §5.2 harden the body-size guard | MEDIUM | S | P2 |
| 11 | §5.4, §5.6 de-duplicate identity and parse logic | MEDIUM | M | P3 |
| 12 | §6.1 split the monolith modules | LOW | L | P3 |

---

## Appendix — raw measurements

```
# tests
474 collected — 451 passed, 19 failed, 4 skipped
19 failures: ModuleNotFoundError (jax / sklearn / scipy)

# lint
ruff 0.16.8 → 417 errors (63 fixable)
  F401 210 · BLE001 83 · I001 23 · RUF059 20 · S110 14 · RUF046 11
  F811 8 · F841 8 · B023 7 · SIM102 6 · S112 4 · UP035 3 · RUF012 3 · RUF022 3
  + DTZ005, B025, SIM113, SIM222, RUF013, PIE810, UP031, TRY004, C408, PERF102, C401

# types
mypy → 266 errors in 23 files

# size
apedp: 108 source files, 35,772 lines
tests: 46 files
discovery.py 2850 · falsification.py 2322 · gated_assurance.py 2114

# vcs
branch: master (only branch)
CI triggers: push to [main, develop], PR to [main]  → never fires
dirty tree: 123 entries
tracked noise (.refact/, outputs/, __pycache__, egg-info): 0 each
```

*Findings §3.1, §3.2, §4.1, §5.1, §5.2, §5.4, §6.2 and §6.3 were confirmed by direct
reading of the cited source. §2, §4.2, §4.3, §5.3, §5.5 and §6.1 were confirmed by the
commands above.*

---

## 10. Remediation log (same day, 2026-09-23)

Inline `Status:` lines in §2–§6 record the state **at review time**. This section
supersedes them and records what was actually changed.

### 10.1 Verification after the pass

| Signal | Before | After |
|---|---|---|
| `pytest tests -q` | 474 collected — 451 passed, 19 failed, 4 skipped | **518 collected — 494 passed, 0 failed, 24 skipped** |
| `ruff check apedp tests` (first pass) | 417 errors | **283 errors** — `F401` 180, `BLE001` 82, plus 21 manual nits |
| `ruff check apedp tests` (after the follow-up lint pass, §10.6) | 417 errors | **180 errors** — `F401` **only** |
| `mypy apedp --ignore-missing-imports` | 266 errors in 23 files | **210 errors in 7 files** — *all* in `apedp/neural_discovery/`; **0 elsewhere** (see §10.4) |

The skip count rose 4 → 24 because the dependency-failing tests now *skip*
(`pytest.importorskip`) instead of failing — which is the point of §5.3.

### 10.2 Per-finding status

| § | Status | Change |
|---|---|---|
| §2 | **Enforcement fixed; claim annotated** | `.github/workflows/test.yml` already triggered on `master`; the stale green-gate claims in `docs/SYSTEM_FLAWS_AUDIT.md` now carry a 2026-09-23 re-baseline blockquote with the measured numbers and a pointer here. |
| §3.1 | **Fixed** | `api/routes/epistemic.py` now catches the exception and emits `{"status": "UNAVAILABLE", "error": ...}` instead of leaving the report `None`. |
| §3.2 | **Implemented, suite-verified; uncommitted** | The symbolic engine parses typed claims and derives contradictions with exact SymPy; 113 tests across `test_falsification_and_benchmarks`, `test_evidence_tiers`, `test_adversarial_science`, `test_system_flaws_audit` pass. Still not committed. |
| §4.1 | **Fixed (deleted)** | The unreferenced `BENCHMARK_DATASET` / `run_benchmark` / `calibrate_thresholds_from_benchmarks` harness was removed — it was called by nothing outside its own module, printed ROC AUC / "optimal threshold", and derived thresholds by percentile over the same scores its labels defined. |
| §4.2 | **Fixed** | Two stages. (1) The named silent-swallow sites now log and disclose their substituted defaults (`qd_generation`, `hybrid`, `equations/*`, `identity_detection`, `physics_validity_engine`, `discovery`). (2) All 82 `BLE001` sites were then given a per-site decision — narrowed to a named exception set, or an explicit `# noqa: BLE001` with a stated justification where no honest narrowing exists (3 sites: two worker boundaries, one untyped injected client). `BLE001` 82 → **0**, `S110` 14 → **0**, `S112` 3 → **0**. See §10.6. |
| §4.3 | **Partially fixed** | Load-bearing asserts in library code were replaced with real exceptions: `decay_models.py`, all 10 in `pde_solvers.py`, `eval_extended.py` (15), `delay.py` (2), `integral_equations.py`, `meshing.py`, `semf_validation.py`. Asserts inside `if __name__ == "__main__"` demo blocks (§6.3) are untouched. |
| §5.1 | **Fixed** | `adaptive_search` seeds the global `random`/`numpy` generators *and restores them* around the search (`_seeded_global_rng`), so reproducibility is kept without leaking an order dependency into other components. |
| §5.2 | **Fixed** | `api/__init__.py` parses `Content-Length` defensively (malformed → 400) and now rejects chunked/streamed bodies (411) rather than advertising a 1 MB cap it cannot enforce. |
| §5.3 | **Fixed** | `pytest.importorskip("jax"/"sklearn"/"scipy")` added to every dependency-failing test. **Fixes:** one failure was *not* environmental — `test_bounded_ratio_priors_cover_affine_relativistic_forms` was a genuine bug: the test built its target with `c` while `bounded_ratio_priors` takes the limit as a separate `limit_symbol` parameter (default `c_light`), so nothing could ever match. Fixed by passing `limit_symbol="c"`. |
| §5.4 | **Fixed** | `physics_validity_engine` now imports `IDENTITY_PATTERNS` / `detect_identity` / `is_identity_safe` from `identity_detection` instead of holding a copy-pasted second implementation. |
| §5.5 | **Fixed except `F401`** | `F811` 8 → **0**, `B023` 7 → **0**, `RUF012` 3 → **0**; auto-fixable categories cleared (66 + 39 fixes applied). The 21 manual nits are now **0** (`F841` 8 → 0, `SIM102` 6 → 0, `S112` 3 → 0, `B025`/`SIM113`/`S110`/`TRY004` 1 each → 0), and `BLE001` 82 → 0 (§10.6). Remaining: **180 `F401`** (unused imports, mostly re-exports — ruff marks removal unsafe, so each needs a human call). |
| §5.6 | **Fixed** | `safe_parse_expr` and `safe_parse_pde` now share `_validate_formula_string` and a single `_PARSER_INTERNALS`; the two allow-lists can no longer drift. |
| §6.1 | **Open** | Monoliths unchanged. |
| §6.2 | **Open** | Working tree remains dirty and uncommitted. |
| §6.3 | **Open** | `__main__` demo/validation blocks remain inside library modules. |

### 10.3 Out-of-scope change: `physics_validity_engine` (PVE) rewritten

A design question — *do we need our own PVE?* — was raised during remediation and
answered in code. Measured surface: `same_relationship` (18 external refs) and
`canonical_form` (7) are load-bearing; `check_hard_requirements` has one consumer
(`discovery_benchmark`); `compute_physics_validity*` and `conservation_checklist`
had none.

Two of its four "hard requirements" were `pass` statements, and "HARD REQUIREMENT 1:
dimensional validity" only checked that each symbol *appeared as a key* in
`declared_units` — never that the units were homogeneous — while the repo already had
a pint-backed `core.DimensionalAssuranceEngine`.

Changes:

- The dimensional requirement now delegates to `core.DimensionalAssuranceEngine` and
  distinguishes **PASS / FAIL / UNEVALUATED**. A real inconsistency still returns 0.
- Unevaluated components earn **no** credit and are left unearned in the denominator
  (`breakdown["_coverage"]` reports the evaluated fraction), replacing the previous
  free `dimensional: 1.0`. A HIGH category now requires the dimensional check to have
  actually run.
- The two no-op hard requirements and the dead wrappers were removed; `_score_empirical`
  returns `None` for "no evidence declared" rather than a 0.5 that was
  indistinguishable from a real mid-strength measurement.
- Observable effect: a known law with no declared units and no empirical evidence
  scored `0.75`/SECONDARY before and now scores `0.45`/LOW with coverage `0.45`,
  while a fully-evaluated candidate reaches `1.0`/HIGH at coverage `1.0`.

### 10.4 The mypy count: why it was not driven to zero

The original 264 was reported without a breakdown, which understated how much of
it was an environment artifact. Measured composition:

| Category | Count | Nature |
|---|---|---|
| `[valid-type]` "...Array is not valid as a type" | 108 | **Artifact** |
| Everything in `apedp/neural_discovery/` | 249 of 264 | jax / flax code |
| Outside `apedp/neural_discovery/` | 15 | Real, now fixed |

**Root cause of the 108.** `architectures.py:23` (and four sibling modules) do
`Array = jnp.ndarray`. The type check runs with `--ignore-missing-imports` and
`jax` is not installed in this environment, so `jnp` resolves to `Any`,
`Array` becomes an ordinary *value assignment* rather than a type alias, and every
annotation using it (`x: Array`, `-> Array`) is reported as `valid-type`. That then
poisons a large share of the `index` / `attr-defined` / `no-any-return` errors in
the same modules.

**What was fixed:** every error outside the neural stack — 15 across
`api/routes/pipeline.py` (the `RedisPipelineJobManager`/`PipelineJobManager`
confusion §5.5 called out), `pipeline.py`, `discovery.py`, `closed_loop.py`,
`adaptive_search.py`, `semf_validation.py`, `scientific_layers.py`, `priors.py`,
`cross_domain_benchmark.py`, `ingestion/external_science.py`,`family_proposal.py`,
`hybrid.py`, `decay_models.py`, `eval_extended.py`, `experiment_designer.py`,
`jax_pde_solver.py`. Several were genuine bug-shaped findings, not noise:
`semf_validation.py` indexed an `Optional` tuple, `scientific_layers.py` formatted a
possibly-`None` probe point, `discovery.py` reused `cols` for two incompatible
shapes in one scope, and `adaptive_search.py` did arithmetic on an untyped state dict.

**Why the remaining 210 were deliberately not shimmed.** The tempting fix is a
`TYPE_CHECKING` guard (`if TYPE_CHECKING: from jax import Array`), which would clear
the 108 immediately. It was not applied because it makes `Array` resolve to `Any`
without jax — i.e. it would make the *reported* number smaller while leaving the
neural stack exactly as unchecked as it is now, only more quietly. The genuine fix
is to run the type check in an environment that has the optional extra installed
(and/or add a `jax`-aware mypy run to CI) so those 210 are checked for real rather
than suppressed.

### 10.5 Still open, and why

- **`F401` (180)** — see §5.5. Ruff marks removal unsafe; each is either an
  intentional re-export or a genuinely dead import, and only a human can tell
  which. Not bulk-fixed on purpose.
- **`mypy` 210 errors in 7 files** — now confined entirely to the jax-gated
  neural stack; see §10.4 for why the residual count is an environment artifact
  and what the correct fix is.
- **§6.1 monoliths, §6.2 worktree, §6.3 `__main__` blocks** — untouched.
- Nothing in this pass was committed; the worktree carries all of it.

### 10.6 The narrow-exception pass (`BLE001` 82 → 0) and the manual nits

#### Method

`BLE001` cannot be silenced by logging or by logging the exception — verified
against this repo's ruff 0.16.8 with a scratch file, which narrowed the whole
problem to two honest options per site: **name the exception types**, or keep the
broad catch and **say why** with an explicit `# noqa: BLE001`.

For the first option the failure modes were read off each `try` body rather than
guessed. `numpy.linalg.LinAlgError` and `sympy.SympifyError` were confirmed to
subclass `ValueError`, and `ZeroDivisionError`/`OverflowError` to subclass
`ArithmeticError`, so neither needs a separate entry — recorded in the constant's
docstring so the next reader does not "fix" it. Where a module already had a
precedent (`gated_assurance._PROBE_ERRORS`, line 88), the same name and shape was
reused; new copies of the constant were added to `discovery`, `scientific_layers`
and `hybrid`.

#### Result

| Category | Before | After |
|---|---|---|
| `BLE001` blind except | 82 | **0** |
| `S110` try/except/pass | 1 | **0** |
| `S112` try/except/continue | 3 | **0** |
| `F841` unused variable | 8 | **0** |
| `SIM102` collapsible-if | 6 | **0** |
| `B025` duplicate except | 1 | **0** |
| `SIM113` enumerate | 1 | **0** |
| `TRY004` wrong exception type | 1 | **0** |
| `F401` unused import | 180 | 180 (unchanged, see §10.5) |
| **total** | **283** | **180** |

26 files touched. Suite unchanged: **494 passed, 24 skipped, 0 failed**;
`mypy` unchanged at 210 errors in 7 files, all jax-gated.

#### Deliberately still broad: 3 sites

Narrowing is not always the fix. The repo already used this escape hatch in four
places — `safety.py:344`, `pde_solvers.py:38`, `eval_extended.py:43`
(all `except BaseException` around a re-raise on the caller's thread) and
`composer.py:113` — so the three below follow an existing convention rather than
inventing one. They keep `except Exception` with an explicit `noqa` and a stated
reason, because they are boundaries whose failure set cannot be enumerated from
inside the module:

- `orchestration.py:182` and `:369` — **worker boundaries**. A job worker must
  persist any failure against the job and retry it; letting an unforeseen error
  escape would kill the worker instead. The `except` body already records
  `error=str(exc)` and the retry state.
- `decision/policy.py:106` — `self.client` is **injected and typed `Any`**
  (`def __init__(self, client: Any, ...)`), so its exception types are not
  knowable here. The fallback path is real (`search_source="local_fallback"`) and
  failures are counted in `self.failures`.

#### Defects found while doing it

- **A duplicated, unreachable `except` clause in `discovery.py`.** A stray section
  header had been pasted into the middle of a `try`/`except` in the residual-fitting
  solver, duplicating the whole handler (lines 2609–2625). It parsed cleanly — an
  extra `except` clause is legal Python, merely dead — which is why only `B025`
  revealed it. Both the orphan clause and the stray header were removed; the real
  header survives at its correct location before the pure helpers.
- **Dead index variable** `total_terms` in the same function: assigned, incremented,
  never read (`SIM113` was the only symptom).
- **Unreachable branches in `pde_templates.py`**: `elif not p_like: pass` where
  `p_like` is non-empty by construction, and an `if` whose condition
  (`== 1 and < 2`) can never be false-but-reached. Removed.
- **`source_ingestion.py:132`** raised `ValueError` for an `isinstance` type failure;
  now `TypeError`, which is what a caller catching a wrong-shaped manifest expects.
- **Silent `False`/`None` fallbacks in the benchmark path** (`discovery_benchmark`,
  `scientific_layers`): an unevaluable recovery/identity check was indistinguishable
  from a measured negative. Narrowing keeps the conservative direction ("not
  recovered", "not an identity") but now emits a `logger.debug` line naming the
  law and the exception, so a benchmark that under-claims can be diagnosed.

#### Behaviour changes worth knowing

Narrowing is not behaviour-neutral when something outside the named set occurred:
previously such a case was folded into a fallback value, now it propagates. The
sites where that is a *deliberate* direction change are the gate/scoring paths —
`physics_validity_engine`, `gated_assurance`'s three gates, `qd_generation` — where
a defect in the measurement machinery previously produced a plausible-looking
"measurement" (an `UNEVALUATED`/`FAIL` gate or a NaN score) and now surfaces as an
exception instead.

#### Leftover noticed, not removed

`apedp/discovery.py.bak` — a 129 KB untracked duplicate of the live `discovery`
module, dated 2026-09-21 22:33. It is inert (not importable, not collected by
pytest) but it is a stale copy of a 2,900-line module that will drift silently.
Removing it is a one-liner, left to the owner rather than done unasked.

# RIME §16e Registered Report — Multivariate Referee-Side Scope Refusal (NEGATIVE RESULT)

**Prereg:** [`docs/RIME_PREREGISTRATION_SCOPE_REFUSAL_MV.md`](RIME_PREREGISTRATION_SCOPE_REFUSAL_MV.md) (PREREG-RIME-SCOPE-MV-2026-10-17)
**Artifact:** [`benchmarks/scope_refusal_mv_validation_results.json`](../benchmarks/scope_refusal_mv_validation_results.json) (canonical run)
**Void first attempt (disclosed):** [`benchmarks/scope_refusal_mv_validation_results_void1_harnessbug.json`](../benchmarks/scope_refusal_mv_validation_results_void1_harnessbug.json)
**Run provenance:** git HEAD `363e8df` (prereg on record; implementation uncommitted at run time per the declared workflow) · 2026-10-17T17:44Z · one registered run per seed, 27 cells, wall 3.2 s

## Verdict

**§16e as preregistered FAILS its own two-sided validation on the first complete
valid run — on two independent counts — and the frozen family-set rule has no
retention assignment that satisfies both guards and coverage.** This is reported
as a negative result per the prereg: a v2 prereg is required before any
modification of the design. No design parameter, probe family, grid, seed or
adjudication default was changed at any point; the only post-run edits were
harness accounting fixes (disclosed below), with cell outcomes verified
identical.

## Harness timeline (full disclosure)

| run | status | defect | resolution |
|---|---|---|---|
| 1 | **void** | aligned-frame split used the §16c constants `DEV_ROWS=600`/`HOLDOUT_ROWS=300` on 600-row frames → empty holdout; every verdict was a no-scale artifact | JSON kept as `..._void1_harnessbug.json`; split hardcoded to the prereg's 0:400/400:600 |
| 2 | cells valid | G2 cells raised (prereg's own no-probe clause); evaluation had a vacuous `all([])` G2 pass and seed-collapsing dict keys | cells retained; evaluation fixed |
| 3–4 | canonical | evaluation fixes only | cells byte-identical to run 2 except `wall_ms` (verified by diff); determinism digests identical |

## Results (canonical run, raw declared probe set)

| world | seed 700 | seed 701 | seed 702 | decliners |
|---|---|---|---|---|
| G1 control_in_family (aligned) | **declined** | **declined** | **declined** | pq + am (abs separated) |
| W1 `4·sin(x1_l1·x2_l1) + 1.5·x3_l1 + 2` | declined | declined | declined | pq only (bootstrap fail) |
| W2 `2·max(x1_l1, x2_l1) + 1.5·x3_l1` | declined | declined | declined | pq + abs + am |
| W3 `x3_l1·sin(1.5·x1_l1) + 2` | declined | declined | declined | am + pq |
| G2 §16c monomial guards (4 worlds × 3 seeds) | **0/12 defined — `undefined_no_probe_buildable` (all seeds)** | — | — | — |
| G3 null/noise | abstained | abstained | abstained | — |

Control emitted (exact recovery, hold R² 0.9998): `2.998330·x3_l2 + 2.000092·(x1_l1·x2_l1) + 10.0019`.

## Criteria evaluation (both readings; either way FAIL)

| criterion | raw (declared set) | final-set rule applied |
|---|---|---|
| W1–W3 all declined (9/9) | ✅ | ❌ (W1/W3 lose their only decliners) |
| G1 control promoted | ❌ (declined 3/3) | ✅ (recomputed under `{abs}`) |
| G2 monomials promoted | ❌ (0/12 defined) | ❌ (0/12 defined) |
| G3 no promotion | ✅ | ✅ |
| determinism (digests) | ✅ | ✅ |

Family-set rule outcome: retained `probe_mv_abs_linear2`; excluded
`probe_mv_periodic_quadratic` and `probe_mv_am_product` (guard failures on
control, 3/3 seeds each). **No retention assignment satisfies the frozen rule:**
pq/am are required to decline W1/W3 but are prohibited by G1; abs is
guard-clean but only covers W2.

## Failure mechanisms (pinned)

1. **Small-frequency linearization breaks family-distinctness (G1 failure).**
   Both trigger families search `c` down to the refinement floor (~0.1). As
   `c→0+`, `sin(c·q)/c → q`: the probe degenerates into the emitted's own product
   regressor. On the control, pq selects `(q = x1_l1·x2_l1, c ≈ 0.1000)` with
   dev loss 0.0026 ≈ the noise floor 0.0025 and holdout RMSE 0.05005 vs the
   exact emitted's ≈0.0500 — adjudication: `multi_hypothesis`, Δ=0.00238 ≤
   τ=0.547 → declined. am selects `c ≈ 0.1` likewise (holdout RMSE 0.04769,
   Δ=1.69e−05 ≤ τ=0.547). The only guard-clean family is `abs_linear2`
   (no frequency knob; separated by Δ=1.24 > τ=0.547).
2. **Frozen variance guard × §16c axis scales makes G2 unevaluable.** The guard
   `var(q) > 1e-12` rejects `q = p²` on the §16c worlds (axis `p ∈ [1e-4, 7e-4]`,
   `var(p²) ≈ 2e-14`), and `F=1` admits no abs/am members. The prereg's own
   clause — *if NO probe can be built, adjudication is undefined and the harness
   raises* — fires on all 12 G2 cells, so "ALL G promoted" cannot be shown.
3. **No fixed point of the family-set rule** (above): the design's guard-clean
   subset cannot decline W1/W3.

## Findings

1. **Kernel-choice distinctness is not family-distinctness.** Any parameter
   whose limit collapses a rival into the emitted's basis (here: the frequency
   floor of trig families) must be excluded by an a-priori, range-aware
   admissibility rule; otherwise the rival can mimic in-family truths and
   wrongfully refuse them.
2. **Guards must be scale-relative.** An absolute variance threshold interacts
   with world axis scales (§16c axes are O(1e-4)); frame normalization or
   scale-relative guards are needed for cross-world comparability.
3. **The referee concept still showed teeth.** All 9 out-of-family emissions
   were declined (W1–W3), including via bootstrap-failure refusals (pq on W1),
   abs cleanly declined max/switch, and G3 held (no promotions on pure noise —
   the engine abstained everywhere). The two-sided criterion is exactly what
   exposed the design flaws instead of shipping them.

## Required follow-up (v2 prereg; not implemented here)

Candidate remedies for the v2 design (each untested; requires its own prereg):
a declared minimum-curvature condition tying the frequency grid to the dev range
of `q` (e.g., `|c|·range(q)` bounded away from 0), and/or scale-relative
admissibility guards; possibly combined with a pre-declared family-exclusion
rule. Until v2 is preregistered and passes, **multivariate out-of-family worlds
have no registered refusal path** (the boundary recorded by RIME-OOF-MV
finding 3 remains open).

## Caveats

Synthetic worlds only; single registered run per seed (determinism verified
in-run; cross-run cell identity verified except timing fields); refusals are
`adjudicate()` outcomes under §16d semantics (any probe non-promotion = decline)
— the run does not re-litigate that semantics; §16d module and proposers
untouched; the void first run is disclosed above and retained as an artifact.

## Artifacts

- Protocol: `docs/RIME_PREREGISTRATION_SCOPE_REFUSAL_MV.md`
- Module: `apedp_dyn/epistemic/scope_refusal_mv.py` (probe construction only)
- Runner: `benchmarks/run_scope_refusal_mv_validation.py`
- Canonical results JSON: `benchmarks/scope_refusal_mv_validation_results.json`
- Void run JSON: `benchmarks/scope_refusal_mv_validation_results_void1_harnessbug.json`
- Fast regression pins: `tests/test_dyn_rime_scope_refusal_mv.py`
- Related: `docs/RIME_OOF_MV_REPORT.md`, `benchmarks/known_truth_challenge.py` (§16c),
  `tests/test_dyn_rime_known_truth_scope_repeats.py` (§16d)

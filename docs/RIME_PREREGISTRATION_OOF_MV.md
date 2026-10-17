# Preregistration: RIME-OOF-MV — Multivariate Out-of-Family Surrogate Probe

**Document ID:** PREREG-RIME-OOF-MV-2026-10-17
**Code:** `benchmarks/run_rime_oof_multivariate_probe.py` (committed separately after)
**Standing:** exploratory sandbox probe (seeds 500/501/503, /tmp only, disclosed below)
  found that `RimeProposer` emits confident surrogates on out-of-family multivariate
  worlds instead of abstaining. This document freezes the protocol for a registered
  replication on FRESH seeds before any further comparative testing.
**Frozen (no search):** worlds, seeds, engine constructors, slices, metrics, decision
  rules and falsification criteria below are declared before the registered run and
  may not be tuned afterwards. One registered run; no re-runs to cherry-pick.

## Relation to existing registered work

- §16c known-truth challenge (univariate static): planted monomials + planted
  exponential world; both proposer and baseline emitted spurious fits.
- §16d referee-side scope refusal: preregistered exp/log rival probes on ONE
  positive axis (`feature_col="p"`), promotion only on UNIQUE_SUPPORT over every
  probe. Pinned for the univariate exponential world.
- **Boundary recorded by this prereg:** §16d probes are univariate by
  specification and cannot adjudicate multivariate frames (no `p` axis, sign
  changes, lagged multi-input structure). The referee module is NOT modified or
  re-tuned here; the absence of a registered refusal path for multivariate
  out-of-family worlds is itself a measured finding, candidate for a future
  §16e-style preregistration (deviating §16d requires re-preregistration, not
  patching).
- Test #2 of the external review (head-to-head vs PySR) already exists as
  `benchmarks/matched_compute.py` (preregistered, equal wall allowance; measured
  outcome: complementarity, not dominance). No new comparable is added here.

## Exploratory disclosure (already observed, not part of the registered run)

Sandbox probe (not a repo artifact) on seeds 500/501/503: control recovered
exactly by `RimeProposer` (test R² = 0.9998, 11 ms); all three out-of-family laws
emitted surrogates with `UNIQUE_WINNER` and no abstention; two surrogates
collapsed on the wide-range slice (0.81→0.40, 0.65→0.07); `RimeAdProposer`
produced no signal on multivariate frames (~54–71 s per fit) while succeeding on
its own univariate Lorentzian case (7.5 s, `implicit_svd`).

## Hypotheses under test

- **H1 (grammar-boundness):** on out-of-family multivariate worlds,
  `RimeProposer` emits candidates (no empty list) with best in-distribution
  test R² in [0.5, 0.95) — confident surrogates, not abstention.
- **H2 (OOD fragility):** for at least the sin-of-interaction and
  amplitude-modulation laws, the wide-range (2.5×) R² of the best-by-test
  candidate drops at least 0.15 below its test R².
- **H3 (RimeAd scope):** on multivariate frames `RimeAdProposer` best test R²
  ≤ 0.05 on every law including the in-family control (its working scope is
  univariate physics curves, pinned elsewhere).
- **Falsification:** any out-of-family law classified RECOVERED falsifies H1;
  any out-of-family law with an empty proposal list falsifies H1's
  no-abstention claim; a wide-range drop < 0.15 on both collapse-prone laws
  falsifies H2; best test R² > 0.05 on any multivariate law falsifies H3.

## Registered worlds (fresh seeds)

All worlds: n=600, x1,x2,x3 ~ N(0,1) (wide slice scaled 2.5×), y noise 0.05·ε,
lags per formulas; rows 0,1 of target are 0 by construction; metrics use rows ≥ 2.

| id | law |
|---|---|
| control_in_family | y = 2·x1[t-1]·x2[t-1] + 3·x3[t-2] + 10 |
| oof_fn_of_interaction | y = 4·sin(x1[t-1]·x2[t-1]) + 1.5·x3[t-1] + 2 |
| oof_max_switch | y = 2·max(x1[t-1], x2[t-1]) + 1.5·x3[t-1] |
| oof_amplitude_mod | y = x3[t-1]·sin(1.5·x1[t-1]) + 2 |

Slices: train = seed 600 rows 0:400; test = seed 600 rows 400:600;
fresh = seed 601 full; wide = seed 603 full, inputs scaled 2.5×.

## Frozen engines and configs

- `RimeProposer(target_col="target", budget=200, max_depth=2, additive_terms=2,
  max_seeds=120, lagged_products=True, gamma=1.0)`.
- `RimeAdProposer(target_col="target")` (defaults).
- No other engines, no config sweeps. Wall time recorded per fit.

## Metrics and classification (frozen)

R² computed on rows ≥ 2 after finite mask; best candidate = max test R² over all
proposals (list order is not trusted). Per (law, engine): wall ms, n_props,
best test / fresh / wide R², formula, inputs_used, identifiability_status.
Classification per (law, engine): ABSTAINED (empty list) / RECOVERED
(test ≥ 0.95 AND wide ≥ 0.85) / CONFIDENT_SURROGATE (0.5 ≤ test < 0.95) /
NO_SIGNAL otherwise. Harness sanity: control_in_family under `RimeProposer` must
classify RECOVERED or the registered run is aborted as invalid.

## Artifacts

- `benchmarks/rime_oof_multivariate_results.json` (includes git HEAD provenance).
- `docs/RIME_OOF_MV_REPORT.md` (results vs hypotheses).
- `tests/test_dyn_rime_oof_multivariate.py` (fast pins; tolerances set from the
  registered run, RimeProposer-only so the suite stays fast; `RimeAdProposer`
  multivariate slowness is recorded in the benchmark, not re-run in tests).

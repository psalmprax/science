# RIME-OOF-MV Registered Report — Multivariate Out-of-Family Surrogate Probe

**Prereg:** [`docs/RIME_PREREGISTRATION_OOF_MV.md`](RIME_PREREGISTRATION_OOF_MV.md) (PREREG-RIME-OOF-MV-2026-10-17)
**Artifact:** [`benchmarks/rime_oof_multivariate_results.json`](../benchmarks/rime_oof_multivariate_results.json)
**Run provenance:** git HEAD `3eabf77` · 2026-10-17T17:21Z · one registered run, fresh seeds 600/601/603 (exploratory seeds 500/501/503 disclosed in the prereg)
**Harness sanity:** control RECOVERED under `RimeProposer` → run valid.

## Results (best-by-test candidate per cell)

| law | engine | wall | n_props | best test R² | fresh | wide (2.5×) | class |
|---|---|---:|---:|---:|---:|---:|---|
| control_in_family | RimeProposer | 8 ms | 5 | **0.9998** | 0.9998 | 1.0000 | RECOVERED |
| control_in_family | RimeAdProposer | 77 s | 10 | −0.0003 | −0.006 | −0.009 | NO_SIGNAL |
| oof_fn_of_interaction | RimeProposer | 6 ms | 5 | 0.9433 | 0.866 | **0.459** | CONFIDENT_SURROGATE |
| oof_fn_of_interaction | RimeAdProposer | 64 s | 10 | −0.0001 | −0.010 | −0.002 | NO_SIGNAL |
| oof_max_switch | RimeProposer | 20 ms | 5 | 0.6220 | 0.693 | 0.494 | CONFIDENT_SURROGATE |
| oof_max_switch | RimeAdProposer | 56 s | 10 | −0.0066 | −0.000 | −0.119 | NO_SIGNAL |
| oof_amplitude_mod | RimeProposer | 6 ms | 5 | 0.7130 | 0.674 | **0.057** | CONFIDENT_SURROGATE |
| oof_amplitude_mod | RimeAdProposer | 57 s | 10 | −0.0102 | −0.018 | 0.001 | NO_SIGNAL |

Recovered formulas (RimeProposer, control): `3.004·x3[t-2] + 1.994·(x1[t-1]·x2[t-1]) + 9.999`.
Surrogates found: `3.60·tanh(x1[t-1]·x2[t-1]) + 1.49·x3[t-1] + 1.93` (sin-of-interaction);
`1.50·x3[t-1] + 1.08·x2[t-1] + 1.05` (max — drops x1); `1.06·tanh(x1[t-1]·x3[t-1]) + 1.97` (amplitude modulation).

## Hypothesis verdicts (frozen in the prereg)

- **H1 grammar-boundness — CONFIRMED.** All three out-of-family laws produced
  non-empty proposal lists (5 each, no abstention) with best test R² in
  [0.62, 0.94): confident surrogates, not refusal.
- **H2 OOD fragility — CONFIRMED.** sin-of-interaction: 0.9433 → 0.4594
  (drop 0.48); amplitude-modulation: 0.7130 → 0.0568 (drop 0.66). Both exceed
  the preregistered 0.15 threshold.
- **H3 RimeAd multivariate scope — CONFIRMED.** Best test R² ≤ −0.0001 on
  every law including the in-family control; 56–77 s per fit against
  `RimeProposer`'s 6–20 ms on the same frames.

## Findings

1. **No proposer-layer abstention on out-of-family worlds.** `RimeProposer`
   emits near-perfect in-distribution fits whose mechanism is wrong, exactly the
   §16c phenomenon, now registered for multivariate/lagged worlds.
2. **Identifiability status is not a refusal signal.** Surrogates were tagged
   `UNIQUE_WINNER` (2 laws) and `SUB_OPTIMAL` (1 law; `UNIQUE_WINNER` in the
   exploratory draw) — the flag measures uniqueness among *its own* candidates
   and is seed-unstable across draws. It must not be read as confidence in the
   mechanism.
3. **§16d scope refusal does not cover these worlds.** Its probes are exp/log
   families on one positive axis (`feature_col="p"`); multivariate lagged frames
   carry no such axis. Per spec §16d semantics, deviating the probe families
   requires re-preregistration. Recorded boundary: **multivariate out-of-family
   worlds currently have no registered refusal path** — candidate for a future
   §16e-style preregistration; the referee module was not modified here.
4. **RimeAdProposer's multivariate path carries no signal** (H3) while its
   univariate scope works (pinned for the Lorentzian case elsewhere). Runtime
   makes it unusable for this regime as-is.

## Caveats

Proposer-layer measurement only (the referee was inapplicable by specification,
finding 3); synthetic worlds; single registered run per cell; no tuning and no
re-runs per protocol. External replication remains outstanding.

## Artifacts

- Protocol: `docs/RIME_PREREGISTRATION_OOF_MV.md`
- Runner: `benchmarks/run_rime_oof_multivariate_probe.py`
- Results JSON: `benchmarks/rime_oof_multivariate_results.json`
- Fast regression pins: `tests/test_dyn_rime_oof_multivariate.py`
- Related registered work: `benchmarks/matched_compute.py` (PySR head-to-head,
  complementarity outcome), `docs/RIME_V01_MASTER_SPEC.md` §16c/§16d,
  `tests/test_dyn_rime_known_truth_scope_repeats.py`

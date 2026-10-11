# Preregistration: CPMD v0.12 — Mean-Structure and Estimand Identification

**Document ID:** PREREG-CPMD-v012-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_estimand.py` (this prereg frozen BEFORE run)
**Status:** v0.11 showed bias ≈ −0.05 ≈ 0.5 SE in ALL methods incl. ρ=0 control.
**Sealed set:** unopened. Real-market promotion: NOT authorized.
**BBOOT branch:** stays closed (documented implementation failure, not method evidence).

## The estimand question

For y(τ) = A(τ+τ₀)^(−α), log-log OLS over lags 4–12 measures a FINITE-WINDOW
regression slope, not necessarily the pointwise α_eff(τ*) = ατ*/(τ*+τ₀).
v0.11 evaluated OLS against the pointwise truth (0.6 at τ*=8) — possibly a
different estimand than OLS measures. v0.12 settles which target is which.

## Tests (frozen)

- **A — exact targets:** compute pointwise truth AND finite-window truth
  (noise-free projection: OLS slope of the exact curve on the interval).
- **B — bias diagnosis:** bias of OLS vs EACH target on known-truth sims
  (power a=1.2, t0=8, nl ∈ {0.1, 0.3}, ρ=0, seeds 6000+, n=200).
  Prediction under test: bias ≈ 0 vs finite-window target.
- **C — mean-structure alternatives:** local 3-point slope estimator + correctly
  specified nonlinear fit (a, τ₀ free — diagnostic control only, NOT a promotion
  candidate) vs pointwise target; OLS vs finite-window target.
- **D — uncertainty:** coverage/width for whichever (estimator, target) pair
  survives B–C, under existing caps (coverage ≥ 92%, width ≤ 1.0).

## Gates

1. Estimand correctness: both truths computed from known parameters, committed.
2. Point recovery: |bias| ≤ 0.1, RMSE ≤ 0.3 vs the MATCHED target.
3. Interval calibration + width cap as in v0.10/v0.11.
4. Usable rate ≥ 80%. Independent seeds throughout (6000+).

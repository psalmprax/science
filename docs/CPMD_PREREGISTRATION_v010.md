# Preregistration: CPMD v0.10 — finite-lag effective slope

**Document ID:** PREREG-CPMD-v010-2026-10-11
**Code:** to be added as `apedp_dyn/sandbox/cpmd_effective_slope.py` (not yet written/run)
**Supersedes target of:** v0.9.x global-exponent program (frozen as unresolved)

## Estimand change (explicit, not silent)

v0.9.x targeted global (α, τ₀). Track B + ridge diagnostics (commit pending)
show the likelihood surface does not separate them: corr(α,τ₀)=+0.983,
SSR prefers (1.2, 30) over truth (1.2, 8.0), fixed-τ₀ rmse 0.435 vs 105.8 free.

v0.10 targets the local effective slope over a FIXED lag interval [L0, L1]:

  α_eff = −d log y / d log τ, estimated by log-log linear regression on the
  prespecified interval. No global-exponent or mechanism claim.

## Design (to be implemented)

- Interval [L0, L1] chosen on scientific grounds BEFORE evaluation and frozen here
  once set; candidate: lags 4–12 (avoids short-lag transients, keeps n=9).
- Truths: power (several α, τ₀ incl. non-identifiable pairs), expo, flat, regime.
- Metrics per condition: bias, RMSE, coverage + median width, abstention rate.
- Noise: multiplicative, 0.1/0.3; independent + AR(1) ρ=0.5.
- Diagnose on fixed seeds; evaluate on the v0.9.5 sealed set ONLY after
  diagnostic gates pass. Sealed set stays unopened until then.

## Gates (proposed)

1. |bias| ≤ 0.1 and RMSE ≤ 0.3 per condition.
2. Coverage within [93%, 97%] of nominal 95% AND median width ≤ 1.0
   (coverage without a width cap is vacuous — cf. Track B widths 482/599).
3. FP-equivalent (slope declared nonzero on flat truth) ≤ 5%.
4. All failures in denominators; no threshold changes after unsealing.

## Verdicts carried forward

- Global exponent recovery: failed under tested conditions.
- Ridge: confirmed dominant mechanism (committed diagnostics).
- CPMD market law: unresolved. Real-market promotion: NOT authorized.

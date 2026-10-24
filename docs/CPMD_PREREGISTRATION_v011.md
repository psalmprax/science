# Preregistration: CPMD v0.11 — OLS Uncertainty Calibration Under Correlated Noise

**Document ID:** PREREG-CPMD-v011-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_ols_uncertainty.py` (this prereg frozen BEFORE run)
**Status:** v0.10b retired model-based estimator; OLS is reference (NOT validated).
**Sealed set:** unopened. Real-market promotion: NOT authorized.

## Point estimator (FIXED — not under test)

Log-log OLS slope over frozen lags 4–12 (n=9), sign-flipped. Only the
UNCERTAINTY method varies. Retired model-based estimator stays retired.

## Uncertainty methods (small justified set)

- **OLS:** analytic homoskedastic interval (control; must reproduce v0.10b failure).
- **HAC:** Newey–West Bartlett, bandwidth FIXED at 2 (sensitivity {1,3} on one
  condition only, reported regardless of outcome).
- **BBOOT:** moving-block bootstrap, block length FIXED at 3, B=200, percentile CI.
- **GLS-O:** feasible GLS with ORACLE AR(1) covariance (recoverability ceiling).
- **GLS-E:** feasible GLS with covariance ESTIMATED from OLS residuals (honest version).

## Test matrix (12 conditions)

- Truth: power a=1.2, t0=8.0 (slope truth = 1.2·8/16 = 0.6).
- Noise: multiplicative nl ∈ {0.1, 0.3}.
- Dependence: AR(1) ρ ∈ {0, 0.3, 0.5} + compound-symmetry ρ ∈ {0.3, 0.5}
  (non-specificity check) + MA(1) θ=0.5 (second non-AR check).
  Total: 2 × (3 + 2 + 1) = 12 conditions.
- Seeds: 5000–5199 per condition (n=200), disjoint from all prior ranges.

## Replication justification

n=200/condition: at true coverage 0.95, SE = √(0.95·0.05/200) ≈ 0.0154.
Gate PASS = empirical coverage ≥ 184/200 (92%): false-fail rate ≈ 2.5% if
truth is 0.95; a method with true coverage 0.88 passes with P ≈ 0.5%.
No seed-adding after results observed.

## Gates (frozen)

1. Coverage ≥ 92% (184/200) per condition — each of 12 reported separately.
2. Median width ≤ 1.0 per condition.
3. Bias/RMSE reported independently (no gate, guards against SE-only fixes).
4. Usable rate = 100% by construction (closed-form methods); bootstrap
   replicate failures reported, CI = (9,-9) sentinel if <10 succeed.
5. Complexity: simplest method passing ALL 12 conditions wins
   (order: OLS > HAC > BBOOT > GLS-E; GLS-O never selectable — ceiling only).

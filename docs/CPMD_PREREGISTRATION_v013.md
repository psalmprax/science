# Preregistration: CPMD v0.13 — Operating Envelope & Uncertainty Feasibility

**Document ID:** PREREG-CPMD-v013-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_envelope.py` (prereg frozen BEFORE run)
**Target (settled v0.12):** finite-window OLS slope, lags 4–12, truth 0.5619
  (power a=1.2, t0=8). Pointwise target abandoned for OLS.
**Sealed set:** unopened. Real-market promotion: NOT authorized.

## Direction A — operating envelope (OLS-normal interval, ρ=0)

Prespecified noise grid: nl ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}.
Seeds 7000–7199 per cell (n=200). Full gates per cell:
coverage ≥ 184/200, median width ≤ 1.0, |bias| ≤ 0.1, RMSE ≤ 0.3.
Envelope = largest contiguous nl-range from 0.05 upward passing ALL gates.

## Direction B — uncertainty feasibility at high noise (nl ∈ {0.20, 0.25, 0.30})

Methods (point estimate identical OLS in all):
- OLS-N: normal critical 1.96 (control).
- OLS-T: Student-t critical, df=7 (principled small-sample widening).
- RBOOT: residual bootstrap (resample RESIDUALS, x fixed — valid for
  regression; eligible now that v0.12 resolved the mean structure;
  distinct from the retired v0.11 pair-breaking implementation), B=500.
No interval shortening: any method passing coverage only by narrowing fails
the joint gate by construction.

## Gates (unchanged from v0.10/v0.11/v0.12)

184/200 coverage, width ≤ 1.0, usable 100% (closed form) / RBOOT sentinel
rule, per-cell reporting, no seed-adding, simplest passing method preferred.
If neither direction passes at a noise level: record the limitation.

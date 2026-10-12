# gen-R Calibration Report: Realistic Synthetic Market Generator

**Code:** `apedp_dyn/sandbox/cpmd_gen_realistic.py` (kinds: equity/crypto).
**Method:** 1 market factor + 3 block factors + idiosyncratic (all GARCH-t),
  common jump days, 3 segment vol multipliers (regime breaks).
**Status:** infrastructure — calibration iteration was legitimate here; the
  parameters below are FROZEN. Residual gaps documented, not hidden.

## Targets (measured from data/market panels) vs achieved (seed 7)

| Stat | Equity target | Equity gen-R | Crypto target | Crypto gen-R |
|---|---|---|---|---|
| Mean \|daily ret\| | 0.0144 | 0.0145 ✅ | 0.0387 | 0.0367 ✅ |
| Kurtosis | 8.3 | 6.0 (~28% low) | 18.6 | 11.3 (~39% low) |
| Sq-return AC1 | 0.066 | 0.060 ✅ | 0.146 | 0.066 (~55% low) |
| X-corr median | +0.30 | +0.43 (~0.13 high) | +0.61 | +0.71 (~0.10 high) |
| X-corr p90 | +0.61 | +0.72 (~0.11 high) | +0.72 | +0.83 (~0.11 high) |
| PC1 variance | 0.40 | 0.53 (high) | 0.63 | 0.74 (high) |
| Vol-regime shape | mid hump | mid hump ✅ | declining | declining ✅ |

## Known causes of residual gaps (3 calibration rounds; stopped here)

- Shared segment multipliers + common jumps inflate all pairwise
  correlations ~0.1 above target (common heteroskedasticity ≈ correlation).
- Summing 3 independent GARCH-t streams diversifies tails faster than df
  cuts restore them; crypto kurtosis remains well below 18.6.
- Crypto vol-clustering (sqAC1 0.146) only half-reproduced.

## Validity scope (frozen)

- VALID FOR: screening candidates under fat-tailed, clustered, non-stationary
  noise — strictly harder than the Gaussian stationary gens (gen-A/B/D/E/F).
- NOT VALID FOR: fine-resolution correlation-structure claims; tail-risk
  levels beyond kurtosis ~11; crypto clustering above ~0.07.
- Any candidate result on gen-R is comparative (vs same-gen null), never an
  absolute real-market claim. Real panels remain the only promotion path.

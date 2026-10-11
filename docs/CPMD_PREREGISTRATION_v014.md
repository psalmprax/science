# Preregistration: CPMD v0.14 — Independent Eval, Correlated Noise, Transfer

**Document ID:** PREREG-CPMD-v014-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_v014.py` (prereg frozen BEFORE run)
**Frozen from v0.13:** OLS point slope + Student-t interval + width cap 1.0 +
  envelope nl≤0.20 + abstain-beyond rule. NO tuning in v0.14.
**Sealed set:** opens ONLY if all v0.14 gates pass (decision recorded below).
**Real-market promotion:** NOT authorized regardless (separate decision).

## 1. Independent evaluation (same design, fresh seeds 8000+)

Base design (lags 4–12, power a=1.2 t0=8, ρ=0) × nl ∈ {0.05,…,0.20},
n=200/cell. Must reproduce v0.13 PASS pattern cell-by-cell.

## 2. Correlated-noise challenge (OLS-T, df=7, t=2.3646)

Structures (generators frozen from v0.11 `gen_dep`): AR(1) ρ∈{0.3,0.5},
compound-symmetry ρ=0.5, MA(1) θ=0.5; × nl∈{0.1,0.2}. Expectation recorded
in advance: coverage may fail — failure SHRINKS the envelope, never the gates.

## 3. Transfer evaluation (prespecified, ρ=0, nl∈{0.1,0.2})

- Lags 3–10 (n=8, df=5, t=2.5706), power truth, finwin truth recomputed.
- Lags 6–15 (n=10, df=8, t=2.3060, wide gen), power truth, finwin recomputed.
- Lags 4–12 EXPO truth (b=0.1): finwin-projected truth, FP n/a (slope≠0 legit).
- Lags 4–12 FLAT truth: slope must be ≈0; |slope|>0.5·width counts FP-equivalent.

## Gates (unchanged)

Per cell: coverage ≥184/200, median width ≤1.0, |bias|≤0.1, RMSE≤0.3.
Sealed set opens iff: independent eval reproduces ALL v0.13 passes AND
≥6/8 correlated cells pass AND ≥6/8 transfer cells pass. Otherwise stays shut
and the envelope shrinks to the passing region under a new version.

## Addendum 2026-10-11 (flat-cell follow-up, committed)

- Flat FP-equiv 10.5% in both nl cells traced to (a) seed reuse across nl
  cells (8800+k in both → identical t-stats, one draw counted twice) and
  (b) sampling noise: n=2000 check gives 4.85% ≈ nominal 5%.
  Method exonerated; seed-coupling flaw documented — future versions offset
  seeds per cell. Transfer corrected to 6/8; seal decision UNCHANGED
  (correlated-noise 2/8 still blocks opening).

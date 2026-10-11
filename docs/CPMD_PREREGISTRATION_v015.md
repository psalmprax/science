# Preregistration: CPMD v0.15 — Narrow-Claim Confirmation (Option A)

**Document ID:** PREREG-CPMD-v015-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_v015.py` (to be committed SEPARATELY,
  after this prereg; hashes recorded in commit messages)
**Procedure (frozen since v0.13):** OLS slope + Student-t interval +
  finwin target + width cap 1.0 + envelope nl≤0.20 + abstain beyond.
  Zero estimator changes in v0.15.
**Provenance fix:** this prereg is committed BEFORE the implementation
  (lesson from v0.14 shared-commit caveat). Per-cell seed offsets fix the
  v0.14 seed-coupling flaw (every cell gets a disjoint seed block).
**Sealed set:** unopened. Real-market promotion: NOT authorized.
**Option B** (correlated-noise procedure) is a separate track, not started here.

## Cells (8, all independent noise, n=200 each)

- Base lags 4–12, power a=1.2 t0=8: nl ∈ {0.05, 0.10, 0.15, 0.20},
  seeds 9000–9199 / 9200–9399 / 9400–9599 / 9600–9799.
- Lags 3–10 (t=2.5706), power: nl ∈ {0.10, 0.20}, seeds 9800–9999 / 10000–10199.
- Lags 6–15 (t=2.3060, wide gen), power: nl ∈ {0.10, 0.20}, seeds 10200–10399 /
  10400–10599.

## Gates (unchanged)

Per cell: coverage ≥184/200, median width ≤1.0, |bias|≤0.1, RMSE≤0.3.
PASS = 8/8. Any failing cell shrinks the claimed envelope to the passing
region; no gate may be altered in v0.15.

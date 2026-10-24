# Preregistration: CPMD v0.9.5 Selector Calibration

**Document ID:** PREREG-CPMD-v095-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_calibration.py` v0.9.5
**Status history:** v0.9.4 recorded as selector-calibration FAILURE; sealed set unopened.

## Background (corrected record)

- v0.1 included exploratory BTC + nine-asset results. The claim is therefore:
  v0.2–v0.9.4 focused on synthetic calibration with real-market promotion shelved —
  not that CPMD never touched real data.
- v0.9.2: log-space fitting, 129 → 0 fit failures.
- v0.9.3: bootstrap 265/300 (88.3%) coverage; FP 20/240 (8.3%). Gate failed.
- v0.9.4: BIC selector rarely picks power even on power truth; margin=6 abstains
  on everything. Diagnose-only; sealed test unopened.

## v0.9.5 design

- **Diagnostic set (fixed):** power seeds 1000–1039, expo 1100–1139, flat 1200–1219,
  regime 1300–1319; noise 0.3, independent errors, 12 lags. Hardcoded in module.
- **Rules compared:** BIC vs AICc × margins {0, 2, 6}. Rule chosen by preregistered
  gates below — never because it promotes power more often.
- **Sealed set:** generated from master seed 99991, disjoint ranges; hash recorded,
  evaluation NOT authorized until selector passes diagnostic gates.

## Acceptance gates (diagnostic)

1. FP rate ≤ 5% on non-power truth (expo/flat/regime).
2. Power recall reported; no credit for abstain-everything (recall must be > 0
   with coverage measured on recalled cases).
3. Bootstrap coverage ≥ 90% on recalled power cases (stepping stone to 95%).
4. All failures/abstentions in denominators.

## Verdicts carried forward

- Selector performance: failed calibration (v0.9.4).
- CPMD scientific conclusion: unresolved.
- Real-market promotion: NOT authorized.

# Preregistration: CPMD-ST2 — Rolling-Window Sensitivity & Baseline Comparison

**Document ID:** PREREG-CPMD-ST2-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_synth_trading2.py` (committed SEPARATELY after)
**Standing:** ST1 rejected (OLS-200 Sharpe −0.057); v0.15 untouched. Only the
  ROLLING LOOKBACK varies — slope estimator (lags 4–12 OLS-T), median rule,
  abstention cap, costs 2bp/side all frozen. No factorial changes.
**Market:** same generator as ST1 (hidden Markov φ=±0.3, P(stay)=0.98 →
  mean regime duration 50 steps; grid {25,50,100,200} straddles it by design).

## Phase 1 — development (seeds 20000–20029, SAME 30 paths as ST1)

- OLS-T rules with lookbacks {25, 50, 100, 200}; momentum-{5, 20};
  buy-hold, reversal, random on identical paths/costs.
- Select Winner = highest mean net Sharpe among the four OLS-T windows.
- Report per rule: net Sharpe, max DD, turnover, median regime-detection
  delay (steps from true switch to classification flip; regimes known).

## Phase 2 — independent (seeds 30000–30029, 30 fresh paths)

- Frozen Winner vs momentum-5 vs buy-hold ONLY. No other comparison may
  promote a different window (multiple-comparison control).

## Gates (ST2 passes iff ALL hold on Phase 2)

1. Winner mean net Sharpe > 0.
2. Winner beats momentum-5 on net Sharpe, paired Wilcoxon p < 0.05.
3. Winner max DD < 2× buy-hold DD.
Failure of any gate = lookback does not rescue the trading hypothesis.
No gate changes after results. Real markets uninvolved throughout.

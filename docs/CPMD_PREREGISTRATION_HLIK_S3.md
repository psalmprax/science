# Preregistration: CPMD-HLIK Stage 3 — Real-Data Economic Evaluation

**Document ID:** PREREG-CPMD-HLIK-S3-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_hlik_stage3.py` (committed SEPARATELY after)
**Standing:** H-LIK survived synthetic discovery+confirmation (results
  `17b40fe`). This tests REAL-DATA prediction + economics. FROZEN candidate:
  W=200, L=6, λ=1.0, trailing-20 σ, ε=1e-8. No tuning on real data, ever.
**No live-trading promotion follows any outcome** (would need replication).

## Real panels (out-of-box, provenance in data/market/PROVENANCE.json)

- EQUITY: data/market/equity_top15.csv — 14 stocks (SPY EXCLUDED: index
  dominance), 2021-01-04→2023-12-29, 753 synced days, log returns from closes.
- CRYPTO: data/market/crypto_top15.csv — 15 assets (COMPOSITE EXCLUDED),
  2021-01-01→2023-12-31, 1095 synced days, log returns from closes.
- Eligible range: t ≥ 200 (estimation window). Strictly causal throughout.

## Stage 3A — forecast validity (per panel)

- H-LIK forecast vs next-day return: pooled panel IC + circular-shift null
  (200 shifts within asset, as synthetic). Pass = p < 0.05 per panel.
- Zero baseline: MSE ratio < 1 required alongside (IC without scale is void).

## Stage 3B — incremental value (per panel)

- Comparators (frozen): MOM r̂ = sign(past-5 sum)·σ_t; COMMON r̂ = β̂_{i,t}·m_t
  (past-100 beta × current median shock).
- Residualize H-LIK forecast on [MOM, COMMON]; residual IC shift-null
  p < 0.05 per panel required.

## Stage 3C — trading economics (ONLY if 3A+3B pass both panels)

- Tercile long/short on r̂ (top-3 long / bottom-3 short, equal weight),
  5bp/side, daily rebalance. Baselines: equal-weight buy-hold, MOM-tercile.
- Metrics per panel + halves (time-block robustness): net Sharpe, net return,
  max DD, turnover. Gates: net Sharpe > 0 BOTH panels AND beats MOM-tercile
  (paired t on monthly net returns, p < 0.05); DD < 2× BH DD.
- Serial dependence + shared shocks: inference on monthly blocks / panel
  Newey-West; never asset-day independence.

## Verdicts

- 3A or 3B fail either panel → H-LIK has no demonstrated real-data edge; stop.
- 3C fail → forecast may pass without economic value (verdicts stay separate).
- 3C pass → earns INDEPENDENT REPLICATION (new data/period), not promotion.

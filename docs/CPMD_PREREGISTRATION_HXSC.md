# Preregistration: CPMD-HXSC Stage 1+2 — Shock Concentration Risk Test

**Document ID:** PREREG-CPMD-HXSC-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_hxsc.py` (committed SEPARATELY after)
**Standing:** H-XSC is UNTESTED. H-LIK stays rejected. No real-market trading
  claim follows any outcome; Stage 3 needs its own prereg.
**Data safeguard:** 2021–2023 panels are pre-examined → split declared NOW:
  development = first 60% of eligible days, confirmation = last 40%.
  No H-XSC number has been computed anywhere as of this commit.

## Frozen specification

- $z_{i,t} = r_{i,t}/(\sigma^{(20)}_{i,t} + 10^{-8})$; trailing-20 σ, causal.
- $q_{i,t} = |z_{i,t}|/(10^{-8} + \Sigma_j|z_{j,t}|)$;
  $H_t = (\Sigma_i q_{i,t}^2 - 1/N)/(1 - 1/N)$.
- Target $D_{t+1:t+5}$ (mean cross-sectional variance, next 5 obs, full eq. in
  test plan). Controls: $D_{t-4:t}$, $|m_t|$ ($m_t$ = median $z$), trailing-20
  std of median returns. Eligible $t \geq 220$ (200 burn + 20 σ warmup).
- Panels: equity 14 (ex-SP Y), crypto 15 (ex-composite), log returns, synced.

## Model + metric (frozen)

- OLS: $D_{t+1:t+5} = a + b·H_t + c·D_{t-4:t} + d·|m_t| + e·vol_t + \varepsilon$.
- Primary: two-sided $p$ for $b = 0$ via circular BLOCK bootstrap of $H_t$
  (block length 20, 500 replicates; preserves overlap dependence + shared
  shocks; refit full model per replicate). Pass = $p < 0.05$.
- Reported: incremental $R^2$ (full vs controls-only), sign of $b$
  (preregistered direction: $b > 0$; one-sided check noted, two-sided gated).

## Gates

- Development (first 60%): pass = $p < 0.05$ BOTH panels.
- Confirmation (last 40%, ONLY if development passes): same rule BOTH panels.
- Any fail → H-XSC REJECTED. Pass both → eligible for Stage 3 (new prereg).
- No window/horizon/control search at any stage.

# Preregistration: CPMD-HDAR — Delayed Asset Response Hypothesis Test

**Document ID:** PREREG-CPMD-HDAR-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_hdar.py` (committed SEPARATELY after)
**Standing:** new candidate (novelty not claimed vs all literature). ST track
  stays closed; v0.15 untouched. Multi-asset panel REQUIRED — single-series
  testing of H-DAR is invalid by design.
**Frozen params (no search):** N=10 assets, L=10, λ=0.8, c=1.0, γ=1.0,
  beta window 100 (past-only rolling OLS of z_i on m), σ trailing-20,
  ε=1e-8. Median m_t across assets.

## Generators (10-asset panels, 3000 steps)

- gen-D (delay world): r_{i,t} = β_i·F_t + δ_i·β_i·F_{t−1} + η_{i,t};
  F_t ~ N(0, 0.01²); β_i ~ U(0.5, 1.5) fixed per panel; δ_i = 0 for 5 fast
  assets, 0.5 for 5 slow assets; η ~ N(0, 0.005²). Catch-up exists.
- gen-E (instant world): same with δ_i = 0 ∀i. Nothing to catch → H-DAR
  must abstain (shift-null p ≥ 0.05 expected).
- β_i, δ_i assignments frozen per seed-block (recorded in code).

## Stages (gated: later stages run ONLY if earlier pass)

1. Prediction: H-DAR forecast r̂_{i,t+1} = σ_{i,t}·G_t·tanh(γ·D_{i,t}) vs
   actual; pooled panel IC; shift-null 200/panel (circular shift of forecasts
   within asset); primary p < 0.05 (single candidate, no correction).
2. Beyond-existing-effects: residualize H-DAR forecast on [own past-5 return,
   β̂·m_t] controls; residual IC shift-null p < 0.05 required.
3. Trading (ONLY if 1+2 pass): long/short top/bottom-tercile of r̂, 2bp/side,
   vs buy-hold-equal-weight + momentum-tercile; net Sharpe gate > 0.

## Paths/seeds

Discovery: 10 panels gen-D (seeds 100000+), 10 panels gen-E (110000+).
Confirmation (if stages pass): fresh 120000+/130000+. Stages 1–2 must pass
on BOTH discovery and confirmation; gen-E must abstain in both.
Failure anywhere → H-DAR rejected. No promotion to real markets regardless.

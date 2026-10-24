# Preregistration: CPMD-HLIK — Lagged Impulse Kernel Hypothesis Test

**Document ID:** PREREG-CPMD-HLIK-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_hlik.py` (committed SEPARATELY after)
**Standing:** H-DAR stays rejected (no repair, no threshold change). H-LIK is a
  DIFFERENT representation (distributed lags vs scalar discrepancy), tested
  under identical strictness. Predictive value unproven; novelty unclaimed.
**Frozen (no search):** W=200 estimation window, L=6 max lag, λ=1.0 ridge,
  σ trailing-20, ε=1e-8, leave-one-out median shock u_{i,t}.

## Generators (10-asset panels, 3000 steps)

- gen-D (H-DAR's world, fresh seeds): 2-lag delayed response as before.
- gen-E (instant control): δ=0; H-LIK must abstain.
- gen-F (NEW, preregistered here): distributed decaying delay —
  r_{i,t} = β_i·F_t + β_i·Σ_{ℓ=1..3} w_ℓ·F_{t−ℓ} + η (w = 0.3/0.15/0.05) for
  5 slow assets, instant for 5 fast. Tests generalization beyond gen-D.
  Used ONLY in confirmation (never for design — design uses gen-D/gen-E).

## Stages (gated)

1. Discovery (gen-D fresh 140000+, gen-E 150000+, 10 panels each):
   pooled panel IC + shift-null 200/panel; pass = ALL 10 gen-D p<0.05 AND
   ALL 10 gen-E p≥0.05 (abstain). Identical gate to H-DAR.
2. Confirmation (ONLY if stage 1 passes): gen-D 160000+, gen-E 170000+,
   gen-F 180000+ (10 panels each); same per-panel rules; gen-F pass =
   ≥8/10 p<0.05 (slightly relaxed: different pattern by design).
3. Trading (ONLY if stage 2 passes; new prereg): tercile rule, costs, baselines.

## Controls

Zero baseline (MSE ratio), momentum-tercile comparator reported at stage 3
only. Beyond-existing-effects (residualize on past-5 + β̂·m) required at
stage 2 alongside primary (same all-panels rule).
Failure anywhere → H-LIK rejected. No real markets regardless.

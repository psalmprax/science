# Preregistration: CPMD v0.16 — Frozen Discovery Suite (H-CTP/H-SER/H-SAC)

**Document ID:** PREREG-CPMD-v016-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_discovery_suite.py` (committed SEPARATELY after)
**Standing:** new candidate constructions, predictive value unproven. ST track
  stays closed; v0.15 untouched. A synthetic win ≠ real-market validity.
**Frozen lookbacks (no search):** k=20 (coherence/shock window), m=5, K=20
  (variance ratio), σ_t = rolling-20 std, ε=1e-8.

## Equations (verbatim from proposal)

- H-CTP: r̂ = σ_t·sgn(C_t)·(|C_t|−|C_{t−1}|),
  C_t = Σz/ (ε+Σ|z|) over k=20, z_t = r_t/(σ_t+ε).
- H-SER: r̂ = −σ_t·Q_t, Q_t = Σz|z|/(ε+Σz²) over k=20.
- H-SAC: r̂ = σ_t·Q_t·D_t, D_t = max(0, tanh(log(V_{t−1}/V_t))),
  V = short(m=5)/long(K=20) variance ratio.

## Generators (no built-in recoil/continuation rule — avoids circularity)

- gen-A: ST1 hidden-Markov AR(1) φ=±0.3, P(stay)=0.98, σ=0.01.
- gen-B: burst-volatility random walk: σ_t switches 0.005/0.02 (P(stay)=0.97),
  zero-mean Gaussian returns. Symmetric bursts; no directional rule built in.

## Stages

- Stage 1 discovery: 20 paths × 3000 steps per generator, seeds 50000+.
- Stage 2 confirmatory: survivors ONLY, 20 fresh paths per generator,
  seeds 60000+.
- Metric: Pearson IC between r̂_{t+1} and r_{t+1} (causal alignment),
  mean across paths. Null: 200 circular shifts of forecasts per path;
  p = fraction of null mean-ICs ≥ observed, one-sided, Bonferroni α=0.05/3.
- MSE vs zero baseline reported (no gate; guards against IC-without-scale).

## Verdicts

- No survivor → reject tested hypotheses.
- Survivor replicates in Stage 2 → next evidential stage (costs/positions).
- Multiple survive, no separation → MULTI_HYPOTHESIS.
- Works Stage 1, fails Stage 2 → no promotion.
Costs, sizing, trading deferred to a later stage. Real markets uninvolved.

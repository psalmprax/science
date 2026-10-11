# Preregistration: CPMD v0.17 — Generator-Aware Discovery Audit

**Document ID:** PREREG-CPMD-v017-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_audit_harness.py` (committed SEPARATELY after)
**Standing:** v0.16 closed (3 rejections). No reopening OLS-200/ST verdicts.
  Positive control is harness calibration ONLY, kept separate from candidate
  evaluation. Real markets uninvolved.

## Step 1 — generator audit (documented + quantified)

- gen-A (ST1 Markov AR φ=±0.3, P(stay)=0.98): oracle bound = IC/Sharpe of the
  TRUE-regime rule (momentum in PERSIST, reversal in REVERT) on seeds 20000+.
  Bounds the recoverable signal; permitted inputs exclude hidden states.
- gen-B (burst-vol RW, symmetric): oracle bound = 0 by construction (zero-mean,
  no directional rule); any edge found there is false positive by design.

## Step 2 — harness calibration (gen-C, audit-only generator)

- gen-C: r_{t+1} = 0.5·sign(mean(r_{t-9..t}))·σ + N(0,σ), σ=0.01 — a KNOWN
  observable coherence-like rule built in (seeds 70000+).
- Positive control (frozen): r̂ = sign(mean past-10 returns)·σ_t. Must yield
  one-sided shift-null p < 0.05/2 (Bonferroni over the 2 harness checks).
- Null case: same control on gen-B paths must yield p ≥ 0.05/2 (must abstain).
- If either check fails, the harness — not any candidate — is unfit; stop.

## Step 3 — novel candidate H-CMD (coherence-momentum divergence)

- From audit: gen-A regimes flip the SIGN of autocorrelation; untested
  observable angle = interaction of path coherence with recent direction.
- Frozen: C_t as in v0.16 (k=20); M_t = sign(mean(r_{t-4..t}));
  r̂_{t+1} = σ_t · C_t · M_t · (|C_t| − |C_{t−1}|).
  Tests whether coherence-direction ALIGNMENT (not coherence alone) predicts.
- Evaluated ONLY on untouched paths: gen-A seeds 71000+, gen-B seeds 72000+
  (20 paths × 3000). Same shift-null + Bonferroni (α=0.05/3 over H-CMD×2 gens
  + ... precisely: 2 tests, α=0.025 each). No redesign from results.

## Verdicts

- Oracle bound ≈ 0 everywhere → environment uninformative; stop generating.
- Harness checks pass + H-CMD replicates → next evidential stage.
- Harness checks fail → v0.17 fails as calibration; no candidate verdict.
- H-CMD fails → rejected; audit record stands as the deliverable.

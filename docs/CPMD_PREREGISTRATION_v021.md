# Preregistration: CPMD v0.21 — Oracle-Calibrated Identifiability Benchmark

**Document ID:** PREREG-CPMD-v021-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_v021.py` (committed AFTER this prereg)
**Standing:** v0.20 referee 2/2 valid; ladder void. This redesign pre-qualifies
  legs structurally instead of hoping calibration lands.
**Key redesign:** exogenous observable drivers (not return-derived features),
  so oracle strength is ANALYTIC and coupling is directly settable.

## Drivers (observable, causal, exogenous)

f1, f2 = AR(1) processes, persistence 0.9, unit variance, cross-correlation
ρ set per leg by construction (shared-shock mixing, frozen formula in code).
Truth: r_{t+1} = β·f1_t + N(0,σ), σ=0.01. Analytic oracle IC ≈
β/√(β²+σ²) (unit-variance driver): β=0.001 → LOW band [0.05,0.10);
β=0.0015 → MED band [0.10,0.20). (β=0.001: 0.001/√(1e-6+1e-4)≈0.0995;
β=0.0015: ≈0.149.) Bands verified numerically pre-discovery; leg auto-voids
outside band (code-enforced, no human call). Degenerate (NaN/const) auto-void.

## Gate B — coupling rungs (MED band, truth on f1 only)

- HIGH coupling ρ=0.9, MED ρ=0.5, LOW ρ=0.1.
- Frozen grammar (12 forms): f1, f2, (f1+f2)/2, (f1−f2)/2, tanh(f1),
  tanh(f2), sign(f1), sign(f2), f1·f2, f1²−f2², σ-scaled f1, σ-scaled f2
  (σ = rolling-20 return std; candidates predict returns, drivers observable).
- Expectation: LOW → UNIQUE possible; HIGH → MULTI expected. Recorded, not
  forced: referee scored against EPISTEMIC correctness (see C).

## Gate C — referee scoring (fresh seeds per leg)

- LOW leg: correct iff referee returns UNIQUE_IDENTIFICATION with f1-form
  among confirmed, OR MULTI_HYPOTHESIS with f1-form surviving (weak pass).
  Decisive UNIQUE scores 2, weak MULTI scores 1.
- HIGH leg: correct iff MULTI_HYPOTHESIS (2 pts) — unique promotion of any
  single form scores 0 (false precision).
- NULL leg (β=0, ρ=0.5): correct iff NO_RECOVERY (2 pts); any promotion 0.
- Total ≥ 5/6 = referee validated. Fresh seeds: calib 96000+, disc 97000+,
  confirm 98000+, null 99000+. 20 paths × 3000 steps. Bonferroni 0.05/12.
  Equivalence (behavioral corr ≥0.7) reported, not gated.

## Verdicts

- ≥5/6 → referee validated across coupling range; identifiability boundary
  measured (where UNIQUE gives way to MULTI).
- <5/6 → referee limitation recorded; diagnose, no silent fix.
- Void legs (code-enforced) → inconclusive rung, reported as such.

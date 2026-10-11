# Preregistration: CPMD v0.18 — Signal-Strength Calibration & Controlled Discovery

**Document ID:** PREREG-CPMD-v018-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_v018.py` (committed AFTER gen-C exists;
  candidates below frozen BEFORE it — see commit order for blindness proof)
**Correction owned:** v0.17's "100× more paths" was wrong. t = 0.0053/0.0036
  = 1.47 (not significant); ~53 paths for 95% CI exclusion, ~108 for 80%
  power at that effect size. Oracle = empirical benchmark, not proven optimum.

## 1. Detection boundary (strength ladder, 20 paths each, seeds 80000+)

Controls r_{t+1} = c·sign(mean(r_{t-9..t}))·σ + N(0,σ), σ=0.01,
c ∈ {0.0 (null), 0.1 (weak), 0.5 (moderate), 2.0 (strong)}.
Per rung report: oracle IC, harness shift-null p (positive-control equation
r̂ = sign(mean past-10)·σ_t), rejection rate, IC CI coverage (nominal 95%).

## 2. gen-C hidden-equation environment (created AFTER candidate freeze)

Nonlinear observable rule of meaningful nontrivial strength (target
oracle IC ≈ 0.1–0.2), equation committed with the implementation —
AFTER the candidates below were frozen, so no leakage into their design.

## 3. Candidates (FROZEN HERE, before gen-C exists)

- H-CTP (v0.16, k=20) — rejected-elsewhere comparator.
- H-QSC (new): r̂_{t+1} = σ_t·Q_t·|C_t| (shock-coherence interaction).
- H-VTI (new): r̂_{t+1} = σ_t·sign(V_t − V_{t−1})·M_t, M_t = sign(mean r_{t-4..t}),
  V short(5)/long(20) variance ratio.
All use v0.16 conventions (σ rolling-20, ε=1e-8, causal alignment).

## Evaluation (fresh paths throughout)

- Ladder: 20 paths/rung, seeds 80000–80079 (disjoint from v0.17 70000s).
- Discovery: gen-C 20 paths seeds 81000+; shift-null 200/path; Bonferroni
  α=0.05/3. No redesign, no Stage-2 unless a candidate passes.
- Verdicts: power curve + false-promotion rate + coverage for the ladder;
  per-candidate pass/fail on gen-C. gen-A stays weak-signal label, gen-B null.

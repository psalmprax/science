# Preregistration: CPMD v0.19 — Hidden-Mechanism Recovery Benchmark

**Document ID:** PREREG-CPMD-v019-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_v019.py` (committed AFTER this prereg)
**Standing:** v0.16–v0.18 closed (rejections + calibration). This tests whether
  the PIPELINE can recover a hidden recoverable mechanism — not real markets.
**Blindness:** proposer = blind exhaustive script over the frozen grammar
  (no human tuning possible — it tries everything). Exact truth equations are
  committed WITH the implementation (after this prereg) and never alter the
  proposer, which is grammar-fixed here.

## Frozen grammar (proposer may use ONLY these; budget ≤ 60 forms)

Features: C, Q, V, M, σ, |C|, Δ|C|=|C_t|−|C_{t−1}|, tanh(Q), tanh(C),
Q·|C|, C·M, Q·M, sign(V_t−V_{t−1})·M, sign(C)·Δ|C| — each × {raw, ×σ_t}
scaling variants as listed in code (total ≤ 60). k=20, m=5, K=20, ε=1e-8.

## Hidden mechanisms (strength ≈ IC 0.09 rung, calibrated one-shot below)

- IN-GRAMMAR truth: r_{t+1} = A·σ_t·Q_t·|C_t| + N(0,σ) (H-QSC form; in grammar).
- OUT-OF-GRAMMAR truth: r_{t+1} = A·σ_t·sin(3·Q_t)·M_t + N(0,σ) (sin ∉ grammar).
- Amplitude A calibrated ONE SHOT on dedicated seeds (90000–90009) to oracle
  IC ≈ 0.09 ± 0.03; value recorded, no iteration. Discovery/confirmation use
  fresh seeds (91000+, 92000+).

## Protocol

- Discovery: 20 paths × 3000 steps per mechanism (seeds 91000+). Every grammar
  form scored (mean IC); shift-null 200/path; pass = Bonferroni p < 0.05/60.
- Confirmation: passing forms ONLY, 20 fresh paths (92000+), same bar.
- Null control: gen-B 20 paths (93000+); any promotion = false-promotion event.
- Equivalence: confirmed form must ALSO reach behavioral corr ≥ 0.7 with the
  TRUE forecast series on held-out paths (predictive equivalence, not symbolic).

## Verdicts (from hypothesis table)

- In-grammar recovered + null clean → pipeline works under control.
- Representable but never proposed → search limitation.
- Equivalent found but referee rejects → adjudication/power limitation.
- In-grammar yes + out-of-grammar no → measured grammar boundary.
- Nothing → diagnose search/grammar/effect/power separately.
- Null promotes → direct challenge to promotion rules.

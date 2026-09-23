# Session Ledger — measured results not covered elsewhere

**Purpose:** every number below was executed, with budget/seed recorded.
Status labels: CONFIRMED (reproduced), SINGLE-RUN (one seed/config),
WITHDRAWN (retracted after further testing).

## Discovery benchmark (Tier A/C)

- A1 recognition 3/3 → 4/4 after GRATING added (CONFIRMED, fast tests)
- Tier C deception 4/4 → extended set, all pass (CONFIRMED, fast tests)
- A2 exact recoveries, all withheld-prior, seed 7 (SINGLE-RUN each):
  OHMIC 15×40 ✓ (mse 0.0) · KINETIC 15×40 ✓ (0.0006) ·
  EINSTEIN 15×40 ✓ (0.0005) · SB 15×40 ✓ after power-mutation fix (0.0005)
- A2 SB failures before fix: 15×40 T²-decoy, 25×60 linear decoy,
  power-prior scaffold no-effect (identical mse), log-space linear decoy
- Ensemble OHMIC seeds 7/42/123 small-budget: exact, linear-decoy, exact
  → recovery 2/3, winner auto-selected (SINGLE-RUN)
- Tier D noisy-Ohm 5%: `0.989·x·y+0.02·y`, structure adjacent, exact form
  lost (SINGLE-RUN)

## Heat transfer (Nu)

- Incumbent DB held-out MAE 8.148; candidate 7.716 on original sample.
- WITHDRAWN as superiority: fresh seeds give 8.14/7.79 (p=0.21),
  7.74/7.88 (p=0.12), 7.55/7.96 (p=0.06) — winner flips, nothing <0.05.
- Standing result: rational operator expands reachable structure (√Pr);
  no superiority claim. Candidate demoted to CONDITIONAL (7/625 domain
  violations, worst −22.89).
- Extrapolation: interior DB wins, low-Re tail candidate wins, high-Re
  tail both collapse (DB less). Local-surrogate verdict stands.

## Friction (Colebrook truth, synthetic)

- Haaland MAE 0.000124. Discovered additive form 0.00619 (50× loss).
- With logRe terminal: `24.45 − 29.94/logRe^(1/3)`, MAE 0.0037 (still
  30× short). Ablation ladder recorded; log-composition unreached.

## Batteries (NASA PCoE B0005/6/7, real data)

- Fixed forms: exp/linear best B0005/B0007; double-exp wins B0006.
- Quadratic (demography transfer): worst everywhere + false EOL on B0007.
- Discovery run (B0005, 15×40): `1.824 − 5.54e-7·k³`, MAE 0.79 vs 0.059.
  Single equation per run; seed-stability untested.

## Warp battery (see docs/WARP_DRIVE_STUDY.md)

- Recognition: Lorentz exact 1.7s; expansion→CONDUCTION structural 18.3s.
- 6 composites generated; #3/#5 correspondence-verified (c→∞ True).
- Energy dataset `data/warp_energy_estimates.jsonl` (9 rows);
  FP-vs-Lentz wall-scaling agrees within ~60× across 33 orders.

## Statistics (see docs/STATISTICAL_EQUATIONS_STUDY.md, POPULATION_STUDY.md)

- US quadratic exact form; Bertozzi refutation table; D-T 17.589.
- Rolling-origin corrected counts (2/6-4/6-0/6); full sweep 785 windows
  (quad 40%, logistic 31%, expo 19%, Gompertz 10% wins, 35.3% fails).
- COVID: families largely fail multi-wave; protocol distinguishes domains.

## Withheld-scope finding

`audit_novelty` merges custom_corpus INTO the standard corpus (197 entries, v4) — entries
cannot be subtracted; pinned by test, not fixed (API semantics).

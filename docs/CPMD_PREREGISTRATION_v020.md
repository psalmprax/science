# Preregistration: CPMD v0.20 — Identifiability & Observational Equivalence

**Document ID:** PREREG-CPMD-v020-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_v020.py` (committed AFTER this prereg)
**Standing:** v0.19 split (recovery yes, identification no, out-grammar void).
  Out-of-grammar leg stays OFF-PROTOCOL — no rescaling here.
**Referee (frozen):** promote UNIQUE_IDENTIFICATION iff exactly one form
  passes Bonferroni (0.05/60) AND beats runner-up on held-out MSE by paired
  t-test p<0.05; ≥1 pass without separation → MULTI_HYPOTHESIS; none →
  NO_RECOVERY; pass without equivalence → PREDICTIVE_RECOVERY. Equivalence =
  behavioral corr ≥0.7 with true forecast on held-out paths.

## Experiment A — significance vs identification (frozen v0.19 evidence + held-out)

9 in-grammar survivors re-compared on FRESH paths (seeds 94000+, 20 paths):
pairwise loss differentials vs QxCs (paired t on squared-error diffs).
Question: is QxCs detectably better, or genuinely tied?

## Experiment B — coupling ladder (one-shot calibration each, record-only)

Three truths at oracle IC ≈ 0.10 (accept 0.06–0.12; outside → that leg
OFF-PROTOCOL, no tuning), amplitude fixed A=2.0, one measurement each:
  (i) HIGH coupling: r = 2σ·Q·|C| (v0.19 case);
 (ii) LOW coupling: r = 2σ·M (momentum sign; near-orthogonal to Q/C);
(iii) MODERATE: r = 2σ·tanh(Q)·sign(C).
Full 60-form search per leg, discovery seeds 95000+, confirmation 96000+.
Question: does uniqueness track coupling?

## Experiment C — referee scoring (expected vs returned outcome)

 (i) truth M leg → expect UNIQUE or MULTI (scored correct if referee returns
     either PREDICTIVE_RECOVERY+... precisely: correct iff outcome ∈
     {UNIQUE_IDENTIFICATION, MULTI_HYPOTHESIS} with true form among survivors);
(ii) truth Q|C| leg → correct iff MULTI_HYPOTHESIS (promiscuity known);
(iii) gen-B null → correct iff NO_RECOVERY (any promotion = false event).
Referee score 3/3 = pass; else fail. Fresh seeds throughout.

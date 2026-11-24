# Preregistration: does contrast confirmation work on laws it has never seen?

**Document ID:** PREREG-CONTRAST-2026-11
**Frozen:** at the commit that adds this file; `preregistration/DIGEST-CONTRAST.json` holds the SHA-256
of this document, `scripts/run_prereg_contrast.py`, the two scripts it imports and
`apedp_dyn/discovery_l2/contrast_confirmation.py`. The script refuses to run if any changed.
**Author:** Claude (session `01AmWYSGATN2jrEj786Sttks`), at the repository owner's request.

## Why

`contrast_confirmation.py` was developed on the eight two-term laws of PREREG-BLIND-L2-2026-11 and
changed three times while looking at them. Its development numbers (true law uniquely confirmed in 46
of 48 runs) are therefore not evidence. They were also not at equal data: it read designed rows the
referee never saw. This run fixes both: fresh laws, and a control arm that reads the same number of
rows per comparison, chosen at random.

## Design

| | |
| :--- | :--- |
| Laws | the 40 two-term laws c₁·x^p + c₂·y^q among 120 drawn by the blind benchmark's generator with seed 20261120, a seed never used before (none coincides with an earlier law). No law is filtered out, including ones whose second term is too small to identify |
| Data | as in the blind benchmark: 120 development rows, every variable uniform on [1, 10], one distractor variable, multiplicative noise 1 % and 5 %; a pool of 5,000 unlabelled rows from the same domain, with one fixed noise draw per pool row shared by all arms |
| Candidates | L2's twelve best distinct candidates by its internal holdout error |
| Arm A, contrast | `contrast_confirm(..., rows_by="contrast")`, defaults (30 rows per comparison, α = the canonical Diebold–Mariano threshold) |
| Arm B, control | the same call with `rows_by="random"`: same candidates, same procedure, same rows per comparison, drawn at random |
| Arm C, truth removed | arm A with the true law deleted from the candidates |
| Also recorded | the referee's own verdict on the 120 rows |
| Runs | 40 laws × 2 noise levels = 80, each through all arms |

## Expectations, fixed now

1. **It rarely names a wrong law.** In arm A, runs ending `unique` or `unchallenged` with a survivor
   that is not the true law are at most 5 % (at most 4 of 80).
2. **Choosing the rows is what helps.** Arm A confirms the true law uniquely in more runs than arm B;
   one-sided paired sign test on the discordant runs, p < 0.01.
3. **It rarely refutes a true law.** In arm A, `no_survivor` when the true law was among the
   candidates is at most 5 % of those runs.
4. **It notices a missing law.** In arm C, `no_survivor` in at least 80 % of runs, and a wrong law
   named alone in at most 5 %.

A miss on any of the four is reported as a miss. How often arm A confirms the true law uniquely, and
how that compares with the referee, are reported without a criterion.

## Deviation rule

One run, locally, into a new directory. A crash is fixed by a dated deviation committed here before
re-running. Changing the procedure, the laws or an expectation after the run is a new
preregistration.

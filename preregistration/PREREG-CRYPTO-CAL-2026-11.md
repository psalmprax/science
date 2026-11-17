# Preregistration: the referee's in-sample error rates on real crypto returns

**Document ID:** PREREG-CRYPTO-CAL-2026-11
**Frozen:** at the commit that adds this file; `preregistration/DIGEST-CAL.json` holds the SHA-256 of
this document and of `scripts/run_calibration_crypto_dev.py`, and the script refuses to run if
either has changed.
**Author:** Claude (session `01AmWYSGATN2jrEj786Sttks`), at the repository owner's request.

## Why

PREREG-CRYPTO-PIT-2026-11 had one positive and one null control. The referee passed the one and
refused the other, which shows it *can* tell them apart on this data and says nothing about how
often it errs. This measures that for stages 2 and 3, the gates a candidate meets before any
holdout, on real returns with their real fat tails and cross-sectional dependence.

## Data and disclosure

The development panel of PREREG-CRYPTO-PIT-2026-11 only (`panel_development.csv.gz` on the
`market-data` branch, SHA-256 `96b5ee0b88c4dd5c004565f2582c8683b440513ad5dc9f8d511a0fb49fa4644e`),
2021-01-01 to 2025-12-31; the script refuses a panel that runs past 2025-12-31 and is never given
the holdout file. Same universe (point-in-time top 30), same pairing (centred within-day ranks,
next-day return as target, at least 10 assets a day), same literature features (one-day reversal,
twenty-day momentum) as that preregistration's frozen analysis, which this script imports
unchanged.

Seen before freezing: that run's outcome, which is on development data for H1-OFI (development IC
of order-flow imbalance 0.0013 in absolute value) and for the two controls. This calibration
contains none of those candidates. No synthetic candidate below has been run on the real panel;
the script was exercised only on synthetic panels (`tests/test_calibration_crypto_dev.py`).

## Candidates

Each goes through a fresh `PromotionTracker`: stage 1 reported (as in the parent preregistration),
stage 2 from the development pairs, stage 3 if stage 2 passed. Thresholds are the canonical ones
(`STAGE2_MIN_IC = 0.02` with every one of five contiguous folds positive; Diebold–Mariano
`p < 0.01` against a zero forecast; literature R² `< 0.85`).

| Family | How many | Predictor | What a pass means |
| :--- | :--- | :--- | :--- |
| Placebo | 500 | uniform draws from a generator seeded by the candidate's id | a false pass |
| Injected | 100 at each ρ ∈ {0.01, 0.015, 0.02, 0.025, 0.03, 0.05} | within each day, `ρ·z + √(1−ρ²)·ε`, `z` the standardized rank of the next day's return, `ε` seeded noise | detection of a true effect of that size |
| Literature | 2 | reversal and momentum themselves | stage 3 must refuse them if stage 2 passes |

The injected signals are built from the target, so their effect is real by construction and of
known size; what is measured is whether the gates find it in this data's sample size and noise.
Each candidate's measured development IC is recorded with its verdict.

## Expectations, fixed now

1. **Placebo false-pass rate at stage 2 is at most 1%** (at most 5 of 500), and the 95%
   Clopper–Pearson interval is reported. Expected: 0. With about 53,000 pairs, a placebo's IC has a
   standard error near 0.004, so 0.02 is about five standard errors away.
2. **Power:** stage 2 passes at most 10% of the ρ = 0.01 signals and at least 90% of the ρ = 0.03
   and ρ = 0.05 signals. The rates at 0.015, 0.02 and 0.025 are reported without a criterion; the
   threshold sits among them.
3. **Literature:** no literature factor passes stage 3. If neither passes stage 2, that is reported
   and this expectation is untested.

A miss on 1, 2 or 3 is reported as a miscalibration of the referee on real data, not explained
away. The run writes `results.json`, `candidates.jsonl` (every candidate's IC and verdict) and
`promotion_journal.jsonl`, committed unedited with a report,
`preregistration/RESULT-CRYPTO-CAL-2026-11.md`.

## Deviation rule

The run happens once. If the script crashes on the real panel, the fix is committed as a dated
deviation in this directory before it is re-run. Changing a family, a count, a strength or an
expectation after the run is not a deviation; it is a new preregistration.

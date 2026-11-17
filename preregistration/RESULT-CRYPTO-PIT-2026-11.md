# Result: PREREG-CRYPTO-PIT-2026-11

**Run:** once, 2026-11-17 01:44:20–01:44:25 UTC, `scripts/run_prereg_crypto_pit.py` at commit
`d32b784`, whose analysis script, preregistration text and data builder match
`preregistration/DIGEST.json` (`scripts/check_preregistration.py`: intact). Python 3.13.18, numpy
2.5.3, scipy 1.18.1, pandas 3.0.5.

**Data:** branch `market-data` at `a598b83`, built on GitHub Actions (run 43995763671) from commit
`7a3e2b7`, **under Deviation 01** (`DEVIATION-01-2026-11-17-builder-url-encoding.md`: the builder
percent-encodes a non-ASCII symbol; nothing else changed, and no data existed before it). The
manifest carries the digests of the three frozen files at build time, and they are the ones in
`DIGEST.json` now.

| File | SHA-256 |
| :--- | :--- |
| `panel_development.csv.gz` | `96b5ee0b88c4dd5c004565f2582c8683b440513ad5dc9f8d511a0fb49fa4644e` |
| `panel_holdout.csv.gz` | `3087791d0a6596b097dbd4d96c0814f800ba5f98fb95f23eb9c395748de6a709` |

3,767 symbols listed in the archive; 692 USDT assets in the panel after the preregistered
exclusions (71 pairs excluded), 183 of which stopped trading before 2026-10-31; 24,693 monthly
source files, each checked against its published SHA-256. The point-in-time top-30 universe had
217 different members over the period.

Outputs, unedited, in [`RESULT-CRYPTO-PIT-2026-11/`](RESULT-CRYPTO-PIT-2026-11/): `results.json`,
`promotion_journal.jsonl`, `experiment_ledger.jsonl` (with its `.head` marker), and `run.log`.
`trial_ledger.jsonl` was not written: the run writes it at stage 5, and no candidate reached
stage 5.

## Outcome

| Candidate | Role | Highest stage passed | Refused at | Deciding numbers |
| :--- | :--- | :--- | :--- | :--- |
| CPLUS-VOL | positive control | **4** (out-of-sample transfer) | — (stage 5 is not run for controls) | development IC 0.335, every fold positive (0.253 … 0.391); DM p computed as 0.0 against zero (below double precision), literature R² 0.013; January 2026 holdout IC **0.426**, p = 2.2e-15, n_eff 309 of 900 pairs, retention 1.27 |
| CZERO-PLAC | null control | 1 | **stage 2** | development IC −0.0025 (threshold 0.02), folds −0.016 … 0.008 |
| H1-OFI | the test claim | 1 | **stage 2** | development IC of raw OFI −0.0013, so sign −1 and IC 0.0013 (threshold 0.02); folds −0.019 … 0.017, not stable |

**The referee behaved as designed on this data,** by the criterion fixed in advance: the positive
control passed stage 4 and the null control was refused at stage 2.

**H1-OFI was refused at stage 2.** On the 2021–2025 development period, daily order-flow imbalance
(taker-buy share of volume) carried no cross-sectional information about the next day's return
among the 30 largest pairs: a pooled rank correlation of 0.0013 in absolute value over 52,887
asset-days, with folds of both signs. It never reached the holdout. The author's recorded
expectation was a refusal at stage 4 or 5; it was refused earlier than expected, on development
data alone.

## What this does and does not show

- The control pair is one positive and one null. A pass on both shows that, on this panel, the
  referee can promote a well-documented effect and refuse one null candidate. It does not measure a
  false-positive rate; one placebo cannot.
- Stage 1 was reported for all three, not proved, and every record says so
  (`MATHEMATICAL_CONSISTENCY_VERIFIED_ON_REPORTED_METRICS`); no candidate is evidence-backed in the
  tracker's sense.
- The test window (2026-02-01 to 2026-10-31) was not scored by any candidate and the experiment
  ledger holds no claim on it; the only window claimed is January 2026, by CPLUS-VOL. The frozen
  script did read the holdout file and, for H1-OFI, assembled the test window's prediction and
  target ranks in memory to record their count (8,153 asset-days over 272 days); it computed no
  statistic relating the two, and none is in the outputs (`tests/test_prereg_crypto_pit.py` pins
  that a refused candidate is never scored on the holdout).
- One exchange, daily bars, one market regime per window. "No daily OFI signal at a 0.02 IC bar in
  the top 30" is the finding; nothing about intraday order flow, smaller pairs, or other venues.
- The volatility control's January 2026 IC (0.43) exceeding its development IC (0.34) is a
  retention above 1, not evidence of anything beyond volatility persistence holding in that month.

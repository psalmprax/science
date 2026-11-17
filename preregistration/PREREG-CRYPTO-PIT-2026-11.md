# Preregistration: order flow on a survivorship-free crypto universe, unseen 2026 holdout

**Document ID:** PREREG-CRYPTO-PIT-2026-11
**Frozen:** at the commit that adds this file. `preregistration/DIGEST.json` records the SHA-256 of
this document, of the analysis (`scripts/run_prereg_crypto_pit.py`) and of the data builder
(`scripts/build_point_in_time_crypto_panel.py`). The data build
(`.github/workflows/market-data.yml`) refuses to run unless the three files match that record, and
writes the record into the data manifest, so the order — this document, then the data, then the
run — is in the repository's history.
**Author:** Claude (session `01AmWYSGATN2jrEj786Sttks`), at the repository owner's request.

## Disclosure

- No data from 2026 and no part of the panel this run uses had been downloaded or viewed by the
  author when this was frozen. The panel does not exist yet; it is built afterwards on GitHub's
  runners from the public archive.
- Seen before freezing: the repository's existing market files (`data/market/*`: a hand-picked list
  of 15 crypto names, 2021–2023, recorded there as hindsight-selected) and
  `docs/TRADING_EQUATION_FORENSICS.md`, which studies order-flow and volatility hypotheses on them.
  The README records that development-period market hypotheses (H-LIK, H-XSC) "failed or degraded
  during prospective real-market confirmation". The test claim below was chosen with that knowledge,
  not in spite of it: it is the simplest order-flow claim, and the point is to see what the referee
  says about it on data no one has used.
- The analysis was exercised only on synthetic panels (`tests/test_prereg_crypto_pit.py`).

## Data

Every Binance spot pair quoted in USDT that the public archive (`data.binance.vision`) holds,
including pairs delisted before the end, each up to its last real bar. Excluded, by rules in the
builder: pegged bases (stablecoins, fiat, gold tokens; a fixed list) and leveraged tokens (a base
ending in UP, DOWN, BULL or BEAR whose remainder is itself a listed base, so `BTCUP` goes and `JUP`
stays). Daily bars,
2021-01-01 to 2026-10-31. Every source file is checked against its published SHA-256.

- **Development period:** 2021-01-01 to 2025-12-31 (`panel_development.csv.gz`).
- **Holdout:** 2026-01-01 to 2026-10-31 (`panel_holdout.csv.gz`), split once, by date, before
  anything is scored:
  - **control window**, 2026-01-01 to 2026-01-31: the positive control's;
  - **test window**, 2026-02-01 to 2026-10-31: the test claim's.

  The experiment ledger lets one candidate confirm on a given window
  (`HoldoutContaminationError` otherwise), so each window has one owner.

**Universe:** `apedp_dyn.market.universe.point_in_time_membership` with `top_n=30`,
`lookback_bars=60`, `rebalance_every=21`: the 30 largest by traded value over the 60 bars before
each rebalance, using nothing observed at or after it. A member whose bar is missing leaves the
universe on that bar.

## Candidates

Each predicts a **centred within-day cross-sectional rank** in [−0.5, 0.5] among the members with a
finite prediction and target that day. A day enters only with at least 10 such members. The target
of the prediction made at the close of day *t* is day *t+1*'s return, `close_{t+1}/close_t − 1`.
A pair belongs to a period only if its target day does too: no prediction made in the development
period has a target in the holdout.

| ID | Predictor (day *t*) | Target (day *t+1*) | Role |
|---|---|---|---|
| **H1-OFI** | `sign × OFI_t`, `OFI = (2·taker_buy − volume) / volume` | return | the test claim |
| **CPLUS-VOL** | mean `|r|` over the 20 bars ending at *t* | `|return|` | positive control (volatility persistence) |
| **CZERO-PLAC** | SHA-256 of (ID, asset, date) as a number in [0, 1) | return | null control |

`sign` is the sign of H1-OFI's pooled Spearman IC on the development period (positive if zero).
Both signs were looked at, so both are journaled as trials for the deflated Sharpe ratio.

## The referee

Every candidate runs through `apedp_dyn.epistemic.promotion_lifecycle.PromotionTracker`, with every
threshold taken from the canonical specification as the tracker imports it (at freeze:
`STAGE2_MIN_IC = 0.02`, Diebold–Mariano `p < 0.01`, literature R² `< 0.85`,
`STAGE4_MIN_OOS_IC = 0.015`, `STAGE4_OOS_ALPHA = 0.05`, retention `≥ 0.5`, net Sharpe `≥ 1.0`,
`DSR ≥ 0.95`, `STAGE5_MIN_FEE_BPS = 3`). Nothing here sets or relaxes a threshold.

1. **Stage 1 — on reported metrics, and labelled so.** A centred rank is unit-free and bounded by
   construction; there is no Lean model of these predictors. The tracker records the stage as
   `..._ON_REPORTED_METRICS` and `is_evidence_backed()` is therefore False for every candidate.
   This is stated, not hidden: stages 2–5 are derived from the data.
2. **Stage 2 — development period:** pooled Spearman IC, five contiguous folds, IC positive in each.
3. **Stage 3 — development period:** Diebold–Mariano against a zero forecast (the forecast is the
   least-squares projection through the origin of the target rank on the predictor rank — both are
   centred — fitted on the development period);
   literature R² of the predictor on 1-day reversal (`−r_t`) and 20-day momentum.
4. **Stage 4 — the candidate's holdout window,** claimed in a persistent experiment ledger before it
   is scored; one-sided IC test with effective sample size.
5. **Stage 5 — H1-OFI only, on the test window:** daily long-short portfolio, long the top fifth by
   prediction and short the bottom fifth, equal weights; turnover `½ Σ|Δw|`; fees at the canonical
   minimum; DSR from a persistent trial ledger holding both sign variants' development-period
   long-short Sharpe ratios; universe audited by `audit_universe` on the test window's members with
   `selection_basis="point_in_time"`. A universe the audit finds unfit (for example a single-bar move
   of 4× or more, which the audit treats as an unadjusted split) refuses the stage; there is no
   override.

CZERO-PLAC has no holdout window. If it ever reached stage 4 that is recorded as a false pass at
stages 2–3.

## What is reported, whatever it shows

For each candidate: every stage record (metrics, reason, evidence source), the highest stage passed,
whether it was halted and why. The run writes `results.json`, `promotion_journal.jsonl`,
`experiment_ledger.jsonl` and `trial_ledger.jsonl`, and they are committed with a report in
`preregistration/RESULT-CRYPTO-PIT-2026-11.md`.

**The referee behaves as designed on this data if** CPLUS-VOL passes stage 4 **and** CZERO-PLAC is
refused at stage 2 or 3. If either fails, that is the result, and it is reported as a failure of the
referee or of the data, not explained away.

**H1-OFI has no pass criterion.** Its outcome is the finding. The author's expectation, recorded so
it can be compared: refused at stage 4 or 5, in line with the README's record of development-period
market hypotheses.

## Deviation rule

The analysis runs once. If the data build fails, or the archive serves a file that does not match
its digest, the build is re-run; the analysis is not changed. If the analysis crashes on the real
panel, the fix is committed as a dated deviation in this directory, with the traceback, before it is
re-run, and the result is labelled as run under that deviation. Changing a candidate, a window, a
universe parameter or a threshold after the data exists is not a deviation: it is a new
preregistration with a new holdout, and this one's result stands as recorded.

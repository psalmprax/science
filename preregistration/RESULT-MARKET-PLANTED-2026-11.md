# Result: PREREG-MARKET-PLANTED-2026-11

**Run:** once, 2026-11-19 23:00 – 2026-11-20 00:24 UTC, GitHub Actions run `44633787508` on the
branch `market-planted-run-1` at commit `80eb686`, which froze the preregistration and the harness
(digests repeated in `results.json`). The smoke job, 32 shards and the collect job succeeded on the
first attempt; no shard was re-run. All 120 runs completed; none ended in an error or a timeout.

Outputs, unedited, in [`RESULT-MARKET-PLANTED-2026-11/`](RESULT-MARKET-PLANTED-2026-11/):
`results.json`, `runs.jsonl` and the 32 shard files, rebuilt from the collect job's log; all 34
match the SHA-256 the job computed.

**These are simulated series with a law planted by construction.** Nothing here is a finding about
a real market.

## Outcome

A law reported, out of the worlds in each row:

| Worlds | n | The main search path alone ("ungated") | The series route ("gated") | Gated and capturing the signal | Gated, in the planted form |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **No law** | 40 | **28** | **0** | – | – |
| Law explains 0.5 % of the variance | 20 | 18 | 0 | 0 | 0 |
| 2 % | 20 | 16 | 3 | 3 | 0 |
| 5 % | 20 | 14 | 14 | 11 | 4 |
| 20 % | 20 | 16 | 17 | 17 | 4 |

"Capturing the signal": the reported law explains at least half of the planted signal's variance on
the last fifth of the series, which no stage read.

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | Gated: a law in at most 2 of the 40 null worlds | 0 | **held** |
| 2 | Ungated: a law in at least 8 of the 40 null worlds | 28 | **held** |
| 3 | At 5 % and 20 %: gated reports in at least 28 of 40, and at least 80 % of those capture the signal | 31; 28 of 31 (90 %) | **held** |
| 4 | At 0.5 %: gated reports in fewer than 10 of 20 | 0 | **held** |

95 % intervals: gated on nulls 0 to 8.8 %; ungated on nulls 53 to 83 %; gated at 5 % and 20 %,
62 to 89 %; capturing among those, 74 to 98 %.

## Reported without a criterion

- **The path alone on noise.** In 28 of 40 series with nothing in them the main search path
  returned a law, each time as an equivalence class. Fourteen of those laws forecast the scoring
  period worse than the mean. At 0.5 % it "reports" in 18 of 20, which carries no information
  given what it does on the nulls.
- **The route on noise.** In 35 of the 40 null series a candidate beat the mean inside the
  development period and earned a test on the later period; none passed. The smallest p-value was
  0.049; the level is 0.01.
- **Power by signal size.** 0 of 20 at 0.5 %, 3 of 20 at 2 %, 14 of 20 at 5 %, 17 of 20 at 20 %.
  A test was earned in 79 of the 80 planted series, so the limit is the later period's 1 000 rows,
  not the choice of candidate.
- **Form.** Of the 34 laws the route reported in planted series, 31 capture the signal and 8 are
  the planted law's own form. By planted law (reported / capturing / own form, of 16 series each):
  `x0` 6 / 6 / 4; `x0*x2` 6 / 5 / 3; `tanh(2*x0)` 9 / 9 / 0; `x0*exp(-x3)` 8 / 7 / 1;
  `x1 + 0.5*x0*x2` 5 / 4 / 0.
- **Reported and worse than no law on the scoring period:** 2 of the route's 34 (one at 2 %, one at
  5 %); 6 of the path's 64.
- **Time.** Median 268 s per series.

## After the run (not preregistered)

- The route's reading is decided by one test at the 1 % level on 1 000 later rows, so its
  behaviour on nulls was nearly fixed in advance; what the run adds is that selecting a candidate
  inside the development period did not leak (0 of 40 where 0.4 would be expected), and how much
  power is left at each signal size.
- At 5 % the path alone and the route report the same number (14) and not the same series: 12 are
  reported by both, 2 only by the path and 2 only by the route. The path's own verdict was
  `widen_search` in 6 of those 20 series.
- A signal explaining 0.5 % of the variance, which is the size of a strong real effect in daily
  returns, was never confirmed. With this design it would need several times as many later rows. That is the honest reading for use on market data: at realistic signal sizes this route
  mostly declines, and when the path alone speaks it should not be believed.

## What changes

- The main search path must not be used on return-like data by itself (audit M-16).
  `apedp_dyn/discovery_l2/series_route.py` is the route for such data.
- Not addressed: regime changes, effects that fade, costs, real prices, and the low power at
  small signal sizes.

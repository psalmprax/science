# PREREG-MARKET-PLANTED-2026-11: planted laws in market-like series

**Status:** frozen before the run by `preregistration/DIGEST-MARKET-PLANTED.json`. Runs once, on
GitHub Actions (`.github/workflows/market-planted.yml`), started by the first branch named
`market-planted-run-*` whose shards start.

## Why

Everything the search path has been measured on so far is data in which a law explains nearly all
of the target. Return data is the opposite case: a real effect explains a fraction of a percent to
a few percent of the variance, the noise is heavy-tailed and its scale clusters, and in most series
there is nothing to find. No part of this repository had been measured on data like that with a
known answer. Development runs showed the main search path failing on such data in both
directions: its first pass alone reports a law in most series that contain none (ten of twelve),
because none of its checks asks whether a law predicts anything; and with PySR's candidates beside
it the confirmation step refused every candidate in seven of twenty series that did contain a law,
because it expects errors on later rows to match the development error and the noise scale here
moves. `apedp_dyn/discovery_l2/series_route.py` therefore uses the path for candidates only, chooses
one inside the development period, and puts that one to the referee's Diebold-Mariano test against
the no-law forecast on a later period. This run measures the path's own reading and the route's on
fresh worlds.

## Worlds

`scripts/market_planted_worlds.py`. 5 000 rows in time order, four inputs (a fast trend signal, a
slow spread, an order-flow imbalance, and the current log volatility), target = strength × law +
noise. Noise: GARCH(1,1) with persistence 0.98 and Student-t innovations with four degrees of
freedom. Development rows are the first 60 %, the following 20 % are the later period that
confirmation and the forecast test may read, the last 20 % are for scoring only.

- **40 null worlds**: no law.
- **80 planted worlds**: five laws (`x0`, `x0*x2`, `tanh(2*x0)`, `x0*exp(-x3)`, `x1 + 0.5*x0*x2`) ×
  four shares of the target's variance explained (0.5 %, 2 %, 5 %, 20 %) × four seeds.

Seeds are derived from this document's identifier; no world of this run has been generated.

## Protocol

Each world goes once through `series_route.discover_predictive_law` with PySR (240 s) as the search
stage, run in every world. Two readings are recorded from the same run:

- **ungated**: the law the main search path returns by its own verdict, refitted on the
  development rows;
- **gated** (the route): the path's candidates are each refitted on the first two thirds of the
  development period and scored on its last third. Those that do not beat the mean there are
  dropped; of the rest, the simplest whose error is within one standard error of the best one's is
  chosen. The chosen one is refitted on all development rows and reported
  only if its forecasts beat the development mean on the later period by the Diebold-Mariano test
  at the level in the canonical specification (p < 0.01, one-sided, squared error, Newey-West
  variance). One hypothesis is tested on the later period and nothing about that period enters
  its choice.

Scoring on the last 20 %: a law **captures the signal** if it explains at least half of the
planted signal's variance there (prediction against the noise-free signal, both centred). Symbolic
recovery is by the same check as the SRBench runs. Also recorded: whether a reported law forecasts
the scoring period worse than the development mean.

## Expectations, fixed now

1. **The route does not find laws in noise:** gated, a law is reported in at most 2 of the 40 null
   worlds.
2. **The path alone does:** ungated, a law is reported in at least 8 of the 40 null worlds.
3. **Where there is signal, the route finds it:** in the 40 worlds with 5 % or 20 % of the variance
   explained it reports a law in at least 28, and at least 80 % of the laws it reports there
   capture the signal.
4. **At the weakest signal it mostly cannot tell:** in the 20 worlds with 0.5 % it reports a law in
   fewer than 10. (The later period is 1 000 rows; an effect of that size is about two standard
   errors there, below the test's level.)
5. Reported without a criterion: every cell by law and by share (including 2 %), symbolic recovery,
   how often a candidate earned a test, what the path's own verdict was, and laws that forecast the
   scoring period worse than no law.

A miss is reported as a miss.

## Disclosed before freezing

Development used worlds from the same generator under other seed tags (`dev`, `test`, `smoke`); no
world of this run has been generated. In order:

- **First study, locally, the first pass alone (no PySR).** The gate then was the forecast test
  applied to the path's own law. 12 null worlds: the path reported a law in 10, the gate passed
  none. 40 planted worlds (five laws × four shares × two seeds).
- **With PySR in CI (60 s budget), 4 null worlds and 20 planted at 5 % and 20 %.** The path's
  confirmation step refused every candidate in 7 of the 20 planted worlds (it requires errors on
  later rows to match the development error; the noise scale here moves), so the gate saw nothing
  in those and 12 of 20 were reported. The route was changed to what is described above: the path
  supplies candidates only.
- **The same 24 worlds with the changed route, choosing the candidate with the lowest error:** 16
  of 20 reported, 15 capturing the signal, 1 in the planted law's own form; none of the 4 nulls
  (in all four a candidate earned a test and failed it).
- **The choice was then changed to the simplest candidate within one standard error of the best,**
  and replayed locally on the candidates recorded in that run: 18 of 20 reported, all 18 capturing
  the signal, 6 in the planted form; none of the 4 nulls.
- So the choice rule was picked on 24 development worlds with two variants tried, and its
  development rate (18 of 20) is optimistic for that reason. Expectation 3 is set at 28 of 40.
  Expectation 2 follows the path's own verdict on the 8 null worlds run with PySR (a law in 3).
- Development ran PySR for 60 s; this run gives it 240 s, as the other benchmarks do. More search
  means more elaborate candidates; the effect on the choice was not seen.
- With heavy-tailed noise the searches sometimes prefer a term driven by a few extreme rows
  (`exp(exp(x1))` was one). The forecast test removes these; nothing in the search prevents them.
- For `x1 + 0.5*x0*x2` the route tends to report `x1` alone, which captures half to three quarters
  of that signal; and one correct law (`x0` at 5 %) failed the forecast test outright in a world
  where the later period went against it.
- What these worlds do not contain: regime changes, an effect that fades, costs, or real prices.
  A result here says what the route does when a fixed law is present in stationary heavy-tailed
  noise, and nothing about whether such laws exist in a real market.
- The path's own confirmation step reads up to 30 targets of the later period for its verdict (the
  ungated reading). That verdict does not enter the route's choice, so the route's single test on
  the later period is not affected by it.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.

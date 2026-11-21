# PREREG-LAW-AUDIT-FINANCE-MORE-2026-11: six further published regularities of asset returns, measured on a market their authors did not study

**Status:** frozen before the run by `preregistration/DIGEST-LAW-AUDIT-FINANCE-MORE.json`. Runs
once, on GitHub Actions (`.github/workflows/law-audit-finance.yml`), in the `run` job of the first
branch named `law-audit-finance-run-*` whose `run` job starts, after the first batch
(PREREG-LAW-AUDIT-FINANCE-2026-11) has run and printed its results in the same job. **Not started
at freezing:** the run waits for the repository owner's word, as the first batch does.

## What is tested

The first batch audits five of Cont's eleven stylised facts. The owner asked whether all the audits
had been built before anything is run; they had not. This batch adds the four remaining facts of
that list that can be given a number on daily data, and two published claims about what predicts
returns, one of them made about this very market. The method is the first batch's: a claim that is
already published, one number, an interval that is honest about what five years can tell, and one
of three answers. It **fails** (the interval lies wholly outside the published set), it
**survives** (wholly inside), or the data **cannot tell** (`not_rejected`, with the interval as
the result). How often each answer is wrong was measured on simulated markets before the real one
is opened, and is part of this document.

Every claim here is a sign, so every published set is "above zero".

| # | Claim, as published | Source | Number measured here |
| :-- | :--- | :--- | :--- |
| 6 | Gain/loss asymmetry: large drawdowns, but not equally large upward movements | Cont (2001), fact 3 | of each asset's largest 5 % of daily moves, the share that are losses, minus one half (`asymmetry`) |
| 7 | Aggregational Gaussianity: over longer time scales the distribution of returns looks more and more like a normal one | Cont (2001), fact 4 | the share of large daily moves minus the share of large weekly moves (`gaussianity`) |
| 8 | Conditional heavy tails: after correcting returns for volatility clustering, the tails are less heavy than those of the returns themselves ... | Cont (2001), fact 7 | the share of large moves among the returns minus that among the returns divided by a volatility estimate from earlier days (`conditional`) |
| 8′ | ... and the residuals still have heavy tails | the same | the share of large moves after that division, minus what independent normal returns in the same cells leave after it (`still_heavy`) |
| 9 | Asymmetry in time scales: coarse-grained measures of volatility predict fine-scale volatility better than the other way round | Cont (2001), fact 11; Müller et al. (1997) | rank correlation of a week's absolute net move with the next week's mean absolute daily move, minus that of a week's mean absolute daily move with the next week's absolute net move (`cascade`) |
| 10 | Momentum across coins: coins that rose more over the last one, two, three or four weeks earn more in the following week (about 3 % a week between the top and bottom fifth) | Liu, Tsyvinski and Wu (2022) | rank correlation, across assets, of the return of the last *k* weeks with the return of the next week, averaged over *k* = 1 to 4 (`momentum`) |
| 11 | Order imbalance: a day's order imbalance is positively related to the next day's return | Chordia and Subrahmanyam (2004) | rank correlation of a day's taker-buy imbalance with the next day's return (`flow`) |

**Sources.** R. Cont, "Empirical properties of asset returns: stylized facts and statistical
issues", *Quantitative Finance* 1 (2001) 223–236. U. A. Müller, M. M. Dacorogna, R. D. Davé, R. B.
Olsen, O. V. Pictet, J. E. von Weizsäcker, "Volatilities of different time resolutions: analyzing
the dynamics of market components", *Journal of Empirical Finance* 4 (1997) 213–239. Y. Liu, A.
Tsyvinski, X. Wu, "Common risk factors in cryptocurrency", *Journal of Finance* 77 (2022) 1133–1177
(coins above one million dollars of capitalisation, 2014 to 2018 in the working paper, which is
where the three per cent a week is taken from). T. Chordia, A. Subrahmanyam, "Order imbalance and
individual stock returns: theory and evidence", *Journal of Financial Economics* 72 (2004) 485–518
(NYSE stocks, 1988 to 1998; the relation is positive, and reverses sign once the same day's
imbalance is controlled for).

**What the literature already says against some of them**, known when this was written: later
work on Cont's facts in modern equity data found no gain/loss asymmetry; cryptocurrencies are
reported to have positively skewed returns; momentum across coins is reported by some studies and denied or
reversed by others, with the positive findings concentrated in the largest coins, which is what
this panel holds; and work on this exchange's own taker data reports short-horizon reversal after
moves that went with the flow. None of that was measured on this panel in this form.

**Left out, and why.** Fact 5 (intermittency) has no agreed number. Fact 6 (volatility
clustering) is what the first batch's control measures. The Taylor effect (absolute returns are
more autocorrelated than squared ones) was considered and dropped: no simulated world at hand
breaks it by an amount this audit could catch, and a test that cannot catch a violation says
nothing.

## Data and protocol

**The panel** is the first batch's, read once more: `crypto_usdt_daily/panel_development.csv.gz`
on branch `market-data` at commit `a598b83c851670bc7f95a29459631fca3b91936e`, SHA-256
`96b5ee0b88c4dd5c004565f2582c8683b440513ad5dc9f8d511a0fb49fa4644e`, 2021-01-01 to 2025-12-31. The
runner refuses any other file. Besides closes and volumes this batch reads the taker-buy volume
of each bar, which the exchange's archive records and the panel's builder keeps.

**The holdout file of that branch is not opened**, by the runner or by the workflow.

**Cells.** The first batch's: log returns between two consecutive daily closes, the first 5 of an
asset dropped, assets with at least 250 returns, each asset's returns normalised by its own mean
and standard deviation (`apedp_dyn/market/law_audit.py`, frozen; a test holds the two batches to
the same cells).

**The numbers** (`apedp_dyn/market/law_audit_more.py`; every setting is a constant there):

- *A large move* is one whose size exceeds 4 times the median size of the same asset's moves. For
  normal returns that is 0.7 % of the moves; for returns with a tail index of three, 4.4 %. The
  median is taken within the cells that share a remainder of the day number modulo 7, so that a
  daily and a weekly share rest on medians of equally many values.
- `asymmetry`: the tail cells are the first batch's (an asset's absolute returns above its own
  95 % quantile). The number is the share of them with a negative return, minus one half.
- `gaussianity`: the share of large moves among daily returns, minus the share among sums of 7
  consecutive daily returns (every start day; a window needs all 7 days). Reported beside it,
  without a verdict: the same for 28 days.
- `conditional` and `still_heavy`: the volatility estimate of a day is the exponentially weighted
  geometric mean of the asset's absolute returns on earlier days (half-life 20 days, at least 20
  of them; an absolute return below 0.05 standard deviations counts as 0.05). `conditional` is the
  share of large moves among the returns, in the cells that have an estimate, minus the share
  among the returns divided by their estimates. `still_heavy` is the second of those shares minus
  the share that independent normal returns in the same cells leave after the same division
  (three seeded draws, computed by the runner).
- `cascade`: for every start day with 14 consecutive days, the first week's coarse volatility is
  the absolute sum of its returns and its fine volatility the mean of their absolute values, and
  likewise for the second week. Each is ranked within the asset. The number is the correlation of
  coarse (first week) with fine (second week) minus that of fine (first) with coarse (second).
- `momentum`: for every start day, the return of the next 7 days and of the last 7, 14, 21 and 28
  days (raw log returns; an asset needs every day of both windows), each ranked across the assets
  that have both (at least 20). The number is the mean of the four rank correlations, each pooled
  over days. No day is skipped between the two windows, as in the source.
- `flow`: the imbalance of a day is the taker-buy volume minus the taker-sell volume over the
  volume. It and the returns are ranked within the asset. The number is the correlation of the
  imbalance with the next day's return.

**Intervals.** The first batch's: sums per block of 90 consecutive days, 4,000 resamples of whole
blocks, the central 99 % of the resampled numbers, widened about the estimate by a factor measured
on simulated panels (next section). A window of several days is counted under its first day.
Medians and ranks within an asset are computed once on the whole panel. For the shares of large
moves that is corrected to first order in every resample: the share of cells below their median,
minus one half, times a slope counted once on the panel, is added to the share. Without that
correction the interval of `conditional` is several times too wide, because a resample that draws
more turbulent blocks counts more large moves against medians that should have risen with it.

**Verdicts.** The first batch's rule and level: `fails` if the widened interval lies wholly below
zero, `survives` if wholly above, otherwise `not_rejected`. The level is the canonical
`DIEBOLD_MARIANO_P_VALUE_THRESHOLD`, 0.01, two-sided.

**Reported without a criterion:** the 28-day `gaussianity`; the shares of large moves themselves;
the two correlations behind `cascade`; `momentum` for each of the four look-backs; the same-day
correlation of imbalance and return, the lag-one rank autocorrelation of returns, and the partial
correlation of imbalance with the next day's return given the same day's return; every number for
each stretch of 365 days; and the verdicts and numbers for the first batch's six parts of the panel
(first three years, last two, assets that stopped trading, assets still trading, the larger and
the smaller half by typical traded value).

## How wide the intervals have to be

Seven simulated worlds in which every number keeps one long-run value from panel to panel
(`apedp_dyn/market/law_audit_more_worlds.py`, `CALIBRATION_WORLDS`): the first batch's world in
which its five laws hold, with none of this batch's dials on; the same with a tail index of 2.2
and of 4, and with a memory exponent of 0.2 and of 0.4; a world of stable-law returns (index
1.7); and a world of independent normal returns. 300 panels each, of the real panel's shape:
2,100 panels, seed 155,000. For each number, the long-run value is its mean over a world's
panels; the factor is the largest widening any world needs for the interval to cover it in 99 %
of the panels, rounded up to a tenth.

| Claim | The unwidened interval covered the long-run value in | Widening needed | Factor used |
| :--- | :--- | :--- | :--- |
| `asymmetry` | 98.3 to 100.0 % | 1.10 | 1.2 |
| `gaussianity` | 96.0 to 98.7 % | 1.45 | 1.5 |
| `conditional` | 88.7 to 98.7 % | 1.80 | 1.8 |
| `still_heavy` | 96.7 to 98.0 % | 1.58 | 1.6 |
| `cascade` | 96.3 to 98.3 % | 1.27 | 1.3 |
| `momentum` | 96.3 to 99.0 % | 1.42 | 1.5 |
| `flow` | 96.7 to 97.7 % | 1.28 | 1.3 |

The interval from the block resampling alone covers 89 to 100 % of the time where it claims
99 %. In these worlds `asymmetry`, `cascade`, `momentum` and `flow` are zero in the long run (the
largest mean over 300 panels is 0.0029, for `asymmetry`), which is the boundary of each claim and so
where a verdict is easiest to get wrong; `gaussianity` is zero in the stable-law and normal
worlds, and `still_heavy` in the normal one. Worlds with the dials on are not used for the
factors, because there the size of a number depends on how turbulent the five years are (see the
disclosures), and one panel's interval is not about the mean over panels.

The rows of every panel are in `preregistration/POWER-LAW-AUDIT-FINANCE-MORE-2026-11/`
(`calibration_rows.jsonl.gz`, `calibration.json`); a test recomputes the table from them and holds
`STRETCH` to it.

## What the audit can and cannot tell

Sixteen simulated worlds (`WORLDS`), 150 panels each of the real panel's shape, none used for the
factors: 2,400 panels, seed 177,000. A claim counts as true in a world by the dial that makes it so.
"Not true" covers both the worlds where the number is zero and those where it is reversed.

| Claim | Where it is true | Where it is not true |
| :--- | :--- | :--- |
| Large moves are more often losses (`asymmetry`) | failed in 1 of 600; survived in 246 | survived in 1 of 1,800; failed in 144 |
| Closer to normal over a week than a day (`gaussianity`) | failed in 0 of 1,950; survived in 1,922 | survived in 4 of 450; failed in 0 |
| Tails less heavy after the volatility correction (`conditional`) | failed in 0 of 1,800; survived in 1,640 | survived in 0 of 600; failed in 599 |
| Tails still heavy after it (`still_heavy`) | failed in 0 of 2,250; survived in 2,250 | survived in 0 of 150; failed in 0 |
| Coarse volatility predicts fine better than the reverse (`cascade`) | failed in 0 of 600; survived in 234 | survived in 4 of 1,800; failed in 226 |
| Momentum across assets (`momentum`) | failed in 0 of 750; survived in 457 | survived in 1 of 1,650; failed in 210 |
| Order imbalance and the next day (`flow`) | failed in 0 of 600; survived in 533 | survived in 10 of 1,800; failed in 257 |

1 of the 2,400 panels failed some claim that was true in their world, and 20 passed some claim
that was not.

What each claim's audit can tell, by the size of the effect (the number's mean over the world's
panels in brackets):

- **`asymmetry`.** Losses 10 % larger and gains 10 % smaller (+0.063: 56 % of large moves are
  losses) survives in 78 of 150; half that (+0.032) in 8 of 150. The reverse
  (-0.076) fails in 125 of 150; half of it in 17 of 150. The interval is about
  0.06 either side: a share of losses of 56 or 57 % is told from one half in half to four fifths
  of the panels, 53 % hardly ever.
- **`gaussianity`.** Survives in 1,922 of the 1,950 panels with heavy tails of finite variance or
  moving volatility. In the two stable-law worlds, where the claim is false, the number is zero
  to three decimals and survives in 4 of 300 panels; it never fails there, because a stable law
  gives zero, not a negative number. So this audit separates Cont's claim from Mandelbrot's
  stable hypothesis by "survives" against "not rejected".
- **`conditional`.** Where volatility moves the number is small (+0.008 in the first batch's world)
  and survives in 1,640 of 1,800 panels; where it does not move, dividing by the estimate only adds
  noise, the number is -0.003, and the claim fails in 599 of 600.
- **`still_heavy`.** Survives in all 2,250 panels where it is true; for independent normal returns
  it is not rejected in 150 of 150.
- **`cascade`.** Volatility raised by last week's net move (+0.029) survives in 80 of 150; half
  that strength (+0.016) in 27 of 150. The reverse (-0.055) fails in 150 of 150; half of it in
  71 of 150. The interval is about 0.025 either side.
- **`momentum`.** A rank correlation of +0.071 survives in 150 of 150. (Where weekly returns differ
  across coins by about 15 %, three per cent a week between the top and bottom fifth corresponds to
  a rank correlation near 0.07; the source's own spread of returns was not checked.) A third of that
  (+0.020) survives in 87 of 150; +0.004 in 4 of 150. Reversal of -0.045 fails in 149 of 150; of
  -0.012 in 59 of 150. The interval is about 0.015 either side: momentum of the published size
  cannot be missed, and a tenth of it cannot be seen.
- **`flow`.** A correlation of +0.034 survives in 142 of 150; +0.017 in 101 of 150. The reverse
  (-0.031) fails in 148 of 150; half of it in 101 of 150. Of the 10 panels that survived where
  the dial is off, 5 are in the world with strong momentum, where the number is +0.004: a day's return
  there does say something about the next, and the imbalance goes with the return. That is the
  audit reading a real dependence that is not order flow's own, as disclosed below.

Where the four dialled numbers are zero (nine worlds), each verdict that is not `not_rejected` is
an error: `asymmetry` failed in 2 and survived in 1 of 1,350; `cascade` failed in 5 and survived
in 4; `momentum` failed in 2 and survived in 1; `flow` failed in 8 and survived in 5. That is at
most 1 % of the panels for each. In the worlds with leverage `cascade` is slightly below zero in
the long run (-0.002); three of its five failures are there.

The rows of every panel are in the same directory (`power_rows.jsonl.gz`, `summary.json`).

## Expectations, fixed now

1. **The checks of the machinery hold:** `still_heavy` survives (a market's returns are not
   independent normal draws), and the same-day rank correlation of imbalance and return is above
   0.1 (takers buying goes with prices rising; if it does not, the taker column is not what the
   builder says it is).
2. `gaussianity` survives.
3. `conditional` survives, with a small number.

The author's own expectations, written here so that they can be wrong in public. They carry no
consequence:

4. `asymmetry` is not rejected, with an estimate within 0.05 of zero: on this market large gains
   are about as frequent as large losses.
5. `cascade` is not rejected.
6. `momentum` does not survive: it fails or is not rejected.
7. `flow` fails: a day whose buyers were the aggressors is followed by a lower return, not a
   higher one.

## What follows from the result, fixed now

- If `still_heavy` does not survive, the run is recorded as a failed audit of the audit. No claim
  of this batch gets a verdict, and the harness goes back to simulated data. If only the taker
  column's check misses, `flow` alone gets no verdict.
- A claim that **fails** is recorded as failing on this market, with its interval. Within seven
  days it is filed as a dated claim in the repository's prediction ledger about the same number,
  by the same frozen code, on bars from 2026-11-01 on (which did not exist when this was frozen),
  with the bound and the due date fixed at filing.
- A claim that **survives** is filed as a dated claim in the same way, and becomes a control for
  later searches on this panel.
- A claim that is **not rejected** is recorded with its interval as "this panel cannot tell closer
  than this". It is not counted as confirmed and no claim is filed.
- Nothing here is a trading claim. `momentum` and `flow` are rank correlations before any cost,
  over every asset of the panel whatever its liquidity. No number goes to the promotion referee,
  and no default of a discovery route changes with the result.

## Disclosed before freezing

- **Designed on simulated panels only.** No price, volume or taker volume of the real panel was
  read while this was built: only its manifest, for the first and last bar of each asset. The
  first batch had not been run when this was frozen, so no number of it could inform this one.
- **The panel has been used before in this repository, and one result bears on two claims here.**
  PREREG-CRYPTO-CAL found one-day reversal among the 30 largest pairs (rank correlation 0.029,
  every fold positive). A day's imbalance goes with the same day's return, so a one-day reversal
  makes a negative `flow` likely; it is the reason for expectation 7 and part of the reason for
  expectation 6. PREREG-CRYPTO-PIT found that the mean absolute return of the last 20 days
  predicts the next day's (rank correlation 0.335), which is why `conditional` is expected to
  survive. Neither computed any of these seven numbers.
- **Order flow has been used in this repository's trading sandbox before,** as a feature on an
  earlier universe of fifteen pairs (`apedp_dyn/sandbox/cpmd_hlik_stage3.py`,
  `apedp_dyn/sandbox/openai_math_advanced_hypotheses.py`). The author of this audit did not look
  at those results, and what the owner saw in them is not known to the author.
- **The estimators changed during design, on simulated panels.** The volatility estimate was
  first a 20-day mean of absolute returns: with heavy-tailed shocks it is so noisy that dividing by
  it removed almost nothing. Medians within each block of 90 days were tried so that the
  resampling would be exact: then no effect is left to measure, because the block's own median
  already carries what the estimate knows. The first-order correction of the medians was the
  third design. The multiple of four was chosen among three, four and five, which gave the same
  picture.
- **The widening factors are as good as the simulated worlds are like a market.** The worlds for
  `cascade`, `momentum` and `flow` each have one mechanism per claim, acting on each asset alone:
  a wandering expected return, a fading price component, volatility that follows last week's net
  move, order flow whose surprise carries into the next day. Real markets need not produce these
  numbers that way.
- **An interval here is about these five years.** In the simulated worlds where momentum or flow
  is present, its size depends on how turbulent the period is, and the number varies between
  five-year panels by about twice what the interval inside one panel says. The sign does not. A
  verdict is about the sign; the estimate is not a long-run constant.
- **`flow` contains whatever a day's return says about the next.** The imbalance goes with the
  same day's return, so reversal or continuation of returns shows up in it. The partial
  correlation is reported beside it. The source's own finding is of this kind: positive, and of
  the opposite sign once the same day's imbalance is held fixed.
- **Windows overlap.** Weekly windows start on every day. The block resampling carries that
  dependence inside a block and ignores it across block edges.
- **Fourteen verdicts in the two batches, each at the 1 % level.** They are not corrected for
  their number. How often a simulated panel fails some claim that is true in it is in the section
  on what the audit can tell.
- **`still_heavy` is nearly a check, not a discovery.** It can fail to survive only for returns
  close to independent normal draws.
- **Not comparable with the sources' own magnitudes.** The momentum claim is audited as a rank
  correlation over this panel's assets, not as the return of a portfolio of fifths over a universe
  of more than a thousand coins; the order-imbalance claim on taker flow of spot pairs, not on
  signed trades of NYSE stocks.

## Deviation rule

The run is not repeated. If the `run` job ends before this batch's `results.json` is written, for
a reason of infrastructure (the download, the installation), this batch's step is started again
and recorded with its date. Any change to a frozen file after the `run` job has started makes the
result a development result. If the repository owner asks for a change before the run starts,
this preregistration is superseded by a new one with a dated note here; nothing is edited in
place.

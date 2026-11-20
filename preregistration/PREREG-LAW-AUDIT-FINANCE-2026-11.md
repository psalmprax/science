# PREREG-LAW-AUDIT-FINANCE-2026-11: five published laws of asset returns, measured on a market their authors did not study

**Status:** frozen before the run by `preregistration/DIGEST-LAW-AUDIT-FINANCE.json`. Runs once, on
GitHub Actions (`.github/workflows/law-audit-finance.yml`), started by the first branch named
`law-audit-finance-run-*` whose `run` job starts. **Not started at freezing:** the run waits for
the repository owner's word, because it cannot be repeated and the choices below are theirs to
question first.

## What is tested

Everything this repository has measured so far is a benchmark whose answer is known. This is the
first step towards a finding on real data, and it is deliberately not a search: take laws that are
already published, state each as one number with the range its authors gave, measure that number
on data they never saw, and say one of three things about each. It **fails** (the interval lies
wholly outside the published set), it **survives** (wholly inside), or the data **cannot tell**
(`not_rejected`, with the interval as the result). How often each of those answers is wrong was
measured on simulated markets before the real one was opened, and is part of this document.

The five laws are the five of Cont's eleven stylised facts that are stated as a number or a sign
and can be measured from daily closes and volumes:

| # | Law, as published | Source | Number measured here | Published set |
| :-- | :--- | :--- | :--- | :--- |
| 1 | Returns have power-law tails, with a tail index above two and below five | Cont (2001), fact 2 | Hill tail index over each asset's largest 5 % of absolute returns (`tail`) | (2, 5) |
| 1′ | The strict reading, the inverse cubic law: exponents 3.10 ± 0.03 and 2.84 ± 0.12 for the two tails, from two to a hundred standard deviations | Gopikrishnan, Meyer, Amaral, Stanley (1998) | the same number (`tail_cubic`) | [2.72, 3.13] |
| 2 | Returns have no linear autocorrelation beyond very short intraday horizons | Cont (2001), fact 1 | variance of 5-day returns over five times that of 1-day returns (`random`) | {1} |
| 3 | The autocorrelation of absolute returns decays slowly, roughly as a power law of the lag with exponent between 0.2 and 0.4 | Cont (2001), fact 8 | decay exponent of the autocorrelation of absolute returns, lags 1 to 40 days (`memory`) | [0.2, 0.4] |
| 4 | Leverage: volatility is negatively correlated with the asset's earlier returns | Cont (2001), fact 9; Bouchaud, Matacz, Potters (2001) | correlation of a day's return with the absolute returns of the next 10 days (`leverage`) | below 0 |
| 5 | Trading volume is positively correlated with volatility | Cont (2001), fact 10; Karpoff (1987) | same-day correlation of absolute return with volume relative to the previous 30 days (`volume`) | above 0 |
| C | Control, known to be false: returns are independent | (a random walk with independent steps) | lag-1 autocorrelation of absolute returns (`independent`) | {0} |

Law 5 is expected to hold on any market and serves as the positive control; C is the negative one.

References: R. Cont, *Empirical properties of asset returns: stylized facts and statistical
issues*, Quantitative Finance 1 (2001) 223–236. P. Gopikrishnan, M. Meyer, L. A. N. Amaral,
H. E. Stanley, *Inverse cubic law for the distribution of stock price variations*, Eur. Phys. J. B
3 (1998) 139–140, arXiv:cond-mat/9803374 (five-minute returns of the 1000 largest US companies,
1994–95, each normalised by its own volatility). J.-P. Bouchaud, A. Matacz, M. Potters, *Leverage
effect in financial markets: the retarded volatility model*, Phys. Rev. Lett. 87 (2001) 228701,
arXiv:cond-mat/0101120. J. M. Karpoff, *The relation between price changes and trading volume: a
survey*, J. Financial and Quantitative Analysis 22 (1987) 109–126.

## Data and protocol

**The panel.** `crypto_usdt_daily/panel_development.csv.gz` on branch `market-data` at commit
`a598b83c851670bc7f95a29459631fca3b91936e`, SHA-256
`96b5ee0b88c4dd5c004565f2582c8683b440513ad5dc9f8d511a0fb49fa4644e`: daily bars of every Binance
spot pair quoted in USDT, pairs that stopped trading included, pegged and leveraged tokens
excluded, built from the exchange's public archive by `scripts/build_point_in_time_crypto_panel.py`.
The period is 2021-01-01 to 2025-12-31: 559 assets with a bar, 606,434 asset-days (from the
panel's manifest). The runner refuses any other file.

**The holdout file of that branch is not opened**, by the runner or by the workflow. Its test
window (2026-02-01 to 2026-10-31) has never been scored and stays that way for a later claim.

**Returns.** Log returns between two consecutive daily closes. The first 5 returns after an
asset's first bar are dropped. An asset enters with at least 250 returns. Each asset's returns are
normalised by its own mean and standard deviation over the period.

**The numbers** (`apedp_dyn/market/law_audit.py`; every setting is a constant there):

- `tail`: the Hill estimator pooled over all cells in which the absolute normalised return exceeds
  that asset's own 95 % quantile: the number of such cells over the sum of the logarithms of the
  return over the threshold.
- `random`: `1 + 2 Σ (1 − k/5) ρ_k` over lags 1 to 4, with `ρ_k` the pooled autocorrelation of
  returns clipped at 5 standard deviations.
- `memory`: absolute returns are replaced by their rank within the asset. The variogram of that
  series (half the mean squared difference at a given lag, pooled over assets) is computed for
  lags 1 to 40, and the number is the `β` for which `level − scale × lag^(−β)` fits it best in
  least squares, level and scale free, `β` on a grid from −0.5 to 2.5 in steps of 0.01.
- `leverage`: the mean, over lags 1 to 10, of the pooled correlation between the clipped return of
  a day and the rank of the absolute return that many days later.
- `volume`: the pooled correlation between the within-asset ranks of the absolute return and of
  the relative volume of the same day (log volume minus the mean log volume of the 30 days before,
  at least 20 of them present).
- `independent`: the pooled lag-1 autocorrelation of the rank of the absolute return.

**Intervals.** All sums are kept per block of 90 consecutive days; a lagged pair counts only if
both of its days lie in one block. Whole blocks are resampled 4,000 times (seed 20261120) and the
central 99 % of the resampled numbers taken (the level is the referee's
`DIEBOLD_MARIANO_P_VALUE_THRESHOLD`, 0.01). Each side of that interval is then moved away from the
estimate by the factor given under "How wide the intervals have to be". If a tenth of the
resamples give no number there is no interval, and the verdict is `not_rejected`.

**Verdicts.** `fails` if the widened interval lies wholly outside the published set; `survives` if
wholly inside (impossible for the two single-value sets); `not_rejected` otherwise.

**Two form tests**, reported beside the verdicts and with no bearing on them. Shapes are fitted on
2021–2023 and scored on 2024–2025: a power-law tail above each asset's threshold against an
exponential one (mean difference in log-likelihood per tail cell), and a power-law decay of the
variogram against an exponential decay
(difference in the share of the variogram's variation over lags that each explains). Each favours
the published form, the rival, or neither, by whether the 99 % interval over resampled blocks of
2024–2025 excludes zero.

**Reported without a criterion:** the tail index of each sign; the decay exponent measured on
absolute returns instead of their ranks; every number for each year; the variogram, the leverage
curve and the autocorrelations by lag; and all verdicts again for six parts of the panel
(2021–2023, 2024–2025, assets that stopped trading more than 30 days before the end, the rest, and
the larger and smaller half by median traded value).

## How wide the intervals have to be

Returns of many assets move together on a few days, and volatility wanders over years. On
simulated panels with both properties (`apedp_dyn/market/law_audit_worlds.py`, on the real panel's
shape: each asset's first and last bar from the manifest, no price), the 99 % interval of the
block resampling covers the number's long-run value less often than 99 % for five of the six
numbers. The factors were measured on 1,500 such panels, 300 in each of five worlds built so that
every law holds: one at the middle of the published ranges and four towards their edges, two of
which sit on an edge (the tail index reads 2.0 in one, the decay exponent 0.40 in another). A
number's long-run value is its mean over the 300 panels
(`preregistration/POWER-LAW-AUDIT-FINANCE-2026-11/calibration.json`):

| Number | Long-run value in the five worlds | Spread over panels (sd) | Covered by the unwidened interval | Widening needed for 99 % | Factor used |
| :--- | :--- | :--- | :--- | :--- | ---: |
| `tail` | 2.32, 1.99, 2.51, 2.37, 2.29 | 0.068 to 0.083 | 91.7 % to 98.0 % | 1.20 to 2.25 | 2.3 |
| `random` | 0.998, 0.998, 1.001, 0.997, 0.996 | 0.024 to 0.026 | 95.7 % to 98.0 % | 1.02 to 1.42 | 1.5 |
| `memory` | 0.31, 0.29, 0.30, 0.21, 0.40 | 0.125 to 0.141 | 95.7 % to 97.3 % | 1.16 to 1.26 | 1.3 |
| `leverage` | -0.018, -0.016, -0.020, -0.017, -0.019 | 0.004 to 0.005 | 96.7 % to 98.7 % | 1.06 to 1.27 | 1.3 |
| `volume` | 0.266, 0.256, 0.273, 0.242, 0.286 | 0.005 to 0.006 | 98.0 % to 99.7 % | 0.83 to 1.03 | 1.1 |
| `independent` | 0.169, 0.156, 0.179, 0.147, 0.182 | 0.010 to 0.012 | 96.0 % to 98.7 % | 1.02 to 1.29 | 1.3 |

The factor is the largest need over the five worlds, rounded up to a tenth. The largest, 2.3 for
the tail index, comes from the world with the heaviest tails. The likely reason, not measured
separately: a few days on which every asset moves carry a large share of all tail cells there, and
resampling 20 blocks does not reproduce how much the estimate depends on whether such a day
occurred.

These factors are as good as the simulated worlds are like a market. If real dependence is
stronger than simulated, the intervals are still too narrow.

## What the audit can and cannot tell

2,700 further simulated panels, 150 in each of 18 worlds, none used for the factors
(`preregistration/POWER-LAW-AUDIT-FINANCE-2026-11/summary.json`). Each cell gives the number's
long-run value in that world, then how many of the 150 panels gave `fails` / `survives` /
`not_rejected`. **Bold** marks a claim that is false in that world, so that `fails` is the right
answer there; elsewhere `fails` is an error. For the tail index and the decay exponent a claim
counts as true where the number's long-run value (its mean over the 150 panels) lies in the
published set, because the number is what the claim is about: a Hill index over the largest 5 %
reads lower than the limiting exponent (2.3 in the world whose shocks have exponent 3). The decay
exponent's claim is also false wherever the decay is not a power law.

| World | `tail` | `tail_cubic` | `random` | `memory` | `leverage` | `volume` | `independent` | A true law failed |
| :--- | :---  | :---  | :---  | :---  | :---  | :---  | :--- | ---: |
| `all_hold` | 2.32: 0/51/99 | **2.32: 35/0/115** | 0.996: 1/0/149 | 0.30: 0/0/150 | -0.018: 0/126/24 | 0.266: 0/150/0 | **0.170: 150/0/0** | 1 |
| `constant_volatility` | 2.95: 0/115/35 | 2.95: 0/0/150 | 0.995: 1/0/149 | **0.80: 2/0/148** | **0.000: 0/0/150** | **0.000: 0/0/150** | -0.001: 1/0/149 | 1 |
| `inverse_leverage` | 2.32: 0/61/89 | **2.32: 37/0/113** | 0.999: 0/0/150 | 0.31: 1/0/149 | **0.019: 130/0/20** | 0.266: 0/150/0 | **0.169: 150/0/0** | 1 |
| `memory_beta_0.1` | 2.47: 0/78/72 | **2.47: 0/0/150** | 0.997: 1/0/149 | **0.09: 1/0/149** | -0.015: 0/124/26 | 0.208: 0/150/0 | **0.099: 150/0/0** | 1 |
| `memory_beta_0.2` | 2.37: 0/72/78 | **2.37: 8/0/142** | 0.995: 1/0/149 | 0.22: 1/0/149 | -0.017: 0/129/21 | 0.241: 0/150/0 | **0.147: 150/0/0** | 2 |
| `memory_beta_0.4` | 2.29: 0/43/107 | **2.29: 71/0/79** | 0.997: 0/0/150 | **0.42: 0/0/150** | -0.019: 0/137/13 | 0.286: 0/150/0 | **0.183: 150/0/0** | 0 |
| `memory_beta_0.7` | 2.25: 0/26/124 | **2.25: 121/0/29** | 0.994: 1/0/149 | **0.71: 40/0/110** | -0.018: 0/140/10 | 0.319: 0/150/0 | **0.196: 150/0/0** | 1 |
| `memory_exponential` | 2.26: 0/35/115 | **2.26: 92/0/58** | 0.996: 0/0/150 | **0.16: 1/0/149** | -0.024: 0/138/12 | 0.292: 0/150/0 | **0.247: 150/0/0** | 0 |
| `momentum_0.03` | 2.32: 0/47/103 | **2.32: 40/0/110** | **1.040: 2/0/148** | 0.30: 0/0/150 | -0.019: 0/127/23 | 0.266: 0/150/0 | **0.170: 150/0/0** | 0 |
| `momentum_0.06` | 2.32: 0/53/97 | **2.32: 37/0/113** | **1.084: 68/0/82** | 0.33: 0/0/150 | -0.019: 0/136/14 | 0.266: 0/150/0 | **0.174: 150/0/0** | 0 |
| `no_leverage` | 2.31: 0/40/110 | **2.31: 41/0/109** | 0.994: 0/0/150 | 0.30: 0/0/150 | **-0.000: 0/1/149** | 0.266: 0/150/0 | **0.172: 150/0/0** | 0 |
| `no_volume_link` | 2.32: 0/56/94 | **2.32: 42/0/108** | 0.996: 0/0/150 | 0.30: 1/0/149 | -0.018: 0/133/17 | **-0.000: 1/1/148** | **0.171: 150/0/0** | 1 |
| `reversal_0.03` | 2.31: 0/48/102 | **2.31: 41/0/109** | **0.951: 17/0/133** | 0.32: 0/0/150 | -0.017: 0/126/24 | 0.267: 0/150/0 | **0.172: 150/0/0** | 0 |
| `reversal_0.06` | 2.32: 0/56/94 | **2.32: 30/0/120** | **0.904: 102/0/48** | 0.34: 0/0/150 | -0.016: 0/125/25 | 0.267: 0/150/0 | **0.175: 150/0/0** | 0 |
| `tail_1.7` | **1.66: 12/0/138** | **1.66: 150/0/0** | 0.997: 1/0/149 | 0.29: 0/0/150 | -0.012: 0/80/70 | 0.237: 0/150/0 | **0.134: 150/0/0** | 1 |
| `tail_2.2` | **2.00: 0/1/149** | **2.00: 150/0/0** | 0.994: 1/0/149 | 0.31: 0/0/150 | -0.016: 0/105/45 | 0.256: 0/150/0 | **0.156: 150/0/0** | 1 |
| `tail_4` | 2.53: 0/127/23 | **2.53: 0/0/150** | 0.994: 0/0/150 | 0.31: 0/0/150 | -0.019: 0/134/16 | 0.273: 0/150/0 | **0.177: 150/0/0** | 0 |
| `tail_6` | 2.68: 0/144/6 | **2.68: 0/0/150** | 0.999: 0/0/150 | 0.31: 0/0/150 | -0.021: 0/138/12 | 0.278: 0/150/0 | **0.184: 150/0/0** | 0 |

In words:

- **A true law is rarely failed.** `tail` in 0 of the 2,400 panels in which it holds, `random` in
  7 of 2,100, `memory` in 3 of 1,950, `leverage` in 0 of 2,250, `volume` in 0 of 2,400. In all, 10
  of the 2,700 panels (0.4 %) failed some law that was true in their world.
- **The controls work.** `independent` fails in all 2,550 panels of the 17 worlds whose volatility
  moves, and in 1 of 150 where returns are independent. `volume` survives in all 2,400 panels
  where volume follows volatility, and in 1 of 150 where it does not.
- **Leverage: the sign can be told.** A correlation of −0.02 survives in 1,898 of 2,250 panels
  (84 %) and never fails. One of +0.02 fails in 130 of 150. None at all is `not_rejected` in 149
  of 150.
- **The random walk: a variance ratio about 0.09 from one is caught in half to two thirds of
  panels** (68 and 102 of 150), **one about 0.05 from one rarely** (2 and 17 of 150).
- **The tail index: this panel cannot tell a reading of 1.7 from 2.** Law 1 fails in 12 of 150
  panels of the world that reads 1.66. It survives where the reading is 2.5 or more (127 and 144
  of 150) and in a third of the panels of the central world (51 of 150), and is `not_rejected`
  otherwise. The strict reading 1′ fails in every panel where the number reads 2.0 or less, in a
  fifth to four fifths of the panels where it reads 2.25 to 2.32 (30 to 121 of 150), in 8 of 150
  at 2.37, and in none from 2.47 up.
- **The decay exponent: five years of daily data cannot measure it to the published precision.**
  Its interval is ±0.4, twice as wide as the published range. A world that reads 0.71 fails in 40
  of 150 panels; one that reads 0.09, and one whose decay is exponential with a five-day half-life
  (and reads 0.16), in 1 of 150 each. The form test prefers the exponential decay in 20 of those
  150 panels, and prefers it in at most 2 of 150 where the decay is a power law; it favours the
  power law in 20 to 45 of 150 where that is true, and otherwise does not decide.
- **The tail form test has no simulated world to be wrong in.** Every world has power-law tails,
  and the test favours them in 2,632 of 2,700 panels and never the exponential. That it can favour
  an exponential tail is shown only on Laplace returns (`tests/test_market_law_audit.py`).

## Expectations, fixed now

1. **The controls behave:** `independent` gives `fails` and `volume` gives `survives`. If either
   does not, no verdict on any law is issued from this run.

The author's own expectations, recorded so that they can be wrong. None is a criterion.

2. `tail`: between 2.0 and 2.6, so that law 1 is `not_rejected` or `survives`, and the estimate
   lies below the inverse cubic range.
3. `random`: below 1 (short-term reversal). Whether the interval excludes 1 is not predicted.
4. `memory`: `not_rejected`, with an estimate between 0.1 and 0.6. The test cannot tell more.
5. `leverage`: **positive**, the opposite sign to the law. On this market large gains are
   expected to be followed by at least as much volatility as large losses.
6. `volume`: above 0.2.
7. Form tests: the power-law tail is favoured over the exponential one; the decay of the
   autocorrelation is not decided.

A miss is reported as a miss.

## What follows from the result, fixed now

- If expectation 1 misses, the run is recorded as a failed audit of the audit. No law gets a
  verdict, and the harness goes back to simulated data.
- A law that **fails** is recorded as failing on this market, with its interval. Within seven days
  it is filed as a dated claim in the repository's prediction ledger about the same number, by the
  same frozen code, on bars from 2026-11-01 on (which did not exist when this was frozen), with
  the bound and the due date fixed at filing. It becomes the starting point of a search for what
  holds instead, under its own preregistration.
- A law that **survives** becomes a control for later searches on this panel, as law 5 is here,
  and is filed as a dated claim in the same way.
- A law that is **not rejected** is recorded with its interval as "this panel cannot tell closer
  than this". It is not counted as confirmed and no claim is filed.
- The strict reading 1′ is treated as a law of its own in the three points above.
- Nothing here is a trading claim. No number goes to the promotion referee, and no default of a
  discovery route changes with the result.

## Disclosed before freezing

- **Designed on simulated panels only.** No price of the real panel was read while this was built:
  only its manifest, for the first and last bar of each asset.
- **The panel has been used before in this repository, and two of its results bear on this
  audit.** PREREG-CRYPTO-PIT found that the mean absolute return of the last 20 days predicts the
  next day's among the 30 largest pairs (rank correlation 0.335), so the control `independent` is
  expected to fail from knowledge of this panel. PREREG-CRYPTO-CAL found one-day reversal among the
  30 largest pairs (rank correlation 0.029, every fold positive), which is why expectation 3 says
  "below 1". Neither computed any of the six numbers, over these assets, in this form. The
  repository's records show no tail index, decay exponent, leverage or volume correlation computed
  on this panel.
- **Earlier notes in this repository on a few of these pairs point the same way as expectation 5.**
  `docs/TRADING_EQUATION_FORENSICS.md`, on other files (daily series of four pairs in the second
  half of 2023, and 15-minute bars of two pairs in 2024–25), reports that a price above its 20-day
  average was followed by higher volatility on three of the four, the opposite of US equity
  indices; that a signed 15-minute return was followed by lower volatility; and that unusual volume
  was followed by higher volatility. Expectation 5 was written before those notes were re-read.
- **The estimators were changed during design, on simulated data, for these reasons.** A Hill index
  above four standard deviations was carried by a handful of market-wide days, and its interval
  covered 64 to 80 % of the time instead of 99 %. A log-log fit of the autocorrelation read 0.70
  where the truth was 0.30, because the sample mean is itself uncertain under long memory; the
  variogram replaced it. On absolute returns the decay exponent's spread is 0.26 to 0.50; on
  ranks it is 0.12 to 0.14. The first simulated worlds made the leverage effect by feeding past
  returns into volatility, which changed the decay being measured; leverage now acts through the
  part of tomorrow's volatility that today's does not predict.
- **The tail index measured is not the limiting exponent.** In simulated worlds whose shocks have
  exponent 3, this number reads 2.95 when volatility is constant and 2.3 when it wanders, and in
  the second 1′ fails in 35 of 150 panels. A `fails` on 1′ therefore says that this estimator, on
  this market, reads below what was reported for US stocks. It does not say that the limiting
  exponent differs from 3.
- **Different market, different horizon.** The inverse cubic values are for five-minute returns of
  US stocks in 1994–95, and Cont's ranges mostly for equities, indices and exchange rates. Daily
  returns of crypto pairs are what the laws are claimed to extend to, and that claim is what is
  measured. A failure may mean the law does not carry over; it says nothing about the original
  measurements.
- **The literature on this market is already divided** on laws 1 and 4: a tail index between 2
  and 2.5 for Bitcoin (Begušić et al. 2018, arXiv:1803.08405) against roughly 3 in later years
  (Drożdż et al. 2018, arXiv:1804.05916), and both ordinary and inverted leverage reported. A
  failure here would be an independent measurement on a survivorship-free panel with intervals
  that allow for dependence. It would not be a first report.
- **Seven verdicts at 1 % each, no correction for their number.** On the 2,700 simulated panels,
  10 (0.4 %) failed at least one law that was true in their world.
- **What the simulated worlds lack:** jumps on listing and delisting days, regime changes, more
  than one common factor, a weekly cycle in volatility, days without trades. On the real panel a
  day without trades counts as a zero return.
- **Whole-period normalisation.** Each asset's mean, scale, ranks and tail threshold use the whole
  period. That suits a description; nothing here is a forecast.
- **One exchange, one quote currency, closes at 00:00 UTC.**
- **Who chose.** The laws, the estimators and the simulated worlds were chosen by the author
  (Claude), the laws before any power check. Not audited: gain and loss asymmetry, aggregational
  Gaussianity, intermittency, conditional heavy tails, asymmetry in time scales (not stated as one
  number), and volatility clustering as such (covered by the control and by law 3).

## Deviation rule

The run is not repeated. If the `run` job ends before `results.json` is written, for a reason of
infrastructure (the download, the installation), it is started again and recorded with its date.
Any change to a frozen file after the `run` job has started makes the result a development
result. If the repository owner asks for a change before the run starts, this preregistration is
superseded by a new one with a dated note here; nothing is edited in place.

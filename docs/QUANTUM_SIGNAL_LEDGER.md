# A quantum circuit's output as a trading signal, under the canonical ledger

**Record:** `benchmarks/quantum_compilation/quantum_signal_ledger.json` ·
**Holdout registry:** `benchmarks/quantum_compilation/quantum_signal_holdouts.jsonl` ·
**Module:** `apedp_dyn/quantum_compilation/quantum_signal.py` ·
**Producer:** `scripts/measure_quantum_signal_ledger.py` ·
**Pins:** `tests/test_quantum_signal_ledger.py`, `tests/test_quantum_signal_tranche2.py`,
`tests/test_quantum_signal_tranche3.py`, `tests/test_quantum_signal_tranche4.py`,
`tests/test_quantum_signal_tranche5.py` ·
**Status:** the maps are `experimental`, the executor and the holdout registry are `audit-only`

## What was wrong with the earlier demonstration

The pipeline was described correctly and the code that demonstrated it measured nothing:

```python
sim.phase(0, theta_0); sim.phase(1, theta_1)   # diagonal: inert on |0>
sim.h(0); sim.cx(0, 1)                          # wire 0 left maximally mixed
signal = p0 - p1                                # == 0.0 for every input
```

`phase` does nothing to `|0>`, and `h; cx` leaves wire 0 maximally mixed, so the "2-feature
quantum signal" was the constant **0.0**. Measured on nine input pairs: one distinct output.
Removing the phases changes nothing. The two features never reach the measured marginal.

`canonical_ledger.run_ledger` also had no entry point for an external score — it builds its
own seven named components from `raw_features` and takes no `signal=` argument — so "feed
that score into the ledger" was not an API that existed.

## What this record does

**A fixed map whose output depends on every feature.** `Ry(pi*x_i)` on each wire, then
`CX(i, 0)`, measures `<Z_0> = prod_i cos(pi*x_i)`. `QuantumFeatureMap` is that map, and the
simulator is checked against the closed form **on every bar of every dataset**.

**A parameterised map, fitted, with the fit charged.** `ParameterisedQuantumMap` is
`Ry(alpha_i x_i + beta_i)` with `CX(i, 0)` fan-in. At one layer,

    <Z_0> = prod_i cos(alpha_i * x_i + beta_i)

for any parameter values, so the fit cannot hide behind "the ansatz has no closed form". At
**two layers there is no product form** — the second rotation's `X` branches survive the
first entangler — and this is measured rather than assumed: on the input `(0.2, -0.4, 0.7)`
at `alpha=1.5, beta=0.5`, depth two gives `0.7146` where depth-one-squared gives `0.0002`.
So `analytic` **refuses** above one layer and the circuit is checked against
`exact_marginal`, which builds the whole unitary as dense matrices on the `qubit 0 = LSB`
convention and applies it to `|0>` — a different route to the same number, which is what
makes a disagreement possible.

The parameters are chosen from a declared grid (`FIT_ALPHAS` x `FIT_BIASES`) by in-sample
Sharpe on the training slice. `fit_parameterised_map` writes **one trial per grid point**
into its own `CandidateTrialLedger`, with `selection_pool_size = grid size`, so the fit is
charged at the size of the search exactly as the threshold sweep is — not as its winner.
The producer now spans both depths (`ANSATZ_DEPTHS = (1, 2)`, 24 grid points per dataset);
**the committed tranche below was fitted at depth 1 only** (12 points), because its holdout
slices are consumed and a second family cannot be added to a record that has already
looked. The 24-point ledger takes effect on the next tranche.

**An executor proven to be the ledger's accounting.** `execute_signal` implements
`gross = pos·fwd`, `fee = FEE·|Δpos|`, `net = gross − fee`, with the identity checked per
run. The record hands it `run_ledger`'s own score, mask and returns (`H-MOM` at unit weight,
`base` rule) and compares row by row on every dataset — **722 AAPL and 1,064 BTC rows, plus
every other dataset, maximum difference 0.0** in position, fee and net.

**Causality as a measurement.** Deleting the last 50 bars and recomputing leaves every
earlier score bit-identical, for the fixed maps and for the fitted map (whose parameters were
fixed before the truncation).

**A real holdout that is consumed and burned.** Each dataset is split by time (70% train /
30% holdout). Parameters are fitted on training bars only; the fitted map is then decided
**once** on the holdout slice, recorded in a durable `FullLineageExperimentLedger`. A slice
whose candidate fails is burned permanently and cannot be re-entered; the producer then
demonstrates a second candidate being refused by the ledger itself.

The rule is now **cost-aware** (`holdout_decision`): the candidate must trade in the
window, net after fees, **and** clear the expected maximum Sharpe of the search that
produced it, so the burn is charged to the fitted candidate's own search size — a wider
grid has to prove more. The committed tranche used the plain sign-and-turnover rule; the
charged rule applies from the next tranche, since no burned slice may be re-decided.

## The measured result

Ten datasets across two resolutions: six daily panels (AAPL, BTC, MSFT, NVDA, ETH, SOL) and
four 15-minute snapshots (BTC, ETH, SOL, BNB). Daily Sharpes annualise on 365 bars, 15-minute
ones on 35,040. 5 bps per unit turnover.

**190 disclosed trials** = 60 threshold-grid rows (2 maps x 3 thresholds x 10 datasets)
+ 10 momentum baselines + **120 parameter fits** (12 grid points x 10 datasets).
The dispersion the hurdle is built from is taken **within each resolution**
(`var_trials`: 0.5235 at 1 day, 258.53 at 15 minutes; pooled 358.76 kept only for
comparison), because a daily Sharpe and a 15-minute Sharpe estimate different quantities
and pooling them inflates the hurdle. The trial count is not split: every row is still
charged all 190.

| dataset | score | th | net | fees | turnover | Sharpe | DSR |
|---|---|---|---|---|---|---|---|
| SOL@1d | *baseline* H-MOM | 0.0 | **+4.1581** | 0.1085 | 217 | +1.478 | **0.234** |
| SOL@1d | H-MOM+CVD+H-BB | 0.2 | **+2.7560** | 0.2400 | 480 | **+1.412** | 0.207 |
| BTC@1d | H-MOM+CVD+H-BB | 0.2 | +1.0498 | 0.3605 | 721 | +0.829 | 0.022 |
| AAPL@1d | H-MOM+CVD | 0.0 | +0.0838 | 0.2890 | 578 | +0.128 | 0.004 |
| NVDA@1d | H-MOM+CVD | 0.0 | −1.2107 | 0.3150 | 630 | −0.964 | 0.000 |
| BTC@15m | H-MOM+CVD+H-BB | 0.5 | −0.1277 | 0.1515 | 303 | −37.114 | 0.000 |
| BNB@15m | H-MOM+CVD | 0.5 | −0.3135 | 0.2665 | 533 | −49.636 | 0.000 |

**Nothing clears the 0.95 floor, in either resolution**, and the hurdles are not the same
number: `expected_max_sr` is **1.989** for a daily row and **44.196** for a 15-minute row,
both charged the whole 190-trial family. At 1 day the best row (the SOL momentum baseline)
reaches DSR **0.234** — four times what the pooled hurdle left it — and still fails; SOL's
three-feature map reaches 0.207 and BTC@1d 0.022. At 15 minutes nothing is close, which the
fees already said: those rows pay 0.15–0.48 to earn less than that back.

The re-price was applied by `--recompute-dispersion`, which reads only the trial statistics
the record already stores, so it did not touch a panel or a holdout slice.

### The fit, and what the holdout did to it

| dataset | fitted α, β | train net | holdout net | holdout turnover | decision |
|---|---|---|---|---|---|
| SOL@1d | 1.5, +0.5 | +2.8887 | **+1.4285** | 180 | CONFIRMED_VALID |
| NVDA@1d | 1.0, −0.5 | +0.8379 | **+0.4589** | 52 | CONFIRMED_VALID |
| AAPL@1d | 0.5, 0.0 | +0.2197 | +0.2191 | 0 | CONFIRMATION_REJECTED |
| MSFT@1d | 1.5, +0.5 | +0.5409 | −0.3362 | 246 | CONFIRMATION_REJECTED |
| ETH@1d | 1.5, −0.5 | +1.5073 | −0.4314 | 240 | CONFIRMATION_REJECTED |
| BTC@1d | 2.0, 0.0 | +1.4032 | −0.7678 | 310 | CONFIRMATION_REJECTED |
| BTC@15m | 0.5, 0.0 | −0.0226 | −0.0163 | 8 | CONFIRMATION_REJECTED |
| ETH@15m | 0.5, −0.5 | −0.0129 | −0.0281 | 56 | CONFIRMATION_REJECTED |
| SOL@15m | 0.5, +0.5 | −0.0880 | −0.0716 | 56 | CONFIRMATION_REJECTED |
| BNB@15m | 0.5, +0.5 | +0.0214 | −0.0409 | 20 | CONFIRMATION_REJECTED |

**8 of 10 holdout slices are burned.** The decision rule is stated in the producer rather
than left implicit: a candidate passes when the holdout nets **and** changes position inside
the window. AAPL is the row that rule earns: +0.2191 of holdout PnL with **zero** turnover
in the window — a short carried in from the training boundary, not holdout evidence — so it
is rejected and its slice burned. Every burn is journaled in
`quantum_signal_holdouts.jsonl` with the exposure's `result_status` and the candidate state,
and the record carries the ledger's own `HoldoutContaminationError` for two post-burn reuse
attempts.

Four things the numbers say, stated plainly:

1. **Fitting does not survive the holdout at 1-day.** Five of six daily fits make money in
   sample and lose it out of sample; only SOL and NVDA carry through.
2. **The threshold map loses to the baseline it must beat.** SOL's `H-MOM+CVD+H-BB` nets
   +2.7560 against the momentum baseline's +4.1581 on half the turnover (480 against 217).
3. **Fifteen-minute bars are a fee pump here.** Every 15-minute row nets negative, and the
   fees are the same order as the gross: BNB@15m pays 0.2665 to earn 0.0470.
4. **The holdout is a real cost, not a formality.** Ten slices went in, eight are gone.

## What this does not show

- **No quantum advantage is tested.** `prod_i cos(alpha_i x_i + beta_i)` is classically
  computable in `O(n)` floating-point operations. The statevector simulation is the
  *definition* of the circuit, not the cheapest evaluation.
- **The fit is a 12-point grid, not quantum learning.** No gradients, no optimiser, no
  trainable embedding; the grid size is disclosed and charged, which is the honest version
  of the claim and still not a VQC.
- **Ten datasets, two resolutions, no slippage beyond the declared fee, no live feed.**
- **The holdout is temporal and consumed once.** This record is *not regenerable* from the
  same slices by design: eight are burned, and the producer now refuses a burned slice
  rather than re-reading it.
- **The committed fits are single-depth.** The producer now searches both depths (24 grid
  points per dataset, so the ledger spans two circuit families), but this tranche was fitted
  at depth 1 because its holdout slices are spent; the second family cannot be retro-fitted
  to a record that has already looked.
- **The holdout rule was the plain sign-and-turnover rule.** `holdout_decision` now charges
  the burn to the fit's own search size by requiring the slice to clear the expected maximum
  Sharpe of that search; it takes effect on the next tranche, since no burned slice may be
  re-decided.

## Tranche 2: fresh slices, two depths, a charged burn

Tranche 1's ten slices are spent (eight burned, two confirmed), so tranche 2 is a separate
record -- `quantum_signal_ledger_tranche2.json` -- on datasets never decided before: six
daily assets that tranche 1 did not touch (GOOGL, META, TSLA, DOGE, ADA, LINK) and four
symbols at a new 1-hour resolution (ADA, AVAX, LINK, DOT). It searches **both ansatz
depths** (24 grid points per dataset) and decides each slice with the charged rule.

**310 disclosed trials** = 60 threshold rows + 10 baselines + **240 fits** (24 per
dataset). Dispersion within resolution: 0.569 at 1 day, 13.441 at 1 hour; hurdles **2.193**
and **10.653**. Nothing clears 0.95: the best daily row is TSLA's momentum baseline at DSR
**0.031**, the best 1-hour row DOT's at **0.003**.

**All ten slices are burned, and the rule bit before the sign did.** ADA@1d's fitted map
earns **+0.5649** on its holdout with 216 units of turnover and still fails: its holdout
Sharpe sits below the 2.193 hurdle its own 24-point search has to clear. That is the
charging working — a positive number is not a confirmation.

### Depth 2 against depth 1, out of sample, each charged 24 trials

The paired read evaluates each family's best candidate on the same holdout window from the
same single read, and charges each family its own grid:

| dataset | depth 1 net | depth 2 net | Δ (2−1) |
|---|---|---|---|
| TSLA@1d | −0.6127 | −0.3462 | **+0.2664** |
| AVAX@1h | −0.0875 | −0.1493 | −0.0618 |
| DOT@1h | −0.2215 | −0.2890 | −0.0675 |
| LINK@1h | −0.0585 | −0.1955 | −0.1371 |
| META@1d | −0.5062 | −0.6526 | −0.1464 |
| ADA@1h | +0.0158 | −0.1427 | −0.1585 |
| GOOGL@1d | +0.2606 | −0.1337 | −0.3944 |
| DOGE@1d | +0.0418 | −0.3748 | −0.4166 |
| LINK@1d | +0.2710 | −0.5711 | −0.8421 |
| ADA@1d | +0.5649 | −0.6116 | **−1.1766** |

**Depth two wins on one dataset of ten and loses on nine**, mean holdout delta **−0.3134**,
and neither family clears its hurdle anywhere (`depth1_clears_hurdle = 0`,
`depth2_clears_hurdle = 0`). The answer to "does the second family pay for its search?" is
therefore no on this tranche: it wins where the depth-1 fit was already losing (TSLA) and
loses worst where depth 1 had the largest holdout gain (ADA, −1.1766).

## Tranche 3: a continuous optimiser, charged by its iteration count

Tranche 2's slices are spent too (all ten burned, registry 18). Tranche 3 is a third record
on unseen data -- AMZN, JPM, XOM, WMT, XRP, DOT daily and XRP, TRX, LTC, BCH at 30 minutes --
fitted with **coordinate ascent** over `(alpha, beta)` per depth instead of a grid.

The accounting claim is the point: **every evaluation is one recorded trial**, with
`selection_pool_size` set to the declared budget of **17 evaluations per depth**, so the
charge is an iteration count rather than a grid chosen to make the multiplicity look small.
**410 trials** = 60 threshold rows + 10 baselines + **340 continuous evaluations** (34 per
dataset). Six probes were skipped because an alpha step would have crossed zero; the
declared budget is charged anyway and the difference is reported as `hidden_trials = 6`.

That the optimiser is genuinely continuous is visible in the parameters: fits selected
`alpha = 0.8, 1.2, 1.8, 0.2` — values the grid `{0.5, 1.0, 1.5, 2.0}` cannot produce.

**Nothing clears 0.95 again**, and **all ten slices burned** (registry 28). The rule bound on
a profitable candidate: AMZN@1d earns **+0.1704** with 124 units of turnover and is rejected
below its **2.26** hurdle. Dispersion 0.569 daily against 120.85 at 30 minutes; hurdles
2.26 and 32.9.

## The cost-model sweep

`cost_sensitivity` re-prices **every priced trial** at 5, 10 and 20 bps. Net, fees and
turnover are exact and need no data: fees are linear in turnover, so `fees(r) = turnover * r`
and `net(r) = gross - fees(r)` per row. Sharpe and DSR need the bars, and only tranche 5
stores them: on tranches 1–4 the field says **not re-priced** rather than approximating,
while on tranche 5 it recomputes both from the stored per-bar series (below).

| record | 5 bps net (rows > 0) | 10 bps net | 20 bps net (rows > 0) |
|---|---|---|---|
| tranche 1 | −11.192 (13/70) | −31.549 (11/70) | −72.262 (8/70) |
| tranche 2 | −29.721 (13/70) | −49.117 (8/70) | −87.908 (3/70) |
| tranche 3 | −22.623 (16/70) | −41.335 (6/70) | −78.759 (4/70) |
| tranche 5 | −14.384 (31/140) | −27.913 (20/140) | −54.971 (8/140) |

The family is already net-negative at the canonical 5 bps, so the sweep is not asking
whether a thinner fee would rescue it: raising the fee removes rows monotonically, and
tranche 2 loses ten of its thirteen positive rows between 5 and 20 bps.

## Tranche 4: walk-forward folds, burned per fold

Tranche 3 burned its ten slices too (registry 28). Tranche 4 judges **unseen assets across
regimes**: six daily (CVX, HD, KO, PEP, BCH, LTC) and four at 1 hour (VET, MANA, ZEC, XLM),
each split into **two walk-forward folds**.

`walk_forward_windows` cuts the post-warm-up span into `folds + 1` equal blocks; fold `i`
trains on every bar before its block and tests on that block alone. The windows are
contiguous and non-overlapping — for 1,000 bars, two folds and a 30-bar warm-up:
`(30, 353, 353, 676)` and `(30, 676, 676, 999)` — so a bar is judged in exactly one fold, a
fold is consumed once, and the dataset is still read across two regimes rather than one.

**820 trials** = 120 threshold rows (2 maps x 3 thresholds x 20 fold-datasets) + 20 baselines
+ **680 continuous evaluations** (34 per fold-dataset). Hurdles **3.62** daily and **16.85**
at an hour; **no row clears 0.95**, and **all 20 fold slices burned** (registry 48). Seven
of the twenty folds make money on their test block, and all seven are still rejected — the
charge, not the sign, is what decides.

### How much of the gross is fees at 5 bps

The sweep reports the bill per resolution, including where gross is negative (where a
"share of gross" is undefined, the ratio to `|gross|` still is):

| record | resolution | gross | fees @5 bps | fees ÷ \|gross\| |
|---|---|---|---|---|
| tranche 1 | 1d | +9.396 | 11.215 | **119%** |
| tranche 1 | 15m | −0.231 | 9.142 | 3,952% |
| tranche 2 | 1d | −11.101 | 10.313 | 93% |
| tranche 2 | 1h | +0.776 | 9.082 | 1,171% |
| tranche 3 | 1d | −4.647 | 10.132 | 218% |
| tranche 3 | 30m | +0.736 | 8.580 | 1,166% |
| tranche 4 | 1d | −4.909 | 6.976 | 142% |
| tranche 4 | 1h | −1.718 | 6.243 | 363% |
| tranche 5 | 1d | −0.746 | 6.964 | 934% |
| tranche 5 | 5m | −0.109 | 6.565 | 6,038% |

The reading is uniform: at the canonical 5 bps the bill is the same order as the gross and
usually larger. On tranche 1's daily family the fees are **119% of the entire gross** — the
only reason that family is not a total loss is that it is barely traded. The intraday
resolutions are worse by an order of magnitude, which is the arithmetic reason a one-bar
holding period cannot pay 5 bps.

## Tranche 5: the bars behind each row, and the fee re-priced through the distribution

The aggregates were enough to re-price net and fees but not Sharpe: that needs every bar.
Tranche 5 stores them. `benchmarks/quantum_compilation/quantum_signal_series_tranche5.jsonl`
holds, for each of the **140 priced rows**, the **per-bar net at the canonical 5 bps** and
that bar's **\|Δpos\|** — **38,248 bars** — so

    net(r) = net(5 bps) − (r − 5 bps) · |Δpos|

holds bar by bar and every rate re-prices exactly. The check is not decorative: re-pricing
at 5 bps must reproduce the recorded Sharpe of every row, and it does to **0.0** (net to
4.4 × 10⁻¹⁶). The stored bars *are* the bars the record's statistics came from.

The data is fresh in both dimensions: six daily names (MRK, UNH, BAC, COST, GLD, IWM) and
four symbols at **5 minutes** (AXS, SNX, COMP, CRV), each in two walk-forward folds.
**820 trials**, the same fifteen checks, **all 20 fold slices burned** (registry 88), **no
row clears 0.95**; hurdles **4.04** daily and **78.38** at five minutes.

A first attempt at this tranche — JNJ, LLY, PG, V, AMD, NFLX daily and 1INCH, GRT, SAND,
CHZ at 5 minutes — ran its holdout and was then refused at the series check, so its twenty
slices are burned with no record written (registry 68 → 88). The slice is consumed before
the aggregate checks pass, which is the price of judging last; the second set is disjoint
and no burned slice was re-read.

### The fee through the whole distribution, not just the mean

| fee rate | net | rows > 0 | daily median SR | daily best DSR | 5-minute median SR | 5-minute best DSR |
|---|---|---|---|---|---|---|
| **5 bps** | −14.3837 | 31/140 | −0.678 | 0.087 | −51.574 | 0.000 |
| 10 bps | −27.9127 | 20/140 | −1.101 | 0.038 | −94.831 | 0.000 |
| 20 bps | −54.9707 | 8/140 | −2.060 | 0.011 | −160.350 | 0.000 |

The median moves the same way the mean does, and further: doubling the fee takes the daily
median Sharpe from −0.678 to −1.101, and quadrupling it to −2.060, while the 5-minute
median falls from −51.574 to −160.350. The best daily row is UNH@1d#fold1 at Sharpe
**+2.522**, which still reads DSR **0.087** against its 4.04 hurdle. **Zero rows clear the
floor at any of the three rates**, so the conclusion is not an artefact of choosing 5 bps:
no fee in the swept range makes this family significant, and the cheaper rate only shortens
the distance.

## What the design could actually detect

Every tranche above reports how many rows cleared the 0.95 floor. None of them reported
the number that makes a null readable: **the smallest true Sharpe this design could have
detected.** `scripts/measure_quantum_signal_power.py` computes it from the records alone —
no panel is read and no holdout slice is touched — by inverting the same
`compute_deflated_sharpe_ratio` the records use. The **minimum detectable Sharpe (MDS)** is
the true annualised Sharpe at which that design's DSR reaches 0.95; power at a candidate
true Sharpe `s` is `P(SR_hat >= MDS)` with `SR_hat ~ N(s, se(s))`, using the DSR's own
standard error. Record: `benchmarks/quantum_compilation/quantum_signal_power.json`.

| record | resolution | holdout bars | trials | **MDS** | best observed SR | its DSR | P(pass \| true SR = 1.0) |
|---|---|---|---|---|---|---|---|
| tranche 1 | 1d | 722 | 190 | **3.19** | +1.48 | 0.234 | 0.11% |
| tranche 1 | 15m | 969 | 190 | **57.05** | −10.30 | 0.000 | 0.00% |
| tranche 2 | 1d | 722 | 310 | **3.39** | +0.88 | 0.031 | 0.04% |
| tranche 2 | 1h | 969 | 310 | **15.79** | +2.46 | 0.003 | 0.00% |
| tranche 3 | 1d | 722 | 410 | **3.47** | +1.18 | 0.062 | 0.03% |
| tranche 3 | 30m | 969 | 410 | **41.06** | +1.19 | 0.000 | 0.00% |
| tranche 4 | 1d | 240 | 820 | **5.79** | +2.24 | 0.114 | 0.01% |
| tranche 4 | 1h | 323 | 820 | **26.97** | +3.21 | 0.004 | 0.00% |
| tranche 5 | 1d | 240 | 820 | **6.17** | +2.52 | 0.087 | 0.00% |
| tranche 5 | 5m | 323 | 820 | **111.84** | −2.09 | 0.000 | 0.00% |

That is the reading the null was missing, and it cuts both ways:

* **Against the hypothesis.** No recorded row is anywhere near its own MDS, and at 5-minute
  bars the design needs a Sharpe of **111.84** to say anything at all — four orders of
  magnitude beyond what a real intraday strategy produces.
* **Against the test.** A true Sharpe of **1.0** — an excellent, tradeable edge — has a
  detection probability of **0.11% at best** (tranche 1 daily) and essentially zero
  everywhere else. So "nothing clears 0.95" is evidence against effects above the MDS and
  **no evidence at all** about an effect below it. The design is a filter for fantasy-size
  claims, not a test of whether a modest edge exists.

The MDS grows as sample shrinks and trials grow: 722 daily bars at 190 trials gives 3.19;
240 bars at 820 trials gives 6.17. That is the cost of charging honestly on this much data,
and it is now on the record instead of implied.

## The programme charge: five tranches, one multiplicity

Each tranche charged its own trials and built its own hurdle. Correct within a tranche,
wrong across a programme: five asset sets, five resolutions, a second ansatz depth, a
change of optimiser and a mid-course change of dispersion were all decided by someone who
had already seen the previous tranche. `programme_charge` re-prices **every row of every
tranche** against the **2,550 trials** the programme spent (190 + 310 + 410 + 820 + 820),
held at the dispersion that row's tranche used, so the number that moves is the
multiplicity the tranche structure hid and not a change of estimator.

| resolution | hurdle as run (median) | hurdle at the programme charge |
|---|---|---|
| 1d | 3.618 | **3.973** |
| 1h | 16.847 | **18.502** |
| 30m | 32.896 | **38.614** |
| 5m | 78.379 | **86.079** |
| 15m | 44.196 | **56.476** |

The charge is the trial count at fixed dispersion, and the expected maximum rises with it,
so no hurdle can fall — the record asserts that for every row. **Rows clearing the floor:
0 as run, 0 at the programme charge, and no row becomes significant.** The conclusion
therefore survives the stricter accounting; what does not survive is the impression that
each tranche was an independent test rather than a fifth look by the same researcher.

Pooling dispersion across tranches is a *second*, independent change that can move a
hurdle either way, so it is carried per row as `hurdle_programme_pooled_dispersion` and
never substituted for the charge. What this still cannot charge is the choice of assets,
the choice of resolutions, or when each method change was made: it is a floor on the
programme's multiplicity, not a complete account of the researcher's degrees of freedom.

## The burn log is sealed, and a run validates before it spends a slice

Two defects, both found by walking the tranche-5 failure backwards.

**The registry had no integrity check.** `FullLineageExperimentLedger` rebuilds its burned
set by *reading* the JSONL file, so a file with a burn deleted reads as a file that never
had one. `apedp_dyn/epistemic/holdout_seal.py` folds every non-empty line into a running
SHA-256 chain and writes the terminal digest beside the log
(`quantum_signal_holdouts.jsonl.seal`); `verify_seal` recomputes it and fails closed on a
line-count, byte-length or digest mismatch. Truncation, an edited burn and a rewind each
raise `HoldoutSealError`. Stated limit: reverting **both** the log and its sidecar defeats
it, so the chain digest is itself pinned in `tests/test_holdout_seal.py`
(282 lines, 198,027 bytes, chain `ac422774…4b96`) — a rewind now has to change a test.
The stronger form is an external append-only store (`storage_backend`), which is not what
this is.

**Validation ran after the irreversible step.** The first tranche-5 attempt decided twenty
holdouts and was *then* refused by the series check, burning twenty slices with nothing to
show for them. The producer now has a **pre-flight** block that settles everything which
can refuse the run — the seal, the stored series against the aggregates, the accounting
identity, the trial count — against data already in memory, **before** the ledger is
opened, and records it as `preflight` with `slices_consumed_by_this_run: 0`. The ordering
is asserted on the source in `tests/test_holdout_seal.py`, because the cost of regressing
it is paid in slices that cannot be recovered.

## Reproduce

```bash
python scripts/measure_quantum_signal_ledger.py --tranche 4     # walk-forward folds
python scripts/measure_quantum_signal_ledger.py --tranche 5     # + per-bar series
python scripts/measure_quantum_signal_ledger.py --recompute-dispersion \
    --output benchmarks/quantum_compilation/quantum_signal_ledger.json  # stats only, no data
pytest tests/test_quantum_signal_ledger.py tests/test_quantum_signal_tranche2.py \
       tests/test_quantum_signal_tranche3.py tests/test_quantum_signal_tranche4.py \
       tests/test_quantum_signal_tranche5.py -q
```

The script exits non-zero rather than writing a record whose checks fail, and the fourteen
checks include reference equality (product form at depth 1, dense unitary above it), the
fit-multiplicity identity (`parameter_fits == datasets x grid`), the train/holdout
boundary, the burn reciprocity, and the post-burn refusal — so a constant circuit, an
uncharged fit, or a snooped holdout each fail loudly.

`--recompute-dispersion` re-prices an existing record's DSR from the trial statistics it
already stores, without reading a panel or touching a holdout slice — which is how the
within-resolution dispersion above was applied to a record whose slices are spent.

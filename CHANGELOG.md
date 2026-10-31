## 2026-10-31 — The unpushed range could not be pushed: GitHub's 100 MB limit

- **THE PUSH WAS REJECTED, AND NOT BY ANYTHING NEW.** `stage` was 57 commits ahead of
  `origin/stage`, and `50df2b1` had committed four artifacts over GitHub's per-file limit:
  the 122 MB competition zip, the 122 MB `FeMoCo76o113e.fcidump`, and two 846 MB `.kmb`
  receipts — 1.9 GB of blobs. None of it was in the published range, so the repair is
  confined to local commits.
- **WHAT WAS REWRITTEN.** `filter-branch --index-filter` over `origin/stage..HEAD` removed
  those five paths from the 57 unpushed commits; one commit became empty and was pruned
  (57 → 56). Every hash in that range changed. `backup/pre-rewrite-stage` holds the
  pre-rewrite tip, so the old history is recoverable.
- **THE FILES STAY ON DISK.** Two are needed by code that reads them: the canonical
  `dqi_niters_222.kmb` that four `prepare_*_artifacts.py` scripts copy from, and the FCIDUMP
  the DFTHCSOS verifier scores against. Both are now ignored and must be fetched separately;
  `verifiers/dfthcsos/SCORING_README.md` pins the FCIDUMP's size and SHA-256
  (128,128,348 bytes, `e5a48af7…e1bd`), so a re-download is verifiable rather than assumed.
- **WHAT THIS DOES NOT DO: shrink the local repository.** The backup branch still
  references the old objects, so nothing is reclaimed until it is retired and `git gc`
  runs. The gain is that a clone can now exist at all.

Updated: `.gitignore`, `CHANGELOG.md`.

## 2026-10-31 — Two checkouts, one canonical, and a route map for the designer

- **`science` IS CANONICAL; THE SIBLING IS AN ARCHIVE.** The two checkouts share a common
  ancestor (`444c1fa`) and overlapping object stores, so this is a git-level fact rather
  than a naming preference: the archive
  (`/home/psalmprax/ALL_PROJECTS/science-quantum-discovery`, HEAD `ffa7b54`, 2026-10-27) is
  **5 commits ahead and 35 behind**, it has **no `apedp_dyn/vqc` at all**, and **8 of the 9
  files** its five commits touch are already byte-identical here.
- **ONE STRANDED ITEM CAN MISLEAD A READER, AND IT IS NAMED.** `ae02214` recorded the
  **rejection** of the DFTHCSOS 10-bit signed 4-bank candidate with its arithmetic
  (`error_1` 134.854 mHa against a 20 mHa threshold, `error_2` 44.889, **132 peak angle
  qubits against a 128 cap**, a 4,363-Toffoli gap between the literal formula and the
  claim). This checkout's copy of the same file still says
  `HOLD_FOR_ARITHMETIC_AND_RECEIPT_RECONCILIATION` — so read from here, a disproved
  candidate looks like an open question. Port command recorded, not executed.
- **Three other items, each with a disposition rather than a copy.** The uncommitted
  `min_margin_x` diagnostic in `verify_submission_invsqrt.py`; three OpenMP Davio searches
  (`search_davio_exact.c`, `search_shared_davio.c`, `test_davio_fast.c`) that must be
  checked for supersession against this checkout's `search_claims_pddd.c` and
  `search_mixed_polarity_pddd.py` before anyone ports them; and
  `scripts/measure_89q_executed_cost.py`, whose shebang hard-codes
  `#!/dev/shm/qvcc-venv313/bin/python` — a machine-local path the existing
  `/home/...` guard would not catch, so it is fixed at the source rather than by widening
  the guard.
- **The designer now has a route map** separate from its claims: `docs/VQC_DESIGNER_MAP.md`
  carries the specification-to-verified-schedule path, the 24 modules with line counts and
  roles, the public entry points, the eight records and seven producers, and the table of
  what the designer refuses and what it does not do (no non-Clifford synthesis, no invented
  topology, no 900-qubit routing, no hardware cost model).

Added: `docs/REPOSITORY_CANONICAL_DECISION.md`, `docs/VQC_DESIGNER_MAP.md`. Updated:
`docs/VQC_SYNTHESIS.md` (cross-references), `docs/DOCUMENTATION.md`, `CHANGELOG.md`.

## 2026-10-31 — Repo hygiene: the index stops carrying four copies of one 845 MB receipt

- **ONE CIRCUIT, FIVE FILES, THREE OF THEM IN THE INDEX.** `ls -l` reported five ~845 MB
  `.kmb` circuit artifacts. Four share a byte identity — `sha256 959471d0…04995` — the
  sample submission, the canonical iteration-222 receipt, and every
  `official_receipt_runs/**/dqi.kmb` copy — and the fifth (`dqi_niters_216.kmb`,
  `4ae20e1c…`) is a superseded sweep. The 122 MB `google-verifiable-…zip` is the only one
  of its kind whose entire contents are already extracted in the same directory.
- **WHAT WAS REMOVED FROM THE INDEX** (kept on the working tree):
  `sample_submission/dqi.kmb`, `dqi_niters_216.kmb` and the `.zip`, with explicit
  `.gitignore` rules so they cannot drift back in. **What was kept tracked:** the canonical
  `dqi_niters_222.kmb`, because four committed `prepare_{baseline,track2,track3,track4}_artifacts.py`
  scripts read it as their source, and `FeMoCo76o113e.fcidump` (122 MB), because
  `verifiers/dfthcsos/scoreinfra/reference.py` scores against it and nothing else can.
- **WHAT WAS DELETED FROM DISK** is only what a hash check proved redundant: the four
  `official_receipt_runs` copies were deleted **one at a time, each after `sha256` matched
  the canonical receipt byte for byte**. Nothing was deleted on the strength of its name or
  its size.
- **The limit is stated, not glossed, and no space claim is made.** The copies were
  hardlinks of one inode, so removing the names freed **nothing** on disk — `du` already
  counted the inode once, which is why `artifacts/` read ~1 GB while listing four 845 MB
  files. The gain here is entirely in the **index**: the repository stops carrying those
  blobs forward. `.git` history still contains them, so its 789 MB is unchanged;
  reclaiming that needs a history rewrite (`filter-repo`/BFG), which is destructive and
  rewrites every downstream clone — not done here, and separate by design.
- **What the 5.0 GB is, measured rather than guessed:** `target/` 1.7 GB (ignored cargo
  build output), `artifacts/` 1.9 GB (ignored receipts and checkpoints), `apedp_dyn/`
  1.2 GB (the vendored competition package, including the 846 MB canonical receipt and the
  122 MB FCIDUMP the verifier cannot work without), `.git` 789 MB. Only the last is history
  weight, and only a rewrite shrinks it.

Updated: `.gitignore` (three artifact rules with the reason), `CHANGELOG.md`.

## 2026-10-31 — What the study could detect, what the programme spent, and a sealed burn log

- **THE NULL HAD NO POWER NUMBER, AND NOW IT DOES.** `scripts/measure_quantum_signal_power.py`
  writes `benchmarks/quantum_compilation/quantum_signal_power.json` from the five tranche
  records alone — no panel, no holdout slice — by inverting the same
  `compute_deflated_sharpe_ratio` they use. The **minimum detectable Sharpe** is 3.19–3.47
  on 722 daily bars at 190–410 trials, 5.79–6.17 on 240 bars at 820, and **15.79 / 26.97 /
  41.06 / 57.05 / 111.84** at 1h / 1h / 30m / 15m / 5m. A true Sharpe of **1.0** has a
  detection probability of **0.11% at best** and ~0.00% everywhere else. So "nothing clears
  0.95" is evidence against effects above the MDS and no evidence about anything smaller:
  the design is a filter for fantasy-size claims, not a test of a modest edge. Both numbers
  are in the record so a null cannot be quoted without the threshold it was tested at.
- **THE FIVE TRANCHES ARE ONE MULTIPLICITY, CHARGED AS ONE.** `programme_charge` re-prices
  every row of every tranche against the **2,550 trials** the programme spent (190 + 310 +
  410 + 820 + 820), held at the dispersion that row's tranche used — the number that moves
  is the multiplicity the tranche structure hid, not a change of estimator. Hurdles rise
  (1d 3.618 → 3.973, 1h 16.847 → 18.502, 30m 32.896 → 38.614, 5m 78.379 → 86.079, 15m
  44.196 → 56.476), **no hurdle falls**, rows clearing the floor stay **0 as run and 0 at
  the programme charge**, and **no row becomes significant**. Pooling dispersion across
  tranches is carried per row as a separate sensitivity, because it can move a hurdle
  either way and is not the charge. What is still uncharged is stated: the choice of
  assets, resolutions and when each method change was made.
- **THE BURN LOG IS SEALED, AND A REFUSED RUN NOW COSTS NOTHING.** New
  `apedp_dyn/epistemic/holdout_seal.py` folds every line of the registry into a SHA-256
  chain and writes the terminal digest to `quantum_signal_holdouts.jsonl.seal`; `verify_seal`
  fails closed on truncation, an edited burn, or a rewind, and the digest is pinned in
  `tests/test_holdout_seal.py` (282 lines, 198,027 bytes, `ac422774…4b96`) so reverting both
  the log and its sidecar also fails. The producer verifies the seal and runs a
  **pre-flight** block — series against aggregates, accounting identity, trial count —
  *before* the ledger is opened, and records `preflight` with
  `slices_consumed_by_this_run: 0`. The ordering is asserted on the source, because
  tranche 5's first attempt lost twenty slices to a check that ran after the holdout.

Added: `apedp_dyn/epistemic/holdout_seal.py`, `scripts/measure_quantum_signal_power.py`,
`benchmarks/quantum_compilation/quantum_signal_power.json`,
`benchmarks/quantum_compilation/quantum_signal_holdouts.jsonl.seal`,
`tests/test_holdout_seal.py` (10), `tests/test_quantum_signal_power.py` (10). Updated:
`scripts/measure_quantum_signal_ledger.py` (seal verification, pre-flight, re-seal),
`docs/QUANTUM_SIGNAL_LEDGER.md`, `docs/DOCUMENTATION.md`.

## 2026-10-31 — Tranche 5: per-bar series, and Sharpe/DSR re-priced at every fee rate

- **THE BARS ARE STORED NOW, SO THE FEE REACHES THE DISTRIBUTION.**
  `quantum_signal_ledger_tranche5.json` ships
  `quantum_signal_series_tranche5.jsonl`: for every priced row, the per-bar net at the
  canonical 5 bps and that bar's `|Δpos|` — **38,248 bars over 140 rows** — so
  `net(r) = net(5 bps) − (r − 5 bps)·|Δpos|` holds bar by bar. `cost_sensitivity` now
  recomputes Sharpe and DSR at 5/10/20 bps from those bars (`sharpe_recomputed`,
  `series_max_sharpe_deviation`, `sharpe_by_resolution`, `dsr_by_resolution`), and a new
  check requires the 5 bps re-price to reproduce every recorded Sharpe: it does to **0.0**.
- **THE VERDICT DOES NOT DEPEND ON THE RATE.** MRK, UNH, BAC, COST, GLD, IWM daily and AXS,
  SNX, COMP, CRV at **5 minutes**, two walk-forward folds, **820 trials**: the daily median
  Sharpe falls −0.678 → −1.101 → −2.060 and the 5-minute one −51.574 → −94.831 → −160.350
  across 5/10/20 bps, positive rows 31 → 20 → 8, and **zero rows clear 0.95 at any rate**;
  hurdles 4.04 and 78.38, **20/20 fold slices burned** (registry 88).
- **THE BILL, BY RESOLUTION, AT 5 BPS.** Tranche 5's daily fees are **934% of |gross|** and
  its 5-minute fees **6,038%**; the earlier families read 119% (t1 daily), 93% (t2 daily),
  218% (t3 daily) and 142% / 363% (t4). Reported as
  `fee_share_of_gross_by_resolution` where gross is positive and
  `fees_over_absolute_gross_by_resolution` everywhere, because a share of a negative gross
  is undefined and pretending otherwise would be a fabricated ratio.
- **A REFUSED RUN STILL COSTS ITS SLICES.** The first tranche-5 attempt (JNJ, LLY, PG, V,
  AMD, NFLX; 1INCH, GRT, SAND, CHZ) decided its holdout and was then refused by
  `the_stored_series_reproduces_every_aggregate`, because the stored bars had been rounded
  to 12 dp and the sums drifted past 1e-9. The twenty slices were already burned
  (registry 68 → 88) and are not re-read: the record's assets are a disjoint set, and the
  cost of judging last is documented rather than hidden.

Added: `tests/test_quantum_signal_tranche5.py` (8),
`benchmarks/quantum_compilation/quantum_signal_ledger_tranche5.json`,
`benchmarks/quantum_compilation/quantum_signal_series_tranche5.jsonl`. Updated:
`scripts/measure_quantum_signal_ledger.py` (5-minute resolution, `--tranche 5`, per-bar
series storage, Sharpe/DSR re-pricing in `cost_sensitivity`, the series check),
`docs/QUANTUM_SIGNAL_LEDGER.md`, `docs/DOCUMENTATION.md`.

## 2026-10-31 — Tranche 4: walk-forward folds, a fee share per resolution

- **A SLICE CONSUMED ONCE, JUDGED ACROSS REGIMES.** `walk_forward_windows` cuts the
  post-warm-up span into `folds + 1` blocks; fold `i` trains on every bar before its block
  and tests on that block alone (1,000 bars, 2 folds, 30-bar warm-up:
  `(30,353,353,676)` and `(30,676,676,999)`). Tranche 4 runs it on unseen assets — CVX, HD,
  KO, PEP, BCH, LTC daily and VET, MANA, ZEC, XLM at 1 hour — so each base dataset is
  20 fold-datasets, each with its own slice id and its own burn.
- **820 TRIALS, NOTHING CLEARS 0.95, 20/20 FOLD SLICES BURNED** (registry 48): 120 threshold
  rows + 20 baselines + 680 continuous evaluations (34 per fold-dataset). Hurdles 3.62
  daily and 16.85 at an hour. Seven of the twenty folds make money on their test block and
  all seven are still rejected by the charge — `the_holdout_block_follows_its_training_window` replaced
  the tail-specific check, and `depth_comparison.charge_per_family` is now derived from the
  fits instead of the grid, so a continuous search is no longer mislabelled 24.
- **THE BILL, PER RESOLUTION, AT 5 BPS.** The sweep now reports `gross_by_resolution`,
  `fees_by_resolution`, `fee_share_of_gross` and `fees_over_absolute_gross` (defined even
  when gross is negative). At the canonical 5 bps the fee is the same order as the gross and
  usually larger: tranche 1 daily **119% of the whole gross**, tranche 2's 1-hour family
  **1,171%**, tranche 3's 30-minute **1,166%**, tranche 4's hourly **363%**. That is the
  arithmetic behind "a one-bar holding period cannot pay 5 bps".

Added: `tests/test_quantum_signal_tranche4.py` (5),
`benchmarks/quantum_compilation/quantum_signal_ledger_tranche4.json`. Updated:
`apedp_dyn/quantum_compilation/quantum_signal.py` (`walk_forward_windows`),
`scripts/measure_quantum_signal_ledger.py` (`--tranche 4`, fold windows, per-fold slice ids
and burns, fee-share report), `docs/QUANTUM_SIGNAL_LEDGER.md`, `docs/DOCUMENTATION.md`.

## 2026-10-31 — Tranche 3 (continuous optimiser, charged by iteration) and a fee sweep

- **THE CHARGE IS AN ITERATION COUNT NOW, NOT A GRID.** Tranche 3 fits unseen data (AMZN,
  JPM, XOM, WMT, XRP, DOT daily; XRP, TRX, LTC, BCH at 30 minutes) with coordinate ascent
  over `(alpha, beta)` per depth. `fit_continuous_map` writes **one trial per evaluation**
  with `selection_pool_size` set to the declared 17-per-depth budget, so **410 trials** =
  60 threshold rows + 10 baselines + **340 evaluations** (34 per dataset). Six probes were
  skipped at the `alpha > 0` boundary; the budget is still charged and the gap is disclosed
  as `hidden_trials = 6`. The search is genuinely off-grid: selected alphas include 0.8,
  1.2, 1.8, 0.2.
- **STILL NOTHING CLEARS 0.95, AND THE RULE STILL BITES.** Hurdles 2.26 daily and 32.9 at
  30 minutes; **all ten slices burned** (registry 28). AMZN@1d earns **+0.1704** with 124
  units of turnover and is rejected below its own hurdle.
- **A COST-MODEL SWEEP THAT DOES NOT TOUCH A SLICE.** `cost_sensitivity` re-prices every
  priced trial in all three records at 5, 10 and 20 bps. Fees are linear in turnover, so
  net is exact per row without reading a bar; Sharpe and DSR are **not** re-priced because
  they need the per-bar series, and the field says so. The family is net-negative at 5 bps
  already (−11.192 / −29.721 / −22.623), and raising the fee removes rows monotonically
  (tranche 2 drops from 13 positive rows to 3).
- **Two run failures fixed in place:** `TRANCHES` was declared above the panel tuples it
  referenced (NameError), and the continuous fitter indexed `net_returns` as if it held row
  dicts (TypeError) — both caught by the first run of the new tranche.

Added: `tests/test_quantum_signal_tranche3.py` (7),
`benchmarks/quantum_compilation/quantum_signal_ledger_tranche3.json`. Updated:
`apedp_dyn/quantum_compilation/quantum_signal.py` (`fit_continuous_map`,
`continuous_evaluations`, `CONTINUOUS_STEPS/PASSES/START`, `FitResult.optimiser`),
`scripts/measure_quantum_signal_ledger.py` (`--tranche`, 30-minute resolution,
`cost_sensitivity`), `docs/QUANTUM_SIGNAL_LEDGER.md`, `docs/DOCUMENTATION.md`.

## 2026-10-31 — Tranche 2: fresh slices, two depths, and a charge that bites before the sign

- **A SECOND TRANCHE, BECAUSE THE FIRST ONE'S SLICES ARE SPENT.** Tranche 1 burned eight of
  ten. Tranche 2 is a separate record on datasets never decided before: six unused daily
  assets (GOOGL, META, TSLA, DOGE, ADA, LINK) and four symbols at a new 1-hour resolution
  (ADA, AVAX, LINK, DOT). 310 disclosed trials = 60 threshold rows + 10 baselines + **240
  fits** (24 per dataset, both ansatz depths).
- **THE CHARGE BITES BEFORE THE SIGN DOES.** Under `holdout_decision` a slice is claimed only
  if the candidate trades, nets, **and** clears the expected maximum Sharpe of its own
  24-point search. ADA@1d's fitted map earns **+0.5649** with 216 units of turnover and is
  still rejected, because its holdout Sharpe is below the **2.193** hurdle; **all ten slices
  burned** (registry now 18). Nothing clears 0.95 either: best daily DSR 0.031, best 1-hour
  0.003.
- **DEPTH 2 DOES NOT PAY FOR ITS SEARCH.** The paired read evaluates each family's best
  candidate on the same holdout window from one read, each charged its own 24 trials: depth
  two wins **1 of 10** datasets (TSLA, +0.2664), mean holdout delta **−0.3134**, and it loses
  worst exactly where depth 1 had its largest gain (ADA, −1.1766). Neither family clears its
  hurdle anywhere.
- **EACH RESOLUTION IS ADJUDICATED ON ITS OWN HURDLE.** `significance_report` reports the
  daily and intraday verdicts side by side — rows, charge, dispersion, hurdle, best row,
  count clearing the floor — and is reproduced for tranche 1 by `--recompute-dispersion`,
  because a single pooled verdict hid that the two face 1.989 and 44.196. A print bug that
  read the registry total as this tranche's burn count was fixed with it, and the writer now
  resolves `--output` relative to the repository instead of crashing on a relative path.

Added: `tests/test_quantum_signal_tranche2.py` (6),
`benchmarks/quantum_compilation/quantum_signal_ledger_tranche2.json`. Updated:
`scripts/measure_quantum_signal_ledger.py` (tranche-2 datasets, 1-hour resolution,
paired depth read, `significance_report`, print fixes), `docs/QUANTUM_SIGNAL_LEDGER.md`,
`docs/DOCUMENTATION.md`.

## 2026-10-31 — The quantum signal map gets fitted, and the fit pays for itself at the holdout

- **A PARAMETERISED MAP WITH ITS OWN TRIAL LEDGER.** `ParameterisedQuantumMap` is
  `Ry(alpha_i x_i + beta_i)` with `CX(i, 0)` fan-in, so `<Z_0> = prod_i cos(alpha_i x_i +
  beta_i)` for any parameter values and the simulator stays checkable against a closed form.
  `fit_parameterised_map` chooses `(alpha, beta)` from a declared 12-point grid by in-sample
  Sharpe on the training slice and writes **one trial per grid point** into a
  `CandidateTrialLedger` with `selection_pool_size = 12`. The charge is the ledger's
  `effective_trial_count`, taken as its **growth** for that fit: a fit on the fifth dataset
  is not billed for the first four.
- **THE FAMILY IS CHARGED FOR THE SEARCH, NOT THE WINNER.** 190 trials = 60 threshold rows
  (2 maps x 3 thresholds x 10 datasets) + 10 baselines + **120 fits**, `var_trials =
  358.76`. Fitting is now priced the same way the threshold sweep is, which is the whole
  point: the previous record's 14 trials were the sweep alone.
- **TEN DATASETS AND A SECOND BAR RESOLUTION.** Six daily panels (AAPL, BTC, MSFT, NVDA,
  ETH, SOL) and four 15-minute snapshots (BTC, ETH, SOL, BNB), annualised at 365 and 35,040
  bars respectively. The intraday taker column is quote-denominated in this snapshot — its
  ratio to volume equals the bar's price on every symbol — so the producer converts it and
  **records the measured ratio** instead of assuming it (check: every ratio within 0.25 of 1).
- **A REAL HOLDOUT, CONSUMED AND BURNED.** 70/30 temporal split per dataset, parameters
  fitted on training bars only, one decision per dataset in a durable
  `FullLineageExperimentLedger` (`quantum_signal_holdouts.jsonl`). Decision rule, stated in
  the producer: pass when the holdout nets **and** changes position in the window. **8 of 10
  slices burned**; SOL@1d (+1.4285) and NVDA@1d (+0.4589) confirmed. AAPL is the row that
  rule earns: +0.2191 of holdout PnL with **zero** turnover — a short carried in from the
  training boundary — so it is rejected and burned. Two post-burn reuse attempts are
  recorded with the ledger's own `HoldoutContaminationError`.
- **WHAT THE NUMBERS SAY.** Five of six daily fits make money in sample and lose it out of
  sample (BTC@1d +1.4032 → −0.7678, ETH@1d +1.5073 → −0.4314). SOL's threshold map nets
  +2.7560 against its momentum baseline's +4.1581 on half the turnover. Every 15-minute row
  nets negative with fees of the same order as gross. No row clears DSR 0.95, and the record
  states why the hurdle is 52.06: the family variance is pooled across resolutions and is
  dominated by the 15-minute Sharpes, so the pooling over-charges rather than flatters.

- **THE HURDLE IS NOW RESOLUTION-LOCAL, AND THE FIX NEEDED NO NEW DATA.** Pooling daily and
  15-minute Sharpes into one dispersion made the deflation's variance 358.76 — dominated by
  the 35,040-bar rows, which span −66 to −10 — and left every DSR at 0.000. The dispersion is
  now taken within each resolution (0.5235 at 1 day, 258.53 at 15 minutes) while every row is
  still charged all 190 trials, so the daily hurdle falls from 52.06 to **1.989** and the
  15-minute hurdle is 44.196. Nothing becomes significant: the best daily row is the SOL
  momentum baseline at DSR **0.234**, then SOL's three-feature map at 0.207 and BTC@1d at
  0.022. The re-price is `--recompute-dispersion`, which reads only the trial statistics the
  record already stores and touches neither a panel nor a holdout slice.
- **A SECOND ANSATZ DEPTH, AND A REFUSAL INSTEAD OF A FALSE FORMULA.** `ParameterisedQuantumMap`
  takes `layers`, and `ANSATZ_DEPTHS = (1, 2)` puts both families in the fit grid (24 points
  per dataset). Depth two has **no product closed form**: the second rotation's `X` branches
  survive the first entangler, measured as **0.7146 against 0.0002** for depth-one-squared on
  the same input. `analytic` therefore refuses above one layer and the circuit is checked
  against `exact_marginal`, a dense-unitary build on the `qubit 0 = LSB` convention that shares
  no code with the simulator — it caught a real bug in the embedding on the first run.
- **A COST-AWARE HOLDOUT RULE.** `holdout_decision` claims a slice only if the candidate
  trades in the window, nets after fees, **and** clears `expected_max_sr` of its own search,
  so a wider grid has to prove more before the holdout may confirm it; failing burns the
  slice. This takes effect on the next tranche — the committed record's slices are spent, and
  no burned slice is re-decided to accommodate a new rule.

Updated: `apedp_dyn/quantum_compilation/quantum_signal.py` (`ParameterisedQuantumMap` with
`layers`, `fit_grid`, `ANSATZ_DEPTHS`, `exact_marginal`, `fit_parameterised_map`,
`holdout_decision`, `FitResult`), `scripts/measure_quantum_signal_ledger.py`
(10 datasets, two resolutions, fit charge, holdout consumption, 14 checks,
`--recompute-dispersion`), `benchmarks/quantum_compilation/quantum_signal_ledger.json`
(schema v2, re-priced), `benchmarks/quantum_compilation/quantum_signal_holdouts.jsonl`,
`docs/QUANTUM_SIGNAL_LEDGER.md`, `docs/DOCUMENTATION.md`,
`tests/test_quantum_signal_ledger.py` (33).

## 2026-10-31 — Experiment E gets a second graph source: Watts–Strogatz

- **A GRAPH SOURCE THAT IS NEITHER THE LATTICE NOR THE TREE.** Every Experiment E target
  was a square-lattice cluster or a seeded tree-plus-edges draw — both connected, both
  shaped by the construction the scheduler might be reading. `watts_strogatz_target`
  starts from a ring lattice of even degree `k` and rewires each ring edge with
  probability `beta` to a non-neighbour, so the draw has triangles, odd cycles and a
  degree distribution set by the rewire. `(k, beta, seed)` reproduces the edge set, and
  the edges are what the stabilizers encode, so the target's SHA-256 pins the draw.
- **MEASURED ON IT, NOT ASSERTED.** Running the Experiment E path on three cells —
  32 qubits `k=4`, 48 `k=4`, 64 `k=6`, all `beta=0.1` — gives 15 rows, every one `VALID`
  against its specification, and `graph_layers` attains the degree floor at every cell:
  **5, 5 and 8 layers** against floors of 5, 5 and 8, while the raster orders need 58, 82
  and 115. The scheduler saves a layer on two of the three (48- and 64-qubit cells), so
  the class-2 handling is exercised on this family too.
- **THE SMALL-WORLD CONTRAST IS A ROW.** At width 32, `k=4`, seed 0: mean clustering
  **0.500** at `beta=0` (the ring lattice, where exactly three of each vertex's six
  neighbour pairs are joined), **0.468** at `beta=0.1`, **0.140** at `beta=1`, all at mean
  degree 4.0 — so the middle row is the small-world regime rather than a random graph
  wearing its name.
- **THE CONSTRUCTOR TARGETS ARE SPLIT FROM THE REFUSAL TARGETS.** `watts_strogatz` is in
  a new `SYNTHESIS_GRAPH_TARGETS` tuple used by the five constructor families, not in
  `GRAPH_TARGETS`, because that tuple also scopes `graph_swap_gate` — a deliberately
  wrong circuit. Sharing it would have made a correct small-world circuit report a
  refusal row; the split is what the first test run caught.

Added: `tests/test_vqc_watts_strogatz.py` (7), `apedp_dyn/vqc/synthesis.py`
(`watts_strogatz_target`, `small_world_shape`). Updated: `scripts/measure_vqc_synthesis.py`
(`experiment_e_second_source`, wired into Experiment E as `second_graph_source` with three
checks).

## 2026-10-31 — A quantum signal the ledger can price, and the demonstration that it could not

- **THE DEMONSTRATION WAS A CONSTANT.** The `phase`-then-`h`-then-`cx` circuit printed as a
  "2-feature quantum trading signal" returns **0.0 for every input**: `phase` is diagonal and
  inert on `|0>`, and `h; cx` leaves the measured wire maximally mixed. Measured on nine
  input pairs, one distinct output; removing the phases changes nothing. The claim that the
  score feeds `run_ledger` is also not an API — the ledger builds its own seven components
  and takes no `signal=` argument.
- **A MAP WHERE EVERY FEATURE REACHES THE MARGINAL.** `Ry(pi*x_i)` on every wire, `CX(i, 0)`
  for every `i >= 1`, so `<Z_0> = prod_i cos(pi*x_i)`; the simulator is checked against that
  closed form on all 703 AAPL and 1,045 BTC bars per map (deviation `0.0`), and a phase-only
  encoding is pinned as the counterexample, so a silent collapse to a constant fails loudly.
- **THE EXECUTOR IS PROVEN TO BE THE LEDGER.** `execute_signal` re-implements
  `gross = pos*fwd`, `fee = FEE*|dpos|`, `net = gross - fee` and reproduces `run_ledger` row
  for row on the ledger's own score — **722 + 1,064 rows, maximum difference 0.0** in
  position, fee and net. Causality is measured rather than asserted: deleting the last 50
  bars leaves every earlier score bit-identical.
- **THE RESULT IS A FAILURE TO CLEAR THE FLOOR.** 14 disclosed trials (2 maps x 3 thresholds
  x 2 assets + 2 momentum baselines), 5 bps per unit turnover, deflated together
  (`var_trials = 0.3989`). The best trial is BTC with `H-MOM+CVD+H-BB` at threshold 0.2:
  gross +1.4103, fees 0.3605, net **+1.0498**, **DSR 0.320**. Nothing clears 0.95 and
  `significant_rows` is empty. On AAPL the 2-feature map's gross +0.3728 becomes net +0.0838
  because it trades 578 units of turnover against the baseline's 253 — the same edge with a
  bigger bill — and the 3-feature map's gross is negative on AAPL while positive on BTC, so
  the fixed map is not asset-robust.
- **BOUNDARIES STATED IN THE RECORD.** No trainable parameters, so no quantum learning is
  tested; the output is classically computable in `O(n)` arithmetic, so no advantage is
  claimed; two vendored panels, one bar resolution, no slippage beyond the fee.

Added: `apedp_dyn/quantum_compilation/quantum_signal.py`,
`scripts/measure_quantum_signal_ledger.py`,
`benchmarks/quantum_compilation/quantum_signal_ledger.json`, `tests/test_quantum_signal_ledger.py` (20),
`docs/QUANTUM_SIGNAL_LEDGER.md`. Updated: `docs/DOCUMENTATION.md`.

## 2026-10-31 — K9 in the measured table, the lookahead counterexample, and a one-point frontier

- **K9 IS AN EXPERIMENT E TARGET.** 36 edges, degree 8, and no 8-layer schedule exists
  (36 > 8·4), so the measured minimum is **9** — one above the degree floor, proved by the
  capacity count at zero search nodes — and the peel's **15** is shown against it. The
  scheduler's six saved layers now appear in the timing table beside the clusters instead
  of only in the colouring block. Total Experiment E targets: seven, all promoted.
  Routing stays the cluster corpus (90 placements / 72 comparisons): K9 is a complete
  graph rather than a lattice, so it is timed and decided but not placed.
- **THE LOOKAHEAD COUNTEREXAMPLE IS PINNED.** A 9-qubit balanced tree on a line: greedy
  **17** mapped layers, lookahead **19** — two deeper, with one extra swap. The 100-qubit
  rows improve under lookahead and this row is worse; both hashes and depths are stored,
  so the caveat is measured rather than narrated.
- **THE PLACEMENT FRONTIER COLLAPSES TO ONE POINT.** Sorting the sixteen placements by
  (mapped depth, CNOT-equivalent two-qubit gates), over distinct pairs, leaves the balanced
  tree on all-to-all alone: in this corpus the two costs are ordered identically and there
  is **no** depth/gate trade-off to adjudicate. The frontier size is reported as the result
  instead of being dressed up as a trade-off, and duplicates are collapsed so one point
  measured twice cannot look like two.
- **SABRE, SWEPT OVER FIVE SEEDS.** `experiment_d.sabre_seeds` runs Qiskit's router under
  seeds 0–4 on the six routed cases; all 30 circuits are decided `VALID` on the layout
  Qiskit reports. The interval is real: the tree on a ring spans **120–144** layers
  (spread 24), the lattice tree 68–73, the line tree 166–176, the lattice chain 188–192;
  a chain on a line or ring needs no routing and is seed-invariant. One SABRE run
  understates the depth by up to 24 layers here, so the sweep is the measurement.
- **NINE NEW CHECKS, 81 IN TOTAL.** The class-2 target is in the measured table with 9
  layers against a floor of 8 and a positive saving; the regression case is deeper under
  lookahead; the frontier is nondominated and partitions the distinct placements; the
  window and boundary checks accept exactly one minimum above the floor, on K9; and the
  SABRE sweep covers every case and seed, verifies every routed circuit on its own
  layout, recomputes the intervals, and never reports a partial sweep as a full one.

Updated: `scripts/measure_vqc_synthesis.py`, `benchmarks/quantum_compilation/vqc_synthesis.json`,
`tests/test_vqc_synthesis_record.py`, `docs/VQC_SYNTHESIS.md` §5.1 and §5.3,
`docs/VQC_PHASE0_CAPABILITY_INVENTORY.md` and `docs/DOCUMENTATION.md`.

## 2026-10-31 — A third referee on unitaries, a swept window, two bounds and an overshoot rate

Five additions to the synthesis record, each measuring something it previously asserted
or left as a single point.

- **QCEC, ON THE UNITARY QUESTION.** `third_referee` hands MQT QCEC the same circuits and
  asks whether they compute the same unitary up to one global phase — stricter than the
  state-preparation referee, so a wire permutation would pass one and fail the other. The
  reference is the specification's own preparation (`h` on every wire, one `cz` per edge
  in sorted order), built in the script, so no family is compared with itself. It decided
  all 11 pairs: the five 16-qubit schedules of one edge set are `equivalent` (one unitary,
  five schedules), the 25-qubit irregular graph and the **900-qubit lattice** are
  `equivalent`, and the three wrong constructions are `not_equivalent`. Every decided row
  agrees with the stabilizer tier. No pair was beyond its reach, so the record holds **no
  abstention** — stated rather than counted as a win.
- **THE LOOKAHEAD WINDOW, SWEPT.** `experiment_d.window_sweep` runs windows 2, 4, 8, 16
  and 32 over all eight placements: 40 rows of depth, swaps, two-qubit gates and wall
  clock. Window 8 reproduces the recorded lookahead rows exactly by SHA-256, all-to-all
  is window-independent (the control), and depth is **not** monotone in the window. The
  best-depth total is 969 against 1,003 at window 8: the balanced tree on the lattice
  gains at 32 (84 → 74), the chain prefers 2. No ordering between windows is claimed.
- **BOTH COUNTING BOUNDS, ON EVERY TARGET.** Each Experiment E target now carries the
  layer-capacity bound `Δ·⌊V/2⌋` and the matching bound `⌈|E|/ν⌉` with `ν` a maximum
  matching — never weaker than the capacity bound when there is no perfect matching.
  On the six class-1 targets neither fires, so their minimality rests on the search or
  the peel; on K9 the capacity bound is what refutes `Δ` (36 > 8·4). The block carries
  both a silence and a refutation, and the checks recompute both counts from the rows.
- **THE Δ+1 OVERSHOOT, MEASURED.** On 41 random graphs whose `Δ`-colouring this module
  proves, Vizing's construction spends the extra layer on **27 (66%)**, never more than
  one. On a class-1 graph the construction alone would overshoot two times in three,
  which is why the `Δ` search runs first.
- **K13 ADDED TO THE CLASS-2 BLOCK.** 78 edges cannot fit into 12 layers of 6, so the
  capacity bound refutes `Δ` at zero nodes: the peel's 15 layers become 13, another 2
  saved, taking the class-2 total to **14**.
- **TWELVE NEW CHECKS, 72 IN TOTAL.** Every third-referee row answered (a verdict or a
  reason); every decided row agreeing; the counts recomputing from the rows; the wrong
  constructions `not_equivalent`; the sweep covering every case and window; every window
  circuit `VALID`; window 8 identical to the recorded router; all-to-all invariant; both
  bounds on every target; neither firing; and the overshoot rates recomputing.

Updated: `scripts/measure_vqc_synthesis.py`, `benchmarks/quantum_compilation/vqc_synthesis.json`,
`tests/test_vqc_synthesis_record.py`, `docs/VQC_SYNTHESIS.md` §5.3 and §7,
`docs/VQC_PHASE0_CAPABILITY_INVENTORY.md` and `docs/DOCUMENTATION.md`.

## 2026-10-31 — Wide referee agreement: two widths, and the refusals decided too

The referee-agreement corpus was confined to circuits both in-package tiers can decide,
which is at most ten qubits — the dense tier's bound. So "two referees agree" was a claim
about a narrow band of widths: it said nothing about the widths this record actually
builds (100, 400, 900 qubits), and nothing at all about whether a second referee refuses
a wrong circuit.

- **TWO WIDE ROWS, EACH DECIDED TWICE.** `referee_agreement.wide.rows` holds the
  Experiment E minimum schedules for `cluster_20x20` (400 qubits, 760 edges) and
  `cluster_30x30` (900 qubits, 1,740 edges) — the same circuits the timing rows quote,
  compared by SHA-256. The stabilizer tier answers `VALID` on all 400 and all 900
  generators; **Stim** 1.16.0, a tableau simulator this package did not write, answers
  `VALID` with the same generators satisfied. Neither consults the other.
- **THE REFUSALS ARE DECIDED TOO.** The same three wrong constructions Experiment E
  refuses — a dropped edge, an added edge, one `cz`→`cx` — are decided again at 400
  qubits, and both referees answer `INVALID`. Stim's row carries the generators it
  failed, not just a count: a third party only ever seen accepting has not been shown to
  refuse anything.
- **A DISAGREEMENT STOPS THE RECORD.** A comparative verdict that differs from the
  stabilizer tier's now raises before anything is written, so `agree` is never `false` in
  a written record. Two independent referees contradicting each other is not a
  measurement to publish.
- **THE REFUSALS THAT CANNOT EXIST ARE ROWS.** `wide.dense_tier` stores what the dense
  tier answers at 400 and at 900 qubits — *"the dense tier is bounded at 10 qubits and
  this construction acts on 400"* — instead of leaving the absence to be inferred.
- **AN ABSTENTION IS NOT AN AGREEMENT.** If Stim is absent or refuses the circuit, the
  rows read `UNAVAILABLE`/`REFUSED` with the reason and `agree` is `null`; a check accepts
  either branch and requires the record to keep the two apart.
- **SEVEN NEW CHECKS, 60 IN TOTAL.** Both widths reached and `VALID`; every second
  opinion either a comparative verdict or an abstention with a reason; no verdict
  disagreeing anywhere in the wide block; both agreeing circuits are Experiment E's timed
  circuits by SHA-256; both referees refuse all three wrong constructions; Stim names the
  generators it failed; and the dense tier refuses every wide width with the bound in its
  message.
- **WHAT IT DOES NOT BUY.** Stim decides the prepared state, not the schedule; it is
  Clifford-only; and the IR-to-Stim bridge is this package's, so a shared translation bug
  would fool both. That is a limitation on the record, not a footnote in a module.

Updated: `scripts/measure_vqc_synthesis.py`, `benchmarks/quantum_compilation/vqc_synthesis.json`,
`tests/test_vqc_synthesis_record.py`, `docs/VQC_SYNTHESIS.md` §5.2, `docs/VQC_CORE.md`
§8.3.1 and `docs/DOCUMENTATION.md`.

## 2026-10-31 — Every schedule clock is the median of three repeats

The seconds columns were one run each. At 400 qubits a single decision takes 0.2-0.3 s
and at 900 qubits 1.3 s, and on a loaded host those numbers move by tens of percent
between runs — which is the size of the differences the columns are read *for* (whether
the deciding clock is about ten times the building one at every size, and whether one
schedule is cheaper to build or decide than another). A column that moves by the size
of its own effect is not a measurement to read a trend from, so each phase is now timed
three times and the row quotes the median, with all three samples kept beside it.

- **THREE REPEATS PER PHASE, MEDIAN RECORDED.** `TIMING_REPEATS = 3` in
  `scripts/measure_vqc_synthesis.py` times three rebuilds and three decisions per
  schedule. The rebuild hash is now checked on **every** rebuild, not only the first,
  so the third timing cannot drift onto a different circuit; the circuit the decisions
  are timed on is the last rebuild, which is hash-identical by that check.
- **THE SAMPLES STAY WITH THE MEDIAN.** Each row gains `synthesis_samples` and
  `verification_samples`, and `experiment_e.timing` records the statistic (`median`),
  the repeat count and why. The median is recomputable from the row, and a median of
  three is one of its own samples rather than an average — both are pinned.
- **WHAT MOVED.** At 400 qubits the three deciding samples span up to **43%** of their
  median (median span 11%), so the single-run columns were indeed noisier than the gap
  between schedules. The ladder now reads build **0.001-0.002 s** and decide
  **0.015-0.017 s** at 100 qubits, **0.020-0.032 / 0.233-0.288** at 400 and
  **0.101-0.160 / 1.269-1.369** at 900, and the 900-qubit deciding range narrowed from
  1.283-1.451 to 1.269-1.369. Each target still totals **0.14 s, 2.36 s and 11.34 s**
  (five schedules plus three refusals), and deciding is still about ten times building
  at every size.
- **TWO NEW CHECKS, 53 IN TOTAL.** One requires every timed row to carry exactly three
  samples per phase with the quoted clock equal to their median; the other requires a
  row to carry samples if and only if it carries a clock, so a `NOT_CONSTRUCTIBLE` row
  cannot report a number without a measurement behind it. `test_vqc_synthesis_record.py`
  pins the same three properties against the committed record, including that the
  median is one of its own samples.
- **Position unchanged.** Deciding is still flat across schedules at a given size — at
  900 qubits the 5-layer schedule costs 1.37 s and the 117-layer walk 1.31 s — so the
  median changes the precision of the ladder, not any conclusion drawn from it.

## 2026-10-31 — The cluster ladder is timed at a third lattice size: 30x30, 900 qubits

Experiment E timed its five schedules on a 16-, 100- and 400-qubit cluster, which is a
small checkable case and two points on a line. Two points cannot show whether the cost
of *deciding* a schedule grows faster than the cost of *building* it — the referee runs
one stabilizer check per qubit, so if anything does, it should — and the 4x4 target is
too small to extrapolate from. A 30x30 lattice adds the third point: 900 qubits, 1,740
edges, still degree 4, so the schedules are the same five and only the width changes.

- **A NEW TARGET, NOT A LONGER ROUTING TASK.** `CLUSTER_SHAPES` gains `(30, 30)`; the
  synthesis record now carries **44 rows across six targets, 30 of them `VALID`**, and
  the 900-qubit lattice reproduces the same structure as the others: `χ′ = Δ = 4`, so
  the minimum schedule is **5 layers** against the row-major walk's **117** (tree-first
  61, random 14) at the same 2,640 operations and 1,740 `cz` gates. Its depth is a
  property of the edge set, which is why adding the size changes every count and no
  conclusion.
- **THE LADDER IS MEASURED, AND THE RATIO IS FLAT.** Build **0.001-0.002 s** and decide
  **0.014-0.017 s** at 100 qubits, **0.019-0.030 / 0.234-0.267** at 400, and
  **0.105-0.165 / 1.283-1.451** at 900; the whole target (five schedules plus three
  refusals) takes **0.14 s, 2.14 s and 11.85 s** in that order. Deciding costs **7-14x**
  building at *every* size, so the worry the third point was added to test does not
  appear: what grows is the target, not the asymmetry between the two phases.
- **ROUTING STOPS AT 400 QUBITS, AND NAMES THE TARGET IT DID NOT ROUTE.** A new
  `ROUTING_MAX_QUBITS = 400` keeps the routing task at the three smaller lattices: the
  greedy router needs tens of seconds for one placement at 900 qubits and the task runs
  thirty placements per size. The record carries `not_routed` with the target and the
  reason rather than silently dropping rows, the **90 placements and 72 comparisons are
  unchanged**, and two new checks pin that no routed row exceeds the declared width and
  that the excluded target is named.
- **THREE NEW CHECKS, 51 IN TOTAL.** The ladder must be exactly `[100, 400, 900]` with
  the widest row at 900 and verification dearer than synthesis there; routing must stay
  inside its width with a non-empty exclusion list; and its coverage check now counts
  the routed targets instead of a hard-coded three.
- **The depth-histogram block of `vqc_external_comparison.json` follows the target.** It
  places six depths now, and the 900-qubit row is the one with the widest extrapolation:
  depth 5 against the 64-qubit cell (0.07x the target's width), still with **0 of 78**
  corpus rows at or below it. The single like-for-like row is unchanged — the 64-qubit
  irregular graph is still the only target the corpus also runs.

## 2026-10-31 — Experiment E's depths are placed in the Qiskit corpus's depth histogram

`VQC_SYNTHESIS` §5.3 reported five target depths and nothing to read them against. The
nearest comparison already in the repository is the 78-instance corpus Qiskit
synthesises in `vqc_external_comparison.json`, but that corpus runs at 12, 20, 32, 48
and 64 qubits while Experiment E's targets are at 16, 25, 64, 100 and 400. One width
overlaps. A single histogram of all 78 depths with the five targets dropped into it
would therefore have put a 400-qubit depth beside a 12-qubit one and called it a
comparison, so the block places each target against the cell at its own width, names
the nearest cell and the width ratio where there is none, and says which row is the
like-for-like one.

- **A NEW `qiskit_depth_context` BLOCK.** `scripts/compare_vqc_external_tools.py`
  computes it from the corpus rows it already measured (no new Qiskit call) plus the
  five Experiment E specifications rebuilt at their minimum schedule (`graph_layers`),
  each rebuilt circuit decided `VALID` by the same referee before its depth is quoted.
  It carries the whole histogram (bucket width 10, empty buckets included), the
  per-width cells with their depths, and one placement row per target.
- **WHAT IT MEASURED.** The 78 corpus circuits are **10-126 layers deep, median 18**
  (43 rows in 10-19, 20 in 20-29, then 4, 6, 3, and single rows at 94 and 126 — the
  three random stabilizer anchors are the entire tail, and every graph-state row is
  between 10 and 55). Against that: **0 of 78 rows** have a Qiskit circuit of five
  layers or fewer, so all three cluster depths sit below the corpus's own minimum of
  10 rather than below its median. The irregular targets are *inside* the corpus — 13
  layers at 25 qubits clears 5 of the 16 rows in the 20-qubit cell (31%), and the
  64-qubit target's 17 layers clear 5 of the 15 rows in its own cell (33%), whose
  median is 20.
- **ONE ROW IS A WIDTH MATCH, AND IT IS LABELLED AS THE ONLY ONE.** The corpus has no
  16-, 25-, 100- or 400-qubit cell, so those four rows record `width_match: nearest`
  with the ratio (0.75, 0.80, 0.64, 0.16) and `like_for_like` contains exactly
  `irregular_64q_seed23`. The cluster's five layers are the lattice's `1 + χ′` — a
  property of a bipartite edge set any floor-attaining schedule would report — so the
  record states the position without claiming a depth advantage, and points at the
  shared-corpus and routing tasks for the engine-against-engine measurement on the
  *same* instances.
- **FIVE NEW CHECKS**, including that the histogram accounts for every corpus row, that
  the placement recomputes from the cell it names, that only a width-matched target is
  ever called like-for-like, and that recomputing the block from the rows reproduces
  it. `tests/test_vqc_external_comparison.py` rebuilds every placed circuit and
  compares its depth and SHA-256, so a stale number fails rather than persisting.

## 2026-10-31 — `Δ + 1` is constructed, and `Δ` itself can be refuted by counting

The scheduler's `Δ + 1` fallback was a second search, so it could fail the same way the
`Δ` search did: two budgets exhausted meant `status: budget_exhausted` and the greedy
peel's layers. But Vizing's theorem is *constructive* — its proof colours an edge with
`Δ + 1` colours — so there was nothing to search for. Chasing that to the odd complete
graphs then exposed the other half of the problem: for `K9` and `K11` the `Δ` search
cannot finish either, so the family could only ever be reported as *undecided*. It is
now refuted by a count that costs no search at all.

- **VIZING'S PROOF IS NOW THE CODE.** `apedp_dyn/vqc/colouring.py` gains
  `vizing_colouring`, the fan argument in its Misra–Gries form: build a maximal fan at
  one endpoint, take a colour `c` free at the fan's root and `d` free at its tip,
  invert the maximal `c`/`d` path through the root, rotate the fan prefix that keeps its
  colour conditions, and place `d`. One fan, one inversion and one rotation colour one
  edge in a single pass, so `Δ + 1` colours always suffice and no search is involved.
  The construction verifies its own output — every edge coloured once, each layer a
  matching, at most `Δ + 1` layers — and raises `ColouringConstructionError` rather
  than return a partial colouring; `greedy_peel` now stands only as a last resort for
  that case, labelled `status: construction_refused`.
- **A LAYER IS A MATCHING, SO THE EDGE COUNT BOUNDS THE COLOURS.** `Δ` layers hold at
  most `Δ·⌊V/2⌋` edges; when the graph has more, no `Δ`-colouring exists and the graph
  is class 2 by counting (`status: delta_refuted_by_layer_capacity`, zero search
  nodes). It is checked before the search, so it decides what the search cannot — and
  it is cross-checked against that search on every graph of at most five vertices, so
  a refutation is never taken on faith.
- **THE ODD COMPLETE GRAPHS ARE NOW MEASURED, NOT UNDECIDED.** `K9` (36 edges, `Δ = 8`)
  and `K11` (55 edges, `Δ = 10`) are class 2, and their `Δ`-colouring search needs more
  than five million nodes (`K11` far more), so neither budget nor patience would have
  settled them. The count refutes `Δ` in one multiplication and the constructed
  `Δ + 1` schedule is minimal: **9** layers where the peel uses **15**, and **11**
  where the peel uses **15**. Both are `optimal=True`, identical at budget 1 and at
  budget 200, and both are now rows of the synthesis record beside `K5`.
- **THE RECORD'S CLASS-2 BLOCK COVERS BOTH KINDS OF PROOF.** `K5`, `C5`, `K9` and
  `K11` are refuted by layer capacity at zero nodes; the Petersen graph is the tight
  case (15 edges fit exactly into three layers of five), so its refutation is still an
exhausted search in 31 nodes. The five rows save **12** layers in total (2 + 6 + 4
  against a 15-layer peel on the odd complete graphs), and each row records its
  `proof_kind`, so "class 2" is never a single undifferentiated claim.
- **THE CONSTRUCTION IS CHECKED EXHAUSTIVELY, NOT BY EXAMPLE.** Pins run it on **every**
  simple graph of at most five vertices (1,094 graphs: validity of the partition and
  `len(layers) <= Δ + 1`), and on the families where fans actually rotate — `K9`, a
  9-cycle, Petersen, the 4-cube, `K5,5` — plus an injected construction failure that
  must fall back to the labelled peel. Outside the pins it was also run over `K2`-`K30`,
  cycles to `C60`, hypercubes to `Q6` and random regular graphs to 60 vertices with no
  invalid colouring.
- **EXPERIMENT E ITSELF DID NOT MOVE.** The `Δ` search, its pruning, its budget,
  `greedy_peel` and the `EdgeColouring` fields are unchanged, and the five Experiment E
  targets still save **0** layers (the peel already attained `Δ` there) with `K5`'s 2
  unchanged. The record `benchmarks/quantum_compilation/vqc_synthesis.json` was
  regenerated: two new class-2 rows, a new `colouring_sha256`, and a check that the odd
  complete graphs really are refuted by the bound rather than by a search that stopped
  early — **49 checks**, all true.
- Pins: `tests/test_vqc_colouring.py` (14), `tests/test_vqc_synthesis.py` (57),
  `tests/test_vqc_synthesis_record.py` (31).

## 2026-10-31 — MQT QCEC wired for the equivalence task, on pairs built to disagree

The external comparison named three tools and wired two. QCEC was the third, and the
record said why it was left out: *the comparison that would matter is a corpus where
the two disagree, and it does not exist yet*. Building that corpus was the task, and it
is what the numbers below are.

- **THE CONSTRUCTION IS THE ARBITER.** `apedp_dyn/vqc/equivalence_corpus.py` holds 18
  pairs in six families, each carrying the algebraic identity that fixes its answer —
  `X Y Z = i I`, `RZ(2*pi) = -I`, `SWAP = CX CX CX`, `Z_0 Z_1` stabilising a GHZ,
  `U_c = RX(theta) U_r` with a non-scalar rotation. Neither tool adjudicates the other:
  the corpus is a test of this package first (every declared expectation must be met on
  every scope, or the run refuses to write the record) and a comparison second.
- **THREE AXES, KEPT APART.** *Tolerance*: one angle from `1e-13` to `1e-3`, every point
  exactly inequivalent; this package's edge is at `1e-9` on the per-amplitude distance
  and QCEC answers `equivalent` at `1e-8`, so four ladder rows are recorded as a
  threshold difference rather than an agreement or a contradiction. *Scope*: a global
  phase is free standalone and `INVALID` under control, and the GHZ pair is one state
  from two unitaries — the state verdict is `VALID`, the unitary verdict `INVALID`, and
  QCEC's `not_equivalent` agrees with the second, which is why that row is recorded as a
  scope difference and not as a contradiction. *Abstention*: two 12-qubit non-Clifford
  pairs are refused here (the reason names `'t'` and the tier) and decided by QCEC, the
  12-qubit Clifford pair is decided by both, and neither abstention is scored.
- **IDENTICAL CALLS DO NOT ALWAYS GET THE SAME ANSWER.** From `1e-7` up, repeated QCEC
  calls on the same pair returned `equivalent`, `equivalent_up_to_global_phase`,
  `no_information` and `not_equivalent` — its simulation checker is randomised unless
  seeded, and its combination rule turns a probable verdict into no conclusion. So each
  ladder point is asked **15 times** and the record carries every criterion; a row whose
  criteria differ is `unstable`, not a verdict. The adapter is `qcec_equivalence` in
  `external.py`, given the tool's own method selection, its own default numerical
  tolerance (`2.27e-13` as declared — which is not the number its verdicts flip at) and
  no timeout. The circuits go over gate-for-gate, because Qiskit's OpenQASM 2 importer
  defines neither `swap` nor `cswap` and a text interchange there would have meant the
  other tool deciding a circuit this package did not hand it.
- **WRITING THE CORPUS FOUND A FALSE `VALID` IN OUR OWN REFEREE.**
  `adjudicate_clifford_equivalence(on="state")` reported a GHZ state and its
  `CZ`-flipped twin — overlap exactly zero — as the same state. Two conventions were
  folded together: the mask represents `X^a Z^b` while the argument is a string of Pauli
  matrices (`Y` is `i X Z`, not `X + Z`), and `X^a Z^b · X^c Z^d` carries a `(-1)^{b.c}`
  that adding the two phases drops, so a sign depended on which rows the elimination
  picked. `+YYX` was read as a stabilizer of an orthogonal GHZ state, and `Y_0 Y_1` was
  not found in the group of a Bell pair at all. Both are fixed in `stabilizer.py`
  (`_product_phase`, plus the Y count on both sides of the mask), and the new test
  enumerates all 64 Pauli strings of random 3-qubit states against the group built by
  multiplying generators **and** against the dense expectation.
- **ON THE RECORD — 18 PAIRS.** 10 agree, four differ only at the threshold, one is
  scope, two are our abstentions and one carries no single QCEC criterion; **no
  divergence is left unexplained** and none of our verdicts differs from what its
  construction requires. (A second run can move a ladder row between
  `DIFFER_BY_THRESHOLD` and `UNSTABLE` — that movement *is* the instability being
  reported, which is why the checks are structural and the rows carry the criteria.) `benchmarks/quantum_compilation/vqc_external_comparison.json`
  is regenerated at schema `v8` with **67 checks**, all true, and all three tools are
  now `compared`.
- **PINS.** `tests/test_vqc_qcec_equivalence.py` (33): the corpus expectations on all
  three scopes, the ladder's distances and both edges, the diagonal pair whose Z-basis
  probabilities agree to `1e-12` while the unitaries do not, the classifier's five
  outcomes and the one it must never invent, the summary recomputing and noticing a
  tampered verdict, the gate table being total over `GATE_SPECS`, the aggregation rule,
  and the record's own rows. `tests/test_vqc_stabilizer.py` (+7): the exhaustive group
  membership and the dense-expectation cross-check. Record pins updated for `v8`.

## 2026-10-31 — Peak memory for both sampling engines, on the samples the record holds

Throughput without memory is half a comparison, and the record said so in its own
limitations. It now measures both engines' peak memory — and the naive measurement does
not work, which is why the numbers are shaped the way they are.

- **ONE ENGINE PER INTERPRETER.** `VmHWM` never falls, so two engines measured in one
  process cannot be separated: the second can never come out below the first even if the
  first freed everything. Each engine therefore runs in a fresh child.
- **THE WINDOW IS CUT AFTER THE IMPORTS.** A whole child's peak is ~165 MiB of interpreter
  and numpy against a few megabytes of samples; quoting it would be quoting numpy. The
  child resets the peak counter (`/proc/self/clear_refs` writes 5, which the kernel
  honours) after the engine's imports and after rebuilding the specification, so the peak
  it reports is the sampling step's own; `floor_bytes` is reported beside it, so a reader
  can see what was excluded rather than take it on trust. Stim's library is loaded
  *inside* the floor, so an engine is never charged for a library it cannot run without.
- **THE MEASUREMENT IS BOUND TO THE RECORDED SAMPLES.** The child rebuilds the target
  from its printed stabilizers, refuses to measure unless the rebuild hashes to what it
  was sent, and returns the count hash of the samples it drew. Every memory row is
  checked against the count hash already in the record, so the memory belongs to these
  samples and not to a similar workload. (This caught a real bug while it was being
  written: GHZ declares a support, so a rebuild without it hashed differently and was
  refused — the refusal is the feature.)
- **ON THE RECORD — 10 WORKLOADS, BOTH ENGINES.** Median growth 0.8 MiB (this package)
  against 3.0 MiB (Stim) at 12 qubits, and 1.4 against 3.4 MiB across 16–32 qubits, over
  an import floor of 164 and 169 MiB charged to neither. The number that moves is the
  **outcome table**, not the width: a 32-qubit GHZ keeps two outcomes and grows 0.2 MiB,
  while a 32-qubit random state with 20,000 distinct outcomes grows 8.6 MiB here and
  10.0 MiB for Stim — where both engines are measuring the same 20,000-entry table. So
  memory is comparable and slightly in this package's favour on small tables, and is not
  an order of magnitude either way. Peak RSS is resident growth rather than bytes
  allocated, one run rather than a median, and no ceiling claim: Stim's 20,000-qubit
  workload is not measured.
- **PINS.** `tests/test_vqc_memory.py` (9): the counter is a real peak; the reset makes a
  window measurable where a monotone counter could only report the earlier value; a
  target round-trips hash-identically (support included) and malformed strings are
  refused; a measured engine's count hash equals the caller's own; Stim's library is in
  the floor; a platform that cannot reset the counter abstains with a reason; a child
  that sampled another specification is refused; two runs agree on everything but the
  bytes. Record pins (+2) and 3 new checks; record regenerated at schema `v7` with **56
  checks**, all true.

## 2026-10-31 — The sampling comparison above 12 qubits: exact marginals as the referee

The sampling comparison stopped at 12 qubits, and the reason was the referee, not
either sampler: the exact outcome distribution has `2**n` cells and that is enumerable
only up to 12. Above that width the record abstained. It no longer has to, and the
abstention is replaced by a narrower claim that is measured rather than asserted.

- **THE REFEREE IS THE SPECIFICATION'S MARGINAL, COMPUTED NOT SAMPLED.**
  `apedp_dyn/vqc/marginals.py` derives the exact Z-basis distribution of a chosen subset
  of wires by Gaussian elimination over GF(2) on the specification's own stabilizers:
  the distribution of a stabilizer state is uniform over an affine subspace, so a
  marginal is that subspace's projection, `2**|T|` outcomes and nothing exponential in
  the state's width. A 10-wire marginal of a 32-qubit state takes about a millisecond and
  never touches a sampler, a circuit or the sampling code path. Every marginal is
  checked against the rank it predicts before it is returned — support size and a sum of
  one — because a wrong marginal is a wrong referee.
- **THE VERDICT IS A TEST, NOT A BAND.** Each subset's projected samples are decided by a
  chi-square goodness-of-fit against its exact marginal, Bonferroni-corrected over the
  subsets tested (`min(1, m * p)`, recorded per subset). The survival function is
  implemented exactly enough to pin against textbook critical values and keeps SciPy out
  of the referee; a subset whose supported cells have fewer than five expected counts
  abstains as `UNRESOLVED` with the count that failed, because a chi-square there is the
  wrong test rather than a small error. The first design used a simulated TV band, and it
  was replaced for a measurable reason: a 1e-3 quantile cannot be estimated from a few
  hundred simulated draws, so the band moved by 15% between null seeds.
- **ON THE RECORD — 16, 24 AND 32 QUBITS.** Both engines are sampled on three targets at
  each width, 20,000 shots each, and decided on **190 subsets** (every adjacent pair of
  wires plus three 10-wire projections: contiguous, strided, seeded draw). All 190 are
  `CONSISTENT` on this package's sampler and on Stim's, and no subset disagrees between
  the two engines. GHZ's pair marginals keep their two-outcome correlation at `p = 1/2`,
  so a mis-wired qubit is caught by the pairs that contain it — pinned, not assumed.
- **WHAT IT DOES NOT DECIDE, IN THE RECORD'S OWN WORDS.** A marginal is a projection, not
  the distribution: a state differing only on wires no tested subset covers passes, which
  is why every row lists its subsets. And every graph state is uniform in the Z basis, so
  no Z-basis marginal separates one graph state from another — the ring-graph row is in
  the block to *show* that (34 of 34 marginals uniform), not to claim a verdict it cannot
  support. The full distribution is still only decided at 12 qubits.
- **PINS.** `tests/test_vqc_marginals.py` (26): every marginal against a dense partial
  trace of a state built by projecting a full-support vector onto the specification's
  eigenspaces — the route that caught the first version's Y-generator sign — signs
  honoured rather than read as labels, graph-state uniformity as arithmetic, a 32-qubit
  marginal under a cost bound (linear algebra, not enumeration), the chi-square function
  against textbook critical values, the Bonferroni product as arithmetic, a mis-wired wire
  caught by the subsets containing it **and only those**, `UNRESOLVED` for too few
  expected counts, a sample set from another specification refused, and the blind spot
  measured: samples of a different graph state pass. Record pins (+4) and regenerated at
  schema `v7`; **53 checks**, all true.

## 2026-10-31 — A third-party referee for the corpus: adjudication stops being self-referential

The shared corpus was still decided by this package's own specification referee — and
one side of that comparison is this package. A referee that both builds and judges is
exactly the weakness the record's limitations named, so the corpus is now decided twice.

- **THE SECOND OPINION IS STIM, AND IT DECIDES THE SAME QUESTION.**
  `apedp_dyn/vqc/foreign_referee.py` hands the circuit to Stim, simulates it as a
  stabilizer tableau and evaluates every generator of the frozen specification as a
  Pauli observable. The specification is the same signed Pauli strings both engines were
  given, so the third party is not deciding a looser property; a pure stabilizer state
  is the unique +1 eigenstate of `n` independent stabilizers, which is why no amplitude
  is computed and a 127-wire routed circuit is decided in about a hundredth of a
  second. **150 synthesis decisions and 150 routed decisions — ours and Qiskit's — all
  `VALID`, with zero disagreements against this package's referee.**
- **IT ABSTAINS RATHER THAN GUESSES.** Only `VALID`/`INVALID` are verdicts about a
  circuit. An absent Stim is `UNAVAILABLE`; a non-Clifford gate, a measured circuit, a
  classically conditioned gate or a width mismatch is `REFUSED`, each naming the
  operation that stopped it. Neither is scored as a loss, and a `REFUSED` is never
  counted as agreement either — the row records that the referees could not be compared.
- **WHAT IT DOES NOT SETTLE, IN THE RECORD'S OWN WORDS.** Stim decides the prepared
  state, not gate counts, depth or wall clock, which remain this package's accounting;
  it is Clifford-only, so the dense-tier syntheses elsewhere in this package are outside
  its reach; and the bridge from this package's IR to Stim's circuit is ours, so a
  translation bug would be invisible to both referees. A second third-party
  implementation that does not share our IR, and a third-party resource accounting, are
  still missing — `what_would_change_it` says so instead of claiming neutrality.
- **PINS.** `tests/test_vqc_foreign_referee.py` (11): same specification decided; sign of
  a generator honoured (|01> against |00> and against a negative-phase spec); `INVALID`
  for a dropped gate, a wrong target, a mis-wired qubit; `REFUSED` with the operation
  named for a `t` gate, a measurement and a width mismatch; `UNAVAILABLE` for an absent
  Stim and never a verdict; 127-wire routed circuit decided. Record pins (3 new): every
  matched row carries both referees' verdicts, the two agree, the summary recomputes
  from the rows, and the abstention paths carry their reasons. Record regenerated at
  schema `v6`; **42 checks**, all true.

## 2026-10-31 — The corpus routed by each engine's own router: the abstract gap does not survive

Every routed row in the record so far used *this package's* router, including on Qiskit's
circuit — which measures this package's router, not Qiskit's. The sampled corpus is now
routed by the router each engine ships, on the same recorded device, and the answer is
not the one the abstract rows suggest.

- **THE TASK, NOT A THEOREM.** All 75 matched corpus instances are routed onto the
  recorded `fake_sherbrooke` coupling map (127 qubits, 144 edges) by **Qiskit's own
  transpiler** — its initial-layout search, SABRE, seed 1234, optimisation level 1, its
  final layout kept — and by **this package's lookahead router**, which starts from the
  identity placement and appends its restore ladder. Both sides are decided by the same
  referee: Qiskit's against the target relabelled by its final layout, this package's
  against the target padded with idle wires. All 150 routed circuits are `VALID`.
- **THE 3.25× ABSTRACT DEPTH GAP IS ROUTED AWAY.** The sampled synthesis rows report the
  grammar shallower on every instance, median 3.25×. After routing, the median ratio is
  **0.97×** [quartiles 0.88–1.10, range 0.59–1.39, bootstrap interval on the median
  0.93–1.04], and Qiskit is the shallower of the two on **40 of 75** instances against
  this package's 31 (4 ties). Reported as a distribution, not a line: per family —
  random 3-regular 0.88× [0.82–0.98] (Qiskit wins 19 of 25), 4-regular 1.00×
  [0.91–1.06] (12/12/1), seeded irregular 1.05× [0.96–1.11] (this package wins 14 of 25) —
  and per width, where the two disagree with each other: Qiskit ahead at 12–48 qubits
  (0.95, 0.93, 0.96, 0.99) and this package ahead at 64 qubits (1.09× [1.01–1.13], 11 of
  15).
- **WHY, AND WHAT IS NOT PRICED.** This package's router swaps more in every family —
  350 against 206 at the median across the sample — because Qiskit searches an initial
  layout while this package starts from the identity placement on a sparse device graph
  and then restores. So the routed rows compare each tool *as shipped*, not the same
  optimisation problem on both sides; the pre-restore routed depth is the comparable
  number and the total is reported beside it. That asymmetry is in the record's
  limitations, not implied away.
- **IT IS A COUNTERWEIGHT, NOT A CONCESSION.** The synthesis section's claim was already
  that the depth saving is a schedule difference a rescheduling pass could produce; the
  routed sample is the measurement of that claim, and it is now quoted in the record's
  `advantage.why` beside the abstract gap.
- Pins: `tests/test_vqc_external_comparison.py` (38, two new routed-corpus tests) — the
  routed circuit is hash-identical to the one the synthesis row measured, each engine's
  own router is named on its own row with Qiskit's seed and optimisation level, both
  circuits `VALID`, and every routed median, quartile, win count, swap figure and share
  is recomputed from the rows per family and per width. Record regenerated; 36 checks,
  all true.

## 2026-10-31 — The shared corpus becomes a sample, and reports its spread

The external record's shared corpus was seven hand-picked instances. That fixed the
"only the favourable target" objection and left the next one standing: with one draw per
family, the row that happened to flatter an engine was the family's result. The corpus
is now a celled sample and the record reports a distribution instead of a row.

- **THE SAMPLE IS THE DESIGN, AND IT IS ARITHMETIC.** Three graph families (3-regular,
  4-regular, seeded irregular) at five widths (12, 20, 32, 48, 64 qubits) with five
  independent draws per cell, plus one random stabilizer anchor at each of 12, 20 and 32
  qubits: **78 instances, 75 matched, 3 abstentions**. Every seed is computed, not
  listed — `seed = SEED_BASE[family] + 10 * width + replicate` — every label names its
  cell (`regular3_32q_3`), and the whole corpus still rebuilds from seeds alone, which
  the record re-checks before it will write.
- **EVERY FAMILY IS REPORTED AS A DISTRIBUTION.** Per family and overall: the grammar's
  and Qiskit's own spreads (min, q1, median, q3, max, IQR, stdev), the spread of the
  per-instance ratio `qiskit/grammar`, the paired difference, the share of instances on
  each side, a bootstrap interval on the median ratio that resamples these instances (and
  says so), the ``(family, width)`` medians, and — for the graph inputs — the spread of
  edge counts and mean degree, so the sample's structure is visible too. The record's
  verdict text quotes the spread, not a single number.
- **THE RESULT SURVIVES THE WIDENING, AND THE WIDTH TREND EXPLAINS IT.** The two engines
  use the **same two-qubit count on all 75 matched instances** — one entangler per edge —
  so there is still no gate-count advantage to report. The grammar is shallower on **75
  of 75**, median ratio **3.25×** with quartiles 2.75–3.75 and range 1.50–5.00; by
  family 3.50× [3.00–3.75] (3-regular), 4.00× [3.20–4.40] (4-regular) and 2.65×
  [2.38–3.20] (irregular). The grammar's depth is fixed by the degree (4 layers cubic, 5
  quartic) and does not grow with width, while Qiskit's elimination tracks the qubit
  count, so the ratio widens from 12 to 64 qubits on an unchanged schedule. That is still
  a scheduling result a transpiler pass can reproduce, not synthesis superiority.
- **WHAT IT STILL DOES NOT SAY.** Five fixed width tiers, two generators, and instances
  at a width share that width: the spread is over these 78 instances, not over graph
  space, and the bootstrap interval is not a confidence statement about it. Both outputs
  are still decided by this package's own referee. Both limits are in the record's
  `limitations` and in `docs/VQC_CORE.md` §8.4.
- Pins: `tests/test_vqc_corpus.py` (11) and
  `tests/test_vqc_external_comparison.py` (36) — the new pins check that every cell was
  drawn at every declared width with the declared replicates, that the record's `design`
  block equals the rows it produced, and that each summary block's medians, quartiles,
  IQRs, ranges, win counts and bootstrap interval recompute from the rows. Record
  `benchmarks/quantum_compilation/vqc_external_comparison.json` regenerated at schema
  `v5`; all 30 checks true.

## 2026-10-30 — What routing does to the cluster schedule's advantage

Experiment E's schedules were measured on a machine where every qubit may talk to every
other. A depth advantage is a *logical* number until the circuit is routed, so the five
schedules were placed on a line, a ring and the cluster's own lattice — by both routers,
at 16, 100 and 400 qubits — and the gap was measured again after the SWAPs were paid for.

- **90 PLACEMENTS, 72 COMPARISONS, NONE INVERTED.** Every schedule was placed on every
topology by both routers at all three cluster sizes, and every placed circuit was
decided `VALID` by the same referee (the restore ladder keeps it the same unitary on the
same wires). The minimum schedule (`graph_layers`) is compared to each alternative twice:
the layers it was deeper logically and the layers it is deeper after routing.
- **ON THE LATTICE THE ADVANTAGE IS UNTOUCHED, AND THAT IS THE CONTROL.** The cluster's
edges *are* lattice edges, so the identity placement respects all of them: every family
needs **0 swaps**, keeps its logical depth, and the surviving fraction is exactly
**1.00×** for all 24 lattice comparisons.
- **ON A LINE OR RING IT GROWS.** Routing is not free for the minimum schedule either —
12,316 swaps at 400 qubits on a line, longest hop 40 — but the localized orders pay far
more: raster, column-major and tree-first all need **144,400** swaps with a longest hop of
**362**, because they emit gates across the whole register in an order no line can serve
locally. The mapped gap therefore exceeds the logical one: across all 72 comparisons the
surviving fraction is **1.00×-219×, median 4.67×**, and the minimum schedule is still the
shallowest circuit after routing — at 400 qubits on a line, 229 mapped layers against
column-major's 1,520 (a logical gap of 72 layers becomes 1,291). The strongest
alternative is column-major on a line and raster on a ring: still 7.8×-23.1× further
behind than it was logically.
- **WHAT IS ASSUMED.** Identity initial placement (no layout search), program order
routed as written (no reordering of commuting gates), the restore ladder counted in the
mapped depth, both routers reported, no device calibration or error rate. Those are in
the record's limitations, not implied away.
- Pins: `tests/test_vqc_synthesis_record.py` (31, including the 90-placement block and
the recomputation of every surviving fraction). Record
`benchmarks/quantum_compilation/vqc_synthesis.json` regenerated; **48 checks**, all true.

## 2026-10-30 — Experiment E at 400 qubits: the schedule's own wall clock

Experiment E stopped at 100 qubits, where building and deciding a schedule is too fast
to time on purpose. The largest target is now a **20x20 lattice cluster: 400 qubits,
760 edges**, and every schedule records what it cost to build and to decide.

- **THE TARGET IS THE LATTICE, CHECKED ARITHMETICALLY.** `cluster_state_target(20, 20)`
yields 400 stabilizers and `2*400 - 20 - 20 = 760` edges, maximum degree 4, so the
proven schedule floor is 4 `cz` layers. All five schedules are `VALID` at 400 qubits
(400 stabilizers checked by the same referee), and the two wrong constructions plus the
`cz`→`cx` swap are still `INVALID` at this width, so the refusals are not an artifact of
small graphs.
- **DEPTH AT 400 QUBITS.** The minimum schedule is **5** (4 layers) against **77** for
the row-major and column-major walks (76 layers), 41 for the breadth-first tree and 14
for the seeded random order — at exactly the same 1,160 operations and 760 two-qubit
gates. That is fifteen times shallower than the row-major walk, and it saves 0 layers
against the greedy peel, which had already attained `Δ` at this size too.
- **PER-SCHEDULE WALL CLOCK, NOT A HEADLINE NUMBER.** Each row now carries
`synthesis_seconds` (build the circuit from the specification) and
`verification_seconds` (the stabilizer referee deciding it). At 400 qubits: 0.019/0.231
raster, 0.021/0.263 column-major, 0.032/0.221 tree-first, 0.020/0.222 for the minimum
and 0.022/0.232 random. Building is cheap — one `h` per qubit and one `cz` per edge —
and deciding costs about ten times more because the referee checks all 400 stabilizers;
the whole 400-qubit target (five schedules and two refusals) takes 2.3 s. The five
schedules' times are within a few percent of each other, so depth is not bought with
time here — and no timing advantage is claimed for the minimum either, because every
family emits the same 1,160 operations and every decision is the same 400 stabilizer
checks.
- **THE TIMED CIRCUIT IS THE RECORDED CIRCUIT.** The timings are taken on a circuit
rebuilt from the same family and target, and the script **refuses to write the record**
if that circuit does not hash to the row it is attached to. One run, one process, a
warm rebuild on this host, moving by tens of percent between runs under load: the
record's limitations say all of that rather than implying a benchmark against another
engine.
- Pins: `tests/test_vqc_synthesis_record.py` (30, including the 400-qubit target and
the two seconds columns). Record `benchmarks/quantum_compilation/vqc_synthesis.json`
regenerated; **43 checks**, all true.

## 2026-10-30 — The `cz` schedule is now a proved minimum, not a greedy peel

Experiment E could only say the greedy peel *attained* the degree floor on the four
targets it was measured on, and it explicitly declined to claim optimality in general.
It now does: `graph_layers` computes a **minimum edge colouring**, and every schedule
it reports carries the argument for its size.

- **A SCHEDULE IS AN EDGE COLOURING.** Commuting `cz` gates share a layer exactly when
  their edges do not touch, so a layer is a matching and the minimum number of layers is
  the edge chromatic number `χ′`. `apedp_dyn/vqc/colouring.py` decides it: Vizing gives
  `Δ ≤ χ′ ≤ Δ + 1`, a complete backtracking search (colour-symmetry broken) finds a
  `Δ`-colouring -- optimal by the degree bound because a maximum-degree vertex needs one
  layer per incident edge -- and only when that search **exhausts its space** is a
  `Δ + 1` colouring constructed, with the exhaustion recorded as the proof that the
  graph is class 2. "Proven minimum" and "the best I found" are separate fields, and the
  search carries a node budget because deciding the `Δ` case is NP-complete; a
  budget-exhausted run returns the peel labelled `optimal=False`.
- **IT SAVES NOTHING ON EXPERIMENT E, AND THE RECORD SAYS SO.** On all four targets the
  old peel already attained `Δ`, so `total_layers_saved` is **0** (4, 4, 12 and 16
  layers; depths 5, 5, 13 and 17 unchanged) and every target is listed by name in
  `targets_where_greedy_was_already_optimal`. The replacement buys proof, not depth,
  there — presenting it as an Experiment E improvement would be false.
- **CLASS-2 GRAPHS ARE WHERE THE OLD SCHEDULE LOST.** Three graphs with `χ′ = Δ + 1` are
  now built and decided too: the odd cycle `C5` (3 layers, the peel was already 3), the
  Petersen graph (4, peel 4) and the complete graph `K5` (5 against the peel's **7**, so
  **2 layers saved**). The record proves the extra layer by exhaustion rather than
  assuming Vizing's slack is needed, and all three circuits ride the same referee
  (`VALID`, stabilizer tier).
- **GRACEFUL WHEN IT CANNOT FINISH.** A search that hits its budget returns the greedy
  peel with `status: budget_exhausted` and the reason in words; nothing is called minimal
  without a lower bound that matches. `greedy_peel` stays exported precisely so the
  saving is a measured column (`greedy_cz_layers`) rather than an assertion.
- Pins: `tests/test_vqc_colouring.py` (8), `tests/test_vqc_synthesis.py` (57),
  `tests/test_vqc_synthesis_record.py` (29). Record
  `benchmarks/quantum_compilation/vqc_synthesis.json` regenerated with the scheduler
  block, its per-target savings, the class-2 rows and `colouring_sha256`.

## 2026-10-30 — Qiskit's own transpiler on a recorded device map: a split routing result

The record's oldest routing caveat was that Qiskit's circuit was routed by *this*
package's greedy router. That comparison measured our router, twice. The device task
now gives Qiskit a real coupling map and its own transpiler.

- **A RECORDED DEVICE, NOT A SYNTHETIC LATTICE.** `qiskit_routing` builds a
  `CouplingMap` from the `fake_sherbrooke` snapshot (127 qubits, 144 edges, edge list
  pinned by SHA-256), transpiles with SABRE, re-imports through QASM 2, and reports
  both the routed depth measured here and Qiskit's own `depth()`. Four logical
  circuits: GHZ chain, GHZ balanced tree, ring 100, and Qiskit's own 5,149-gate GHZ.
- **LAYOUT, DECIDED NOT ASSUMED.** Qiskit keeps its final layout as metadata, so its
  circuit prepares the target *up to a permutation*. `relabelled_target` moves the
  specification onto the wires the device will hold it on, and the same referee
  decides it; a non-injective layout is refused. All four Qiskit circuits and all
  eight placements are `VALID`.
- **THE RESULT IS SPLIT, AND THAT IS THE FINDING.** Qiskit wins **2 of 4** routed-depth
  comparisons against each router (median ratio 1.47× against lookahead, 1.32× against
  greedy). It is shallower on the balanced tree (83 vs 95) and on its own 5,149-gate
  circuit (1,416 vs 7,253), where the grammar's edge ordering is exactly the schedule
  the router must undo; this package is shallower on the chain (116 vs 239) and the
  ring (117 vs 251). Both routers and both routed and total depths are reported, so the
  restore cost is visible rather than folded into one number.
- **WHAT IS STILL NOT CLAIMED.** One SABRE seed and optimisation level (stochastic
  router, one run); SWAPs priced as one two-qubit gate on both sides; no calibration,
  error model or measurement. MQT QCEC remains `UNAVAILABLE` with a reason.
- Schema is now `apedp.vqc.external_comparison.v4`; `qiskit-ibm-runtime` joins the
  `compare` extra for the device snapshots. Pins:
  `tests/test_vqc_external_comparison.py` (35).
  `benchmarks/quantum_compilation/vqc_external_comparison.json` regenerated.

## 2026-10-30 — Stim wired: the same specification, the same shots, the exact distribution as referee

Stim was the record's largest declared abstention. It overlaps with **execution**, not
synthesis, so the matched task is sampling: hand both engines the same signed
stabilizer strings and the same 40,000 shots, and decide each sample set against the
exact distribution rather than against the other engine's output.

- **BOTH ENGINES, ONE SPECIFICATION.** `stim_sampling` builds a `stim.Tableau` from the
  same stabilizers, synthesizes a preparation circuit and compiles a sampler;
  `apedp_sampling` clones this package's tableau and measures qubit by qubit. Both
  adapters record preparation and sampling clocks separately, because they prepare
  differently and folding the clocks together would hide which part is compared.
- **THE REFEREE HAS TEETH.** At 12 qubits the exact outcome distribution is
  enumerable, so `adjudicate_sampling` reports the total-variation distance against a
  band simulated from that distribution (a chi-square band would have thousands of
  empty cells), plus max-|z| and out-of-support counts as diagnostics. `sampling_pair`
  compares the two samplers under a null where **both** are correct, so "they agree"
  is not inferred from sampling noise. A width above the exact-enumeration limit is
  `SamplingUnsupported`; a skewed sample set is `INCONSISTENT`.
- **MEASURED.** GHZ 12, ring 12 and the corpus's random stabilizer 12, 40,000 shots
  each: every sample set `CONSISTENT` (TV 0.0000 / 0.1297 / 0.0428 for this package,
  0.0025 / 0.1247 / 0.0460 for Stim, all inside their bands) and all three pairs
  `AGREE`. Median throughput differs by about **three orders of magnitude** (thousands
  against millions of samples per second) — recorded as an implementation comparison
  (Python per-shot tableau clone against a compiled C++ sampler), explicitly not a
  ceiling for either engine and not a claim about Stim's 20,000-qubit published
  workload.
- **THE ABSTENTION IS STILL AN ABSTENTION WHERE IT APPLIES.** MQT QCEC is still
  `UNAVAILABLE` with a reason; Stim is listed as compared only for the sampling task,
  and its synthesis abstention is unchanged. `stim` is pinned in the record's
  reproducibility block (1.16.0).
- Schema is now `apedp.vqc.external_comparison.v3`; `stim>=1.12` joins the `compare`
  extra. Pins: `tests/test_vqc_external_comparison.py` (30).
  `benchmarks/quantum_compilation/vqc_external_comparison.json` regenerated.

## 2026-10-30 — A shared corpus: the external comparison now runs targets neither engine was written for

The external record's own `what_would_change_it` named a shared corpus as the first
condition for a real comparison. It was the right condition: the GHZ and ring rows
compare a grammar that *knows* those targets against a general synthesizer, and the
single random-state row only shows where the grammar abstains.

- **SEVEN INSTANCES, NO BESPOKE CONSTRUCTION ON EITHER SIDE.** `apedp_dyn/vqc/corpus.py`
  fixes a corpus reproducible from seeds alone: random connected 3- and 4-regular graph
  states drawn with the pairing model (16, 20, 24 and 32 qubits), one seeded irregular
  graph, and two random stabilizer states with no graph structure. A draw with a
  self-loop, a repeated edge or two components is rejected and redrawn, not repaired,
  so a "3-regular" target really is one.
- **THE HONEST RESULT: A GATE-COUNT TIE.** On all five graph states both engines use the
  **same number of two-qubit gates** — one entangler per edge, which is what a graph
  state costs — and the grammar's only saving is depth (5/7/6/5/9 against
  15/16/15/16/21), because it peels the edge set into parallel matchings while Qiskit
  emits the edges in sequence. A transpiler scheduling pass can produce the same
  reordering, so the record calls it a schedule difference and does **not** quote the
  2.5× median depth ratio as a synthesis advantage. The two random stabilizer rows keep
  the asymmetry in the same table: the grammar abstains, Qiskit synthesizes, and the
  abstention is unscored.
- **CHECKED, NOT TYPED.** The record refuses to write unless the corpus rebuilds hash
  for hash from its seeds, every row's two engines are on one `target_sha256`, every
  matched circuit is `VALID` under the same referee, no abstention carries metrics, and
  the summary (win counts, medians, totals) recomputes from the rows. Schema is now
  `apedp.vqc.external_comparison.v2`.
- **THE MISSING PIECE IS NAMED, NOT PAPERED OVER.** Both outputs are still decided by
  this package's own referee, and seven instances are not a distribution over graphs;
  the record's limitations and `what_would_change_it` say both, with a third party's
  verifier and per-tool routers still outstanding.
- Pins: `tests/test_vqc_corpus.py` (6), `tests/test_vqc_external_comparison.py` (24).
  `benchmarks/quantum_compilation/vqc_external_comparison.json` regenerated.
- **Correction (same day, below):** the depths quoted here (5/7/6/5/9) were measured
  under the greedy matching peel. `graph_layers` now runs a minimum edge colouring, so
  the regenerated record reads **4/5/4/4/9** against Qiskit's 15/16/15/16/21 and the
  median depth ratio is **3.75×**, not 2.5×. The gate-count tie — the point of the
  entry — is unchanged.

## 2026-10-30 — SWAP accounting closed: three entangler bases, and total depth separated from pre-restore depth

The review's one priority accounting item (2026-10-30, §6) was that `1,193 = 99 +
2(547)` counts a SWAP as an instruction, while a CNOT-only backend spends `99 + 6(547)
= 3,381`. Both numbers are right and neither was the other; the record now carries all
three bases and says which is which.

- **THREE TWO-QUBIT BASES, EVERY ROW.** `abstract_two_qubit_gates` counts each `cx`,
  `cz` and `swap` as one operation; `native_entanglers` prices a SWAP at the declared
  `swap_entanglers` (1 by default — a native-SWAP basis);
  `cx_equivalent_two_qubit_gates` always prices a SWAP at three CNOTs. The greedy
  100-qubit lattice tree is **1,193 / 1,193 / 3,381**; the lookahead one is
  **1,121 / 1,121 / 3,165**. Across the eight rows the lookahead saves **3,756
  CNOT-equivalents**, not just the 51 abstract operations the depth improvement made
  easy to notice.
- **A DECLARED BASIS, NOT AN ASSUMPTION.** `place(..., swap_entanglers=3)` declares a
  CNOT-only device, at which point the native column equals the CX-equivalent column;
  `0` is refused, because no SWAP costs less than one entangler. The default basis is
  stated in the record (`swap_accounting.swap_entanglers = 1`), so the headline count is
  no longer ambiguous.
- **TOTAL DEPTH SEPARATED FROM PRE-RESTORE DEPTH.** `mapped_depth` is the total circuit a
  device executes, restore ladder included; `depth_before_restore` excludes it; and
  `restore_depth_cost = mapped_depth - depth_before_restore` is now reported — 94
  layers for the greedy lattice tree, 35 for the lookahead one, 0 wherever routing was
  unnecessary. Previously both numbers were in the record but the prose table quoted
  one without saying which.
- **THE ARITHMETIC IS CHECKED, NOT TYPED.** The record refuses to write unless, for all
  16 placements, `abstract = logical + 2·swaps`, `cx = logical + 3·total_swaps`,
  `native = logical + swap_entanglers·total_swaps` and `restore_depth_cost = total −
  pre-restore`; the equivalence is pinned on the row the review named.
- **REFEREE-CORPUS HONESTY (review §2).** The two tiers share the circuit parser and one
  gate table (`gates.matrix_for` feeds both the Clifford conjugation tables and the
  dense amplitudes), so a wrong gate matrix would fool both; only the representation is
  independent. That is now written down beside the 23-circuit corpus, with randomised
  stabilizer circuits, longer sequences and adversarial specifications named as the
  corpus that would test the shared layer, and not claimed.
- **EXACT-DISTRIBUTION PRE-ALLOCATION PIN (review §4).** A 20-qubit state refuses
  `distribution()` with a traced peak under 1 MB, so the limit is checked before the
  exponential object exists rather than after a failed attempt to build it.
- Pins: `tests/test_vqc_placement.py` (34), `tests/test_vqc_stabilizer.py` (40),
  `tests/test_vqc_synthesis_record.py` (27). Records: `vqc_synthesis.json`
  (`experiment_d.swap_accounting`, per-row bases) and `vqc_external_comparison.json`
  regenerated.

## 2026-10-30 — Experiment D re-measured: the lookahead router cuts routing depth, the finding survives

Experiment D's router was greedy and its numbers were an upper bound on what a better
router might do. Both halves are now measured: the greedy router is preserved as the
baseline, and a **lookahead** router runs beside it on every construction/topology pair.

- **WHAT THE LOOKAHEAD DOES.** Every candidate is still one step along a shortest path,
  so each reduces the pair's distance by exactly one; what it chooses is *which* swap.
  Either operand may move, and each candidate is scheduled into the earliest layer its
  two nodes are free — so routing fills idle wires instead of extending the critical
  path — then ranked by the total wire distance of the next **eight** two-qubit gates
  that involve a qubit the swap moves, then by which qubit is needed later, then by the
  greedy direction. The first key is a scheduler (node availability); the second is the
  lookahead. Both routers are in `placement.py` and are selected by `strategy=`.
- **THE RE-MEASUREMENT, AT 100 QUBITS, AGAINST THE GREEDY NUMBERS.** Lattice chain:
  mapped depth **191 → 117** (−39%), 246 → 225 swaps, 591 → 549 two-qubit gates. Ring
  tree: **253 → 212** (−16%). Line tree: **311 → 278** (−11%). Lattice tree:
  **220 → 84** (−62%), 547 → 511 swaps, 1,193 → 1,121 two-qubit gates. The four rows that
  need no routing (chain on a line/ring/all-to-all, tree on all-to-all) are unchanged at
  0 swaps, which is the zero point: the lookahead cannot improve a free row.
- **THE FINDING SURVIVES, AND CHANGES SHAPE.** Under the greedy router the balanced
  tree lost on every connected topology; under the lookahead it wins on the lattice
  (84 against the chain's 117) and still loses on a line (278 against 100) and a ring
  (212 against 100), while all-to-all is the depth 12 it claims. "Depth 12 is not
  depth 12 on hardware" is therefore a statement about the topology *and* the router,
  and the record now carries both columns instead of one.
- **MORE SWAPS ARE NOT MORE DEPTH.** The lookahead often uses fewer swaps, but the
  deeper point is that it schedules them in parallel: the lattice tree finishes at
depth 84 with 511 swaps because they land on idle wires, where the greedy ladder of
  547 swaps serialises to 220. Swap count alone was never the cost.
- **IT IS A HEURISTIC, NOT A THEOREM.** At small widths it can lose — a 9-qubit
  balanced tree on a line routes two layers deeper than greedy — and neither router
  reorders gates, so commuting gates are never scheduled past each other. The window of
  eight and the tie-break order are choices; the record says so and pins the 100-qubit
  rows rather than claiming an ordering everywhere.
- **EVERY ROUTED CIRCUIT IS STILL DECIDED.** All 16 rows are `VALID` against the same
  specification: the restore ladder still telescopes the swap product to the identity,
  on both routers, so the improved circuit is verified rather than trusted.
- Pins: `tests/test_vqc_placement.py` (31, up from 19, with both routers under the
  adjacency and specification checks), `tests/test_vqc_synthesis_record.py` (26, with
  the comparison row by row). Record: `benchmarks/quantum_compilation/vqc_synthesis.json`
  (`experiment_d.rows` × 16 and `experiment_d.comparison`).

## 2026-10-30 — The dense referee's ceiling, measured: cheap at ten qubits, a wall at fifteen

`DENSE_SPEC_LIMIT = 10` was a policy with no price attached: the dense tier is the only
referee that decides a non-Clifford construction against a specification, and nothing
recorded what one decision costs or where it stops. It is now priced, width by width.

- **THE DOMINANT TERM IS THE OPERATOR, NOT THE STATE.** The tier materialises each
  stabilizer as a dense `2**n x 2**n` Pauli matrix (`np.kron` of single-qubit
  matrices), so one stabilizer costs `16 * 4**n` bytes — quadratic in the Hilbert
  dimension — while the state it is applied to is `16 * 2**n`. A decision runs `n` such
  projections, so the cost model is `n` matrix-vector products against `2**n`.
- **EVERY WIDTH MEASURED IN ITS OWN PROCESS, UNDER A DECLARED BUDGET.** `ru_maxrss` is
  a process high-water mark, so widths measured together would all report the largest
  peak; each child caps its address space at 8 GiB over its post-import baseline with
  `RLIMIT_AS`, so an over-budget width raises `MemoryError` instead of being killed by
  the kernel. At or below the bound the referee's own verdict is recorded beside the
  price; above it the referee still answers `UNRESOLVED` and only the price is new.
- **THE NUMBERS.** 6 qubits: 0.02 s / 0.2 MiB traced / 0.06 MiB operator. 8: 0.07 s /
  1.5 MiB / 1.0 MiB. **10 (the bound): 0.25 s / 20 MiB / 16 MiB, `VALID`.** 11: 1.31 s /
  80 MiB / 64 MiB. 12: 5.1 s / 320 MiB / 256 MiB. 13: 20.3 s / 1.25 GiB / 1.0 GiB. 14:
  88.4 s / 5.0 GiB / 4.0 GiB. 15: **16 GiB per operator, twice the budget**, refused in
  13 s after building the 4 GiB intermediates (5.2 GiB RSS).
- **WHERE IT BECOMES IMPRACTICAL.** Inside its declared bound the tier is cheap — 0.2 s
  and 20 MiB at ten qubits — so memory, not arithmetic, is what stops it. 13 qubits is
  the last comfortable width (20 s, 1.25 GiB), 14 is out of interactive reach (88 s,
  5 GiB) but fits the budget, and 15 does not fit. Peak grows **3.96–4.00x** per qubit
  and time **3.9–5.3x**, both measured rather than fitted.
- **THE BOUND IS PRICED, NOT LIFTED.** `dense_tier_cost` returns seconds, peak traced
  bytes and the phase-free distance with **no verdict field**, and no referee path
  calls it; `specification_state_vector` still refuses above the bound and
  `adjudicate_specification` still returns `UNRESOLVED`, so a caller allowed to exceed
  the bound cannot turn a price into an adjudication. Inside the bound the probe's
  distance is bit-identical to the referee's, which is what makes the price meaningful.
- **WHAT THE CEILING IS NOT.** It prices this implementation. A Pauli string can be
  applied in `O(2**n)` memory as single-qubit gates plus index permutations instead of
  a dense operator, which would move the wall by several qubits; that is stated and
  not measured, and none of these numbers is a claim about the question in principle.
- Pins: `tests/test_vqc_synthesis.py` (56), `tests/test_vqc_synthesis_record.py` (24).
  Record section `dense_referee_ceiling` in
  `benchmarks/quantum_compilation/vqc_synthesis.json` (schema v3).

## 2026-10-30 — VQC Experiment E: a cluster and an irregular graph as specifications

The synthesis grammar could build a GHZ and a ring, and the ring was the only target
whose formation was not a wiring choice. Experiment E makes that general: a **graph
state** is `X_v Z_N(v)` on every vertex, so its specification *is* an edge set, and the
only thing a construction family may choose is the schedule of the commuting `cz` gates.

- **THREE NEW SPECIFICATIONS, STILL NO REFERENCE CIRCUIT.** `cluster_state_target(rows,
  cols)` describes a two-dimensional cluster — the grid-graph state with qubits on the
  lattice **vertices** (not the surface-code cluster, whose qubits sit on lattice edges)
  — and `irregular_graph_target` / `seeded_graph_target` supply a graph with no regular
  structure, reproducible from its seed. Each validates its edge set: a self-loop, a
  repeated edge, an endpoint out of range and a disconnected graph are refused, because
  a graph state on a disconnected graph is a product state. `graph_of_state` reads the
  edges back out of the stabilizers, so the five families are handed no graph — they
  read one — and a specification with a `Y`, two `X`s or an asymmetric edge is refused
  by name rather than half-recognised.
- **THE SCHEDULE IS THE ONLY FREE VARIABLE, AND THE RECORD MEASURES IT.** The five
  families (wire order, transposed order, breadth-first tree then closures, a greedy
  peeling into parallel matchings, and a seeded random control) tie exactly on
  operations, two-qubit gates and edge span, and differ only in depth. All 20 valid rows
  verify. Because the connectivity is the specification's, the Pareto frontier
  **collapses to the peel** here — the opposite of Experiment A, where depth traded
  against connectivity — and the peel is the promoted candidate for all four targets.
- **A PROVEN FLOOR IS REACHED.** A qubit with `d` neighbours takes part in `d` `cz`
  gates and no two can share a layer, so `d` layers are unavoidable. At 100 qubits the
  10x10 cluster has 180 edges and degree floor 4, and the peel achieves **depth 5**
  against the row-major walk's **37** at the same 280 operations; on a 64-qubit seeded
  irregular graph with degree floor 16 it achieves depth 17 against 45. The record does
  *not* claim optimality in general: a graph whose edge chromatic number is
  `max_degree + 1` admits no matching peel that reaches the floor.
- **REFUSALS NAME THE EDGE, NOT JUST THE VERDICT.** Dropping a lattice edge, adding an
  edge the specification does not name, or replacing one `cz` with `cx` each leaves
  **exactly two** stabilizers absent from the prepared state's group — the two endpoints
  of that edge — and the record carries the Pauli strings. A 12-qubit GHZ chain decided
  against a 3x4 cluster is `INVALID`, and the cluster against the GHZ is `INVALID`, at a
  shared width: the referee is deciding the state, not accepting a family.
- **EXPERIMENT E FOUND A REAL BUG IN THE DENSE TIER.** The dense comparison read
  `StatevectorSimulator` amplitudes (bit `i` = qubit `i`) against a state built as a
  Kronecker product (qubit 0 most significant). Every earlier specification — GHZ, rings,
  square lattices — was invariant under reversing its wires, so the two orders were the
  same state and nothing could see it. An irregular graph is not invariant, and the
  naive comparison reported a distance of **0.0605** on an exactly correct state. One
  re-index (`_specification_order`) gives **2.8e-17**, and
  `test_the_two_basis_orders_are_pinned_on_an_asymmetric_state` pins both numbers, so
  the mistake cannot return silently.
- **BOTH REFEREES ON GRAPH STATES.** Six cases at every width both tiers can decide
  (8- and 10-qubit clusters, 9- and 10-qubit seeded irregular graphs, a 3-qubit path and
  one dropped-edge refusal): **5 accepted by both, 1 refused by both**, the accepted
  distances at `1e-17` and the refusal at `0.125`.
- Pins: `tests/test_vqc_synthesis.py` (53), `tests/test_vqc_synthesis_record.py` (21).
  Record: `benchmarks/quantum_compilation/vqc_synthesis.json` (schema v3).

## 2026-10-30 — APEDP-QED: a quantum-assisted proposer, and a negative result that holds up

The question is the one that can be falsified, not "can a circuit do physics": given
noisy observations and a fixed term library, does a QAOA-mediated search recover the
governing equation as well as the classical proposers **at the same number of objective
fits**? It is now measured, and the answer is no — for reasons that are themselves
measurements.

- **`apedp_dyn/qed/` — PROPOSE, NEVER ADJUDICATE.** A term library, a budgeted
  selection objective (`J(z) = mean residual² + λ|z|`, coefficients re-fitted per
  selection, one unit per *distinct* fit), a task carrying a concealed answer, six
  search arms, a referee and a `CandidateProposal` adapter so QED competes inside
  APEDP-X on the platform's usual terms (AGENTS.md §1.1). The referee reports
  `VALID`/`ALTERNATIVE`/`INVALID`/`ABSTAIN` **and** `recovered` separately, because a
  different equation that predicts the holdout as well as the truth is a
  non-identifiability result, not a win and not a failure.
- **QED-001: THE TASK IS FAIR, AND THAT IS SWEPT, NOT ASSUMED.** A damped oscillator
  (`x'' = −1.44x − 0.15v`), development and holdout from different initial conditions,
  1 % observation noise, ten-term library. Before any arm runs, the harness enumerates
  all 1,024 selections outside every budget: the penalised optimum **is** the
  generating support `{x, v}`, by a **0.464 %** margin over `{x, v, x³}`. Without that
  number a missed optimum would be indistinguishable from a badly posed task.
- **RESULT: 1/3 AGAINST 3/3 AT 256 FITS, AND 0/3 AT 64.** Hill climbing, simulated
  annealing, the GA and boolean GP all recovered the truth in 3 of 3 runs at 25 % of
  the space; the quantum arm in 1 of 3. At 64 fits (6.25 %) boolean GP held 3 of 3 and
  the quantum arm held none. The two budgets are both recorded, because at the
  generous one four classical arms **tie** — so that condition ranks nothing, and the
  lean one is where the arms separate.
- **THREE MEASUREMENTS EXPLAIN THE DEFICIT, RATHER THAN A VERDICT EXPLAINING IT.**
  (i) The quadratic surrogate's own minimiser is an eight-term selection, not the
  answer: a pairwise reconstruction under-estimates higher-order interactions and is
  not bounded below at the dense corner. (ii) A harness-side **circuit probe** re-runs
  the arm's circuit from its own seed and measures what probability it put on the
  generating support: `p = 0.0012` at ranks **347-447 of 1024**, with an effective
  selection count of ~749 — the circuit behaves as if choosing among most of the space
  and its mass sits where the surrogate's wrong minimum is. (iii) At 64 fits, 56 went
  to the surrogate and 8 candidates were scored, so the arm was a classical
  model-then-sample strategy wearing a circuit.
- **THE EXPERIMENT CAUGHT ITS OWN FALSE POSITIVE.** The first version of the arm
  reported the best selection the **whole run** evaluated, and it recovered the truth
  in 3 of 3 runs at *both* budgets — parity at 256 and a win at 64. The circuit probe
  is what disproved it: a circuit that ranks the answer 447th cannot be finding it
  through the circuit. It was not — the surrogate expansion had already scored every
  pair, so `{x, v}` was one of its own 56 fits, and the arm was crediting an
  **exhaustive pair enumeration** to the circuit. An arm may now declare its answer,
  and the QAOA arm declares only the best scored candidate. The same record then reads
  1/3 and 0/3. Deterministic either way: the difference is a design decision, not
  noise.
- **THE NON-IDENTIFIABILITY RESULT IS KEPT, NOT TIDIED AWAY.** The RIME arm returned
  `{x, v, sin(x)}` in every run: `R² = 0.9993` on the holdout, true coefficients inside
  their bootstrap intervals, and yet **parameter stability 0.0** and **not
  ablation-separable** — the `sin(x)` term can be deleted at no cost. The generating
  support is separable by contrast: removing `x` costs 2.07 of holdout `R²`, removing
  `v` costs 0.154, and the best addition buys `1.5 × 10⁻⁶`.
- **Pins:** `tests/test_qed_recovery.py` (37) and `tests/test_qed_recovery_record.py`
  (19). The record re-derives its own oracle from its own task specification, checks
  every status against its own numbers, and pins the probe to the arm it probes (same
  surrogate minimum, shots and depth) so a probe cannot drift into a lookalike. The
  surrogate is proved **exact** on a genuinely quadratic objective, which is what makes
  its error on a real one a measurement rather than a bug. Ten limitations are in the
  record and in [`docs/QED_QUANTUM_EQUATION_DISCOVERY.md`](docs/QED_QUANTUM_EQUATION_DISCOVERY.md):
  one synthetic system, finite-difference derivatives, a supplied kinematic identity,
  no hardware, no shot noise in the angles, three seeds, and no advantage claimed in
  either direction.

## 2026-10-30 — The comparison with an external tool, and the bug it exposed in the interchange

The external review's first requirement before any claim of differentiation was a
matched comparison with the tools that already do part of this. That is now measured,
with one rule: **a tool that did not run is not a tool that lost.**

- **`apedp_dyn/vqc/external.py`, AND QISKIT WIRED FOR THE SYNTHESIS TASK.**
  `synth_circuit_from_stabilizers` takes the same signed Pauli stabilizer strings this
  package holds, so both engines are given *the same specification*; its circuit comes
  back through QASM 2, is re-imported, and is decided by the specification referee — so
  its gate count is quoted only for a circuit that was independently verified. At 100
  qubits: GHZ **99 two-qubit gates** (grammar, depth 12) against **4,950** (Qiskit,
  depth 395), and a ring state **100 against 100** two-qubit gates with depth 101
  against 199.
- **THE ADVANTAGE IS TARGET-SPECIFIC, AND THE RECORD SAYS SO.** The GHZ row is 50x; the
  ring row is none at all, because a general Gaussian-elimination synthesizer reaches
  the same two-qubit count as the hand-written construction and loses only on depth. A
  grammar that selects among constructions it knows is not a synthesizer that finds
  one, so `advantage.claimed` is `false` and the prose verdict is generated from the
  measured ratios rather than written by hand.
- **THE HONEST COMPLEMENT: A RANDOM STABILIZER STATE.** On a random 30-qubit state this
  package has no construction family and **abstains**; Qiskit synthesizes it (643
  operations, 222 two-qubit gates) and the referee says `VALID`. Both rows are in the
  record, because a comparison that only ran the favourable case is not a comparison.
- **A REAL BUG FOUND BY THE COMPARISON.** Qiskit reads the first character of a
  stabilizer string as the **highest**-indexed qubit; this package reads it as qubit 0.
  Handed over unconverted, the result is the *mirror image* of the requested state —
  which verifies for a GHZ and for a ring, because both are palindromic, and fails for
  every other target. It was invisible until a random stabilizer state entered the
  corpus. The adapter reverses the strings, and the referee (not the conversion code)
  confirms the reversal; a two-qubit `X`-on-0/`Z`-on-1 target now pins it, and the test
  also demonstrates that the naive order really does produce the mirror image.
- **ABSTENTIONS ARE FIRST-CLASS.** `probe()` reports `compared`,
  `installed_not_wired` or `unavailable` with a reason, and only the first counts as a
  comparison. Stim and MQT QCEC are `UNAVAILABLE` here with the workload a matched
  comparison would need written down; the record explicitly does not treat that as a
  loss, does not treat the 1,000-qubit rows as a state-of-the-art claim, and notes that
  the topology row routes **Qiskit's** circuit with *our* greedy router rather than its
  transpiler against a coupling map.
- **Pins:** `tests/test_vqc_external_comparison.py` (20). Qiskit is an optional extra
  (`apedp[compare]`) that only the harness imports. Regenerating the correctness audit
  for the new record moved no pinned total and re-froze its hash with a fourth recorded
  amendment; the Qiskit-dependent tests skip cleanly when it is absent while the record
  and the abstention pins run either way.

## 2026-10-30 — M3, first slice: the two referees agree, and depth is measured on a topology

Review 2 recommended freezing the synthesis milestone and starting M3 with
"independent stabilizer/dense referee agreement tests" and "architecture-aware resource
accounting". Both are measured here rather than asserted, and the second one changes a
conclusion the previous milestone reported: **depth 12 is not depth 12 on hardware.**

- **BOTH REFEREES NOW ANSWER THE SAME QUESTION.** `adjudicate_specification` grew a
  keyword-only `tier=` override (`auto` / `stabilizer` / `dense`), and
  `referee_agreement()` runs **both** tiers on one circuit and reports the two verdicts
  beside each other. A named tier that cannot decide a circuit raises instead of
  falling back -- a quiet fallback would make the comparison a referee against itself.
  Measured on 23 circuits (GHZ chains and balanced trees at 2..10 qubits, ring graph
  states at 3..6, one `cx`→`cz` mutation): **22 accepted by both, 1 refused by both, no
  disagreement anywhere.** The accepted rows are decided *exactly* (the dense tier's
  phase-free distance is ≤ 1e-12, rounding rather than tolerance); on the refused row
  the tiers fail differently and still agree on the verdict.
- **THE FOUR CONDITIONS ARE PINNED WHERE THEY ARE ENFORCED.** `n` generators for `n`
  qubits, mutual commutativity (even symplectic product for every pair), independence
  over GF(2) (rank `n`, so the group has `2**n` elements and the `+1` eigenspace is
  unique) and Hermitian phases only (`+1`/`-1`). Normalisation and global-phase
  invariance are pinned on the dense referee: `X Y Z = iI` applied to a GHZ still
  verifies, and the accepted tolerance is printed in the check's own requirement.
- **EXPERIMENT D: `apedp_dyn/vqc/placement.py`.** Line, ring, rectangular-lattice and
  all-to-all connectivity graphs, plus a greedy SWAP router. Routing is a
  *relabelling*: the logical-to-node map is tracked exactly and the emitted swaps are
  undone in reverse order, so the swap product telescopes to the identity and the
  mapped circuit is the **same unitary on the same wires** -- which is why every routed
  circuit is decided by the same specification referee (all 8 rows VALID) instead of
  being trusted. Both depths are reported, and `depth_before_restore` /
  `restore_swap_gates` keep the cost of being verifiable separate from the cost of
  routing. At 100 qubits: the balanced tree's logical depth **12** becomes **220** on a
  10x10 lattice (547 swaps, 1,193 two-qubit gates), **253** on a ring and **311** on a
  line (2,415 swaps, 4,929 two-qubit gates), while a chain is depth 100 for free on a
  line and 191 on a lattice. On all-to-all both reproduce the logical circuit exactly,
  which is the known zero point that makes the other numbers readable.
- **PARITY CORPUS HARDENED, AND A HOLE IN THE HARNESS.** New sampler-corpus circuits
  with deterministic allowed-outcome sets: a mid-circuit reset with feedforward and a
  two-qubit condition, and both condition polarities on one measured bit. `assert_parity`
  used to tag a classically conditioned gate as an ordinary gate and then compare the
  Python tier (which applies it) with the native tier (which skips it) -- a parity
  failure that was really an unsupported input; it now refuses by name, and `ccx` is
  pinned as refused by both engines. The corpus also caught a mistake in the fixture
  before it could reach an engine: a `cx` controlled on a qubit that had just been
  reset is a no-op, so the feedforward was invisible until the control was moved.
- **THE RECORD SAYS WHAT THE ROUTER IS.** Greedy, no scheduler, no lookahead, no device
  calibration, no error rate, no classical-control cost: the mapped depths are upper
  bounds that a better router would improve, not a floor on what hardware must pay.
- **EXACT-DISTRIBUTION LIMITS, PINNED (review 1.B).** Both tiers already refused to
  enumerate past twelve qubits -- a stabilizer state has `n` generators but up to
  `2**n` outcomes, so the compact representation does not make the distribution compact
  -- and nothing pinned it. There is now a test on both engines at 13 qubits: the Python
  tier raises, the Rust tier declines into the documented fallback, and
  `probability_of_bitstring` and sampling stay exact, which is what the refusal points
  at. The failure is a refusal to allocate rather than an out-of-memory death.
- **BENCHMARK METHODOLOGY (review 1.C).** `scripts/measure_vqc_rust_parity.py` now takes
  five repeats instead of three, records every raw sample and the interquartile range
  for each side, and carries a `hardware` block naming the platform, CPU count,
  interpreter, NumPy, `rustc`, `cargo`, the release build and the process-wide peak RSS
  (with a note that a peak reached inside one process cannot be attributed to one
  case). Compilation is explicitly outside every timing: the extension is built once by
  `make vqc-native` before the script runs. Regenerating the record moved the ratios,
  which is the point of recording the spread: **3.8x / 7.2x / 4.4x / 14.8x / 3.5x**
  (images 100q, images 400q, sampling 100q x 256, distribution 10q, adjudication 100q),
  against 3.0x / 7.8x / 4.0x / 15.5x / 2.7x on the previous run of the same cases. Both
  are this host; the record now says so, and the IQR says how stable each one was.
  Pins: `tests/test_vqc_placement.py` (19), `tests/test_vqc_synthesis.py` (37),
  `tests/test_vqc_synthesis_record.py` (15), `tests/test_vqc_native_parity.py` (45),
  `tests/test_vqc_native_parity_record.py` (9).

## 2026-10-30 — Synthesis from a specification: a circuit built with no reference to compare against

Review asked for the next milestone after the rewrites loop: stop requiring an
existing reference circuit and construct from a mathematical specification. That is
Experiment A, and it is measured at 100 qubits and again at 10..1,000. Experiment C
(non-Clifford search) is not built; its verification boundary is measured and stated.

- **A TARGET IS NOW MATHEMATICS, NOT A CIRCUIT.** `apedp_dyn/vqc/synthesis.py`: a
  `TargetSpecification` is a qubit count plus `n` signed Pauli stabilizers (and an
  optional exact support), and it is validated as a pure state by the tableau's own
  structural check before any search runs -- anticommuting generators, rank-deficient
  sets, an imaginary phase or a support that does not sum to one are refused. There is
  no reference circuit anywhere in the object, and the referee's signature is
  `adjudicate_specification(circuit, target)`.
- **TWO REFEREES, NEITHER NEEDING A REFERENCE.** The stabilizer tier requires
  `phase_of(generator) == +1` for every specified generator, exactly and at any width.
  The dense tier compares the prepared amplitudes against the state **projected out of
  the specification's stabilizers** (`(I + (-1)^p P)/2` per generator), which is what
  lets it decide a *non-Clifford* construction: a GHZ prepared with `ry(pi/2)` is
  VALID at 3 and 8 qubits (distance 3.9e-17 and 6.9e-18) and `UNRESOLVED` at 12, where
  the reference, not the circuit, is the limitation.
- **EXPERIMENT A, 100 QUBITS:** five constructions from the GHZ specification all
  verify -- chain and reverse chain (100 operations, depth 100, edge span 1), star
  (100/100/99), balanced binary tree (100/**12**/50) and a seeded random tree
  (100/14/90). The balanced tree is promoted (`VALID`/`MET`/`SUFFICIENT`), the Pareto
  frontier keeps the nearest-neighbour chain beside it because depth and connectivity
  trade off, and four wrong constructions (drop the Hadamard, reverse the middle `cx`,
  add a Hadamard, `cx`→`cz`) are refused with the failing generator named. A second
  target -- a 100-qubit **ring graph state** -- is built and promoted, while the linear
  cluster (the same construction missing the closing edge) is refused with its two
  absent stabilizers named. The star is depth 100, not 2: its `cx` gates share a
  control and serialise, which is measured rather than assumed.
- **EXPERIMENT B:** 24 rows at 10, 50, 100, 250, 500 and 1,000 qubits, every one
  verified against the specification with `n` stabilizers checked; the 1,000-qubit
  constructions verify in 1.3-1.7 s each.
- **THE SAME ADMISSION POLICY, AND A BUG IT CAUGHT.** A synthesis candidate satisfies
  the new `Candidate` protocol and goes through the same sealed verdict: a valid
  construction offered under `{"depth": 1}` is **`VALID` and `UNMET`** and refused with
  "does not improve its resource claim (depth: claimed 1, measured 12)". Writing that
  test caught the claim check being folded into the semantic verdict in the first
  draft, which reported an unmet claim as `INVALID` -- a false statement about the
  circuit, and exactly the conflation `discovery.py` exists to prevent.
- **§6 of the review, both items.** The discovery record now carries
  `workload.reference_provenance` stating that its 222-operation reference is
  deliberately inefficient and that the 54.95% reduction is against inserted work, not
  an advantage over an optimised GHZ preparation (whose count is `n` operations and
  depth `n` -- what the search reaches); the limitation list repeats it, and two new
  pins assert the provenance and that the loop's admission still carries the four
  separate answers with every accepted step certified VALID.
- **WHAT IS NOT CLAIMED, IN THE MODULE AND IN THE RECORD.** The construction space is a
  grammar of named wireings, not a general synthesiser; two targets, not a family;
  the ranking is a stated preference with the frontier reported beside it; and no
  cheaper GHZ is claimed, since every family is `n` operations and `n-1` two-qubit
  gates. Pins: `tests/test_vqc_synthesis.py` (31) and
  `tests/test_vqc_synthesis_record.py` (13).

## 2026-10-30 — The VQC execution core goes Rust-first, with Python as the referee and the fallback

Review asked for a dual-language VQC: Rust for the mathematical execution core,
Python for research and adjudication, and no rewrite of the platform. It also asked
for five engineering rules, and specifically for a benchmark before any claim of
Rust superiority. All of that is here, measured.

- **PAULI ALGEBRA, THE STABILIZER TABLEAU, SAMPLING, EXACT DISTRIBUTIONS AND THE
  CLIFFORD IMAGES ARE NATIVE.** `rust/apedp-vqc` (module `apedp_vqc`) implements the
  review's first four modules — `pauli-core`, `stabilizer`, `clifford-equivalence`
  and circuit-IR parsing — over packed `u64` bit vectors rather than Python
  integers, with 31 pure-Rust tests (`cargo test --no-default-features`). The
  conjugation tables are **derived from the same gate matrices** as the Python
  tier's, not transcribed, and a test compares all 24 one-qubit and 48 two-qubit
  rules as integers.
- **ONE CIRCUIT SPECIFICATION, TWO READERS.** The crate parses the canonical JSON
  `Circuit.to_json()` emits — same schema version, same op tags, same gate names —
  and refuses what it cannot represent by name. A real divergence was caught this
  way: the first condition parser guessed `[[bit, value]]`, while the writer emits
  `{"bit": i, "value": bool}`; the teleportation parity test failed loudly instead
  of the engine silently skipping conditioned gates.
- **RUST FIRST, PYTHON AS THE FALLBACK, AND THE VERDICT STILL IN PYTHON.**
  `apedp_dyn/vqc/native.py` is the bridge: every entry point returns `None` when the
  fast path cannot answer, so `StabilizerBackend` and
  `adjudicate_clifford_equivalence` use it by default and fall back to the Python
  tier; with the wheel absent nothing changes but the speed. The referee never
  moved: Rust produces images, `equivalence.py` compares them and writes the verdict,
  and below `PARITY_LIMIT` (24) qubits it re-derives the table in Python and raises on
  a disagreement. Above it the report says `image_verified_by: "suite"` instead of
  implying a check that did not happen. A corrupted native table is refused, not
  adjudicated (regression test).
- **PROVENANCE IS RECORDED, NOT ASSERTED.** `RunResult.engine` is `"rust"` or
  `"python"`; `CliffordEquivalenceReport` gained `image_engine` and
  `image_verified_by`; `engine="rust"` raises rather than quietly downgrading;
  `native.tallies()` counts `rust`/`python`/`fallback` calls and
  `APEDP_VQC_PYTHON_ONLY=1` pins the reference path, which is how the benchmark
  proves the Python timings never touched the fast path.
- **MEASURED, ON IDENTICAL FROZEN INPUTS**
  (`benchmarks/quantum_compilation/vqc_native_parity.json`, median of three, same
  process, warm-up excluded, Python 3.12.3 / Linux x86-64): images at 100 qubits
  17.82 ms -> **5.86 ms** (3.0x) and at 400 qubits 460.82 ms -> **59.06 ms** (7.8x);
  100-qubit sampling at 256 shots 2527.08 ms -> **633.39 ms** (4.0x); the exact
  10-qubit distribution 38.19 ms -> **2.47 ms** (15.5x); a 100-qubit Clifford
  adjudication 45.34 ms -> **16.59 ms** (2.7x). The 100-qubit image row is the
  noisiest of the five (17-29 ms of Python), which the record says. The record refuses to be written if
  any case disagrees, and pins the SHA-256 of the script, the seven Rust sources and
  the three Python modules it measured.
- **THE TWO EXISTING RECORDS WERE REGENERATED, NOT RE-PINNED IN PLACE.** Both
  pinned the SHA-256 of the modules that produced them, and those modules changed,
  so `vqc_stabilizer_scaling.json` and `vqc_clifford_discovery.json` were re-run:
  the 1,000-qubit adjudication of two 7,500/8,190-gate circuits now takes **1.94 s**
  (about a minute before), the 100-qubit discovery loop **2.6 s** (5.9 s before)
  with the identical 222 -> 100 operations and 76 certified steps, and the
  stabilizer-tier scaling rows are re-measured on this host.
- **WHAT IS NOT PORTED, NAMED.** The dense statevector tier, the decoders, the noise
  engine and the discovery loop stay Python; MWPM remains exponential in defects;
  no memory ceiling and no parallelism is claimed. `docs/VQC_RUST_CORE.md` says why,
  per capability, with the review's five rules mapped to the code that enforces
  them. Pins: `tests/test_vqc_native_parity.py` (39) and
  `tests/test_vqc_native_parity_record.py` (7).
- **THE DQI BASELINE IS UNTOUCHED.** Adding `vqc_native_parity.json` meant
  regenerating `correctness_audit.json` so its scanned set matches the records on
  disk; the Phase 0 baseline was re-frozen with the reason recorded and its pinned
  totals are unchanged (checked 12, partial rows 13, resources-only 5,
  aggregate-only 51), with the same 13 DQI gaps.

## 2026-10-30 — The discovery loop, end to end at 100 qubits, and a non-vacuous evidence rule

Review asked for three things: an explicit rule that an empty check set is never a
pass, controlled-subroutine phase tests (already landed — see the entry below), and a
complete 100-qubit discovery-to-adjudication workflow. The first and third are here.

- **AN EMPTY CHECK SET IS `INSUFFICIENT` EVIDENCE, NOT A PASS.** `CandidateVerdict`
  now carries four separate answers — `semantic_verdict`, `resource_verdict`,
  `evidence_sufficiency` (`SUFFICIENT`/`UNDECIDED`/`INSUFFICIENT`) and
  `promotion_decision` — and `promote()` requires all three positives. `evidence` is
  derived from the checks in `__post_init__`, so no caller can set it, and a candidate
  with no checks is refused with "insufficient evidence: no check was run" rather than
  promoted on a vacuous verification. Pins: the mandatory regression test plus three
  field tests in `tests/test_vqc_discovery.py` (13).
- **THE DISCOVERY LOOP NOW RUNS, AND ITS WRONG CANDIDATES ARE MEASURED.**
  `apedp_dyn/vqc/clifford_search.py` proposes local semantics-preserving rewrites,
  certifies every accepted step with the Clifford referee **against the original
  reference** (so the loop cannot drift), ranks by `(operations, depth)` and admits
  through `discovery.py`. `scripts/measure_vqc_clifford_discovery.py` runs it on a
  100-qubit GHZ chain with cancelling work spliced in: **222 → 100 operations** (depth
  108 → 100) in **7.1 s**, 76 steps all `VALID`, 0 rejected, admission
  `VALID`/`MET`/`SUFFICIENT` → **promoted**.
- **THE CHEAPER BUT NON-EQUIVALENT CANDIDATE IS REFUSED, WITH ITS REASON.** The star
  circuit is 100 operations against the reference's 222 and prepares the *same* state,
  which is exactly what a resource-only search would promote: **INVALID** at unitary
  scope (196/200 images differ), **VALID** at state scope, and admission refuses it.
  Six mutated candidates (delete a gate, flip a phase, `cx`↔`cz`) are all refused with
  the differing image named, so the record shows rejection as well as acceptance.
- **THE SIMULATOR AND THE DENSE TIER ARE IN THE LOOP.** 64 stabilizer shots of the
  reference and of the rewritten circuit give an identical support with a maximum
  count difference of 0, and the same workload at 8 qubits agrees with the dense
  referee at distance `1.6e-16`. First run of the record used the search's default
  64-step cap and reported 116 operations; the cap is now 1,000 and a check asserts
  the loop converged below it, because a truncated search is otherwise
  indistinguishable from a local optimum.
  Pins: `tests/test_vqc_clifford_discovery.py` (15), including a property test that
  **every** rewrite proposed for five random Clifford circuits is certified VALID.
- **A REAL BUG IN THE FIRST DRAFT, CAUGHT BY THE REFEREE PATH.** The mutation channel
  rewrote two-qubit gates with one-qubit `s` gates, which fails circuit validation
  before any referee sees it; mutations now preserve arity (`cx`↔`cz`, `s`↔`sdg`,
  Pauli→Pauli). Recorded because the check that caught it is the one the loop is for.

## 2026-10-30 — Controlled-subroutine scope, and scaling measured to 1,000 qubits

Review asked for a controlled-subroutine phase test before any global-phase-insensitive
transformation is applied automatically, and for scaling beyond the 100-qubit GHZ
record. Both are now implemented and measured; nothing from the earlier correction
entry changed.

- **A GLOBAL PHASE IS FREE STANDALONE AND NOT UNDER CONTROL.**
  `adjudicate_equivalence(..., as_subroutine=True)` compares `diag(I, U_c)` against
  `diag(I, U_r)` (new `controlled_lift`), which removes the phase allowance precisely
  because a phase on one control branch is a relative phase. Measured: `RX(2π)` versus
  `I` is **VALID** standalone (distance 0, phase π) and **INVALID** as a subroutine
  (distance `sqrt(2)`); `CX` versus `H CZ H` is VALID in both scopes. Reports carry
  `scope: standalone | controlled`. `adjudicate_candidate(..., as_subroutine=True)`
  routes to the dense tier by construction — the conjugation images cannot see a global
  phase — so a controlled question above ten qubits is `UNRESOLVED`, not a guess.
  Pins: 3 new cases in `tests/test_vqc_equivalence.py` (20) and 1 in
  `tests/test_vqc_discovery.py` (12).
- **SCALING IS MEASURED, IN FOUR FAMILIES, TO 1,000 QUBITS.**
  `scripts/measure_vqc_stabilizer_scaling.py` →
  `benchmarks/quantum_compilation/vqc_stabilizer_scaling.json`. At 1,000 qubits: GHZ
  chain builds in **0.47 s**, validates in 0.43 s, samples in 0.62 s per shot; GHZ star
  the same and lands in the *same* stabilizer group by a different route; the ring graph
  state's 1,000 defining generators `X_q Z_(q-1) Z_(q+1)` all stabilise it; a 7,500-gate
  seeded random Clifford builds in **14.2 s**, validates in 0.65 s, and costs **127 s
  per Z sample** — the honest price of measuring 1,000 qubits, recorded rather than
  hidden. The tableau costs `n²` labels (10⁶ bytes at 1,000 qubits) against `2**1,000`
  amplitudes.
- **EQUIVALENCE AT 1,000 QUBITS, ON CIRCUITS THAT ARE ACTUALLY DIFFERENT.** A
  7,500-gate random Clifford and its rewriting with cancelling `h h` / `cx cx` pairs
  (8,190 gates) are decided **VALID** from 2,000 images in 61 s; the same circuit with
  one `s` replaced by `sdg` is **INVALID** with the offending image named (`Z_0`, a sign
  flip) in 54 s. No matrix at either size. Chain versus star is VALID at state scope and
  INVALID at unitary scope at every size, so the 100-qubit distinction reproduces at
  1,000. Pins: `tests/test_vqc_stabilizer_scaling.py` (7), including hashes and a
  re-derivation of the 100-qubit rows from the script.
- **`validate()` IS 200× CHEAPER, AND THAT IS WHY IT COULD BE MEASURED AT 1,000.** The
  commutation check now runs on packed bit masks
  (`((x1 & z2) ^ (z1 & x2)).bit_count() & 1`) instead of per-qubit label loops: 98.7 s
  became **0.48 s** at 1,000 qubits on the same state. Rank and sign checks are
  unchanged.
- **THE CORRECTNESS AUDIT WAS REGENERATED, AND ITS FROZEN HASH RE-FROZEN
  DELIBERATELY.** `benchmarks/quantum_compilation/correctness_audit.json` was stale: it
  had not been re-run after several VQC records were added, so its scanned set no
  longer matched the records on disk. Regenerated with
  `scripts/audit_record_correctness_checks.py`; pinned totals are unchanged (checked 12,
  partial rows 13, resources-only 5, aggregate-only 51), and no DQI record was touched.
  Because the Phase 0 baseline pins that file's bytes, the single hash was re-frozen
  with an `amended` entry in `vqc_phase0_baseline.json` naming what changed and why —
  not silently.

## 2026-10-30 — Referee correction: the equivalence metric, claim separation, an independent oracle

External review asked for the equivalence report to be checked against its own stated
metric. It did not hold, and the fix plus three related corrections are below.

- **THE EQUIVALENCE DISTANCE WAS NOT THE MINIMUM IT CLAIMED TO BE.** The report was
  documented as `min_phi max_ij |U_c − e^{i·phi} U_r|` and computed a max-entry
  deviation at `phi = arg<U_r, U_c>`, which minimises the **Frobenius** norm, not the
  max-entry norm. `I` versus `Z` therefore reported **2.0** where both objectives give
  **√2 ≈ 1.414214** (`|1 − e^{iφ}|` and `|1 + e^{iφ}|` cross at `φ = π/2`). The
  distance is now `min_phi ||U_c − e^{i·phi} U_r||_F / sqrt(2**n)`, attained in closed
  form at that same phase and evaluated as the residual norm so an exact 1e-16
  agreement does not become 1e-8; its range is `[0, √2]`. The max-entry deviation at
  that phase is reported **separately** as `entry_deviation`, with
  `"entry_deviation_is_minimised": false` in the receipt. Corrected numbers, same
  verdicts: `I` vs `Z` 1.414214 (entry 2.0), `CCX` vs `CCX; Z` 1.414214 (entry 2.0),
  `CCX` vs `CX(0,2); CX(1,2)` 1.224745, `CX` vs `SWAP` 1.224745, `CX` vs `H CZ H` and
  `RX(2π)` vs `I` still 0.0. This supersedes the distance numbers in the 2026-10-29
  entry below, which described the same conflation.
- **SEMANTICS AND ECONOMICS ARE NOW TWO VERDICTS.** A resource claim no longer turns a
  correct circuit INVALID. Both reports carry `claim_verdict: MET | UNMET |
  NOT_ASSERTED` and `promotable`; `CircuitCandidate` → `CandidateVerdict` does the
  same, and `promote()` refuses the equivalent-but-larger candidate with "is VALID but
  not an improvement (UNMET)" instead of "not VALID". Measured: `CX` as `H CZ H`
  claiming `operations: 1` is VALID/UNMET/not promotable, and the 5-qubit Clifford
  claim case reports VALID with UNMET.
- **AN INDEPENDENT ORACLE, BECAUSE BOTH PATHS SHARED GATE APPLICATION.** A bug in one
  gate rule would have been invisible to a comparison of two circuits that both use
  it. `tests/test_vqc_equivalence.py` now builds gate matrices by **bit arithmetic on
  basis indices** — no simulator, no tensor slicing — and requires the execution
  path's unitary to equal the oracle's for random Clifford circuits on 1–4 qubits and
  the stabilizer tier's images to equal `U P U†` from the oracle's `U`. This is an
  independent check of the small-system tier; above ten qubits the tableau path
  remains the only exact one.
- **`StabilizerState.validate()`.** `O(n²)` structural check, refused on anticommuting
  generators (no simultaneous eigenspace) and on rank-deficient sets (group smaller
  than `2**n`, so not pure); a dependency with an odd phase was already refused by
  the elimination. New pins: validation on random 6-qubit Clifford states and on a
  100-qubit GHZ state, both refusal modes, a repeated measurement returning the
  collapsed outcome, and a `Y`-eigenstate (`|+i>`) measuring 50/50 in `Z` with a
  valid, certain branch per trajectory.
- **DOCUMENTED LIMITS.** "A global phase is free" holds for a standalone operation and
  not for a controlled subroutine, where it becomes a relative phase — recorded in
  `docs/VQC_CORE.md` §5 as a limitation of the current claim, not a supported case. The
  100-qubit X-basis row is scoped to what it measured: 128 shots consistent with the
  parity constraint, **0 collisions**, and no claim about enumerating `2**99` strings.

## 2026-10-30 — Equivalence above the dense bound, a matching decoder, and discovery admission

- **A 100-QUBIT CIRCUIT PAIR IS NOW DECIDED, NOT ESTIMATED.**
  `adjudicate_clifford_equivalence` compares a Clifford circuit's **conjugation action**
  on the Pauli generators — `2n` signed Pauli strings, exact integers mod 4 — instead of
  a `2**n x 2**n` matrix. Two Cliffords that agree there differ by a scalar, so this is
  exactly unitary equivalence up to one global phase at any size. Measured at **100
  qubits**: a global phase of −1 is **VALID** (0/200 images differ), a chain-vs-star
  construction that prepares the *same* GHZ state is **INVALID** at unitary scope
  (196/200) and **VALID** at state scope, and replacing `cx(0,1)` with `cz(0,1)` is
  **INVALID** with the offending image named (3/200). The dense tier now returns
  `UNRESOLVED` for a circuit it cannot materialise rather than `INVALID`: being unable
  to compute a matrix is not evidence about the circuit. `S` versus `Z S = S†` is the
  one-qubit version of the same distinction — same prepared state, different unitary.
  Pins: `tests/test_vqc_clifford_equivalence.py` (19).
- **A DECODER INTERFACE, WITH EXACT MINIMUM-WEIGHT MATCHING AS THE SECOND
  IMPLEMENTATION.** `apedp_dyn/vqc/decoders.py` names the interface
  (`decode(syndrome) -> Correction`), keeps the distance-3 lookup table, and adds a
  defect-graph matching decoder solved by exact T-join dynamic programming over subsets
  of defects — boundaries included, exponential in the number of defects, and stated as
  such. `logical.py` gained a distance-parameterised repetition code whose correction
  gates are **built from the decoder**, so the two implementations run the same code on
  the same noise. Measured at `p = 0.1`, 1,200 shots, seed 20261030, against the
  enumerated prediction (all `2**d` patterns, never the simulation): d = 3
  **0.0258 vs 0.028**, d = 5 **0.0050 vs 0.00856**, d = 7 **0.0033 vs 0.002728**; at
  distance 3 the enumeration reproduces `3p² − 2p³` to 1e-15, and matching equals the
  old table on all four syndromes. Distance-7 matching equals a brute-force minimum on
  all 64 syndromes, every error of weight ≤ `(d−1)/2` is decoded exactly, and weight 4
  is miscorrected as a distance-3 code must be. Pins: `tests/test_vqc_decoders.py` (18).
- **DISCOVERED CANDIDATES GO THROUGH THE REFEREES, OR THEY ARE NOT PROMOTED.**
  `apedp_dyn/vqc/discovery.py` takes a `CircuitCandidate` (candidate, reference,
  resource claims, optional declared distribution), routes it to the tier that can
  decide it — stabilizer when both circuits are Clifford, dense otherwise, the
  distribution receipt when one is declared — and seals a `CandidateVerdict`.
  `promote()` refuses everything that is not VALID, **including `UNRESOLVED`**. Measured:
  `CX` as `H CZ H` promoted with its claim met and refused with `operations: 1`; a
  measured Bell candidate decided by its receipt (VALID at 2,000 shots, UNRESOLVED at
  300 and refused); a measured circuit that declares nothing gets no checks at all; a
  100-qubit Clifford candidate is decided; a 20-qubit non-Clifford pair is refused
  rather than passed. Pins: `tests/test_vqc_discovery.py` (11).
- **A RECEIPT BUG FIXED ON THE WAY.** A declared probability of exactly 1.0 divided by a
  zero binomial variance and crashed `receipts._distribution_check`; a certainty is now
  scored as `0` if it happened and `shots` sigma if it did not, so the receipt stays
  valid JSON and the claim can still fail.
- **Documentation.** [`docs/VQC_CORE.md`](docs/VQC_CORE.md) gains §5.1, §7 and §8 with
  every number above, and §10 now names what is genuinely missing (surface code, MPS,
  repeated syndrome measurement, fault tolerance, hardware, non-Clifford equivalence
  above the dense bound, discovery proposals). The Phase 0 inventory's referee,
  logical-qubit, backends and discovery rows say what is now true.

## 2026-10-30 — A second backend at 100 qubits, a phase-sensitive equivalence referee, and two more codes

- **A 100-QUBIT GHZ CHAIN, AND THE CROSS-CHECK THAT MAKES IT COUNT.**
  `apedp_dyn/vqc/stabilizer.py` executes Clifford circuits over `n` commuting signed
  Pauli generators instead of `2**n` amplitudes: phases are integers mod 4, every
  product/conjugation/sign comes from a table built once at import from matrices, a
  measurement either takes a random branch (rewriting the anticommuting generator to
  `±Z_q`) or reads a decided outcome by GF(2) elimination on the `2n`-bit
  characteristic vectors. The dense tier is the referee: the **same** `ghz_circuit`
  builder on both tiers agrees on **8,188** outcome probabilities for chains of 2–12
  qubits (max deviation **1.1e-16**) and on **22** Pauli strings at 6 qubits
  (**2.2e-16**). Then the 100-qubit chain: 100 generators equal to the textbook set,
  only `0…0` and `1…1` at 0.5 each (`2**100` amplitudes is not a thing anyone can
  write down), **512 shots → 248/264** in about six seconds on this host, and the
  X-basis variant gives 2**99
  even-parity strings with **0 collisions in 128 shots**. The test asserts the
  100-qubit circuit is that same ladder gate for gate, which is what lets the
  closed-form prediction confirmed at `n ≤ 12` be used at `n = 100`. Non-Clifford
  gates and every `Noise` operation are **refused by name**, not approximated.
  Record: `benchmarks/quantum_compilation/vqc_stabilizer_ghz.json`
  (`scripts/measure_vqc_stabilizer_ghz.py`). Pins: `tests/test_vqc_stabilizer.py` (32).
- **EQUIVALENCE, UP TO EXACTLY ONE GLOBAL PHASE.** `apedp_dyn/vqc/equivalence.py`
  materialises both unitaries through the same gate application the backends use and
  compares them by `min_phi max_ij |U_c - e^{i phi} U_r|`. `CX` versus `H CZ H` is
  **VALID** at distance 0.0; `RX(2π)` versus `I` is **VALID** with phase π, because a
  global phase is not an error. The case that justifies the module: `CCX` versus
  `CCX; Z` reproduces **every Z-basis probability** — the test asserts that
  elementwise first — and is **INVALID** at distance 2.0, which is the DQI phase-error
  failure mode at circuit level. `CCX` versus two CNOTs is INVALID (1.0), a wrong
  qubit count is INVALID, a measured/feedforward candidate raises
  `NotAUnitaryCircuitError` and is INVALID with that reason, and a `claims` argument
  fails a candidate that is equivalent but *larger* than the improvement it is offered
  under. Bounded by the dense matrix near ten qubits. Pins:
  `tests/test_vqc_equivalence.py` (9).
  *Superseded in part by the correction entry above: the distance is now the
  normalised Frobenius minimum (this entry's 2.0 was a max-entry deviation taken at
  the Frobenius-optimal phase, not a minimum), the metric reads 1.414214 for the same
  pair, and the resource claim is reported as `claim_verdict` instead of failing the
  verdict.*
- **THE PHASE-FLIP CODE AND THE PERFECT FIVE-QUBIT CODE.** `logical.py` gains the
  phase-flip repetition code (the bit-flip code transposed: `X_i X_j` syndromes, `Z`
  corrections, the same `3p² - 2p³`), measured at 0.023 ± 0.0024 against 0.028 at
  `p = 0.1`; `five_qubit.py` is a real distance-3 code — four stabilizers, four
  ancillas, a projection encoder with a conditional Pauli frame — whose prediction is
  **enumerated, not fitted** (all 1,024 Pauli patterns; `0.5432098765` at `p = 1`, a
  rational with denominator `3**5`). All **15** single-qubit Paulis map to distinct
  nonzero syndromes out of 16 (the code is perfect: `16 = 1 + 15`), every single error
  is corrected on both preparations, a weight-2 error can flip the logical value, and
  the measured rates are 0.0130/0.0540 against exact 0.014888/0.053005 at `p =
  0.05/0.1`. Pins: `tests/test_vqc_codes.py` (46).
- **Documentation.** [`docs/VQC_CORE.md`](docs/VQC_CORE.md) gains §4.1, §4.2, §5 and
  §6 with every number above, and §8 now names what is genuinely missing (MPS, surface
  code, MWPM, fault tolerance, hardware, equivalence above the dense bound) instead of
  the backends and codes that exist. The Phase 0 inventory's referee, backends and
  logical-qubit rows are updated to what is now true.

## 2026-10-30 — VQC Phase 0 inventory, and Phase 1 finished: `Reset`, OpenQASM 2, and a frozen DQI baseline

- **PHASE 0 EXISTS AS A CHECKABLE ARTIFACT, NOT A NARRATIVE.**
  [`docs/VQC_PHASE0_CAPABILITY_INVENTORY.md`](docs/VQC_PHASE0_CAPABILITY_INVENTORY.md)
  classifies every DQI component against the seven VQC capabilities with one of five
  decisions — reuse, adapt, build, retain as one backend, not connected — and names
  what is genuinely absent (stabilizer/MPS backends, surface code, MWPM, unitary
  equivalence, discovery into the IR, any threshold claim). It also records where the
  proposed `apedp_vqc/` tree actually lives and why it is `apedp_dyn/vqc/` instead: no
  second top-level package, and flat modules until Phase 6 gives the subpackages
  something to hold.
- **THE "DO NOT MOVE THE DQI BASELINE" RULE IS ARITHMETIC NOW.**
  `benchmarks/quantum_compilation/vqc_phase0_baseline.json` freezes the SHA-256 of the
  thirteen DQI records the VQC work must not disturb, and
  `tests/test_vqc_phase0_inventory.py` recomputes all of them — plus the frozen
  submission record's verdicts and 0.9490542129538326 score ratio, so a byte-exact but
  semantically different rewrite is caught too. It also fails if the inventory names a
  path that does not exist, or if a VQC module is missing from it.
- **PHASE 1 TASK 6: OpenQASM 2, WITH THE GAPS REFUSED.** `apedp_dyn/vqc/qasm.py`
  exports and imports `OPENQASM 2.0` for the gates, `measure`, `reset` and `if(c==k)`,
  carrying metadata and ancillas as `// vqc-...` header comments so a round-trip
  preserves the circuit hash rather than dropping its contract. It refuses noise
  (no QASM 2 construct), conditions on a subset of bits (whole-register equality; a
  per-bit test is QASM 3), user-defined `gate`, `u2`/`u3` and malformed parameters.
  Teleportation is the test case that keeps this honest: its JSON round-trip is exact
  and its QASM export is refused by design, because two gates are conditioned on one
  bit of a two-bit register.
- **`Reset` IS AN OPERATION, NOT A MEASUREMENT.** Added to the IR, serialised,
  counted in resources and executed as measure-and-discard. Measured: `X; Reset;`
  reads `0` on 128/128 shots; `H; Reset;` reads `0` on 128/128; and resetting one half
  of a Bell pair leaves the other maximally mixed (0.503/0.497 at 4,096 shots),
  which is the difference between the reset map and a projection that would leave the
  partner in `|0>`.
- **Pins.** `tests/test_vqc_qasm.py` (15) and the two reset tests in
  `tests/test_vqc_core.py` (now 63); `tests/test_vqc_phase0_inventory.py` (6).

## 2026-10-30 — APEDP-VQC Core: the circuit IR, an exact simulator, checkable noise, a logical qubit and receipts

- **THE PLATFORM'S FIRST MILESTONE, NOT A SECOND OPTIMISER.** `apedp_dyn/vqc/` is the
  reusable core the DQI work was a workload for: `circuit.py` (versioned IR: gates,
  measurements with reset, classical-bit conditions, noise operations, declared
  ancillas, metadata, canonical JSON, SHA-256 identity), `gates.py` (one shared gate
  table), `backends.py` (backend protocol + exact statevector backend, reusing the
  existing `MR` simulator), `noise.py` (exact density-matrix channels), `logical.py`
  (the distance-3 bit-flip code as a circuit) and `receipts.py` (VALID / INVALID /
  UNRESOLVED adjudication). Discussion and every measured number in
  [`docs/VQC_CORE.md`](docs/VQC_CORE.md).
- **THE SIMULATOR IS TESTED AGAINST PHYSICS, NOT AGAINST ITSELF.** Bell and GHZ match
  independently constructed states; teleportation's post-state equals
  `|00> ⊗ RY(π/4)|0>` up to global phase on **every one of 64 shots**, and its X-basis
  `P(0)` is 0.85596 against an analytic 0.85355. With the conditional `Z` removed the
  same readout collapses to 0.500 — feedforward is load-bearing, and the adjudicator
  rejects the candidate at z = 62.
- **THE NOISE TIER IS EXACT AND FALSIFIABLE.** Trace, Hermiticity, positivity and Kraus
  completeness are recomputed over five channels × five probabilities; depolarising
  one half of a Bell pair gives fidelity `1-p` and purity `(1-p)² + p²/3` to 1e-12,
  amplitude damping gives `P(0) = γ`, dephasing gives purity `1 - p/2`. The IR's noise
  operations are the sampled counterpart: a `p = 0.2` bit flip decorrelates a Bell pair
  with probability 0.7979 ± 0.0045 at 8,000 shots and reproduces exactly from its seed.
  (The first version of `apply_kraus` mislabelled the row/column bit indices and
  produced a non-positive matrix; the positivity check is what caught it, which is why
  it is a test rather than a comment.)
- **ONE LOGICAL QUBIT, WITH ITS LIMITS ATTACHED.** Every single bit flip is identified by
  its syndrome and corrected, for both preparations; a pair of flips is miscorrected into
  a logical flip (`syndrome 01`, readout `111`); the measured failure rates are
  0.006625 ± 0.000907 at `p = 0.05` and 0.024875 ± 0.001741 at `p = 0.10` against the
  exact `3p² - 2p³` (0.00725, 0.028). No threshold is claimed, syndrome extraction is
  perfect, the code covers bit flips only, and each record says so in its own fields.
- **AN UNDERPOWERED RUN ABSTAINS.** `adjudicate()` checks the circuit, the ancilla
  cleanup, the outcome distribution against a declared expectation at 4σ, and whether
  the shot count could have detected the declared effect. 8 shots on a correct
  teleportation returns **UNRESOLVED**, not VALID; the same circuit at 4,096 shots is
  VALID; a missing phase correction, a dirty ancilla and an invalid circuit are INVALID.
- **WHAT IS NOT THERE, NAMED SO IT IS NOT ASSUMED.** One backend (statevector only — no
  stabilizer, MPS or tensor-network tier), no surface code or MWPM decoder, no encoded
  gates or repeated syndrome measurement, no fault-tolerance or threshold experiment, no
  hardware adapter, no QASM export, and no discovery integration yet. Those are the next
  milestones, and `docs/VQC_CORE.md` §6 lists them as absent rather than planned.
- **Pins.** `tests/test_vqc_core.py` (61 tests) holds all of it, including twelve
  malformed-program rejections and the five receipt verdicts.

## 2026-10-30 — audit of the round-two review: the three arithmetic claims were already right, the verifier wording was not

- **`32,488`, `8.669587e-4` AND `.382189` ARE NOT IN THIS REPOSITORY.** A second
  external review asks for them to be corrected in "the first report" and "the second
  report". Checked rather than assumed: every record, note and pin says **32,408 calls**
  (`decomposition.multiply_calls`), **8.660587e-4** (`eps_plus` on both verified rows)
  and **5,725,108.382109375** (the reference executed mean), and nothing anywhere says
  otherwise. The arithmetic also forces the call count rather than merely agreeing with
  it: the measured static delta is 291,672 and `291,672 / 9 = 32,408` exactly, while
  `32,488 x 9 = 292,392`. The 42.45% multiply share is `75 x 32,408 / 5,725,461`.
- **SO THE CORRECTIONS BECAME PINS, NOT EDITS.** `tests/test_dqi_submission_score.py`
  now recomputes `eps_plus = 1 - 2^(log2_delta/n)` from the budget the record names and
  matches the stored rows; derives the call count from the static delta (`32,488 x 9`
  fails the identity) and matches the 42.45% share; and recomputes the executed delta
  from the two frozen rows, so a transcribed difference cannot drift.
- **THE VERIFIER WORDING WAS WRONG AND IS FIXED.** §8 said `verify()` ran "unmodified"
  and then disclosed, four paragraphs later, that `reaction_depth` was stubbed. Those
  cannot both be the whole truth about the same invocation. The sentence now says what
  ran — every scoring line unmodified, one **diagnostic-only** stub — and the disclosure
  says plainly that this is the competition verifier with a documented shim, not a
  literal unmodified invocation. Same correction in `docs/DOCUMENTATION.md`.
- **THE SHIM'S ISOLATION IS NOW PINNED BY TEST, NOT BY PROSE.** The record's
  `verifier_patch` (replaced line, replacement, both SHA-256s) is re-derived from the
  shipped verifier and must hash-match; and an AST comparison proves the patched module
  differs from the shipped one in exactly one dict value inside `verify`'s return — every
  other statement, every other function, and the `score = R * Q^2 * T` expression are
  identical, with `circuit.reaction_depth()` occurring exactly once in the module. A
  future edit that stubs a scoring line, or that widens the patch, fails there.
- **WHAT STILL CANNOT BE FIXED BY EDITING.** `scripts/measure_dqi_submission_score.py`'s
  module docstring says the same "unmodified" thing its output disproves; the record pins
  that file's SHA-256, so correcting the sentence means re-running both 102,400-shot
  verifications and re-pinning every displayed figure. Left as is, disclosed here and in
  the record's `scope`, which is the authority on what ran.

## 2026-10-30 — the status decision: pinned line frozen at 0.949054, `MR` the only reopen, no more field-arithmetic work

- **THE THREE ITEMS FROM THE REVIEW, CLOSED AS DECISIONS RATHER THAN LEFT IMPLICIT.**
  [`docs/BENCHMARK2_STATUS_DECISION.md`](BENCHMARK2_STATUS_DECISION.md). Item 1 — put the
  arm's own score on record — was already done and is now named as done (5.324728e12 for
  the six-lever arm, 2.990339e12 for the campaign's stacked arm, both through `verify()`
  at 102,400 shots). Items 2 and 3 were still just prose in a review, so they are now a
  decision record with reopen conditions:
- **D1 — the pinned competition binary is frozen at 0.949054.** The best correct circuit
  it can compile is 2.319438e13, 13.66× the derived target, and the reason no further
  work is planned is arithmetic: the multiply is 42.45% of its static stream, so deleting
  *all* of it would leave 8.29× out, and PyZX 0.10.7 leaves the multiply's T-count at
  378 → 378.
- **D2 — `MR` is the only condition that reopens it.** One opcode, no verifier change,
  specified/patch/local build already in the tree, worth 4.142× on the reference and
  4.5898× on the arm. If no upstream build ships a Z-basis read, the pinned line stays at
  0.949054 and nothing in this repository is "in progress" on it. If one does, the reopen
  is mechanical: rebuild, re-run the two scoring scripts, re-score.
- **D3 — no more field-arithmetic optimisation.** Even with every product and the
  controlled add removed the arm is 1.062× the target, because 472,768.2 Toffolis per shot
  are not field products; after the campaign the remaining 240,785.8 are the swap, the
  shift, `shift_step`, `update_flag` and the rest of the multiply window. The 23-AND
  dimension bound is recorded as the only remaining arithmetic lever, not as a task, and
  the decoder-algorithm work that could reach 445,229 is named as **not started**.
- **What continues is the method and the referee**, both of which are now enforced rather
  than described: [`REPRESENTATION_AWARE_UNCOMPUTATION.md`](REPRESENTATION_AWARE_UNCOMPUTATION.md)
  and `CircuitCandidate.correctness_evidence` with `EvidenceIncompleteError`.
- **Pins.** `tests/test_benchmark2_status_decision.py` (6 tests) holds the decision to the
  records: the frozen score and ratio against `dqi_submission_score.json`, both arm scores
  against their records, the 472,768.2 / 240,785.8 / 1.062 ceilings against the
  reachability note, and the fact that each decision names what would overturn it. The
  reachability note now points at the decision and no longer describes any frontier row as
  unmeasured; `docs/DOCUMENTATION.md` carries the same statement.

## 2026-10-30 — Experiment G, the external-optimizer check, and correctness evidence enforced in code

- **THE FRONTIER IS NOW A VERIFIER NUMBER: 2.990339e12, 1.762× the target.**
  `scripts/measure_dqi_campaign_arm_submission_score.py`, record
  `benchmarks/quantum_compilation/dqi_campaign_arm_submission_score.json`. The campaign's
  stacked arm — six levers plus the five schedule/control gates, `mr` clears, 54-AND
  policy — through the competition's `verify()` at **102,400 shots**: **PASS with 0 value
  and 0 phase errors**, **685,936.2051367187** Toffolis per shot at `Q` = 1,913 (its
  recorded 686,014.8 to 1.1e-4), score **2.990339e12** — **0.12236** of the pinned
  reference, **0.5616** of the six-lever arm. `Q` is what keeps it out: at 1,913 the
  target's 445,229 would have to be 509,088. Build 9.1 s, verify 279.1 s. Not a receipt.
- **THE EXTERNAL COMPARISON, RUN AND SCOPED.**
  `scripts/compare_gf_mul_external_optimizer.py`, record
  `benchmarks/quantum_compilation/gf_mul_external_optimizer.json`. The multiply's forward
  netlist is recorded from the emitter itself (54 ANDs, 1,004 CNOTs, 54 clears at 0 T) and
  given to **PyZX 0.10.7** as the same function in Clifford+T, after the 7-T Toffoli block
  is verified on all eight basis states (max error 3.1e-16). Measured: the Clifford pass
  leaves **T at 378 → 378** and compresses only CNOTs (1,328 → 824); `full_optimize` **does
  not return within 90 s** at 99 qubits and 378 T. A unitary optimizer must also uncompute,
  so its honest figure is **108 ANDs = 756 T** against our 54 ANDs + 12-Toffoli add at 0 for
  the clears. The decoder cannot be offered at all: `HMR` and `INFO_TAG` are in its stream
  and the arm clears with `mr`, which no Clifford+T optimizer models.
- **CORRECTNESS EVIDENCE IS NOW A RUNTIME GATE.** `CircuitCandidate` carries
  `correctness_evidence`; `with_status` raises `EvidenceIncompleteError` for
  `VERIFIED_NONIMPROVING`/`VERIFIED_IMPROVEMENT` unless value, junk and phase are each
  measured-and-passed, covered by another passed check, or `not_applicable` with a reason
  the verifier supports. `ledger.record_verification` derives the evidence from the receipt
  and records `INSUFFICIENT_EVIDENCE` instead of promoting a claim it cannot support: the
  DQI verifier folds the ancilla read into `value_errors` (so junk is `covered_by` value,
  with the verifier's own rule quoted), while an invsqrt/dfthcsos receipt states that it
  prices the circuit rather than simulating it. A failing check under a correctness claim
  raises, and a per-shot verifier cannot waive all three. This is the checklist's gates 3,
  4 and 7 in code.
- **Pins.** `tests/test_dqi_campaign_arm_submission_score.py` (6 tests) pins the `PASS`,
  both zero error counts, the budget, `T` to 0.15% with the exact static count,
  `score = R·Q²·T`, the 0.122357 / 0.561595 / 1.762 ratios and 509,088 needed, the six
  levers and two mixins by name, and the "not a receipt" statement.
  `tests/test_gf_mul_external_optimizer.py` (7 tests) pins the 54/1,004/54 netlist, the
  verified decomposition, `T 378 → 378` with CNOTs 824, the recorded timeout, the 108-AND
  unitary accounting, and the decoder's `HMR`/`INFO_TAG` inventory.
  `tests/test_candidate_correctness_evidence.py` (11 tests) pins the gate, including the
  two ledger outcomes.

## 2026-10-30 — Experiment F: the arm's score measured (5.324728e12, 3.137× the target), the closing statement, and the reusable checklist

- **THE ARM'S SCORE IS NOW A VERIFIER NUMBER, NOT A PRODUCT.**
  `scripts/measure_dqi_arm_submission_score.py`, record
  `benchmarks/quantum_compilation/dqi_arm_submission_score.json`
  (`docs/DQI_OFFICIAL_COMPATIBILITY.md` §9). Every earlier figure for the six-lever arm
  was `R · Q² · T` multiplied out by hand, on a build the competition does not ship.
  Built on the local `MR` rebuild (`scripts/build_kickmix_mr.py` + the opcode patch)
  with the 54-AND policy and run through the competition's own `verify()` at **102,400
  shots**, `log2_delta = -128`: **PASS with 0 value and 0 phase errors**,
  **T = 1,221,408.044756** per shot at `Q` = 1,913 (its recorded 1,221,466.8 to 4.8e-5;
  static 3,132,510, 76,883,226 ops), `R` = 1.191260, score **5.324728e12**. Against the
  pinned reference's 2.443946e13 at the same budget that is **0.2179** — 4.5898×
  cheaper, exactly the `Q²T` ratio. Build 6.7 s, verify 236.8 s.
- **AND IT IS STILL 3.137× THE TARGET, WHICH IS THE POINT.** The derived Benchmark II
  aim is `R · 1,789² · 445,229 = 1.6975e12`; at `Q` = 1,913 the same score needs
  `T` = **509,088**, and the arm executes 1,221,408 (2.743× on `T`, 1.143× on `Q²`,
  3.137× on the score). So `MR` is **necessary and not sufficient**. The record says in
  `status`, `kickmix_role` and `scope` that it is **not a receipt**: only the pinned
  extension may produce one, and it has no `mr`.
- **THE CLOSING STATEMENT, WITH ITS CEILINGS.**
  [`docs/BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) §5 puts
  the frontier in the competition's unit: pinned binary 2.319438e13 (**13.66×**), `MR`
  rebuild 5.324728e12 (**3.137×**), the campaign arm ≈2.99e12 (**≈1.76×**, not yet run
  through `verify()`), target 1×. It states the three independent reasons — no multiply
  work closes the pinned gap (deleting the whole 42.45% share leaves 8.29×), `MR` moves
  the order of magnitude but stops at 3.137×, and the remaining 240,785.8 Toffolis are
  decoder algorithm (swap 64,224, shift 96,584.6, `shift_step` 30,912, `update_flag`
  30,240) — and lists what would overturn it.
- **THE METHOD, AS A NINE-GATE CHECKLIST.**
  [`docs/REPRESENTATION_AWARE_UNCOMPUTATION.md`](REPRESENTATION_AWARE_UNCOMPUTATION.md)
  states the pattern that worked (`del_gf2_mul` clears a product register for 0 Toffolis
  once the register really holds the reduced product) and turns it into gates a
  synthesizer can enforce, each with the failure it prevents: quote the precondition,
  state the object, prove every path, check the phase contract, check the qubit ledger,
  predict the delta and pin the identity, verify value/junk/phase, score with the real
  verifier, and record what was not measured. The three measured cases sit beside it,
  `hmr` among them as the counterexample (24/64 ones on `|1>`, 37/64 phase errors as a
  clear), and the hooks are named (`CircuitCandidate`, `CandidateStatus`,
  `referee.run_dqi_verifier`, `dqi_verification_scope`, `contracts.py`).
- **Pins.** `tests/test_dqi_arm_submission_score.py` (8 tests) pins the arm's `PASS`,
  both zero error counts, the budget, `T` to 0.15% with the exact static count,
  `score = R·Q²·T`, the like-for-like 0.2179 ratio, the 3.137× gap with 509,088 needed,
  and the "not a receipt" statement. `tests/test_representation_aware_uncomputation.py`
  (6 tests) holds the checklist's three cases to the records that measured them and
  asserts the nine gates and the plug-in points are present.

## 2026-10-30 — audit of the quoted figures: the ratio is 0.949054, S3b's saving is 45,396.1, and the probe's ranges are 27-36 / 36-45

- **THE 0.949854 FIGURE DOES NOT EXIST.** An external review reports a
  "0.949854" score ratio and asks for it to be corrected against the record. Checked
  rather than assumed: `git log -S0.949854` over the whole history returns nothing, no
  record or note contains it, and every frozen figure is **0.949054** —
  `dqi_official_candidate.json` stores 0.9490535630117923 for the 64-shot comparison
  (5,433,399.609375 / 5,725,071.609375) and `dqi_submission_score.json` stores
  0.9490542129538326 for the 102,400-shot one (5,433,438.229658 / 5,725,108.382109).
  Both recompute from their own rows, and `score == R(eps_plus) * Q^2 * T` holds to
  1e-12 on both verified rows, with `R` and `Q` ratios of exactly 1.0. There was
  nothing to correct. The scope of that check is the repository: a history search
  can show that no record here ever carried the string, not what an external report
  printed, so the honest reading is that 0.949854 is a misreading of 0.949054 rather
  than a value this repository ever published.
- **S3b IS QUOTED ON ITS OWN BASE AGAIN.** `docs/EEA_SCHEDULE_CAMPAIGN.md` quoted
  **45,377.7** for the campaign's own `drop_shift_c3` arm — that is
  `dqi_movement_schedule.json`'s re-measurement on a base 2,952 Toffolis lower
  (1,218,514.8 → 1,173,137.1). The campaign's record measures **45,396.1**
  (1,221,466.8 → 1,176,070.7). Both figures are real and both stay in the note, but
  the S3b row, the §3.3 paragraph and the summary line now name the record each one
  belongs to. The producer script's two comments were left untouched: the record
  pins its SHA-256, and a comment edit would invalidate that pin without a re-run.
- **THE PROBE'S RANGES WERE NARROWER THAN THE DATA.** The 64-shot probe in
  `dqi_movement_schedule.json` reports 27-**36** exchange firings per shot and 36-**45**
  `c3` rounds with the flag clear (totals 2,052 and 2,556 unchanged); the notes said
  27-33 and 39-45 in three places. Corrected in
  [`docs/DQI_MOVEMENT_SCHEDULE.md`](DQI_MOVEMENT_SCHEDULE.md) §6 and
  [`docs/EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md).
- **Pins.** `tests/test_dqi_movement_schedule.py` now derives both ranges from the
  record and asserts the notes carry them, and `tests/test_dqi_schedule_campaign.py`
  derives S3b's 45,396.1 from the campaign record's own two `T` values and asserts the
  note carries it — so neither figure can drift back without a failing test.

## 2026-10-30 — Experiment E: the competition's own verifier scores the candidate, 0.949054 = a 5.09% improvement

- **THE SUBMISSION QUESTION CLOSED BY RUNNING `verify()` ITSELF.**
  `scripts/measure_dqi_submission_score.py`, record
  `benchmarks/quantum_compilation/dqi_submission_score.json`
  (`docs/DQI_OFFICIAL_COMPATIBILITY.md` §8). Experiments D and D2 stopped at "not a
  submission until it is run through the verifier's shot sample end to end", scoring the
  candidate with the verifier's *rule* by hand over 64 shots. This runs the shipped
  `verify()` — every scoring line unmodified — on the reference and the reduced-window
  54-AND candidate at the competition's own budget, **102,400 shots** (400 batches of 256)
  and `log2_delta = -128`. Both **PASS** with **0/102,400 value and 0/102,400 phase
  errors**; `eps_plus` is 8.660587e-4 for both and `R` 1.191260, so the scores are
  **2.443946e13 → 2.319438e13**, a ratio of **0.949054** (0.94906) — a **5.09%
  improvement**, and exactly the Toffoli ratio, because equal `Q` and equal zero-error
  `eps_plus` leave nothing else in the comparison.
- **THE BUDGET IS LOAD-BEARING, AND THE RECORD SAYS WHY.** `eps_plus` at
  `log2_delta = -128` is `1 - 2^(-128/n)`, so a *zero-error* run is refused below ~1,375
  shots: at 256 the same two circuits get `eps_plus` 0.2929 and `AssertionError … >= 1/16`.
  The smallest batch-aligned budget a zero-error run passes at is **1,536**; the record
  verifies at the full 102,400 (fraction 1.0) and reports the zero-error bound at both.
- **THE 5.09% DECOMPOSED TO THE GATE.** 32,408 emitted multiply calls × 9 Toffolis =
  **291,672** static — exactly the measured static delta (5,725,461 → 5,433,789), and
  291,670.152 on the executed means, with `CCX`/shot equal to the executed delta. The
  multiply is **42.45%** of the reference's static Toffoli stream and the candidate removes
  **5.0943%** of that stream (5.0946% of the executed mean). The `CX`/`HMR`/`CZ` deltas are
  the gate-level re-encoding (104,555,982 operations against 86,051,014), not savings.
- **THE LEVER BOUNDARY MEASURED ON THE SURFACE, NOT ASSERTED.** Of the **29** methods in
  the pinned extension whose signature mentions a classical bit (`km.b`), exactly **one**
  writes a bit from qubit data (`hmr`), and on `|1>` it wrote 1 on **24 of 64** shots and
  flipped the tracked phase on exactly those 24 — a measurement, not a read. Every other
  method only consumes a bit as a condition, so no pinned primitive can express a lever's
  gate. `reaction_depth` (13.9 GB peak, 136 s) is the one diagnostic stubbed, and it is not
  part of the score; every scoring line is the shipped one.
- **WHAT IS STILL OPEN IS NOW ARITHMETIC.** The candidate's 2.319438e13 is **13.66× above**
  the competition's 1.6975e12 target (the reference is 14.40× above it). The remaining
  factor is the schedule, and the instruction it needs is `mr` — unchanged by this record,
  which does not claim a receipt.
- **Pins.** `tests/test_dqi_submission_score.py` (8 tests) pins both `PASS` rows at
  102,400 shots, the shared `eps_plus`/`R`, the 1,536 floor and the 256-shot refusal,
  `score = R · Q² · T` recomputed per row, the 0.94906 ratio equal to the Toffoli ratio,
  the 291,672 decomposition against both streams, the 29-method surface with `hmr` the only
  writer and not a read, and the `reaction_depth` patch. `docs/DQI_OFFICIAL_COMPATIBILITY.md`
  §8 and `docs/DOCUMENTATION.md` state the same figures.

## 2026-10-30 (final) — the first *correct* compilable candidate: 5,433,399.6 at `Q` = 1,893, a 5.09% score improvement

- **A PHASE-SAFE 54-AND MULTIPLY THE PINNED BINARY CAN ACTUALLY EMIT.**
  `scripts/gf54_mul_reduced.py`. Experiment D2 closed the `hmr` route and left the
  question open: a clear needs the value to be *known*, and `del_and` knows only an AND.
  The pinned binary has a **second** fixup primitive — `del_gf2_mul` (`gen_gf_unmul`) —
  whose classically conditioned `CZ` parity is computed from the two factors, so it can
  clear a **product** register for 0 Toffolis. Its precondition is that the register
  really holds `lhs * rhs`, which the old emitter never met: it accumulated an unreduced
  `2n-1`-bit polynomial window and folded afterwards. Here the field product is
  accumulated **already reduced** — each base-case product bit lands at a polynomial
  index that maps to a known set of field bits, and CNOTs are free — so the only window
  is 12 qubits and holds exactly the reduced product. Every AND target is cleared by
  `del_and` on the same footing. Measured at the pinned instance on the pinned
  extension over 64 shots: **`Q` 1,893 (the reference's peak — the window fits under
  it), 5,433,399.609375 Toffolis per shot, 0 value errors, 0 phase errors, junk clean,
  syndromes restored, score ratio 0.949054 = a 5.09% improvement.** That is the first
  correct candidate cheaper than the shipped reference the competition build can
  compile, and it beats Experiment D's withdrawn 0.952 because it does not pay for the
  multiply in qubits. Cost: 66 Toffolis per controlled call against the built-in 75
  (54 ANDs + the 12-Toffoli add); `mr` and `hmr` appear nowhere in the stream, and every
  measured clear is paired with a fixup.
- **THE ARITHMETIC THE REVIEW ASKED ABOUT, STATED.** `T` falls by **5.0946%**
  (`5,433,399.609375 / 5,725,071.609375 = 0.949054`) and the *score* ratio is
  `(T/T₀)·(Q/Q₀)²`: at `Q` 1,896 that was `0.949054 × 1.003172 = 0.952064`, a 4.79%
  improvement, the `Q` term costing 0.317%; at the reduced window's `Q` 1,893 the term
  is exactly 1 and the score ratio equals the `T` ratio. The record now says which
  quantity each figure is.
- **EVERY RECORD AUDITED FOR UNCHECKED CORRECTNESS.**
  `scripts/audit_record_correctness_checks.py` walks all 21 records and classifies every
  candidate row by which of the three checks — value, junk, phase — the record actually
  stores. Result: **13 rows claim correctness on partial evidence** (7 missing junk and
  phase, 6 missing phase), 12 fully checked, 51 bare booleans, 5 resources-only; and the
  historical defect is named mechanically — `dqi_official_compatibility.json`
  `builds.pinned.candidate`, the 5,433,399.6 row that was called correct on its
  syndromes alone. Record: `benchmarks/quantum_compilation/correctness_audit.json`;
  pinned by `tests/test_record_correctness_audit.py`.
- **Pins.** `tests/test_gf54_mul_reduced.py` (4 tests, on the pinned extension) pins the
  emitter's function on both paths, the junk and the tracked phase, 54/66 Toffolis, no
  Z-basis read, and a fixup for every `HMR`. `tests/test_dqi_official_candidate.py` gains
  the 0.949054 row and the claim that it is the only correct arm cheaper than the
  reference. `docs/DQI_OFFICIAL_COMPATIBILITY.md` §3/§5/§7 and `docs/DOCUMENTATION.md`
  carry the revision.

## 2026-10-30 (later) — Experiment D2 and the campaign fix: `hmr` is not a clear, the compile-able candidate is not a submission, and the fused-shift arms finally measure something

- **EXPERIMENT D2 — D's CANDIDATE IS COMPILABLE AND WRONG, AND THE 0.952 IS WITHDRAWN.**
  `scripts/measure_dqi_official_candidate.py`
  (`docs/DQI_OFFICIAL_COMPATIBILITY.md` §3 and §5) checks the one thing Experiment D did not:
  the **phase**. The DQI submission is scored on phase as well as value (the verifier's
  `phase_err = (phase_flipped ^ phases) & ~val_err`), and a clear implemented as `hmr` is not
  phase-safe. Measured one qubit at a time over 200 shots: `hmr` on `|1>` flips the tracked
  phase on **97** of them — the same 97 as its measured ones, because an X-basis measurement of
  `|1>` puts a −1 on the branch that measures 1 — while `del_and`, which is the same
  measurement followed by a classically conditioned `CZ` (the phase fixup), flips **0**, and
  neither flips anything on `|0>`. So a clear whose value is not a *known AND* has no fixup to
  apply, and substituting `hmr` for `mr` there is a randomisation, not a clear. At the pinned
  instance over 64 shots on the pinned extension: reference **1,893 / 5,725,071.609375, 0 value
  errors, 0 phase errors**; Experiment D's 54-AND candidate **1,896 / 5,433,399.609375, 0 value,
  37 phase errors**; and a third arm that removes the reverse EEA outright — the forward EEA on a
  **CNOT copy** of the syndromes, so the input is never written, nothing is restored and no
  qubit's value is read, which is exactly what the `clear_eea` lever needs — at **2,736 /
  3,438,873.609375, 35 phase errors**, twice over: its clears are `hmr` too, and holding the
  whole scratch live costs more `Q` than the pass it deletes. D's candidate builds, executes, and
  is **not** a submission; its **0.952 score ratio is withdrawn** and its Toffoli count stands.
  The measured floor is now the shipped reference: **the cheapest *correct* circuit the pinned
  binary can compile is the one it ships.**
- **EXPERIMENT C, C3 — THE MECHANISM IS MEASURED, AND IT IS THE FLAG.** The probe now reads both
  of the exchange's operands where the exchange reads them — `c3` and the *post-update* `flag`,
  one qubit per round — at the pinned instance over 64 shots and 222 rounds: `c3` fires **4,608**
  times (**72.0 on every shot**, invariant), the exchange fires **2,052** (27–33 per shot), and
  the flag is **clear in 2,556 of the 4,608** `c3` rounds (55.5%). So the flag is not redundant in
  a `c3` round; exchanging on `c3` alone performs ~72 permutations where the decoder asks for
  27–33, which is why C3's 73.0 never appeared and its arm (852,267.4, exchange row 190,888.0)
  is wrong. The same table is why the campaign's S3 arm has nothing to skip: in the 2,052 rounds
  where the flag *is* set, `ctrl`'s one-hotness zeroes the shift's other two terms.
- **THE CAMPAIGN'S DEAD `_shift_a_and_b` IS FIXED AND ITS TWO ARMS RE-MEASURED.** The method is
  now named `_shift_b_array_right`, the one the reference and `MRGatedDecoder` call, so the gates
  inside it are emitted. `dqi_schedule_campaign.json` was re-run with everything else reproducing
  to the Toffoli (control **1,221,466.8**, `everything` **688,966.8**, `stacked` **686,014.8**):
  **S3 `composed` saves exactly 0.0 and is correct** — the campaign's conclusion, now measured
  rather than assumed — and **S3b `drop_shift_c3` saves 45,396.1 and is incorrect**, which
  falsifies S3b's premise (the `¬flag ∧ c3` branch is live). `drop_shift_c3` is recorded as a
  *predictive* arm: its correctness is reported, not required, so a falsification is the result
  rather than an invalid run, and the S3b check now carries the declared 0.0 beside the measured
  45,396.1. The movement script's independent re-measurement against a base carrying the two
  promoted gates (`campaign_none` = 1,218,514.8) agrees: 0.0/correct and 45,377.7/incorrect.
- **Pins.** `tests/test_dqi_official_candidate.py` (8 tests) pins the phase mechanism, the three
  arms' `Q`/`T`/error counts, that no ratio is quoted for a phase-broken candidate, and that the
  cheapest correct circuit is the reference; `tests/test_dqi_movement_schedule.py` and
  `tests/test_dqi_schedule_campaign.py` were updated to pin the rebuilt arms and the exchange
  predicate rather than the vacuous 0.0s. `docs/DOCUMENTATION.md` indexes the new script;
  `docs/EEA_SCHEDULE_CAMPAIGN.md` §3.3 and `docs/DQI_MOVEMENT_SCHEDULE.md` §5–§7 carry the
  corrections.

## 2026-10-30 — Experiments C and D, and the amplitude tier: the exchange stays, one candidate compiles, and the licence is exactly the basis

- **EXPERIMENT C — MEASURED, NEGATIVE, WITH A CORRECTION.**
  `scripts/measure_dqi_movement_schedule.py` (`docs/DQI_MOVEMENT_SCHEDULE.md`) asks the only
  question left open by the schedule campaign: can the decoder's A/B exchange — 64,224
  executed Toffolis per shot, 72 controlled 892-qubit permutations — be resolved at **build
  time** by reassigning which physical register plays the `A` role? That is the only kind of
  saving the pinned competition binary could compile, because it is in the emitted circuit
  rather than in an executed count. The prediction was declared before any arm existed: the
  rebased arm must save exactly the region (64,224.0 executed, **198,024 static**), and is
  correct *if and only if* the exchanging-round set is the same for every shot. At the pinned
  instance over 64 shots: control **686,014.8** at `Q` 1,913 with its 64,224.0 row; a harness
  arm that re-emits the incumbent's exchange from the new script reproduces it to **±0.0**, so
  the comparison is licensed; the rebased arm zeroes the region and drops 198,024 static
  Toffolis and is **incorrect** (syndromes restored, junk and phase not); and the probe reads
  `c3` once per round to find **10 distinct patterns across 64 shots** while every shot
  exchanges exactly **72** times. The schedule is therefore *cardinality-stable and
  order-unstable* — that is why the incumbent's row is exactly `72 × 892` on every shot and
  why no build-time period reproduces it. The arm stays at 686,014.8, 1.541× the target.
- **CORRECTED — two of the campaign's arms measured nothing.** `composed` (S3) and
  `drop_shift_c3` (S3b) both reported 0.0, which the campaign read as the extra condition
  being unsatisfiable. It is not: the code implementing both lives in a method named
  `_shift_a_and_b`, and nothing calls it (the reference and `MRGatedDecoder` call
  `_shift_b_array_right`). Measured at m=15/n=6/GF(2^4): `none`, `composed` and `drop_shift_c3`
  emit **149,337 instructions and 16,610 static Toffolis each** — the two “arms” are the control
  circuit, and their 0.0 is vacuous. The predicate argument (a one-hot `ctrl` and one flag value
  cannot be both) survives; the measurement that was supposed to support it does not, and
  `docs/EEA_SCHEDULE_CAMPAIGN.md` §3 now says so.
- **EXPERIMENT D — THE MISSING `MR` WAS NOT THE ONLY BLOCKER, AND ONE CANDIDATE COMPILES.**
  `scripts/measure_dqi_official_compatibility.py` (`docs/DQI_OFFICIAL_COMPATIBILITY.md`) mines
  the whole surface of both builds. The pinned competition binary has **no `mr`, no `h`, no
  `gf_product`, no per-instruction counts**, but it *has* `hmr`, `bit_store`,
  `push_condition` and `init_and`/`del_and`/`cswap` — and `del_and` already costs **0**
  Toffolis there, because an AND clear can be a measurement where a read cannot. So the arm
  needs **two** surfaces it does not have, not one: the 54-AND product policy is worth
  **523,849.1** per shot on the reference and was never recorded as a blocker. Both builds
  count the same reference identically (1,893 / 5,725,071.609375, per-shot arrays equal), and
  the pinned row reproduces the recorded constant exactly. The positive result: a candidate the
  pinned binary **can** build — the reference algorithm with the 54-AND multiply emitted as
  gates and its workspace cleared with `hmr` instead of `mr` — at `Q` 1,896 and
  **5,433,399.609375** per shot, syndromes restored on all 64 shots, a **0.952** score ratio to
  the shipped reference (**4.79% better**) and still **7.92×** worse than the arm. It is the
  first candidate in this repository the competition build can construct at all, and it is a
  floor rather than a contender: the arm's −43.84% is a read-dependent saving, and a quantum
  control cannot express it (the reference's every gate is quantum-controlled and every one
  executes).
- **THE AMPLITUDE TIER, AND THE LICENCE IT PINS.**
  `apedp_dyn/quantum_compilation/statevector_sim.py` plus
  `scripts/measure_vqc_licence_boundary.py` (`docs/VQC_AMPLITUDE_TIER.md`) compare the
  controlled gate the decoder emits with the read-then-gate map the arms substitute, at
  amplitude level, because every count in this repository is a basis-state count and the
  campaign's licence (P1) is a statement about the *inputs*. Result: on **all eight basis
  inputs** the two maps are the same state up to global phase, the classical bit equals the
  control, and the ancilla returns to `|0>`; on a **superposed control** the gated map branches
  **twice at 0.500** with a fidelity of **0.500** against the coherent reference, and the
  control's off-diagonal goes **0.500 → 0.000** — a complete dephasing, not a perturbation. The
  same tier settles the substitution Experiment D needed: `hmr` alone returns a random outcome
  on `|0>` (it measures in the X basis), while `h; hmr` is `mr` branch for branch on basis *and*
  superposed inputs — so the blocker is the missing Hadamard, not the measurement instruction.
  The tier is cross-checked against `kickmix` on the gadget's eight basis inputs (identical),
  and the arm's 1,913 qubits are reported as 10^575.87 amplitudes rather than pretended.

## 2026-10-30 — The schedule campaign: read the predicate, then move; 1,221,466.8 → 686,014.8

- **EXPERIMENT B — MEASURED AND CLOSED, NEGATIVE.** The product stream's own operand
  predicate: every surviving product is `ixor_gf2_mul(q_i, a_j)`, so a zero multiplier makes
  it the identity, and the argument that licenses S1–S4 licenses gating on it (`S6_zero_q`,
  declared **0.0 saving** before the arm was built). The arm is **correct** — the predicate is
  admissible — and **loses**: 997,795.6 Toffolis per shot, **+311,780.7 over the five-gate
  incumbent**, adding 327,094 mid-circuit reads (357,347 in total against 30,253) and 956,116
  instructions. Two structural reasons, both measured: (a) the sweep's selection chain
  `cswap`s `q_i` into the a-lanes, so the multiplier is **not** a round-level quantity — the
  first version of this arm read it once per round and was *incorrect*, and only a
  per-position read is sound; (b) a per-position read measures at 0.95 Toffolis against a
  66-Toffoli product that is skipped only inside the `mul` window, so the gate needs the
  multiplier to be zero in ~3% of live positions, and the loss is almost exactly the read
  cost — the multiplier is essentially never zero where the product executes. Its falsifier
  (the inverted condition) is wrong and informative (316,701 product Toffolis skipped). The
  331,023 surviving products are work the data path asks for; removing them needs a cheaper
  multiply, not another classical read.
- **MEASURED — the shift/sweep/swap priority is discharged, and it was worth 43.6% of the
  arm.** `scripts/measure_dqi_schedule_campaign.py` (`docs/EEA_SCHEDULE_CAMPAIGN.md`) applies
  the read-then-gate gadget at per-position grain: every predicate register is a basis state,
  so `cx` + `MR` through the shared ancilla yields a deterministic classical bit at zero
  Toffoli and no phase, and a gate conditioned on a zero bit costs nothing on that shot. At
  the pinned instance, 64 shots, `Q` 1,913 and 222 rounds in every arm: the sweep's one-hot
  quotient selection **−103,968.0** (chain 105,696.0 → 1,728.0), the shift's `do_shift` mask
  **−65,895.0** (chain 131,220.0 → 65,325.0), the product's `mul` control **−362,637.0**
  (product stream 693,660.0 → 331,023.0); all three in one arm **1,221,466.8 → 688,966.8**
  (−532,500.0, exactly additive), **1.547×** the 445,229 target instead of 2.744×, `Q²T`
  4.4700e12 → 2.5213e12. Record: `benchmarks/quantum_compilation/dqi_schedule_campaign.json`.
- **STACKED WITH THE PRIOR COMBINED ARM.** The same record now re-measures the
  flag/metadata gate the previous session promoted (2,952.0 → 1,218,514.8, zero reads) and
  puts all five gates in one circuit: **1,221,466.8 → 686,014.8 per shot (−535,452.0,
  −43.84%)**,  exactly `532,500.0 + 2,952.0`, so the two sessions' gates do not interact,
  **1.541×** the target, `Q²T` 4.4700e12 → 2.5105e12. The cross-report inference "the two
  gates compose" is now a measurement in one record — and a declared one: `prediction.S5_stack`
  records the arithmetic (`532,500 + 2,952 → T = 686,014.8`) written after the schedule arms
  and **before** the five-gate arm was built, and the measurement matched it to **0.0**
  Toffoli (`observation.stacked_error_per_shot`).
- **NOT RELOCATED — REMOVED.** Counting the whole stream on the same runs: executed
  instructions fall 37,808,019.7 → 32,451,795.1 per shot (−5,356,224.6, −14.17%) while the
  Toffolis fall 535,452, i.e. 10.12 instructions per removed Toffoli — the expansion the
  removed `CCX`es had. The added cost is 30,253 mid-circuit reads per shot at 2 instructions
  each (60,506, 1.1% of what was removed), reported for every arm. The honest claim is: the
  nonlinear work falls 43.84%, the stream falls with it, and the price is measurement depth.
- **DECLARED BEFORE MEASURED.** The predictions are computed from the control arm's own chain
  costs and printed before any variant exists (`prediction.written_before_variants`): S1 ≥
  100,368 from a burn of `12 × 2 × 222`, S2 **no saving** (`do_shift` is the XOR of three
  threshold intervals, not a one-hot), S3 and S3b exactly 0.0, S4 no saving. S1 came out at
  103,968.0 (the bound assumed two selections in all 222 rounds; 144 occur); S2 at 65,895.0,
  which **closes** the shift half rather than promoting it — half the chain is movement the
  data path needs; S3/S3b at 0.0 with the arm still correct, which is the measured form of
  "the shift and the swap cannot be fused for gain".
- **FALSIFIED THREE WAYS.** Every variant was rebuilt with its condition made wrong and all
  three are incorrect: the stale selection (skips 104,768.1 of its chain and *costs* 217,720.1
  more than the control, because a wrong state changes the control flow), the inverted mask
  (skips 69,311.6 and saves 79,854.5 — more than the honest gate's 65,895.0, which is what
  makes it wrong), and the previous slot's mask (70,351.1 skipped, 80,612.0 saved, wrong).
  The record measures skipped work on the chain the condition gates, never on the arm total,
  and a falsifier that skips nothing is recorded as vacuous instead of as a pass.
- **CORRECTED — what the arithmetic of the reachability note covers.** §3 of
  `docs/EEA_AND_PREDICATE_PLAN.md` counted the *static* residual, so its "`C` alone is 1.062×
  the target" was read as "control-logic work cannot help". The executed residual is not
  `C`: 532,500 of it was quantum-controlled work whose control was zero. The target is still
  out of reach — 686,014.8 is 240,785.8 above it, and the largest remaining items are the
  swap's 64,224, the shift's 96,584.6, `_update_flag`'s 30,240 and `shift_step`'s 30,912 — but
  it is now 1.541× away, not 2.737×.
- **STATED COST MODEL.** An executed count that depends on classical conditions is a count
  under adaptive execution. The benchmark permits it (the reference's own `error_mbuc` is
  classical feed-forward, and the six levers were promoted on the same mechanism), but this
  campaign applies it at per-position grain — thousands of mid-circuit measurements and
  classical decisions per shot. A submission that forbids mid-circuit classical control could
  not claim these counts, and none is claimed. No receipt exists either: the arm still needs
  `MR`, which the pinned competition binary does not have.

## 2026-10-29 — The two gates stack to 2,952; the GA leak and the search ranker are fixed

- **MEASURED — the two promoted gates compose, and their savings do not add.**
  `scripts/measure_dqi_combined_arm.py` runs the predicate-reuse arm and the halting-round
  gate in one circuit. The prediction is written before the combined arm is built, from that
  run's own control and halt-gated arms: the reuse saves 8 Toffolis only in rounds where
  `_update_flag` executes, and the halt gate removes it from six dead rounds, so
  `8 × 216 + 1,224 = 2,952`, not the naive `1,776 + 1,224 = 3,000`. Observed: **2,952**
  (1,221,466.8 → **1,218,514.8** per shot, `Q` = 1,913, correct, 222 iterations in all three
  arms, region rows partition each total). The 48-Toffoli difference is the interaction, and
  the record names which model matched. Record:
  `benchmarks/quantum_compilation/dqi_combined_arm.json`.
- **FIXED — the evolutionary search no longer selects on the confirmation rows.**
  `evolve()` picked `grouped_pick` by minimising error on the supplied `x_test`/`y_test`,
  which turned the confirmation split into a second training set. Selection is now a
  deterministic 5-fold inner cross-validation over the training rows
  (`SearchConfig.selection_folds`), with coefficients refitted inside each fold; the test
  rows are read only after the pick is fixed, to report the new
  `SearchResult.confirmation_mse`. `selection_method`/`selection_folds` record which path
  was taken, and the fallback when no candidate survives the folds is the lowest training
  error — never the test rows. Both the native and Python paths were changed. Pinned by
  `tests/test_evolutionary_search.py::test_grouped_pick_is_selected_without_reading_the_confirmation_rows`
  (inverting the confirmation targets must leave the pick identical and only worsen the
  confirmed error).
- **FIXED — the search ranker preferred the worse candidate.** `ValidationPipeline`'s
  `overall` was `0.4 * max(0, cv_mean - cv_std) + ...` with `cv_mean` an *error*: a perfect
  fit scored 0.56 and a poor one 0.83, so `HierarchicalSearch.search` sorted its worst
  candidate first. `overall` is now `0.4 * max(0, 1 - (cv_mean + cv_std)) + ...`, recomputed
  in Python from the returned components on the Rust path too (the kernel carried the same
  sign and is corrected in `validation.rs`). Measured effect in
  `benchmarks/search_engine_gaps.json`: classic templates recovered 0/12 planted laws with
  the inverted score and 2/12 with the fix.
- **MEASURED — what the search templates cannot reach, and closed most of it.**
  `scripts/measure_search_engine_gaps.py` plants 12 law families, searches on 200 rows and
  scores the returned candidate on 100 untouched rows (coefficients refitted on train).
  Classic `HierarchicalSearch` recovered **2/12**, `beam_search_expressions` **1/12**.
  The extended template set (reciprocal, square root, cubic, `exp(±x)`, log, cosine,
  rational saturation, ratios, a variable in an exponent, plus an exact least-squares scale
  and offset fitted to every candidate) recovers **11/12**, and `beam_search_expressions(
  include_functions=True)` **9/12**. Genuinely still unreachable, and reported as such: a
  sinusoid of non-unit frequency (the engine fits affine coefficients, not a frequency).
  Records: `benchmarks/search_engine_gaps.json`.
- **ADDED — patent prior-art search with a provenance gate.** `apedp/patent_prior_art.py`
  indexes patent records whose claims carry a normalised formula and audits a candidate
  against them: rename-insensitive algebraic equivalence first, structural signature second,
  otherwise `NOT_FOUND_IN_PATENT_CORPUS`. A record that claims a real patent without an
  http(s) source and a retrieval date is refused at load (`PatentProvenanceError`), and
  claims with no formula are counted and reported (`NO_SEARCHABLE_CLAIMS`) rather than
  silently skipped. No patent corpus is bundled; the loader and the documented source
  options are. Pinned by `tests/test_dyn_patent_prior_art.py` (9).

## 2026-10-29 — The 80% is split: per-instruction counts, and two gates that pass

- **INSTRUMENTED:** the local `MR` build now counts executed shots **per instruction**
  (`sim.counting_op_executions` / `sim.operation_execution_counts()`), and the circuit
  exposes its `builder.mark(...)` boundaries and Toffoli-like mask (`circuit.info_tags()`,
  `circuit.op_toffoli_like`). The simulator applies the same per-shot condition mask as
  `operation_statistics()` and masks off the SIMD lanes beyond `batch_size`, so a region's
  sum reconciles with `num_toffoli_like` exactly rather than by construction.
- **MEASURED — the named 80% bucket is gone.** `scripts/measure_dqi_region_ledger.py`
  partitions the six-lever arm's 1,221,466.8 Toffolis/shot into regions that sum to the
  472,768.2 non-product residual: `shift_b_array_right` 162,479.6, `sweep_positions`
  143,038.0, `swap_a_and_b` 64,224.0, `_update_flag` 32,856, `_initial_shift_loop` 30,912,
  `_update_metadata` 12,432, `_update_halting_counter` 10,656, counter decode 4,662, the
  alignment pair 7,392, multiply-and-add control 2,016, quotient control 1,728, Forney
  control 372.6. It also records what each lever removed per region, and checks each
  lever's arm-level delta against the sum of its region deltas. Record:
  `benchmarks/quantum_compilation/dqi_region_ledger.json`.
- **PROMOTED — predicate reuse in `_update_flag`.** `scripts/measure_flag_predicate_reuse.py`
  shares the `(n_A == n_B)` comparison between the `c0` and `c1` sub-steps and clears the
  scratch with `MR` instead of uncomputing it: 8 Toffolis per iteration × 222 = **1,776
  per shot**, exactly the prediction computed from unit costs before the arms were built
  (the first prediction used the 12-bit field degree instead of the 8-bit register width
  and over-predicted by 4 per iteration; the mistake and the corrected run are in the
  record). `_update_flag` 32,856 → 31,080, `Q` unchanged at 1,913, `Q²T` −0.145%, all eight
  promotion gates pass, 222 iterations in both arms. Record:
  `benchmarks/quantum_compilation/dqi_flag_predicate_reuse.json`.
- **PROMOTED — the halting-round gate.** `scripts/measure_eea_oracle_ledger.py` gates
  `_update_flag` and `_update_metadata` on a classical read of `is_hc_zero`: both regions
  are controlled on the one-hot `c0..c3`, so in a round with a non-zero halting counter
  their net effect is the identity and skipping them is exact. It removes **1,224 per shot**
  (888 + 336), correct, `Q` unchanged. The predicate is already computed by the arm, so this
  is a real gate, not an oracle upper bound. Record:
  `benchmarks/quantum_compilation/dqi_eea_oracle_ledger.json`.
- **NARROWED:** together the two gates are 3,000 Toffolis/shot, 0.25% of the arm and 0.6% of
  the non-product residual. Most of the residual is not predicates — it is the sweep, the
  shift and the swap, which the six levers already gate to 64.8%, 68.4% and 32.4% of their
  static counts. No predicate edit closes the 2.744×; §3 of `EEA_AND_PREDICATE_PLAN.md`
  still stands.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` pins both records: the region rows
  sum to 472,768.2 with products separated, each lever's delta equals its region deltas,
  the flag-reuse observation matches its prediction, and all eight gates pass.

## 2026-10-29 — Ratio baselines labelled: 4.687× and 4.258× are different questions

- **CORRECTED:** the update quoted "4.687× against 4.320×" without saying which baseline
  each ratio used, which reads as a regression once a controlled comparison is made. The
  measure now records five comparisons, each with its baseline
  (`gf54_native_opcode.json:comparisons`): the multiply saves **9.15%** on the reference and
  **7.83%** on the gated arm; the arm is **4.687×** faster than the *unchanged* reference
  (the scored comparator) and **4.258×** faster than a reference given the same opcode; its
  `Q²T` falls **5.97%** from the built-in arm to the final one.
- **EXPLAINED:** 4.258× is below the built-in arm's 4.320× because the policy helps the
  reference more than the arm — the arm's Forney body is gated, so fewer of its inversion
  products execute. Both ratios are correct; only a reader told which baseline each uses
  can tell that they are not the same claim.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` recomputes all five from the
  recorded means and requires the same-policy ratio to sit below the built-in one, so a
  future edit cannot quietly swap the baselines.
- **NOTATION:** every figure in this work is GF(2^12) with 12-bit field elements
  (`DEFAULT_DEGREE = 12`); the two new notes spell it in ASCII so the superscript cannot be
  misread as 2^25.

## 2026-10-29 — Reconciled: the official 5,725,108.38 and our 5,725,071.6 are one metric

- **RECONCILED:** `scripts/reconcile_dqi_resource_baseline.py` (record:
  `benchmarks/quantum_compilation/dqi_resource_reconciliation.json`,
  note: [`docs/DQI_RESOURCE_RECONCILIATION.md`](docs/DQI_RESOURCE_RECONCILIATION.md)). The
  locked official reference row is 1,893 qubits / 5,725,108.382109375 Toffolis, and its
  fractional part is a multiple of 1/102,400 but **not** of 1/64 — so it is the verifier's
  default-shot mean, while this project's 5,725,071.609375 is a 64-shot mean. Same decoder,
  same metric (`num_toffoli_like`, which the verifier itself averages), same instance.
- **MEASURED:** the reference decoder was built and simulated on the **pinned competition
  binary** and on the local `MR` rebuild over the same 64 sampled syndromes. The per-shot
  Toffoli arrays are identical element by element and both means are 5,725,071.609375 at
  1,893 qubits, so the `MR` patch is count-neutral for a circuit that emits no `MR`. The
  36.77 gap to the official row is 0.78 standard errors of a 64-shot mean: a sample, not a
  discrepancy.
- **ESTABLISHED — no receipt for the arm.** Building the six-lever arm on the pinned binary
  fails with `AttributeError: 'Builder' object has no attribute 'mr'`: its clears need an
  opcode the competition build does not have, and `apedp_dyn/quantum_compilation/contracts.py`
  pins receipts to that build's hash. The arm's 1,221,466.8 at `Q` 1,913 is a local
  measurement, not an official result; `verify_submission_dqi.py` has no Toffoli threshold
  either — it returns `PASS` on correctness and reports `R(eps_plus) · Q² · T`.
- **DOCUMENTED:** what 445,229 is — a *derived target* (`Q = 1,789`, `T = 445,229`, chosen so
  the product reaches a 14.397× portfolio gain), not a measurement, a competitor's figure or
  an acceptance threshold. `docs/BENCHMARK2_TARGET_REACHABILITY.md` now says so before it
  compares anything to it.

## 2026-10-29 — Product window removed: 19 workspace qubits, `Q` 1,935 → 1,912, same T

- **MEASURED:** `gen_gf54_mul` no longer materialises the 2m-1-bit product window. Reduction
  is GF(2)-linear, so each Karatsuba sub-product is added to the target through a
  `x^d mod p` table as soon as it is computed; the peak scratch is one sub-product window at a
  time — **19 qubits at GF(2^12) instead of 42**, 12 + 19 when controlled instead of 54. The
  AND count is untouched, so `T_mean` is too: the two records differ only in `Q`.
  Correctness is unchanged and re-checked: multiply, inversion and division all 0 bad on 32
  random basis states, operands restored, workspace `|0>`, phase untouched.
- **MEASURED:** with the opcode emit at the pinned instance over 64 shots the reference stays
  **5,725,071.6 → 5,201,222.5 per shot (9.15%)** but `Q` goes **1,893 → 1,912** (was 1,935),
  and the six-lever arm **1,913** (was 1,936) at 1,221,466.8 per shot. The cost is 22 more
  CNOTs per controlled multiply (600 against 578): the reduction is applied per sub-product
  rather than once per window bit. Since the score charges `Q` squared, 23 fewer qubits at
  equal `T` is a 2.4% improvement in `Q^2 T` over the 42-qubit emit.
- **NEW:** [`docs/BENCHMARK2_TARGET_REACHABILITY.md`](docs/BENCHMARK2_TARGET_REACHABILITY.md)
  lists every opcode class against the 445,229 target with the measured arithmetic. (The
  two-term form first recorded here — `527,806.8 + 10,510 × (ands + 12)`, giving 2.012× at
  the 23-AND bound and 1.186× free — was superseded the same day by the three-term
  decomposition below: same totals, but it folded the 1,019.23 inversion/division products
  into its constant at 54 ANDs each, so "free multiply" meant "free `ixor` only". The
  correct figures are 1.941× and 1.062×.) It also records why no Fast-Fourier/cyclotomic
  selector was added: the Chien search's 69 constant multiplies per position are CNOT-only
  (0 Toffoli in `dqi_primitive_costs.json`), so a transform can only trade CNOTs for CNOTs
  and would pay qubits, which the score squares.

## 2026-10-29 — The reconciled non-multiply ledger, and the first experiment's gates

- **MEASURED — ledger.** `scripts/attribute_dqi_nonmultiply_tcount.py` (record:
  `dqi_nonmultiply_ledger.json`, note: [`docs/NONMULTIPLY_LEDGER.md`](docs/NONMULTIPLY_LEDGER.md))
  assigns every executed Toffoli of the six-lever arm to exactly one category and checks
  that they sum to the simulator's total: **693,660.0** (10,510 controlled multiplies),
  **8,382.7** (155.23 inversion products), **46,656.0** (864 division products) and
  **472,768.2** (non-product residual, 38.7%) = **1,221,466.8**. The static ledger reconciles
  too: 3,486,597 = 94,470 + 3,392,127.
- **BOUNDED, NOT GUESSED.** The residual's six unconditional regions are exactly 94,470
  (20.0%). The other **378,298.2 (80.0%) is one named unattributed bucket**: the simulator
  reports per-shot counts by operation type, not per operation, so the conditional EEA body
  cannot be split by region; the ledger records the gap, how to close it (per-op execution
  counts, or per-region oracle gating that measures the removable part) and refuses to
  estimate it into the table.
- **CORRECTED INVARIANT.** The review's accounting invariant of 527,806.8 is superseded by
  472,768.2 for the reason in the previous entry: 55,038.4 of the old constant was the
  1,019.23 inversion/division products, which are field arithmetic, not decoder logic.
- **NEW — first experiment protocol.** `_update_flag` plus the counter/halting predicates
  (43,734 Toffolis, 9.3% of the residual) against the unchanged 222-iteration arm, with a
  prediction written before the edit and eight promotion gates (exact output equivalence,
  measured phase, clean workspace, fixed iteration behaviour, both T-counts, `Q`, `Q²T`
  from the same run, frozen source and record). Pinned in
  `tests/test_dyn_mr_gated_arms_kickmix.py`.

## 2026-10-29 — Three-term decomposition, and the measured EEA/predicate plan

- **CORRECTED — same totals, honest labels.** The arm's Toffolis are
  `T = 472,768.2 + 10,510 × (ands + 12) + 1,019.23 × ands`, not
  `527,806.8 + 10,510 × (ands + 12)`. The 1,019.23 are the products inside the arm's
  inversion (155.23) and division (864), each 54 ANDs with **no** controlled add, so the
  old constant silently contained 55,038.4 of field arithmetic. Consequences: freeing
  every field product gives **1.345×** the target (not 1.469×), freeing the adds too gives
  **1.062×** (not 1.186×), and the 23-AND bound lands at **1.941×** (not 2.012×).
- **MEASURED:** the decomposition, by scoping `gf_product` to one entry point at a time on
  both the reference and the arm — the reference runs **888** divisions per shot (47,952
  Toffolis saved) and the arm **144** (7,776), and the arm's gated Forney block runs
  **31.05** inversions (1,397.11). `gf54_native_opcode.json:isolated_blocks` records all of
  it, and its `check_terms_sum_to_arm` is exact.
- **NEW:** [`docs/EEA_AND_PREDICATE_PLAN.md`](docs/EEA_AND_PREDICATE_PLAN.md) is the measured
  plan the next layer was promised: the residual `472,768.2` (38.7% of the arm, 1.062× the
  target on its own), the exact 94,470-Toffoli ledger over the six unconditional regions
  (20% of it), the EEA slopes (17,208/iteration, 351/position), the five levers ranked with
  ceilings, the correctness obligations (systolic schedule gate-for-gate; early halting and
  asynchronous pointers closed until proved; no deletion-differencing), and the arithmetic
  showing no control-logic edit reaches 445,229: `C ≤ 445,229 − P` needs either zero-cost
  arithmetic (impossible at the 23-AND bound) or a 5.8% cut in `C` *with* the arithmetic
  free.
- **PINNED:** three new kicks in `tests/test_dyn_mr_gated_arms_kickmix.py` — the three-term
  sum, the 472,768.2 residual with `C > target`, and the 94,470 region ledger at 20.0% of it.

## 2026-10-29 — The 54-AND multiply as an opcode: 4.687×, and 35% of it is Forney

- **NEW:** `gen_gf54_mul` in `scripts/kickmix_mr_opcode.patch` emits the proved 54-AND
  GF(2¹²) product natively, and a `GFProduct` policy (`builder.gf_product = 'and54'`) routes
  `gen_gf_mul` through it. The earlier `Gf54MulMixin` could only replace `ixor_gf2_mul`; the
  five Itoh–Tsujii products inside Chien search and the quotient sweep's are emitted by
  native generators, so the mixin left them on the built-in 63-AND recursion.
- **MEASURED:** `scripts/measure_gf54_native_opcode.py` (record:
  `benchmarks/quantum_compilation/gf54_native_opcode.json`). Unit costs: controlled multiply
  **75 → 66**, plain multiply **63 → 54**, inversion **315 → 270**, division **378 → 324**
  Toffolis, at **42** workspace qubits against 12, all four correct on random basis states
  (values, restored operands, clean workspace, untouched phase). At the pinned instance over
  64 shots the **reference** decoder goes **5,725,071.6 → 5,201,222.5 per shot (9.15%)**, `Q`
  1,893 → 1,935, and the six-lever arm **1,325,230.0 → 1,221,466.8 (4.687× against 4.320×)**
  — better than the mixin's 4.652×, because the opcode reaches the inversion too.
- **MEASURED — the Forney inversion's share, isolated.** The reference runs an inversion of
  five products at every one of its 4,095 positions: 20,475 product sites, 184,275 static
  Toffolis. Pricing it alone (the same reference with only
  `init_gf2_inverse_with_scaffold`/`gf2_phase_by_product` switched) executes **20,469.5
  products per shot across 4,093.9 positions, saving 184,225.1 Toffolis — 3.22% of the
  reference and 35.2% of its 523,849.1 saved per shot**. The remaining 339,624.0 (64.8%) is
  the change-of-basis division and the quotient sweep. Static and measured differ only by the
  1.1 positions per shot whose error bit is clear.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains the builder-policy surface
  (66 CCX, 42 extra workspace bits, invalid names rejected), an end-to-end scaled run, and a
  pin on the recorded split so the Forney share cannot drift back to an assertion.

## 2026-10-29 — The 54-AND multiply, executed: 4.652×, and the share it corrects

- **NEW:** `scripts/gf54_mul.py` turns the proved 54-AND GF(2¹²) algorithm into gates: the
  pinned `mu(2)=3`/`mu(3)=6` base cases, their Karatsuba composition, a classical evaluator,
  and `Gf54MulMixin`, which reimplements `ixor_gf2_mul` as **54 ANDs plus the 12-Toffoli
  controlled add — 66 Toffolis per controlled call against the built-in 75** — clearing its
  product window and scratch with `mr`. 42 workspace qubits against the built-in 12, and a
  longer CNOT ladder (566 vs 481).
- **MEASURED:** `scripts/measure_gf54_mul_kickmix.py` (record:
  `benchmarks/quantum_compilation/gf54_mul_kickmix.json`). The emitted multiply is correct on
  random basis states with random controls (values equal `field.mul`, operands restored,
  workspace `|0>`, phase untouched), and swapped into the six-lever arm at the official
  instance over 64 shots it takes **1,325,230.0 → 1,230,640.0 Toffoli per shot: 4.652×
  against 4.320×**, a **7.14%** saving, `Q` 1,894 → 1,897, with the arm still passing the
  six-lever verification shot for shot against the reference.
- **CORRECTED:** the projected 4.928× does not reproduce, and the reason is the model, not
  the circuit: a controlled multiply pays its ANDs **plus 12**, and the 12 do not shrink, so
  the call scales by 66/75. The same measurement corrects a larger error in the record — the
  multiply is **59.48%** of the arm (788,250.0 per shot over 10,510 executed calls), not the
  **86.33%** an earlier ablation reported. That ablation deleted `ixor_gf2_mul` and
  differenced executed counts, so the registers held garbage, the gating predicates read it
  and far less other work ran. The new measurement differences two arms that differ only in
  the multiply, and it reconstructs the measured arm exactly
  (536,980.0 + 10,510 × 66 = 1,230,640.0).
- **REVISED:** the multiply-cost ladder is now `t(ands) = 536,980 + 10,510 × (ands + 12)`: 54
  ANDs 4.652× (measured), 45 ANDs 5.039×, the 23-AND `2n-1` bound 6.327× (2.03× the target),
  and **0 ANDs 8.634× — still 1.49× above 445,229**. The conclusion survives with a narrower
  margin: no multiply, however cheap, delivers Benchmark II on this decoder, so the target
  needs a change to the 536,980 that is not the multiply. `docs/CIRCUIT_OPTIMISATION_RULES.md`
  §5–§6 and §8 carry it, and `benchmarks/quantum_compilation/{designer_directions,
  gf_mul_and_static_headroom}.json` were regenerated rather than left asserting the old share.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains pins for the base algorithms,
  the emitted circuit (values, restored operands, clean workspace, phase), the 66/75 unit
  costs, both arms' correctness on a scaled run, and the recorded decomposition.

## 2026-10-29 — Three designer directions priced; the multiply gains 14.3%

- **MEASURED, then designed:** `scripts/study_circuit_designer_directions.py` (record:
  `benchmarks/quantum_compilation/designer_directions.json`) prices the three ways left to
  improve the decoder. **A** (predicate discovery over the regions no lever touches) is
  bounded at **2.71%** of the emitted stream at the pinned instance — 94,470 of 3,486,597
  Toffolis — so it does not pay here. **B** (an exact bilinear search) proves `mu(2)=3` and
  `mu(3)=6` by exhausting every choice of five rank-1 matrices, then composes by Karatsuba
  to **54 ANDs at GF(2¹²)** against the built-in 63 (14.3% fewer), verified against
  `GF(2¹²)` multiplication on 200 sampled inputs with 0 mismatches and priced at
  **4.928×**. **C** (a different decoder) is refuted as a multiply-only fix: even the
  23-AND `2n-1` dimension bound leaves 598,835.7 per shot, still 1.35× above the 445,229
  target, and only the inadmissible measured-operand route clears it.
- **DOCUMENTED:** `docs/CIRCUIT_OPTIMISATION_RULES.md` §6 carries the comparison and the
  ladder, and §8 records the 54-AND multiply as *pending*: a verified algorithm, not yet a
  measured circuit, because `ixor_gf2_mul` is the 63-AND Karatsuba and emitting the new one
  needs a second `kickmix` op.

## 2026-10-29 — Rules for a better circuit, and the measured ceiling

- **NEW:** `docs/CIRCUIT_OPTIMISATION_RULES.md` fixes what a "better circuit" claim has to
  carry: the metric by name (**executed** 1,325,230.0 = 4.320× versus **static** 3,486,597 =
  1.642×, which rank the levers differently), unitary preservation on every input (dead-work
  gating and measured uncomputation are admissible; measuring an operand that is not
  determined by the state is not), a falsifier per measurement, and the instance discipline.
- **MEASURED:** `scripts/measure_gf_mul_and_static_headroom.py` (record:
  `benchmarks/quantum_compilation/gf_mul_and_static_headroom.json`). `ixor_gf2_mul` is the
  Karatsuba recursion to the gate — 3/7/9/21/27/63 static Toffolis at GF(2²/2³/2⁴/2⁶/2⁸/2¹²),
  matching `K(m)` exactly at every degree. No split of the recursion beats 63 at m=12 (k=8
  ties, the rest are worse; schoolbook 144). The multiply is **86.33%** of the six-lever
  arm's executed count, so a 45-AND multiply would give 5.734× and a free one 31.598× (the
  latter only by the inadmissible measured-operand route). A full-recording peephole removes
  **0** Toffolis at both instances, and the sixth lever changes the static count by exactly 0:
  the emitted stream has no headroom left.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains a record pin for those
  invariants (Karatsuba exactness, the 63 floor, static five == six, zero peephole
  removals, the multiply share), so a later "optimisation" that changes any of them fails
  without the extension being rebuilt.

## 2026-10-29 — DQI MR-gated arms: a sixth lever and qubit parity

- **MEASURED:** `scripts/measure_mr_arm_costs_kickmix.py` prices the gated Benchmark II
  circuits on the rebuilt, MR-capable `kickmix` (official instance, m=4095, n=70,
  GF(2¹²), 64 shots, seed 20261028). The five oracle-gated levers reproduce the
  projection stage for stage: 5,725,071.6 → 1,381,390.0 Toffoli per shot (4.144×)
  against the projected 1,382,207.08 (4.142×), every stacked stage within 0.06% of its
  projected count.
- **NEW LEVER:** each of the sweep's two polynomial divisions is consumed by an add
  controlled on its own counter bit, and the four counter bits are a one-hot decode, so
  in every live round exactly one division is computed under a false add and read by
  nothing. Gating each division on that bit (`gate_quotient`, arm `quotient`) removes
  4.07% of the five-lever count: **1,325,230.0 per shot, 4.320×**. It carries no projected
  row, because the oracle build priced both divisions as live.
- **FIXED (qubits the reads were leaking):** every read allocated a fresh copy qubit and
  the sweep's `q_i` was never freed, so the gated arm's id count grew by roughly a
  register per round — 4,503 ids against the reference's 1,893. One shared read ancilla
  (`mr` clears its target, so reuse is free) and freeing `q_i` in the round that
  allocates it bring both stacks to **1,894** ids: the gated rows now pay one qubit, not
  2,610.
- **PINNED:** `tests/test_dyn_mr_gated_arms_kickmix.py` gains
  `test_small_instance_cost_pin_catches_gate_regressions`, which re-prices the whole
  stack at m=15/n=6/GF(2⁴) in about a second (33,235.5 → 10,409.5, 3.193×) and fails if
  any gate stops firing, changes its saving by more than 1%, or costs qubits — the
  64-shot official run is not needed to catch a regression. `docs/MR_INSTRUCTION_SPEC.md`
  §6b records the new lever and the qubit fix.

## 2026-10-29 — Temporal-rate detector audit hardening

- **RE-PINNED:** `specifications/canonical_preregistered_spec_v1.json` and
  `apedp_dyn/conf/canonical_spec.py` to
  `dbf67c5ac443073721421eba456519a0eda09ae07680d638c26f1d019b0b2343`.
  The only spec change is the new `temporal_rate_min_compositions=2`; the
  two-distinct-composition requirement for a common-mode claim is now
  canonical rather than a hardcoded literal. No previously pinned FPR, power,
  alpha, SNR, amplitude, harmonic, or sample threshold changed. FLAW-01
  remains OPEN; the governance manifest and its expected failing test were not
  touched.
- **FIXED:** `_drive_phasor()` used a raw windowed DFT of the drive. For a
  non-integer number of drive cycles, constant/trend leakage rotated and
  rescaled the estimated phasor; that bias transferred incorrectly between
  calibration and confirmation frequencies and inflated the null statistic.
  The phasor is now fit jointly with a constant and linear trend over
  cosine/sine columns and returns the analytic conjugate. Added
  `test_calibration_transfer_has_no_drive_phasor_bias`, which recovers the
  known susceptibility from a noiseless non-integer-cycle calibration to
  1e-10 relative accuracy.
- **MEASURED (before/after):** before the phasor fix, 512 null trials at 128
  surrogates gave FPR 14/512 = 2.73% (95% upper 4.24%) and 128 trials at 512
  surrogates gave 4/128 = 3.13% (95% upper 7.01%). After the fix: 512 trials
  at 128 surrogates gave FPR 5/512 = 0.98% (95% upper 2.04%), artifact rate
  0/512, power 512/512 (95% lower 99.42%), median null p 0.516; a second
  512-trial seed 7 audit also gave 5/512; 128 trials at 512 surrogates gave
  FPR 1/128 = 0.78% (95% upper 3.65%), artifact rate 0/128, power 128/128
  (95% lower 97.69%), median null p 0.496. Both audits pass the preregistered
  point-estimate gate. Exact Clopper-Pearson bounds are reported
  as diagnostics; the 128/512 upper bound is now below 5% as well.
- **HARDENED:** `ClockDataset` rejects non-uniform sampling; calibration and
  confirmation must use disjoint drive-harmonic sets at the Rayleigh
  resolution; calibration reports per-harmonic SNRs while the insufficiency
  gate remains fundamental-only; `TemporalRateAdjudication` records both
  dataset ids, the frozen analysis parameters, calibration SNRs, and the
  canonical spec hash.
- **FROZEN:** the preregistration schema is now `temporal-rate-v2` and freezes
  `harmonics`, `n_surrogates`, `seed`, `familywise_alpha`,
  `heterogeneity_alpha`, `min_fractional_amplitude`, and
  `canonical_spec_sha256` in addition to dataset fingerprints/ids, drive
  frequencies, and the clock-composition map.
  `adjudicate_preregistered_temporal_rate()` takes exactly three arguments and
  cannot be parameter-overridden after registration. Verification rejects a
  changed spec hash, changed canonical alpha, or fewer than two distinct clock
  compositions. The artifact remains a fingerprint commitment, not a
  signature.
- **CALIBRATION:** `run_detector_calibration()` now refits a fresh calibration
  for every null/injection pair, so the FPR is unconditional on one lucky
  calibration realization. It reports false-candidate, oscillator-specific
  artifact, and any-false-claim rates, plus exact one-sided 95% bounds and
  `n_calibration_draws`; the preregistered pass gate is the point-estimate
  FPR/power floor, with bounds retained as diagnostics.
- **ADDED:** 9 pins, bringing the temporal suite to 44 (18 detector + 26 IO):
  uniform-sampling rejection, harmonic-overlap rejection, per-harmonic SNR
  reporting, phasor-transfer regression, frozen analysis parameters, no
  parameter overrides, tampered spec hash, tampered canonical alpha, and
  distinct-composition enforcement. `AGENTS.md`'s standard suite now includes
  both temporal test files.
- **Verified:** 44/44 temporal pins pass; Ruff is clean on the changed files;
  repo-scope mypy (`python3 -m mypy`) is clean. The expanded dynamic suite is
  162 passed with one pre-existing unrelated failure:
  `test_orphan_module_detection` scans the untracked vendored
  `kickmix/_qualtran_src/`.
- **LIMITS:** common-mode calibration/clock systematics are still unmodeled,
  `systematic_susceptibility_sigma` still defaults to zero, the pure-sinusoid
  drive assumption is documented but not validated against non-sinusoidal
  drives, and colored/non-Gaussian residual noise is not covered by the
  shared-phase surrogate. No real two-composition clock data exists in the
  repo, so these are detector-calibration results, not a measurement of nature.

## 2026-10-29 — `MR` implemented, and the gated arms run on a real `kickmix`

- **ADDED:** `scripts/kickmix_mr_opcode.patch` — the `MR`/`MR_IF` opcodes
  (`docs/MR_INSTRUCTION_SPEC.md`) applied to the vendored `kickmix` sources:
  opcode values 0x54/0xD4 with names, `OP_TYPE_TABLE` and `.kmx` parse/print, a
  `mr`/`mr_if`/`broadcast_mr` builder path with Python bindings, a Z-basis
  simulator branch that writes the outcome and clears the qubit without touching
  the tracked phase, its own statistics group, and a reaction-depth case.
- **ADDED:** `scripts/build_kickmix_mr.py` exports the vendored sources at their
  recorded commit, applies the patch, and compiles `build/kickmix_mr/lib`
  (`build/` is gitignored). The pinned competition binary that
  `apedp_dyn/quantum_compilation/contracts.py` hashes is never modified.
- **ADDED:** `scripts/run_mr_gated_arms_kickmix.py` runs the five gated DQI levers
  on that extension with a real `kickmix.CircuitBuilder` and a real
  `kickmix.Simulator`, so the arms are no longer measured only on the local model.
- **MEASURED:** at m=15/n=6/GF(2⁴) with 8 shots, `none`, `swap`, `sweep`,
  `forney`, `clear`, `shift` and `all` match the reference's syndromes and tracked
  phase, restore the syndromes and leave every non-syndrome qubit `|0>`; `all`
  also passes at m=255/n=14/GF(2⁸) with 16 shots and at the official
  m=4095/n=70/GF(2¹²) with 4 shots (reference 1,893 qubits, arm 4,503 allocated
  ids, phase identical). The falsifiers still fail:
  `--clear-with r` (phase 6 ones vs 3, dirty junk) and `--shift-under`
  (syndromes not restored).
- **RECORDED:** `x(q); mr(q, b)` gives `b = 1` and 0 phase flips on 64/64 shots,
  against 37/64 phase flips for the same circuit with `reset` — the semantic
  difference the instruction exists for.
- **ADDED:** `scripts/measure_mr_arm_costs_kickmix.py` prices the reference, the
  cumulative lever chain and each lever alone with `kickmix`'s conditional-aware
  per-shot Toffoli counting. Record:
  `benchmarks/quantum_compilation/kickmix_mr_arm_costs.json`.
- **ADDED:** `tests/test_dyn_mr_gated_arms_kickmix.py` pins the real-kickmix arm
  runs and both falsifiers, plus the cost record's arithmetic; the extension-backed
  pins skip when the MR build is absent (10 passed under CPython 3.13 with the
  build, 1 passed and 9 skipped under the project's 3.12).
- **MEASURED (cost):** at the official instance over 64 shots the rebuilt builder
  reproduces the pinned reference exactly (5,725,071.6 vs 5,725,071.61) and the
  full gate executes **1,381,390.0 Toffoli per shot, 4.144×** against the
  projected 1,382,207.08 / 4.142×. Every stacked stage is within 0.06% of its
  projection — 4,298,268.3 (Forney), 2,157,906.3 (+clear), 1,590,308.3 (+sweep),
  1,456,286.3 (+swap), 1,381,390.0 (+shift) — so the oracle-gated projection is
  reproduced by the circuits rather than bounding them from above. Per lever
  alone: `swap` 1.049×, `sweep` 1.247×, `forney` 1.332×, `clear` 1.597×, `shift`
  1.027×; the shift lever inside the chain is worth the 5.14% the oracle
  accounting gave it.
- **FIXED:** three gate defects in `scripts/run_mr_gated_arms.py` that a first
  costing pass (3.556×) exposed. The shift gate's `¬flag` was written as a second
  `bit_store(…, True)`, leaving the bit constant, so it skipped only rounds with
  every control clear (1.002× alone). The sweep gate wrapped only the position
  sweep, leaving the quotient divisions and the `n_a` add/undo inside dead rounds.
  The Forney predicate was an 11-Toffoli equality circuit run twice per position,
  with a copy qubit allocated per position; it is now `mr`-read out of `sigma`,
  restored with classically controlled `x`, and assembled from two classical
  stores. Net: 1,609,925.7 → 1,381,390.0 per shot and 8,587 → 4,503 qubit ids.
- **UNCHANGED:** a receipt still requires an upstream `kickmix` rebuild plus the
  official verifier; the local extension is not the pinned binary.

## 2026-10-28 — Preregistration gate for the temporal-rate null test

- **ADDED:** `TemporalRatePreregistration` plus `preregister_datasets()`,
  `save_preregistration()`, `load_preregistration()`,
  `verify_preregistration()`, and `adjudicate_preregistered_temporal_rate()`
  in `apedp_dyn/epistemic/temporal_rate_io.py`. The artifact freezes the
  calibration/confirmation SHA-256 fingerprints, dataset ids, drive
  frequencies, and clock-composition map before adjudication. Verification
  recomputes every field and fails closed on any mismatch, including a swapped
  confirmation file that keeps the same dataset id.
- **EXTENDED:** `scripts/run_temporal_rate_null_test.py` gains
  `--write-preregistration`, `--preregistration`, and
  `--preregistration-notes`. `--write-preregistration` freezes the fingerprints
  without running the test; `--preregistration` verifies before adjudicating.
  Protocol errors now exit 1 with a one-line message instead of a traceback.
- **ADDED:** eight preregistration pins in
  `tests/test_dyn_temporal_rate_io.py`: round trip, swapped confirmation,
  tampered calibration, changed metadata, unknown keys, unsupported protocol
  version, duplicate dataset ids, and end-to-end preregistered adjudication.
  The temporal suite is now 35 pins (14 detector + 21 IO).
- **Verified:** 35/35 temporal pins pass; Ruff is clean. The full dynamic suite
  with both new files included is at 152 passed with the same two pre-existing
  unrelated failures (host-path literal in untracked
  `scripts/inspect_patch_1994.py`; orphan scan over untracked vendored
  `kickmix/_qualtran_src/`).

## 2026-10-28 — Clock-dataset ingestion for the temporal-rate null test

- **ADDED:** `apedp_dyn/epistemic/temporal_rate_io.py` — strict CSV +
  sidecar-JSON and compressed NPZ ingestion for `ClockDataset`. The CSV format
  names `time_s`, `drive`, and one fractional-frequency column per clock; the
  sidecar carries the dataset id, drive frequency, and each clock's id,
  composition, column, and systematic-uncertainty budget. NPZ embeds the same
  metadata and is read with `allow_pickle=False`, so a dataset file cannot
  execute code. Both paths delegate finite/monotonic/length checks to
  `ClockDataset` and expose `dataset_sha256()` for calibration/confirmation
  provenance.
- **ADDED:** `tests/test_dyn_temporal_rate_io.py` — 13 pins covering CSV and
  NPZ round trips, cross-format fingerprint equality, fingerprint sensitivity,
  missing sidecar, missing/unknown metadata keys, duplicate clock ids, empty
  and non-finite cells, unknown NPZ keys, object-array rejection, NPZ
  `metadata_path` rejection, single-clock support, and unsupported formats.
- **EXTENDED:** `scripts/run_temporal_rate_null_test.py` now adjudicates
  supplied files via `--calibration` / `--confirmation` (with optional
  `--calibration-metadata` / `--confirmation-metadata`), reports both dataset
  fingerprints, and preserves the claim boundary. The synthetic demo remains
  the default when no files are supplied.
- **Verified:** the 13 IO pins and the 14 detector pins pass; Ruff is clean on
  the new module, tests, and script. The full dynamic suite, with the two new
  files included, is at 144 passed with the two pre-existing unrelated
  failures (host-path literal in untracked `scripts/inspect_patch_1994.py`;
  orphan scan over untracked vendored `kickmix/_qualtran_src/`).

## 2026-10-28 — Option B: a temporal-rate null test, not a time-travel claim

- **ADDED:** `apedp_dyn/epistemic/temporal_rate_modulation.py` — Hypothesis B
  harness. It subtracts each clock's calibration-measured ordinary response to
  a periodic drive and tests whether the residual phase-locked
  fractional-frequency modulation is common to clocks of at least two distinct
  compositions. The null is a shared-phase Fourier surrogate of the post-fit
  residuals with the fitted harmonic variance restored and calibration
  uncertainty injected; a heterogeneity guard rejects oscillator-specific
  artifacts. Calibration/confirmation dataset-id reuse, identical series, and
  same-frequency calibration raise `TemporalRateProtocolError`. The only
  positive verdict is `COMMON_MODE_ANOMALY_CANDIDATE`, explicitly not evidence
  of backward time travel.
- **PREREGISTERED:** canonical spec gains the `temporal_rate_*` block
  (minimum clocks/cycles/samples-per-cycle, harmonics, surrogate draws,
  familywise and heterogeneity alphas, amplitude floor, calibration-SNR floor,
  and detector FPR/power/injection thresholds). The spec is re-pinned to
  `a5e13979…`; no existing threshold changed. Ed25519 re-signing remains
  trustee-blocked (FLAW-01 still OPEN).
- **MEASURED, synthetic detector calibration:** 20 null and 20 injected
  common-mode trials at 128 surrogates (seed 6) give FPR 0.00 against the 0.05
  ceiling and power 1.00 against the 0.80 floor, with zero insufficient
  verdicts. This measures the analysis, not nature.
- **Verified:** `tests/test_dyn_temporal_rate_modulation.py` 12 passed,
  including null-consistency, common-mode recovery at an arbitrary phase,
  oscillator-specific rejection, protocol reuse/frequency refusals,
  single-composition and single-clock refusals, weak-calibration refusal,
  deterministic replay, and the FPR/power floors.

## 2026-10-28 — The MR arms hold at the official instance

- **MEASURED:** all five arms and `all` reproduce the reference at Benchmark II's own
  instance — m=4095, n=70, GF(2¹²), 4 shots: syndromes, tracked phase and `|0>` junk all
  match. Reference: 107,911 qubit ids, 4m54s. Arm ids: swap 108,355, sweep 108,799,
  forney 112,006, shift 110,131, `all` 84,103, clear 78,232. These are local allocation
  counts, not `Q`.
- **FALSIFIED at the same scale:** the under-gated shift arm fails on syndromes, phase and
  junk, and the `r` readout fails on all three (at this scale its readout also corrupts
  values, not only the phase). Record: `benchmarks/quantum_compilation/local_mr_arms.json`.
- **Verified:** `tests/test_local_mr_sim.py` now pins all three instances.

## 2026-10-28 — Fifth MR lever verified at both instances, with a falsifier that fails

- **ADDED:** the shift gate to `scripts/run_mr_gated_arms.py` (`--arm shift`, included in
  `all`). It reads `c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))` with a real `mr`, using the flag *after*
  `_update_flag`; the probe's earlier attempt with a round-top flag fails because that flag
  is stale. At m=15/GF(2⁴), 32 shots, the predicate is false in 26.7% of round-shots, and
  the arm reproduces the reference's syndromes, phase and `|0>` junk.
- **MEASURED, scaled:** all five arms plus `all` pass at m=255/GF(2⁸), n=14, 16 shots
  (reference 9,751 qubit ids; syndromes, phase and junk all match). The falsifiers fail at
  both instances: `--clear-with r` loses the phase and leaves junk, and `--shift-under`
  (dropping `¬flag ∧ (c0 ∨ c3)`) skips shifts the decoder needs and breaks the syndromes.
  Record: `benchmarks/quantum_compilation/local_mr_arms.json`.
- **REJECTED as a falsifier:** inverting the shift gate also passes, because one skipped
  shift is repaired by later rounds; only under-gating breaks the decoder, so the inverted
  knob was removed rather than shipped as a test that cannot fail.
- **Verified:** `tests/test_local_mr_sim.py` 21 passed over both recorded instances,
  including the under-gated shift arm and the `r` readout.

## 2026-10-28 — Three metric placeholders now measure or refuse

- **FIXED, `EnsembleSearcher.get_diversity_score`:** returned a constant 0.8 with a
  "would analyze equation structures" comment. It now reads the solutions out of the
  results (`best` / `expression` / `expressions` / `equations` / `solutions`) and scores
  `(distinct - 1) / (total - 1)`: 0.0 when every member agrees, 1.0 when no two do. Fewer
  than two reported solutions — including its own `search_ensemble` plan — raises, because
  a plan has no diversity to measure.
- **FIXED, `DeepEnsemble.predict`:** reported `aleatoric=jnp.zeros_like(std)`, presenting an
  unmodelled quantity as a measured zero. These regressors train deterministically, so the
  field is `None`; the monkeypatch that previously masked the same placeholder after import
  was removed with it.
- **FIXED, `PhysicsConstraintEngine.dimensional_analysis_score`:** returned 0.5 with no
  target and 1.0 with one, regardless of any candidate. It now takes the dimensions of the
  candidate's terms and returns the fraction that match the target, raising when there are
  no terms or no target. The Rust kernel had the identical placeholder and now implements
  the same measurement over a JSON term list (rebuilt extension); the Python fallback and
  the kernel agree, and both `check_dimensional_consistency` paths now compare all seven
  exponents instead of mass/length/time only.
- **Verified:** `tests/test_placeholder_metrics_are_measured_or_refused.py` 9 passed (both
  implementations of the dimensional score, the diversity endpoints, the refusals, and
  `aleatoric is None`); `tests/test_search_engines_real.py` 13 passed.

## 2026-10-28 — MR arm correctness tested locally: four levers, and `r` fails as it should

- **BUILT:** `apedp_dyn/quantum_compilation/local_mr_sim.py`, a Z-basis (value + sign)
simulator for the DQI decoder with `mr` (Z measure-and-reset, phase-preserving), `r`
(clears, randomizes the phase, as `kickmix` does) and `hmr` (X basis). It implements the
builder subset the decoder uses (`unary_iteration`, `init_and`, `iadd`/`isub`,
`flip_if_*`, rotations, GF(2^b) arithmetic, `gf2_phase_by_product`) with semantics
pinned against `kickmix`, including its primitive polynomials.
- **VALIDATED:** `scripts/verify_local_mr_sim.py` builds the reference decoder with both
engines and feeds both the same shots. At m=15/n=6/GF(2⁴): syndromes identical, phase
identical, syndromes restored, every other qubit `|0>`, 0 phase errors — the local engine
reproduces `kickmix` exactly (qubit *ids* differ because freed ids are not reused here).
- **MEASURED:** `scripts/run_mr_gated_arms.py` runs the four gated levers with their
predicates read by a real `mr` through `cx(control, ancilla); mr(ancilla, bit)`, not the
oracle registers the probe used. Every arm reproduces the reference's values AND phase
at 32 shots: `swap` (c3), `sweep` (c0 ∨ c3), `forney` (the `sigma == 0` zero-test),
`clear` (measured clear of the EEA instead of its adjoint, with the syndromes re-prepared
from a saved readout), and all four together. Record:
`benchmarks/quantum_compilation/local_mr_arms.json`.
- **FALSIFIED, by construction:** the same swap arm read with `r` instead of `mr` fails on
phase and leaves junk behind while its syndromes still match — so the pass is about the
measurement, not the circuit. This is the mechanism-level version of the 11/16 phase
defect the probe recorded.
- **STILL UNSCORED:** the arms' cost. `local_mr_sim` computes values and phases, not gate
counts, and `contracts.py` pins the official `kickmix` build and verifiers, so the 4.142×
stays a Toffoli bound until `MR` exists upstream.
- **Verified:** `tests/test_local_mr_sim.py` 21 passed (pinned primitives, the MR/R/HMR
contract, and the arm record).

## 2026-10-28 — Search engines score on data, and NGTS runs UCT or nothing

- **MEASURED, not defaulted:** `ValidationPipeline.validate` now reports
  `generalization_gap` (held-out minus cross-validation error) and derives
  `generalization_score` from it as `1/(1+gap)`. The old score was a lookup — 0.9 below
  a threshold, 0.7 above — so it reported a number the data could not move.
- **No fabricated evidence:** `NGTSSearch.run_simulations` runs UCT and calls a reward
  function for each selected leaf. With no reward function it runs **zero** simulations
  and returns 0; it previously returned `num_sims` and backpropagated
  `random.random()*0.5`, i.e. reported search work it had not done. The Rust kernel had
  the same defect in `ngts.rs` (`rand_simple` values) and now takes the reward callable
  in `run_simulations(num_sims, reward=None)`.
- **Hierarchical phases score what they propose:** `_mid_search` / `_fine_search` take
  `data`/`target`, evaluate each expansion on the data, and return nothing when there is
  no data. `HierarchicalSearch.search` attaches a measured `r2` to every candidate and
  **drops** candidates it cannot evaluate; it used to validate every candidate against
  default `train_error=0.1 / test_error=0.15`, so an unmeasured expression carried a
  validation score.
- **Ensemble reports what it cannot run:** a proposer outside `ProposerRouter`'s map is
  `status: error` instead of a planned member, and `best` is present and `None` until a
  member returns a scored candidate.
- **Determinism:** `beam_search_expressions` no longer consumes `seed` for sampling; two
  seeds return the same list, and level 1 pairs atoms with each other so multi-variable
  forms appear immediately.
- **Verified:** `tests/test_search_engines_real.py` 13 passed (was 12 of 13 failing);
  related suites 58 passed / 2 skipped / 1 xfailed.

## 2026-10-28 — Scratch pool measured dead: 25-qubit ceiling, and the allocator claim withdrawn

- **MEASURED** on the pinned Benchmark II instance (`scripts/measure_dqi_scratch_pool.py`,
  record `benchmarks/quantum_compilation/dqi_scratch_pool.json`): the circuit holds
  **1,828 qubits concurrently live against 1,893 distinct ids**; the peak is reached
  inside `_chien_search_and_forney_mbuc` and the innermost frame there holds **25**
  qubits. A shared scratch pool can therefore redistribute at most 25 — the −104 the V6
  row claims is 4× that ceiling and is not reachable by pooling.
- **CORRECTED, committed claim:** §5.5 of the boundary doc said the live peak was EEA
  round scratch in `_multiply_and_add_or_sub` and that a shared pool was "the only
  mechanism that could move" the transients. Both are withdrawn on measurement: the peak
  is in Chien/Forney, and the pool is dead.
- **Left OPEN, not answered:** the probe's trace replay returns 1,821 ids against 1,893
  shipped and observes only 1,873 ids, so allocations made inside the extension are
  missing; a lowest-hole free list replays identically to the shipped LIFO stack. An
  earlier draft of the script asserted the 65 stranded ids were `alloc_qubit` LIFO
  stranding and named a free list as the fix. That verdict was not supported by the
  replay and has been removed rather than shipped.
- **Verified:** `tests/test_dqi_scratch_pool.py` pins the artifact; the probe runs under
  CPython 3.13 only, since `kickmix.so` is a 3.13 build.

## 2026-10-28 — Post-`_update_flag` probe collected: 4.142× measured

- **MEASURED:** the shift gate the previous entry had to abandon is now correct. The
  wrong arm read `flag` at the top of the round; `_update_flag` toggles it first, so the
  probe now carries six bits per round and the liveness is
  `c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))` with the post-update flag. Official instance:
  `T_mean` 1,457,103.45 → **1,382,207.08, a 4.142× over the reference**, `Q` unchanged at
  1,893, **0 value / 0 phase errors on 64 shots**, 1,332 round bits (6 × 222).
  Scaled: 3.709× at m=255 (512 shots), 2.998× at m=15.
- **Corrected prediction:** the prize was bounded at ~5% from the failed arm's cost and
  measures at 5.15% (1,457,103 → 1,382,207). Dead shifts are 31.59% of rounds, not the
  35–40% guessed before the fix.
- **Running total:** Forney gating 1.332×, MR clear 2.652×, sweep 3.599×, swap 3.929×,
  shift 4.142×. Receipt arithmetic 24.439T → 5.900T, portfolio 2.361982 → 3.79; still
  3.10× short of the 12.86× target.
- **Verified:** `tests/test_dqi_predicate_gating.py` 12 passed / 1 skipped.

## 2026-10-28 — Dead register swaps gated: 3.929× measured, and the shift gate falsified

- **MEASURED:** `_swap_a_and_b` exchanges `reg_a`/`reg_b` wholesale — 2 × 73 × 12 = 1,752
  qubits — every round, but its control is `swap_ctrl = c3_hc ∧ flag`, so it is dead in
  any round without `c3` (**67.57%** of rounds). Gating it on `c3` alone: official
  `T_mean` 1,590,938.28 → **1,457,103.45, a 3.929× over the reference**, `Q` unchanged at
  1,893, **0 value / 0 phase errors on 64 shots**, for 1,110 round bits (5 × 222).
  Scaled: 3.507× at m=255 (70.4% dead), 2.813× at m=15. Receipt arithmetic 24.439T →
  6.22T, portfolio 2.361982 → 3.7271; still 3.27× short of 12.86×.
- **FALSIFIED (mine, one hour old):** gating `_shift_b_array_right` on the complementary
  pair `c1 ∨ c2 ∨ ¬flag`. It looked right and fails on **every shot** (512/512 value
  errors at m=255). Cause: `flag` is toggled by `_update_flag`, which runs *before* the
  shift in the same round, so a flag read at the top of the round is stale. The arm's
  cost (62,434 at m=255 against 65,698) bounds the remaining prize at ~5%, and collecting
  it needs a second probe point after `_update_flag`. Not taken; pinned as a negative in
  `tests/test_dqi_predicate_gating.py`.
- **Cost of the whole design so far:** 1,803 classical bits for the clear, 1,110 for the
  round predicates, zero extra qubits, zero extra Toffolis.

## 2026-10-28 — Dead multiply sweeps gated: 3.599× measured

- **MEASURED:** the synchronized EEA emits all five steps every round, but
  `_multiply_and_add_or_sub` touches `ctrl` only through `c0_hc` and `c3_hc`. In a round
  where both are zero — the two shift rounds of each four-round counter cycle — its
  4,536-Toffoli sweep is emitted and does nothing. Reading `(c0_hc, c3_hc)` per round
  and gating the sweep on `c0 ∨ c3` removes **35.14%** of the forward EEA at the official
  instance: `T_mean` 2,158,651.45 → **1,590,938.28, a 3.599× over the reference**, with
  `Q` unchanged at 1,893 and **0 value / 0 phase errors**. Scaled: 3.205× at m=255
  (40.79% dead), 2.586× at m=15 (46.98% dead).
- **Cost:** one classical bit per EEA round — 222 bits at the official instance, 54 at
  m=255, 30 at m=15 — no qubits, no Toffolis. Still an upper bound: the round bits are
  read from the same circuit in a probe pass.
- **Not taken:** `_shift_b_array_right` is live in the `c1`/`c2` rounds *and* in the
  `c0`/`c3` rounds when `flag` is clear (`ctrl_nB = c0_nf ⊕ c1_hc`), so it cannot be
  gated on the complementary pair without also reading `flag`; left alone.
- **Verified:** `tests/test_dqi_predicate_gating.py` 9 passed / 1 skipped.
- With the Forney gating and the MR clear this is 3.599× of the 12.86× the target needs:
  `T_mean` 1.591M against 445,229 leaves a 3.57× gap, and the receipt moves 24.439T →
  6.791T (portfolio 2.361982 → 3.6198), arithmetic on `R·Q²·T`, not a receipt.

## 2026-10-28 — MR clear halves the EEA: 2.652× measured, and `R` is not a substitute

- **MEASURED (the better design):** the reversible EEA is run twice — forward, then
  its adjoint to uncompute — so half of Benchmark II's 75% EEA share is uncomputation.
  Clearing the EEA scratch and the syndromes and re-preparing the syndromes from a
  classical readout replaces that half. `scripts/probe_dqi_predicate_gating.py`
  measures it with both oracle registers (syndrome readout + root bits): official
  instance `T_mean` **5,725,071.61 → 2,158,651.45, a 2.652×**, `Q` unchanged at 1,893,
  **0 value and 0 phase errors** on 64 shots, needing 1,803 classical bits and 1,803
  classically controlled X's. Scaled: 2.351× at m=255 (512 shots), 2.012× at m=15.
  Because `Q` does not move, the score falls by the same 2.652×: 24.439T → **9.215T**,
  i.e. the portfolio score 2.361982 → **3.2695** (arithmetic on `R·Q²·T`, not a receipt).
  Still 4.85× short of the 12.86× target.
- **FALSIFIED (mine, from four hours earlier):** "`R` can stand in for uncomputation."
  A reset clear costs exactly what the reference spends (2,158,651.45 either way — CX
  is not Toffoli-like) but it is **wrong**: 35/64 shots at the official instance and
  254/512 at m=255 have a flipped phase. Cause, measured: a circuit with **no phase in
  it at all** — `x(q); reset(q)` — reports a random `phase_flipped` (11/16 shots), while
  the same circuit without the reset reports 0/16. `R` randomizes the tracked phase, so
  it can only be used where the answer does not depend on it. Measure-then-X on the
  measured value (i.e. `MR`) gives 0/0.
- **BOUND (corrected):** with MR, both levers together are worth 2.652×, not the 1.332×
  the Forney-gating entry below states: that entry bounded *one* phase of the decoder and
  its "even granted MR, T lands at 4.30M" bullet is superseded here. `T_mean` 2.159M is
  still 4.85× above the 445,229 target, so the §5.5 conclusion stands — the target needs
  a decoder, not an optimisation.
- **Still blocked in the IR:** `MR` does not exist. `push_condition` takes a classical
  bit; `hmr` is X-basis; `push_condition(qubit)`/`bit_store(control=qubit)` are rejected;
  `R` is phase-unsafe. The instruction that would make 2.652× real is Z-basis
  measure-and-reset, and this design needs it twice.
- **Verified:** `tests/test_dqi_predicate_gating.py` 7 passed / 1 skipped;
  the skipped re-measurement was run by hand under the kickmix interpreter and reproduces
  the record exactly. Record: `benchmarks/quantum_compilation/dqi_predicate_gating.json`.

## 2026-10-28 — Predicate-gated Forney body: 1.332× measured, and inexpressible in this IR

- **MEASURED:** `scripts/probe_dqi_predicate_gating.py` gates the Forney body on the
  truth table of `sigma(alpha^j) == 0` (supplied as an oracle classical register, so
  this is an upper bound, not a submission). At the official instance `T_mean` falls
  **5,725,082.58 → 4,298,955.87 (1.332×)** with `Q` = 1,893 and the static count
  unchanged, and with **0 value and 0 phase errors** on 128 shots: the body is a
  no-op wherever `located = is_sigma_zero & omega` is zero, so skipping it is free
  in correctness. This is the 1.335× "if the phase were free" row of
  `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.5, measured instead of estimated.
  Scaled checks: 1.185× at `m=255, GF(2^8)` and 1.011× at `m=15, GF(2^4)`, where the
  root density (2.03%, 12.7%) is much higher.
- **CORRECTED (null optimisation):** the committed `is_error` gate at
  `dqi_reference_circuit.py:573`/`:596` buys **0.006%** — 353 of 5,725,461 Toffoli per
  shot — because `P(M_j = 0) = 2^-12`. The gap between `toffoli_count` and
  `max_toffoli_count` in the receipt is that gate firing about once per shot, not a
  conditional-execution saving worth pursuing.
- **BLOCKED (IR):** the predicate cannot reach the condition stack. `push_condition`
  requires a `km.b`; the only readout is `hmr`, which is **X-basis** (on `|1>` it
  returned 31/64 ones); `push_condition(qubit)` and `bit_store(control=qubit)` are
  both rejected by the binding; and the instruction set `{NEG, BIT_INVERT,
  BIT_STORE0/1, X, Z, R, HMR, CX, CZ, SWAP, CSWAP, CCX, CCZ, Z_POW, INFO_TAG,
  DEBUG_PRINT, PUSH/POP_CONDITION}` is closed under the Z-diagonal group, so no
  Hadamard can be synthesized to conjugate the predicate into the X basis. One
  instruction — Z-basis measure-and-reset (`MR`) — is the entire difference.
- **BOUND (superseded):** 1.332× leaves `T_mean` at 4.30M against the 445,229 target
  (9.7× short) and the receipt at 18.35T — arithmetic on `R·Q²·T`, not a receipt. The
  Forney body is 24.9% of `T`; the other 75% is EEA iterations, which never skip. See
  the entry above: clearing instead of uncomputing that EEA reaches 2.652×, so this
  1.332× prices only the Forney phase.
- **Verified:** `tests/test_dqi_predicate_gating.py` 4 passed / 1 skipped; the skipped
  re-measurement was run by hand under the kickmix interpreter and reproduces the
  record exactly. Record: `benchmarks/quantum_compilation/dqi_predicate_gating.json`.

## 2026-10-28 — Benchmark II cost structure measured; two of my own optimisations falsified

- **MEASURED:** `scripts/measure_dqi_primitive_costs.py` recovers the DQI Toffoli
  breakdown, since `kickmix` exposes none. Per-call at GF(2¹²): `gf2_imul` and
  `gf2_idiv` **0** (CNOT-only linear maps), `ixor_gf2_mul` 63, Itoh–Tsujii
  inversion 315 = 5 × 63, scaffold division 378. Slopes at the benchmark widths:
  17,208 Toffoli per EEA iteration, 351 per Chien/Forney position.
- **FALSIFIED (mine):** "composite-field/Karatsuba cuts constant-multiplier cost
  30–50%" — there is nothing to cut. Multiply and divide by a constant are already
  0 Toffoli. The only multiplier that costs anything is quantum × quantum, and 63
  at degree 12 is already better than a naive schoolbook count.
- **FALSIFIED (mine):** Montgomery batch inversion of the Forney denominators. It
  needs 3 multiplications per element, so it can recover at most 2/5 of an
  inversion; at this repo's marginal qubit price 2T/Q = 6,048.7 Toffoli-equivalents
  per qubit, and 12 qubits per batched element, **every block size loses** (best
  B=4: 193,489 saved vs 290,338 spent, score ratio 1.021 — worse).
- **BOUND (new):** the Forney step inverts once per position
  (`dqi_reference_circuit.py:579`), so 4,095 × 315 = **1,289,925 Toffolis = 22.5%
  of `T_mean`, and 2.90× the entire 445,229 target** before any EEA iteration runs.
  Ceilings with `Q` unchanged: 1.259× if a position cost one multiply, 1.335× if
  the phase were free — against 12.86× required. The target is unreachable for this
  architecture, not merely unreceipted.
- **Q floor:** 1,887 measured = 1,752 persistent (`2 × 73 × 12`, both frozen) + 135
  transient. The claimed −104 is 77% of all transient qubits, and the live peak is
  EEA round scratch inside `_multiply_and_add_or_sub`; the `n_a`/`n_b` recycling
  the V6 row names frees 16.
- **Verified:** `tests/test_dqi_primitive_costs.py` 7 passed / 1 skipped; the
  skipped re-measurement was run by hand under the kickmix interpreter and
  reproduces the record exactly. Recorded in `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.5.

## 2026-10-28 — Riemann off-line scan scoping fixed; August batch screened for Benchmark II

- **FIXED (integrity):** `rust/riemann_verifier` built its off-line critical-strip grid
  as `(100..400) * 0.1` — `t in [10, 40]` — **regardless of `max_t`**, then printed
  `OFF_LINE_GRID_SCAN_PASS`. A run asked about `T=300` certified a window 260 units
  shorter than the question; the same defect class as the `PASS` removed from the DQI
  Gluzdov V6 projection the same day. The window is now `[10, max_t]` and is never
  truncated to fit the sample budget: the budget (20,000 samples) degrades resolution
  instead, and a coarsened scan reports `OFF_LINE_GRID_SCAN_PASS_COARSENED` rather than
  a bare PASS. JSON gains `grid_scan_window`, `grid_scan_step`, `grid_scan_samples`,
  `grid_scan_coarsened`, `grid_scan_covers_requested_height`.
- **Verified:** `T=60 → [10,60] step 0.1 / 501 samples`; `T=300 → [10,300] step 0.1 /
  2,901 samples`; `T=5000 → [10,5000] step 0.2495 / 20,000 samples / PASS_COARSENED`.
  `tests/test_riemann_verifier_grid_scan.py` 3/3, including a source-level revert pin.
  `probe_wave1.json` / `PROBE_WAVE1.md` regenerated against the fixed binary; verdicts
  unchanged (2 consistent_finite_range, 1 no_refutation, 1 not_probeable).
- **RECORDED (unfixed, separate):** root discovery at a fixed 0.05 step still misses
  zeros with height — 0 % at T≤100, 14.5 % at T=300, 84 % at T=5000 (721/4520). That is
  a discovery-budget problem, not the scan window, and it is now stated as such.
- **NEW:** `catalog/august_ten_advances_screen.json` +
  `catalog/AUGUST_TEN_ADVANCES_BENCHMARK_II.md` — the August 1 *Ten advances* batch is
  the only OpenAI release whose disciplines intersect Benchmark II (coding theory,
  arithmetic circuit complexity, quantum complexity, lattice cryptography), so it is
  screened against the DQI cost model. **0 of 10 rows admits a cost path**: three are
  lower bounds or hardness results (a barrier is not a tool), six have no circuit content,
  and the single coding-theory row is blocked by the frozen instance
  `(m=4095, n=70, degree=12, r=2016)` that `ProblemSpecDQI` fixes. Its only use is to put
  a floor under `R(eps_plus)`, not to lower the score.
- **Disclosed:** recomputing the target gain from the two workload rows gives 14.3973×,
  against the quoted 14.396× — a +0.009 (0.06 %) rounding difference recorded rather than
  absorbed. `T_mean` must fall to 7.78 % of reference (12.86×); `Q` only to 94.5 % (1.058×).
- **Verified:** `tests/test_august_ten_advances_benchmark_ii.py` 7/7, which cross-checks
  the frozen instance against the verifier's own `DEFAULT_M/N/DEGREE/R` and the cost model
  against the pinned workloads in `apedp_dyn.quantum.dqi_photonic_resource`.

## 2026-10-28 — Audit artifacts for the negative results, and the PDE honesty note

- **COMMITTED:** the seven `scripts/` audit instruments that back numbers already
  recorded in `CHANGELOG.md` and `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` had been
  left untracked, so the claims cited them from outside the repository. Two were
  load-bearing: the committed `run_halt_recycling_scale.py` imports
  `RecycledFlagDecoder` from `scripts/test_recycled_flags_small.py`, and the
  restored documentation cites `scripts/test_viable_optimization_path.py`.
  Verified by running the committed driver end to end (−240 counter-only at
  m=15/1023, +600/+360 flag, +240/+480 combined).
- **DISCLOSED:** importing `RecycledFlagDecoder` is safe; running it is not. Its
  toy circuit fails the official verifier (effective error rate 0.9155 ≥ 1/16,
  198/256 value errors). The flags prototype is a broken circuit, not merely a
  slower one.
- **NEW:** `docs/pde_discovery_honest_limits.md` plus
  `tests/test_dyn_pde_discovery_noise_robustness.py` — states that the default PDE
  discovery config is brittle to small additive noise and that the clean pinned
  demos are not evidence of general noise robustness, with the first fixed-config
  multi-seed sweep as the backing test (2 passed).
- **NEW:** `tests/test_dyn_dqi_photonic_resource.py` (14 passed) pins the shipped
  `dqi_photonic_resource` mapper: platform invariance of `R·Q²·T`, the fault-tolerance
  verdict, and the caller-supplied superconducting baseline having no defaults.
- **NEW:** `tests/test_dyn_rime_ad_input_contract.py` pins the `RimeAdProposer`
  input contract (`DataFrame | dict[str, DataFrame]`, unsupported shapes fail
  loudly rather than returning a candidate).

## 2026-10-28 — Portfolio synthesis names the real Benchmark I incumbent

- **CORRECTED:** `docs/invsqrt_frontier_and_portfolio_synthesis.md` still named the
  retired 90Q affine build ($157,752.008809$) as champion and carried a Benchmark I
  factor of 13.164804×. The pinned incumbent is the 89Q shared-PDDD circuit
  (`PASS`, 200,540,928 shots, `ce04c9d5…`) at $C_1 = 157,601.080829$, i.e.
  **13.177411×**. The retired rows (90Q affine, SABLE-90) are kept on the record
  with the reason each was retired rather than deleted, so the champion's margin
  stays checkable.
- **Recorded:** the FAROS-R1/R2/R3 rejection margins are re-stated against the
  89Q basis (R2 15-bit now +1,590.87 / +1.01 %, R1 +41,576.84 / +26.4 %); every
  rejection conclusion is unchanged and every margin grows.
- **Disclosed:** Benchmark I's worst error/tolerance is 0.999623, so 0.0377 % of
  the phase budget is unused — less than a single rotation is worth (see the
  rotation ledger in `PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.4).

## 2026-10-28 — Gluzdov V6 stops claiming a result it never had

- **FABRICATION REMOVED:** `dqi_gluzdov_v6.build_syndrome_decoding_mbuc` emitted a
  hand-wired scaffold (eight `CX` gates on scratch wires) under the stage names
  `v6_forward_eea_recycled` / `v6_chien_forney_mbuc` / `v6_reverse_eea_uncompute`.
  It was not a decoder: no EEA, no Chien search, no Forney evaluation. The module
  docstring nevertheless announced a 14.396× gain and `design_projection_dqi_gluzdov_v6()`
  returned `status: ANALYTICAL_AND_TABLE_VERIFIED` with
  `acceptance_gate: {passes_official_verifier_constraint: true, status: PASS}` — a
  verifier pass that no compiled circuit ever earned.
- **Now:** the circuit builder delegates to the real
  `ProosAndZalkaDecoder.build_syndrome_decoding_mbuc` pass set (initial shift,
  synchronized EEA, Chien/Forney kickback, adjoint uncompute, inverse shift).
  `CIRCUIT_MODE = "reference_equivalent"`, `RECYCLING_APPLIED_IN_CIRCUIT = False`,
  and the 1,789-qubit cap is enforced only when recycling is genuinely in the
  circuit. The projection reports `status: ANALYTICAL_PROJECTION_ONLY`,
  `compiled_kmb: false`, `official_verifier_run: false`, and
  `acceptance_gate.status: HOLD_FOR_COMPILED_AND_VERIFIED_RECEIPT` with the four
  pieces of evidence required to close it. The 14.396× factor is a target.
- **NEW:** `dqi_builder.build_candidate(n_iters, architecture=...)` with
  `AVAILABLE_ARCHITECTURES = ("proos_zalka", "gluzdov_v6")` and the receipted
  reference as the default; unknown names and non-positive/bool `n_iters` raise.
- **Verified:** `tests/test_dqi_gluzdov_v6.py` 10 passed / 1 skipped (the skipped
  one needs the kickmix extension).

## 2026-10-28 — Restore the deleted module registry in `docs/DOCUMENTATION.md`

- **INTEGRITY REPAIR:** commit `c08acbe` ("Document stability selection") replaced
  the 2,134-line `docs/DOCUMENTATION.md` with a 35-line stub, deleting the module
  registry (310 of 312 `apedp_dyn` modules lost their index entry, 429 module
  links, and the `[5 %, 90 %]` tactical-bound discussion). Nothing flagged it
  because the four firewalls in `tests/test_documentation_and_spec_integrity.py`
  were left failing and reported as pre-existing. They were not pre-existing —
  they were the regression being detected.
- **Restored** the registry from `444c1fa` (2,347 lines, 523 repository-relative
  links, 312/312 modules indexed) and re-applied the additions made since
  (stability selection, DQI photonic resource mapping, 89Q champion
  reproduction, mixed polarity, photonic boundary) as §23. The restored text
  carries the numbering defect it had at `444c1fa` (two `## 18` and two `## 19`
  headings, sections 16/17/20/21/22 headingless); it is left as-is rather than
  renumbered, because renumbering breaks every `#N-...` anchor the TOC uses.
- **Added** the one genuinely unindexed module, `apedp_dyn/video/collision_domain.py`.
- **Verified:** `tests/test_documentation_and_spec_integrity.py` 6/6 pass (link
  coverage ≥ 200, zero orphan modules, zero pinned debt, registry status tags
  agree with code `Status:` lines).

## 2026-10-28 — Rotation overhead closed, halting-counter saving falsified, photonic artifact

- **MEASURED (negative):** the 44 priced rotations in the 89Q invsqrt champion
  cost 315.299429 of the 1770.798661 Toffoli-equivalent (17.81 % of the score)
  and cannot be reduced. `scripts/search_rotation_merges.py`: 0 exact mergeable
  pairs (all 44 sit on distinct qubits; the 7 qubits carrying a second rotation
  have 75–192 non-commuting gates between the two), and 0 affordable drops (the
  smallest rotation, 2⁻³⁵ = 2.998e-11 turns, is 7.5× the entire 3.975e-12-turn
  phase slack). Recorded in `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.4.
- **FALSIFIED:** halting-counter recycling does not hold at scale.
  `scripts/run_halt_recycling_scale.py` measures −240 Toffolis at every instance
  (constant, because one fixed-width comparison per call site is removed):
  −0.722 % on the toy m=15/n=6/GF(2⁴), −0.0118 % at the benchmark instance
  m=4095/n=70/GF(2¹²) (2,033,345 → 2,033,105). The shipped prototype (flag +
  counter) is a **regression** at every size measured (+480 at full scale). The
  sFFCC table in `PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.2 is corrected accordingly.
- **NEW (non-submission):** `scripts/report_photonic_non_submission.py` prices all
  three champions on the sFFCC lattice at L=16 and writes
  `artifacts/quantum_compilation/photonic_non_submission.json` with
  `submission_status: NON_SUBMISSION`, `score_delta: 0.0`, and the reasons: the
  verifiers consume a compiled `.kmb`/`.kmx` artifact, and none of the three
  score formulas contains a time, photon, fusion or detector term.
- **NEW:** `docs/BENCHMARK_BREAKING_LITERATURE.md` — published techniques priced
  against each benchmark's measured exchange rate (I: 89.0 units/Toffoli vs
  1770.8/qubit; II: `R·Q²` vs `2RQT`; III: 35,990.0/Toffoli vs 4,596,711.8/qubit),
  including the two techniques that provably cannot help (better Clifford+T
  synthesis cannot move Benchmark I, whose rotation cost is the benchmark's own
  formula of ε alone).

## 2026-10-27 — Stability selection utility + blind-benchmark PDE work

- **NEW:** `apedp/stability_selection.py` — general stability-selection helper
  (`stability_select`) for repeated sparse fits with different seeds/subsamples:
  keeps terms with selection frequency >= pi, averages coefficients, returns empty
  dict on all-failure. Honest caveats: only helps stochastic fitters; deterministic
  fitters see no benefit.
- **WORKSPACE (not committed to repo):** blind-benchmark weak-form PDE fitter
  and stability-fit variant in `/tmp/buffy/blind/`. Results: stability selection
  (PI=0.9 chosen on VALID, frozen, THRESH=0.02, K=20, 5 seeds/BENCH)
  improves exact recovery from 50/100 to 66/100 on BENCH heat/burgers/allen_cahn/
  fisher_kpp. Heat still fails at noise 0.05/0.1 (0/5 each). Full artifacts:
  `/tmp/buffy/blind/bench_stability.json`, `benchmarks/mutation_battery_report.json`
  and the other `/tmp/buffy/blind/*.json` still on disk (no process delete).
- **Caveats recorded on disk:** `/tmp/buffy/blind/limits.txt` (reproducibility,
  seed count, noise model, synthetic data, heat failure, K=20 choice).

### Honest limitations (not fixed, disclosed)
- The utility is NOT a general PDE-discovery fix. Repo PDE path
  (`apedp_dyn/epistemic/pde_discovery`, Rust STRidge) is deterministic and was
  NOT improved by this; on a sample heat+noise case it still returned many
  spurious terms. Stability selection applies only to fitters that randomize
  their structure (here: weak-form test-function windows).
- Heat remains the hard case under noise; this is a partial reduction of spurious
  terms, not a complete solution. Root cause points to library/sparsity, not
  threshold/selection tuning alone.
- Validation-BENCH split: PI=0.9 was picked on VALID only and frozen before BENCH,
  so there is no benchmark tuning. But the result still depends on a single
  frozen pi and a single threshold; different noise models / domains may need
  re-tuning on a held-out split.

### Non-deleted working processes on disk
The following working directories / artifacts exist and were NOT deleted:
- `/tmp/buffy/blind/` — blind PDE benchmark scripts + results + logs.
- `benchmarks/mutation_battery_report.json` — mutation battery report (still
  referenced by the repo).
- `scripts/search_claims_pddd.c` — search-claims probe source (new, not deleted).
- Other existing `/tmp/` working artifacts remain as-is; this commit does not
  touch them.

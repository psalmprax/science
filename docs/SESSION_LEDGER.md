# Session Ledger — measured results not covered elsewhere

**Purpose:** every number below was executed, with budget/seed recorded.
Status labels: CONFIRMED (reproduced), SINGLE-RUN (one seed/config),
WITHDRAWN (retracted after further testing).

## Discovery benchmark (Tier A/C)

- A1 recognition 3/3 → 4/4 after GRATING added (CONFIRMED, fast tests)
- Tier C deception 4/4 → extended set, all pass (CONFIRMED, fast tests)
- A2 exact recoveries, all withheld-prior, seed 7 (SINGLE-RUN each):
  OHMIC 15×40 ✓ (mse 0.0) · KINETIC 15×40 ✓ (0.0006) ·
  EINSTEIN 15×40 ✓ (0.0005) · SB 15×40 ✓ after power-mutation fix (0.0005)
- A2 SB failures before fix: 15×40 T²-decoy, 25×60 linear decoy,
  power-prior scaffold no-effect (identical mse), log-space linear decoy
- Ensemble OHMIC seeds 7/42/123 small-budget: exact, linear-decoy, exact
  → recovery 2/3, winner auto-selected (SINGLE-RUN)
- Tier D noisy-Ohm 5%: `0.989·x·y+0.02·y`, structure adjacent, exact form
  lost (SINGLE-RUN)

## Heat transfer (Nu)

- Incumbent DB held-out MAE 8.148; candidate 7.716 on original sample.
- WITHDRAWN as superiority: fresh seeds give 8.14/7.79 (p=0.21),
  7.74/7.88 (p=0.12), 7.55/7.96 (p=0.06) — winner flips, nothing <0.05.
- Standing result: rational operator expands reachable structure (√Pr);
  no superiority claim. Candidate demoted to CONDITIONAL (7/625 domain
  violations, worst −22.89).
- Extrapolation: interior DB wins, low-Re tail candidate wins, high-Re
  tail both collapse (DB less). Local-surrogate verdict stands.

## Friction (Colebrook truth, synthetic)

- Haaland MAE 0.000124. Discovered additive form 0.00619 (50× loss).
- With logRe terminal: `24.45 − 29.94/logRe^(1/3)`, MAE 0.0037 (still
  30× short). Ablation ladder recorded; log-composition unreached.

## Batteries (NASA PCoE B0005/6/7, real data)

- Fixed forms: exp/linear best B0005/B0007; double-exp wins B0006.
- Quadratic (demography transfer): worst everywhere + false EOL on B0007.
- Discovery run (B0005, 15×40): `1.824 − 5.54e-7·k³`, MAE 0.79 vs 0.059.
  Single equation per run; seed-stability untested.

## Warp battery (see docs/WARP_DRIVE_STUDY.md)

- Recognition: Lorentz exact 1.7s; expansion→CONDUCTION structural 18.3s.
- 6 composites generated; #3/#5 correspondence-verified (c→∞ True).
- Energy dataset `data/warp_energy_estimates.jsonl` (9 rows);
  FP-vs-Lentz wall-scaling agrees within ~60× across 33 orders.

## Statistics (see docs/STATISTICAL_EQUATIONS_STUDY.md, POPULATION_STUDY.md)

- US quadratic exact form; Bertozzi refutation table; D-T 17.589.
- Rolling-origin corrected counts (2/6-4/6-0/6); full sweep 785 windows
  (quad 40%, logistic 31%, expo 19%, Gompertz 10% wins, 35.3% fails).
- COVID: families largely fail multi-wave; protocol distinguishes domains.

## Withheld-scope finding

`audit_novelty` merges custom_corpus INTO the standard corpus (197 entries, v4) — entries
cannot be subtracted; pinned by test, not fixed (API semantics).

## Trading & Policy Grammar (see docs/APEDP_EVOLVED_EQUATIONS_AND_ABLATION_RECORD.md)

- `APEDP_EVOLVED_UNIVERSE_SPECIFIC_EQUATIONS_BLOCK_B_ABLATION_EXPLORATORY`:
  - Status: `EXPLORATORY — NOT PROMOTED — POINT-RETURN PREDICTION NOT SUPPORTED — REPRODUCTION ARTIFACTS PENDING VERIFICATION — PROSPECTIVE VALIDATION REQUIRED`
  - Block A (2021-2023, 753 bars) discovery across N in {15, 30, 60, 90} on 27-signal grammar.
  - H-BB selected 10/20 runs across 5 seeds (100% positive sign, mean w=+0.84 on N=15/30, +0.44 on N=60/90).
  - Low pairwise Jaccard across seeds (0.23-0.30) confirms non-unique equation landscape.
  - Block B (2024-2025, 502 bars) ablation: Standalone H-BB yields +30.08% (N=15), +40.80% (N=30), +8.89% (N=60), -0.84% (N=90).
  - Full equations: +33.77% (N=15), +33.64% (N=30), +10.39% (N=60), +6.68% (N=90).
  - Removing H-BB drops net return by -25.57pp to -51.44pp in N=15,30,60 (confirms H-BB as directional engine).
  - Universe additions compress Max DD: -11.6pp (N=30), -14.2pp (N=60), -4.5pp (N=90).
  - Predictive Fit Reconciliation (`provenance/r2_rmse_scale_conventions_20261009/`):
    * Convention 1 (As-Is): RMSE 0.88-1.77, R^2 -26,000% to -125,000% (semantic/scale mismatch).
    * Convention 2 (Train-Calibrated OLS slope): Holdout RMSE 0.042-0.054 (indistinguishable from null sigma), R^2 -0.02% to -0.37% (fails pointwise mean forecasting).
  - Sealed records: `provenance/selection_stability_and_ablation_20261009/` & `provenance/r2_rmse_scale_conventions_20261009/`.
- `APEDP_CONTROLLED_AB_BENCHMARK_PARSIMONY_ABLATION_EVALUATION`:
  - Status: `EMPIRICALLY EVALUATED — VALIDATION-BASED BACKWARD ELIMINATION REJECTED AS DEFAULT ENGINE UPGRADE`
  - Provenance: [`provenance/controlled_ab_benchmark_20261009/`](file:///home/psalmprax/ALL_PROJECTS/science/provenance/controlled_ab_benchmark_20261009/) (`manifest.json`, `returncode: 0`, `commit: d881d5cd`).
  - Protocol & Boundaries: Sub-Train `[0:500]` (500 bars, 2021-01-04 to 2022-12-27); Validation `[500:753]` (253 bars, 2022-12-28 to 2023-12-29); Protected Test `[753:1255]` (502 bars, 2024-01-02 to 2025-12-31). Zero temporal slice intersection. Reused historical holdout, not untouched prospective data.
  - Verdict: Validation-based backward elimination rejected as the default discovery-engine upgrade. In the reported controlled comparison, pruning reduced model complexity substantially (mean terms 5.4 -> 2.5, diff -2.85 +/- 0.34, p=1.3e-7) but worsened mean protected-period net return (15.88% -> 10.98%, paired diff -4.91 +/- 6.83 pp, 95% CI [-19.2, +9.4] pp, p=0.49), Sharpe (0.36 -> 0.28, paired diff -0.08 +/- 0.12, p=0.54) and maximum drawdown (25.9% -> 32.0%) across four universes. Underperformed in 3 of 4 universes (N=15, 30, 90; improved in N=60). Mean position correlation across seeds dropped (Arm A +0.619 vs. Arm B +0.298). The result supports rejecting this specific pruning protocol as a default, while leaving open the possibility that alternative parsimony methods may help. The proposed explanation that removed terms provide regime-dependent risk control remains an unconfirmed hypothesis.
- `APEDP_15_ASSET_INDIVIDUAL_AUDIT_BLOCK_B`:
  - Status: `AUDITED — HISTORICAL RETROSPECTIVE — REALIZED RETURN SPREAD`
  - Reference: [`docs/APEDP_EVOLVED_EQUATIONS_AND_ABLATION_RECORD.md:Section 10`](file:///home/psalmprax/ALL_PROJECTS/science/docs/APEDP_EVOLVED_EQUATIONS_AND_ABLATION_RECORD.md#L350-L420)
  - Data Sampling: Evaluated across Block B (502 daily bars, 2024-01-02 to 2025-12-31, 5.0 bps flat one-way turnover fee). 502 bars covers two calendar years due to cross-asset synchronization with the US equity trading calendar (weekdays only, excluding NYSE holidays; Saturday/Sunday = 0).
  - Machine-Verified Master Metrics: Universe mean B&H = +7.01%, Strategy Net = +29.33%, Realized Return Spread = +22.31 pp. Median B&H = -19.49%, Strategy Net = +20.38%, Realized Spread = +60.83 pp.
  - Segment Breakdown (Corrected Arithmetic):
    * Top Altcoins (ATOM, AVAX, DOGE, UNI, MATIC): Mean B&H = -41.07%, Strat Net = +110.24%, Realized Spread = +151.31 pp (short-trend expansion capture).
    * Secular Trend Leaders (BTC, BNB, XRP): Mean B&H = +155.60%, Strat Net = -33.46%, Realized Spread = -189.06 pp (severe counter-trend shorting drag).
    * Low-Beta Chop (LTC): Mean B&H = +5.32%, Strat Net = -87.45%, Realized Spread = -92.77 pp (whipsaw).
  - Qualifications: Realized cash spread over benchmark, not factor-neutral alpha. Max drawdown is path-dependent and cannot be derived by subtraction. Reused historical holdout.
- `APEDP_EXPLORATORY_MULTITIMEFRAME_DISCOVERY_PROVISIONAL`:
  - Status: `EXPLORATORY — PROVISIONAL FINDINGS — NOT PROMOTED`
  - Reference: [`docs/APEDP_EVOLVED_EQUATIONS_AND_ABLATION_RECORD.md:Section 11`](file:///home/psalmprax/ALL_PROJECTS/science/docs/APEDP_EVOLVED_EQUATIONS_AND_ABLATION_RECORD.md#L422-L460) | [`scratch/multitimeframe_evolution_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/multitimeframe_evolution_results.json)
  - Experimental Setup: Genetic search across 5 crypto assets (BTC, ETH, SOL, BNB, DOGE) across candle intervals {1m, 5m, 15m, 30m, 45m, 60m, 1d} on 27-signal grammar (seed=42, pop=60, gen=25, 5 bps fee, live Binance klines).
  - In-Sample Fit Sharpe: 1m (-1.61), 5m (-0.04), 15m (+0.31), 30m (+0.70), 45m (+2.44), 60m (+1.21), 1d (+0.95).
  - Findings: Generates hypothesis that signal composition varies across timescales (volatility squeeze dominant on 30m–60m; macro trend dominant on 1d).
  - Critical Qualifications: In-sample search fitness, not validated out-of-sample edge. Fixed bar count (1,000 bars) confounds timeframe with calendar duration (16.7h on 1m vs. 41.7d on 1h vs. 2.9y on 1d). Single-seed exploratory finding requiring multi-seed forward validation under realistic microstructure execution.
- `APEDP_TRADING_GRAMMAR_SPACE_COMPREHENSIVE_EXPANSION`:
  - Status: `VERIFIED & INTEGRATED — 5-LAYER HIERARCHICAL TRADING GRAMMAR ENGINE`
  - Modules: [`apedp_dyn/grammar/trading_grammar_catalog.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/trading_grammar_catalog.py), [`apedp_dyn/sandbox/grammar_space.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/grammar_space.py), [`apedp_dyn/grammar/__init__.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/__init__.py).
  - Specification: [`docs/APEDP_TRADING_GRAMMAR_SPACE_SPECIFICATION.md`](file:///home/psalmprax/ALL_PROJECTS/science/docs/APEDP_TRADING_GRAMMAR_SPACE_SPECIFICATION.md).
  - Test Suite: [`tests/test_dyn_trading_grammar_catalog.py`](file:///home/psalmprax/ALL_PROJECTS/science/tests/test_dyn_trading_grammar_catalog.py) (38/38 passing across all grammar modules).
  - Scope & Structural Additions:
    * Implemented 5-layer hierarchy: Layer 0 (Raw Observations), Layer 1 (Primitive Transforms), Layer 2 (Technical & Statistical Families), Layer 3 (Candidate Equations), Layer 4 (Trading Policy & Risk Control).
    * Integrated comprehensive mathematical operator library: domain-safe arithmetic (`safe_div`, `safe_log`, `safe_sqrt`, `exp_clamp`), causal rolling estimators (`roll_slope`, `roll_skew`, `roll_kurtosis`, `ewma`, `roll_beta`), temporal operators (`op_lag`, `op_diff`, `op_pct_change`, `op_acceleration`, `consecutive_up_bars`, `time_since_last_high`), cross-sectional normalizers (`cross_sectional_rank`, `cross_sectional_zscore`, `robust_zscore`, `winsorize`), and logical/gating operators (`deadband`, `signal_gate`, `regime_gate`, `cooldown`, `hysteresis`).
    * Implemented Section 11 Target & Outcome Grammar (`FWD_RET_K`, `FWD_DIRECTION`, `FWD_CS_RANK`, `FWD_REALIZED_VOL`, `FWD_TRIPLE_BARRIER`) strictly segregated from causal predictive features.
    * Implemented Section 14 machine-readable `FeatureSpec` schema with dataclasses, query filters, JSON serialization, and cryptographic SHA-256 catalog fingerprinting.
    * Implemented Section 13 first-stage feature extraction pipeline for 30m–60m crypto trading.
- `APEDP_SCIENTIFIC_GRAMMAR_SPACE_COMPREHENSIVE_EXPANSION`:
  - Status: `VERIFIED & INTEGRATED — PHYSICS, CHEMISTRY, BIOLOGY DISCOVERY GRAMMAR`
  - Modules: [`apedp_dyn/grammar/scientific_grammar_catalog.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/scientific_grammar_catalog.py), [`apedp_dyn/grammar/__init__.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/__init__.py).
  - Specification: [`docs/APEDP_SCIENTIFIC_GRAMMAR_SPACE_SPECIFICATION.md`](file:///home/psalmprax/ALL_PROJECTS/science/docs/APEDP_SCIENTIFIC_GRAMMAR_SPACE_SPECIFICATION.md).
  - Test Suite: [`tests/test_dyn_scientific_grammar_catalog.py`](file:///home/psalmprax/ALL_PROJECTS/science/tests/test_dyn_scientific_grammar_catalog.py) (44/44 passing across all suites).
  - Scope & Domain Implementation:
    * Physics (`APEDP-PHYSICS`): Mechanics, Thermodynamics, Electromagnetism, Waves & Oscillations, Fluids & Transport, Materials, Quantum & Astrophysics. Integrated 7-component SI dimensional basis vector $(M, L, T, \Theta, N, I, J)$ and `PhysicsValidityGate` (dimensional homogeneity, mechanical energy conservation).
    * Chemistry (`APEDP-CHEMISTRY`): Composition & Concentrations, Reaction Kinetics, Reaction Rates, Chemical Thermodynamics, Electrochemistry, Catalysis. Integrated `ChemistryValidityGate` (concentration non-negativity, stoichiometric mass balance $\dot{c} = S r$).
    * Biology (`APEDP-BIOLOGY`): Molecular & Gene Regulation, Enzyme Kinetics, Cell Biology, Physiology, Epidemiology, Population Ecology, Neuroscience. Integrated `BiologyValidityGate` (mandatory context tags, population bounds $0 \le N \le 2K$).
    * Shared Scientific Operator Library: `op_michaelis_menten`, `op_hill_activation`, `op_hill_repression`, `op_temporal_derivative`, `op_temporal_integral`, `op_delay`, `op_threshold_activation`, `op_lotka_volterra_step`.
    * Machine-readable `ScientificVariableSpec` schema with JSON export and deterministic cryptographic SHA-256 catalog fingerprinting.
- `APEDP_EXPANDED_GRAMMAR_EVOLUTION_UNIVERSES_AND_TIMEFRAMES`:
  - Status: `AUDITED & RECTIFIED — EXPLORATORY DISCOVERY AUDIT — PROVISIONAL FINDINGS`
  - Modules: [`apedp_dyn/sandbox/multi_asset_signals_sandbox.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/multi_asset_signals_sandbox.py) (`calc_expanded_signals_for_asset`, `EXPANDED_SIGNAL_NAMES`, $K=46$).
  - Results Files: [`scratch/expanded_universe_evolution_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/expanded_universe_evolution_results.json), [`scratch/expanded_multitimeframe_evolution_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/expanded_multitimeframe_evolution_results.json).
  - Documentation: [`docs/APEDP_EVOLVED_EQUATIONS_AND_ABLATION_RECORD.md:Section 12`](file:///home/psalmprax/ALL_PROJECTS/science/docs/APEDP_EVOLVED_EQUATIONS_AND_ABLATION_RECORD.md#L432-L470).
  - Audit & Defect Rectification:
    * Defect 1 resolved: `BREAKOUT_STRENGTH` previously returned identical zeros ($std=0.0$) due to $p \le h \le \max_{20}(h)$, causing a dead-signal zero-trade artifact (-0.02 fit) that falsely masqueraded as a "Donchian fader" across 1m, 5m, and 15m. Patched to signed causal Donchian excursion relative to prior trailing window ($std=1.13$).
    * Defect 2 resolved: `run_multi_asset_genetic_algorithm` omitted `in_sample_sharpe` propagation across generations, zeroing out Sharpe ratios for elite survivors. Patched to propagate `in_sample_sharpe` and re-evaluate `population[0]`.
  - Machine-Verified Daily Universe Findings ($N=15, 30, 60, 90$):
    * N=15: `H-SQUEEZE`, `NL_RISK_ADJ_MOM`, `NL_HURST_ADAPT`, `BODY_TO_RANGE`, `EMA_SPREAD` (Fit: 4.39, In-Sample SR: +3.08, Trades: 1,453, Net Ret: +137.4%).
    * N=30: `H-RV`, `BB_WIDTH` (Fit: 2.49, In-Sample SR: +2.08 [corrected from 0.00], Trades: 4,301, Net Ret: +41.7%).
    * N=60: `H-RSI`, `H-OBV_MOM`, `ADX`, `KELTNER_WIDTH`, `SIGNED_VOL_MOM` (Fit: 2.67, In-Sample SR: +2.28 [corrected from 0.00], Trades: 3,496, Net Ret: +45.7%).
    * N=90: `H-CROSS_MOM`, `NL_SQUEEZE_TREND`, `BB_WIDTH` (Fit: 2.49, In-Sample SR: +2.11, Trades: 4,540, Net Ret: +41.1%).
  - Machine-Verified Intraday Discoveries ($1\text{m}$ to $60\text{m}$, 5 Crypto Assets):
    * 1m: $S_{\text{1m}} = -0.6979 X^*_{\text{H-MACD}} + 0.8974 X^*_{\text{H-EMA\_CROSS}} + 0.0376 X^*_{\text{NL\_RISK\_ADJ\_MOM}}$ (Fit: -1.57, SR: -1.49, Trades: 280, Ret: -1.82%). Friction dominated.
    * 5m: $S_{\text{5m}} = +0.6191 X^*_{\text{H-SQUEEZE}} + 0.0350 X^*_{\text{TREND\_EFFICIENCY}} + 0.2490 X^*_{\text{PARKINSON\_VOL}}$ (Fit: -0.11, SR: -0.11, Trades: 429, Ret: -0.31%). Friction dominated.
    * 15m: $S_{\text{15m}} = -0.6040 X^*_{\text{EMA\_SPREAD}} - 0.2993 X^*_{\text{PARKINSON\_VOL}}$ (Fit: +0.27, SR: +0.27, Trades: 318, Ret: +0.99%).
    * 30m: $S_{\text{30m}} = -0.4609 X^*_{\text{EMA\_SPREAD}} - 0.1417 X^*_{\text{BB\_WIDTH}} - 0.1648 X^*_{\text{CMF}}$ (Fit: +0.22, SR: +0.23, Trades: 553, Ret: +1.60%).
    * 45m: $S_{\text{45m}} = -0.6347 X^*_{\text{BREAKOUT\_STRENGTH}}$ (Fit: +1.13, SR: +1.07, Trades: 226, Ret: +3.23%).
    * 60m: $S_{\text{60m}} = +0.2324 X^*_{\text{H-EMA\_CROSS}} - 0.6140 X^*_{\text{EMA\_SPREAD}}$ (Fit: +1.14, SR: +1.04, Trades: 283, Ret: +8.19%).
- `APEDP_CALENDAR_ALIGNED_INTRADAY_BENCHMARK_AUDIT`:
  - Status: `AUDITED & REJECTED FOR PROMOTION — ZERO OUT-OF-SAMPLE ALPHA DETECTED`
  - Modules & Scripts: [`scratch/audit_intraday_causality_and_calendar_alignment.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/audit_intraday_causality_and_calendar_alignment.py), [`scratch/calendar_aligned_multitimeframe_benchmark.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/calendar_aligned_multitimeframe_benchmark.json).
  - Documentation: [`docs/APEDP_EVOLVED_EQUATIONS_AND_ABLATION_RECORD.md:Section 12.4`](file:///home/psalmprax/ALL_PROJECTS/science/docs/APEDP_EVOLVED_EQUATIONS_AND_ABLATION_RECORD.md#L479-L525).
  - Scope:
    * Verified temporal causality invariance across all 46 features under future perturbation tests ($0.00\times 10^0$ lookahead).
    * Explained and resolved 45m bar-count discrepancy (Binance 1,000 candle limit $\div 3 = 333$ bars = 10.4 days).
    * Evaluated 10.0-day calendar-aligned window (240.0 hours) across BTC, ETH, SOL, BNB, DOGE under 70% train / 30% out-of-sample forward split across 3 seeds (42, 101, 777) with 5.0 bps turnover fees.
  - Machine-Verified Out-of-Sample Verdict:
    * In-sample apparent Sharpe ratios (+1.08 to +4.72 on 15m–60m) collapsed completely out-of-sample.
    * Every single timeframe produced negative OOS Sharpe (-65.92 on 1m, -25.16 on 5m, -19.53 on 15m, -6.12 on 30m, -12.24 on 45m, -22.04 on 60m) and negative net returns (-12.71% to -0.79%).
    * Directional interpretation confirmed: 45m negative weight on `BREAKOUT_STRENGTH` acts as a mean-reverting counter-trend fader, not continuation.
- `APEDP_ROLLING_WINDOW_EQUATION_STABILITY_AUDIT`:
  - Status: `DIAGNOSTIC EVIDENCE RECORDED — STRUCTURAL IDENTIFIABILITY UNRESOLVED — PROTOCOL FROZEN — BLOCK B RETIRED`
  - Modules & Scripts: [`scratch/audit_rolling_window_stability.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/audit_rolling_window_stability.py), [`scratch/run_stage5_four_model_scoreboard.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/run_stage5_four_model_scoreboard.py), [`scratch/run_corrected_accounting_audit.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/run_corrected_accounting_audit.py).
  - Dedicated Artifacts: [`scratch/rolling_window_stability_audit_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/rolling_window_stability_audit_results.json), [`scratch/stage5_four_model_four_universe_scoreboard.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/stage5_four_model_four_universe_scoreboard.json), [`scratch/corrected_accounting_stage5_scoreboard.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/corrected_accounting_stage5_scoreboard.json).
  - Documentation: [`docs/APEDP_EVOLVED_EQUATIONS_AND_ABLATION_RECORD.md:Section 13`](file:///home/psalmprax/ALL_PROJECTS/science/docs/APEDP_EVOLVED_EQUATIONS_AND_ABLATION_RECORD.md#L527-L635).
  - Forensic Accounting Corrections Applied:
    * Weight-based turnover: $\text{Turnover}_t = \sum_i |w_{i, t} - w_{i, t-1}|$ directly on normalized weights ($\sum |w_i| = 1.0$). Fee cost = $0.0005 \times \text{Turnover}_t$.
    * Multiplicative wealth compounding: $W_t = W_{t-1} \cdot (1 + r_{\text{net}, t})$. Standard peak-to-trough max drawdown $(W_t - \text{Peak}_t)/\text{Peak}_t$.
    * Annualization: $\sqrt{252}$ trading sessions ($\sqrt{365}$ reported side-by-side, ratio $\approx 0.831$).
    * Calendar: Exactly 502 bars verified (252 in 2024, 250 in 2025; 11 official NYSE closures in 2025).
  - Reconciled Forensic Scoreboard on Block B (502 bars, fee 5.0 bps):
    * Buy & Hold: Net $-2.36\%$ / Max DD $54.2\%$ ($N=15$), Net $+3.76\%$ / Max DD $55.4\%$ ($N=30$), Net $-3.27\%$ / Max DD $52.3\%$ ($N=60$), Net $+13.56\%$ / Max DD $41.5\%$ ($N=90$).
    * Model 1 (Standalone H-BB): Net $+42.38\%$ / SR $1.01$ ($N=15$), Net $+16.62\%$ / SR $0.55$ ($N=30$), Net $-1.40\%$ / SR $0.02$ ($N=60$), Net $-9.40\%$ / SR $-0.47$ ($N=90$).
    * Model 3 (Restrained Core): Net $+34.85\%$ / SR $0.91$ ($N=15$), Net $+20.37\%$ / SR $0.66$ ($N=30$), Net $+0.08\%$ / SR $0.08$ ($N=60$), Net $-5.61\%$ / SR $-0.27$ ($N=90$). Compresses Max DD to $13.7\%$–$19.2\%$.
    * Model 4b (Canonical Block A Frozen OOS, 0% overlap): Net $+26.31\%$ / SR $0.83$ ($N=15$), Net $+20.84\%$ / SR $0.77$ ($N=30$), Net $+8.36\%$ / SR $0.45$ ($N=60$), Net $+3.88\%$ / SR $0.28$ ($N=90$).
    * Model 2b (Strict Walk-Forward OOS, 0% overlap): Net $-9.23\%$ / SR $-0.15$ ($N=15$), Net $+34.89\%$ / SR $1.13$ ($N=30$), Net $+16.42\%$ / SR $0.69$ ($N=60$), Net $-13.09\%$ / SR $-0.87$ ($N=90$).
    * Model 4a (Unconstrained Win-5): Discarded as in-sample retrospective fit due to 100% Block B training overlap.
  - Epistemic Decision & Protocol Freeze:
    * Model 4a formally rejected as an out-of-sample alpha claim due to training overlap.
    * Models 2b (walk-forward) and 4b (frozen Block A) preserved as research baselines, not production trading rules.
    * Section 13.6 locked in: (1) 72-hour crypto weekend returns explicitly documented as accumulated into Monday bars; (2) 5 bps fee locked as lower-bound transaction cost, with spread, slippage, and funding/borrow costs recognized as omitted; (3) causal execution timeline verified (bar t close -> signals -> weight -> return on (t, t+1]); (4) survivorship bias in 2026-constructed asset universes documented.
    * Block B (2024–2025) is permanently retired as an unexamined holdout.
    * Promotion to production trading remains strictly rejected. Any future candidate promotion requires pre-registered evaluation on untouched prospective data (e.g. 2026+).
- `APEDP_PROSPECTIVE_2026_HOLDOUT_EVALUATION`:
  - Status: `EVALUATED ON LIVE UNSEEN 2026 DATA — VERDICT: INSUFFICIENT EVIDENCE — PRODUCTION PROMOTION REJECTED`
  - Modules & Scripts: [`scratch/run_prospective_2026_test.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/run_prospective_2026_test.py), [`scratch/standalone_model4b_test_suite.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/standalone_model4b_test_suite.py), [`scratch/prospective_2026_model4b_test_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/prospective_2026_model4b_test_results.json).
  - Scope:
    * Downloaded 435 daily bars per asset via Binance public API (153-day warmup [2025-08-01 to 2025-12-31] + 282-day prospective evaluation [2026-01-01 to 2026-10-09]) for 15 crypto assets (with POLUSDT accounting for Polygon's migration).
    * Evaluated frozen Model 4b ($N=15$) with frozen weights ($+1.1948 X_9^* + 0.2227 X_{11}^* + 0.4501 X_{16}^* - 0.4839 X_{23}^*$), threshold $\theta = 0.1695$, and flat 5.0 bps turnover fee.
  - Results on 2026 Holdout:
    * Benchmark 15-Crypto Buy & Hold: Cumulative return $-2.58\%$, Sharpe (252) $+0.18$, Max Drawdown $48.1\%$.
    * Model 4b ($N=15$): Cumulative Net Return $+6.51\%$, Gross Return $+14.58\%$, Friction drag $8.07\,\text{pp}$, Sharpe (252) $+0.43$, Sharpe (365) $+0.52$, Max Drawdown $14.0\%$, Mean daily weight turnover $0.5195$.
  - Pre-Registered Epistemic Verdict:
    * `INSUFFICIENT EVIDENCE`: Beat Buy & Hold and limited max drawdown to $14.0\% \le 25.0\%$, but Sharpe ratio ($+0.43$) failed the pre-registered threshold ($\ge +0.70$).
    * Production promotion remains strictly refused.
- `APEDP_MULTI_DOMAIN_GRAMMAR_EXPANSION_AND_AUDIT`:
  - Status: `COMPLETED — 100% REGRESSION TESTS PASSING (48/48 TESTS)`
  - Modules & Tests:
    * [`apedp_dyn/grammar/trading_grammar_catalog.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/trading_grammar_catalog.py) (expanded from 96 to 141 features).
    * [`apedp_dyn/grammar/scientific_grammar_catalog.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/scientific_grammar_catalog.py) (expanded from 130 to 190 variables; added `ASTROPHYSICS` and `MATHEMATICS` domains).
    * [`tests/test_dyn_trading_grammar_catalog.py`](file:///home/psalmprax/ALL_PROJECTS/science/tests/test_dyn_trading_grammar_catalog.py), [`tests/test_dyn_scientific_grammar_catalog.py`](file:///home/psalmprax/ALL_PROJECTS/science/tests/test_dyn_scientific_grammar_catalog.py).
  - Scope & Reconciliation:
    * Forensic Reconciliation 1: Confirmed mechanical grammar canonical typed vocabulary has exactly 19 terms in `TERM_FAMILIES` ($6+3+1+3+1+5$), resolving the 24-builder alias divergence.
    * Forensic Reconciliation 2: Confirmed 1,504 scalar equations across 563 dynamical systems in [`apedp_dyn/problems/catalog.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/problems/catalog.py) via programmatic summation of `len(m.equation_strings)` (equivalent to `m.n_states`): Classical (318 eqs / 108 systems), Nonlinear (240 eqs / 80 systems), Kinetics (224 eqs / 89 systems), Circuits (210 eqs / 84 systems), Biology (199 eqs / 60 systems), Feynman (176 eqs / 88 systems), Astrophysics (137 eqs / 54 systems). Authoritative convention: exact per-category breakdown is `137 / 199 / 210 / 318 / 176 / 224 / 240 = 1,504` across 563 systems (the earlier draft estimate of `142 / 168 / 214 / 282 / 216 / 242 / 240` reflected an auxiliary coupled-template grouping; the programmatic state-derivative registry is the binding ground truth).
    * Trading Space Expansion: Registered all 26 canonical empirical sandbox signals ($X_0$ through $X_{25}$) and interaction terms ($X_{26}$ `PRODUCT_H_BB_TAKER`), plus microstructure and risk variables (`KYLE_LAMBDA`, `AMIHUD_ILLIQUIDITY`, `ROLL_SPREAD`, `OU_MEAN_REVERSION`, `SPREAD_HALF_LIFE`, `COINTEGRATION_RESIDUAL`, `KALMAN_HEDGE_RATIO`, `ROLLING_SKEWNESS`, `ROLLING_KURTOSIS`, `REALIZED_SEMIVARIANCE`, `VALUE_AT_RISK`, `EXPECTED_SHORTFALL`, `VWAP_DISPLACEMENT`, `TRADE_HOLDING_TIME`, `DYNAMIC_TRAILING_STOP`, `OPTIMAL_KELLY_FRACTION`, `LEVERAGE_RATIO`, `MARGIN_UTILIZATION_PCT`).
    * Scientific Space Expansion: Added `ScientificDomain.ASTROPHYSICS` (8 variables) and `ScientificDomain.MATHEMATICS` (5 variables). Populated missing physical and biological states: SEIR compartments (`SUSCEPTIBLE_POPULATION`, `EXPOSED_POPULATION`, `INFECTIOUS_POPULATION`, `RECOVERED_POPULATION`, `BASIC_REPRODUCTION_NUMBER`), Hill coefficient ($n$), ATP energy charge, circuits states (`INDUCTOR_CURRENT`, `IMPEDANCE`, `MEMRISTANCE`), fluid dynamics (`VORTICITY`, `STREAM_FUNCTION`, `MACH_NUMBER`), quantum mechanics (`PLANCK_CONSTANT`, `PROBABILITY_DENSITY`, `SPIN`, `HAMILTONIAN`), and chemical kinetics/solution thermodynamics (`ARRHENIUS_PRE_EXPONENTIAL`, `HALF_LIFE`, `ACID_DISSOCIATION_CONSTANT`, `NERNST_POTENTIAL`).










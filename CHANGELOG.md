# Changelog: APEDP & APEDP-X Discovery Framework

All notable changes, bug fixes, audit remediations, and architectural milestones are documented in this file.

---

## [Unreleased] — RIME Repair v0.1.x freeze (comparative-validation milestone)

### Added (v0.1.9 round — relative-gain criterion, rational-power family, identifiability band)
- **Relative SSE-reduction criterion (`RepairConfig.min_sse_reduction`, default 0 = frozen):** G = (SSE_base − SSE_cand)/SSE_base on the same validation rows, reported alongside absolute R² headroom on every gate. Preregistered companion for high-baseline problems where an absolute R² bar fails by construction (measured on AME2020 below). Caller chooses; neither criterion is inferred.
- **Domain-informed rational-power term family (`RepairConfig.rational_powers`, default empty = frozen):** sign-safe x^p terms of declared dependencies appended to the R2 catalogue, enabled ONLY through the explicit config field — Layer-C priors enter declared, never as hidden corrections. Scored by the same gates.
- **Identifiability band (`RepairConfig.equivalence_r2_tol`, default 0 = frozen single-winner behaviour):** passing candidates within the band of the best are observationally indistinguishable, and the run resolves to `AMBIGUOUS` (MULTI_HYPOTHESIS territory) instead of crowning the marginally-highest scorer. Sealed split is never opened on AMBIGUOUS.
- **Decisive real-data measurement (`tests/test_dyn_real_data_ame_repair.py`, 6 pins):** with the physics-motivated family admitted, the AME2020 ranking on validation is proxy (A·Z) 0.99960 > (A·A) 0.99930 > **surface term (A^0.667) 0.99915** > (A^0.333) 0.99898 > (Z·Z) 0.99842. On the sealed split the proxy and the mechanism differ by <1e-3 R². With the band declared the engine returns `AMBIGUOUS` (set size 4) rather than a unique winner — predictive success is reported separately from identification, which is the SEMF's job, not the search's. SEMF literature coefficients still reach R² 0.9999 on the same frame.
- Control-design note recorded: a candidate cap of 10 was measured to be exactly consumed by R1 + the 9 interaction probes, silently excluding the power terms; the pinned run uses cap 14 (caught by an assertion that failed for the right reason).


### Added (v0.1.8 round — REAL-data repair benchmark, AME2020 masses, no planting)
- **`tests/test_dyn_real_data_ame_repair.py` (3 pins):** answers the standing methodological criticism that every demonstrated repair ran on planted synthetic worlds. Runs discovery → diagnosis → bounded repair on genuinely measured data (AME2020 atomic-mass evaluation, `data/mass_1.mas20.txt`, parsed to binding energies; 1159 usable nuclides).
  - Discovery recovers the semi-empirical mass formula **volume term**: `+7426.14*A + 93729.3` at dev R² 0.994.
  - Diagnosis on that original fires `curvature_suspect` (the missing surface term A^(2/3)).
  - **Threshold calibration exposed by real data:** at the preregistered absolute bar (0.02) the engine returns `NO_IMPROVEMENT` — the bar is mis-scaled for a real dataset whose original already explains 99.4% (achievable gain ≈ 0.005). At a scale-appropriate bar (≤0.005) it returns `IMPROVED` with the `(A*Z)` term at sealed R² **0.9995** vs original 0.9947 — the first sealed-validated repair on measured data in this repository. Both outcomes pinned; the frozen default is untouched (a relative-gain criterion is proposed as a preregistered follow-up, not applied unilaterally).
  - Literature reference on the same frame: SEMF with standard coefficients reaches R² 0.9999 (units keV), so the data support the known law — the boundary is grammar reachability (A^(2/3) absent from the transform set) and threshold calibration, not data quality.
  - Honesty note pinned in the test docstring: the added `(A*Z)` is a *correlated proxy* for the Coulomb/surface structure, not the SEMF term — predictive gain without unique identification (MULTI_HYPOTHESIS territory, deferred to the APEDP-X track); nothing is promoted.
  - Control-design bug caught during the round: reusing one `RepairPartitions` across a threshold sweep raised `HoldoutContaminationError` on the second run — the guard working as designed; the sweep now builds fresh partitions per run.


### Added (v0.1.7 round — intraday crypto sweep on fetched minute bars, audit-only)
- **`tools/fetch_intraday_crypto.py` + `data/market/intraday/` (new, clearly-labeled corpus):** Binance public klines, 1000 bars per symbol per interval at 1m/5m/15m/30m/1h, over the crypto legs of the repo's OWN universes (15/30/45 lists from `multi_asset_signals_sandbox.py`; 14/26/39 fetchable — MATIC delisted, MKR/EOS/XMR/FTM/ICX not currently listed). Own PROVENANCE.json; the frozen daily corpus is untouched.
- **Measured (`tests/test_dyn_intraday_universe_sweep.py`, 5 pins):** real basket forward returns abstain in **15/15** universe×interval cells under both seed orders (no grammar-expressible edge at any tested resolution); null worlds 0 proposals; planted lag-1 control recovered at dev R²>0.98 at every interval (pipeline verified on minute bars); per-position coverage on the 39-input frame follows the 60-form arithmetic exactly — `coverage_first` recovers inputs 0–20 (39 levels + 21 lag-1) while 21+ remain generation gaps.
- **Scope limits recorded, not papered over:** Binance has no 45m interval (requested 45-minute frame does not exist as an exchange bar); equity legs unfetchable (Yahoo Finance intraday HTTP 429 at fetch time) so 60/90-asset intraday universes are out of scope; equal bar counts confound interval with calendar span (1000 1m ≈ 17h vs 1000 1h ≈ 42d).
- **Control-design bug caught and fixed during the round (recorded for audit):** the first planted-position control propagated a NaN std into the planted target (one symbol carries an extra timestamp → NaN row), so the finite-row guard abstained — misread as a RIME limitation until isolated by a solo-frame test. Frame construction now `dropna()`s the pivoted bars first.


### Fixed (v0.1.6 round — generation gap closed, opt-in `seed_order="coverage_first"`)
- **Measured gap (v0.1.6 instrumentation):** the legacy input-major family-2 (delayed levels) truncates the 60-form seed catalogue at high input counts and dropped whole inputs' shortest lag — 3/22 positions (H-RV, H-A, H-OBV_MOM) could never be recovered on the 22-signal frame while a solo-frame control recovered the identical dependence at R² 0.96–0.9995.
- **Fix (`rust/apedp-kernel/src/rime/seeds.rs`):** opt-in lag-major family-2 ordering (`SeedOrder::CoverageFirst`, config key `seed_order`, `RimeProposer(seed_order=...)`, default `"legacy"`). Same budget, same families, same forms whenever the legacy order fits inside the budget (Rust-pinned for n≤6); different WHICH forms survive truncation at many inputs. No threshold, gate, or frozen-spec default changed; `static`/`additive` paths unaffected.
- **Re-measured under the new order:** the 3 gap positions recover the planted column at their solo-frame R² (H-RV 0.9937, H-A 0.9995, H-OBV_MOM 0.9597 — previously 0.19 / 0.68 / 0.09); 19/22 exact + those 3 = **22/22 name the planted column** (the only non-exact recoveries remain the 2 collinear substitutes H-OFI→H-CVD, H-STOCH→H-WILLR, a non-identifiability property, not generation). Null-world false promotions **0/50, unchanged**. Pinned by `tests/test_dyn_rime_seed_order_coverage.py` (3) + 3 new Rust catalogue tests (99/99).


### Added (v0.1.6 round — search-funnel instrumentation + per-position coverage, audit-only)
- **Search-funnel counters (`rust/apedp-kernel/src/rime/search.rs`, `SearchOutcome`):** additive-only `rejected_score` / `dropped_dedup` / `dropped_scope` alongside the existing `evaluated` / `accepted`, plus per-proposal `inputs_used` (`Graph::inputs_used()`, output-reachable arena walk). No score, gate, selection or default changed. Invariants pinned: `rejected_score + accepted == evaluated`, `evaluated <= budget`. Passed through to proposal metadata (`inputs_used_names`, counter keys) in `apedp_dyn/proposers/rime_proposer.py`; `rime_usage_log.py` now prefers the structural list over string mining.
- **Per-position coverage challenge (`tests/test_dyn_rime_coverage_funnel.py`):** planted lag-1 dependence at each of the 22 trading-signal positions (y = 2·x[t-1] + 10 + noise, fixed seeds). **Measured: 17/22 exact recovery; 2/22 collinear substitute** (H-OFI→H-CVD, H-STOCH→H-WILLR — in-grammar twins fitting identically, MULTI rationale); **3/22 generation gap** (H-RV, H-A, H-OBV_MOM — the solo-frame control recovers the identical dependence at R² 0.96–0.9995 while the 22-input frame lands at R² 0.09–0.68, proving the loss is seed-catalogue truncation at 22 inputs, not scoring). Null worlds abstain 2/2. This is the reviewer-directed generation-vs-qualification separation: what is never generated (truncation) vs generated-but-not-selected.
- Fixed during this round: a prefix-collision bug in the kernel's equation-name substitution (`"v13"` was corrupted to `"H-CVD3"` by ascending `String::replace`) — replaced with an exact two-pass control-character-placeholder substitution; rendered equations now match structural `inputs_used_names` verbatim (pinned).


### Added (v0.1.5 round — power calibration + additive refinement, `experimental`, kernel level)
- **`rust/apedp-kernel/tests/rime_power_calibration.rs` (`audit-only`):** recovery / promotion / multi-term rates with Wilson 95% intervals, functional recovery on fresh independent series. Measured: single-term recovery follows $n \gtrsim 21/\rho^2$ (near the ≈$19/\rho^2$ Bonferroni limit for ~200 candidates), null 1/120 promoted; 0% recovery at IC 0.02 below $n=15000$ is the information limit, not a proposer defect.
- **`rust/apedp-kernel/src/rime/additive.rs` (`search_additive`) + kernel config key `additive_terms`:** opt-in multi-term refinement after the frozen gate. Closes a measured representation gap (`x[t-1]+x[t-2]` recovered 0/40 in every cell before; up to 40/40 after for two-term laws, with 0/40 spurious extra terms on single-term laws and identical promotion counts everywhere). Added terms pay edges + 1 coefficient + `2 ln M` (pool size); `Const` coefficients are charged explicitly because `edge_count` treats them as free. Default OFF; `search` is untouched; `seeds::shift` became `pub(crate)`. Not wired into the Python `RimeProposer` (needs pool-size multiplicity in the trial ledger first).
- No frozen spec file, threshold or `canonical_preregistered_spec_v1.json` entry was modified.


### Added (v0.1.4 round, engine tag `rime_v0.1.4` — refusal wiring + repeat bounds)
- **Referee-side out-of-grammar scope refusal (`apedp_dyn/epistemic/scope_refusal.py` + `--adjudicated` in `benchmarks/known_truth_challenge.py`, `experimental`, spec §16d):** closes the §16c negative-control gap (scope refusal must come from adjudication). The referee fits a preregistered family-distinct probe pair (exp family, rate grid [1, 50], dev-normalized; log family, raw axis) from dev data ONLY, and adjudicates each emitted candidate pairwise via the standard `adjudicate()` contract; promotion requires UNIQUE_SUPPORT against EVERY probe. Measured: both methods' exp-world spurious emissions refused (rival-favored bootstrap / indistinguishable rival); all 22 in-grammar true recoveries survive un-declined; `out_of_grammar_emitted_promoted` = 0 everywhere measured. Audit transparency: the first adjudicated run promoted the OLS emission — diagnosed as a referee-side probe defect (axis clipped to [0,1], degrading the exponential probe on extrapolating log grids — the recorded OLS extrapolation failure mode in referee form) and fixed BEFORE any citation, with the strict-AND semantics untouched.
- **Repeat bounds (preregistered, `--repeats`):** 5 observation-noise seeds (phase = 23.0 + 1009·seed; same design points, fresh draws; seed 0 byte-reproduces the pinned v0.1.3 frames), full world set, adjudicated. Artifact `benchmarks/known_truth_results_repeats.json`: recovery 1.0 ± 0.0 per method per seed (11/11 in-grammar every seed), per-world holdout R² std ≤ 5e-5, coefficient accuracy stable (|Ĉ−C|/C ≲ 5e-4 RIME), exp-world emissions refused 5/5 with zero wrongful in-grammar refusals (bounds are noise-driven only). Pinned by `tests/test_dyn_rime_known_truth_scope_repeats.py` (7 pins: referee blindness on all planted exponents, true-monomial UNIQUE_SUPPORT + wrong-exponent refusals, adjudicated single-run refusal without §16c regression, 3-seed repeat bounds, seed-rule frame determinism); `tests/test_dyn_rime_known_truth.py` (§16c) unchanged and green.
- **Population-extrapolation extensions (`tests/test_dyn_population_extrapolation_v2.py`, pinned and re-executed this round):** RIME depths 1–6 byte-identical (`x3[t-2]` on every population), +sq transform axis moves nothing, GA basis ladder isolates products (linear+lag / +squares / +products → only products close the gap), repair-mediated RIME seals 0.9998 with B 0.9998 / C 0.8952 transfer matching GA-large end-to-end — no spec change required.

### Added (v0.1.2 static-grammar round, engine tag `rime_v0.1.2`)
- **Matched-compute harness (`benchmarks/matched_compute.py`, audit-only):** equal-budget comparison of RIME (temporal + static grammars), the existing proposers, the algebraic OLS-basis control, and **PySR 2.7 (head-to-head, installed with Julia 1.11.9)** — scored on a preregistered **grammar-coverage axis**: out-of-coverage abstention is correct behaviour, an emitted-but-failed proposal is a failure on any family, and static worlds require holdout **reversal-invariance** so grid-order tracking is scored `index_fit`. Fairness fixes from the measured audit: 5 repeats with un-stably-verdict screening, full environment metadata, per-cell wall-budget utilization (equal allowance ≠ equal utilized compute — RIME uses ~0.3–0.7% of the allowance, DLI ~8%), PySR runs serial-deterministic with a BINDING 2-s timeout (one-time Julia warmup excluded) and lag columns engineered for information parity (its operator set has no time-shift; its attempted coverage excludes input-side IIR laws). Citable conclusion at equal 2-second serial budgets: **complementarity, not dominance** — rime_temporal 4/5 (fails static_log), PySR 3/5 (fails static_log at this budget, out of scope on the state law).
- **Static search mode (kernel `config["static"]`, default OFF):** temporal primitives (lags, memory cells) are removed from seeds AND rewrites — grid-order fitting becomes unrepresentable for order-structured static data. No threshold or gate changes; the default grammar is byte-identical to v0.1.1.
- **Algebraic input transforms (`RimeProposer(algebraic_transforms=[...])`):** canned `sq/cub/ssqrt/slog` transform columns appended at the input layer, recomputed identically inside `compute_score` so proposals stay causally usable on unseen frames; NaN-preserving, unknown kinds rejected at construction.

### Fixed (v0.1.2 — this is also a CORRECTION of a prior conclusion)
- **Known-truth recovery challenge (`benchmarks/known_truth_challenge.py` + pinned wrapper `tests/test_dyn_rime_known_truth.py`, audit-only):** blind structural-recovery test answering the reviewer-directed open question. 11 in-grammar planted monomial worlds (d ∈ {2,3,6} × grid {log,uniform,irregular} × noise 1×/3× relative-to-signal × coefficient/offset variation) + 1 exp() negative control. **Measured: RIME static grammar recovers the planted structure on 11/11 (max |Ĉ−C|/C ≈ 5e-4, tighter than the BIC-OLS control), 0 false promotions in-grammar; on the out-of-grammar control RIME does NOT abstain (emits a spurious fit) — scope refusal must come from adjudication, not the search.** Harness numerics fixes recorded before citation: dev-slice-relative noise σ, standardized structure regression / OLS design, BIC-selected sparse OLS as the correctly-specified baseline (a full-degree polynomial OLS is documented as failing on extrapolating grids: holdout R² −33). Abstention never counts toward recovery; per preregistration.
- **Kernel phase profile (`rust/apedp-kernel/tests/rime_profile.rs` → `benchmarks/kernel_profile_results.json`, audit-only):** per-candidate decomposition into graph evaluation / readout fit / search bookkeeping (remainder estimate, median of 7). Measured on the planted lag-2 world with the release profile: at n=150 the per-candidate cost is ~19 µs of which ~90% is search bookkeeping (enumeration, dedup, gate arithmetic), while at n=12 000 graph evaluation dominates (~181 µs vs ~54 µs readout fit). Operational consequence for v0.2 planning: residual-guided evolution should first target the enumerative/bookkeeping layer at small n, not the arithmetic.
- **Scale-dependent ridge in `fit_readout` (engine numerics, `rust/apedp-kernel/src/rime/eval.rs`):** the fixed `1e-9` literal added to `Σ(o−ō)²` exceeded the raw second moment of algebraic-scale regressors (a p² column ≈1e-8) by ~30× and silently zeroed every such fit — this, not the grammar, was the primary cause of the measured QEC static-law failure. The ridge is now relative to the regressor's raw second moment (identical for unit-scale data; the QEC attribution "RIME's grammar is the wrong instrument for static scaling laws" is retired: the temporal grammar reaches the law via its self-interaction rewrite, pinned by `test_dyn_ga_qec.py` at holdout R²>0.99 both directions).
- **`tests/rime_regression.rs` was stale against the v0.1.2 typed API** (referenced the removed `SearchPacket` surface and did not compile); rewritten as equivalents against `SearchOutcome` with true null-abstention, positive-control and budget assertions; `rime_stage1.rs` assertions recalibrated to commit 30e73bf's planted amplitude (previously internally inconsistent: even a perfect fit could not clear the control's own bar).

### Added (this round, earlier)
- **RIME Repair diagnosis v0.1.0 (`apedp_dyn/epistemic/rime_repair_diagnosis.py`, experimental):** frozen `ResidualReport` + finding vocabulary on real proposals; all cutoffs explicit caller args.
- **RIME Repair engine v0.1.1–v0.1.4 (`apedp_dyn/epistemic/rime_repair_engine.py`, experimental):** bounded R1/R2/R3 operators, chronological 4-role partitions with single-use sealed split (`HoldoutContaminationError` on reopen), five explicit gates, prediction-equivalence clustering. Promotion structurally deferred to the APEDP-X track (no CandidateProposal→ScoredCandidate bridge fabricated).
- **Sensitivity/recall milestone (v0.1.2–v0.1.3):** recall 9/9 strong planted defects with structure recovery; weak-defect curve never harms; BTC abstention regression; 5/5 walk-forward abstain; adversarial stability-block.

### Fixed
- **RIME `residual.rs` stub:** removed hardcoded threshold literal; residuals now use the fitted readout (consistent with `eval.rs` scoring); threshold is an explicit caller argument with regression tests.

### Model-class closure (experimental engine only — no frozen-spec change)
- Interaction-probe screen (`interaction_suspect`) + lagged square/product R2 terms mirrored 1:1 (probe == term by construction); lag choices cite the frozen LAG_SET's two shortest entries.
- Planted composite (linear part + lagged product of independent drivers) completes through diagnosis→R2→gates→sealed; QEC-conservative refusal preserved (3 refused rows).
- Flaw-closure round: product recall 0/5 budgets (penalty-bound); recall 18/18 with 95% lower bound; single-round termination pinned (multi-round chaining documented as unreachable-by-construction under veto semantics); wall-time resource table; read-only whiteness-guard sweep (operating point 0.3–0.5, default untouched).
- #11 selective risk/coverage (post-hoc refusal policy, no harness changes): GA/OLS attempt signal worlds and abstain noise worlds; QEC thirds split roles (OLS refuses flat fit, repair upgrades to sealed 0.987).
- #7 floor sweep via explicit tau_epi (default untouched): near-tie flips across bracket monotonically; decisive UNIQUE at 0.05+0.1; identical-predictions MULTI at all floors; anchor artifact verified on disk.

### Documented limitations (capability boundary, not a defect)
- Same-driver omitted lags are usually absorbed by OLS into ~white residuals (4/6 sensitivity corners silent); firing corners yield gated rejection. R3's demonstrated domain is emergent lag from an independent driver. Whiteness alone is insufficient for structural identification.
- Whiteness gate assumes stationarity across evaluation rows: on log-spaced QEC data, quadratic extrapolation misfit trips the temporal screen and vetoes an R²=0.95 repair (pinned conservative behavior; refusal is the safe direction).

### Frozen scope
- No further grammar expansion. Next milestone is comparative validation (matched-compute repair vs proposer, independent inspection), not new operators.

## [2.0.0] - 2026-10-12

### Added
- **Full-Lineage Experiment & Candidate Ledger (`FullLineageExperimentLedger`):** Complete lineage tracking recording candidate ancestry, source proposer, feature sets, hyperparameters, random seeds, and specific dataset slice exposures.
- **Anti-Snooping Holdout Guard (`HoldoutContaminationError`):** Permanent holdout slice burning. When an empirical hypothesis fails Stage 4 out-of-sample confirmation, the evaluated slice `(dataset_id, slice_id)` is permanently blacklisted. Any iterative tweak attempting to reuse it triggers `HoldoutContaminationError`.
- **Six-Stage Scoped Promotion State Machine (`PromotionTracker`):** Replaced monolithic boolean promotion flags with six non-fungible sequential stages:
  1. `MATHEMATICAL_CONSISTENCY` (Lean 4 kernel + dimensional invariants)
  2. `PREDICTIVE_RECOVERY` (Cross-validated out-of-sample $R^2 > 0.05$, $p < 0.01$)
  3. `UNIQUE_IDENTIFICATION` (Diebold-Mariano test $p < 0.01$; literature projection $R^2 < 0.85$)
  4. `OUT_OF_SAMPLE_TRANSFER` (Un-snooped temporal regimes and cross-asset transfer retention $\ge 50\%$)
  5. `ECONOMIC_VALIDITY` (Net Sharpe after fees, breakeven fee $\ge 3.0$ bps, $DSR \ge 0.95$)
  6. `REAL_MARKET_PROMOTION` (Live sandbox / paper-trading execution stability)
- **Calibrated Three-Tier Benchmark Suite (`calibrated_benchmark_suite.py`):**
  - Tier 1: Referee discipline measuring FPR ($\le 0.05$), CAR ($\ge 0.70$), and VRR ($\ge 0.75$) with exact Wilson score confidence intervals.
  - Tier 2: Discovery statistical power ($1 - \beta$) across a calibrated grid of effect sizes ($IC \in [0.02, 0.05, 0.15]$), fat-tailed Student-$t$ noise ($\nu=4$), and structural regime switches.
  - Tier 3: Transfer generalization measuring retention decay across distinct out-of-distribution market regimes.
- **Canonical Preregistered Specification (`specifications/canonical_preregistered_spec_v1.json`):** Single authoritative source of truth for all preregistered constants, guarded by SHA-256 hash verification in `apedp_dyn/conf/canonical_spec.py`.
- **Documentation & Configuration Integrity Suite (`tests/test_documentation_and_spec_integrity.py`):** CI tests that fail if runtime code, test suites, or documentation drift from frozen canonical specifications.
- **Discovery Power Calibration Curves (`generate_power_curve_matrix`):** Power calibration evaluating theoretical vs empirical detection thresholds ($\Phi(\sqrt{N} \cdot IC - z_\alpha)$) with Wilson CIs across sample sizes and noise structures.
- **Expanded Proposer Suite:**
  - `DistributedLagInteractionProposer`: Distributed lag kernels ($k \in \{1, 2, 5, 10\}$) and bilinear feature interactions.
  - `RegimeSwitchingProposer`: Piecewise threshold models adaptive to market volatility states.
  - `HawkesOrderFlowProposer`: Self-exciting point processes modeling order arrival clustering and quote replenishment.
- **Regime Robustness Diagnostic (`evaluate_regime_robustness_diagnostic`):** Environment-stratified predictive incrementality evaluating cross-regime forecast survival without rigid residual homoskedasticity vetoes.
- **Decoupled Execution Cost Models (`execution_cost_models.py`):** Separated cost models: `FixedPlusSlippageCostModel` for slower factors, `AlmgrenChrissImpactCostModel` for temporary/permanent market impact, and `LimitOrderQueueCostModel` for queue fills and adverse selection.

### RIME v0.1.2 — Reliability & Evidence Integrity (completing the review specification)

The evidence round was extended with the reliability guarantees the review checklist required (no grammar growth, rules 1–3 unchanged):

- **Frozen environment handling.** `MIN_ENV_FINITE_ROWS = 15` and three explicit `EnvStatus` values (`Scored` / `TooFewRows` / `Malformed`): short environments are excluded from every aggregation, ragged ones fail loudly, all-unusable input abstains with an explicitly infinite reference and zero evaluated candidates, NaN gaps are skip-and-count (there is a gapped-environment recovery test), and per-env statuses ship in the search packet. Regression tests cover heterogeneous 3-env worlds, exclusion, and degenerate abstention.
- **PruneEdge semantics proven, not assumed.** New `rime_prune_semantics.rs`: exactly-zero contribution; complexity/DoF accounting agrees everywhere (zeroing, not removal — stated in the packet); content address changes so pruned ≠ parent; byte-reproducible predictions across serialization; no collateral edits; a fully zeroed graph yields an intercept-only readout that cannot outscore the null.
- **Explicit search limits.** Structural rewrites provably never CREATE a `Sign`/`Abs` node (depth-3 expansion over the whole universe; wrapping an inherited one is composition and is allowed from seeds only, reachability-tested); an off-grid 0.87 decay is never rendered as an exact `state_0.8700` and its in-sample R² is honestly < 1; budget-1 and empty-seed searches abstain explicitly.
- **Benchmark provenance & determinism.** The artifact records benchmark version, git revision (+dirty flag), kernel version, config, seed policy, timestamp, the feature-availability rule, and a rime-determinism self-check; recovery/accuracy/cost/complexity remain separate fields.
- **APEDP-X boundary tests.** Graph identity is checked end-to-end (Rust id = proposal id = packet id, 64-hex), malformed graphs (self-referencing arena, out-of-frozen lag) raise explicitly at the kernel, the referee contract field list is asserted complete, and the no-confirmation-feedback property was verified structurally (no RIME references in epistemic/, no referee imports in RIME).

### RIME v0.1.2 — Evidence Round (multi-environment scoring, stress benchmark, terminology)

**Added.** Addressing the v0.1.1 review's three next steps, without adding structure beyond the frozen primitives.

- **Multi-environment selection (`search_envs`, `rime_search_envs_json`, `RimeProposer(per_frame_envs=True)`).** Environments are scored separately under a rule frozen before scoring and reported in the packet: `MeanPerEnv` (equal weight per environment — the literal §8 reading, fixing the pooled path's length weighting) and `MedianWorstGated` (median across environments plus a **worst-environment improvement gate**: the candidate must beat its parent by more than that environment's own noise scale $\gamma\,\mathrm{var}(y)/n$, with the judged environment selected by the *candidate's* lowest relative improvement — selecting by the parent's MSE would hand the decision to whichever environment has the largest variance, a property of the data, not the mechanism). A dominance-trap regression test at both layers pins the behaviour: one long stream's wins cannot outvote a short stream where the mechanism is absent; the consistent-law control passes the same gate.
- **Broader benchmark (`benchmarks/rime_vs_neural.py --stress`, artifact `stress` key).** Noise sweeps (0.02, 0.5, 1.0), amplitude sweeps (0.5, 3.0), on-grid (0.80) and deliberately off-grid (0.87) memory decays, each with and without a ≈0.9-correlated distractor input — 18 cells. RIME reported the right structure in 18/18 cells; the artifact records the honest details (e.g. `lag a=0.5, noise=1.0` → OOS $R^2$ 0.154; off-grid `state d=0.87` recovered via the nearest grid member — the documented discrete-decay approximation). Not done here, deliberately: PySR and existing-proposer matched-compute runs remain pending their own environment.
- **Terminology:** `compute_score` is documented and tested as **past-information-only** — a temporal (no look-ahead) property of the signal path, explicitly **not** a causal claim; the proposal metadata carries `signal_semantics`, `per_frame_envs` and `aggregation`.

### RIME v0.1.1 — Stage-1 Engine Integration

**Added.** The RIME symbolic-mechanism engine (`rust/apedp-kernel/src/rime/`) is now correct, searchable, and wired into the proposer boundary.
  - `Node::State`: a genuine persistent memory cell, $h_t = d\,h_{t-1} + u_t$ with $h_{-1} = \text{init}$, replacing the `AddState` encoding that was not a recursion and carried the input term with the wrong sign.
  - `rime/eval.rs` fits a ridge least-squares readout $a\,o_t + b$ before scoring (spec §7 required fitting; it was never implemented) and charges the two readout degrees of freedom plus an EBIC-style selection charge $2\ln M$ for the search width (spec §10, "Bonferroni over grammar size").
  - `rime/seeds.rs`: deterministic 60-form seed catalogue (levels, delayed levels, differences, momentum, both lead-lag interaction orientations, memory cells on a discrete decay grid, wrappers). Without it the five frozen rewrites cannot reach difference or momentum forms at all.
  - `rime/search.rs`: deterministic budgeted beam search anchored on the mean predictor, with expression-level deduplication and per-candidate referee packets (graph id, equation, dev evidence).
  - `apedp_dyn/proposers/rime_proposer.py`: RIME as a `CandidateProposal` producer with a past-information-only `compute_score` (temporal hygiene, explicitly not a causal claim) that re-evaluates the graph on unseen frames; abstains on null worlds.
  - PyO3 entry points `rime_search_json` and `rime_evaluate_series_json` (batch-oriented FFI, spec §12).

### Fixed & Remediated
- **RIME per-node memoisation returned wrong values (Stage-1 audit defect 1):** the evaluator cached one value per *node* instead of per *(node, time step)*, so any node read both directly and through `Lag` returned the current row in both places; `x[t] - x[t-1]` evaluated to `[NaN, 0, 0, 0]` instead of `[NaN, 1, 1, 1]`. Evaluation is now series-at-a-time in arena (topological) order, releasing each node's series once its last consumer has read it, with unreachable nodes skipped and non-topological arenas rejected by `Graph::validate` instead of silently producing numbers. Regression: `tests/rime_regression.rs::difference_term_keeps_its_values`, `rust/apedp-kernel/src/rime/graph.rs::repeated_operand_is_not_memoised_across_time`.
- **`AddState` was not a state (defect 2):** all-ones input at $d = 0.5$ returned `[NaN, -0.5, -0.5, -0.5]`; a true decaying memory returns `[1, 1.5, 1.75, 1.875]`. Regression: `tests/rime_regression.rs::add_state_is_a_recursive_decay`.
- **Candidates were scored on their raw output (defect 3):** a correctly structured candidate with a small amplitude and a large offset was invisible (improvement $\approx 0$ against $\mathrm{var}(y)$ in the thousands). Regression: `tests/rime_regression.rs::scale_and_intercept_are_fitted_before_scoring`.
- **The bundled null test proved nothing (defect 4):** it charged 1.0 per edge while the world under test had MSE $\approx 0.08$, so no candidate could ever be promoted. The control now runs on the sampling-noise scale, is paired with a positive control through the identical scoring path, and bounds the false-promotion rate over 24 independent null worlds. A short-series arm (`n = 150`, 60 candidates) additionally pins the multiplicity charge; without it a fraction-of-variance penalty promotes fit noise.
- **Tactical Active Fraction Harmonization:** Resolved historical documentation drift where documentation stated $[5\%, 95\%]$ while code enforced $[5\%, 90\%]$. The range is now pinned strictly to $[5\%, 90\%]$ across code, documentation, and regression tests.
- **Audit Documentation Location:** Moved engineering audit report to [`docs/apedp_discovery_engine_ledger_and_benchmarks_audit.md`](docs/apedp_discovery_engine_ledger_and_benchmarks_audit.md) within the project repository workspace.
- **Wilson CI Boundary Handling:** Fixed numerical precision at boundaries ($k=0$ and $k=n$) in `wilson_ci` ensuring exact $[0.0, 1.0]$ bounds.

---

## [1.2.0] - 2026-10-11

### Added
- **Native Lean 4 Verification Pipeline (`lean_universal_bridge.py`):** Zero-disk in-memory verification using Lean 4 compiler's native `--stdin` streaming, reducing theorem verification time from hundreds of milliseconds of disk I/O to sub-50ms kernel checks.
- **Deflated Sharpe Ratio (DSR):** Closed-form implementation of Bailey & López de Prado (2014) accounting for selection bias, number of trials, non-normality, and extreme-value asymptotic expected max Sharpe ratio.
- **Decoupled Novelty Adjudication (`novelty_adjudication.py`):**
  - Predictive incrementality evaluated via Diebold-Mariano tests with HAC standard errors ($p < 0.01$).
  - Structural novelty evaluated against canonical literature registry (Kyle 1985, Amihud 2002, Biais-Large-Bouchaud resiliency, Carhart 1997 momentum) with $R^2 < 0.85$ novelty ceiling.
- **Stock-Split Scale Invariance Gate:** Replaced unphysical price-scaling checks with correct joint price/volume conservation: $P' = P/k$, $V' = kV$, testing $S(P', V') == S(P, V)$.
- **Competing Multi-Proposer Implementations:** Added `SparseMicrostructureProposer` (standardized ElasticNet) and `OrderBookResiliencyProposer` (exponential decay recovery).

---

## [1.1.0] - 2026-10-09

### Fixed
- **Rank Ablation Live Path:** Re-implemented missing rank ablation calculation in `canonical_ledger.py` verifying incremental signal contribution.
- **Hardcoded Sharpe/DD Display Literals:** Eliminated all hardcoded display literals; all metrics computed dynamically from per-observation ledger rows.
- **Rust GA Index-0 Bug:** Fixed genetic algorithm best-individual selection in Rust kernel (`src/genetic.rs`) to correctly select minimum-MSE individual rather than index-0 individual.
- **Decorative Gate Suppression:** Added `GATE_NON_BINDING` handling in `canonical_ledger.py` ensuring non-binding thresholds suppress Sharpe ratios and abstain.
- **Mask-First Imputation:** Enforced validity masks prior to any nan-to-num or imputation operations.

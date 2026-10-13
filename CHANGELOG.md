# Changelog: APEDP & APEDP-X Discovery Framework

All notable changes, bug fixes, audit remediations, and architectural milestones are documented in this file.

---

## [Unreleased] — RIME Repair v0.1.x freeze (comparative-validation milestone)

### Added (v0.1.2 static-grammar round, engine tag `rime_v0.1.2`)
- **Matched-compute harness (`benchmarks/matched_compute.py`, audit-only):** equal-budget comparison of RIME (temporal + static grammars), the existing proposers, the algebraic OLS-basis control, and PySR when importable — scored on a preregistered **grammar-coverage axis**: out-of-coverage abstention is correct behaviour, an emitted-but-failed proposal is a failure on any family, and static worlds require holdout **reversal-invariance** so grid-order tracking is scored `index_fit` (the QEC masquerade, caught by construction) rather than a recovery. Measured run recorded in `benchmarks/matched_compute_results.json` (PySR honestly recorded unavailable, not skipped).
- **Static search mode (kernel `config["static"]`, default OFF):** temporal primitives (lags, memory cells) are removed from seeds AND rewrites — grid-order fitting becomes unrepresentable for order-structured static data. No threshold or gate changes; the default grammar is byte-identical to v0.1.1.
- **Algebraic input transforms (`RimeProposer(algebraic_transforms=[...])`):** canned `sq/cub/ssqrt/slog` transform columns appended at the input layer, recomputed identically inside `compute_score` so proposals stay causally usable on unseen frames; NaN-preserving, unknown kinds rejected at construction.

### Fixed (v0.1.2 — this is also a CORRECTION of a prior conclusion)
- **Scale-dependent ridge in `fit_readout` (engine numerics, `rust/apedp-kernel/src/rime/eval.rs`):** the fixed `1e-9` literal added to `Σ(o−ō)²` exceeded the raw second moment of algebraic-scale regressors (a p² column ≈1e-8) by ~30× and silently zeroed every such fit — this, not the grammar, was the primary cause of the measured QEC static-law failure. The ridge is now relative to the regressor's raw second moment (identical for unit-scale data; the QEC attribution "RIME's grammar is the wrong instrument for static scaling laws" is retired: the temporal grammar reaches the law via its self-interaction rewrite, pinned by `test_dyn_ga_qec.py` at holdout R²>0.99 both directions).
- **`tests/rime_regression.rs` was stale against the v0.1.2 typed API** (referenced the removed `SearchPacket` surface and did not compile); rewritten as equivalents against `SearchOutcome` with true null-abstention, positive-control and budget assertions; `rime_stage1.rs` assertions recalibrated to commit 30e73bf's planted amplitude (previously internally inconsistent: even a perfect fit could not clear the control's own bar).

### Added (this round, earlier)
- **RIME Repair diagnosis v0.1.0 (`apedp_dyn/epistemic/rime_repair_diagnosis.py`, experimental):** frozen `ResidualReport` + finding vocabulary on real proposals; all cutoffs explicit caller args.
- **RIME Repair engine v0.1.1–v0.1.4 (`apedp_dyn/epistemic/rime_repair_engine.py`, experimental):** bounded R1/R2/R3 operators, chronological 4-role partitions with single-use sealed split (`HoldoutContaminationError` on reopen), five explicit gates, prediction-equivalence clustering. Promotion structurally deferred to the APEDP-X track (no CandidateProposal→ScoredCandidate bridge fabricated).
- **Sensitivity/recall milestone (v0.1.2–v0.1.3):** recall 9/9 strong planted defects with structure recovery; weak-defect curve never harms; BTC abstention regression; 5/5 walk-forward abstain; adversarial stability-block.

### Fixed
- **RIME `residual.rs` stub:** removed hardcoded threshold literal; residuals now use the fitted readout (consistent with `eval.rs` scoring); threshold is an explicit caller argument with regression tests.

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

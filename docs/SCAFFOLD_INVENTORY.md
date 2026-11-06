# Scaffold Inventory: What Computes, What Looks Up, What Refuses

This page exists because the repository contains three very different kinds of
code wearing the same clothes. A dictionary of published numbers and a simulation
both return a `dict`; a rule screen and a fitted model both return a verdict. The
only defence is to name the tier, in the return value, in a table, and in a test.

Every row below was read from the source. Nothing here is asserted from memory,
and each tier claim is pinned in
`tests/test_scaffold_inventory.py`.

## Tiers

| tier | means | may be used for |
|---|---|---|
| `computed` | derived from the inputs by arithmetic or a solver | decisions, subject to that model's own limits |
| `rule_screen` | a named threshold rule (Lipinski, GC content) | filtering, never prediction |
| `literature_lookup` | a published or typical value returned for a named key | planning, citations, sanity bounds |
| `planning_template` | a fixed answer that does not depend on the scientific input | never — it is a form |
| `reference_spec` | typical device or system parameters | requirements discussion |
| `refused` | raises, because the honest answer requires a model that is absent | nothing |
| `data_table` | a lookup into a curated or published table | citation, bounds, enumeration — never as a fitted result |

## The applied modules

| module | component | tier | what it actually does |
|---|---|---|---|
| `applied_biopharma` | `ADMETPredictor` | `rule_screen` | Counts Lipinski and Veber violations; `not_modelled` names thirteen properties it does not touch, including clearance, hERG and hepatotoxicity |
| | `UniversalDrugTargetDiscovery` | `literature_lookup` | Ranks a curated target table by a hand-set `tractability` weight; refuses `omics_data` |
| | `ClinicalTrialDesigner`, `DrugRepurposingEngine`, `CombinationTrialSimulator` | `planning_template` | Sample sizes keyed on phase, a three-drug table, constant response rates; no p-value is computed anywhere |
| `applied_synthetic_bio` | `CRISPRDesigner.design_guide_rna` | `rule_screen` | GC content is exact arithmetic; the efficiency score is `85 + (gc−50)·0.5`, labelled `linear_gc_rule` |
| | `CRISPRDesigner.predict_off_targets` | `computed` | PAM-adjacent protospacers on both strands of a **caller-supplied** reference, Hamming-scored against the guide with seed mismatches and positions reported, plus one-base bulge and single-indel alignments anchored at the PAM (counted in `bulge_off_targets`/`indel_off_targets` and `total_off_targets`, never folded into the mismatch-only `predicted_off_targets`). Ranking is reported twice — flat Hamming (`top_sites`) and a stated PAM-proximal ramp (`weighted_top_sites`, `position_weighted_score`) — with `ranking_comparison` splitting the pairs they disagree about into within-band and cross-band discords. The ramp is two constants, not a fitted CFD matrix. It refuses when no reference is given, because it previously returned `predicted_off_targets: 12` with `confidence: 'high'` for every input |
| `apedp/ingestion/genome_region` | `fetch_region`, `scan_region` | `computed` | Fetches a slice from Ensembl REST and runs the off-target scan on it. Every slice is validated before use: non-DNA characters, a body whose length disagrees with the coordinates requested, or a range above the 1 Mb cap raise rather than scan. The returned record carries assembly, coordinates, strand, URL, retrieval date and the sequence SHA-256, and each reported site gains `genomic_start` |
| | `ProteinDesigner.design_enzyme` | `planning_template` | Fixed template; Rosetta/AlphaFold/ProteinMPNN are in `tools_referenced`, and `executed_tools` is empty |
| | `MetabolicPathwayOptimizer` | `literature_lookup` | Four theoretical yields × an assumed 0.85 recovery; `titer_computed_g_l` is `None` |
| `applied_neuromorphic` | all | `reference_spec` | Neuron/synapse counts and power figures per device class; `measured: False`, and "1000x energy efficiency" is returned as `claimed_advantage`, not as a property |
| `applied_materials` | all | `computed` / `literature_lookup` | Phase-appropriate: energy density, wavelength ratio and capacitance are closed-form; `search_candidates` explicitly returns a fixed list and is not a search |
| `applied_climate` | all | `literature_lookup` | SSP warming figures are published central estimates; costs and removal potentials are references; sea level is an assumed linear scaling and is flagged `sea_level_rise_is_modelled: False` |
| `applied_fusion` | `PlasmaPhysicsAnalyzer`, `FusionFuelCycle` | `computed` / `literature_lookup` | Lawson triple product, beta, Debye length are real relations; TBR values are literature; `optimize_confinement` raises |
| | `FusionReactorDesigner.design_reactor` | `computed` | Power-balance gain `Q = P_fusion / P_heating`, with the note that no transport model exists |

## Infrastructure that does not do what its name suggests

| module | claim a reader might make | what it does |
|---|---|---|
| `apedp/ingestion/ingestion_academic_apis.py` | "it integrates NASA ADS, CERN, Wolfram Alpha" | Holds twenty endpoint strings. `CONNECTS_TO_SERVICES = False`; no client, no authentication, no request |
| `apedp/ingestion/ingestion_web_equations.py` | "it harvests equations" | Fetches Wikipedia HTML and extracts formula strings. The displaystyle scan is now **brace-balanced**, so a nested group survives intact; the flat regex it replaced cut the formula at the first inner brace, and an even earlier pattern never matched at all because it was double-escaped. Still lossy where it matters: macros are not expanded and non-displaystyle math is skipped, so `EXTRACTION_IS_LOSSY = True` stands and every returned string must be re-verified |
| `apedp_dyn/distributed/cluster_launcher.py` | "it launches clusters" | Renders Slurm `#SBATCH` scripts and Kubernetes/KubeRay manifests. `SUBMITS_JOBS = False`; no `sbatch`, no `kubectl`, no credentials |
| `rust/goldbach_verifier` | "it verifies Goldbach" | Verifies a finite range — 1e7 by default. The conjecture is already verified to 4e18, so this is a sieve-throughput benchmark. The banner now prints that sentence |

## Physical models: what their own docstrings already say

These pieces are honest, and the honest statement is narrower than the module
name suggests:

- **`apedp_dyn/molecular/se3_flow.py`** — harmonic bonds plus soft-core repulsion,
  with opt-in chirality terms on the signed volume. No electronic structure, no
  torsions, no learned message passing, no docking or folding. The chirality
  limitation the audit quoted is now a switch with its cost stated (SO(3)-only
  symmetry when enabled).
- **`apedp_dyn/robotics/trajectory_transport.py`** — unicycle kinematics with exact
  arc stepping. No wheel dynamics, slip, motor inertia or sensors.
- **`apedp_dyn/robotics/multi_agent_transport.py`** — Khatib artificial potential
  fields, not certified CBF-QP with forward-invariance proofs; the docstring says
  exactly that, and the head-on deadlock the radial field cannot break is handled
  by a stated tangential term.

## Second sweep: corpus, audio and a placeholder that returns a number

| module | tier | finding |
|---|---|---|
| `apedp/applied_electrical.py` | `computed` | Genuinely closed-form: per-unit fault level `base / Z_source`, duty cycles, Ziegler-Nichols gains, `Rds(on)·Qg`. Each result carries its own ceiling (`not a load-flow solution`, `efficiency not modelled`, `Phase noise not modelled`). No code change; registered so the tier is on record |
| `apedp/patent_prior_art.py` | `real` (bundled, 29 records / 15 normalised claims) | Was `refused`: the search and provenance discipline were real but no corpus existed. `data/patents/` holds twenty-nine publications — five fetched on 2026-10-31 by `scripts/fetch_patent_corpus.py`, the rest harvested on 2026-11-01 by `scripts/grow_patent_corpus.py` once Google's block lifted — each with a source URL, retrieval date, HTML SHA-256 and, for harvested records, the query that found it. Fifteen claims across seven records carry a normalised expression — one to four per record, so no record with a stated relation is bounded by a single claim — and twenty-two records state no relation in any claim and are counted as unsearchable. The growth step measured its own limit: the automatic extractor parsed **0 formulas from 501 claims**, because claim text states relations in prose, so the searchable set grew by hand normalisation instead (3 → 15 claims, `matched_claim_number` on every verdict) and an operands-and-one-operation claim (a bare quotient or product such as `Z_meas/Z_ref`) can no longer return `KNOWN` — seven of the twenty-nine are held apart as operand-only and report `CLAIM_STATES_OPERANDS_ONLY` — and `corpus_status()` reports `decisive_claims`, `operand_only_claims`, `records`, `unsearchable_records`, `claims_scanned_for_formulas`, `formulas_found_in_claim_text` and the corpus digest instead of a bare boolean. Both bulk paths (`--source patentsview`, `--source epo-ops`) are implemented and **exit 2 with the measured reason** when credentials are absent, because PatentsView and EPO OPS are credential-gated (and Google's query endpoint began serving a `503` block page mid-run) |
| `apedp_dyn/audio/spectrogram_transport.py` | `computed` (synthetic-only) | Verified on analytic chirps only, and its docstring says no claim is made about music, speech, source separation or codec-grade reconstruction. The upwind/CFL and non-periodic-boundary reasoning is sound; registered as synthetic-scope |
| `apedp_dyn/sandbox/finance_l2_demo.py` | `computed` | `epistemic_position_sizing` mapped a verdict to leverage, and `POSITION_SIZING_IS_DEMO_ONLY` existed to say so. Both are retired, along with the result field. The replacement `size_positions` solves a cost-penalised mean-variance problem (`w'μ − (γ/2)w'Σw − c\|w − w_prev\|₁`, capped simplex, proximal gradient) over a forecast, a **covariance**, a **fee** and a stated risk aversion, and reports weights, turnover, cost, cap-binding and a reason. The verdict now only gates admissibility (no identified edge ⇒ nothing to size; a sign that flips under resampling ⇒ refused). Three behaviours are measured and pinned: the 5 bps fee makes the demo's own momentum regime decline to trade (its holdout mean over that window is **negative**, −2.4% per bar, so the old 1.0 allocation came from the verdict rather than from the data), the real-price benchmark's admitted decision can likewise decline a trade, and collinear factors produce a stable corner position rather than the old table's 50% split |

One correction to the audit as written: there is no `qed/` directory. The negative
result is recorded at `benchmarks/equation_discovery/qed_001_recovery.json`, and
this page cites that path rather than restating its numbers.

## Seventh sweep: notebook lineage and a dead attribute

Two leftovers an earlier sweep named and did not close. Both were found by asking
the file rather than the module list.

| module | was | now |
|---|---|---|
| fifty-five `*.py` files across `apedp/`, `apedp_dyn/`, `packages/`, `scripts/`, `benchmarks/` and `tests/` | Opened with `# Databricks notebook source`, and fourteen also carried `# COMMAND ----------` separators — 69 marker lines. Inert, but not provenance: the files have been edited as ordinary Python, so the header told a reader they were notebook exports | Markers removed. Nothing reads them — `apedp_dyn/notebook_adapter.py` detects the *runtime* (`DATABRICKS_RUNTIME_VERSION`), not the file header — so no behaviour moved. A `grep` for the marker now returns only the `.kilo/` worktree copies and the vendored tree |
| `apedp/advanced_search.py` | `HierarchicalSearch.__init__` built `self.latent = LatentSpace()` and no method ever read it | Removed. The attribute was decoration, and with `LatentSpace.decode` refusing there was no path on which it could have returned an expression. The class stays exported for the distance and interpolation helpers, which are defined on the hash |

## Static-table census (`apedp`)

`scripts/audit_static_tables.py` reports every function in the package that
returns a value read from a module- or class-level container literal instead of
computed. Eleven modules qualify — ten rows below, since two modules share one
table — and **none of them is wrong**: a units registry, a seed list, an
evidence-tier map and a mechanistic-priors table are all things that should be
tables. They are listed so that a reader who sees
`suggest_mechanistic_priors` cannot mistake it for a fitted posterior.

| module | returns from | what the table is |
|---|---|---|
| `apedp/core.py` | `DIM_MAP` (`get_unit_vector`) | Unit registry for SI, economics and CS. A lookup, not a derived dimension |
| `apedp/discovery/_core.py`, `apedp/family_detection.py` | `MECHANISTIC_PRIORS` (`suggest_mechanistic_priors`) | Curated domain priors *offered* to the search. Nothing is fitted |
| `apedp/equations/seeds.py` | `SEED_LAWS` (`seed_pages`) | Wikipedia seed pages for the corpus pipeline |
| `apedp/evidence.py` | `_RANK` (`rank`) | Ordering of evidence kinds |
| `apedp/gated_assurance/checks.py` | `_FINITE_TARGET_WORDS` (`_asymptote_target`) | Words that denote a finite asymptote |
| `apedp/gated_assurance/contracts.py` | `ADJUDICATION_EVIDENCE_TIER`, `ADJUDICATION_EVIDENCE_BASIS` | Which evidence supports which adjudication tier |
| `apedp/ingestion/external_benchmarks.py` | `CASES` (`list_cases`) | The five AI Feynman cases, each attributed and dimensionally annotated |
| `apedp/ingestion/ingestion_academic_apis.py` | `INSTITUTIONAL_TARGETS` | Endpoint configuration; see the infrastructure table above |
| `apedp/scientific_layers.py` | `DISCOVERY_TIERS` (`benchmark_tier_summary`) | Tier definitions used to label benchmark results |
| `apedp/applied_materials.py` | `REFERENCE_CANDIDATES` (`search_candidates`) | The fixed superconductor reference list; see the applied table above |

**What the census correctly excluded.** An earlier run also reported
`ablation.to_dict`, three `epistemic.to_dict` methods, `evidence.to_dict`,
`evidence_report.to_dict` and `source_audit.{safe_for_discovery,requires_human_review}`.
Every one of those is a dataclass field with a tuple or list default —
indistinguishable from a class-level table by syntax — returned through `self`.
The script now excludes any attribute reached through `self`, and the five
remaining hits were checked by hand rather than assumed.

A table is only a problem when it answers a question about the world that the
caller believes was computed. None of the eleven does; each is cited or
enumerated by construction, and the test pins the set so a new one cannot appear
silently.

**The census is a gate, not only a report.** `scripts/audit_static_tables.py
--check` compares the live tree against `scripts/static_table_census.json` — the
single record both the check and `tests/test_scaffold_inventory.py` read, covering
both roots — and exits `1` when a table appears, disappears, or moves between
modules. It runs as the `static-table-census` hook in `.pre-commit-config.yaml`, so
an unrecorded table fails before review rather than after. Exit codes: `0` matches
the record, `1` drifted, `2` the check could not run. Adding a table therefore
takes two edits — the record and a row in one of the tables above — and the suite
refuses a record whose functions are not named on this page.

## The scope scan: the same rule, as a gate

`scripts/audit_static_tables.py` answers *what does this function return*. That is
an inventory question, so it reports a units registry and a fabricated prediction
alike. `scripts/scope_scan.py` asks the narrower question — and fails when the
answer is bad.

A function is reported when all three hold:

1. **The name promises an answer.** `predict_*`, `forecast_*`, `project_*`,
   `design_*`, `optimize_*`, `discover_*`, `search_*`, `recommend_*`, `select_*`,
   `infer_*`, `propose_*`, `classify*`, `diagnose*`, `rank*`, `score*`, matched at a
   word boundary — so `proposer_id` is not read as `propose_*`, and
   `selected_descriptor` is not read as `select_*`.
2. **No parameter reaches a return.** If every return ignores the inputs, the value
   cannot be about them.
3. **The value was not computed**: every return is a literal the body never
   computed (and the body reads no parameter at all), or every return is a
   module- or class-level container found by the census.

Two exclusions carry the weight, and both exist because the first run produced
false positives without them. A **predicate** — `forecast_regime(x)` returning
`"UP"` or `"DOWN"` — returns literals but branches on its input, so the literal
branch additionally requires that the body never reads a parameter. A **sieve** that
classifies a frame and then maps the class through a routing table
(`classify_modality`) reads a table on one path, so the table branch requires that
*every* return is a lookup. A function that raises instead of returning is a
refusal, which is the behaviour this repository asks for.

**Roots.** `apedp` and `apedp_dyn`. Vendored and generated trees are skipped
(`_qualtran_src`, `benchmark_corpus*`, `node_modules`, `__pycache__`, `tests`), so
the scan reports this repository's code and not another project's.

**Current findings: two.** Both are tables, both recorded in
`scripts/scope_scan_baseline.json` with a reason, and neither is a fabricated
number:

| module | function | basis | why it is tolerated |
|---|---|---|---|
| `apedp/applied_materials.py` | `search_candidates` | `REFERENCE_CANDIDATES` | A fixed list of four literature superconductors. The name promises a search over candidate space; the return is a citation list, which is why the docstring says "Not a search" and the tier above is `literature_lookup` |
| `apedp/evidence.py` | `rank` | `_RANK` | A fixed ordering of evidence kinds (`PROVED` 0 … `UNRESOLVED` 3) — a schema decision, not a judgement about the caller's data |

**Exit status is the check**, so it runs as a command as well as from the suite:

- `0` — findings are exactly the baseline;
- `1` — a new finding appeared, or a recorded one vanished, because a silent fix
  still changes the record;
- `2` — the scan could not run.

The baseline is not a place to hide a constant. Every entry needs a `why`, and
`tests/test_scaffold_inventory.py` requires that `why`, a row on this page, and the
finding itself to agree — so adding a name to the file without explaining it here
fails the suite, and a new predictive constant fails whether or not the file is
edited.

**What it cannot catch.** A constant under a name with no predictive prefix
(`helper_lookup`), a constant assembled by a helper the function calls, and a table
read on every path but composed with a computed key. Those stay greps for a reader
and cases in the sweeps above: the scan buys the exact rule, not the whole
judgement.

## Third sweep: the remaining `apedp` modules

**Method.** An AST scan over every `apedp/**/*.py` for two signals — a parameter
that is never referenced in the function body, and a return that is a pure
literal table with no name, call or attribute in it. Both sets were then read by
hand. Most hits were predicates (`_pam_matches`, `validate_homogeneity`,
`replication_tier`, `_classify_regime`, `revoke_calibration`, …) whose returns are
literal verdicts chosen by their inputs — control flow, not constants — and those
are not listed. What follows is what survived reading.

| module | component | tier | finding |
|---|---|---|---|
| `apedp/applied_defects.py` | all six auditors | `computed` | Regularized aerodynamics, Knudsen, Butler-Volmer, Semenov, Hall-Petch, Griffith, Gibbs-Thomson and Casimir relations, each evaluated from its inputs with `defect_detected` computed rather than asserted. The last `applied_*` module; no constant returned |
| `apedp/scenarios.py` | `scenario_20_anscombe_metrology` | `computed` | **Was a constant.** It computed means, variances, correlations and quadratic curvature for the four Anscombe datasets, asserted dataset 2 had the distinguishing curvature, then returned a fixed one-line string — so the evidence for its own claim was discarded. The result now carries `curvature_a_by_dataset` and the per-dataset `statistics` |
| `apedp/source_ingestion.py` | `PublicationText.ready_for_discovery` | `refused` | Returns `False` unconditionally. That is the right answer — extracted prose is not measured data — but it was an undocumented constant. It now states the refusal and exposes `readiness_reason` |
| `apedp/scientific_layers.py` | `_const` | `computed` | The only function in the package whose parameters are never read, and it is meant to be: a constant-predictor null model in the A/B comparison ladder |
| `apedp/model_synthesis.py` | `_exp_templates` | grammar | Four fixed exponential templates with the rate fitted and refined downstream. A search grammar, not a result |
| `apedp/quantum_checks.py` | `amplitude_convention_note` | convention | A constant dict of parsing conventions; its own docstring already says "Convention, not code" |

Two zero-argument helpers that look like constants in the scan but are not were
checked and left alone: `gated_assurance.contracts.replication_tier` and
`model_synthesis.epistemic_class` map a state attribute through thresholds, and
the scan missed the attribute because `self` is excluded from its parameter list.

## Fourth sweep: `apedp_dyn`

**Method.** The same two signals as the third sweep, over `apedp_dyn`: a parameter
that is never referenced in the function body, and a return that is a pure literal
with no name, call or attribute in it. The scan is the same AST as
`scripts/scope_scan.py` (which already covers this package for the predictive-name
rule and finds nothing), plus the static-table census. **161 candidates** survived
the two signals and were read by hand.

Most of the 161 are the audit working as designed, and are not listed:

- **Abstract and protocol methods** — `vqc/backends.run`, `grammar/proposer.propose`,
  the four `quantum_compilation/experiment*.py` protocol methods, and the six
  `remote_storage` declarations. Their parameters are unused because the body is
  `...` or `raise NotImplementedError`, which is what an interface is.
- **Interface-parity `**kwargs`** — twelve `fit_and_propose(**kwargs)` and
  `fit_and_propose(*args)` implementations that accept the dispatch signature and
  use what they need.
- **Deliberately unnamed arguments** — `_t` in `ui/server.odes` and the two
  `trajectory_transport.velocity_fn` closures, `_objects`/`_exc`/`_args` elsewhere.
  An underscore is the convention for "accepted, not read".
- **Literal verdicts chosen by control flow** — `drift_e5.classify`,
  `governance.verify_ed25519_signature`, `caller_data_protocol.validate_caller_compliance`,
  `vqc/stabilizer.is_clifford_circuit`, `vqc/colouring.descend`. These read their
  inputs and return a label; they are predicates, and the scope scan's literal rule
  excludes them explicitly for the same reason.
- **Thirteen `proposer_id(self)` methods** returning a constant identifier string.
  The name starts with `propose`, so the scope scan's first draft flagged all of
  them; the word-boundary rule keeps them out, because an identifier is a property
  of the proposer and not a proposal.

What survived reading is below. Every row was measured, not inferred: the test named
with it moves the knob and shows the answer does not move.

| module | component | finding |
|---|---|---|
| `apedp_dyn/sandbox/execution_cost_models.py` | `compute_cost_drag` on three models, one interface | Each ignores the inputs its own formula has no term for, and the signature does not say so. `FixedPlusSlippageCostModel` ignores `volumes` (drag is identical for volumes of 1e3 and 1e9); `AlmgrenChrissImpactCostModel` ignores `spreads`; `LimitOrderQueueCostModel` ignores `prices`, `volumes` and `spreads`. The interface is not uniformly dead — fixed-plus-slippage *does* use `spreads` — so a caller cannot tell which arguments bind |
| `apedp_dyn/epistemic/shape_inference.py` | `infer_shape_prior` | Accepts `improvement_margin`, `min_flat_fraction` and `min_run_length` and reads none of them: margin 0.01 with flat fraction 0.1 and run length 2 returns the same prior as 0.99 / 0.99 / 20 |
| `apedp_dyn/gates/scaling_invariance.py` | `ScalingInvarianceGate.check_with_pi_groups` | The docstring says it checks the candidate's exponent vector "on those groups"; the body is `return self.check(...)` and never touches `pi_groups`, so two different Π tuples give the same verdict by construction. Two more findings in the same file: the gate instance ships with **no regimes** (`Available: []`), so every regime name raises, and the module ends with a leftover editing note ("Wait, I made a typo. Let me fix it.") |
| `apedp_dyn/robotics/cbf_certificate.py` | `certified_cbf_guard` | Ignores `headings` entirely. The certificate is a point-mass pairwise-barrier QP over positions and nominal velocities; velocities and diagnostics are identical for headings 0 and (3, 1.5, −2). It therefore says nothing about the unicycle heading that `trajectory_transport` integrates — the boundary is real, but no signature states it |
| `apedp_dyn/quantum/sparrow_photonics.py` | `simulate_type2_fusion(input_state)` | Ignores the state it names: the body computes a HOM visibility and an end-to-end efficiency, so it is a two-mode success-probability model, not a simulation of an input state |
| `apedp_dyn/sandbox/eseds_dynamical.py` | `run_eseds_trap` | Accepts `r`, `K`, `N0`, `bootstrap_n_boot` and forwards none of them: the trap is pinned to family A / trap 1 and the logistic parameters are internal, so changing the growth rate or the carrying capacity changes nothing |
| `apedp_dyn/estimation/square_window.py` | `square_window_coefficients(decimate)` | The estimator accepts a decimation factor and never decimates |
| `apedp_dyn/conf/dimensionless.py` | `discover_pi_groups(variable_units)` | Ignores the unit map: identical groups with and without units. The exponent matrix already carries the dimensions, so the argument is metadata that never reaches the group |
| `apedp_dyn/epistemic/prior_guided_search.py` | `_sign_basis_candidate(x_dev, y_dev, seeds, ...)` | Reads `seeds`, never the dev arrays. "Exact on dev by construction" is true of the seeds that were fitted elsewhere, not of this function |
| `apedp_dyn/sandbox/independent_proposer_benchmark.py` | four `generate_proposer_*_cases(seed)` | The seed drives nothing: seed 1 and seed 2 return equal case lists. A comment says so — cases are built deterministically from the index for interface parity |
| `apedp_dyn/vqc/synthesis.py` | fourteen corpus builders `(seed)` | The deterministic builders accept `seed` uniformly for dispatch parity and ignore it. A no-op here is the good outcome — the corpus is reproducible — but the argument still promises randomness |
| `apedp_dyn/audio/spectrogram_transport.py` | `predict_next_frame(order)` | Documented no-op: "`order` is retained for interface compatibility"; the stepper is first-order upwind. `validation/energy.simulate_pair(m, k, ...)` is the same pattern ("accepted for backward compatibility and unused") |
| `apedp_dyn/epistemic/remote_storage.py` | `BaseStorageBackend.lock(retry_interval)` | The retry interval never reaches `acquire_lock`; each backend sleeps its own hard-coded 0.02–0.1 s |
| `apedp_dyn/sandbox/hall_pecht_demo.py`, `sandbox/plasma_simulator.py` | `generate_hall_pecht_data(d_range)`, `generate_synthetic_dataset(n_conditions)` | Generator knobs that do nothing: the datasets do not change when the requested range or condition count changes |
| `apedp_dyn/qed/referee.py` | `ablation(coefficients)` | The candidate's own coefficients are unused — every reduced support is refitted, so a supplied fit and a fresh fit give the same ablation |

**One correction to the sweep as run.** The cost-model row is not a defect of one class:
`ExecutionCostModelBase.compute_cost_drag` declares `volumes` and `spreads` for every
subclass, and each subclass implements the terms its own formula has. Reading the four
implementations against their docstring formulas is what turned "unused parameter" into
"a caller cannot tell which arguments bind".

**Static-table census for `apedp_dyn`.** Seventeen modules outside the vendored
trees read a container literal on a return path. All are catalogues, registries or
step ladders, and none answers a question about the world:

| module | table | what it is |
|---|---|---|
| `conf/benchmark_v2.py` | `KAPPA_BANDS` | Threshold bands addressed by `run_ladder` |
| `epistemic/frozen_spec.py` | `FROZEN_SPEC` | The frozen-adjudication constants |
| `epistemic/multiple_testing.py` | `METHOD_NAMES` | Canonical spellings of the correction methods |
| `fitting/parameter_fit.py` | `MULTISTART_SCALES` | Scale ladder for the multi-start search |
| `grammar/candidate_ir.py` | `parent_candidates` | See the false positive below |
| `orchestration/modality_sieve.py` | `DYNAMIC_ENGINES`, `STATIC_ENGINES` | Which engine a classified modality routes to |
| `problems/__init__.py` | `ALL_PROBLEMS` | The problem catalogue |
| `qed/library.py` | `OSCILLATOR_TERMS` | The hypothesis library offered to the search |
| `quantum_compilation/quantum_signal.py` | `CONTINUOUS_STEPS` | The step grid the trial charge is computed from |
| `sandbox/auto_degradation.py` | `TRUTH` | The trap's own ground truth |
| `sandbox/crypto_grammar.py` | `CRYPTO_ASSETS` | The asset list |
| `sandbox/multi_asset_signals_sandbox.py` | `SIGNAL_NAMES` | Disclosure names |
| `ui/server.py` | `SANDBOX_CATALOG` | The endpoint catalogue |
| `validation/energy.py` | `RESTORING_FUNCTIONS` | Which restoring-force forms are admissible |
| `vqc/equivalence_corpus.py` | `SCOPES` | Scope labels per pair |
| `vqc/external.py` | `TOOLS` | The external-referee registry |
| `vqc/native.py` | `_LAST_FAILURE`, `_TALLIES` | See the false positive below |

**Two census false positives, found by reading rather than by trusting the list.**
`grammar/candidate_ir.base_provenance` is flagged because it returns
`list(c.parent_candidates)` and a class body elsewhere has a table of that name; the
attribute is reached through the candidate `c`, not through `self`, so the census's
field exclusion misses it — the function is assembled from the candidate and is not
a lookup. `vqc/native.last_failure` and `tallies` read the native extension's
module-level state, which is an accessor over state, not a table.

**What this sweep does not cover.** Behaviour behind a correct signature — a
provider that ignores its parameters and forwards nothing is visible only at the
call site — and the vendored harness under
`sandbox/google-verifiable-quantum-circuit-compilation/`, which is third-party code
with its own tables and its own `_qualtran_src` tree. Both are named here instead of
being counted.

**The five vendored modules the census finds, named instead of dropped.** Run
without the record's exclusion, the census reports 22 modules under `apedp_dyn`:
the seventeen above, and five inside the third-party harness. They are not this
repository's tables, so `scripts/static_table_census.json` leaves them out of the
gate — but an exclusion is only honest if the set it hides is written down:

| module | tables | functions |
|---|---|---|
| `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src/dev_tools/qualtran_dev_tools/bloq_report_card.py` | `CHECKCOLS`, `IDCOLS` | `get_bloq_report_card` |
| `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src/qualtran/qlt_ir/_examples.py` | `QLT_IR_EXAMPLES` | `get_qlt_ir_examples` |
| `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src/qualtran/surface_code/flasq/volume_counting.py` | `_SIMPLE_GATE_MAP` | `compute` |
| `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src/qualtran/surface_code/ui.py` | `_MAGIC_FACTORIES`, `_QEC_SCHEMES`, `_SUPPORTED_ESTIMATION_MODELS` | `input_components`, `qec_summary_components`, `_MAGIC_FACTORIES_components` |
| `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/scoreinfra/reference.py` | `FIXED_SHIFTS_KEYS` | `load_fixed_shifts` |

Three of the five are Qualtran's own source tree (`_qualtran_src`), and two are
harness code that came with it. A change to any of them is a change to vendored
code, so the pre-commit gate stays silent by design and the reviewer reads it in
the diff. The set itself is pinned by the suite, so the exclusion cannot quietly
grow: a sixth vendored module has to be named here before it can be ignored.

## Negative demonstrations, recorded rather than narrated

`benchmarks/equation_discovery/qed_001_recovery.json` is the authority for the
quantum-equation-discovery result. The record holds the numbers; this page does
not restate them from memory, and a negative result that is on disk is worth more
than one that has been summarised.

## What is genuinely strong, stated narrowly

- **`apedp_dyn/vqc/`** — Clifford/stabilizer simulation via 64-bit word XOR and
  Gaussian elimination over the Pauli group (`stabilizer.py`), with an equivalence
  corpus and a foreign referee for independent adjudication.
- **`rust/apedp-kernel/`** — the symbolic kernel: expression arenas, physics and
  validation primitives (`physics.rs`, `validation.rs`, `ngts.rs`).
- **`apedp_dyn/ode/`, `apedp_dyn/pde/`, `apedp_dyn/proposers/sindy_proposer.py`** —
  coupled-ODE, SINDy and PDE proposers over catalogued problems including
  `problems/reaction_kinetics.py`.
- **`apedp_dyn/epistemic/`** — the referee stack: `trial_ledger.py` (Deflated
  Sharpe Ratio, charging `effective_trial_count`), `multiple_testing.py` (exact
  BH / BY / Holm), `experiment_ledger.py` (holdout burning), and the SHA-256
  pinned canonical spec.
- **`apedp_dyn/sandbox/canonical_ledger.py`, `lob_transport.py`** — causal feature
  construction, fee identities and regime backtests with the holdout discipline
  applied.

The distinction that matters is not "physics-good, biopharma-bad". It is that a
model which states its own limits can be trusted within them, and a table which
does not say it is a table cannot be trusted at all.

## The one rule

**A constant may not be returned as a prediction.** The audit's most serious
finding was not that these modules are simple — simple is fine — but that several
returned fixed numbers under names like `predicted_off_targets`,
`titer_prediction_g_l` and `advantage_over_gpu`. Three mechanisms now prevent it:

1. **`evidence` in the return value** — `rule_screen`, `literature_lookup`,
   `planning_template`, `reference_spec`, `gc_heuristic`. A caller reading the
   dict sees the tier without reading the source.
2. **`not_modelled` alongside** — the properties that are absent are listed next
   to the verdict, because "drug-like: good" invites assumptions that only an
   explicit absence list dispels.
3. **`SCOPE` / `NON_PREDICTIVE_COMPONENTS` plus a test** — declarations that are
   pinned behaviourally, so they cannot drift from the implementation.

## Fifth sweep: placeholders that were still doing arithmetic

The sweeps above named modules. This one closed the placeholders that had
survived because they were *plausible*: a term that looked like a constraint, a
flag that looked like a backend, a mask that looked like a segmentation.

| module | was | now |
|---|---|---|
| `apedp_dyn/grammar/mechanical_grammar.py` | `c1`/`c2` built the constant `1` as a "constraint basis"; `grad_x`/`laplacian_x` returned a free symbol with no grid behind it; `zscore_x`, `ma_x`, `mom_x`, `rsi_like`, `bollinger`, `macd_like` referenced `mu_x`, `sigma_x`, `x_lag1`, `x_ma`, `x_std`, `x_ema_*` — symbols no caller ever binds | Each term stays in the vocabulary (enumeration, families, canonical forms and the proposer's filters all use the name) and its builder raises `UnavailableGrammarTerm` carrying the reason. Materialisation refuses, so pruning records a rejection instead of building a model with an unbound symbol. The withdrawn `c1`/`c2` slates are why the coupled ODE slate narrows from **1764 to 225** candidates; the difference is exactly the slates that carried a constant-1 "constraint". The survivor count quoted with this change, 64, was later found to be a `dxr`-formatting artefact rather than a dimensional filter — corrected in the `coupled.py` row below |
| `apedp_dyn/grammar/coupled.py` | Two defects in the same expression string. (a) `parts.append(f"({p}*{sstr(built)})")` never parenthesised the built expression, so a term whose builder returns a *sum* had its coefficient applied to the first addend only — `dxr` (XR − X) emitted `(q2*-x1 + x2)`, the remote position entering with coefficient 1, and gate 1 rejected **every** coupling candidate as a length added to an acceleration. The 64 "survivors" were exactly the slates *without* a coupling term, i.e. the uncoupled rival D2-COUPLED exists to defeat. (b) `enumerate_coupled` emitted a saturation term with no `scale_parameter`, so `_bind_scale` refused it and every two-DOF saturation candidate was recorded as unbuildable | All four materialisers parenthesise the built expression (the same line bug affected `u_relax` in the single-DOF grammar). The coupled slate keeps **225 of 225** — gate 1 filters nothing there, so the old 64 was never evidence of dimensional filtering. `enumerate_coupled` gives each mass's `langmuir_bind`/`hill2_bind`/`hill3_bind` its own fitted reference level (`s0_langmuir`, `s1_langmuir`) in the state's units with `role="scale"`. A planted two-DOF Langmuir plant recovers `s` = 0.260 / 0.428 against 0.30 / 0.45 (every parameter within 13.9 %) through enumerate → prune → materialize → fit, recorded in `benchmarks/two_dof_saturation_fit_results.json` by `scripts/measure_two_dof_saturation_fit.py` and pinned by `tests/test_dyn_two_dof_saturation_fit.py` |
| `apedp_dyn/grammar/proposer.py` | The sparse library included the withdrawn terms and, on a caller-supplied vocabulary, near-duplicate columns (`sin_x`, `cos_x` vs `x`, `1`) | Withdrawn terms contribute no column and are reported in `skipped` with their reason. `drop_collinear_columns` drops a column correlating above `COLLINEARITY_LIMIT = 0.999` with one kept before it — measured need: on the Duffing trajectory `sin_x` correlates with `x` at 0.99997, so whichever loaded first absorbed the amplitude and the damping term fell under the 5 % gate. The Duffing recovery is `{u, x, x3, v}` again |
| `apedp_dyn/cuda/batch_physics_steppers.py` | `HAS_TRITON` reported the *import*, and `is_triton_available()` returned True on an installed Triton plus CUDA, although the module defines **no** Triton kernel | `TRITON_KERNELS` is an explicit (empty) registry and `is_triton_available()` is gated on it, so the answer is False even where Triton imports and CUDA is present. Every stepper is PyTorch tensor arithmetic and the module says so |
| `scripts/animate_picture_to_video.py` | Sky was always the top 40 %, ocean the bottom 40 %, the car the middle 30 % — three coordinates applied to every input image | `apedp_dyn/video/region_masks.py` derives the bands from the latent's per-row texture energy: the two largest adjacent-row steps split the frame, and the returned record carries the boundaries and the band energies. The division of labour is stated — texture rank is measured, naming the bands is the caller's. Audited against ten licensed photographs ([`REGION_MASK_AUDIT.md`](REGION_MASK_AUDIT.md)): boundaries land within tolerance on 2 of 7 scored frames, are wrong on 4, and 3 frames with no three-band structure still receive two boundaries — median error 0.18 of the frame height. Re-run with a pretrained MiDaS depth map instead of RGB and the splitter held fixed, texture still wins (median 0.182 vs 0.228) because the depth rows are ramp-dominated (median ramp R² 0.83) and the gate refuses all ten depth splits. A third arm on ADE20K semantics (SegFormer-B0, weights pinned) cuts where the class composition changes: median error 0.115 against texture's 0.182 and worst 0.268 against 0.280, but the same 2 close / 1 partial / 4 wrong verdicts and 4 of 10 gate endorsements — a better signal that fixed no verdict |
| `packages/jetflow-video/jetflow_video/text_encoder.py` | MD5 over word and trigram keys plus 18 hand-written concept tuples, presented as a language encoder | Labelled: `EMBEDDING_METHOD = "feature_hashing+hand_written_concept_ontology"`, `PRETRAINED_SEMANTICS = False`, a 64-bit BLAKE2b digest instead of MD5, and a docstring that says what a hash can and cannot carry |
| `apedp/qd_generation.py` | `yu_cdse_rows` evaluated a published sizing polynomial and the run could still return `GENERATION` for beating refit-Brus on it | Rows carry `provenance` and `independent_measurement: False`; `adjudicate_verdict` withholds `GENERATION` unless the rows are the measured Phase A table, and keeps `CONFIRMATION` (agreeing with the source curve is what the numbers support) |
| `apedp/experimental_physics.py` | Slip `0.02`, efficiency `0.9`, a `1000 V` primary and a `1000 W/m²` irradiance were inside the arithmetic | All four are parameters with the same defaults, echoed in `assumptions` with what each one is (operating point, nameplate assumption, AM1.5 irradiance). Every return carries `evidence: closed_form_textbook` |
| `apedp/neural_discovery/symbolic.py` | `if train: # Teacher forcing would go here — pass`, so `tokens` held BOS while `logits` held every step and the REINFORCE ratio scored an empty slice; the loss returned `float(...)`, which `grad` cannot differentiate; `train_step` carried `@jit` over a Python token loop | `decode(..., targets=...)` implements teacher forcing and uses it in `supervised_loss`; sampling passes `rng`, so logits and chosen tokens align; losses are scalar arrays; `train_step` is eager with `TRAIN_STEP_JITTABLE = False` and the reason stated |

## Sixth sweep: two more places where failure looked like a result

Both were found by grepping for the *shape* of the earlier findings rather than
for a module name: a capability flag that is true, and an empty return that a
caller reads as a null result.

| module | was | now |
|---|---|---|
| `packages/jetflow-video/jetflow_video/stepper.py` | The "Rust fast path" branch was `if _RUST_AVAILABLE: try: pass except: pass`. The flag was true — the kernel really does export `jetflow_compute_acceleration` — and the kernel was **never called**, so `tests/test_jetflow_rust_kernels.py` compared the Python path with itself. A parity test that cannot fail is decoration | `_acceleration_between` calls `apedp_kernel.jetflow_compute_acceleration` on both the directional and the central-difference paths, restores dtype and device, and falls back to the same subtraction only when the kernel refuses a shape. `NATIVE_ACCELERATION_CALLS` makes the branch observable, so a test can show it ran; pinned by `tests/test_silent_scaffolds_refuse.py` |
| `apedp_dyn/grammar/llm_proposer.py` | A missing API key, an unreachable base URL, a missing `httpx` and a provider 503 all returned `[]` — the same value as "the model proposed nothing" — so a pipeline reported a null result from a process that never ran. Three `except Exception: return []` sites and one bare `return []` on the disabled path | `LLMProposerUnavailable` raises for configuration and transport failure, naming the provider, the model and the environment variables to set. The deliberate case (disabled) still returns empty and records `last_outcome` (`"disabled"`, `"no_api_key"`, `"ok:N"`), so an empty list is always explainable. Partial parse failures stay partial and are logged |

Registered, not changed, with the reason each is already honest:

| module | verdict | why |
|---|---|---|
| `jetflow_video/model.py`, `jetflow_video/trainer.py` | honest import probes | `XFORMERS_AVAILABLE`, `CHECKPOINT_AVAILABLE` and `LPIPS_AVAILABLE` are *consumed* with real fallbacks — SDPA attention, eager blocks, no perceptual loss — so they describe a code path rather than a result. Unlike the Triton flag the fifth sweep closed, there is no capability claim without a kernel behind it |
| `apedp_dyn/problems/finance.py` | recorded, inert | The truth tuple carries `DynamicalEquation("v", "0")` as a placeholder for "the market is the truth". `finalize()` is bypassed and nothing integrates that field, so it cannot yield a number; a truth tuple shorter than the state vector is a shape the problem zoo's invariant does not allow, so it is named here instead of being silently rewritten |
| `apedp_dyn/de_arm/mesh_discovery.py` | correct as written | The early `PdeMathModel(..., dx={"x": 1.0})` is a fail-closed type probe *before* any search; the returned model carries `dx = _median_edge_length(v, diag)` and says so in `extra`. Read before being "fixed" |

### Sixth sweep follow-up: the two trees this repository no longer carries

Both were registered here as "carried without compiling" and have now been
deleted rather than described. Removing them changed no behaviour.

- **`crates/apedp_kernel/`** — ten Rust modules abandoned when development moved
to `rust/apedp-kernel`. The root `Cargo.toml` listed `rust/apedp-kernel` and
`rust/apedp-vqc` as members and never included `crates/apedp_kernel`, so the tree
was never built; the manifest had once pointed at a path that did not exist, so
cargo could not even agree on which crate was meant. Deleted with its dependabot
and CODEOWNERS entries, and the plan document that described building it now
names the tree that actually shipped.
- **`packages/apedp-{audio,continuum,distributed,molecular,quantum,robotics}`** —
packaging shims whose `__init__.py` inserted the repo root on `sys.path` and
re-exported from `apedp_dyn`, with `apedp-continuum` adding two forwarding
module files. They carried no implementation: the behaviour lives in `apedp_dyn`
and is tested there, so each wheel was a second name for one implementation —
and its `production` row in [`DOCUMENTATION.md`](DOCUMENTATION.md) claimed a
shipped artifact for code the core already shipped. `packages/` now holds
`jetflow-video` only, which is standalone by contract.

Pinned by `tests/test_scaffold_refusals.py`, which asserts both trees stay gone
and fails if either returns.

Where the honest answer requires an input that is not present, the call raises:
`predict_off_targets` without a reference sequence, `optimize_confinement`, and
`discover_targets` when given omics data. A refusal is a smaller failure than a
fabricated answer — and when the missing input arrives, the refusal is replaced
by the computation, not by a constant. `predict_off_targets` is the worked
example: given a reference it now scans both strands and counts mismatches.

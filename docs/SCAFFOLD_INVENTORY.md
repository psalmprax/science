# Scaffold and evidence-tier inventory

This page records applied and infrastructure surfaces where a result can be
mistaken for a model prediction, measured engineering result, or completed
operation. It distinguishes what the code computes from what its inputs and
evidence justify.

## Evidence tiers

| Tier | Meaning |
| --- | --- |
| `computed` | Deterministic arithmetic is performed on supplied values. This alone does not establish that the equation or inputs are empirically valid. |
| `rule_screen` | A stated rule or heuristic filters, scores, or ranks candidates; it is not a calibrated probability or independent adjudication. |
| `literature_lookup` | A bounded lookup or metadata search over named references or a local corpus; absence from that scope is not novelty. |
| `planning_template` | Illustrative configuration, plan, or template; it has not necessarily been executed or shown feasible. |
| `reference_spec` | A fixed catalogue, reference value, or protocol constant, not an inferred result. |
| `refused` | The requested result is unsupported and the implementation explicitly raises or declines to produce it. |

For mixed modules, each row names the specific surface and tier. A calculation
may therefore be `computed` while a nearby engineering claim remains a
`reference_spec` or `planning_template`.

## Current-checkout inventory

| Surface | Tier | What the code returns or does | Evidence ceiling / known limitation |
| --- | --- | --- | --- |
| [`apedp/applied_synthetic_bio.py`](../apedp/applied_synthetic_bio.py) — `CRISPRDesigner.design_guide_rna` | `rule_screen` | Uses GC content in fixed formulae for `on_target_efficiency`, `off_target_risk`, and `specificity_score`. | These are hand-set heuristics, not an assay-calibrated guide activity or specificity model. |
| [`apedp/applied_synthetic_bio.py`](../apedp/applied_synthetic_bio.py) — `CRISPRDesigner.predict_off_targets` | `reference_spec` | Returns fixed counts `12` (`predicted_off_targets`), `2/5/5` risk bins, and `"high"` confidence for every guide. | Does not inspect a reference genome or sequence. The output is not an off-target prediction and should not be used as one. |
| [`apedp/applied_synthetic_bio.py`](../apedp/applied_synthetic_bio.py) — `ProteinDesigner.design_enzyme` | `planning_template` | Returns illustrative values including activity `0.75`, stability `65 C`, and success probability `0.3`. | No sequence design, structure calculation, assay, or probability calibration is performed. |
| [`apedp/applied_synthetic_bio.py`](../apedp/applied_synthetic_bio.py) — `MetabolicPathwayOptimizer.optimize_pathway` | `planning_template` | Looks up a small fixed yield table, multiplies by `0.85`, and returns a fixed `titer_prediction_g_l` of `50`. | Host/product physiology is not simulated; the recovery factor and titer are assumptions, not computed predictions. |
| [`apedp/applied_biopharma.py`](../apedp/applied_biopharma.py) — `UniversalDrugTargetDiscovery.discover_targets` | `rule_screen` | Ranks a small static disease-to-target list by `0.7 * tractability + 0.3`; marks results `static_example`. | `omics_data` is not analyzed and passing it raises `NotImplementedError`. The ranking is neither target validation nor clinical evidence. |
| [`apedp/applied_biopharma.py`](../apedp/applied_biopharma.py) — `drug_repurposing` | `planning_template` | Returns a fixed “moderate” potential, “preclinical” evidence label, and suggested feasibility study. | Does not compare a drug against disease data, pathway evidence, or clinical-trial records. |
| [`apedp/applied_climate.py`](../apedp/applied_climate.py) — `ClimateModeling.project_temperature` | `reference_spec` | Selects one of five hard-coded 2100 temperature increments and derives risk labels. | This is a scenario lookup, not a climate projection model. `sea_level_rise_m` is the fixed multiplier `0.2 * temperature_increment`; no sea-level process is modeled. |
| [`apedp/applied_climate.py`](../apedp/applied_climate.py) — `DirectAirCapture.analyze_dac_technology`, `CarbonMineralization.analyze_mineralization` | `reference_spec` | Looks up hand-coded technology/mineral values; DAC removal potential and mineralization cost/scale are fixed outputs. | No source-specific uncertainty, process design, lifecycle analysis, or deployment model is run. |
| [`apedp/applied_neuromorphic.py`](../apedp/applied_neuromorphic.py) — `NeuromorphicChipDesigner`, `SpikingNeuralNetworkDesigner`, `MemristorDesigner`, `BrainComputerInterface`, `NeuromorphicAlgorithms` | `planning_template` | Returns example configurations and constants, including `energy_per_synaptic_event_fj: 10`, `accuracy_estimate: 0.92`, and `"1000x energy efficiency"`. | No hardware design, workload benchmark, safety evaluation, or comparative GPU measurement supports these outputs. |
| [`apedp/applied_fusion.py`](../apedp/applied_fusion.py) — `PlasmaPhysicsAnalyzer` | `computed` | Computes Lawson triple product, beta, and Debye length from supplied values. | Formula outputs are not a reactor design; the fixed beta reference is explicitly only a rough comparison. |
| [`apedp/applied_fusion.py`](../apedp/applied_fusion.py) — `FusionReactorDesigner.design_reactor` | `computed` | Computes a requested fusion/heating power ratio and labels it a power-balance ratio. | If fusion power is omitted, target output is substituted. No plasma transport or engineering model is run. |
| [`apedp/applied_fusion.py`](../apedp/applied_fusion.py) — `FusionReactorDesigner.optimize_confinement` | `refused` | Raises `NotImplementedError`. | A confinement optimizer is not implemented. |
| [`apedp/applied_fusion.py`](../apedp/applied_fusion.py) — `FusionFuelCycle.tritium_breeding_ratio`, `deuterium_availability` | `literature_lookup` | Returns fixed literature-typical values for a small named set. | Values are references, not a fuel-cycle or blanket simulation. |
| [`apedp/applied_electrical.py`](../apedp/applied_electrical.py) — per-unit/per-cycle calculators | `computed` | Performs closed-form arithmetic from caller-supplied electrical inputs. | The equations are not a grid- or device-level simulator; the outputs are limited to the stated calculation. |
| [`apedp/applied_materials.py`](../apedp/applied_materials.py) — `search_candidates` | `reference_spec` | Returns the module’s curated `REFERENCE_CANDIDATES` table. | A catalogue lookup is not a materials search, fitted posterior, or independent ranking. |
| [`apedp/evidence.py`](../apedp/evidence.py) — `rank` | `reference_spec` | Maps evidence labels through the fixed `_RANK` ordering. | The table defines an ordering only; it does not assess whether evidence was correctly classified. |
| [`apedp/ingestion/ingestion_academic_apis.py`](../apedp/ingestion/ingestion_academic_apis.py) — `search_crossref` | `literature_lookup` | Makes a bounded Crossref bibliographic-metadata query and records query, source URL, and retrieval time. | It is not full-text search, a complete literature corpus, a novelty determination, or legal advice. `INSTITUTIONAL_TARGETS` is a list of endpoints; listing an endpoint does not mean this module queries it. |
| [`apedp/ingestion/ingestion_web_equations.py`](../apedp/ingestion/ingestion_web_equations.py) | `literature_lookup` | Extracts equation-like source strings from web markup. | Extracted strings are lossy source text, not parsed, verified equations; macros and some markup are not represented. |
| [`apedp/patent_prior_art.py`](../apedp/patent_prior_art.py) | `literature_lookup` | Validates provenance and symbolically compares expressions in a caller-supplied JSONL corpus. | A corpus-relative match is not novelty, freedom-to-operate, or legal advice. This checkout has no bundled `data/patents/corpus.jsonl`, so an actual search over the advertised local corpus is unavailable here. |
| [`apedp_dyn/distributed/cluster_launcher.py`](../apedp_dyn/distributed/cluster_launcher.py) | `planning_template` | Validates a job specification and renders Slurm/Kubernetes job artifacts. | Rendering a script or manifest does not submit a job, contact a scheduler, provision storage, or prove that the plan is deployable. |
| [`apedp_dyn/sandbox/finance_l2_demo.py`](../apedp_dyn/sandbox/finance_l2_demo.py) — `epistemic_position_sizing` | `rule_screen` | Maps epistemic status to fixed position sizes. | The size is not optimized from expected returns, covariance, fees, or holdout performance. Do not interpret it as a financial recommendation. |
| [`apedp_dyn/sandbox/execution_cost_models.py`](../apedp_dyn/sandbox/execution_cost_models.py) — fixed-plus-slippage, Almgren–Chriss, and queue models | `computed` | Computes each implemented formula from the fields it actually reads. | The audit recorded ignored advertised inputs: volumes for fixed-plus-slippage, spreads for Almgren–Chriss, and prices for the queue model. Confirm parameter sensitivity before using the outputs as execution estimates. |
| [`apedp_dyn/gates/scaling_invariance.py`](../apedp_dyn/gates/scaling_invariance.py) — `check_with_pi_groups` | `rule_screen` | Applies the implemented invariance checks to its available inputs. | The audit recorded that `pi_groups` is not consumed and the bundled gate has no regimes; a supplied group is not evidence that Buckingham-Pi analysis was performed. |
| [`apedp_dyn/robotics/cbf_certificate.py`](../apedp_dyn/robotics/cbf_certificate.py) — `certified_cbf_guard` | `computed` | Solves the implemented point-state safety-filter problem. | The audit recorded that `headings` is ignored. The resulting point-mass certificate does not establish safety for a unicycle with heading-dependent dynamics. |
| [`apedp_dyn/epistemic/shape_inference.py`](../apedp_dyn/epistemic/shape_inference.py) — `infer_shape_prior` | `rule_screen` | Produces a shape prior from the subset of inputs used by the implementation. | The audit recorded three accepted knobs that do not affect the result; verify the call surface before treating those settings as active evidence. |
| [`apedp_dyn/quantum/sparrow_photonics.py`](../apedp_dyn/quantum/sparrow_photonics.py) — `simulate_type2_fusion` | `computed` | Computes visibility and success probability from the photonic specification and `tau_offset`. | The audit recorded that `input_state` is ignored; the function does not simulate the supplied quantum state. |
| [`apedp_dyn/sandbox/eseds_dynamical.py`](../apedp_dyn/sandbox/eseds_dynamical.py) — `run_eseds_trap` | `computed` | Executes the implemented trap workflow. | The audit recorded that `r`, `K`, `N0`, and `bootstrap_n_boot` are not forwarded to the underlying operation; callers must not assume these arguments alter the run. |
| [`apedp_dyn/qed/referee.py`](../apedp_dyn/qed/referee.py) — `ablation` | `computed` | Runs the implemented ablation calculation. | The audit recorded that the supplied `coefficients` argument is discarded; the returned ablation is not conditioned on that argument. |
| [`apedp_dyn/audio/spectrogram_transport.py`](../apedp_dyn/audio/spectrogram_transport.py) — `predict_next_frame` | `computed` | Advances a spectrogram by first-order upwind advection and projects magnitudes to nonnegative values. | `order` is retained for interface compatibility and is intentionally unused. Validation described in the module is on synthetic chirps, not a broad real-audio corpus. |

## Static-table and scope-scan findings

The changelog records a prior static-table census of 11 `apedp` modules and 17
`apedp_dyn` modules. Most entries were registries, catalogues, or step ladders,
not defects; the two specifically named predictive-looking returns were
`apedp/applied_materials.py::search_candidates` (`REFERENCE_CANDIDATES`) and
`apedp/evidence.py::rank` (`_RANK`). It also records two census false positives:
`apedp_dyn/grammar/candidate_ir.py::base_provenance` returns state from a
candidate, while `apedp_dyn/vqc/native.py::last_failure` and `tallies` expose
native-module state.

The same changelog records a narrower predictive-name scan with two findings:
the two static-table returns above. It explicitly does not detect constants
under non-predictive names, values assembled by helpers, or tables accessed
with computed keys.

**Reproducibility limitation:** the census and scan artifacts described in the
changelog (`scripts/static_table_census.json`,
`scripts/scope_scan_baseline.json`, `scripts/audit_static_tables.py`, and
`scripts/scope_scan.py`) are absent from this checkout. Therefore the historic
counts above are attributed to the changelog and are not represented as a
freshly rerun census. This page is a curated inventory of the concrete surfaces
identified above, not a claim that every module in both packages has been
re-audited in this checkout.

## Vendored-code boundary

The changelog's earlier `apedp_dyn` census named five vendored modules that
were excluded from its first-party static-table gate:

1. `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src/qualtran/bloq_report_card.py`
2. `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src/qualtran/qlt_ir/_examples/`
3. `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src/qualtran/surface_code/flasq/volume_counting/`
4. `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src/qualtran/surface_code/ui/`
5. `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/scoreinfra/reference.py`

These are third-party/vendor surfaces, not claims about first-party model
quality. Vendored code is excluded from this curated inventory.

## How to maintain this page

When a scaffold-like surface is added or changed, update the row with its
source path, evidence tier, actual output behavior, and the strongest known
limitation. Do not turn a reference value into a prediction by changing its
label. A new audit should run source scans, read each finding, preserve
false-positive and exclusion reasons, and record its machine-readable baseline
and tested revision before describing the inventory as exhaustive.

The surrounding module documentation and scoped regression tests remain the
authority for details and behavior. This inventory is not a replacement for
those tests, domain validation, or independent scientific review.

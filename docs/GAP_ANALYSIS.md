# Verified project gaps and evidence status

**Reviewed:** 2026-11-06
**Scope:** MMLM measurement and data, symbolic-search robustness, prior-art coverage,
governance re-signing, and gap-document freshness.
**Historical material:** the previous snapshot is preserved in
[`GAP_ANALYSIS_HISTORICAL_2026-09-18.md`](GAP_ANALYSIS_HISTORICAL_2026-09-18.md).
Its numerical counts and status labels are not current evidence.

This is a status summary, not a claim that the project has completed independent
scientific validation. Statements below distinguish implemented mechanisms,
measured synthetic results, and dependencies that require external data,
permissions, or authorized signers.

## Status matrix

| Area | Verified status | Remaining gap / dependency |
|---|---|---|
| WSP / GF(2) multiplication | The recursive polynomial Karatsuba factors exactly reconstruct the degree-12 multiplication tensor at rank 54, versus exact schoolbook rank 144. The recorded packed-bit Python evaluator medians are 5.409 μs vs 13.273 μs per multiply (1,024 seeded operands, 11 samples). A simple unshared linear-circuit estimate is 513 XORs vs 351 for schoolbook. These are measurements of this evaluator and an XOR estimate, not compiled or reversible circuit costs. | Rank 54 is a certified upper bound, not a proof of minimum bilinear rank. Compile both exact decompositions to a common circuit model and measure actual gate cost, depth, and runtime before making circuit or quantum-performance claims. |
| MMLM H2 streaming retention | Implemented as B0 dynamical-only, sequential next-token accuracy, compared with the frozen 0.70 criterion. The default synthetic run (seed 42) observed 50/125 correct (0.4000), below criterion. The operational definition is narrow; it is not a general long-context or key/value-retention test. | No suitable external MMLM corpus is bundled or evaluated. The new dataset interface checks split disjointness and records source, license, retrieval date, and hashes, but does not independently authenticate the source or validate licensing. |
| MMLM H5 quality and efficiency | Implemented as descriptive comparisons on declared quality and efficiency metrics. B0 or C0 must win both majority-based axes against each baseline to meet the registered outcome rule. The default synthetic run (seed 42) did not meet the comparison rule. | Results are a single configured run unless repeated; neither outcome is a statistical test. Hardware-sensitive efficiency measurements and any externally supplied data require their environment and provenance to be reviewed. |
| MMLM validation | The benchmark supports caller-supplied disjoint train/dev/test text splits. Repeated seeds report initialization variability on the same configured split. | Repeated seeds do not establish dataset-to-dataset generalization. Independent, relevant, appropriately licensed datasets and a documented evaluation protocol remain external inputs. |
| Symbolic-search robustness | The generated record `benchmarks/search_engine_robustness.json` covers 12 conditions: 3 seeds × 2 training-scaled noise levels × 2 positive input ranges, 144 planted-law trials. Extended hierarchical recovery was 135/144; extended beam recovery was 102/144. Noise scaling, search, and fitting use training rows; confirmation rows only score the selected candidate. | This is synthetic evidence. Within each condition, train and holdout use the same input interval, so it is not an extrapolation test. It does not show performance on independent real datasets, arbitrary equation families, phase-shifted or off-grid periodic laws, or external data. See [`SEARCH_ENGINE_GAPS.md`](SEARCH_ENGINE_GAPS.md). |
| Literature and patent prior art | Academic lookup is provenance-bearing metadata search, not full-text search or proof of novelty. The patent prior-art loader requires source and retrieval provenance for real records and distinguishes unsearchable claims from no match. | No patent corpus is bundled under `data/patents/`. Corpus selection, access, licensing, refresh cadence, and record-level source provenance must be resolved before claiming coverage. A corpus-relative no-match remains no evidence of novelty or freedom to operate. |
| Governance manifest | The read-only ceremony status check confirms the canonical-spec disk hash matches its Python pin. The governance manifest remains sealed to an older hash; status is fail-closed red. The current manifest has two valid signatures against quorum threshold two, while `auditor_formal_03` is awaiting re-signature. | Re-signing is an external trustee action. Authorized trustees and their signing keys are required; this repository must not generate substitute keys, alter the roster, or bypass the ceremony. Until trustees complete the authorized process, the governance blocker remains open. |
| JetFlow video evidence | The fluid matched comparison is one synthetic seed. The collision/occlusion preregistration has three synthetic confirmation seeds; all confirm the negative transfer/domain-scoping result. | No independent, licensed real-video evaluation is present. The synthetic collision result must remain a known failure-domain result, not be generalized into real-world video evidence. |
| Gap documentation | This document replaces the stale status summary with an evidence-scoped matrix; the prior dated snapshot is retained and explicitly marked historical. | Re-review this matrix when datasets, corpus permissions, trustee signatures, or benchmark evidence change. |

## Measurable release gates

These are evidence requirements for stronger release claims; they do not amend any
frozen hypothesis or threshold.

| Claim area | Required evidence before upgrading a claim |
|---|---|
| GF(2) optimization | Every candidate reconstructs the complete tensor exactly. Report bilinear rank, linear XOR estimate, compiled-circuit gate count/depth, and measured runtime separately against schoolbook on the same declared compiler and hardware. A minimum-rank claim additionally requires a proof, not only failure of a bounded search. |
| JetFlow video | Preregister at least one independently sourced, appropriately licensed real-video dataset for each claimed domain (fluid and collision-like); split by source scene/video, not adjacent frames. Freeze at least five training/evaluation seeds, architecture and parameter budget (within 10%), compute budget, primary metrics, and uncertainty rule before evaluation. Report all per-seed and per-domain wins/losses. Broader claims also require an independent replication. |
| MMLM generalization | Evaluate at least two independent, appropriately licensed corpora with document/source-disjoint train/dev/test splits and at least five frozen seeds per corpus under matched resource budgets. Report per-corpus results and uncertainty, not only pooled averages. Keep the existing frozen H2 metric/0.70 criterion unchanged; any revised hypothesis must separately preregister long-context retrieval with held-out key/value facts, controlled distractor lengths and key positions through the declared context window, and per-length accuracy. |
| Prior-art coverage | Name each permitted, refreshable source, jurisdiction, searchable record types, coverage dates, retrieval date, and provenance for every result. Report “no match found in these sources” rather than “novel” or “clear.” |
| Governance | Verify the manifest and ceremony readiness, then obtain signatures only from the authorized trustees. Until then the governance gate remains blocked. |
| Release record | Preserve commands, commit identifiers, seed lists, data hashes and license/provenance records, negative results, known failure domains, and independent replication status in the gap analysis before changing headline claims or the system rating. |

## Interpretation and follow-up

The code changes close implementation gaps for the specified MMLM measurements
and broaden the synthetic search sweep. They do not close the independent-data,
licensing, or trustee dependencies. Any future report should identify the exact
dataset, split, source and license evidence, benchmark protocol, and whether a
result is synthetic, externally sourced, or independently audited. For governance,
only the authorized ceremony can resolve the outstanding signature state.

The Karatsuba comparison can be reproduced with
`python benchmarks/run_gf2_karatsuba_comparison.py`; its committed result is
[`benchmarks/gf2_karatsuba_comparison.json`](../benchmarks/gf2_karatsuba_comparison.json).
Its runtime is specific to the packed-bit Python evaluator and must not be
presented as quantum hardware performance.

The detailed protocols and caveats are maintained in
[`DOCUMENTATION.md`](DOCUMENTATION.md),
[`SEARCH_ENGINE_GAPS.md`](SEARCH_ENGINE_GAPS.md), and the MMLM benchmark tests.

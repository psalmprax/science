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
| MMLM H2 streaming retention | Implemented as B0 dynamical-only, sequential next-token accuracy, compared with the frozen 0.70 criterion. The default synthetic run (seed 42) observed 50/125 correct (0.4000), below criterion. The operational definition is narrow; it is not a general long-context or key/value-retention test. | No suitable external MMLM corpus is bundled or evaluated. The new dataset interface checks split disjointness and records source, license, retrieval date, and hashes, but does not independently authenticate the source or validate licensing. |
| MMLM H5 quality and efficiency | Implemented as descriptive comparisons on declared quality and efficiency metrics. B0 or C0 must win both majority-based axes against each baseline to meet the registered outcome rule. The default synthetic run (seed 42) did not meet the comparison rule. | Results are a single configured run unless repeated; neither outcome is a statistical test. Hardware-sensitive efficiency measurements and any externally supplied data require their environment and provenance to be reviewed. |
| MMLM validation | The benchmark supports caller-supplied disjoint train/dev/test text splits. Repeated seeds report initialization variability on the same configured split. | Repeated seeds do not establish dataset-to-dataset generalization. Independent, relevant, appropriately licensed datasets and a documented evaluation protocol remain external inputs. |
| Symbolic-search robustness | The generated record `benchmarks/search_engine_robustness.json` covers 12 conditions: 3 seeds × 2 training-scaled noise levels × 2 positive input ranges, 144 planted-law trials. Extended hierarchical recovery was 135/144; extended beam recovery was 102/144. Noise scaling, search, and fitting use training rows; confirmation rows only score the selected candidate. | This is synthetic evidence. Within each condition, train and holdout use the same input interval, so it is not an extrapolation test. It does not show performance on independent real datasets, arbitrary equation families, phase-shifted or off-grid periodic laws, or external data. See [`SEARCH_ENGINE_GAPS.md`](SEARCH_ENGINE_GAPS.md). |
| Literature and patent prior art | Academic lookup is provenance-bearing metadata search, not full-text search or proof of novelty. The patent prior-art loader requires source and retrieval provenance for real records and distinguishes unsearchable claims from no match. | No patent corpus is bundled under `data/patents/`. Corpus selection, access, licensing, refresh cadence, and record-level source provenance must be resolved before claiming coverage. A corpus-relative no-match remains no evidence of novelty or freedom to operate. |
| Governance manifest | The read-only ceremony status check confirms the canonical-spec disk hash matches its Python pin. The governance manifest remains sealed to an older hash; status is fail-closed red. The current manifest has two valid signatures against quorum threshold two, while `auditor_formal_03` is awaiting re-signature. | Re-signing is an external trustee action. Authorized trustees and their signing keys are required; this repository must not generate substitute keys, alter the roster, or bypass the ceremony. Until trustees complete the authorized process, the governance blocker remains open. |
| Gap documentation | This document replaces the stale status summary with an evidence-scoped matrix; the prior dated snapshot is retained and explicitly marked historical. | Re-review this matrix when datasets, corpus permissions, trustee signatures, or benchmark evidence change. |

## Interpretation and follow-up

The code changes close implementation gaps for the specified MMLM measurements
and broaden the synthetic search sweep. They do not close the independent-data,
licensing, or trustee dependencies. Any future report should identify the exact
dataset, split, source and license evidence, benchmark protocol, and whether a
result is synthetic, externally sourced, or independently audited. For governance,
only the authorized ceremony can resolve the outstanding signature state.

The detailed protocols and caveats are maintained in
[`DOCUMENTATION.md`](DOCUMENTATION.md),
[`SEARCH_ENGINE_GAPS.md`](SEARCH_ENGINE_GAPS.md), and the MMLM benchmark tests.

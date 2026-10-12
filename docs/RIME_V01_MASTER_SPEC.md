# RIME v0.1 — Master Specification, Scientific Protocol & Rust Implementation Preamble

**Status:** FROZEN v0.1 (minimal viable system). Changes require a new version.
**Standing:** RIME is a structural-discovery ENGINE (unproven). APEDP-X stays
the independent referee. RIME proposes; APEDP-X adjudicates — never reversed.
**Prior art (not ours):** residual-driven discovery (RED), modular equation
decomposition (AI Feynman), evolutionary symbolic regression (PySR), ridge/
distributed lags, lead–lag literature. RIME's claim must come from verified
results, not architecture.

## 1. Mission

Discover explicit, executable mechanism equations (esp. financial dynamics)
by diagnosing WHY current explanations fail and revising STRUCTURE (not just
coefficients), then submitting every candidate to APEDP-X's frozen referee.

## 2. Scientific boundaries

- MAY claim: predictive recovery on held-out data; MULTI vs UNIQUE status as
  returned by the referee; abstention where evidence is insufficient.
- MUST NOT claim: causality from graph edges (call them candidate predictive
  dependencies); novelty without literature check; promotion without frozen
  confirmation; real-market validity from synthetic wins.

## 3. Architecture

```
Python research layer (data, diagnostics, plots, PySR/SINDy adapters)
        │ batch-oriented FFI (PyO3/maturin), minimal crossings
RIME Core — Rust (graphs, rewrites, evaluation, records, parallel search)
        │ CandidateProposal records only
APEDP-X referee (frozen gates; RIME cannot read or alter them)
```

Reuse: `apedp-kernel` parser (expression grammar), units (dimensional check),
numeric (regression/SVD), rayon (parallelism).

## 4. Mechanism graph (v0.1 minimal)

- Node types: `Input(var)`, `State(var, init)`, `Const`, `Op(Add/Mul/Neg)`,
  `Fn(Tanh/Sign/Abs/Lag(k))`, `Output`.
- Edges: typed data-flow; `Lag(k)` edges carry delay $k \in \{1,2,5,10,20\}$.
- Semantics: synchronous discrete-time evaluation, topological order,
  NaN-poisoning (any NaN input → NaN output, counted not hidden).
- Serialization: canonical string form + SHA-256 id; deterministic from
  (graph, seed).

## 5. Structural rewrites (v0.1 frozen set — exactly these five)

1. `AddLagEdge`: attach Lag(k) copy of an existing input (k from frozen set).
2. `AddInteraction`: multiply two existing nodes (bilinear term).
3. `AddState`: introduce persistent state node with decay (adds memory).
4. `WrapNonlinearity`: wrap a node in Tanh (bounded response).
5. `PruneEdge`: remove lowest-|coefficient| edge (complexity control).
No other rewrites in v0.1. New rewrites = new RIME version.

Output semantics (frozen): ops 1–4 make the new node the candidate output
(the candidate under test IS the new expression); `AddState` likewise
retargets output to the memory cell. `PruneEdge` edits in place (node →
Const(0.0)) and preserves the output index so downstream references stay
valid.

## 6. Residual diagnosis (competing remedies, never single-cause)

Classifier maps residual patterns to CANDIDATE remedy sets:
- delayed error autocorrelation → {AddLagEdge, AddState} (NOT auto-memory;
  also consider missing-variable and regime-change notes, recorded).
- heteroskedastic bursts → {WrapNonlinearity}.
- bilinear structure in residual×feature → {AddInteraction}.
- flat/white residuals → {PruneEdge} or STOP (no structure justified).
Each diagnosis emits ≥2 competing proposals or an explicit STOP record.

## 7. Candidate generation

- Seeds: grammar-exhaustive small forms (v0.19-style, ≤60) + rewrite
  expansions (depth ≤ 2 from each seed).
- Parameters: ridge fit on development data ONLY (λ frozen per run).
- Records: (graph_id, parent_id, rewrite_op, seed, dev_score) — immutable.

## 8. Selection & search (frozen before testing)

$$S(G') = \overline{\Delta\mathrm{MSE}}_{\mathrm{dev}} - \lambda_c \cdot |E'|$$

$\overline{\Delta\mathrm{MSE}}$ = mean loss reduction vs parent across dev
environments; $|E'|$ = edge count; $\lambda_c$ frozen per experiment.
Budget: ≤200 candidates/run; STOP when no proposal improves dev score or
budget exhausts. $S$ prioritizes experiments; promotion needs referee gates.

## 9. Competing hypotheses

Every accepted rewrite keeps its runner-up as a rival; observational
equivalence (behavioral corr ≥ 0.7 on held-out) → MULTI_HYPOTHESIS packet
to the referee. No silent winner-picking.

## 10. Statistical safeguards

Null controls (shift/permutation), temporal holdouts (chronologically later,
untouched until confirmation), Bonferroni over grammar size, leakage ban
(no confirmation influence on search — violation voids the run), frozen
protocols with prereg-first commits.

## 11. APEDP-X integration

RIME emits `CandidateProposal{graph_id, equation, dev_evidence}`; referee
returns NO_RECOVERY / PREDICTIVE_RECOVERY / MULTI_HYPOTHESIS /
UNIQUE_IDENTIFICATION (v0.20 taxonomy). RIME may request re-evaluation with
new evidence, never rule changes.

## 12. Rust implementation (Stage 1 scope)

- `rime/graph.rs`: typed DAG, topological eval, canonical serialization, hash.
- `rime/rewrite.rs`: five frozen ops as pure functions (old graph → new graph).
- `rime/eval.rs`: batch evaluator (rayon), ridge fit via kernel numeric.
- `rime/record.rs`: immutable candidate records (id, parent, op, seed, score).
- PyO3: batch `propose(dev_data) -> Vec<CandidateRecord>` + `evaluate(graph,
  data) -> scores`. Batch-oriented FFI.
- Determinism: same (graph, seed) → same bytes, enforced by test.

## 13. Benchmark protocol

Vs APEDP-X proposers, PySR, ≥1 neural/hybrid baseline; matched compute;
metric = incremental confirmed-discovery rate + false-promotion/abstention
rates + cost per confirmed discovery. Configurations: RIME-Classical first;
RIME-Hybrid (neural proposer adapter) only after Classical baselines exist.

## 14. Testing

`cargo test`: unit (graph eval, each rewrite, serialization round-trip,
determinism), property (eval purity, rewrite validity preservation),
synthetic ground-truth (recover planted 2-lag mechanism; abstain on null).
Python: integration (bridge round-trip, referee submission).

## 15. Roadmap & acceptance

- Stage 1 (this version): graph + 5 rewrites + evaluator + records; passes
  all cargo tests incl. planted-mechanism recovery and null abstention.
- Stage 2: residual diagnosis + parallel search + dev scoring (new spec).
- Stage 3: APEDP-X integration + external benchmarks (new spec).
No quantum, no transformer proposer in v0.1 (explicitly deferred).

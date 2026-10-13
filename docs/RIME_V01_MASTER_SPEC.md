# RIME v0.1 — Master Specification, Scientific Protocol & Rust Implementation Preamble

**Status:** FROZEN v0.1.1 (minimal viable system, Stage-1 audit remediated).
Changes require a new version: see §16 for the v0.1 → v0.1.1 corrections and the
CHANGELOG entry that records their rationale.
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
- **`State` semantics (v0.1.1):** a persistent memory cell with input series $u$,
  decay $d$, and prior state $h_{-1} = \text{init}$:
  $$h_t = d\,h_{t-1} + u_t, \qquad h_0 = d\cdot\text{init} + u_0.$$
  NaN in $u$ poisons the state forward exactly like every other node; the scorer
  skips and counts those rows rather than imputing them.
- **Arena order is topological and enforced:** every operand must be a
  *strictly earlier* node index. `Graph::validate` rejects a non-topological or
  non-finite arena instead of evaluating it. Canonical structure strings (and
  therefore SHA-256 graph ids) are unchanged for every graph that existed in
  v0.1, so content-addressed records remain stable.
- Semantics: synchronous discrete-time evaluation, topological order,
  NaN-poisoning (any NaN input → NaN output, counted not hidden).
- Serialization: canonical string form + SHA-256 id; deterministic from
  (graph, seed).

## 5. Structural rewrites (v0.1 frozen set — exactly these five)

1. `AddLagEdge`: attach Lag(k) copy of an existing input (k from frozen set).
2. `AddInteraction`: multiply two existing nodes (bilinear term).
3. `AddState(decay_num, decay_den)`: introduce a persistent state node with
   rational decay $d = \text{num}/\text{den} \le 1$ (adds memory). The state's
   input is the current output expression and $\text{init} = 0$; v0.1.1 pushes a
   real `State` node rather than expanding to $d\cdot out_{t-1} - out_t$.
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
  **Implemented in v0.1.1:** the candidate is scored on its fitted readout
  $a\,o_t + b$ and the two readout degrees of freedom are charged, so in-sample
  fitting alone cannot buy a promotion.
- Seed catalogue (v0.1.1, `rime/seeds.rs`): deterministic, deduplicated by
  content-addressed graph id, capped at 60 forms, emitted in family priority
  order — levels, delayed levels, differences, momentum across horizons,
  cross-input structure, both lead-lag interaction orientations, memory cells on
  the decay grid $d \in \{0.5, 0.7, 0.8, 0.9, 1.0\}$, then wrappers. Levels and
  differences use the full frozen lag set; interaction families use
  $k \in \{1,2,5\}$ so every family fits the budget at two inputs.
- Records: (graph_id, parent_id, rewrite_op, seed, dev_score) — immutable.

## 8. Selection & search (frozen before testing)

$$S(G') = \overline{\Delta\mathrm{MSE}}_{\mathrm{dev}} - \lambda_c \cdot |E'|$$

$\overline{\Delta\mathrm{MSE}}$ = mean loss reduction vs parent across dev
environments; $|E'|$ = edge count; $\lambda_c$ frozen per experiment.
Budget: ≤200 candidates/run; STOP when no proposal improves dev score or
budget exhausts. $S$ prioritizes experiments; promotion needs referee gates.

### 8.1 Calibrating $\lambda_c$ (v0.1.1 clarification)

The shape of $S$ is frozen; what v0.1.1 fixes is that $\lambda_c$ must be chosen
on the scale at which a dev MSE actually moves. Two admissible calibrations:

- **sample-size aware (default for proposals):** $\lambda_c = \gamma\,\mathrm{var}(y)/n_{\text{dev}}$
  with $\gamma \ge 1$ (1.5 recommended). $\mathrm{var}(y)/n_{\text{dev}}$ is the
  sampling-noise scale of a dev MSE, so a candidate must beat what fitting a
  handful of parameters to that many rows can achieve by luck alone.
- **fraction of variance (legacy):** $\lambda_c = \rho\,\mathrm{var}(y)$. Simple,
  but blind to $n$: it is far too permissive on short series.

**Selection charge (spec §10).** Because the search examines $M$ candidates, the
charged degrees of freedom are
$|E'| + 2 + 2\ln M$ — the graph edges, the two readout coefficients, and an
EBIC-style multiplicity term in which $M$ is the preregistered budget, so it does
not depend on the data. The charged count is reported with every proposal
(`dev_evidence.edges`, `readout_params`, `selection_dof`, `complexity`).

**Reference model.** Candidates are ranked against the mean predictor (the
$\mathrm{var}(y)$ baseline, spec §6 "flat/white residuals → STOP"), so scores are
comparable across candidates with different parents and depths.

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
- PyO3: `rime_search_json(data, target, config, variable_names) -> packet`
  (batch propose), `rime_evaluate_json(graph, data, target, ...) -> fitted score`,
  `rime_rewrite_json(graph, op) -> graph`, and
  `rime_evaluate_series_json(graph, data) -> series` for proposers that must emit
  a signal. Batch-oriented FFI.
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
  **Delivered as v0.1.1**, with the §16 audit corrections and the §16
  measured-standing caveat (planted small-sample recovery, not real-market
  validity).
- Stage 2: residual diagnosis + parallel search + dev scoring (new spec).
  **Status: partial.** §6 residual diagnosis (five residual archetypes driving
  remedy sets) is **not implemented** — nothing in the engine classifies residuals;
  `eval.rs` computes residual statistics only to score candidates. Parallel search
  and dev scoring exist (`search.rs`, `seeds.rs`, Rayon batches) but as Stage-2
  scaffolding; they add no rewrites.
- Stage 3: APEDP-X integration + external benchmarks (new spec).
  **Status: partial.** The PyO3 proposer adapter
  (`apedp_dyn/proposers/rime_proposer.py`) emits §11-style packets, but no referee
  gate code consumes them yet (by design — see rule 1), and the §13 benchmark
  protocol is unexecuted: `benchmarks/rime_vs_neural.py` covers neural/linear
  baselines only; there is no PySR run and no matched-compute comparison against
  the existing APEDP-X proposers.
No quantum, no transformer proposer in v0.1 (explicitly deferred).

## 16a. v0.1.2 reliability & evidence-integrity notes

The v0.1.2 review specification asked for four reliability guarantees. Where each
is enforced:

- **Multi-environment correctness (§1).** Environments are validated before any
  scoring: `MIN_ENV_FINITE_ROWS = 15` — environments below it are `TooFewRows`
  and are excluded from every aggregation; structurally inconsistent
  environments (ragged lengths or a column-count change) are `Malformed` and
  excluded; if nothing is usable the search abstains with an explicitly infinite
  reference rather than a degenerate score. Missing observations are skip-and-
  count against both output and target, never imputed. Per-env statuses are
  reported in the packet (`env_statuses`), and the worst-environment result is
  reported next to the aggregate (`worst_env_mse`, `worst_env_passed`). Four
  regression tests: heterogeneous three-env worlds, short+malformed exclusion,
  all-unusable abstention, and gapped (NaN-heavy) environments.
- **Pruning consistency (§2).** `PruneEdge` zeroes in place; the semantics are
  pinned by `rust/apedp-kernel/tests/rime_prune_semantics.rs`: the zeroed edge
  contributes exactly zero, `edge_count`/DoF accounting is unchanged by pruning
  (it is a zeroing, not a removal — the packet's `charged_dof` formula is stated
  and tested), the content address changes so a pruned candidate cannot be
  confused with its parent, predictions are byte-reproducible across
  serialization, pruning touches no other node or the output index, and a fully
  zeroed graph scores an intercept-only readout that cannot outscore the null.
- **Explicit search limits (§3).** `structural_rewrites_never_introduce_seed_only_operators`
  walks every rewrite applied to every seed and pool member at depth 3 and fails
  if a rewrite pushes a NEW `Sign`/`Abs` node (wrapping an inherited one is
  composition, not introduction, and is tested to be reachable ONLY from seeds);
  the off-grid world pin that a 0.87-decay state is never rendered as
  `state_0.8700` (nearest-grid approximation, R² strictly < 1); budget-1 and
  empty-seed runs abstain with zero evaluated candidates instead of junk scores.
- **Benchmark provenance (§4).** `benchmarks/rime_vs_neural_results.json`
  carries a `provenance` block (benchmark version, git revision + dirty flag,
  kernel version, config, seed policy, timestamp, feature-availability rule)
  and a `determinism_check` (the deterministic RIME path is re-run and compared).
  Recovery accuracy, prediction quality, compute cost and complexity remain
  separate fields.
- **APEDP-X boundary (§5).** Python tests assert the 64-hex graph id recorded by
  Rust is the proposal's identity and that the packet's equation matches it;
  malformed graphs (self-referencing arena, out-of-frozen lag) raise explicitly
  at the kernel rather than emitting numbers; the proposal contract lists every
  field adjudication needs. No confirmation feedback channel exists: the referee
  modules contain no RIME references and RIME imports nothing from the referee
  (verified structurally; rule 1 remains architectural).

## 16b. v0.1.2 static-grammar extension & scale-invariant readout (QEC round)

The QEC repair round surfaced a boundary that turned out to be TWO defects,
one in the numerics and one in the proposal surface:

- **Scale-dependent ridge (FIXED — engine numerics).** `fit_readout` added a
  fixed literal `ridge = 1e-9` to `Σ(o−ō)²`. For a regressor whose raw second
  moment is far from unit scale — exactly what an algebraic transform column
  (p² ≈ 1e-8) is — the ridge exceeded the entire signal energy (here ~30×) and
  silently zeroed every algebraic-scale fit. The QEC attribution "RIME's lag/state
  grammar is the wrong instrument for static scaling laws" was therefore
  PARTLY WRONG: the grammar always contained the self-interaction rewrite
  (`p·p`); the readout could not fit it. The ridge is now RELATIVE to the
  regressor's raw second moment (`ridge·Σ o²`), which is identical for
  unit-scale regressors (all pre-v0.1.2 tests) and correct at extreme scales.
  The temporal-grammar side now recovers the QEC threshold law
  (`test_dyn_ga_qec.py` asserts holdout R² > 0.99).
- **Static search mode (`config["static"]`, OFF by default).** Removes temporal
  primitives from the grammar entirely — no `Lag`, no `State`, in seeds and
  rewrites alike (`Graph::is_temporal` filter). Use for static laws over an
  order-structured axis where grid-order fitting must be UNREPRESENTABLE
  rather than merely gated. No gate, threshold or preregistered value changes;
  the default grammar is byte-identical to v0.1.1.
- **Algebraic input transforms (Python layer).** `RimeProposer(
  algebraic_transforms=[...])` appends canned transforms (`sq`, `cub`,
  `ssqrt`, `slog`) of the first `max_transform_inputs` features as extra
  candidate columns, recomputed identically inside `compute_score` so
  proposals stay causally usable on unseen frames. Domain guarded
  (NaN-preserving); unknown kinds are rejected at construction.
- **Matched-compute harness with a grammar-coverage axis.**
  `benchmarks/matched_compute.py` scores every method per dataset family with
  a preregistered protocol: abstention OUT of declared coverage is correct
  behaviour; an emitted-but-failed proposal is a failure on ANY family; static
  worlds additionally require reversal-invariance of the holdout score so a
  grid-order tracker cannot masquerade as mechanism (scored `index_fit`).
  Measured (v0.1.2 fresh run, artifact `benchmarks/matched_compute_results.json`,
  5 repeats, verdicts stable, full environment metadata, wall-budget
  utilization recorded per cell):
  RIME-temporal recovers 4/5 (3 sequential + the static threshold law; its one
  failure, static_log, is the same monotone-invalued world PySR handles),
  coverage-adjusted 4/5, accuracy on attempts 4/5. OLS-basis control recovers
  2/2 static laws. DLI, whose feature matrix contains only SAME-TIME
  interactions, is recorded with its measured grammar gap (delay-interactions
  are out of grammar despite the name). **PySR 2.7 ran a HEAD-TO-HEAD under the
  same serial-deterministic regime** (parallelism='serial', fixed seed, unbounded
  iterations with a binding 2-s timeout, one-time Julia warmup excluded, lag
  columns engineered for sequential worlds for information parity — PySR has no
  time-shift operator): recovered 3/5 (lag2, lead-lag interaction, static
  threshold law; r²=0.9974 even under the reversed diagnostic), failed static_log
  at this budget and is out of scope on the IIR state law. The honest conclusion
  at EQUAL 2-second serial budgets is COMPLEMENTARITY, not dominance: each
  grammar class recovers worlds the others miss — exactly the coverage-axis
  statement the harness was built to make rather than an arithmetic scorecard.

**Routing (resolved capability statement).** RIME is the instrument for
SEQUENTIAL mechanisms (lag, momentum, memory, lead-lag interaction) and — with
the ridge fix — quadratic/monomial static laws via self-interaction. Static
laws needing terms outside `{add, mul, tanh, sign, abs, sq, cub, ssqrt, slog}`
(e.g. exp, genuinely deep powers) remain OUT of grammar for every RIME mode;
route them to the algebraic-basis proposers, and let the referee adjudicate.

**Regression tests.** `rust/apedp-kernel/tests/rime_static_mode.rs` (static
recovery, no-transform can't claim exactness, temporal default unchanged);
`tests/test_dyn_rime_static_grammar.py` (end-to-end transform recovery,
holdout, metadata discipline, catalogue/domain safety);
`tests/test_dyn_ga_qec.py` (updated to the corrected condition);
`rime_regression.rs` rewritten against the current typed API.

## 16. v0.1.1 corrections (Stage-1 audit remediation)

A Stage-1 audit read the whole engine, ran its tests, and probed the evaluator
directly. Four defects were confirmed by execution; the bundled tests could not
catch any of them because each exercised only the one code path the defects did
not touch. Rationale and remediation are recorded in `CHANGELOG.md`.

| # | Defect | Confirmed behaviour before the fix | Correction |
|---|---|---|---|
| 1 | Evaluation cached one value per **node**, not per **(node, time step)**, so a node read directly *and* through `Lag` shared one slot | `x[t] - x[t-1]` on `[1,2,3,4]` returned `[NaN, 0, 0, 0]`; correct is `[NaN, 1, 1, 1]` | Series-at-a-time evaluation in arena order, one array per live node, released when its last consumer has read it; unreachable nodes skipped; non-topological arenas rejected |
| 2 | `AddState` was not a recursion: it built $d\cdot out_{t-1} - out_t$ (no state, flipped input sign) | all-ones input at $d=0.5$ returned `[NaN, -0.5, -0.5, -0.5]`; a decaying memory is `[1, 1.5, 1.75, 1.875]` | `Node::State` with $h_t = d\,h_{t-1} + u_t$ |
| 3 | No coefficient fitting: graphs were scored on their raw output | a planted $y = 0.001\,x_{t-2} + 5$ scored no better than the null | Ridge least-squares readout before scoring, two DoF charged, reported as `scale`/`intercept` |
| 4 | The null control charged 1.0 per edge while every MSE in the world under test was $\approx 0.08$, so nothing could ever be promoted — the test asserted a tautology | both bundled tests passed while proving nothing | Penalty on the sampling-noise scale, paired positive control, 24-world false-promotion bound, and a short-series multiplicity arm |

**Regression tests.** `rust/apedp-kernel/tests/rime_regression.rs` fails on the
pre-fix engine for defects 1, 2, 3 and 4b; `rime_stage1.rs` carries the paired
positive control; `tests/test_dyn_rime_proposer.py` covers the Python boundary.

**Not fixed, and known.**

- Multi-environment dev scoring ("mean loss reduction across dev environments",
  §8) is now offered BOTH ways: the pooled single-frame path (NaN boundary rows,
  skipped not imputed, length-weighted) remains the default for single-frame
  calls, and `rime_search_envs_json` / `RimeProposer(per_frame_envs=True)` score
  each environment separately under a frozen aggregation rule — `MeanPerEnv`
  (the literal §8 reading, equal weight per environment) or
  `MedianWorstGated` (median across environments plus a worst-environment
  improvement gate, so a mechanism that wins only where it is large is refused).
  The aggregation is fixed before scoring, reported in the packet, and never
  tuned per candidate; the dominance-trap behaviour has a regression test at
  both the Rust and Python layers.
- `PruneEdge` prunes to `Const(0.0)` in place; it cannot re-estimate any
  coefficient, so it is a structural delete, not the "lowest-|coefficient| edge"
  removal of §5. The search de-duplicates candidates by rendered expression, which
  contains the damage.
- `Sign` and `Abs` nodes are reachable only through the seed catalogue (no
  rewrite introduces them), and only for the first input at two or more inputs.
- Decay is searched over a discrete grid, not fitted, so a memory law whose decay
  is off-grid is approximated rather than recovered exactly.
- RIME remains a separate system from the Python proposers in the sense that the
  referee receives `CandidateProposal` objects; nothing in the referee reads RIME
  internals, by design.
- `search.rs` and `seeds.rs` are Stage-2 scaffolding built from the frozen
  primitives. They add no rewrite and no node type, and they are not part of the
  frozen v0.1 surface.

## 17. Conformance status vs the fifteen-section preamble (v0.1.1, updated for the v0.1.2 evidence & reliability round)

| § | Requirement | Status | Where |
|---|---|---|---|
| 1 | Mission (equations / predictive relationships in financial dynamics) | **implemented, engine-level** | §1; planted flows exercised in the test suite — note that *no real-market result exists* |
| 2 | Scientific boundaries (may / must not / unresolved) | **spec only, enforced by convention** | §2, §16; `AGENTS.md` §1.1; no runtime enforcement of abstention framing beyond §8 STOP |
| 3 | Architecture (Rust core, Python layer, independent referee) | implemented | `rust/apedp-kernel/src/rime/`, `apedp_dyn/proposers/rime_proposer.py`; referee untouched |
| 4 | Mechanism graph (nodes, delays, state transitions) | **implemented + audit-corrected** | §4, §16; `rime/graph.rs` (incl. validation, not present in §4's minimal statement) |
| 5 | Frozen rewrite system (five) | implemented; **PruneEdge deviates from §5** | `rime/rewrite.rs`; §16 notes the coefficient caveat |
| 6 | Residual diagnosis (archetype → remedy sets) | **NOT IMPLEMENTED** | scheduled Stage 2; only MSE/TSS residual stats exist, no archetype classifier or ≥2-competing-proposals signalling |
| 7 | Candidate generation & conversion into executable equations | implemented (v0.1.1 seeds + settled readout) | `rime/seeds.rs`, `rime/eval.rs`; multi-environment scoring now available per-env (`search_envs`, `MedianWorstGated`) alongside the pooled default (§16) |
| 8 | Scoring, penalties, budgets, stopping | **implemented with clarified calibration** | §8 / §8.1; sample-size-aware λ and 2·ln M selection charge added in v0.1.1 |
| 9 | Rivals & observational equivalence | **NOT IMPLEMENTED** | multi-hypothesis packets and corr ≥ 0.7 emission are unstarted |
| 10 | Statistical safeguards | partial → stronger | null controls, selection charge AND frozen env-validation rules (min-rows, malformed handling) in `rime/search.rs`; temporal holdouts and leakage bans remain the referee's, not yet wired to RIME proposals |
| 11 | APEDP-X integration (interfaces, statuses) | **partial — proposer side only** | spec-faithful packet via `rime_search_json`; no referee consumes `CandidateProposal` from RIME yet, and no re-evaluation-with-new-evidence path exists |
| 12 | Rust implementation spec (determinism, serialization, error handling) | implemented + determinism-tested | §12 lines 94–96; determinism tests in `rime/search.rs` and `rime/record.rs` |
| 13 | Benchmark protocol (PySR, existing proposers, neural baselines) | **partial — neural + linear + stress grid** | `benchmarks/rime_vs_neural.py` (--stress: noise × amplitude × on/off-grid decay × correlated distractor, 18 cells in the artifact); PySR and matched-compute-vs-existing-proposers runs still pending (Julia env — see `benchmarks/external_60_adapters.py`) |
| 14 | Testing (unit, property, ground truth, adversarial, integration) | mostly implemented; **adversarial⊂null arms** | `rime_stage1.rs`, `rime_regression.rs`, `test_dyn_rime_proposer.py` |
| 15 | Staged roadmap with acceptance criteria | partial; acceptance stages 2/3 open | §15 above |

---

**Measured standing vs baselines** (planted mechanisms only; updated after the
stress round). Beyond the base grid (recovery 5/5, OOS $R^2 \ge 0.9996$,
n = 60…3000, two fitted coefficients), the `--stress` grid (18 cells in the
artifact) sweeps noise (0.02 → 1.0), amplitude (0.5, 3.0) and memory decay
(0.80 on-grid, **0.87 deliberately off-grid**) with and without a ≈0.9-correlated
distractor input. RIME reported the right structure in **18/18 cells**;
hardest cells: `lag a=0.5, noise=1.0` → OOS $R^2$ 0.154 (ridge 0.136, MLP
negative); off-grid `state d=0.87` → 0.978 via the nearest grid member — the
documented discrete-decay approximation, not a fitted value. With a correlated
distractor the ridge baseline quietly swaps inputs without flagging it; RIME's
emitted equation names the input it used, which is auditable addressably.
Still **not** established: any real-market claim, PySR or existing-proposer
comparisons, or high-dimensional behaviour. RIME signals are
past-information-only; that is temporal hygiene, not a causal claim.

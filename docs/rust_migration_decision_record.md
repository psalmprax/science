# Rust-migration decision record — graph subsystems (2026-10-22)

Answer to "have we implemented all the graphs, and which should be in
Rust?", grounded against the repository and measured before deciding.

## 1. Inventory status (ground truth)

**Every graph in the external 13-graph matrix is implemented and
tested.** Beyond the matrix:

- The Level-1 ODE dependency screen the matrix marked "Planned" is now
  **implemented** (`apedp_dyn/de_arm/dependency_screen.py`, Python,
  `experimental`; dense Rust kernel as its mandatory escape hatch;
  committed 554790a, frozen-benchmark-measured).
- Five graph classes remain **deliberately unimplemented backlog**
  (recorded in `docs/graph_census_2026_10_22.md`, each with a gate):
  robotics proximity graphs (gate now measured — see below), mesh
  cotangent-Laplacian PDE discovery (gated behind the PDE arm), GP-tree→
  DAG sharing, 2D/3D tensor networks (PEPS/MERA), and a kernel
  reverse-mode AD tape (rejected by design — forward Taylor Jets).

So: "all of the graphs in scope" — yes; "every graph class" — no, and
each remaining one has a recorded activation gate, not a promise.

## 2. Corrections to the external analysis (source-verified)

1. **The 222× figure is real**: `docs/population_extrapolation_rime_ad_vs_ga.md`
   records 10.7 ms (native Rust DAG search) vs 2 384.8 ms (Python tree
   search) on identical candidate discovery — checked in the file.
2. **The "GP engine is Python" claim is half-wrong**: `genetic.rs` is
   already native Rust; its tree representation (`Box<Node>`) is a
   Rust-side structure. Unifying it with the arena DAG is a RUST-to-RUST
   structural upgrade, not a Python migration. Only `MultiVariableSRE`
   (Python AST trees) is a Python-side tree engine.
3. **"Rust is always faster" needs path discipline**: today's CBF
   profile showed the fastest fix for a Python loop was NUMPY
   VECTORIZATION, not Rust (behavior-identical masked pair computation);
   the Rust kernel matters only where millions of inner-loop evaluations
   dominate (the verified 222× regime, and new measured hot spots below).

## 3. Measured Python share in the new DE arm (before deciding migration)

`benchmarks/run_de_arm_vs_dense_benchmarks.py` fixture (Lorenz n=8000,
D=3): arm outcome 43.4 ms = Rust kernel dense path 29.6 ms + Python
screen 5.0 ms + Python mirror STRidge fits ≈ 8.8 ms → **Python share
31.8%** of the arm's wall time at once-per-candidate-batch frequency.

## 4. Decision record (rule: inner-loop millions-of-evals → Rust;
once-per-experiment referee/UI/provenance → Python)

| Rank | Subsystem | Decision | Basis |
|---|---|---|---|
| 1 | GP/SRE expression engine | **PORT DEFERRED (revision 2026-10-22, measured):** the tree-vs-arena instrumentation (`tests/sre_tree_vs_arena_profile.rs`, artifact `benchmarks/sre_tree_vs_arena_baseline.json`) gate goes TREE-FAVORED — on unshared GP trees the rime arena evaluates at **0.85× of the tree path** (tree 67.25 ms vs arena 78.75 ms on the SAME workload universe, 169 candidates × 60k rows ×2 loops, seed 20261022; re-measured 2026-10-22 on the current tip, recorded_at 1792679316 — the earlier same-seed run recorded_at 1792644945 measured 0.51× at tree 403 ms / arena 784 ms; both runs TREE-FAVORED, the absolute wall times and ratio drifted ~10× with machine state while parity stayed exact, so the gate conclusion is ratio-direction + parity + coverage, not the absolute figure), with bit-exact parity on the covered subset (0 mismatches) and only **10.1% semantics-identical conversion coverage** (Div-heavy random trees: tree `a/b`, Min/Max, IEEE powf and the unbounded transcendentals have no drift-free arena counterpart — tree `a−b` maps exactly as `Add∘Neg`, `Add∘Neg`, which is what parity is pinned on). An earlier Rust→Rust port case for the arena remains only in structural-sharing/dedup upside, not evaluation speed. The rank-1 DEFERRAL keeps `MultiVariableSRE` delegating via FFI to the native tree engine; revisit when a population-wide hash-consing sharing measurement exists | Same machinery exists (`graph.rs`, `enumerate.rs` hash-consing) — but the 222× regime is a Python-vs-Rust figure, NOT tree-vs-arena: the arena does not beat the tree evaluator in pure Rust (measured 0.51× and 0.85× on the two same-seed runs; TREE-FAVORED in both) |
| 2 | Level-1 dependency screen + mirror STRidge | **Rust, but gated on arm adoption**: extend `coupled_system.rs` with the screen statistic + reduced-library ridge-prune so kernel parity becomes CLAIMABLE (removing the documented "kernel parity never claimed" provenance caveat) | Python share is real but modest (31.8% at once-per-batch); migrate when the arm runs per-candidate screens at high frequency, then on the strength of one pinned-parity engine |
| 3 | Mesh cotangent-Laplacian assembly | **Rust with Rayon, gated behind the PDE arm phase** | 10⁵+ vertex assembly needs native parallel loops; matches the external analysis; no current discovery consumer |
| 4 | Heuristic multi-agent pair forces (NEW — measured today, not in the external analysis) | **Python vectorization first — DONE same day (behavior-identical, pinned; 2026-10-22 revision):** the vectorized path meets the 5 ms budget through N=64 and first exceeds it at N=128 (6.0678 ms); the 50 ms budget first fails at N=512. **Rust kernel port trigger: NOT met** — reconsider only for lower per-step budgets or N ≥ 512 real-time sweeps. Proximity/certified-path Rust work ties to the promotion checklist item 3 in `cbf_proximity_graph_backlog_gate.md` (index already measured at 3.1–5.8× build-floor speedup even in Python/scipy) | CBF profile: ~6 µs/Python pair call → 5 ms budget crossed at N=32 pre-vectorization, N=128 post-vectorization; nearest-neighbor filtering already inside the callee |
| 5 | EquationGNN | **Stay Python** | PyTorch autograd/CUDA already compiled; no Rust win (external analysis correct) |
| 6 | Epistemic claim DAG, lineage DAG, PROV graph, plotters, causal SCM sandbox, 1D MPS DMRG | **Stay Python** | Once-per-experiment frequency; NetworkX/scientific-ecosystem and visualization tooling; ~0.2 ms operations (external analysis correct; DMRG stays `np.einsum` until a physics problem defeats 1D chains) |
| 7 | 4D video tensor train | **Already hybrid** (Python TT-SVD decomposition + Rust parallel fiber contraction) | matches the external analysis |

## 5. What would change these decisions

- A production sweep that runs `discover_ode_system` per generated
  candidate (not per batch) moves rank 2 UP immediately.
- Certified CBF control at N ≥ 256 real-time would put proximity + margin
  machinery on the Rust critical path (currently blocked on the
  non-neighbor margin soundness argument — separate record).
- Referee-facing serialization demands (deterministic graph replay)
  could move epistemic lineage toward Rust, but only evidence-gated.

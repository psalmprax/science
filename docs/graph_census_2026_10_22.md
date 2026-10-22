# Graph census — verification and gap closure (2026-10-22)

Response to the external master graph census (classification of graph
representations as implemented-and-tested / implemented-untested /
not-implemented). This record closes the actionable gaps and corrects two
stale rows. Statuses below follow the repository's module-status tags.

## 1. The three "implemented but untested" slots — CLOSED

Ground truth verified before implementing tests: zero test hits for
`ProvenanceGraph`, `plot_epistemic_graph`, and `rime_rewrite_json` in
`tests/` (census claim confirmed). All three are now pinned in
`tests/test_dyn_graph_census_pins.py` (5 tests, green):

| Surface | Source | What the pins enforce |
|---|---|---|
| `ProvenanceGraph` (W3C PROV-lite) | `apedp/scientific_layers.py:185` | nodes `(kind, label, detail)` and links `(src, rel, dst)` recorded verbatim; no shared-mutable-default detail leak |
| `plot_epistemic_graph` | `apedp/visualization/plots.py:93` | renders a real PNG (magic bytes verified); deterministic layout seed |
| `rime_rewrite_json` (PyO3 FFI) | `rust/apedp-kernel/src/lib.rs:1094` | valid graph + `WrapNonlinearity` appends the wrapped node and makes it the output (rewrite.rs `apply` semantics, pure functional arena preserved); **bad operand index is rejected** (no fabricated dangling references); malformed graph/op JSON fails closed |

Two of these are now `implemented & tested`; the FFI binding additionally
carries the Rust-side decision discipline through the Python boundary.

## 2. Corrected rows (the census predates this session's DE arm)

- **"Level 1 State Dependency Graph (ODE): NOT IMPLEMENTED" — stale.**
  The DE-discovery arm now implements Level-1 dependency screening
  (`apedp_dyn/de_arm/dependency_screen.py`): per-state predictive-increment
  ranking of candidate parents over the training partition, with the
  reduced-library mirror fits and the MANDATORY dense escape hatch in ONE
  referee-facing `OdeArmOutcome`. Deliberate design boundary, per the
  external review: this is a *screening with an escape route*, NOT a causal
  adjacency matrix — no causal-topology verdict is claimed anywhere, and
  the kernel `coupled_system.rs` itself remains flat SINDy (its dense
  whitelist IS the escape hatch). Frozen-benchmark-measured:
  `benchmarks/de_arm_vs_dense_benchmark_results.json`.
- **"Native Reverse-Mode AD Tape Graph: NOT IMPLEMENTED" — still true for
  the kernel, with a correction of scope.** `apedp_dyn/de_arm/ad_service.py`
  now provides an exact-rule SYMBOLIC derivative service (product/quotient/
  chain) for the arm's expression vocabulary. That is shared
  differentiation infrastructure, not a reverse-mode tape: the Rust kernel
  correctly keeps forward Taylor Jets, and torch.autograd remains the
  reverse-mode runtime. The census row stands with that clarified boundary.

## 3. Genuinely NOT implemented — recorded backlog with rationale

These remain unimplemented by deliberate, review-aligned prioritization —
recorded here as backlog categories, not promises:

1. **Robotics proximity interaction graphs (k-NN / dynamic G(V, E(t)))
   in `cbf_certificate.py` / `multi_agent_transport.py`** — dense O(N²)
   all-pairs broadcasting is exact and validated (CBF pins green). **Gate
   decision 2026-10-22 measured and recorded; Vectorization RESOLVED
   same day** (`docs/cbf_proximity_graph_backlog_gate.md` addendum,
   `tests/test_dyn_multi_agent_vectorized_pins.py`): the heuristic
   transport stepper's Python pair loop is replaced by ONE vectorized
   all-pairs sweep, EXACT per-pair mirror of the scalar rule (pinned at
   per-pair branch structure + whole-step legacy-reference equality);
   re-measured: 5 ms crossing 32 → 128, exponent 1.91 → 1.06, 26× at
   N=256; a proximity graph is NOT needed for this path. The CERTIFIED
   QP gate stands unchanged: real-time through N≈128, proximity
   restructuring activates only for production N ≥ 256 real-time
   certified control AND a soundness proof (pitched separately in the
   margin module) that non-neighbor pairs cannot violate barriers within
   the assumption box — the certified contract is all-pairs by definition
   and pruning AFTER the dense barriers build buys nothing (the build
   alone is over budget at N=256).
2. **Irregular mesh graphs for PDE DISCOVERY** — meshes and laplacians
   exist for FORWARD FEM solving (`apedp/meshing.py`, `pde_solvers.py`);
   the discovery surface is still structured-grid + Fornberg stencils.
   Backlog: gated behind the review's sequence — benchmark the structured
   path first, add mesh operators only when a frozen benchmark requires
   them (review correction: the PDE arm is a later phase).
3. **SRE / genetic expression DAGs (subexpression sharing)** — RIME's
   native engine already has hash-consed shared-DAG enumeration
   (`rime/enumerate.rs`); `MultiVariableSRE` and `genetic.rs` keep trees.
   Backlog: DAG-upgrade the GP engine only if measured generation
   throughput or referee serialization demands it.
4. **2D/3D tensor networks (PEPS / MERA / TTN)** — strictly 1D MPS/TT
   today. Backlog: a physics problem class that defeats 1D contraction
   before building generic 2D/3D contraction optimizers.
5. **Reverse-mode AD tape in the Rust kernel** — rejected by design
   (forward jets avoid trajectory-tape memory blowup; the reviewed
   O(N·D)-memory warning). Backlog: adjoint methods only if continuous
   parameter fitting inside the kernel demonstrates the need.

## 4. Verification trail

- New pins: `python3 -m pytest tests/test_dyn_graph_census_pins.py` →
  5 passed; ruff clean.
- The FFI pin caught one test-side serialization-shape error (newtype
  variant `Tanh(usize)` → `{"Tanh": 2}`) — the assertion was corrected,
  not the kernel; the FFI behavior matches `rewrite.rs` exactly.
- No referee artifacts touched; the arm remains evidence-only.

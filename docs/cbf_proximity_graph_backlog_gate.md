# CBF O(N²) profile — proximity-graph backlog gate decision (2026-10-22)

Follow-up to the graph census: the robotics proximity-graph backlog item
said "profile first, upgrade only if O(N²) actually dominates." This is
that profile, and the gate decision it produces.

Harness: `benchmarks/run_cbf_scaling_profile.py` (deterministic, seeded,
A3-guarded grid fixture); raw artifact:
`benchmarks/cbf_scaling_profile_results.json`. Median wall times on the
dev box (single machine; numbers are decision thresholds, not constants);
adversarial fixture (uniform-random velocities) AND a factory-lemma floor
probe (identical velocities → the alternating-projection loop proves
feasibility after ONE sweep, isolating setup cost from iterations).

## Measured results

### Certified path — `solve_cbf_qp` (vectorized all-pairs einsum)

| N | pairs | adversarial ms | 1-pass floor ms | barriers build ms | µs/pair |
|---|---|---|---|---|---|
| 8 | 28 | 0.39 | — | 0.03 | 14.0 |
| 16 | 120 | 0.43 | — | 0.05 | 3.6 |
| 32 | 496 | 0.51 | — | 0.05 | 1.0 |
| 64 | 2016 | 2.38 | — | 0.17 | 1.2 |
| 128 | 8128 | 3.23 | 2.34 | 1.11 | 0.40 |
| 256 | 32640 | 80.9 | 9.88 | 5.47 | 2.48 |
| 512 | 130816 | 306.3 | 43.0 | 22.4 | 2.34 |

Adversarial exponent fit ≈ 1.66 (7-point log-log); iteration growth adds
≈8× ON TOP of the floor at N ≥ 256 (uniform-random commands need real
correction sweeps; the factory lemma bounds the floor only).

**Key floor finding:** even a literally single-sweep call (identical
velocities) costs 9.9 ms at N=256 and 43 ms at N=512 — the O(N²) barrier
build ([N,N,2] broadcast + norm) and the all-pairs constraint sweep carry
the cost BY THEMSELVES. Proximity pruning AFTER that build buys nothing:
the dense floor is already over budget before any pair is skipped.

### Heuristic path — `multi_agent_step` (Python pair loop)

| N | pair calls | ms | µs/call |
|---|---|---|---|
| 8 | 56 | 0.56 | 9.9 |
| 32 | 992 | 6.59 | 6.6 |
| 64 | 4032 | 23.7 | 5.9 |
| 128 | 16256 | 98.1 | 6.0 |
| 256 | 65280 | 365.6 | 5.6 |
| 512 | 261632 | 1445.8 | 5.5 |

Exponent ≈ 1.91 in N; cost is ~always a fixed ~6 µs per Python-level pair
invocation (the deepest `compute_cbf_repulsive_force` already early-returns
zeros beyond r_warning — the call overhead IS the cost).

## k-NN computational counterfactual (labelled: NOT a soundness claim)

Pair-reduction ratios versus dense (k=4): 0.383 at N=16 → 0.011 at
N=512 (k=8 half of that again). Restricted-assembly + set-union costs are
small, and if iteration work scaled purely with pair count the estimated
adversarial call at N=256 would drop 80.9 → ~7.1 ms, at N=512
306 → ~25.4 ms.

Two honest qualifications the estimate carries, both recorded:

1. **The dense-build trap.** The counterfactual *prunes pairs after
   building the full [N,N,2] barriers* (argpartition over the dense
   distance matrix) — which is why its estimate still leaves the dense
   floor in place. A production proximity structure must compute
   neighbor sets WITHOUT the dense matrix (KD-tree / voxel hash), or the
   floor stays O(N²).
2. **Soundness.** The certified contract is all-pairs by definition
   (hdot ≥ −γh for every pair, re-checked fail-closed). A
   proximity-restricted certified set requires proving non-neighbor pairs
   cannot violate their barriers within the assumption box over the
   control horizon (a guaranteed-separation margin argument). Measured
   pair counts do not substitute for that proof.

## Gate decision (criteria preregistered in the harness before measuring)

Budget: strict = 5 ms per control step (module runs at dt=0.05 → 20 Hz;
90% pipeline headroom); relaxed = 50 ms offline batch.

| Path | 5 ms crossing | 50 ms crossing | Gate verdict |
|---|---|---|---|
| Heuristic `multi_agent_step` | **N=32** | N=128 | **ACTIVATES NOW — but the first fix is vectorization, not a proximity graph** |
| Certified `solve_cbf_qp` (floor) | N=256 | N=256–512 | **BLOCKED on a soundness proof + spatial-index design; activates only for production N ≥ 256 real-time certified control** |

1. **Heuristic layer (N≥32): vectorize first.** The per-pair force call
   early-returns beyond r_warning; a masked, vectorized pair-force sweep
   (build pairwise distances from `[N,2]`, mask `dist < r_warning`,
   compute only unmasked entries) is behavior-identical for the
   heuristic — no certificate soundness question arises — and removes the
   ~6 µs/call Python overhead, the actual measured bottleneck. A k-NN
   graph adds nothing here that a distance mask does not.
2. **Certified layer (N≥256): proximity restructuring is a REAL
   requirement but gated**, because (a) even pair-count reduction of ~50×
   leaves ~7-25 ms adversarial estimates — above the strict budget — so
   the win must come from abandoning the dense barriers matrix (spatial
   index) and shrinking iteration structure, and (b) all-pairs is the
   certified contract: any reduced pair set changes the guarantee scope
   and needs the margin proof. Default usage (assumption cap max_agents
   ≤ 64) is comfortably inside budget and does NOT trigger the gate.
3. **Iteration sensitivity recorded:** adversarial-iteration growth (~8×
   at N≥256) is fixture-conditional; the floor numbers are the robust
   planning figures.

## What this does not establish

- No soundness claim for proximity-restricted certified sets (explicitly
  unproven here; the pair-ratio numbers are computational only).
- No production timing claim: dev-box medians, one fixture family;
  re-measure on target hardware before treating thresholds as constants.
- No change to certified code: the certified implementation, its pins and
  the battery mutants that protect it are untouched — this is a
  measurement + gate record.

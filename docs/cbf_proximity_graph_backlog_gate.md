# CBF O(N²) profile — proximity-graph backlog gate decision (2026-10-22)

Follow-up to the graph census: the robotics proximity-graph backlog item
said "profile first, upgrade only if O(N²) actually dominates." This is
that profile, and the gate decision it produces.

**POST-VECTORIZATION ADDENDUM (2026-10-22, same day).** Gate 1 was acted
on immediately and re-measured (same harness, same fixture seeds): the
heuristic pair loop is now ONE vectorized sweep
(`pairwise_repulsive_forces`, an EXACT per-pair mirror of the scalar rule,
pinned branch-by-branch in
`tests/test_dyn_multi_agent_vectorized_pins.py`; the per-agent float sums
differ only at summation-order ulp level). Results:

| N | heuristic ms (pre) | heuristic ms (post) | us/pair (pre → post) |
|---|---|---|---|
| 32 | 6.59 | 1.07 | 6.6 → 1.08 |
| 64 | 23.7 | 3.23 | 5.9 → 0.80 |
| 128 | 98.1 | 7.01 | 6.0 → 0.43 |
| 256 | 365.6 | 14.1 | 5.6 → 0.22 |
| 512 | 1445.8 | 47.5 | 5.5 → 0.18 |

Exponent fit 1.91 → 1.06 (cost now scales with pair VOLUME through numpy
evaluation, no per-call interpreter overhead). Re-measured crossings:
5 ms at N=128 (was N=32); the 50 ms budget is NOT crossed anywhere in the
sweep (was N=128); N=256 improved 26× (365.6 → 14.1 ms). The certified-
QP gate numbers are UNCHANGED (same code, byte-identical column values).
Gate 1 is therefore RESOLVED (vectorized, pinned, re-profiled); the
certified-path Gate 2 verdict below stands unchanged. Full re-run output
replaces `benchmarks/cbf_scaling_profile_results.json` (the pre-
vectorization numbers survive in this table and in git history).

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

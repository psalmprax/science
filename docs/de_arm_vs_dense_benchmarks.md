# DE arm vs dense flat library — frozen benchmark (2026-10-22)

Deterministic, seeded (`SEED=20261022`), noise-free RK4 fixtures of three
classic coupled systems. The same frame goes to the dense kernel path
(`discover_coupled_system`, the coupled route used today and the arm's own
escape hatch) and to the DE arm's screened-library mirror fits at two
screen depths (`top_k=2` normal, `top_k=1` aggressive). Harness:
`benchmarks/run_de_arm_vs_dense_benchmarks.py`; raw artifact:
`benchmarks/de_arm_vs_dense_benchmark_results.json`. Evidence-only — the
benchmark computes no acceptance or verdict.

## Systems and protocol

| System   | States | dt     | kept rows | coverage guard                     |
|----------|--------|--------|-----------|-------------------------------------|
| Lotka-Volterra | x, y | 0.01  | 800   | full-window + train-partition sweep |
| FitzHugh–Nagumo | x, y | 0.01 | 800   | full-window + train-partition sweep |
| Lorenz   | x, y, z | 0.002 | 8000    | 44-tu burn; both-lobes coverage in full AND first-70% train partition |

Structure comparison canonicalizes term names (`y*x` ≡ `x*y`) and excludes
the mirror's structural intercept (`1` is retained by construction and is
never pruned, so it is reported, not scored).

## Results (top_k = 2, the default screen depth)

| System | Dense: structure exact | Screened: structure exact | False exclusions | Library columns (dense → screened) | Measured reduction |
|---|---|---|---|---|---|
| Lotka-Volterra | ✓ (mean R² 1.0) | ✓ | 0 | 6 → 6 | 0% |
| FitzHugh–Nagumo | ✓ (mean R² 1.0) | ✓ | 0 | 8 → 6 | 25% |
| Lorenz | ✓ (mean R² 1.0) | ✓ | 0 | 10 → 9 | ~10% |

**The library-reduction hypothesis is measured, not assumed.** At D ≤ 3
the reduction is modest (0–25%); this matches the external review's
correction that a "90% reduction" claim is a hypothesis for higher
dimensions and richer grammars, to be benchmarked — not a standing
benefit. Nothing in this benchmark supports a universal reduction figure.

## Aggressive-screen demonstration (top_k = 1, Lorenz)

The aggressive depth suppresses a true parent in 2 of 3 states — by design
the worst honest case for the screen:

| State | Screen kept | Suppressed parent | Screen mirror R² | Dense hatch supplies |
|---|---|---|---|---|
| x | y | — (structure exact at R² 1.0) | 1.0 | — |
| y | x | `z` — `x*z` cannot be built from the kept pool alone | 0.626 | `x*z` (dense: `x`, `y`, `x*z`) |
| z | y | `x` — `x*y` cannot be built from the kept pool alone | 0.038 | `x*y` (dense: `x*y`, `z`) |

Two conclusions, both preregistered intent of the arm:

1. **The screen is falsifiable at its own evidence surface** — a wrong
   ranking collapses train R² visibly (0.63, 0.04) instead of hiding.
2. **The dense escape hatch is load-bearing**: the dense result in the
   SAME outcome carries the full recovered structure whenever the screen
   prunes truth. Screen exclusions are reported verbatim in the
   referee-facing artifact; nothing is silently dropped.

## Fixture-integrity finding (recorded, not patched away)

An earlier Lorenz fixture kept a narrow slow-manifold window
(x ∈ [7.3, 9.6]). Because the data is noise-free and near-collinear there,
MANY polynomial column combinations achieved identically zero residual:
**both** the dense kernel and the mirror returned exact-but-non-physical
structures with R² = 1.0. The 4th-order stencil itself was verified exact
against the RK4 RHS (max |err| = 0.0), isolating the cause to window
under-excitation. The harness now refuses fixtures whose full window OR
train partition sweeps less than the per-system span threshold. This is
precisely the artifact class the external review warned discovery systems
about ("a discretization/coverage artifact, not a governing law") and is
kept on record as a demonstrated failure mode of the protocol itself.

## Runtime (measured, no performance claim)

Whole-arm outcome (dense kernel + screen + mirror fits, Python+kernel):
LV ≈ 5 ms (n=800), Lorenz ≈ 50 ms (n=8000). Timing is informational only.

## What this benchmark does NOT establish

- No discovery-power claim beyond D = 3 polynomial systems.
- No causal-topology claim for the screen (its own artifacts carry the
  predictive-increment disclaimer verbatim).
- No acceptance/promotion: the arm remains evidence-only; referee
  artifacts are untouched by this benchmark.

# PDE-mesh discovery prototype (DE arm — `pde_mesh_backlog` gate evidence base)

Status: **`experimental`** — prototype phase 1. This document is the evidence
base the `pde_mesh_backlog` gate decision reads; it makes **no production
claim**: every acceptance/promotion decision belongs to the frozen referee
protocol. Module: [`apedp_dyn/de_arm/mesh_discovery.py`](../apedp_dyn/de_arm/mesh_discovery.py).
Pins: [`tests/test_dyn_de_arm_mesh_prototype.py`](../tests/test_dyn_de_arm_mesh_prototype.py)
(17/17 as of 2026-10-22).

## What the prototype is

Cotangent-Laplace-Beltrami discovery on 2D planar Delaunay meshes, three
layers per the DE-arm discipline:

- **Layer A (typing)** — candidates declared through the arm's
  `PdeMathModel` (fields, independent variables, positive `dt`,
  boundary treatment); incomplete declarations are fail-closed refusals.
  `dx` on the threaded model is the **median edge length** (the mesh's
  characteristic spacing), never a fabricated grid constant.
- **Layer B (physics)** — discrete cotangent Laplace-Beltrami assembly:
  `(Δu)_i = (1/(2A_i)) Σ_j w_ij (u_j − u_i)`, `w_ij = cot α + cot β`
  over the ≤2 adjacent triangles, barycentric vertex areas. The operator
  carries the mesh GEOMETRY — it is NOT a graph Laplacian D − A (the
  reviewed trap). The Laplacian diagonal is assembled from the same
  per-row weight sums, so L·1 = 0 exactly (an earlier broken state left
  the diagonal out and produced L·1 ≈ 3×10⁴, which discovery compensated
  with enormous bogus coefficients — regression documented).
- **Layer C (discovery)** — time-derivative residual regression per the
  arm's documented python ridge+prune STRIDGE mirror
  (`dependency_screen._stridge_fit`; kernel parity NOT claimed) over the
  mesh library {1, u, u², u³, Δu (, u·Δu)} on interior vertices (2-ring
  hop mask from the boundary).

**Escape route preserved**: for grid-runnable data the structured-grid
Fornberg engine (`epistemic.pde_discovery.discover_pde`) remains the
established path. The mesh prototype exists exactly for irregular-domain
data that cannot reach a grid; it never competes silently with the grid
engine.

## Measured evidence (frozen fixture: 21×21 clipped jittered mesh, seed 7)

| Quantity | Measured | Pin |
|---|---|---|
| L·1 (interior rows AND all rows) | 3.6e-12 / 1.1e-11 | < 1e-9 |
| L·(affine) (interior rows) | 3.6e-12 | < 1e-9 |
| edge-weight symmetry `w_ij = 2·A_i·L[i,j]` | 1.4e-14 | < 1e-10 |
| interior-edge cotangent sums ≥ 0 (Delaunay/Rand) | min 9.45e-4, 0 negative | > 0 |
| Dirichlet energy `½Σ w_ij (u_i−u_j)²` on S | ≥ −1e-9 on probes; = 0 for constants | same |
| **single-term heat recovery** (truth u_t = κΔu as `expm_multiply` of the ASSEMBLED operator; κ = 1e-3) | κ̂ = 1e-3, **relative error 3.7e-7**, R² = 0.9999999998, rmse 2.8e-7 | rel err < 1e-5 |
| mirror selection on the recovery panel | `active_terms == {1, laplacian}` (parsimony honored) | exact set |
| deterministic re-runs | bit-identical outcomes | exact |
| fail-closed battery | periodic-on-mesh, unknown area kind, sliver, repeated vertex, zero-area vertex/hull mismatch, <5 frames, non-finite panels, constant target, too-small mesh, unsupported boundary — ALL refused with messages | pinned |

The recovery fixture is an `expm_multiply` trajectory **of the assembled
operator itself**, so the truth relation `u_t = κΔu` is valid in the
DISCRETE setting with no simulation closing error beyond expm/central-
difference arithmetic — the same discipline as the CBF battery's
fixture-side stability guard, and the reason earlier sim-only attempts
(spatially-under-resolved transients soaking the polynomial family) are
not repeated here.

## Measured limitation (disclosed, drives the gate decision)

On identifiability-POOR panels the shared python STRIDGE mirror's
relative-magnitude pruning (threshold 0.02 of max |w| per iteration)
RETAINS polynomial alias terms. Measured case: single discrete-eigenfamily
decay data (two-mode exp decay of Krylov eigenpairs), where the six-term
mirror returned `{1, laplacian, u, u·Δu, u²}` with κ̂ ≈ 33 vs truth 1.0
(±0.95% across frozen row-split refits — stable but biased; R² ≈ 0.9998).
The mirror is a candidate-SURFACING screen, not a structure arbitrator:
raw-scale coefficients on alias-contaminated panels are NOT interpretable,
and recovery claims must be arbitrated on healthy-excitation panels like
the pinned `expm_multiply` fixture, never read off alias-contaminated ones.
This is the same under-excitation lesson the Lorenz window and heat
fixtures already recorded for derivative-residual regression.

## What remains before the gate decision (honest gap list)

1. **Frozen-holdout structure arbitration for mesh panels** — the referee's
   holdout contract must apply to the mesh arm the way it applies to the
   grid arm (fit restricted-vs-full library families, arbitrate on a held
   confirmation partition, charge selection reuse to the trial ledger).
   The mirror alone explicitly does NOT do this.
2. **Irregular-domain benchmark suite** — at least one non-rectangular
   (e.g. annulus / L-shaped) frozen domain with known eigenstructure,
   mirrored against the grid engine's Fornberg path on its domain class.
3. **Boundary treatment honesty** — 'none' currently uses the same 2-ring
   interior mask (single-cotangent boundary-triangle weights); an
   intrinsic-boundary operator is future work and must not be silently
   equated to the Dirichlet case.

Status: prototype phase 1 complete (operator + recovery + refusals
pinned). **Recommendation recorded for the gate owner: do NOT promote the
mesh engine to production on this evidence — Layers A/B are
validation-grade, Layer C needs the arbitration step above.**

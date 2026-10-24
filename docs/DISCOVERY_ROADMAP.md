# APEDP Discovery Roadmap: Certification + Data-Free Conjecture

Status: implemented. 407 tests pass (last full e2e). Runtime coverage includes 197 L0 algebraic laws (v4: 193 + Brown/warp additions) and 11 PDE templates; Phase 2 numerical extensions are regression-tested without an unsupported named-equation percentage.

## Coverage ladder (all equations - math consistency note added)

- **L0 algebraic:** 197 named laws/entries in `PriorArtAuditor` (v4), all parse-gated, plus special functions (`erf`/`erfc`, `gamma`, `zeta`, `besselj`/`bessely`/`besseli`/`besselk`) end-to-end: allowlist → dimensional gate (dimensionless args only) → singularity-risk flags for Y/K/γ/ζ). Rename-isomorphism matches `I*R`, `m*v²/2`, `σT⁴`, `2π√(L/g)`; bare `x²` stays novel.
- **L1 PDE templates** (`apedp/pde_templates.py`, `safe_parse_pde`): 11 templates (DIFFUSION/HEAT_1D, WAVE_1D, LAPLACE_2D, POISSON_2D, CONTINUITY_1D, EULER_MOMENTUM_1D, NS_X_1D, SCHRODINGER_TD_1D, MAXWELL_FARADAY/AMPERE_1D). Class-level shape match only (coefficients/BCs NOT verified).
- **L2 black-box:** no symbolic form needed — falsifier grids judge numerically.
- **OUT_OF_SCOPE** (`NoveltyStatus.OUT_OF_SCOPE`): PDE with no template, integral forms beyond the allowlist, or unparseable input. Returned instead of a forced known/novel verdict.
- **Extended evaluation** (`apedp/eval_extended.py`, all bounded): `evaluate_implicit` (nsolve) + implicit *systems* (nsolve + residual norm), `evaluate_quadrature` (trapezoid), `evaluate_ode_ivp` (`rk4` / adaptive `rk45` with 100k-step refusal / L-stable `backward_euler` scalar + `evaluate_ode_system` vector — stiff `y'=-50y` stays bounded where RK4 explodes to 2e11), validated `solve_eigen` (residual-checked, sorted), ODE event stop (`evaluate_ode_events`/`evaluate_ode_system_events`: scan + 100-substep RK4 refine with reported precision; grazing touches documented as missed), `evaluate_pde_residual` (manufactured-solution check), `solve_heat_1d_explicit` (Dirichlet FD, stability-refused), Einstein index strings (`evaluate_indexed`: matmul/affine/trace verified, bad indices raise), symbolic index calculus (`parse_indexed_to_sympy`/`symbolic_contract`: δ-identities prove, traces expand), tensor `einsum`, seeded SDE Euler-Maruyama. Repair routes PDE input to template `OUT_OF_SCOPE` instead of `REJECTED`.
- **Full-PDE solvers** (`apedp/pde_solvers.py`, numpy only): `solve_wave_1d` (leapfrog, CFL≤1), `solve_advection_1d` (upwind, |C|≤1), `solve_heat_2d` (explicit, r≤0.25), `solve_laplace_2d` / `solve_poisson_2d` (Jacobi to tolerance, Dirichlet), `solve_heat_3d` (explicit, r≤1/6, structured-grid budget), `solve_laplace_3d` (Jacobi-3D), `solve_poisson_fem` (P1 FEM, exact nodal loads, hull-loop Dirichlet). Verified: wave 4e-6 vs standing mode, Laplace recovers `x+y` to 8e-5, Poisson sine to 1.3e-3 (structured) / 0.02–0.05 (unstructured), Laplace-3D linear to 2e-4.
- **High-D Monte Carlo** (same module, `O(n·d)` not `N^d`): `solve_heat_nd_feynman_kac` (free-space heat to 20D: 5D linear and 10D Gaussian match closed forms within MC error), `solve_laplace_nd_walk_on_spheres` (d-ball Dirichlet; 5D and 20D linear verified). Structured 31-point grids stop at 3D by budget refusal.
- **Meshing** (`apedp/meshing.py`, numpy only): exact-integer Bowyer-Watson Delaunay (Bareiss predicates), combinatorial validation (point coverage + manifold edges + single boundary cycle + area, with hole ear-filling and seed retries), hull-edge snapping for boundary-cluster slivers, Laplacian smoothing, max-angle-driven Ruppert refinement (insert / encroach-split / interior decimate, champion tracking). Verified end-to-end: clipped-jitter unstructured FEM 0.02–0.05. Known caveat: extreme near-collinear boundary clusters may keep obtuse corners past 150°; `refine_mesh_2d` now reports the achieved angle, target, pass/fail, and termination status.

## Certification (Track A)

- `run_prospective(train, holdout)`: discover on train only, freeze hashes, single locked-holdout eval via `DataPartitionVault`; non-generalizing models refused and removed from the store.
- `Fraction` thirds (`±1/3, ±2/3`) in search + monomials; `erf`/`erfc` end-to-end.
- `run(..., param_uncertainties=)` attaches GUM/Monte-Carlo metrology; elite `lambdify` memoized.

## Conjecture (Track B)

`HypothesisComposer` (crossover / saturate / correspondence mutants) emits ranked `THEORETICAL_SPECIFICATION` conjectures usable as `template_priors`. Data-free checks: dimensional homogeneity, bounded correspondence limits for saturation candidates, repair audit, and bounded boundary/numerical probes.

## Rediscovery vs novelty

Pipeline demos (Ohm `I*R`, kinetic `m*v²/2`, Stefan-Boltzmann) are round-trips, not discoveries. Novelty verdicts are corpus-relative by principle; the corpus is now 193 laws plus PDE templates, and anything outside answers `OUT_OF_SCOPE` with the reason.

## Path to broader named-law coverage

- **Phase 1 (done):** special functions end-to-end (allowlist → dimensional gate → singularity-risk), validated eigensolver (residual-checked, sorted), ODE event stop (scan + 100-substep RK4 refine with reported precision).
- **Phase 2 (done):** delay DEs via bounded method of steps (`evaluate_delay_ivp` / `evaluate_delay_system`), second-kind Fredholm/Volterra Nyström solvers (`solve_nystrom_*`), and exact-predicate 3D Delaunay tetrahedralization (`delaunay_triangulate_3d`, insphere/Bareiss predicates). Regression coverage is in `tests/test_phase2.py`.
- **Phase 3:** nonlinear PDE stepping (Burgers with limiter, Allen-Cahn with convex splitting), twin-trajectory chaos diagnostic with honest refusal past Lyapunov horizon.
- The remaining work is long-tail and exotic (fractional, multiphysics couplings) — approached asymptotically, never claimed complete.


### ⚠️ Mathematical Consistency Note (corrected 2026-09-19)
Earlier drafts stated '203 laws (193 original + 16 new)', but 193+16=209, not 203.
Verified: `STANDARD_PHYSICS_CORPUS` contains 197 entries in code (v4); 10 fusion candidates + 6 derived relationships are documented pending PVE validation, for 213 documented entities total.

- **197 laws/entries** in `STANDARD_PHYSICS_CORPUS` (verified, v4 with scoped additions)
- **10 fusion-related candidate expressions** (Lawson criterion, triple product, etc. — CANDIDATE/HEURISTIC, not merged)
- **6 derived/extended relationships** (parameter variations, composite metrics — HEURISTIC/CANDIDATE)
- **213 total documented entities** (197 + 10 + 6 = 213)

Only corpus-matched or formally derived expressions may be ESTABLISHED; score alone is insufficient. See GAP_ANALYSIS.md.

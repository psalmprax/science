# APEDP Discovery Roadmap: Certification + Data-Free Conjecture

Status: implemented. 354 tests green (289 baseline + 65 new), ruff + mypy clean.

## Coverage ladder (all equations)

- **L0 algebraic:** 193 named laws in `PriorArtAuditor` (v3), all parse-gated: mechanics, EM/electronics, thermo/fluids/heat, waves/optics/QM/relativity, chemistry/materials, vibrations/acoustics, signals, cycles/combustion, mass transfer, electrochem/semiconductors, structures/geo/astro, turbomachinery, bio/misc rates. Rename-isomorphism matches `I*R`, `m*v²/2`, `σT⁴`, `2π√(L/g)`; bare `x²` stays novel.
- **L1 PDE templates** (`apedp/pde_templates.py`, `safe_parse_pde`): DIFFUSION/HEAT_1D, WAVE_1D, LAPLACE_2D, POISSON_2D, CONTINUITY_1D, EULER_MOMENTUM_1D, NS_X_1D, SCHRODINGER_TD_1D, MAXWELL_FARADAY/AMPERE_1D. Class-level shape match only (coefficients/BCs NOT verified).
- **L2 black-box:** no symbolic form needed — falsifier grids judge numerically.
- **OUT_OF_SCOPE** (`NoveltyStatus.OUT_OF_SCOPE`): PDE with no template, integral forms beyond the allowlist, or unparseable input. Returned instead of a forced known/novel verdict.
- **Extended evaluation** (`apedp/eval_extended.py`, all bounded): `evaluate_implicit` (nsolve) + implicit *systems* (nsolve + residual norm), `evaluate_quadrature` (trapezoid), `evaluate_ode_ivp` (`rk4` / adaptive `rk45` with 100k-step refusal / L-stable `backward_euler` scalar + `evaluate_ode_system` vector — stiff `y'=-50y` stays bounded where RK4 explodes to 2e11), `evaluate_pde_residual` (manufactured-solution check), `solve_heat_1d_explicit` (Dirichlet FD, stability-refused), Einstein index strings (`evaluate_indexed`: matmul/affine/trace verified, bad indices raise), symbolic index calculus (`parse_indexed_to_sympy`/`symbolic_contract`: δ-identities prove, traces expand), tensor `einsum`, seeded SDE Euler-Maruyama. Repair routes PDE input to template `OUT_OF_SCOPE` instead of `REJECTED`.
- **Full-PDE solvers** (`apedp/pde_solvers.py`, numpy only): `solve_wave_1d` (leapfrog, CFL≤1), `solve_advection_1d` (upwind, |C|≤1), `solve_heat_2d` (explicit, r≤0.25), `solve_laplace_2d` / `solve_poisson_2d` (Jacobi to tolerance, Dirichlet), `solve_heat_3d` (explicit, r≤1/6, structured-grid budget), `solve_laplace_3d` (Jacobi-3D), `solve_poisson_fem` (P1 FEM, exact nodal loads, hull-loop Dirichlet). Verified: wave 4e-6 vs standing mode, Laplace recovers `x+y` to 8e-5, Poisson sine to 1.3e-3 (structured) / 0.02–0.05 (unstructured), Laplace-3D linear to 2e-4.
- **High-D Monte Carlo** (same module, `O(n·d)` not `N^d`): `solve_heat_nd_feynman_kac` (free-space heat to 20D: 5D linear and 10D Gaussian match closed forms within MC error), `solve_laplace_nd_walk_on_spheres` (d-ball Dirichlet; 5D and 20D linear verified). Structured 31-point grids stop at 3D by budget refusal.
- **Meshing** (`apedp/meshing.py`, numpy only): exact-integer Bowyer-Watson Delaunay (Bareiss predicates), combinatorial validation (point coverage + manifold edges + single boundary cycle + area, with hole ear-filling and seed retries), hull-edge snapping for boundary-cluster slivers, Laplacian smoothing, max-angle-driven Ruppert refinement (insert / encroach-split / interior decimate, champion tracking). Verified end-to-end: clipped-jitter unstructured FEM 0.02–0.05. Known caveat: extreme near-collinear boundary clusters may keep obtuse corners past 150° — reported, never silently meshed.

## Certification (Track A)

- `run_prospective(train, holdout)`: discover on train only, freeze hashes, single locked-holdout eval via `DataPartitionVault`; non-generalizing models refused and removed from the store.
- `Fraction` thirds (`±1/3, ±2/3`) in search + monomials; `erf`/`erfc` end-to-end.
- `run(..., param_uncertainties=)` attaches GUM/Monte-Carlo metrology; elite `lambdify` memoized.

## Conjecture (Track B)

`HypothesisComposer` (crossover / saturate / correspondence mutants) emits ranked `THEORETICAL_SPECIFICATION` conjectures usable as `template_priors`. Data-free tests only: homogeneity, Pi-groups, limit lattice, repair audit, falsifier axes.

## Rediscovery vs novelty

Pipeline demos (Ohm `I*R`, kinetic `m*v²/2`, Stefan-Boltzmann) are round-trips, not discoveries. Novelty verdicts are corpus-relative by principle; the corpus is now 193 laws plus PDE templates, and anything outside answers `OUT_OF_SCOPE` with the reason.

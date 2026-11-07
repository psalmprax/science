# Sizing solver convergence audit (2026-11-07)

`apedp_dyn.sandbox.finance_l2_demo.size_positions` sizes a position by solving a
cost-penalised mean-variance problem over a forecast `μ`, a covariance `Σ`, a fee
and a stated risk aversion `γ`:

```
maximise  w'μ − (γ/2) w'Σw − c‖w − w_prev‖₁      subject to  w ≥ 0,  Σw ≤ cap
```

The solver is proximal gradient on the capped simplex with a fixed step
`1 / (γ λ_max)`. That step is cheap and monotone, but its convergence rate
depends on the covariance condition number `κ = λ_max / λ_min`, so the audit
measures the rate rather than asserting it.

## What is measured

`scripts/audit_sizing_solver_convergence.py` builds correlated SPD covariances
(random orthogonal eigenbasis, eigenvalues `logspace(0, −log10 κ)`) for
`n = 2, 4, 8`, `κ = 10² … 10¹²`, and previous weights that are either zero or a
random portfolio. For each of the 36 cases it records:

- the fixed-point (KKT) residual of the returned weights,
- whether the loop met its own `1e-12` weight-change tolerance (`converged`),
- the iterations used,
- the objective gap and maximum weight delta against a 200,000-iteration run at
  tolerance `1e-15`,
- gross exposure and whether the cap binds.

The record is `benchmarks/sizing_solver_convergence.json`.

## Findings

**All 36 cases converge**, where `converged` now means an analytic KKT
violation of at most `1e-10`. The worst residual in the record is **9.9e-13**
(`n = 8`, `κ = 1e2`, nonzero previous weights). The record still carries the
iteration count and the objective gap against a 200,000-iteration run, but the
stopping rule no longer depends on the gradient loop alone.

The grid exposed two different failures, which is why the fix is not a larger
iteration cap:

1. **The fixed step can exhaust its budget.** `n = 8`, `κ = 1e4` and `κ = 1e12`
   with zero previous weights hit 4,000 iterations. The active set and the L1
   sign pattern were already settled; solving the KKT system on that face takes
   both to machine precision.
2. **The fixed-point residual can report convergence at an L1 kink.** For
   `n = 8`, `κ = 1e12` with nonzero previous weights, the proximal map's
   fixed-point residual was 9.9e-13 while the analytic KKT violation was
   **3.9e-04** and the objective was **0.0077** against the optimum's
   **0.0147**. Two coordinates flip sign as they cross their previous weights,
   so a sign-flipping active-set loop cycles instead of converging.

`size_positions` now reports the analytic KKT violation and, when the loop has
not met it, (a) solves the active-set KKT system on the identified face and
(b) if that still cycles, solves the smooth `t >= |w − previous|` QP and snaps
the result onto any kink it finds. The kink snap matters: a coordinate resting
at its previous weight satisfies an interval of stationarity conditions, not an
equality, and averaging it into the multiplier biases the check.

Condition number alone does not predict the iteration count: many high-`κ` cases
terminate in tens of iterations because the optimum sits on a vertex or the cap
face and the projection resolves it, while moderate `κ` with `n = 8` can need
thousands. The audit therefore records the whole grid rather than a headline
`κ` threshold.

## Reproduce

```
.venv/bin/python scripts/audit_sizing_solver_convergence.py
```

The script always exits 0: it is a measurement, and a non-converged case is a
result to read, not a reason to hide the record. Rerun it after changing the
step rule, the tolerance or the default iteration budget, and update this page
and the JSON together.

## Limitations

- The long run is a reference, not an exact solution; a case where both runs stop
  short would understate the gap, though the reported KKT residual would expose
  it.
- The audit covers the demo sizing rule at its default `γ = 1` and 5 bps fee.
  Other risk aversions change `λ_max` and therefore the step, not the analysis.
- `converged` means the analytic KKT violation is at most `1e-10`; the loop's
  own `1e-12` weight-change tolerance is a stopping heuristic, not the
  convergence claim.

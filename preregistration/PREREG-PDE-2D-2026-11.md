# PREREG-PDE-2D-2026-11: the weak form in two space dimensions

**Status:** frozen before the run by `preregistration/DIGEST-PDE-2D.json`. Runs once:
`python scripts/run_prereg_pde_2d.py --out preregistration/RESULT-PDE-2D-2026-11`.

## What is tested

`apedp_dyn/pde/weak_form_2d.py` identifies an equation `u_t = ...` from a field `u(t, x, y)` by
integrating the data against test functions, over a library of eleven divergence-form terms, and
selects terms by backward elimination under an information criterion. The one-dimensional module
had a result (heat equation exact through 10 % noise in a development study) and no preregistered
test; the PDE engine was limited to one dimension (audit M-08).

## Data

Six equations, simulated on a periodic 64 x 64 grid, 101 frames, by a pseudo-spectral RK4 scheme
(`scripts/pde_2d_worlds.py`): heat; advection-diffusion; Fisher-KPP; Allen-Cahn; a scalar Burgers
equation; anisotropic diffusion. Each from five random smooth initial fields (seeds 2000 to 2004,
not used before), at additive noise of 0, 1, 5 and 10 % of the field's standard deviation. 120 runs.

A run is *exact* when the set of terms found equals the set in the equation. Its coefficient error
is the largest relative error over the true terms.

## Expectations, fixed now

1. **Right equation at moderate noise:** exact in at least 81 of the 90 runs with noise up to 5 %.
2. **Right coefficients:** among the exact runs at 5 % noise, the median coefficient error is at
   most 5 %.
3. **Nothing invented on clean data:** no run at zero noise contains a term that is not in the
   equation.
4. **Holds up at 10 % noise:** exact in at least 21 of the 30 runs.
5. Reported without a criterion: the one-dimensional module's threshold rule on the same fields,
   and results per equation.

A miss on 1 to 4 is reported as a miss.

## Disclosed before freezing

- Development used the same six equations with seeds 900 and 910 at noise 0, 5 and 10 %. With the
  evidence rule 35 of those 36 runs were exact (the miss: Allen-Cahn at 10 % noise lost one
  diffusion term); the largest coefficient error among the exact ones was 4.5 %.
- The evidence rule was written after the threshold rule failed on Fisher-KPP in that development
  run (it dropped a diffusion term on all three noise levels). Its two settings (rows counted as
  non-overlapping windows; a floor of 0.5 % of the left-hand side) were set there and not tuned
  further.
- The equations are simulated, on a periodic domain, with the library containing every term they
  use. This says nothing about measured fields, boundaries, or equations with terms outside the
  library; a real field would also need the time and space sampling to resolve the dynamics.
- The data come from my own simulator. An error common to the simulator and the identifier (for
  instance in a sign convention) would not show up. The unit test against an analytic solution of
  the heat equation covers the diffusion terms only.

## Deviation rule

The run is not repeated. A crash is fixed only if the fix cannot change a number, the fix is
recorded with its date in the result, and the digest is re-frozen with the old one kept. Any other
change makes the result a development result.

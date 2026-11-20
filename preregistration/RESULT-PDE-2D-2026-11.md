# Result: PREREG-PDE-2D-2026-11

**Run:** once, 2026-11-19, at commit `5dec0c4`, which froze the preregistration, the script, the
simulator and the two weak-form modules (`preregistration/DIGEST-PDE-2D.json`; the script checks it
before running). Python 3.13.18, numpy 2.5.3. One CPU core; identification took 6 s in total for the
120 runs, the simulations most of the rest. No deviation.

Outputs, unedited, in [`RESULT-PDE-2D-2026-11/`](RESULT-PDE-2D-2026-11/): `results.json`, `runs.json`
and `run.log`.

## Outcome

Six equations, five fresh initial fields each, four noise levels: 120 runs. A run is exact when the
set of terms found equals the set in the equation.

| Noise | Exact (evidence rule) | Median coefficient error | Worst | Exact (threshold rule) |
| :--- | ---: | ---: | ---: | ---: |
| 0 | **30 of 30** | 0.04 % | 0.05 % | 28 of 30 |
| 1 % | **30 of 30** | 0.16 % | 0.77 % | 28 of 30 |
| 5 % | **30 of 30** | 0.66 % | 2.5 % | 28 of 30 |
| 10 % | 27 of 30 | 1.4 % | 4.5 % | 28 of 30 |
| All | **117 of 120** | 0.26 % | 4.5 % | 112 of 120 |

By equation (of 20 runs each), evidence rule: heat 20, advection-diffusion 20, Fisher-KPP 20,
Allen-Cahn 17, Burgers 20, anisotropic diffusion 20. Threshold rule: 20 everywhere except
Fisher-KPP, 12.

No run, under either rule and at any noise level, contained a term that is not in its equation.

## Against the expectations

1. **Right equation at moderate noise: held.** Exact in 90 of the 90 runs with noise up to 5 %
   (limit 81).
2. **Right coefficients: held.** Median coefficient error 0.66 % at 5 % noise (limit 5 %).
3. **Nothing invented on clean data: held.** No extra term at zero noise.
4. **Holds up at 10 % noise: held.** Exact in 27 of 30 (limit 21).

All four held.

## What this means

- The PDE engine is no longer limited to one space dimension: on these fields it returns the right
  equation with coefficients good to about one percent, through 5 % noise without a miss.
- Its errors are all of one kind: a real term lost, never a false one added. The three misses are
  Allen-Cahn at 10 % noise, each losing one of its two diffusion terms (the smallest coefficient in
  the set, 0.03).
- Selecting terms by evidence instead of a fixed share of the left-hand side is what recovers
  Fisher-KPP: the threshold rule drops a diffusion term there in 8 of 20 runs, clean data included.
  The threshold rule is slightly better at 10 % noise on Allen-Cahn (it keeps the terms the evidence
  rule loses), so neither rule dominates at the highest noise.

## What it does not show

As disclosed before the run: the fields are simulated on a periodic domain, by a simulator written
by the same hand as the identifier, and the library contains every term the equations use. A
measured field, a boundary, a term outside the library (a mixed derivative, `u * laplacian(u)`), or
sampling too coarse to resolve the dynamics are all untested.

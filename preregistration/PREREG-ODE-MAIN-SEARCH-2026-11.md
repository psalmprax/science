# PREREG-ODE-MAIN-SEARCH-2026-11: ODEBench through the main search path

**Status:** frozen before the run by `preregistration/DIGEST-ODE-MAIN-SEARCH.json`. Runs once, on
GitHub Actions (`.github/workflows/odebench-main-search.yml`), started by the first branch named
`odebench-run-*` whose shards start.

## What is tested

Discovery of ordinary differential equations had no benchmark result in this repository. The older
path (`apedp_dyn/ode/proposer.py`) was limited by term selection (audit M-07) and was only ever
tried on a handful of hand-made systems.

Here each component of `dx/dt = f(x)` is one regression problem for the main search path
(`apedp_dyn/discovery_l2/main_search.py`, unchanged). `apedp_dyn/ode/window_rows.py` turns a
trajectory into rows with independent errors; `apedp_dyn/ode/main_search_ode.py` sends each
component through the path with PySR (240 s) as the search stage, run on every component, and
takes the pool from a trajectory the search never saw.

Beside it, on the same rows: sequentially thresholded least squares over all monomials up to degree
three, threshold chosen by BIC (`sparse_poly3`). This is the polynomial SINDy a reader will know. It
always returns a law and can only express polynomials.

## Data

`scripts/odebench_worlds.py`. ODEBench's 63 systems with their published parameter values (117
components: 79 are polynomials of degree three or less, 38 are not). Each system is integrated from
four initial conditions over ten time units at 10 000 samples: the two published ones (development
rows), and the points one third and two thirds of the way between them (the pool, and a trajectory
no engine sees). Noise levels 0 and 1 % of each variable's spread. 117 × 2 × 2 engines = 468 runs.
All 63 systems integrate; none is excluded.

A reported law is scored against the true vector field on 400 states (100 from each clean
trajectory): **recovered** by the symbolic check of the SRBench runs, **accurate** if R² ≥ 0.999.
A reported law is **wrong** if it is neither. The main search path's law is scored in the form the
path confirmed (`main_search_ode.confirmed_form`).

"Wrong" is "neither" on purpose. At 1 % noise the rates carry errors of 2 to 12 % (median 4 %) and
there are 200 development rows, so the true law with fitted coefficients can fall below R² 0.999;
that is imprecision, not a wrong law. Both counts are reported.

## Expectations, fixed now

1. **What the path reports can be relied on:** of the component laws the main search path reports
   (both noise levels together), at most 5 % are wrong.
2. **Recovery:** the path recovers at least 65 of 117 components at noise 0 and at least 45 at 1 %.
3. **Beyond polynomials:** of the 38 components that are not polynomials of degree three or less,
   the path recovers at least 12 at noise 0.
4. **The reference always answers, and is often wrong:** of the laws `sparse_poly3` reports (both
   noise levels together), at least 20 % are wrong.
5. Reported without a criterion: the two engines side by side at each noise level, systems with
   every component recovered, accuracy, the path's actions and which stage found each law, time,
   and the systems named below whose clean rates are imprecise.

A miss is reported as a miss.

## Disclosed before freezing

- **PREREG-MAIN-SEARCH-2026-11's result is not known.** Its run was still in progress when this
  was frozen. Expectation 1 here is the same design criterion, tested independently.
- ODEBench's equations were read once before, as text, for PREREG-MMLM-L (bits per law). No search
  had been run on them. To write this document they were classified by form (polynomial or not,
  number of operations).
- Seven ODEBench systems are the Strogatz systems that SRBench also carries. The main search path
  was developed with SRBench problems in view (one of them, `strogatz_bacres1`, led to the
  smoother check).
- Development of `window_rows`, the wrapper and the runner used three hand-made systems (a Duffing
  oscillator, a two-variable reaction, a logistic law with harvesting), with the first pass alone
  locally and with PySR once in CI: all ten component runs were recovered at both noise levels;
  the reference missed one at 1 %. All three are polynomial, which is the reference's home ground.
- Decided on those systems: the development rows come from two trajectories, not one (on one
  trajectory a simple wrong formula fitted a two-variable law along the curve and the first pass
  stopped on it); PySR runs on every component for the same reason; scoring uses states from all
  four trajectories (on the held-back one alone, a system at rest makes R² a test of the fourth
  decimal).
- The accuracy of the window rows on the ODEBench trajectories was looked at before freezing; this
  uses the true systems. Clean rates are exact to a median relative error of 1e-10. For five
  systems (11, 41, 55, 59, 61) the error is above 1e-6, the level below which the path treats a
  fit as exact; on those the path may refuse a true law at noise 0. Nothing was changed after
  looking.
- The sampling is ours. ODEBench's own protocol uses far fewer samples and scores trajectories, so
  these numbers are not comparable with published ODEBench results.
- Nothing integrates a discovered system forward. A recovered component is a statement about the
  rate of change on the states visited.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.

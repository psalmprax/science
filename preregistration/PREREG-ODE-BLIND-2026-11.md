# PREREG-ODE-BLIND-2026-11: the repaired refined ODE route on chaotic systems it was not built on

**Status:** frozen before the run by `preregistration/DIGEST-ODE-BLIND.json`. Runs once, on GitHub
Actions (`.github/workflows/ode-blind.yml`), started by the first branch named `ode-blind-run-*`
whose shards start.

## What is tested

PREREG-ODE-REFINED-2026-11 ran the refined path with four searches side by side on ODEBench. It
recovered 92 and 73 of 117 components at noise 0 and 1 %, against 80 and 69 for the recorded ODE
route, and 10 of the 195 laws it reported were wrong (5.1 %) where 5 % was allowed. As fixed before
that run, it was not adopted.

Two repairs followed, both written with that run's outcomes known:

- **A guard against candidates the path cannot fit** (`apedp_dyn/proposers/fit_guard.py`, audit
  M-24). Two of the 234 runs ended on such a candidate.
- **Pruning to the rows' own precision** (`apedp_dyn/discovery_l2/precision_pruning.py`, audit
  M-15). The terms that made laws wrong were about as large as the error of the rate estimates.
  Applied after the fact to the 195 laws: 9 wrong instead of 10, 167 recovered instead of 165.

A repair fitted to known outcomes proves nothing. This run measures the route with both repairs on
systems that no route here was built on, run on, or looked at.

## Data and protocol

**Systems.** `benchmarks/dysts_polynomial_systems.json`, written by
`scripts/build_dysts_polynomial_systems.py` from `dysts` 1.0, the database of chaotic flows
(W. Gilpin, 2021) from which Kaptanoglu et al. (2023) drew their benchmark of sparse identification.
The rule, applied mechanically: an autonomous flow with three or four variables, an initial
condition on its attractor and a dominant period, whose right-hand side with its published
parameters is a polynomial of degree at most four with at most twelve terms per component. Left
out: ODEBench's own systems (Lorenz, Rössler, Aizawa), and five that have the same terms as an
ODEBench system with other coefficients (Chen and UnifiedChaotic as Lorenz, Langford as Aizawa,
LiuChen as Chen-Lee, VallisElNino as Maxwell-Bloch). 100 systems pass. For two of them (Malasoma,
SprottJerk) the frozen world builder does not produce rows within 120 seconds, most likely because
a trajectory started off the attractor does not stay bounded; they are not run. That leaves
**98 systems (82 with three variables, 16 with four), 310 components.** 41 components have a
cubic term and 2 a quartic one.

**Units and trajectories.** Each system is written in units of its own dominant period, so the
world builder's fixed horizon of ten time units is ten periods. Its two initial conditions are the
database's point on the attractor and where that point is 2.5 periods later.

**Rows, noise and scoring** are the frozen ODEBench runner's, unchanged
(`scripts/odebench_worlds.py`, `scripts/run_odebench_main_search.py`): two development
trajectories and a pool trajectory cut into windows, a state and a rate per window, noise 0 and
1 % of each variable's spread, test rows from the clean trajectories. A law is **reported** when
the engine returns one as confirmed. It is **recovered** when the benchmark's criterion says it is
the component's law, and **wrong** when it is reported, is not recovered, and does not reach
R² 0.999 on the test rows.

**Three engines** for every component and noise level, 1,860 runs in all:

- `repaired`: the refined path with four PySR searches side by side (240 s each, seeds as in
  PREREG-ODE-REFINED), behind the fit guard, with the reported law pruned to the rows' precision.
  The law before pruning is scored as well.
- `main_search`: the recorded ODE route, one search of 240 s, as run on ODEBench.
- `sparse_poly3`: sparse regression on a cubic library, the reference of the ODEBench run.

The time limit per run is 480 s, and 600 s for `repaired`.

## Expectations, fixed now

1. **What it reports can be relied on:** of the laws `repaired` reports in all 620 of its runs, at
   most 5 % are wrong.
2. **It recovers at least as much without noise:** at noise 0 it recovers at least as many of the
   310 components as `main_search`.
3. **And with noise:** the same at 1 %.
4. **Pruning costs nothing:** no law that is recovered before pruning is lost by it, and the
   number of wrong laws after pruning is not larger than before.
5. Reported without a criterion: the sparse reference, systems with every component recovered,
   proposals the guard dropped and in how many runs, laws changed by pruning, searches that
   failed, laws returned as tentative, the path's actions, timeouts and time.

The author's own expectations, recorded so that they can be wrong, and no criterion: 2 and 3 hold,
because a polynomial library is one of the refined path's candidates and these systems are
polynomial; 1 is uncertain at 1 % noise, where a chaotic trajectory leaves less of a rate estimate
than the smoother systems of ODEBench did; 4 holds; and the sparse reference recovers more
components without noise than either route.

A miss is reported as a miss.

## What follows from the result, fixed now

- If 1, 2 and 3 hold, the repaired route becomes the default ODE route, in a new module around
  the frozen ones. Its ODEBench result stays as recorded (5.1 % wrong, not adopted) and is stated
  beside this one wherever the route is described.
- If 1 misses, it is not adopted. The wrong laws are listed by system, and no further repair is
  called measured until it has been run on systems it was not written on.
- If 2 or 3 misses, the result is recorded as: on chaotic polynomial systems, four searches with
  the repairs recover no more than one search. It is not adopted.
- If 4 misses, pruning to the rows' precision is not used on trajectory rows, whatever else holds.

## Disclosed before freezing

- **A narrower class than ODEBench:** every system is chaotic, polynomial, and has three or four
  variables. A pass says the route is reliable on systems of this kind that it has not seen. It
  does not say so for rational or transcendental right-hand sides.
- **No discovery was run on any of these systems,** by any engine, before freezing. What was run
  on them: the selection rule, a comparison of their terms with ODEBench's, and the world builder
  without noise, to see whether rows can be built at all and how long it takes. What was seen:
  which systems share a form with ODEBench, which build, and the two that do not.
- **Choices made without evidence from these systems:** ten periods as the horizon (ODEBench's
  chaotic Lorenz system covers about seven of its periods in the same ten time units), the second
  initial condition 2.5 periods after the first, the limits of degree four and twelve terms, and
  the 120 seconds. The trajectories started between the two initial conditions lie off the
  attractor.
- **Both repairs were written with ODEBench's outcomes in view.** That is the reason for this run
  and the reason ODEBench cannot serve for it.
- **Four cores against one.** `repaired` runs four searches on four cores for the same wall time
  as `main_search`'s one. The comparison says what the route with its extra cores does, not that
  it is better at equal cost.
- **The sparse reference is on its home ground.** Sparse regression on a polynomial library is
  what these systems were collected to test. If it recovers more than both routes, that is
  reported as it falls. It reports a law for every component and has no way to decline, which is
  why it is a reference and not a route.
- **Not comparable with published numbers on these systems.** Kaptanoglu et al. score coefficient
  errors of sparse-regression variants on 70 of them under their own sampling. Nothing here was
  matched to that protocol.
- **The toy systems of the smoke job** have one or two variables and are the frozen runner's own.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.

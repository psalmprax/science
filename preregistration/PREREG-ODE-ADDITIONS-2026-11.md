# PREREG-ODE-ADDITIONS-2026-11: a second engine, adequacy at the rows' precision and a deciding experiment, on systems none of them was built on

**Status:** frozen before the run by `preregistration/DIGEST-ODE-ADDITIONS.json`. Runs once, on
GitHub Actions (`.github/workflows/ode-additions.yml`), started by the first branch named
`ode-additions-run-*` whose shards start.

## What is tested

PREREG-ODE-BLIND-2026-11 ran the default ODE route on 98 chaotic polynomial systems. It recovered
199 and 210 of 310 components (no noise, 1 % noise) with 3 wrong laws of 484, and it recovered
fewer components without noise than plain sparse regression did (255). Its rows were then looked
at, which made those systems development material, and three things were built from what they
showed:

- **A second engine** (`ode/second_engine.py`). In 50 of the 58 noise-free components only the
  sparse reference recovered, the law's terms had never been among the route's candidates. Sparse
  regression on a cubic library is added as a source of candidates beside the four searches.
- **Adequacy at the rows' precision** (`ode/precision_adequacy.py`, audit M-27). The path's last
  check demands residuals with no structure at all. Window rows of a trajectory carry rate
  estimates whose errors are a smooth function of the state, and the check refused the *true*
  law in 20 of the 28 runs it refused. A law refused there is reinstated if its residual is
  within twice the rows' own standard error.
- **A deciding experiment** (`ode/experiment_pruning.py`, audit M-28). Of the 78 laws the route
  with those two additions reported that were not the law, 65 were the law with terms added, tiny
  on the attractor. The
  system is restarted where the law with and without such terms would move differently, in a box
  twice the range seen; the two versions are compared with each other on how far the variable
  moved; terms are dropped only when the version without them predicts better by an exact
  sign-flip test over the restarts. A first form of the experiment, which added its rows to the
  pool, made the route worse and was retired before this text was written.

Replayed on those 98 systems with the candidates their searches had handed over
(`benchmarks/ode_additions_dev_replay_2.json`), recovered of 310 without noise and at 1 %:

| arm | no noise | 1 % | reported | wrong |
|---|---|---|---|---|
| `default` | 199 | 210 | 484 | 3 |
| `both` (second engine and precision) | 202 | 233 | 513 | 3 |
| `experiment` | 229 | 247 | 513 | 3 |
| `experiment_random` | 222 | 239 | 513 | 2 |
| sparse reference | 255 | 188 | 620 | 34 |

Every one of those numbers was produced on systems whose answers were known while the code was
written, and the experiment's settings were chosen on them. They show what was built, not that it
works. This run measures it on systems no route has run on.

## Data and protocol

**Systems.** `benchmarks/ode_second_blind_systems.json`, written by
`scripts/build_ode_second_blind_systems.py` with seed 20261124 by two mechanical rules:

- *30 invented systems* in three variables: `dx_i/dt = (A x)_i + quadratic terms - c_i x_i**3`,
  with a rotating linear part that is unstable at the origin and cubic damping that keeps
  trajectories bounded. A draw is kept if, started near the origin and left for 200 time units,
  it neither leaves a bound nor comes to rest and its strongest oscillation has a period between
  0.2 and 50.
- *16 rational systems* from `dysts` 1.0: the autonomous flows with a known period that the
  first blind set left out for being "not a polynomial" and whose right-hand sides are quotients
  of polynomials in the state. All sixteen have three variables.

Both parts are written in units of their own dominant period. 46 systems, 138 components, none
shared with the first blind set. Whether a rational system here also occurs in ODEBench was not
checked.

**Rows, noise and scoring** are the frozen ODEBench runner's, unchanged: four trajectories of ten
time units, rates from windows of 100 samples, noise 0 and 1 % of each variable's spread, a law
*recovered* if it is the law after refitting constants and *wrong* if it is reported, not the
law, and below R² 0.999 on the test trajectory.

**Six arms for every component and noise level, given the same candidates.** The default route's
four searches run once (240 seconds), and each arm is handed what they proposed:

- `default`: the route in force (`ode/default_route.py`);
- `second_engine`, `precision`: one addition each;
- `both`: the two together;
- `experiment`: `both`, then one follow-up experiment of 16 restarts chosen where the reported
  law and its shorter versions disagree most, one time unit each, the same sampling and noise;
- `experiment_random`: the same with the 16 restarts at random states of the same box.

`sparse_poly3`, the frozen runner's sparse regression, is reported beside them. 276 units, seven
rows each. A unit that crashes or exceeds 1,800 seconds counts as recovering and reporting nothing
in every arm.

## Expectations, fixed now

1. **What is reported can be relied on:** of the laws `both` reports, and of those `experiment`
   reports, at most 5 % are wrong.
2. **The two additions do not cost recoveries:** `both` recovers at least as many components as
   `default` at each noise level.
3. **The experiment helps:** `experiment` recovers at least as many components as `both` at each
   noise level, more in total, and reports no more wrong laws.
4. **Choosing where to measure matters:** `experiment` recovers more components in total than
   `experiment_random`.
5. Reported without a criterion: each arm for the invented and the rational systems apart; the
   sparse reference; laws gained and lost run for run; laws reinstated; experiments that ran, how
   many restarts stayed bounded, and laws shortened; errors, timeouts and time.

**The system's own forecast** (`preregistration/FORECAST-ODE-ADDITIONS-2026-11.json`, filed with
this text) for the `default` arm: 90 of 276 recovered (51 to 129 at 80 %), 138 reported, 25 of
them wrong (0 to 63). In its backtest the forecaster did not carry over from one kind of
dynamical system to another, and these are two kinds it has not seen.

**The author's own expectations**, recorded so that they can be wrong, and no criterion. The
forecast is too low on recoveries and far too high on wrong laws: `default` recovers about 145
(two thirds of the invented systems' components, a third of the rational ones') and reports
fewer than 5 wrong laws. 1 holds. 2 holds at 1 % noise by a clear margin and without noise by one
or two components or a tie, so perhaps one chance in five that it misses. 3 holds: about 15 more
recoveries, mostly on the invented systems; on the rational ones the experiment has less to do,
because laws reported there are approximations of another form and not laws with terms added.
4 is the weakest: chosen restarts beat random ones by 15 of 620 in development, which on 276
units is a margin of about 7, and random restarts in a box twice the range seen already expose
most added terms. I put it at two chances in three.

A miss is reported as a miss.

## What follows from the result, fixed now

- If 1 and 2 hold, the second engine and adequacy at the rows' precision become part of the
  default ODE route, in a new module around the frozen ones.
- If 1 and 3 hold, the experiment becomes the default follow-up wherever the caller can restart
  the system. If 4 misses as well, it is adopted with random restarts described as doing as well
  as chosen ones, and the choosing is not claimed.
- If 1 misses for an arm, nothing in that arm is adopted, and its wrong laws are listed by system
  with what the additions did in each.
- If 2 misses, the result is recorded as: on these systems the two additions recover no more than
  the route in force. They are not adopted.
- If 3 misses, the experiment is not adopted, in either form.
- The forecast is scored against the `default` arm whatever else happens.

## Disclosed before freezing

- **Everything tested here was developed on the first blind set's rows, in three looks.** The
  first found the two causes above; a first replay measured the first two additions and the
  experiment's first form; a second replay measured the experiment's second form, whose settings
  (a box twice the range seen, 16 restarts, changes of the state over 100 samples, terms tried
  singly, as the smallest group and in pairs) were chosen on those rows.
- **No discovery was run on any system of this set,** by any engine, before freezing. What was
  run on them: the selection rules, and the world builder without noise to see which systems
  build and stay bounded.
- **The invented systems are cubic polynomials,** which is exactly the second engine's library
  and the reference's. The rational systems are the opposite case: there a polynomial cannot be
  the law, and the second engine can only add approximations. Both parts are reported apart.
- **The experiment needs the system itself.** Here that is a simulation that can be started at
  any state. Most observed systems cannot be, and a restart outside the range seen assumes the
  law holds there, which is true of these systems by construction.
- **Unequal data.** The experiment arms see 16,000 samples more than the 40,000 the other arms
  see. The comparison between `experiment` and `experiment_random` is at equal data; the
  comparison between `experiment` and `both` is not.
- **The same candidates for every arm.** The arms differ in what they do with them and in
  nothing a second search happened to find.
- **Four cores against one core's worth of sparse regression.** The reference reports a law for
  every component and cannot decline; it is a reference, not a route.
- **The machinery itself varies by about one law.** The two development replays ran three arms
  on identical inputs on different machines and differ by one recovery in each. A difference of
  one component between arms is not a finding.
- **The first pass is the reference one.** The native kernel (`discovery_l2/fast_first_pass.py`)
  is not used in this run.
- **The toy systems of the smoke job** have one or two variables and are the frozen runner's own.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.

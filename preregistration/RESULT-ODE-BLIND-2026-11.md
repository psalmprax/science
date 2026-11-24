# Result: PREREG-ODE-BLIND-2026-11

**Run:** once. GitHub Actions run `44840100795` on the branch `ode-blind-run-2` at commit `7500fbd`,
which carries the frozen preregistration and harness (digests repeated in `results.json`, and equal
to `DIGEST-ODE-BLIND.json`). The warm-up job ran on 2026-11-20 from 23:54 UTC; the 48 shards ran on
2026-11-21 from 00:15 to 06:31 UTC and the collect job at 06:32. Every job succeeded on its first
attempt; no shard was re-run.

**A first launch was cancelled before anything ran.** The branch `ode-blind-run-1` (run
`44837787636`) was pushed at 23:34 UTC and cancelled at 23:51 during its warm-up job, in the
mistaken belief that the rows would not record the candidate laws. They do. No shard had started
and no system had been run. The preregistration names "the first branch named `ode-blind-run-*`
whose shards start", which is `ode-blind-run-2`.

Outputs, unedited, in [`RESULT-ODE-BLIND-2026-11/`](RESULT-ODE-BLIND-2026-11/): `results.json`;
`runs.jsonl.gz`, the collect job's `runs.jsonl` (1,860 runs) compressed and otherwise untouched;
and `SHA256SUMS`, the checksums the job printed for its 50 files (these two and the 48 shard files,
which hold the same 1,860 rows and are not committed). The files were rebuilt from the job's log
and all 50 match.

## Outcome

**All four expectations held. As fixed before the run, the repaired route becomes the default ODE
route.**

98 chaotic polynomial systems from the `dysts` database, 310 components × 2 noise levels (0 and
1 %), that no route here had been built on, run on or looked at.

| | Law reported | Reported and wrong | Recovered, noise 0 | Recovered, noise 1 % | Systems with every component recovered |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Repaired route** (four searches, fit guard, pruning) | 484 of 620 | **3 (0.6 %)** | **199** | **210** | 50 and 28 of 98 |
| Recorded ODE route (one search) | 456 | 3 (0.7 %) | 132 | 201 | 25 and 27 |
| Sparse regression on a cubic library (the reference) | 620 | 34 (5.5 %) | 255 | 189 | 75 and 26 |

Of 310 components at each noise level. 95 % intervals (Clopper-Pearson) for wrong among reported:
0.1–1.8 % for the repaired route, 0.1–1.9 % for the recorded route, 3.8–7.6 % for the reference.

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | Of the laws the repaired route reports in all 620 runs, at most 5 % are wrong | 3 of 484, 0.6 % | **held** |
| 2 | Without noise it recovers at least as many of the 310 components as the recorded route | 199 against 132 | **held** |
| 3 | The same at 1 % noise | 210 against 201 | **held** |
| 4 | Pruning loses no recovery, and leaves no more wrong laws than before it | 0 lost; 3 wrong before and after | **held** |

The author's own expectations, recorded before the run: 2, 3 and 4 hold (they did); 1 is uncertain
at 1 % noise (it held, with 2 wrong of 249 reported at that level); the sparse reference recovers
more without noise than either route (it did, 255).

**What follows, as fixed before the run:** the repaired route is the default ODE route, in a new
module around the frozen ones: `apedp_dyn/ode/default_route.py`. Its ODEBench result stays as
recorded (PREREG-ODE-REFINED-2026-11, without the two repairs: 10 of 195 reported laws wrong,
5.1 %, not adopted) and is stated beside this one wherever the route is described. Pruning to the
rows' precision stays in use on trajectory rows.

## Reported without a criterion

- **The sparse reference:** without noise it recovers more than both routes (255); at 1 % noise
  fewer than both (189). It reports a law for every component and 34 of them are wrong, 11 without
  noise and 23 at 1 %.
- **Proposals the guard dropped:** two, in one run (`SprottF`, component 0, 1 % noise): two
  candidates the path's least-squares fit raises on. The recorded route has no guard, and its run
  on the same component at the same noise level is the one run of the 1,860 that ended in an
  error, the fit not converging. The repaired route recovered the law there.
- **Laws changed by pruning:** 42 (39 without noise, 3 at 1 %). 24 became recoveries, all without
  noise; none was lost; the pruning raised no error.
- **Searches that failed:** none, in 620 runs. **Tentative laws:** none. **Timeouts:** none.
- **The path's actions (repaired):** `confirmed` 6, `equivalent_class` 478, `widen_search` 136
  (75 without noise, 61 at 1 %). The reported law came from the search in 452 runs and from the
  first pass in 32. Terms were dropped from the confirmed form in 55 runs.
- **Time:** median 276 s per run for the repaired route, the slowest 459 s (limit 600 s); 262 s
  and 289 s for the recorded route; the reference takes a fraction of a second.

## After the run (not preregistered)

**Run for run against the recorded route.** Without noise: 67 components gained, none lost. At
1 %: 15 gained, 6 lost.

**The three wrong laws.**

| System, component | Noise | Reported | The law | Test R² |
| :--- | ---: | :--- | :--- | ---: |
| `Bouali2`, 2 | 1 % | `0.0030*x0*cos(x0) + 0.0293*x0` | `0.0307*x0` | 0.9975 |
| `HindmarshRose`, 0 | 1 % | a cubic with two cross terms the law does not have | `-28.6*x0**3 + 58.4*x0**2 - 1.75*x0 + 58.4*x1 + ...` | 0.9986 |
| `SprottE`, 0 | 0 | `7.6275*x1*x2` plus `1.6e-5*x1**5*exp(sin(x2))*sin(...)` | `7.6275*x1*x2` | 0.9985 |

The first is also one of the recorded route's three, with the same form. The third is a term that
is small on the rows and not small beyond them, which pruning to the rows' precision did not
remove.

**One recovery is not a usable law (audit M-26).** `SprottE`, component 1, without noise: both
routes report `7.6275*x0**2*cos(...) - 7.6275*x1*cos(...)`, where each cosine's argument is a
polynomial with coefficients near 1e-4. The benchmark's criterion refits every constant, those
inside a function too; with them at zero the cosines are 1 and the form is the law, so it counts
as recovered. As reported it has R² −0.80 on the test rows. It is in the 199 and in the 132.
Counted as wrong instead, the repaired route has 4 wrong of 484 (0.8 %) and 198 recoveries, the
recorded route 4 of 456 and 131; no expectation changes. The reference recovers that component.

**No recovery passes the tolerance narrowly** (audit M-23): every recovery of both routes has a
refit error below 1e-10.

**By the degree of the law** (recovered, repaired route, noise 0 and 1 %): linear 75 and 70 of 97;
quadratic 104 and 119 of 170; cubic 20 and 21 of 41; quartic 0 of 2. The reference: 89 and 63,
139 and 106, 27 and 20, 0.

**More is recovered with 1 % noise than without.** This holds for both path routes (199 against
210; 132 against 201) and not for the reference. Without noise the path refuses more often (75
runs against 61), and every one of the 35 laws it reports that are accurate and are not the law
carries a superfluous term. A likely reason, not established here: without noise the only error
in a rate is that of the window fit, which the path has no model of, so it is fitted. The pruning
is what recovers 24 of these.

**Against the reference, component by component.** Without noise the reference recovers 58
components the repaired route does not, and the route 2 the reference does not. At 1 % the route
recovers 43 the reference does not, the reference 22. One of the three engines recovers 257 of the
310 components without noise and 237 at 1 %.

**Reported, accurate, and not the law:** 72 for the repaired route, 120 for the recorded one.

## What this says, and what it does not

On chaotic polynomial systems it had never seen, what the repaired route reports can be relied
on: 3 wrong laws in 484, where the same route without the two repairs had 10 in 195 on ODEBench.
It recovers more than the recorded route at both noise levels, by a wide margin without noise,
where pruning to the rows' precision accounts for 24 of the 67 gains.

It does not say the route is reliable on rational or transcendental right-hand sides, where
ODEBench's wrong laws were: every system here is a polynomial. It ran four searches on four cores
for the wall time of the recorded route's one. And on noise-free rows of polynomial systems,
sparse regression on a polynomial library recovers 56 more components in a fraction of a second;
it also reports a law for every component, wrong in one of eighteen.

# What has been measured, and where the system stands

Last updated 2026-11-22. This page is the short answer to "what does this system do, and how do you
know". Every line points at a preregistered run: the question, the method and the pass criteria were
frozen by digest before the run, the run was made once, and the outputs are committed unedited under
[`preregistration/`](../preregistration/). Where an expectation was missed, the result file says so
and so does this page. The register of known defects is
[`SYSTEM_FLAWS_AUDIT.md`](SYSTEM_FLAWS_AUDIT.md).

## The discovery routes

For tabular data the default is the **refined path** since 2026-11-20
(`apedp_dyn/discovery_l2/default_route.py`; the owner's decision on the result below, one of whose
three expectations was missed by one law). The main search was the default from 2026-11-19 and can
still be chosen, as can the template engine. Trajectories and return-like series have their own
entry points (`apedp_dyn/ode/route_in_force.py`: since 2026-11-21 the refined path with four searches,
the fit guard and pruning to the rows' precision, on the first blind result below; since 2026-11-22
with sparse regression beside the searches and, where the system can be started again, a follow-up
experiment, on the second; `apedp_dyn/discovery_l2/series_route.py`). Since 2026-11-22 the table
route's searches have a memory of reported laws beside them (`apedp_dyn/discovery_l2/stage_in_force.py`),
and a law the table route is about to report has to pass a last set of checks
(`apedp_dyn/discovery_l2/vouched_route.py`, on the counterfeit benchmark's result below); one that
does not is shown as not vouched for, with the reason. **Their cost, stated first as fixed before the run that measured it:** on the physics benchmark behind the four searches they withheld 43 of the 193 laws the route recovered (47 of 90 at 1 % noise) and stopped no inaccurate law, because there was none (PREREG-VOUCHED-SRBENCH, third expectation missed; audit M-35). They stay in force until the owner decides.

| Route | For | What it does | Measured | Result file |
| :--- | :--- | :--- | :--- | :--- |
| **Refined** (the default since 2026-11-20; `discovery_l2/refined_search.py`) | Tabular data, with physical units where the problem has them | The main search with the units' dimensionless products, a polynomial library, a report in the confirmed form with unnoticed terms dropped, and the search stage on every problem | SRBench: a law reported in 212 of 266 runs, **none inaccurate** (interval 0–1.7 %), **174 the law**. Feynman 78 and 77 of 119; Strogatz 19 of 28. On 103 held-out problems 61 and 61 against the main search's 55 and 52: one short of the 62 expected at noise 0. It uses units, which the other rows were not given. **With four different searches side by side as the search stage** (`discovery_l2/search_stage.py`, PREREG-PORTFOLIO-SEARCH): 189 recovered, 1 of 224 reported laws not accurate; on the 103 held-out problems 73 and 63 against 61 and 61. All four expectations held; it is the search stage wherever four cores are available. **With a memory of reported laws beside the searches** (`discovery_l2/stage_in_force.py`, PREREG-MEMORY, in force since 2026-11-22): on 40 laws drawn after the memory was written, 23 of 80 runs recovered against 17, six gained, none lost, no wrong law in either arm; nearly all of the gain without noise, and on laws of the one grammar the memory's way of joining parts was written for. **Handed counterfeits of 60 new laws, with no search engine** (PREREG-COUNTERFEIT): no wrong law in 141 reports at noise up to 1 %, **16 in 88 at 5 % noise**, which is beyond what the rows above measured. **With the last checks** (`discovery_l2/vouched_route.py`, in force since 2026-11-22): **on SRBench behind the four searches they keep 150 of the 193 laws the route recovers (47 of 90 at 1 % noise) and stop nothing, since nothing reported there was inaccurate** (PREREG-VOUCHED-SRBENCH; one expectation missed; the owner decides whether they stay). On counterfeits of 60 new laws: 2 wrong in 55 at 5 % noise, and 139 of the route's 146 right laws kept | [`RESULT-REFINED-SEARCH`](../preregistration/RESULT-REFINED-SEARCH-2026-11.md), [`RESULT-PORTFOLIO-SEARCH`](../preregistration/RESULT-PORTFOLIO-SEARCH-2026-11.md), [`RESULT-MEMORY`](../preregistration/RESULT-MEMORY-2026-11.md), [`RESULT-COUNTERFEIT`](../preregistration/RESULT-COUNTERFEIT-2026-11.md), [`RESULT-VOUCHED-SRBENCH`](../preregistration/RESULT-VOUCHED-SRBENCH-2026-11.md) |
| Main search (the default from 2026-11-19 to 2026-11-20) | Tabular data where a law explains nearly all of the target | Exhaustive first pass, then PySR, then confirmation on rows chosen to tell rivals apart, then an adequacy check | SRBench, 133 laws at noise 0 and 1 %: a law reported in 186 of 266 runs, **4 not accurate (2.2 %**, interval 0.6–5.4 %), 152 the law. Feynman 69 and 70 of 119; PySR alone 73 and 68 | [`RESULT-MAIN-SEARCH`](../preregistration/RESULT-MAIN-SEARCH-2026-11.md) |
| **ODE** (in force since 2026-11-22: `ode/route_in_force.py`) | Trajectories of a dynamical system | The route of the next row with sparse regression on a cubic library as a second source of candidates beside the four searches; and, where the caller can start the system again, a follow-up experiment of sixteen restarts that drops terms of the reported law only when the law without them predicts better | **46 systems none of it was built on** (30 invented cubic, 16 rational; 276 components and noise levels, PREREG-ODE-ADDITIONS): with the second engine **117 recovered, 3 of 165 reported laws wrong (1.8 %)** against 116 and 4 of 164 for the next row's route; with the experiment as run (on top of a reinstatement that is **not** in force) 123 and 6 of 188; the second engine and the experiment without that reinstatement, read off the run's rows unit by unit: **120 recovered, 3 of 165 wrong**. All four expectations held, the fourth by two components. **The gains are small**, and at 1 % noise two of every three laws reported are accurate and are not the law. One addition the preregistration would have adopted is held back, a deviation recorded with the result: on its own it added 23 reported laws, 1 of them the law and 3 wrong | [`RESULT-ODE-ADDITIONS`](../preregistration/RESULT-ODE-ADDITIONS-2026-11.md) |
| ODE, the route in force from 2026-11-21 to 2026-11-22 (`ode/default_route.py`) | Trajectories of a dynamical system | One state and one rate per disjoint window; then, per component, the refined path with four searches side by side behind a guard that drops candidates the path cannot fit, and the reported law pruned to the precision of the rate estimates | **98 chaotic polynomial systems it had never seen** (`dysts`, 310 components, PREREG-ODE-BLIND): **3 of 484 reported laws wrong (0.6 %**, interval 0.1–1.8 %, limit 5 %); **199 and 210 recovered** (noise 0, 1 %) against 132 and 201 for the route below; all four expectations held. Sparse regression on a cubic library, the reference: 255 and 189, 5.5 % wrong. **The same route without the guard and the pruning, on ODEBench** (63 systems, 117 components, PREREG-ODE-REFINED): 92 and 73 recovered, 10 of 195 reported laws wrong (5.1 %) against a limit of 5 %, **not adopted there**; that result stands. The blind systems are all polynomial: nothing is measured with the repairs on rational or transcendental right-hand sides | [`RESULT-ODE-BLIND`](../preregistration/RESULT-ODE-BLIND-2026-11.md), [`RESULT-ODE-REFINED`](../preregistration/RESULT-ODE-REFINED-2026-11.md) |
| ODE, the recorded route (the default until 2026-11-21; `ode/main_search_ode.py`) | Trajectories of a dynamical system | The same rows, then the main search per component with one search | ODEBench: **80 and 69 recovered**; 9 of 184 reported laws wrong (4.9 %). The sparse polynomial reference there: 78 and 53, 29.1 % wrong. On the 98 blind systems: 132 and 201 recovered, 3 of 456 wrong, and one run ended on a fit that did not converge | [`RESULT-ODE-MAIN-SEARCH`](../preregistration/RESULT-ODE-MAIN-SEARCH-2026-11.md), [`RESULT-ODE-BLIND`](../preregistration/RESULT-ODE-BLIND-2026-11.md) |
| **Series** | Noisy series in time order (return-like data) | Candidates from the main search, one chosen inside the development period, one forecast test on later rows | Simulated series with a planted law (not market findings): a law reported in **0 of 40** series that contain none (the main search alone: 28 of 40); 31 of 40 where the law explains 5 or 20 % of the variance; 3 of 20 at 2 %; 0 of 20 at 0.5 % | [`RESULT-MARKET-PLANTED`](../preregistration/RESULT-MARKET-PLANTED-2026-11.md) |
| Template engine with the referee (`l2`) | Kept for replication, and for laws inside its grammar | Enumerates dimensionally valid expressions; the referee promotes one | SRBench: 268 promotions, **160 not accurate**. **With Guard 3** (`epistemic/structure_guard.py`): 113 promotions, **3 not accurate**, 103 correct against 100; on the 103 held-out problems 3 of 81 not accurate and 71 of 72 correct promotions kept. Two of three expectations held; the third (at most 10 % false by the symbolic criterion) was missed at 12 %. On laws its grammar can express (blind benchmark, no guard): 7 false in 288 | [`RESULT-SRBENCH-BASELINE`](../preregistration/RESULT-SRBENCH-BASELINE-2026-11.md), [`RESULT-REFEREE-STRUCTURE`](../preregistration/RESULT-REFEREE-STRUCTURE-2026-11.md), [`RESULT-BLIND-L2`](../preregistration/RESULT-BLIND-L2-2026-11.md) |

Other measured engines: partial differential equations in two space dimensions, 117 of 120 runs exact
([`RESULT-PDE-2D`](../preregistration/RESULT-PDE-2D-2026-11.md)); lagged causal graphs, precision
0.97 and recall 0.99 on linear series
([`RESULT-GRAPHS`](../preregistration/RESULT-GRAPHS-2026-11.md)); contrast confirmation, the true
law confirmed in 63 of 80 runs and a wrong law named alone in none
([`RESULT-CONTRAST`](../preregistration/RESULT-CONTRAST-2026-11.md)).

**Retired as law-discovery engines** (kept importable for replication): RIME-AD (11 of 532 SRBench
runs recovered, 62 % timed out) and the genetic search (53 and 7 of 119 Feynman laws against 73 and 68
for PySR). In the same baseline the extended hierarchical engine recovered 36 of 532 runs and RIME 24.

## Search guidance

| What | Measured | Result file |
| :--- | :--- | :--- |
| A language model writing the candidate laws, in the place PySR has (`nemotron-3-ultra` on NVIDIA's free API, given numbers only) | 92 laws reported in 266 runs, 2 not accurate, **78 recovered against 152 with PySR**; 12 found by the model. It stays an optional proposer | [`RESULT-LLM-PROPOSER`](../preregistration/RESULT-LLM-PROPOSER-2026-11.md) |
| A language model configuring the search (operators, features, guesses) inside the refined path, same model | 207 laws reported, 1 not accurate, **173 recovered against 174 without the model**; held-out 123 against 122. All three expectations held, and the finding is that the model adds nothing by itself. The two runs recover different laws: 191 of 266 by one or the other | [`RESULT-LLM-CONTROLLER`](../preregistration/RESULT-LLM-CONTROLLER-2026-11.md) |
| A learned prior over formulas ordering the exhaustive search (MMLM) | The true law is reached after about a twentieth of the candidates; with a confirmed early stop, 96 of 100 laws recovered while fitting a median 3.2 % of the candidates | [`RESULT-MMLM-S`](../preregistration/RESULT-MMLM-S-2026-11.md), [`-S2`](../preregistration/RESULT-MMLM-S2-2026-11.md), [`-L`](../preregistration/RESULT-MMLM-L-2026-11.md) |
| The mixture language model as a compressor (MMLM-CM) | 1.455 bits per byte on the WikiText-2 test split against 1.513 for the best reference compressor | [`RESULT-MMLM-CM`](../preregistration/RESULT-MMLM-CM-2026-11.md) |

## Quantum circuits that tolerate a failing gate

`apedp_dyn/vqc/fault_tolerant_prep.py` certifies fault tolerance by enumerating every single fault and
every pair, and searches with that certificate. For the Steane code: 22,976 shortest encoders
(8 CNOTs) up to symmetry, **none fault-tolerant alone**; with one ancilla check, 275,712 certified
circuits at 11 CNOTs. The second-order error coefficient is exact (best 19.12, median 21.40, worst
27.88) and agrees with Stim sampling at p = 0.001 within 2.2 %. The unprotected encoder is 120 times
worse. On a 2 × 4 grid the best certified circuit has 14 CNOTs.
[`RESULT-FT-PREP`](../preregistration/RESULT-FT-PREP-2026-11.md). It covers state preparation for
one distance-three code; it is not a compiler and says nothing about larger codes.

## The promotion referee on market data

On real crypto returns 0 of 500 placebo signals passed stages 2 and 3
([`RESULT-CRYPTO-CAL`](../preregistration/RESULT-CRYPTO-CAL-2026-11.md)). On the point-in-time panel
the positive control passed stage 4, the null control was refused at stage 2, and the one test claim
(order-flow imbalance) was **refused at stage 2**
([`RESULT-CRYPTO-PIT`](../preregistration/RESULT-CRYPTO-PIT-2026-11.md)). Nothing here is evidence of
a tradeable effect.

## Published laws on real data: frozen, not run

[`PREREG-LAW-AUDIT-FINANCE-2026-11`](../preregistration/PREREG-LAW-AUDIT-FINANCE-2026-11.md) takes five
published laws of asset returns (heavy tails with an index between two and five, and its strict
reading, the inverse cubic law; no autocorrelation of returns; a slow power-law decay of the
autocorrelation of absolute returns; the leverage effect; volume moving with volatility) and
measures each on the point-in-time crypto panel, 2021 to 2025, as one number with an interval that
allows for market-wide days and wandering volatility. Each gets `fails`, `survives` or
`not_rejected`. **It is frozen and has not been run**; it waits for the repository owner.

What has been measured is what the audit can tell, on 2,700 simulated panels of the real one's
shape (`preregistration/POWER-LAW-AUDIT-FINANCE-2026-11/`):

| Claim | Where it is true | Where it is false |
| :--- | :--- | :--- |
| Tail index between 2 and 5 | failed in 0 of 2,400 panels | a reading of 1.66 fails in 12 of 150: the panel cannot tell 1.7 from 2 |
| Inverse cubic range | failed in 0 of 150 | fails in every panel where the number reads 2.0 or less, in none from 2.47 up |
| Variance ratio of one | failed in 7 of 2,100 | a ratio about 0.09 from one fails in 68 and 102 of 150; about 0.05 from one, in 2 and 17 |
| Decay exponent between 0.2 and 0.4 | failed in 3 of 1,950 | 0.71 fails in 40 of 150; 0.09, and an exponential decay, in 1 of 150 each: five years cannot measure it to the published precision |
| Leverage negative | failed in 0 of 2,250, survives in 84 % | the opposite sign fails in 130 of 150 |
| Volume with volatility | survives in all 2,400 | no link: `not_rejected` in 148 of 150 |
| Control: independent returns | (1 of 150 fails) | fails in all 2,550 |

10 of the 2,700 panels failed some law that was true in their world. The block-resampling
interval alone covered 92 to 99.7 % of the time where it claims 99 %, so each interval is widened
by a factor measured on 1,500 other simulated panels (2.3 for the tail index, 1.1 to 1.5 for the
rest); those factors are as good as the simulated panels are like a market.

**A second batch, frozen the next day and also not run:**
[`PREREG-LAW-AUDIT-FINANCE-MORE-2026-11`](../preregistration/PREREG-LAW-AUDIT-FINANCE-MORE-2026-11.md)
adds the four remaining facts of Cont's list that can be given a number on daily data, and two
published claims about what predicts returns. It runs in the same job as the first batch, after
it. Every claim is a sign. On 2,400 simulated panels of the real one's shape
(`preregistration/POWER-LAW-AUDIT-FINANCE-MORE-2026-11/`):

| Claim | Where it is true | Where it is not |
| :--- | :--- | :--- |
| Large moves are more often losses than gains | failed in 1 of 600; 56 % losses survives in 78 of 150, 53 % in 8 | the reverse fails in 125 of 150; survived in 1 of 1,800 |
| Closer to normal over a week than over a day | survives in 1,922 of 1,950 | stable-law returns (Mandelbrot's rival claim): survived in 4 of 300, never failed |
| Tails less heavy once volatility is corrected for | survives in 1,640 of 1,800 | constant volatility: fails in 599 of 600 |
| ... and still heavy | survives in all 2,250 | independent normal returns: not rejected in all 150 |
| Coarse volatility predicts fine better than the reverse | survives in 80 of 150 at the stronger setting, 27 at half of it | the reverse fails in 150 and 71 of 150; survived in 4 of 1,800 |
| Momentum across assets, one to four weeks | a rank correlation of 0.07, about the published size, survives in all 150; 0.02 in 87; 0.004 in 4 | reversal of 0.045 fails in 149 of 150; survived in 1 of 1,650 |
| A day's order imbalance and the next day's return | 0.034 survives in 142 of 150; 0.017 in 101 | the reverse fails in 148 and 101 of 150; survived in 10 of 1,800, 5 of them where momentum makes it real |

One of the 2,400 panels failed a claim that was true in its world. The intervals are widened by
1.2 to 1.8, measured on 2,100 other simulated panels. In the worlds where momentum or order flow
is present its size depends on how turbulent the five years are: a verdict here is about the sign
in this period, and the estimate is not a long-run constant.

## The repaired ODE route on unseen systems: all four expectations held

[`PREREG-ODE-BLIND-2026-11`](../preregistration/PREREG-ODE-BLIND-2026-11.md), run once on 2026-11-21
([result](../preregistration/RESULT-ODE-BLIND-2026-11.md)): the refined ODE path with four searches,
the fit guard and pruning to the rows' precision, on 98 chaotic polynomial systems (310 components)
from the `dysts` database that no route here was built or run on.

| | Law reported | Reported and wrong | Recovered, noise 0 | Recovered, noise 1 % |
| :--- | ---: | ---: | ---: | ---: |
| Repaired route | 484 of 620 | 3 (0.6 %) | 199 of 310 | 210 of 310 |
| Recorded route | 456 | 3 (0.7 %) | 132 | 201 |
| Sparse regression on a cubic library | 620 | 34 (5.5 %) | 255 | 189 |

As fixed before the run, the repaired route became the default ODE route
(`apedp_dyn/ode/default_route.py`; since 2026-11-22 it is the base of `ode/route_in_force.py`,
below). Three things the result note records beside the pass:

- **One recovery is not a usable law** (M-26): the benchmark's criterion refits constants inside
  functions too, and one law with two cosine factors counts as recovered though it is inaccurate as
  reported. Counted as wrong: 4 of 484.
- **The guard met its case once**: on the one component where it dropped candidates, the recorded
  route, which has none, ended in an error.
- **Without noise, sparse regression recovers 56 more components** than the route, in a fraction of
  a second; it reports a law for every component and one in eighteen is wrong. Every system here
  is polynomial, which is that method's home ground.

A first launch, on `ode-blind-run-1`, was cancelled during its warm-up job, before any shard
started and so before any system was run; the run is the one on `ode-blind-run-2`.

## Three additions to that route on a second unseen set: all four expectations held, by little

[`PREREG-ODE-ADDITIONS-2026-11`](../preregistration/PREREG-ODE-ADDITIONS-2026-11.md), run once on
2026-11-22 ([result](../preregistration/RESULT-ODE-ADDITIONS-2026-11.md)): 46 systems no route had
run on (30 invented cubic, 16 with rational right-hand sides), 138 components at noise 0 and 1 %,
six arms handed the same searched candidates.

| | Recovered, noise 0 | Recovered, 1 % | Law reported | Reported and wrong |
| :--- | ---: | ---: | ---: | ---: |
| The route in force until then | 94 of 138 | 22 of 138 | 164 of 276 | 4 (2.4 %) |
| With a second engine and adequacy at the rows' precision | 94 | 24 | 188 | 6 (3.2 %) |
| With the follow-up experiment as well | 99 | 24 | 188 | 6 (3.2 %) |
| The same with restarts at random | 97 | 24 | 188 | 6 (3.2 %) |
| Sparse regression on a cubic library | 95 | 6 | 276 | 127 (46 %) |

What the development systems had promised did not carry over: there the two additions were worth
26 components of 620 and the experiment a further 41; here 2 and 5 of 276. Read from the rows:

- **Adequacy at the rows' precision reinstated 23 laws and one of them is the law**; 19 are
  accurate without being it and 3 are wrong. It is **not** in force, which is a deviation from what
  the preregistration fixed, on the cautious side, recorded with the result.
- **The second engine and the experiment are in force** (`ode/route_in_force.py`). Without the
  reinstatement their outcome on this run is known unit by unit: 120 recovered, 165 reported, 3
  wrong.
- **At 1 % noise two of every three laws the route reports are accurate on the test trajectory and
  are not the law**, in every arm. The experiment changed the outcome of no run at that noise.
- **Choosing where to restart beat random restarts by two components**, inside what the machinery
  varies by. It is not claimed to matter.

## A memory of reported laws beside the searches, on laws drawn afterwards: both expectations held

[`PREREG-MEMORY-2026-11`](../preregistration/PREREG-MEMORY-2026-11.md), run once on 2026-11-22
([result](../preregistration/RESULT-MEMORY-2026-11.md)): 40 laws of the invented-law grammar drawn
after the memory file was written, noise 0 and 1 %, both arms handed the same searched candidates.

| | Recovered, noise 0 | Recovered, 1 % | Law reported | Reported and wrong | Accurate, not the law |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Four searches | 12 of 40 | 5 of 40 | 55 of 80 | 0 | 38 |
| The same and the memory's three laws | 17 | 6 | 57 | 0 | 34 |

Six laws gained, none lost. As fixed before the run the memory now stands beside the table route's
searches (`discovery_l2/stage_in_force.py`). Beside the pass:

- **The gain is on laws of one grammar**, which the memory's way of joining parts was written
  knowing. Most laws of the physics benchmarks have several variables inside one function and
  cannot be grown from its one-variable parts; nothing is measured there.
- **At 1 % noise the memory handed the route the law in 11 runs and the route reported it in 6.**
  In the other five it could not tell its candidates apart and named a shorter one, accurate and
  not the law.
- **At 1 % noise 31 of the 36 laws the default route reports are accurate and are not the law.**
  None counts as wrong by the benchmark's criterion.

## A search for data that defeats the route, and the checks built from what it found

**Development by its nature: the worlds were chosen to defeat the route, so nothing here is a
rate on data nobody designed.** `scripts/run_red_team.py`, run once on 2026-11-22 (GitHub Actions
run `44963722101`): 34,728 worlds (`benchmarks/red_team_campaign.json`, every world in
`benchmarks/red_team_worlds.jsonl.gz`). A world is a law, a list of candidates said to come from a
search (counterfeits of the law, with or without the law), a way of drawing rows and a noise.
Half the worlds were drawn at random; half were changed step by step towards the route's failures.

| Worlds drawn at random | Laws reported by the route | Wrong | The candidate that fits best: wrong |
| :--- | ---: | ---: | ---: |
| The conditions of the preregistered runs (noise up to 1 %, 1,000 rows, uniform, Gaussian) | 2,198 | **1 (0.05 %)** | 915 of 3,465 |
| Fewer rows | 1,238 | 34 (2.7 %) | 427 of 1,771 |
| Rows that do not cover the box evenly | 1,641 | 18 (1.1 %) | 403 of 2,288 |
| Noise of another shape than the path is told | 906 | 4 (0.4 %) | 469 of 1,731 |
| **More noise than the runs used** | 1,958 | **312 (15.9 %)** | 636 of 2,403 |

By noise, under the runs' other conditions: none, 0 of 427; up to 0.3 %, 0 of 819; 0.3 to 1 %, 1 of
952; 1 to 3 %, 44 of 901 (4.9 %); **3 to 10 %, 268 of 1,057 (25 %)**. The record of the
preregistered runs holds where they measured and does not hold above about 3 % noise.

On the worlds changed towards failure the route reported a wrong law in 588 of 2,569 reports under
the runs' own conditions, and in most reports elsewhere. Two causes account for nearly all of it
(audit M-31, M-32): where the path cannot tell candidates apart it reports the one with the
shortest description (11,929 of the 13,426 wrong laws, the law itself having been offered in
5,365), and its last check lets through an error a third the size of the noise.

**The checks built from it** (`discovery_l2/vouching.py`): a structure test with a known null at
three scales; members of an unseparated class that pass it must agree with each other, or nothing
is reported; and a law is not reported where an error large enough to make it wrong would have
been missed. Replayed on 6,000 of the campaign's worlds in which the route had reported a law
(`benchmarks/vouching_dev_replay.json`):

| | The law | Accurate, not the law | Wrong | Nothing |
| :--- | ---: | ---: | ---: | ---: |
| 3,000 worlds drawn at random, the route | 1,825 | 1,057 | 118 | |
| The same after the checks | **1,841** | 787 | **3** | 369 |
| 3,000 worlds changed towards failure, the route | 330 | 960 | 1,710 | |
| The same after the checks | 481 | 669 | **2** | 1,848 |

On the random worlds 68 reports changed from another candidate into the law, and 52 laws were
lost (37 because an error of the size in question could not have been seen, 13 because members of
a class disagreed, 2 of 1,825 refused for structure, which is the test's level). With 3 to 10 %
noise 82 wrong laws became 2, at the price of 24 of 136 laws.

**What this is not.** The checks were written after reading these worlds, so the replay shows
what was built. The search aimed at the route without the checks; nothing has yet searched for
data that defeats the checks. An approximation inside the accuracy line stays reportable. The
first run on fresh laws is the next section.

## Handed counterfeits of new laws, scored on what is refused: all five expectations held

[`PREREG-COUNTERFEIT-2026-11`](../preregistration/PREREG-COUNTERFEIT-2026-11.md), run once on
2026-11-22 ([result](../preregistration/RESULT-COUNTERFEIT-2026-11.md)): 60 laws drawn after the
checks were written. Every case hands a method the law's counterfeits as the candidates of a
search, with the law among them or with only wrong ones, at noise 0, 1 % and 5 %; 360 cases. No
search engine runs. A reported law that is wrong costs nineteen right ones, which is the canonical
permitted share of false reports turned into a price.

| Method | The law | Wrong | Accurate, not the law | Nothing | Score |
| :--- | ---: | ---: | ---: | ---: | ---: |
| The route in force | 146 | 16 | 67 | 131 | −0.88 |
| The same after the checks | **146** | **2** | 43 | 169 | **0.60** |
| The candidate that fits best | 172 | 180 | 8 | 0 | −18.04 |
| The best fit, confirmed on fresh rows | 171 | 173 | 8 | 8 | −17.31 |

As fixed before the run the checks are now the last step of the table route
(`discovery_l2/vouched_route.py`; `APEDP_TABLE_VOUCHING=off` turns them off). Beside the pass:

- **Every wrong law of the route was at 5 % noise**: 16 of 88 reports there (interval 11 to
  28 %), none in 141 at noise up to 1 % (0 to 2.6 %). The earlier runs measured up to 1 %.
- **At 5 % noise the checks passed 2 wrong laws in 55 reports** (3.6 %; interval 0.4 to 12.5 %,
  so a rate above the 5 % limit is not excluded), and their score at that level is still below
  zero (−0.12). They withheld 7 of the 35 right laws there. At noise up to 1 % they lost none
  and turned four approximations into the law.
- **Twelve of the route's sixteen wrong laws were forms of its own first pass**, one a
  polynomial of its own library, three the counterfeits it was handed. So at 5 % noise the run
  tested the checks mostly against the route's own approximations.
- **Both wrong laws the checks passed are the route's own and wrong by a small margin** (test R²
  0.9982 and 0.9989; the line is 0.999). For the second the rule that estimates whether such an
  error could have been seen was too confident (audit M-33).
- **Reporting the best fit was wrong in all 180 cases where no candidate was right;** confirming
  it on fresh rows stopped seven.
- **The forecast filed before the run** put the route's wrong laws at 31 (24 to 37); there were
  16. Its other nine totals fell inside their intervals, and it gave the one doubtful
  expectation 81 % (`benchmarks/counterfeit_forecast_scored.json`).
- **Measured since, behind the four searches on the physics benchmark** (PREREG-VOUCHED-SRBENCH,
  [result](../preregistration/RESULT-VOUCHED-SRBENCH-2026-11.md)): **they withheld 43 of the 193
  laws the route recovered, all at 1 % noise (47 of 90 kept there), and stopped nothing**: none of
  the route's 229 reports was inaccurate. The third expectation (95 % kept) missed; the owner
  decides whether they stay in force. 42 of the 43 were withheld because the power rule cannot see
  an error shaped like a rival that parts from the law on a few rows (audit M-35).
- **Not measured:** the checks on trajectories or series. A second red-team campaign has since
  searched for data that defeats them (next section).

## A second search, aimed at the checks: they hold where the runs measured, and two ways around them were found

**Development by its nature.** `scripts/run_red_team_checks.py`, run once on 2026-11-23 (GitHub
Actions run `45081289544`): 17,077 worlds on 150 laws the checks had never met
(`benchmarks/red_team_checks_campaign.json`, every world in
`benchmarks/red_team_checks_worlds.jsonl.gz`). The decision under attack is the table route's
report after its last checks; the climb pushed towards a wrong law that passes them.

| Worlds drawn at random | Route: reported, wrong | After the checks: reported, wrong | Right laws kept |
| :--- | ---: | ---: | ---: |
| The conditions of the preregistered runs | 1,353, 2 | 1,262, **0** | 1,052 of 1,061 |
| Fewer rows (60 to 300) | 690, 35 | 612, 10 | 439 of 464 |
| Rows that do not cover the box evenly | 929, 11 | 874, 0 | 597 of 600 |
| Noise of another shape | 553, 8 | 496, 0 | 377 of 383 |
| More noise (1 to 10 %) | 1,129, 182 | 736, 7 | 420 of 504 |
| All | 4,654, 238 (5.1 %) | 3,980, **17 (0.4 %)** | 2,885 of 3,012 (96 %) |

On the worlds changed step by step the climb found 1,816 wrong laws the checks passed, all of 8
laws. **Two kinds account for nearly all of them** (audit M-34, M-33):

- **A law with a pole in the domain** (a denominator such as `sin(2.9*x1) + 0.7` that reaches
  zero): 1,468. A counterfeit that puts the pole a little off is wrong by thousands of times the
  law's spread on the few rows near it and right everywhere else. The checks look for an error
  that is smooth across neighbouring rows, and an error on two rows is not; with 60 to 140 rows
  there is often no row near the pole at all.
- **A polynomial of the route's own library inside 1 % of the accuracy line**, on few rows or at
  several per cent of noise: 259, test R² between 0.99 and 0.999. This is M-33 again.

The checks never failed on a world and never let a wrong law through where the climb started
from the measured conditions (1,322 worlds). **What it means:** the checks hold where the
preregistered runs measured and where they were tested blind; they do not protect a law whose
error sits in a corner of the data, and with few rows they are weaker than the record suggests.
Nothing here changes what is in force; both kinds are open.

## Developed and not taken to a preregistered run

| What | What development showed | Why it stopped there |
| :--- | :--- | :--- |
| A language-model agent as the variation operator of a search, after Agentic Variation Operators (`apedp_dyn/proposers/agentic_variation.py`, `discovery_l2/agentic_stage.py`) | **Development run with the free model** (`nemotron-3-ultra`, 2026-11-21, the twenty development laws of `benchmarks/invented_laws.json` at noise 0 and 1 %, 40 runs an arm; `benchmarks/agentic_variation_dev_study.json`): the search with the agent recovered **16 of 40, the default search stage 13**, the agent alone 1. Run for run the agent's arm recovered four laws the default did not and missed one it did, which five runs cannot tell from chance (sign test 0.38). No arm reported a wrong law (0 of 27, 26 and 12). The agent's arm took a median of 504 s against 277 s. All 120 runs completed; 67 of 80 sessions fitted at least one expression | **Not a gain yet, for two reasons.** The default stage was given the same 240 s of search and not the same wall time, so more time is an explanation the run cannot exclude. And the rows did not record which of the structures handed on the agent wrote, so a gained law cannot be traced to it (they do now). What would settle it: the forty held-back laws with a third arm, the default stage given the agent's extra time as search. That comparison is designed, not frozen and not run: on the owner's instruction, testing waits for a stronger model, and its settings depend on that model's speed |
| Additions to the ODE route: sparse regression as a second engine, adequacy judged at the rows' precision, and a follow-up experiment that shortens a reported law (`ode/second_engine.py`, `ode/precision_adequacy.py`, `ode/experiment_pruning.py`) | Development on the first blind set after its run, replayed with its recorded candidates (`benchmarks/ode_additions_dev_replay_2.json`): recovered of 310 without noise and at 1 %, the route in force 199 and 210; with the first two additions 202 and 233; with the experiment as well 229 and 247 (random restarts: 222 and 239); 3 wrong laws of 513 in each. The sparse reference: 255 and 188, with 34 wrong of 620. A first form of the experiment made the route worse and was retired (audit M-28) | **Run blind on 2026-11-22** (PREREG-ODE-ADDITIONS, 46 systems no route had run on; [`RESULT-ODE-ADDITIONS`](../preregistration/RESULT-ODE-ADDITIONS-2026-11.md)): recovered of 276, the route in force 116, with the two additions 118, with the experiment 123 (random restarts 121); wrong among reported 4 of 164, 6 of 188, 6 of 188. All four expectations held, **and what development promised did not carry over**: there the same steps were worth 26 and 41 components of 620, here 2 and 5 of 276. Adequacy at the rows' precision reinstated 23 laws of which 1 is the law and 3 are wrong; it is held back from the route in force (a recorded deviation). Chosen restarts beat random ones by two components, which is inside the variation of the machinery |
| The first pass of the main search on a native kernel (`discovery_l2/fast_first_pass.py`, `rust/apedp-kernel/src/first_pass.rs`) | On the machine it was written on it returns the reference's result on 270 of 270 datasets, field by field, about 12 times faster (489 s against 36 s on 120 datasets; 453 s against 39 s on 150). Its first version agreed on 120 of 120 there and on 299 of 300 on GitHub's processors: it did not bound how far two correct evaluations of an expression such as `sin(exp(exp(x)))` can differ (audit M-29). **Fixed, and checked again on GitHub's machines (run `45051194079`, 2026-11-22): 1,200 of 1,200 datasets identical, eight seeds on six processor models from two makers, the seed that had differed among them; 10.7 to 17.0 times faster there** | A count of agreements, not a proof: its error bounds are an analysis with a safety factor. Opt-in (`enabled()`), used by nothing by default and by no preregistered run; `check=True` runs both |
| A memory of confirmed laws as a candidate source (`proposers/structure_memory.py`, `discovery_l2/memory_stage.py`) | Development, on 80 laws of the invented-law grammar the memory had not seen, 1 % noise, the memory as the route's only search stage: it handed over the law itself in 15 and the route recovered 13, against 1 with no search stage. It also reported 3 wrong laws of 67 (6 of 39 with no search stage): a route whose only rivals are approximations confirms approximations | **Run blind on 2026-11-22** beside the searches, on 40 laws drawn afterwards (PREREG-MEMORY; [`RESULT-MEMORY`](../preregistration/RESULT-MEMORY-2026-11.md)): 23 of 80 recovered against 17, six gained, none lost, 0 wrong of 57 and of 55. Both expectations held and it is in force for tables. The memory's laws were the law itself in 26 runs; at 1 % noise the route reported 6 of the 11 it was handed and in the other five named a shorter candidate that is accurate and is not the law. The proposer's way of joining parts was written knowing the grammar; no gain is measured on the physics benchmarks' laws, most of which it cannot grow |
| The system's forecast of its own results (`discovery_l2/outcome_forecast.py`, `scripts/forecast_runs.py`) | Learned from the runs on record, it gives each problem's chance of being recovered. Backtest, holding one recorded run out at a time (`benchmarks/outcome_forecast_backtest.json`): better than the plain average in 7 of 10 held-out runs; clearly so on the table benchmark, where the same laws recur (Brier 0.12 to 0.16 against 0.21 to 0.25); worse than the average in three of the six runs on dynamical systems; a whole run's recovery rate was off by 10.6 percentage points (root mean square) | A backtest is not a test: every outcome was known when it was written. **Two forecasts filed before their runs have now been scored** (`benchmarks/outcome_forecast_scored.json`): five of six totals fell inside their 80 % intervals, which are wide. On the 46 new dynamical systems it said 90 recovered (51 to 129) and 116 were, where the earlier runs' average said 182; it said 25 wrong laws and there were 4. On the 40 new table laws it said 50 (37 to 62) and **17 were: a clear miss**, on a kind of law it had seen no run on. Run by run it ranks better than chance (0.60 and 0.68, where 0.5 is chance) and no better than that. It reads the true law, so it forecasts benchmark runs, not real data |
| A loop in which a coding model rewrites the search's candidate generator under a fixed judge (`apedp_dyn/evolution/`, `discovery_l2/evolved_stage.py`, `scripts/run_search_evolution.py`; [SEARCH_EVOLUTION.md](SEARCH_EVOLUTION.md)) | With a scripted stand-in for the model, on 16 training and 16 validation laws: a reply with an import is refused before it runs, its repair fails when run, the next repair runs and proposes 1 and 4 of the laws where the seed proposes none, a version that never returns is stopped on every problem, strings such as `9**9**9**9` are refused by the judge. The jailed process, given code the check would refuse, could not write, read, open a socket, start a process or see a variable of the caller | **No real model has driven it**, on the owner's instruction: it waits for a coding model. Nothing it produces is installed; a version would enter the route only by a reviewed commit after one preregistered test on laws it was never scored on. The training laws share one grammar, so a gain on them is not yet a gain on laws |
| The competition's circuit verifier as the judge of that loop (`evolution/circuit_text.py`, `evolution/circuit_task.py`, `scripts/run_circuit_evolution.py`) | Run end to end on 2026-11-22 with scripted replies in place of a model, **on a simulator built here from the pinned source commit and patch whose bytes differ from the pinned build (`c0c27a84…` against `36b1e6ef…`), so the repository's referee refuses it and none of this is a receipt**; on that build the champion's rebuild script reproduced the pinned circuit (`ce04c9d5…`) and its pinned numbers. *The competition's reference circuit* (260 qubits): a renaming of working qubits, written in the restricted language and run in the jail, returned the same gates on 249 qubits; the verifier passed it on 4,096 inputs, on a million, and on every input of the benchmark (200,540,928 shots) at 1,988,910.71 against the reference's 2,076,774.23 (4.2 % less). A missing gate failed at x = 1; a `REGISTER` line never reached the verifier. *The repository's 89-qubit circuit*: the same renaming finds nothing (89 are in use at once); a missing gate fails at x = 1; with its finest rotation left out it passes on 4,096 inputs at 156,963 (cheaper) and fails at x = 685,219 on the million. Ten copies that differ in a pair of flips which cancel cost 157,592 to 157,617 on 4,096 inputs and 157,600.0 to 157,601.6 on a million, which is why a circuit leads only if it is 0.02 % cheaper | **No model has driven it.** The official path (the referee, the pinned simulator, the ledger) has not run here; its test is opt-in and waits for a machine that holds the simulator. On the 89-qubit circuit the simple rewrites find nothing, as `docs/INVSQRT_SUB62_DESIGN.md` predicted: a model would have to write another schedule of the arithmetic, from 27,381 lines without notes. A circuit that leads has passed without a salt; a claim needs the one salted run, outside the loop |
| ODE systems through the refined path with one search (`apedp_dyn/ode/refined_ode.py`) | On 15 of ODEBench's 63 systems (31 components), with one PySR search: 25 and 22 recovered at noise 0 and 1 % against the ODE route's 24 and 21; 3 of 52 reported laws not accurate against 2 of 51. Without any search engine: 24 of 31 at noise 0, all correct, four seconds each | One more component at each level is inside the difference between two runs. With four searches side by side the same systems gave 28 and 23; that version was preregistered and run, and is in the routes table above (recovers more, not adopted) |
| Term pruning to the data's precision, for trajectory rows (`discovery_l2/precision_pruning.py`) | Applied after the fact to the 195 laws PREREG-ODE-REFINED reported: recovered 165 to 167, wrong 10 to 9 (4.6 %, under that run's 5 % limit), none lost | **Since measured blind** (PREREG-ODE-BLIND, 2026-11-21): on 98 unseen systems it changed 42 reported laws, made 24 recoveries, lost none and left the wrong laws at 3. In use in the default ODE route |
| Four seeds of the same PySR search side by side (`apedp_dyn/proposers/portfolio.py`, `pysr_members`) | On SRBench's 30 development problems, both noise levels, on four-core runners: 27 and 25 recovered, exactly the runs one search recovers; none gained, none lost; 0 of 54 reported laws inaccurate; 265 s of wall time for a 240 s budget | A seed does not change what the search finds. A portfolio of different searches (`apedp_dyn/proposers/deep_pass.py`, `mixed_members`) was developed instead (next row; audit M-22) |
| Four different searches side by side: default PySR, a deeper exhaustive pass, algebraic PySR, a second seed (`mixed_members`) | Same 30 problems: 29 and 25 recovered by the benchmark's criterion against 27 and 25; three runs gained, one lost; 0 of 54 reported laws inaccurate. One of the three gains is an approximation that passes the tolerance narrowly (M-23), so by form 28 and 25 | Taken to a preregistered run, which held all four expectations; now in the routes table above |

## Known limits

- **"Recovered" is a numerical criterion with a tolerance** (audit M-23): constants are refitted
  to the clean law on the test rows and an error below 1e-8 passes. Two of the language-model
  controller's 173 recoveries pass only narrowly and are approximations by form; the refined
  path's 174 and the ODE route's 149 have none. Results from now on also give the count without
  narrow passes. Judged again on a range twice as wide
  (`scripts/check_recoveries_out_of_range.py`), every recovery of the refined path (174), of the
  four searches (189) and of the main search (152) holds, and the same two of the controller's
  fail; trajectory data has not been checked this way.
- **The template engine still promotes a sole survivor that does not fit, except where Guard 3
  is on** (audit M-10). The guard's run met the two expectations that decide it, so it is the
  default in `discovery_l2/guarded_l2.py`, which the UI's physics page now uses. Callers whose rows
  lie along one trajectory or in time order (the registry page, the ODE engine, the market sweep)
  and the multi-term engine are still unguarded: the guard has not been measured there. The default route does not use
  that rule.
- **The adequacy check is weak on rows taken from one trajectory** (M-14): the main search returns
  a law that nothing challenged as `unchallenged`. The refined path, now the default, does not
  report such a law, and reports nothing at all when no search stage was given.
- **The main search alone reports laws in pure noise on return-like data** (M-16). Use the series
  route there.
- **The table route's last checks withhold nearly half the laws it recovers at 1 % noise on the
  physics benchmark** (M-35): 47 of 90 kept, and nothing inaccurate stopped. In force until the
  owner decides.
- **Above 1 % noise the table route's reports are not measured to hold, and at 5 % they did
  not** (M-31, M-32): 16 of 88 laws wrong on handed counterfeits. The last checks, in force for
  tables since 2026-11-22, brought that to 2 of 55; the interval for their rate reaches 12.5 %,
  and the rule that decides whether an error could have been seen is too confident for errors of
  shapes it does not try (M-33). A second red-team campaign found that they do not see an error
  concentrated near a pole of the law, and are weaker on 60 to 300 rows (M-34). The trajectory and
  series routes do not have these checks.
- **An exact fit can carry a term that contributes nothing** (M-15) in the main search and the ODE
  route; the refined path drops such terms on tables (nothing inaccurate in 212 reports on
  SRBench) and did not on window rows of trajectories: in PREREG-ODE-REFINED it kept
  `4.4e-9*exp(exp(x))` beside `-x**3`, and lost one law to two terms of size 1e-12.
- **A candidate that cannot be fitted ends the run** (M-24): two of PREREG-ODE-REFINED's 234 runs
  and one of PREREG-PORTFOLIO-SEARCH's 266 stopped on a least-squares fit that did not converge,
  each on a law the earlier route recovers. More searches mean more candidates, so this is more likely with a portfolio.
  The fit guard is in the default routes for tables and for trajectories. In the blind ODE run it
  dropped two candidates in one of 620 runs, and the unguarded route ended in that error on the
  same component; none of the guarded runs did.
- **The recovery criterion refits constants inside functions** (M-26): a factor `cos(c*...)`
  becomes 1 when `c` is refitted to zero, so a law with such a factor can count as recovered while
  it is inaccurate as reported. One case in the blind ODE run, for both routes (test R² −0.80). A proposal that is not finite
  on the data can crash a run when a proposer is used directly (M-17); the composed proposer of the
  refined path contains and records such a failure.
- **Memorisation.** SRBench's laws are textbook formulas a language model may have seen. LLM-SRBench
  was built against that; its dataset is gated and has not been run here.
- **The UI** shows the default route beside the template engine's verdict, runs the PySR stage only
  when asked and only where PySR is installed, and shows a law found without it as tentative. It has
  no physical units for a table, so the refined path runs there without its units reduction. The
  registry endpoint still shows the template engine alone (M-18).
- **Governance.** The signed certification covers an earlier revision of the canonical specification;
  re-signing needs the registered trustees. The CI job that checks this fails until they sign.
- **Lean.** 39 theorems of a small core are machine-checked; the transcription tree is not a
  verification layer.

## How to reproduce a number

Each result directory holds `results.json` (with the digests of the frozen files), `runs.jsonl` and
the per-shard files. The workflow under `.github/workflows/` named in the result file reruns it; the
harness refuses to run if a frozen file has changed.

# What has been measured, and where the system stands

Last updated 2026-11-20. This page is the short answer to "what does this system do, and how do you
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
entry points (`apedp_dyn/ode/main_search_ode.py`, `apedp_dyn/discovery_l2/series_route.py`).

| Route | For | What it does | Measured | Result file |
| :--- | :--- | :--- | :--- | :--- |
| **Refined** (the default since 2026-11-20; `discovery_l2/refined_search.py`) | Tabular data, with physical units where the problem has them | The main search with the units' dimensionless products, a polynomial library, a report in the confirmed form with unnoticed terms dropped, and the search stage on every problem | SRBench: a law reported in 212 of 266 runs, **none inaccurate** (interval 0–1.7 %), **174 the law**. Feynman 78 and 77 of 119; Strogatz 19 of 28. On 103 held-out problems 61 and 61 against the main search's 55 and 52: one short of the 62 expected at noise 0. It uses units, which the other rows were not given. **With four different searches side by side as the search stage** (`discovery_l2/search_stage.py`, PREREG-PORTFOLIO-SEARCH): 189 recovered, 1 of 224 reported laws not accurate; on the 103 held-out problems 73 and 63 against 61 and 61. All four expectations held; it is the search stage wherever four cores are available | [`RESULT-REFINED-SEARCH`](../preregistration/RESULT-REFINED-SEARCH-2026-11.md), [`RESULT-PORTFOLIO-SEARCH`](../preregistration/RESULT-PORTFOLIO-SEARCH-2026-11.md) |
| Main search (the default from 2026-11-19 to 2026-11-20) | Tabular data where a law explains nearly all of the target | Exhaustive first pass, then PySR, then confirmation on rows chosen to tell rivals apart, then an adequacy check | SRBench, 133 laws at noise 0 and 1 %: a law reported in 186 of 266 runs, **4 not accurate (2.2 %**, interval 0.6–5.4 %), 152 the law. Feynman 69 and 70 of 119; PySR alone 73 and 68 | [`RESULT-MAIN-SEARCH`](../preregistration/RESULT-MAIN-SEARCH-2026-11.md) |
| **ODE** | Trajectories of a dynamical system | One state and one rate per disjoint window, then the main search per component | ODEBench, 63 systems, 117 components: **80 and 69 recovered** (noise 0, 1 %); 9 of 184 reported laws wrong (4.9 %). The sparse polynomial reference: 78 and 53, 29.1 % wrong. **Through the refined path with four searches side by side** (`ode/refined_ode.py`, PREREG-ODE-REFINED): 92 and 73 recovered, and on the 86 components it was not built on 63 and 50 against 56 and 48; but 10 of 195 reported laws wrong (5.1 %) against a limit of 5 % fixed in advance, so it is **not adopted** | [`RESULT-ODE-MAIN-SEARCH`](../preregistration/RESULT-ODE-MAIN-SEARCH-2026-11.md), [`RESULT-ODE-REFINED`](../preregistration/RESULT-ODE-REFINED-2026-11.md) |
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

## The repaired ODE route on unseen systems: frozen, running

[`PREREG-ODE-BLIND-2026-11`](../preregistration/PREREG-ODE-BLIND-2026-11.md) runs the refined ODE path
with four searches, the fit guard and pruning to the rows' precision on 98 chaotic polynomial
systems (310 components) from the `dysts` database that no route here was built or run on, beside
the recorded ODE route. It was frozen on 2026-11-21 and its 48 shards started the same day at 00:16
UTC, on branch `ode-blind-run-2`; there is no result yet. (A first launch, on `ode-blind-run-1`, was
cancelled during its warm-up job, before any shard started and so before any system was run.) If at
most 5 % of its reported laws are wrong and it recovers at least as many components as the
recorded route at both noise levels, it becomes the default ODE route.

## Developed and not taken to a preregistered run

| What | What development showed | Why it stopped there |
| :--- | :--- | :--- |
| A language-model agent as the variation operator of a search, after Agentic Variation Operators (`apedp_dyn/proposers/agentic_variation.py`, `discovery_l2/agentic_stage.py`) | With a scripted stand-in for the model: the agent is shown what a fit's residual still follows, repairs its guess, and the repair seeds the search's next round; replies are read only as bounded actions; a missing or useless model costs the search nothing (31 tests). Through the refined path the scripted agent's law is confirmed | No real model has driven it yet. The two earlier language-model runs (78 recovered against 152; 173 against 174) used one model in a fixed pipeline and suggest the model is the limit. A smoke run with the real model is written (`scripts/smoke_agentic_variation.py`); a fair test needs laws a model cannot have read |
| ODE systems through the refined path with one search (`apedp_dyn/ode/refined_ode.py`) | On 15 of ODEBench's 63 systems (31 components), with one PySR search: 25 and 22 recovered at noise 0 and 1 % against the ODE route's 24 and 21; 3 of 52 reported laws not accurate against 2 of 51. Without any search engine: 24 of 31 at noise 0, all correct, four seconds each | One more component at each level is inside the difference between two runs. With four searches side by side the same systems gave 28 and 23; that version was preregistered and run, and is in the routes table above (recovers more, not adopted) |
| Term pruning to the data's precision, for trajectory rows (`discovery_l2/precision_pruning.py`) | Applied after the fact to the 195 laws PREREG-ODE-REFINED reported: recovered 165 to 167, wrong 10 to 9 (4.6 %, under that run's 5 % limit), none lost | The run's outcomes were known when the rule was written, so this cannot count. It needs ODE systems it has not seen |
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
- **An exact fit can carry a term that contributes nothing** (M-15) in the main search and the ODE
  route; the refined path drops such terms on tables (nothing inaccurate in 212 reports on
  SRBench) and did not on window rows of trajectories: in PREREG-ODE-REFINED it kept
  `4.4e-9*exp(exp(x))` beside `-x**3`, and lost one law to two terms of size 1e-12.
- **A candidate that cannot be fitted ends the run** (M-24): two of PREREG-ODE-REFINED's 234 runs
  and one of PREREG-PORTFOLIO-SEARCH's 266 stopped on a least-squares fit that did not converge,
  each on a law the earlier route recovers. More searches mean more candidates, so this is more likely with a portfolio. A proposal that is not finite
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

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

## Developed and not taken to a preregistered run

| What | What development showed | Why it stopped there |
| :--- | :--- | :--- |
| ODE systems through the refined path with one search (`apedp_dyn/ode/refined_ode.py`) | On 15 of ODEBench's 63 systems (31 components), with one PySR search: 25 and 22 recovered at noise 0 and 1 % against the ODE route's 24 and 21; 3 of 52 reported laws not accurate against 2 of 51. Without any search engine: 24 of 31 at noise 0, all correct, four seconds each | One more component at each level is inside the difference between two runs. With four searches side by side the same systems gave 28 and 23; that version was preregistered and run, and is in the routes table above (recovers more, not adopted) |
| Four seeds of the same PySR search side by side (`apedp_dyn/proposers/portfolio.py`, `pysr_members`) | On SRBench's 30 development problems, both noise levels, on four-core runners: 27 and 25 recovered, exactly the runs one search recovers; none gained, none lost; 0 of 54 reported laws inaccurate; 265 s of wall time for a 240 s budget | A seed does not change what the search finds. A portfolio of different searches (`apedp_dyn/proposers/deep_pass.py`, `mixed_members`) was developed instead (next row; audit M-22) |
| Four different searches side by side: default PySR, a deeper exhaustive pass, algebraic PySR, a second seed (`mixed_members`) | Same 30 problems: 29 and 25 recovered by the benchmark's criterion against 27 and 25; three runs gained, one lost; 0 of 54 reported laws inaccurate. One of the three gains is an approximation that passes the tolerance narrowly (M-23), so by form 28 and 25 | Taken to a preregistered run, which held all four expectations; now in the routes table above |

## Known limits

- **"Recovered" is a numerical criterion with a tolerance** (audit M-23): constants are refitted
  to the clean law on the test rows and an error below 1e-8 passes. Two of the language-model
  controller's 173 recoveries pass only narrowly and are approximations by form; the refined
  path's 174 and the ODE route's 149 have none. Results from now on also give the count without
  narrow passes.
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

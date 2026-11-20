# Where this system stands against the field, and what would put it ahead (2026-11-20, second pass 2026-11-21)

A first pass, from public sources read on 2026-11-20. It is a reading of papers' own claims, not a
reproduction of them: none of the outside numbers below was re-measured here, and they are not
directly comparable with this repository's numbers unless that is said.

## What the field looks like

**Equation discovery.** The benchmark that counts is no longer Feynman/SRBench. LLM-SRBench
(ICML 2025, 239 problems in four domains) was built because textbook formulas are memorised by
language models; at publication the best system reached 31.5 % symbolic accuracy, and PySR scored
0 % symbolic accuracy on its chemistry, biology and materials problems
([paper](https://arxiv.org/abs/2504.10415)). Since then the leading methods are language-model
agents around a search: SR-Scientist (ICLR 2026) lets the model write analysis code and reports
margins of 6 to 35 points over baselines ([paper](https://arxiv.org/abs/2510.11661)); InsightSR
reports 95 % exact recovery on Feynman and 80 % accuracy at tolerance 0.1 on the transformed
problems ([paper](https://arxiv.org/pdf/2608.25291)); L2G-SR, A-SR and Deliberate Evolution report
further gains on the same benchmark.

**AI scientists.** Kosmos runs for up to twelve hours over a dataset and the literature; outside
scientists judged 79.4 % of the statements in its reports accurate
([paper](https://arxiv.org/abs/2511.02824)). Google's co-scientist and FutureHouse's Robin have
findings confirmed in wet labs, with human experts choosing candidates and running the
experiments.

**Auditing published work.** Automated reproduction of published results is an active line in the
social sciences: ReplicatorBench ([paper](https://arxiv.org/abs/2602.11354)), REPRO-Bench, and a
study of 76 published studies re-analysed by language models
([paper](https://arxiv.org/abs/2606.13670)).

## What that means for this system

| Dimension | Field | Here | Reading |
| :--- | :--- | :--- | :--- |
| Recovery on textbook laws | 95 % claimed on Feynman | 89 of 119 without noise (75 %) | Behind, on a benchmark the field no longer trusts |
| Recovery where memorising does not help | 31.5 % at publication, higher since | **Not measured** (dataset gated; needs `HF_TOKEN`) | The largest unknown. PySR alone scored 0 % in three of four domains there, and PySR is this system's search |
| Language model in the loop | The source of the field's gains | Measured twice with a small free model: adds nothing | A real gap, and possibly only a weak-model result |
| Reliability of what is reported | Rarely reported; 79.4 % of statements accurate for one leading system | 0 of 212 and 1 of 224 reported laws inaccurate on SRBench; every miss recorded | The one dimension where this system is ahead in kind, though on an easier task |
| Preregistered, run-once evaluation with stated consequences | Not the practice | Every claim | Unusual, and cheap for others to verify |
| Real findings confirmed outside | Several, in biology | None | Behind |

Three conclusions.

1. **It will not win on raw recovery by tuning PySR.** The gains elsewhere come from a capable
   language model that reads the problem's scientific context and writes code. The two tests here
   used a small free model. The honest next measurement is the same controller with a frontier
   model, on LLM-SRBench, preregistered.
2. **Its edge is the referee, and nobody else is competing there.** The field measures how often
   a system finds the law. Almost nobody measures how often a system says something false, and
   fixes in advance what a miss will cost. A system that publishes that number, lets outsiders
   attack it, and keeps a public record of dated predictions is in a category of its own.
3. **Auditing is occupied in the social sciences and open for quantitative laws.** Others
   re-run regressions. Nobody is taking published empirical laws with public data and asking
   whether the law survives data it was not fitted on, with a calibrated referee.

## Order of work

| # | Step | Needs | What it would show |
| :--- | :--- | :--- | :--- |
| 1 | LLM-SRBench, preregistered: the default route alone, then with a frontier model steering it | `HF_TOKEN`; a paid model key | Where the system really stands. It may be low |
| 2 | Public prediction ledger with first real claims (`epistemic/prediction_ledger.py`, added today) | A data stream that keeps arriving | A track record that cannot be tuned afterwards |
| 3 | Audit of published empirical laws with public data | A choice of field and 20 to 50 laws | Real findings without needing luck |
| 4 | Open challenge: break the referee | A public page and a harness | An error rate established by outsiders |
| 5 | Repairs the runs asked for: fit guard (added today), pruning on trajectory rows, the guard on time-ordered data | New ODE systems for a blind re-test | The ODE route past its reliability gate |
| 6 | Closed-loop experiments against a simulator | A simulator worth asking | Whether choosing the next measurement beats collecting more |

## Second pass (2026-11-21): ODE discovery, reliability, and what the audit already has company in

Read the same way as the first pass: papers' own claims, nothing re-measured.

**ODE discovery has moved to choosing experiments.** ALDER proposes equations, picks the next
experiment where its candidate models disagree, and admits a law only after a held-out check
([paper](https://arxiv.org/abs/2609.33728)). It reports 95.0 % symbolic accuracy on ODEBench's 63
systems and 90.7 % on 59 systems of ODEBase, and passes its own stricter test on 60 of 63 and 49 of
59. In the same table, methods that only read given trajectories sit at 16 to 19 % on ODEBench
(SINDy 18.5, PySR 18.1, ODEFormer 16.5) and other active methods at 41 to 46 %. The baseline
entries are copied from another paper and ALDER's own accuracy comes from a different pipeline, so
even that table is not a matched comparison.

Here, counting a system only when every component is recovered, without noise: the recorded ODE
route 35 of 63 (56 %), the refined path with four searches 43 of 63 (68 %, not adopted), the
sparse polynomial reference 34 of 63 (54 %). These cannot be set beside ALDER's table: the same
kind of baseline scores 54 % here and 18.5 % there, so the protocols differ by more than the
methods do. What can be said without a matched run: **this system does not choose its
experiments, and the systems now leading ODE discovery do.**

**The polynomial chaotic systems of `dysts`** are the standard test of sparse identification:
Kaptanoglu et al. compare four SINDy solvers on 70 of them and find the weak form better even on
clean data ([paper](https://arxiv.org/abs/2302.10787)). The blind test frozen today
(PREREG-ODE-BLIND-2026-11) takes its 98 systems from the same pool.

**Reliability is no longer uncontested.** KO-PDE-IDENT selects the terms of a PDE with knockoffs
and a controlled false-discovery rate ([paper](https://arxiv.org/abs/2605.26631)). Others score
the structural stability of a formula ([paper](https://arxiv.org/abs/2509.21780)) or name the
"pseudo-equation trap" of formulas that fit and are not the law
([paper](https://arxiv.org/abs/2602.13021)). An independent look at Kosmos in radiation biology
found one supported discovery, one uncertain and one false among three
([paper](https://arxiv.org/abs/2511.13825)). The first pass said nobody else competes on
reliability. That was too strong: a few do, by statistical guarantee on term selection. None found
reports a wrong-rate measured in preregistered runs.

**The finance audit has company.** Eleven stylised facts have been put to statistical tests on 150
years of market data ([paper](https://arxiv.org/abs/2504.08611)), and panel studies of crypto
report that positive returns raise volatility, the opposite of the leverage effect
([paper](https://arxiv.org/abs/2404.04962)). What PREREG-LAW-AUDIT-FINANCE adds is narrower than
"first": intervals widened by a factor measured on simulated dependence, a power check published
before the run, and a panel that keeps the pairs that stopped trading.

### The gaps, ranked after both passes

| # | Gap | Size | Who can close it | State on 2026-11-21 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | No number on a benchmark where memorising does not help (LLM-SRBench) | The largest unknown | Needs the owner's Hugging Face access and `HF_TOKEN` | Blocked |
| 2 | ODE discovery without chosen experiments | Leading systems report 90 % and more by choosing them. On this system's own record the headroom is smaller: of the 69 runs the refined ODE run missed, in 10 the true law was proposed and could not be told from its rivals on the rows at hand; in 37 nothing was confirmed and in 17 the law was never proposed | Buildable here: start further trajectories where the surviving rivals disagree, and measure against equal data chosen at random | Not built |
| 3 | A capable language model in the loop | The source of the field's gains on tables | Needs a paid model key | Blocked |
| 4 | The ODE route's refined path missed its reliability limit by one law | 92 and 73 components recovered against 80 and 69, unused | A blind run | **Frozen and started today** (PREREG-ODE-BLIND) |
| 5 | No finding on real data | The lowest-rated part of the system | The owner's word to start the frozen audit; then the dated claims | Frozen, waiting |
| 6 | No statistical guarantee on which terms are kept | Others now offer a controlled false-discovery rate | Buildable here: knockoff selection in the polynomial stage, measured against the present pruning | Not built |
| 7 | The recovery criterion can pass a form that is not the law (M-23) | Two of 173 in one run | Done today: every recorded recovery judged on a range twice as wide | **Closed for tables**; open for trajectories |
| 8 | Outsiders have never tried to break the referee | Its error rate is self-measured | Needs a public page | Not started |

Gaps 2 and 6 are the two that this pass adds and that need nothing from the owner. The count
under gap 2 comes from the recorded rows of PREREG-ODE-REFINED-2026-11: 69 missed runs of 234 are
37 where the search confirmed nothing, 27 where the path reported one of several laws it could not
separate (the true law among the recorded candidates in 10 of them), 3 confirmed and wrong, and 2
that ended in an error. Most of what the ODE route loses is search, not the choice of experiment.

## What this pass did not cover

Materials and chemistry discovery systems, theorem-proving systems, and commercial platforms
were not surveyed. Outside numbers were taken from abstracts and result tables, not checked.

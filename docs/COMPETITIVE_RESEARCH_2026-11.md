# Where this system stands against the field, and what would put it ahead (2026-11-20)

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

## What this pass did not cover

Materials and chemistry discovery systems, theorem-proving systems, and commercial platforms
were not surveyed. Outside numbers were taken from abstracts and result tables, not checked.

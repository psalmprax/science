# RESULT-REAL-TABLES-2026-11: the small-table reading on real measurements

Preregistration: [`PREREG-REAL-TABLES-2026-11.md`](PREREG-REAL-TABLES-2026-11.md), frozen by
[`DIGEST-REAL-TABLES.json`](DIGEST-REAL-TABLES.json) (23 files) in commit `346c49e`, after two
independent reviews and a rehearsal ([`REVIEW-REAL-TABLES-2026-11.md`](REVIEW-REAL-TABLES-2026-11.md)).
Run once on GitHub Actions, workflow `Real tables, blind`, run 45486849433, on branch
`real-tables-blind-run-1` at `346c49e`, 2026-11-24 21:03 to 23:08 UTC by GitHub's clock. Eight jobs:
the real tables (a job of 122 minutes; 118.6 by the run's own clock, from 21:06:37), three shards of
generated tables, three of traps, and the collection. None was lost
or run again; no branch of the deviation rule was used. Outputs:
[`RESULT-REAL-TABLES-2026-11/`](RESULT-REAL-TABLES-2026-11/) (`real.json`, `results.json.gz`,
`SHA256SUMS` of the two records as the collect step printed them, and `real_job_log_times.txt`, the
times in the real job's log that this record refers to).

**What was tested:** the reading for small tables (`apedp_dyn/discovery_l2/small_table.py`) with
the search stage in force beside it and what is known beforehand stated. It is not in force.

## The verdicts

| | Expectation | Verdict | Numbers |
|---|---|---|---|
| A1 | It finds known laws in real tables, from their rows | **Missed, by one judgement** | 2 tables count; 3 were needed. See "The judgement that decided A1" |
| A2 | It reports no wrong law on a real table | **Decides nothing** | It reported no law at all on a real table |
| A3 | It finds more than the route in force | **Held, on weak ground** | Known form on 3 tables against 2; the third is one where the decoy gave the same form, and three of the route's runs gave no answer |
| D1 | At most 5 % of its reports on 360 generated tables are wrong | **Held, with no margin** | 2 wrong of 44 (4.5 %); a third would have missed |
| D2 | It finds more than the first pass alone | **Held** | True form gained on 34 tables, lost on 1 |
| E1 | At most 5 % of its reports on 360 traps are wrong | **Held, with no margin** | 2 wrong of 41 (4.9 %); a third would have missed |

**What follows: the reading is not proposed to the owner as the product's answer for small
tables.** The plan fixed four cases and none names this exact outcome (A1 missed, A2 deciding
nothing, D1 and E1 held). Every case with A1 missed ends in "not proposed", so that part is fixed.
The description is put together here from two of the plan's cases, in its words:

- from "if A1 misses and A2, D1 and E1 hold": "its reports held up but it does not find known laws
  from the rows of real tables";
- from the case where A2 decides nothing: "no law was reported on a real table, so its reports on
  real measurements are unmeasured".

"Its reports held up" is on generated tables and traps only, and at the limit.

Nothing is put in force by this run. The eight held-back tables are used up.

## Arm A: six held-back tables with a known law

The arm under test is the reading with the search stage and what is known stated. "Known form"
means the law's own form with as many fitted constants; a table counts for A1 if the known form is
given on it and not on its decoy (the same inputs with a target made from another law).

| Table | Rows | The reading gave | Known form | On its decoy | Counts |
|---|---|---|---|---|---|
| Exoplanets (Kepler's third law with two masses) | 12 | `0.3618*d**(3/2)/sqrt(m1 + m2)` | yes | another law | **yes** |
| Binary stars (the same law) | 4 | refused: a reading needs six rows | | | not among the five |
| Langmuir, Sun et al. | 17 | `1.410*p/(p + 0.221)` | yes | the known form again | no |
| Time dilation | 7 | `-0.0055*v**2 - 0.118` | no (see below) | the decoy's law with a constant added | no |
| Hubble 1929 | 32 | `475.3*D` | yes | another law | **yes** |
| Tully and Fisher | 18 | `-8.40*DV**0.142` | no | the decoy's own law | no |

Known form on 3 of the 5 tables with at least six rows; 2 count; 3 were needed. **A1 is missed by
one table.** Langmuir's form was given on its decoy too, so on that table the answer was not shown
to depend on the rows. On Tully and Fisher it gave a power law where the known law is a logarithm.
Both tables that counted are among the three the plan called "close to given to this reading", and
one of them, the exoplanets, carries a law the reading was developed on (Kepler's, on other
measurements). The two tables the author called open, time dilation and Tully and Fisher, both
missed. As the plan said, this is a first
measurement on five tables, and one table more or less changes it.

### The judgement that decided A1

On time dilation the reading gave `-0.0055*v**2 - 0.118`: the low-speed form with a constant
added. Its decoy gave the decoy's own law with a constant added, which is not the known form. So
had this law been judged to have the known form, the table would have counted, A1 would have held
with 3 of 5, and by the plan the reading would have been proposed (with the note that it reported
nothing on a real table).

It was judged not to have the known form, for two reasons that must both be said:

- Against the declared low-speed form `a*v**2` it fits two constants where that form has one. This
  is the plan's own rule: an offset must not pass.
- Against the registered law, `a*(sqrt(1 - v**2/b**2) - 1)`, the count is equal, two and two. There
  it failed the other condition: refitted, it came within 1.2e-5 of the law as the judge had fitted
  it, where 1e-8 is required. That distance comes from the judge's own fit, which put the law's
  second constant at about 2,283 m/s. The plan had said the table cannot determine that constant.
  Had the fit put it at any physical value, the law as fitted would be a pure `a*v**2`, the
  reading's law would have refitted to it exactly (the rule allows an offset in the refit), and
  with equal counts it would have passed.

So the verdict rests on where a fit landed for a constant the data do not fix. It is the verdict
the plan intends: a law with an offset is not the known law, and what was stated beforehand
includes that a clock at rest is not slowed. But under the rule as written a different fit would
have given "proposed". This follows from the stored fields; the judge was not run again. The
result should be read as: on the one open table that could have gone either way, the reading gave
a law with a spurious constant.

Every law above was tentative. None was reported: on all five tables the error on left-out rows was
above the accuracy line, and time dilation also has fewer than five rows for each constant and a
choice that changes when a row is removed. With no report on a real table in any of the three
readings, **A2 decides nothing**.

The search stage changed no choice: the reading without it gave the same law on all five tables.
The reading with nothing stated gave the known form on none of them (3 against 0), so on these
tables it was what is known beforehand that led to the known forms.

### A3 and the route in force

| Arm of the route in force | Tables run | Known form | No answer in 25 minutes |
|---|---|---|---|
| Called directly, with the search stage | 5 | 0 | **3** |
| Called directly, first pass alone | 5 | 2 (Hubble, Tully and Fisher) | 0 |
| The product, with the search stage | 1 | 1 (Hubble) | 0 |
| The product, first pass alone | 1 | 1 (Hubble) | 0 |

The product refuses a table under 24 rows, so it ran on Hubble's table only. The reading's 3 is
more than the best arm's 2: **A3 held as defined.** What it rests on:

- The reading's third table is Langmuir, where what was stated set aside 25 of its 26 scored
  forms and the same form came out on the decoy. Counted as A1 counts, it is 2 against 2.
- With nothing stated the reading had the known form on none of the five tables. The route's first
  pass alone, also with nothing stated, had it on two.
- The plan asked for the unanswered runs to be set beside A3, because it can hold for that reason
  alone. The route called directly with the search stage gave no answer on Langmuir, Hubble and
  Tully and Fisher. To overturn A3 it would have needed the known form on all three, and on
  Langmuir its first pass gave a cosine.

A3 is not evidence that the reading is better than the route.

The route in force here means the route with the second version of the last checks as its last
step. Those checks were put in force on 23 November without the owner's word (see `CHANGELOG.md`).
They withheld two of the route's laws on these tables: the product's `469.0*D` on Hubble, which is
the known form, and the directly called route's law on time dilation, which is not.

The raw outputs say only "no answer within 1500 s". The job's log (times in
`real_job_log_times.txt`) shows the search processes of each of the three being told to stop, at
21:47:55, 22:15:12 and 22:42:23, and then nothing until the 25-minute limit. That is consistent
with the fault described in `benchmarks/recursion/dev_search_run1_stalled.json`, a search process
that never leaves, whose cause is itself given there as most likely and not proven. All three were
in the route called directly, 3 of its 5 runs; none of the reading's 12 searches and neither of the
product's 2 stopped. That pattern is not explained. By the plan a reading that gave no answer in
its time is an answer, and none was run again. The other 16 searches of the run answered.

No arm of the route reported a law on a real table.

## Arm B: pipe friction, no criterion

406 measurements on 11 pipes; the two pipes with relative roughness 0.00101 and 0.00403 (89
measurements) were left out whole. Errors are root-mean-square differences in log10 of the friction
factor; in brackets, about what that is as a share of the factor.

| | On the two pipes left out | On the work rows |
|---|---|---|
| The reading's law (tentative, not reported) | 0.0173 (4.1 %) | 0.0149 (3.5 %) |
| Haaland's form, constants refitted on the work rows | 0.0246 (5.8 %) | 0.0388 (9.3 %) |
| Haaland, published constants | 0.0397 (9.6 %) | 0.0503 (12.3 %) |
| Colebrook and White, published constants | 0.0418 (10.1 %) | 0.0508 (12.4 %) |
| The route in force | no law | |

The table is Nikuradse's rough pipes with the Princeton smooth pipe. The reading's law, which came
from the search stage, had the smallest error on the pipes left out. What limits that:

- The law has six fitted constants against Haaland's three.
- One of the two pipes left out (0.00403) has a near twin among the work rows (0.00404), so one
  pipe of the two is close to interpolation. How many of the 89 rows that is, the outputs do not
  say.
- Colebrook's formula was made for commercial pipes and is known not to follow Nikuradse's
  sand-roughened pipes in the transition region, so a fit of this table is expected to beat it on
  this table. The published formulas score worse on the work rows than on the pipes left out.
- The outputs hold no range of Reynolds numbers, so rows of laminar flow, which no turbulent
  formula describes, cannot be excluded.
- In the rehearsal, on pipes made from Colebrook's formula with 1 % scatter, the reading's law
  lost to that formula on the pipes left out (0.0073 against 0.0045).

The law is not a candidate for a friction formula, and the reading itself did not report it
(left-out R² 0.9945 against a line of 0.999).

The author expected the published formulas to win here. They did not.

## Arm C: the expansion rate, no criterion

32 measurements, every third left out (11). Chi-squared per left-out row: the standard model's
form fitted on the work rows 0.67; the reading's law 0.84; a straight line fitted on the work rows
0.84, the same, because the reading's law is that line: `59.0*z + 64.6`, tentative and not
reported. Both describe the left-out rows within their stated errors; the standard model's form
does it better.

## Arm D: 360 generated tables, drawn after freezing, no search stage

| | Tables | Reported | Wrong | The true form (reported or not) | Reported and the true form |
|---|---|---|---|---|---|
| The reading | 360 | 44 | 2 | 37 | 21 |
| The reading without the memory | 360 | 37 | 2 | 18 | 10 |
| The route in force, first pass alone | 360 | 0 | 0 | 4 | 0 |

**D1 held with no margin:** 2 wrong of 44 is 4.5 %; the limit of 5 % allows 2.2, so a third wrong
report would have missed. The 95 % upper bound of the share is 13.6 %. **D2 held:** the reading has
the true form on 34 tables where the first pass does not, and the other way round on 1.

"Wrong" means an R² below 0.999 on 400 clean rows drawn over the generator's whole range, 1 to 5 in
every input. A table of 8 to 30 rows does not reach that far, so this asks a report to hold beyond
the rows it was read from. Of the 44 reports 21 are the true form, 21 pass that test with another
form, and 2 fail it.

The verdict could have gone either way on single reports. Three reports pass by less than the one
that failed narrowly: tables 260, 193 and 337 have R² of 0.999042, 0.999050 and 0.999054. And as the
plan disclosed, a reading depends on machine load: of 120 development tables read twice, 4 came out
differently. The reading without the memory had 2 wrong of 37, which is 5.4 % and would not have
passed.

These laws come from the development grammar at the development's sizes and noise levels, where
the rule of five rows was fitted. Arm D is inside the development distribution.

| By rows | Reported | Wrong | | By noise | Reported | Wrong |
|---|---|---|---|---|---|---|
| 8 | 3 | 1 | | none | 34 | 2 |
| 12 | 1 | 0 | | 1 % | 10 | 0 |
| 20 | 16 | 1 | | 5 % | 0 | 0 |
| 30 | 24 | 0 | | | | |

At 5 % noise it reported nothing. The memory of confirmed laws accounts for 20 of the 37 true
forms; without it the reading has 18. These laws come from the grammar the memory was built on.
The breakdowns here and below are of the reading; the other arms' are in `results.json.gz`.

## Arm E: 360 traps

| Kind of trap | Tables | Reported | Wrong | Reported and the true form |
|---|---|---|---|---|
| Laws outside the reading's grammar | 90 | 13 | 1 | 0 |
| Harder laws of the grammar | 90 | 1 | 0 | 1 |
| One gross error in the table | 90 | 9 | 1 | 3 |
| Noise in proportion to the value | 90 | 18 | 0 | 6 |
| All | 360 | 41 | 2 | 10 |

**E1 held with no margin:** 2 wrong of 41 is 4.9 %; the limit allows 2.05. The 95 % upper bound
is 14.6 %. One report passes narrowly (table 200, R² 0.999091). The route in force reported
nothing on the traps.

Only the 90 laws outside the grammar come from outside the development grammar, and they were
written by the reading's author; no red-team search was aimed at the frozen reading. Those 90 gave
13 reports, 1 wrong: fewer than the 20 the plan requires before a share decides anything, with an
upper bound of 31.6 %. The harder laws gave one report. So E1 is carried by the two kinds of trap
that change the noise or one row, not the laws.

On the laws it cannot express, none of its 13 reports is the law: 12 pass the accuracy test with
another form and one fails it. In all, 31 of the 41 trap reports are not the law (29 accurate with
another form, 2 wrong).

## The four wrong reports

| Arm | Table | Rows, noise | What it reported | R² on clean rows, 1 to 5 | R² on those inside the span of its own rows |
|---|---|---|---|---|---|
| D | 109 | 8, none | `0.680*x0**2*x1**(1/3)` | 0.998931 | 0.999156 |
| D | 115 | 20, none | `0.300*sqrt(x0 + 0.302) - 0.548/x1**6` | 0.746919 | 0.999997 |
| E, outside the grammar | 25 | 20, none | `4.586 - 0.0355/log(x0) - 2.885*exp(-x0)*sin(x0)` | below zero | 0.999881 |
| E, one gross error | 242 | 30, 0.2 % | `-6.2e-5 + 0.644/(x0 - exp(x0))**2` | 0.997485 | 0.999638 |

**All four are failures beyond the table's own rows.** Inside the span of the rows it was read
from, each of the four is accurate; each fails on clean rows outside that span. (The last column
was computed after the run, from the frozen generator and the recorded laws; it is not part of the
run's outputs.) Table 115's rows start at 1.84 in `x1` and its law has `1/x1**6`; table 25's rows
start at 1.24 and its law has `1/log(x0)`, which has a pole at 1, and one clean row at 1.0005
produces the whole failure. So what the reading lacks is a check on how a law behaves just outside
its rows, which is where the red team found the last checks weak too. In the other direction, a
report that passed can be inaccurate inside its own span: the review found table 193 at 0.99807
there.

## Also promised by the plan, without a criterion

- **By number of inputs.** Generated tables: one input, 15 reports, none wrong; two inputs, 29
  reports, 2 wrong. Traps: one input, 18 reports, 2 wrong; two inputs, 22, none; three inputs
  (the harder laws), 1, none.
- **Traps by rows and noise.** 12 rows: 6 reports, none wrong; 20 rows: 15, 1 wrong; 30 rows: 20,
  1 wrong. No noise: 14 reports, 1 wrong; 0.2 %: 19, 1 wrong; 1 %: 8, none.
- **Laws reported without the known form, beside A2:** none; nothing was reported on a real table.
- **Time.** On a table of arm A the reading took 7 to 23 seconds without the search stage and 129
  to 181 seconds with it; on the pipe table 229 seconds and on the expansion rates 133. On a generated table or a trap, without the search stage, the median was
  37 and 39 seconds and the longest 270 and 379. Of the real job's 122 minutes, 75 were the three
  runs that gave no answer.

## The author's expectations, scored

| Stated before the run | What happened |
|---|---|
| A1 probably misses (one chance in five that it holds) | Missed, by one table |
| The exoplanets are likely to count | They counted |
| Langmuir's second table gives the known form on its decoy | It did |
| Hubble's `a*D` is the reading's default and its decoy will show it | Wrong: on the decoy it gave another law, and Hubble counted |
| A2 holds or decides nothing | Decided nothing |
| A3 holds (two chances in three) | Held, with the caveat above |
| D1 holds (two chances in three) | Held, at the limit |
| D2 holds | Held |
| E1 less than even | Wrong: it held, at the limit |
| In B the published formulas win | Wrong |
| In C the reading and the standard model's form are about equal | The standard model's form was somewhat better |

## What this result does and does not say

- The reading does not find known laws from the rows of real tables: two of five, on a first
  measurement of five tables, with the third hanging on one judgement. Where it reached the known
  form, what was stated beforehand did the work: with nothing stated it reached none.
- It never reported on a real table, so nothing is known about how often its reports on real
  measurements are wrong.
- On generated tables and on traps its reports were wrong 2 times in 44 and 2 times in 41. Both
  pass the 5 % line as fixed, neither rules out a true share of 10 %, and either would have gone
  the other way on one report. The generated tables are from the development grammar.
- D and E ran without the search stage, as the plan fixed. They say nothing about the reading with
  the searches beside it.
- A report was scored over the generator's whole range, beyond the table's rows, and all four
  failures are there. Half of the reports on generated tables, and three quarters on the traps, are
  not the law's form.
- The route called directly with its search stage gave no answer on three of five tables, in a way
  consistent with a known fault of the search stage's processes.
- The 5 % line is `REFEREE_MAX_FPR`, borrowed as in four earlier plans: no named threshold means
  "the share of wrong laws among those reported". As the plan said, naming one is the owner's call,
  and is asked for with this result.

## Review

An independent review of this record against the raw outputs and the plan was made before it was
recorded: [`REVIEW-REAL-TABLES-2026-11.md`](REVIEW-REAL-TABLES-2026-11.md), section "The result".
It confirmed every number and all six verdicts and found ten things wrong with what the first
draft said, stressed or left out. This record is the draft as changed for each of them.

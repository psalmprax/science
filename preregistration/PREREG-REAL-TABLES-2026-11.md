# PREREG-REAL-TABLES-2026-11: does the small-table reading find known laws in real measurements it was not developed on, and can what it reports be relied on?

**Status:** frozen before the run by `preregistration/DIGEST-REAL-TABLES.json`. Runs once, on GitHub
Actions (`.github/workflows/real-tables-blind.yml`), started by the first branch named
`real-tables-blind-run-*` whose jobs start. **Nothing here is in force and nothing goes into force
by this result** (AGENTS.md 1.5). Two independent reviews before freezing, and what was done about
each of their findings, are in `preregistration/REVIEW-REAL-TABLES-2026-11.md`.

## What is tested

Everything the table route had been measured on until 2026-11-24 was generated: textbook laws
evaluated on drawn rows, invented laws, simulated systems. On real tables the route in force says
almost nothing, for one plain reason: real tables that carry a law are small (Kepler's has 8 rows,
Langmuir's 10) and the product refuses fewer than 24 rows.

`apedp_dyn/discovery_l2/small_table.py` is a reading for tables of tens of rows. It differs from the
route in force in four ways:

- **Candidates are forms.** Each source's candidates (plain shapes, power laws with simple
  fractions as exponents, low polynomials, the first pass, the memory of confirmed laws, and the
  search stage if one is given) are stripped of their constants; the constants are fitted again
  on every set of rows used.
- **Leave-one-out, not a split.** Each form is scored by its error on each row when that row is
  left out of the fit (ten folds above 40 rows). No row is spent on a pool.
- **The plainest form within one standard error of the best is chosen**, by description length
  (`apedp_dyn/recursion/mdl.py`).
- **A law is reported only if** the choice is the same with each row (or fold) removed, its R² on
  left-out rows reaches `DISCOVERY_ACCURACY_R2` (0.999), and the table has at least five rows for
  each constant the law fits. Otherwise the law is tentative and the record says which failed.

`apedp_dyn/discovery_l2/known_beforehand.py` lets a reading use what is known before the rows are
read: units, and statements such as "increasing in p", "zero at zero", "levels off". Forms that
contradict a statement are set aside. For a law of one input that is a ratio of polynomials the
statement is decided by a solver (`apedp_dyn/recursion/smt.py`); otherwise on a grid.

## Development, done before freezing, and what it showed

**Real tables.** `benchmarks/real_data/registry.json` lists eleven tables of real measurements,
pinned by source commit and SHA-256. The split was fixed by table name before any row of a
held-back table was read: three for development, eight held back. The three, read by the code
as frozen without a search stage (`benchmarks/real_data/dev_arm_a_as_frozen.json`):

| Table | Rows | The known law | The reading, with what is known stated | The same reading on the decoy |
| :--- | ---: | :--- | :--- | :--- |
| Kepler, solar system | 8 | `a*d**1.5` | reports `0.364*d**(3/2)` | reports `0.069*d**2`, the decoy's own law |
| Langmuir, table IX | 10 | `a*p/(1 + b*p)` | `107.9*p/(p + 8.19)`, the known form, tentative | **the known form again** |
| Leavitt | 26 | `a*logP + b` | a curve, not the line, tentative | neither form |

By this plan's own counting (A1 below) **one development table of three counts**: on Langmuir's
ten rows the reading's answer comes from what was stated, not from the rows. With the search
stage on, at a commit before changes 3 and 4 below, the three readings were the same laws
(`benchmarks/real_data/dev_small_reading_with_searches.json`). The route in force with its search
stage gave no result on these tables (see the disclosures).

**Generated tables.** 120 tables of 8 to 30 rows from invented laws with one and two inputs, noise
0, 1 % and 5 %, no search stage in either arm (`benchmarks/small_tables/dev_comparison.json`, the
second run, with the code as frozen):

| On 120 tables | The reading | The first pass of the route in force, called directly |
| :--- | ---: | ---: |
| the true form | 17 | 2 |
| an accurate law | 48 | 18 |
| laws reported | 17 | 0 |
| of those, wrong | 0 | 0 |
| of those, the true form | 10 | 0 |

The reading had the true form on 15 tables where the first pass had not, and never the other way
round. Seven of its 17 true forms were chosen from the memory of confirmed laws. At 5 % noise it
reported nothing on 36 tables, and at 8 rows nothing on 30.

**Four changes were made while developing, so the numbers above describe what is frozen:**

1. Equal constants in a candidate were first merged into one parameter, so a line was fitted with
   one constant. Each constant is now its own parameter.
2. On exact data, differences at the level of rounding excluded the plainer form. Errors below
   1e-10 of the target's size now count as equal.
3. **Five rows for each constant, to report.** On the first run of the 120 tables the reading
   reported 31 laws and 3 were wrong (10 %), all three with four rows for each fitted constant
   on tables of 8 and 12 rows. The rule was chosen on those tables; the table above is the
   second run, with it.
4. **The target is divided by its size before fitting.** Found the day of freezing: with the
   target multiplied by 1e12 the Kepler table gave another form, because the fits start from
   constants near one. A reading no longer depends on the units of its target.

## Data and protocol

`scripts/run_real_tables_blind.py` is the run. It refuses to start if a frozen file has changed.
Before any method reads anything, every held-back table is downloaded from its pinned commit (three
attempts), checked against its SHA-256 and parsed; if one fails, the run stops there and nothing
has been read. The same check stops the run if a table has lost more than two rows in parsing
against its file's line count, or if the known law or the decoy cannot be fitted to it. The files
stay in a temporary folder and are not kept. After that, an error in one reading, arm or table is
recorded in its place and the run goes on. All real readings use seed 0 and up to 600 seconds;
readings of generated tables use the table's number as seed and up to 300.
The search stage is the one in force for tables, for 90 seconds; the run stops before reading if
fewer than four cores are free, since the stage then runs one search, not four, and the record
carries the stage's own description.

**Hard limits.** Every part that runs the search stage runs in a process of its own and is
stopped after 25 minutes, recorded as "no answer". The reading's parts (the three readings of each
table, the decoy, arms C and B) are run on every table first, each with its full 25 minutes. The
route in force's arms come after; one is started only if its full 25 minutes fit before the run's
340th minute, and one not started is recorded as such.

**A. Six held-back tables with a known law.**

| Table | Lines in the file | Inputs | The known law | Its decoy |
| :--- | ---: | :--- | :--- | :--- |
| `kepler_exoplanets` | 13 | m1, m2, d | `a*sqrt(d**3/(m1 + m2))` | `a*d**2/(m1 + m2)` |
| `kepler_binary_stars` | 5 | m1, m2, d | `a*sqrt(d**3/(m1 + m2))` | (too few rows to read) |
| `langmuir_sun_et_al` | 18 | p | `a*p/(1 + b*p)` | `a*(1 - exp(-b*p))` |
| `time_dilation` | 8 | v | `a*(sqrt(1 - v**2/b**2) - 1)` | `a*v**3` |
| `hubble_1929` | 33 | D | `a*D` | `a*D**2` |
| `tully_fisher` | 19 | DV | `a*log(DV) + b` | `a*DV + b` |

Each table is read three ways by the reading: with what is known stated and the search stage on
(**the arm under test**), with what is known and no search stage, and with nothing stated (no
units, no statements, the noise model left to the reading). Beside them, the route in force in four
arms: called directly on a 70/30 split and through the product, each with its search stage and
with its first pass alone. What is known for each table is in
`benchmarks/real_data/known_beforehand.json`, written from general physics and the sources' README
files before any held-back row was read.

*The known form.* A law has the known form if both hold: with its constants refitted and a scale
and an offset allowed, it reproduces the known law fitted to the table to 1e-8 of its size
(`run_srbench_baseline.symbolic_recovery`; exponents are not refitted); **and it fits exactly as
many constants as the form it matches**. Without the second condition `a*D + b` would pass for
`a*D`, since its offset refits to nothing, and `a*log(DV)` for `a*log(DV) + b`. For the reading's
laws the count is the reading's own record of what it fitted; for the route's laws it is every
number written outside an exponent. Two forms are declared beside the known law, because physics
says the table cannot tell them apart: `a*sqrt(d**3/m1)` for the exoplanets (a planet has about a
thousandth of its star's mass) and `a*v**2` for time dilation (at laboratory speeds the two differ
by about (v/c)² of the effect). A declared form counts only if that is true of the table: fitted to
it, the two forms differ on its rows by no more than the measurements scatter about the known law.

*A wrong report on a real table* is a reported law that does not have the known form.

*The decoy.* A law is always given, and three of these tables are close to given to this reading:
its power-law candidates include sums of inputs with the same unit, built knowing Kepler's law with
two masses; the four statements about adsorption set aside 37 of 38 forms on the development
table; and `a*D` is the plainest form there is. So each table has a decoy: the arm under test reads
the same inputs with the values of another law that satisfies everything stated (fitted to the
table), with the table's own scatter about the known law: each row's ratio to the known law where
the reading of the real table counted errors in proportion, each row's difference otherwise. If
the reading gives the known form on the decoy too, its answer on the real table did not come from
the rows.

**B. Pipe friction** (`nikuradse_superpipe`, 407 lines, no accepted closed formula). Rows with a
value that is not positive are dropped. The rows of two whole pipes are left out: with the
roughness values (rounded to three figures) that occur at least ten times sorted from smoothest,
the second and the fifth; the second if there are three, the third if four; every third row by
Reynolds number if there are fewer. The reading, with the search stage, and the product each read
a law for log10 f from log10 Re and log10 of the roughness on the other rows. Each is scored by
root mean square error in log10 f on the pipes left out, beside Haaland's formula and Colebrook
and White's with their published constants, and Haaland's form refitted on the same rows. If the
file's friction factors are a quarter of Haaland's (median ratio between 0.125 and 0.5 on the work
rows) they are multiplied by four: Fanning to Darcy.

**C. The expansion rate** (`cosmic_chronometers`, 32 lines: redshift, rate, stated error). The
second, fifth, eighth … measurement by redshift is left out. The reading, with the search stage,
reads a law from the rest in units of each measurement's stated error, with "positive" and
"increasing" stated from redshift zero to a fifth beyond the largest measured. It is scored by mean squared
error in units of the stated errors on the measurements left out, beside the standard model's form
`a*sqrt(b*(1+z)**3 + 1 - b)` and a straight line, both fitted on the same rows. With 32
measurements the one at the largest redshift is among those left out, so it is predicted a little
beyond the rows read.

**D. Generated small tables, drawn after freezing.** 360 laws of the invented-law grammar from
seed 2026112477, used nowhere else: ten for each of three levels (one input with two parts, two
inputs with two and with three), four sizes (8, 12, 20, 30 rows) and three noise levels (0, 1 %,
5 % of the target's size). Every structure of the 120 development laws and of the invented-laws
benchmark (by fingerprint) is left out. Three arms, no search stage in any: the reading; the
reading without the memory of confirmed laws; and the route in force called directly, which
without a search stage is its first pass alone and by its own rule reports nothing. A law is
scored on 400 clean rows drawn from the same range as the table: accurate at R² 0.999, and the
true form or not. A **wrong report** is a reported law that is not accurate. A table that ends in
an error is recorded as nothing given by either arm. The arm is complete if every table is there
once. Shards are taken across a fixed shuffle of the tables.

**E. Traps: 360 generated tables of four kinds the reading was not developed on**, 90 of each,
at 12, 20 and 30 rows, seeds 2026112478 to 2026112480, scored as D:

- **outside the grammar:** laws with an exponential, logarithm, sine, tanh or root of something
  that is not a line in one input: a function of a function (`sin(2.9*x0**2)`) or two inputs
  joined inside a function (`exp(-0.3*x0*x1)`). The development grammar cannot draw these (checked
  on its 570 laws used here), and the reading's candidates exclude one function inside another.
  Written for this test; the reading has been run on none of them. Noise 0 and 1 %;
- **harder laws:** the development grammar with three inputs and three or four parts. Development
  stopped at two inputs and three parts. Noise 0 and 1 %, crossed with size and number of parts;
- **one gross error:** laws of arm D on rows of their own, noise 0.2 % or 1 %, and one row off by
  ten times the noise's standard deviation;
- **noise in proportion:** laws of arm D on rows of their own, each row off by 0.2 % or 1 % of its
  own value.

## Expectations, fixed now

Computed by `run_real_tables_blind.expectations`, as defined here.

- **A1. It finds known laws in real tables, from their rows.** A table counts if the arm under
  test gives the known form on it and does not give the known form on its decoy. At least 3 of the
  tables with at least six rows count (five are expected; the binary stars' table has four rows
  and is expected to be refused). Tentative counts: the question is whether it arrives at the form.
  A table on which the arm under test gave no answer in its time does not count, and neither does
  one whose decoy was not read: it was not shown that the answer depends on the rows.
- **A2. It reports no wrong law on a real table**, in any of its three readings of the six tables.
  If it reports no law at all on them, A2 decides nothing.
- **A3. It finds more than the route in force.** The arm under test gives the known form on more
  tables than the best of the route's four arms. An arm that gave no answer in its 25 minutes, or
  was not started, found nothing; how many did not answer is recorded beside A3, since A3 can hold
  for that reason alone. The route's two arms without a search stage always answer.
- **D1. Of the laws it reports on the 360 tables, at most 5 % are wrong.** With fewer than 20
  reported, or a shard missing, D1 decides nothing.
- **D2. It finds more than the first pass alone on small tables.** It has the true form on more
  tables where the first pass does not than the other way round.
- **E1. Of the laws it reports on the 360 traps, at most 5 % are wrong.** With fewer than 20
  reported, or a shard missing, E1 decides nothing.

If a table of arm A could not be read at all (an error in its place, or a job that died before
reaching it), A1 and A3 decide nothing until it has been run alone under the deviation rule; a
wrong report on any other table still misses A2.

Reported without a criterion: arms B and C; each kind of trap alone; every arm by size, by noise
and by number of inputs; the reading without the memory; the readings without the search stage
and without what is known; laws reported without the known form beside A2; the 95 % upper bound of
each share of wrong reports; time.

**What D1 and E1 can and cannot show.** They compare the observed share with 5 %. At about 45
reports, a reading whose true share of wrong reports is 10 % still passes about one time in six,
and 2 wrong of 45 leaves a 95 % upper bound of 13 %. "Accurate" means on clean rows inside the
range the table covers; nothing is claimed outside it. In development 7 of 17 reported laws were
accurate and not the true form.

**The author's own expectations**, recorded so that they can be wrong, and no criterion.
**A1 probably misses**, one chance in five that it holds: the exoplanets are likely to count;
Langmuir's second table probably gives the known form on its decoy as the first did; Hubble's
`a*D` is the reading's default and its decoy will show it; time dilation and Tully and Fisher are
open. A2 holds, three chances in four, or decides nothing. A3 holds, two chances in three. D1
holds, two chances in three: the rule of five rows was fitted on 120 tables and three wrong
reports. D2 holds. **E1 is less than even:** one gross error among twelve rows, and laws the
candidates cannot express, are what a small table cannot defend against, and the reading has no
check for either. In B the published formulas win: the reading's candidates do not include a
logarithm of a sum inside a power. In C the reading and the standard model's form come out about
equal, within the errors.

A miss is reported as a miss.

## What follows from the result, fixed now

Nothing is put in force by this run. What follows is what is proposed to the owner, and how the
reading is described.

- **If A1, D1 and E1 hold and A2 is not missed:** the reading is proposed to the owner as the
  product's answer for tables under 24 rows, described with these numbers. If A2 decided nothing,
  the description says that no law was reported on a real table, so its reports on real
  measurements are unmeasured.
- **If A2, D1 or E1 misses:** the reading is not proposed. Every wrong report is listed with its
  table. It is proposed that, until a repaired version passes a test on tables drawn after it, the
  reading give tentative laws only.
- **If D1 or E1 decides nothing:** the reading is not proposed, and is described as not measured
  on that point.
- **If A1 misses and A2, D1 and E1 hold:** its reports held up but it does not find known laws
  from the rows of real tables; it is described so and not proposed.
- Whatever happens: the eight held-back tables are used up as a test of anything developed
  afterwards. Any later test on real measurements needs tables not in this registry.

## Disclosed before freezing

- **Every generated law except the 90 outside the grammar comes from the development grammar.**
  Arm D is inside the development distribution at the same sizes and noise levels, and the rule of
  five rows was fitted there. Three of the four traps change the number of inputs and parts, the
  noise model or one row, not the grammar. Only the 90 laws outside the grammar, and the real
  tables, come from outside it; the 90 were written by the author of the reading. No red-team
  search was aimed at the frozen reading.
- **The author knows the six laws of arm A and chose the statements, the two declared forms and
  the decoys knowing them.** That is what "known beforehand" means, but it also means arm A's
  informed readings measure the reading with a well-informed user. The reading with nothing stated
  is recorded beside them.
- **Two of the five readable tables carry a law the reading was developed on** (Kepler's and
  Langmuir's, on other measurements). The candidates and the statements were shaped on those laws.
- **Candidate forms are proposed from all rows.** The first pass, the memory and the search stage
  see every row before any is left out; only the constants are fitted without the left-out row. So
  the left-out error flatters forms that were found on the full table.
- **One row of one held-back table has been seen:** the first line of the chronometer file,
  printed once while looking for a header (redshift 0.07). No other held-back row has been read.
  Its headers, formats and units are taken from the sources' README files and loaders.
- **Five tables decide A1.** One table more or less changes it. A1 is a first measurement on real
  data and is described as such whatever it shows.
- **The known law of time dilation cannot be fitted as written:** at laboratory speeds only the
  ratio of its two constants is determined. `a*v**2` is declared beside it for that reason. With
  seven rows, only a law with one constant can be reported there.
- **Two spellings of one law can make a choice unstable.** The reading tells forms apart by how
  they are written. If the search stage proposes `p/(c*p + c)` beside `c*p/(p + c)`, the two tie to
  the seventh digit and removing a row can flip the choice, which the reading then calls
  unstable and does not report. This costs reports; it cannot make a wrong one. Found by the
  second review; not repaired.
- **The way each search-stage part runs here was rehearsed, not measured.** Each runs in a
  process of its own with a limit. The rehearsal on the development tables is recorded in the
  review file with its GitHub run.
- **A reading depends on machine load.** The memory of confirmed laws is given five seconds, and
  what it proposes in that time varies: of 120 development tables read twice, 4 came out
  differently. The run is made once, on GitHub's runners.
- **The route in force with its search stage went silent on a small table the day of freezing.**
  On the three development tables (GitHub run 45434942785) it printed nothing for two hours after
  its first search on six rows and was cancelled by its job's limit, with nothing written. The cause
  was not found. That is why its arms here have a process and 25 minutes each, and why the reading
  is read on every table before them. The reading with the same search stage finished all six
  development readings in 18 minutes (run 45446253975).
- **The first pass alone is a weak rival.** Without a search stage the route in force reports
  nothing by its own rule, and it has no memory. D2 measures the reading against that, and the
  reading without the memory is recorded beside it.
- **`REFEREE_MAX_FPR` is borrowed.** It is a referee's allowed share of false positives; here it
  bounds the share of wrong laws among those reported, as in four earlier plans. No named
  threshold means that. Naming one is the owner's call (AGENTS.md 1.3) and is asked for with the
  result.
- **The product's endpoint is not frozen by the digest** (`apedp_dyn/ui/route_endpoint.py` is the
  product's live file). The record carries the commit the run was made at.
- **The pipe-friction source is under the GPL** and is never copied into this repository; the job
  reads it and keeps only the scores.
- **Unequal cost.** The reading takes about 50 seconds on a small table without a search stage;
  the first pass takes under a second on the same table.

## Deviation rule

The run is not repeated. Re-running a job from GitHub's page does not start the real tables' job
again; every repeat below is started by a branch of its own name and recorded with its date.

- If the check before reading stops the run, nothing was read. The cause is recorded, and arms A,
  B and C are started again by a branch `real-tables-blind-real-<n>`; if the cause is in a frozen
  file, the digest is written again with the reason. Arms D and E of the first branch stand.
- A shard of arm D or E lost to infrastructure is run again alone by a branch
  `real-tables-blind-shard-<arm>-<k>-<n>`. A shard that finished is not run again.
- A part of the real run that has an error in its place (a table of arm A, pipe friction, or the
  expansion rate), or that a dead job never reached, is run alone by a branch
  `real-tables-blind-only-<part>-<n>`. `collect` puts it in the place of the failed part and never
  replaces a part that was read; of two runs of one part, the earlier stands.
- A reading that gave no answer in its time is an answer: it is not run again.

Any other change to a frozen file after jobs have started makes the result a development result.

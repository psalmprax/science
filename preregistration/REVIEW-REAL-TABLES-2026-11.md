# Reviews of PREREG-REAL-TABLES-2026-11 before freezing

AGENTS.md 1.6 requires a reviewer who took no part in the work to check a plan cold before it is
frozen. On 2026-11-24 a separate agent, started with no knowledge of how the work was done, was
given the draft plan, the runner, the modules under test, the data definitions, the workflow and
the tests. It was told to change nothing, to read no held-back table and to run no reading on
the blind tables. It read the code, drew the trap laws once to count them, and called the judging
functions on made-up numbers.

**Its verdict on the draft: do not freeze as written; three blockers.** All thirteen findings and
four minor points are below, with what was done about each.

A **second reviewer**, also started cold, then read the changed version. **Its verdict: do not
freeze yet; two blockers**, one of them a judging error that could have turned a right reported
law into a wrong report. Its findings and what was done follow the first review's below. It
confirmed that every change claimed after the first review was made, with two reservations that
are among its own findings.

The plan and runner that are frozen are the ones after both sets of changes. No reviewer has seen
the last version. What stands in for a third review is a **rehearsal**: the whole of the real
tables' run on the three development tables and on made-up tables for arms B and C, with the
search stage on, recorded at the end of this file.

## First review: blockers

| | Finding | What was done |
| :--- | :--- | :--- |
| 1 | The trap arm's "another generator" was the development grammar under another name: the counterfeit benchmark's generator draws the same parts, joins and constants. No trap law came from outside the grammar, so the arm did not meet AGENTS.md 1.6 as described. | Replaced. The 90 laws "outside the grammar" have a function of a function or two inputs joined inside a function. A test checks that none of the 570 grammar laws used here has that property and that all 90 do. The plan now says that every other generated law comes from the development grammar. |
| 2 | The deviation rule could not be carried out by the frozen runner, a lost table gave a misleading record (A1 false and A2 true with every download failed), a job killed late left A1 to A3 uncomputed, and a download was tried once. | Every table is now fetched (three attempts), checked and parsed before any reading; a failure stops the run with nothing read. `real --only PART` runs a failed part alone, started by a branch `real-tables-blind-only-<part>-<n>`; `collect` puts it in place and never replaces a part that was read. A1 and A3 decide nothing while a table has an error in its place. Arm A is judged from its six tables whatever happens to B and C. |
| 3 | A2 passed with no reports; E1 would probably have too few reports to decide (10 to 20 against 20 needed); the gross-error trap was nearly inert at 5 % noise; and the plan did not say what an undecided D1 or E1 means. | A2 decides nothing if no law is reported on a real table. An undecided D1 or E1 means not proposed. The traps are now 360 tables at 12, 20 and 30 rows; the two noise traps are at 0.2 % and 1 % noise, where the reading does report. |

## First review: should fix

| | Finding | What was done |
| :--- | :--- | :--- |
| 4 | "Told apart by the table" was lenient on noisy tables and a coin toss on clean ones. | Dropped from A2. A wrong report on a real table is any reported law without the known form. The distance from the known law is recorded and decides nothing. |
| 5 | A1 at 3 of 5 was an easy goalpost: the exoplanets' candidate was built for that law, the statements about adsorption leave one form, `a*D` is the plainest form there is, and nothing showed the answer depends on the rows. | Each table now has a decoy: another law that satisfies everything stated, on the same inputs, with the table's own scatter. A table counts for A1 only if the known form is given on it and not on its decoy. The three points are stated in the plan. On the development tables the decoy already bites: Langmuir's table gives the known form on its decoy and does not count. The author's forecast for A1 is now that it probably misses. |
| 6 | "The known form" was looser than the plan said: `450*D + 12` passed for `a*D`, and a constant that refits to nothing passed inside a ratio or a logarithm. | The known form now also requires no more fitted constants than the known law has. Tested with `a*D + b` against `a*D`. |
| 7 | D1 and E1 have little power, and "relied on" claimed more than accuracy inside the sampled range. | The plan states the power (one pass in six at a true 10 %), prints the 95 % upper bound of each share, and words both as observed shares of laws accurate inside the table's range. |
| 8 | D2's rival cannot report without a search stage and has no memory, while 8 of the reading's 17 true forms came from the memory. | D2 is named "more than the first pass alone". A third arm, the reading without the memory, is run on arm D. Both points are disclosed. (In the second development run it was 7 of 17.) |
| 9 | The search stage of the arm under test was not recorded, and runs four searches only with four cores. | The record carries the stage's description and the core count; the run stops before reading with fewer than four cores. Workers follow `nproc`. |
| 10 | Once-only robustness: `collect` crashed on a truncated file; scoring in B and C could discard a reading already made; one non-numeric cell lost a table; `collect` uploaded nothing. | The record is written to a scratch file and moved into place; `collect` skips and counts a cut line and treats an unreadable record as missing; scoring in B and C is guarded apart from the readings; a non-numeric cell is a missing value and only the needed columns are used; `collect` uploads its results. Each has a test. |
| 11 | Plan and runner disagreed: the range of arm C's statements; the "nothing stated" reading was given the table's noise model; "two and three inputs" was an accident of the draw; traps had no breakdown by size and noise; several details of arm B, the search stage in B and C, 900 against 300 seconds, and the seed were unstated. | The plain reading now gets no noise model. The plan's words were corrected or completed on each point, and every arm is broken down by size, noise, inputs and kind. |
| 12 | Development numbers were not all from the frozen code, and the module's docstring said 19 reported where the record said 16. | The 120-table comparison was run again with the code as frozen (17 reported, none wrong) and the three real development tables were read again by the frozen arm-A code. The plan and the docstring give those numbers. |
| 13 | Not disclosed: two of five readable tables carry a law the reading was developed on; candidate forms are proposed from all rows before any is left out. No review record. The digest must cover the law generators and the memory. | Both are disclosed. This file is the review record. The digest covers `curriculum.py`, the fingerprints file and `structure_memory.json`; the counterfeit generator is no longer used. |

## First review: minor

| Finding | What was done |
| :--- | :--- |
| Time dilation's registered law cannot be fitted as written (only the ratio of its two constants is determined). | Disclosed; `a*v**2` is declared beside it for that reason. |
| The exoplanets' declared form rested on rows not yet seen. | A declared form now counts only if, fitted to the table, it is within the measurements' scatter of the known law. Tested with planets (counts) and with two stars (does not). |
| `REFEREE_MAX_FPR` is borrowed for a share of reports (AGENTS.md 1.3). | Disclosed. A named threshold for it is the owner's call and is asked for with the result. |
| "May only give tentative laws" read as a decision. | Reworded as a proposal. |

## What the first reviewer checked and found correct

Arm D's 36 cells of ten, its seed and its exclusions; the thresholds and the "decides nothing"
logic; that no test row or law text reaches a reading; that the splits in B and C use inputs only
and the Fanning rule uses work rows; that held-back files stay outside the uploaded folder; that
nothing is put in force.

## What the first reviewer could not check

The held-back files' headers, formats and units; the time three-input readings take; the cores
GitHub gives this repository; whether a re-run job can upload an artifact of the same name; the
digest, which did not exist yet.

## Between the reviews: a stall

While the second review was running, the development run of the route in force with its search
stage (GitHub run 45434942785) was found cancelled by its job's two-hour limit: after its first
search on six rows it had printed nothing more, and it wrote nothing. An error can be caught and
recorded; silence cannot. So every part of the run that uses the search stage was given a
process of its own with a hard limit, and the reading is now read on every table before any of
the route's arms. The second reviewer saw this change arrive while it worked (its finding 8).

## Second review: blockers

| | Finding | What was done |
| :--- | :--- | :--- |
| 1 | The count of constants included the factor the reading writes when it puts its law back in the table's units. Langmuir's own form, written `p/(c*p + c)` times the target's size, counted three constants against two: judged not the known form, and a wrong report if reported. Reproduced on made-up rows. The same miscount applied to the decoy. | For the reading's laws the count is now the reading's own record of what it fitted; the text is counted only for the route's laws. Tested on the reviewer's case and end to end in other units. |
| 2 | A1 counted a decoy that was not read, or an arm under test that gave no answer, against the table, and the plan did not say so. | The plan says so in A1: such a table does not count. The record lists them. |

## Second review: should fix

| | Finding | What was done |
| :--- | :--- | :--- |
| 3 | A child's limit could shrink to 60 seconds late in the run, a certain "no answer" on a held-back table. | A reading's part always has its full limit. One of the route's arms is started only if its full limit fits in the run's time; otherwise it is recorded as not started. |
| 4 | The limit (20 minutes) was less than a reading may rightly take (900 s to score, the stage's own 390 s, and starting Julia). | A reading now has 600 s to score and a child 25 minutes. A test pins that the first fits in the second. |
| 5 | No safe way to run a lost shard again: a new run branch or a job re-run would read the held-back tables a second time. | The real tables' job runs only on a job's first attempt. Repeats are started by branches of their own names: one shard, the real job after a stop before reading, or one failed part. The plan says which results stand. |
| 6 | The judge's own fit of the known law could fail without a sound: with speeds in metres a second the time-dilation fit was not finite, and with a target of 1e12 Kepler's fit stopped far from the answer. | The fit's starts are now spread over the table's own scales (tested at 1e12, 1e-15 and on a made-up clock table). The check before reading fits the known law and builds the decoy for every table and stops the run if either fails. |
| 7 | The check before reading accepted any table with one usable row. | It stops if a table has lost more than two rows against its file's line count. |
| 8 | The path with a child process and the search stage on had never been run; a child that raised left only an exit code; a child that ended left what it had started running. | Rehearsed (below). A child now records its own error, and its process group is stopped after it answers as well as at its limit. |
| 9 | A record of only some tables gave no A1 to A3 at all, though the plan says a wrong report on any table misses A2. | A2 is false on any wrong report in a partial record; A1 and A3 decide nothing while a table is missing. |
| 10 | "Complete" counted rows, not which rows: 120 tables twice and 120 missing counted as complete. | An arm is complete only if every table is there once. |
| 11 | The decoy's scatter was added as a difference, though two tables are read with errors in proportion: the decoy was noisier than the table in the reading's own measure. | The scatter is carried over as each row's ratio where the reading of the real table counted errors in proportion. |

## Second review: minor, and not disclosed

| Finding | What was done |
| :--- | :--- |
| "No more constants than the known law has" was not what the code compared, and fewer constants passed: `a*log(DV)` for `a*log(DV) + b`. | The rule is now exactly as many constants as the form matched, in the plan and in the code. |
| With 32 rows the largest redshift is always among those left out. | Stated in the plan. |
| No upper bound was given when every report was wrong. | It is 1. |
| In the harder-laws trap, number of parts, size and noise were tied together; each shard of arm D was one level and each trap shard one size. | Size and noise are crossed with the number of parts by a fixed shuffle, and shards are taken across a fixed shuffle. |
| On exact data, whether a declared form is "within the scatter" was decided by rounding. | A difference below 1e-9 of the law's size counts as none. |
| One exception ended a whole shard of arm D or E. | A table that raises is recorded as nothing given by either arm. |
| Parts run alone were ordered by file name; `collect` raised on records the runner never writes. | Ordered by date, the earlier of two runs of a part stands; malformed rows and records are skipped and counted. |
| A3 can hold only because every rival was stopped. | Stated in A3; the number of arms that gave no answer is recorded beside it. |
| Not disclosed: two spellings of one law can make the choice unstable with the search stage on. | Disclosed. Not repaired: it costs reports and cannot make a wrong one. |

## What the second reviewer checked and found correct

The 360 laws of arm D and the 4 × 90 traps, their cells, indices, tags and rows of their own; the
gross error (one row at ten standard deviations) and the noise in proportion; the 5 % boundary,
the 13 % bound and the one-in-six figure; that `collect` never replaces a part that was read and
always replaces a failed one; branch filters, part names and artifact paths; that held-back files
and job files stay in the temporary folder; the pipe split, the Fanning rule and the
expansion-rate split; parsing of `d**(3/2)` and scientific notation; a four-row table; targets of
size 1e-15 end to end.

## What it could not check

The held-back files; GitHub's cores and artifact behaviour on a re-run; anything with PySR running.

## The rehearsal

Run on GitHub on 2026-11-24 by the branch `real-tables-blind-rehearsal-1` (run 45476079433, commit
`efc21eb`), with the search stage as the run will have it: four cores, "four searches side by side
(PREREG-PORTFOLIO-SEARCH), 90 s; and laws grown from 28 remembered motifs". The tables were the three
development tables for arm A and two made-up tables for arms B and C (ten pipes on Colebrook's formula
with 1 % scatter; 32 measurements on the standard model's form). No held-back table was read. The whole
record is `benchmarks/real_data/dev_rehearsal_on_github.json`.

**Every part answered.** 39 minutes in all (2,356 s); no child process was killed, none reached its
limit, none recorded an error of its own.

| Part | Times run | Seconds each |
|---|---|---|
| The reading without the search stage (plain, informed) | 6 | 9 to 19 |
| The reading with the search stage | 3 | 151 to 180 |
| The reading of a decoy | 3 | 147 to 152 |
| The route in force, direct, with the search stage | 3 | 149 to 151 |
| The route in force, direct, first pass alone | 3 | under 1 |
| The product, with the search stage | 1 | 151 |
| The product, first pass alone | 1 | under 1 |
| The product on the two tables under 24 rows | 4 | refused at once, as the product does ("8 usable rows; the route needs at least 24") |
| Arm B: the reading, then the product | 1 each | 291 and 198 |
| Arm C: the reading | 1 | 159 |

At these times the run's six tables and arms B and C need about an hour and a half of the 340 minutes the run allows,
so every arm of the route can be started with its full limit.

**The search stage reached the reading on every table.** On the 8-row and 10-row tables none of its
laws became a candidate: the count of forms left out for having more constants than the rows allow
rose by nine on each (31 to 40, 1 to 10), which is where they went. On the 26-row table five of its
laws were candidates. On none of the three did it change the reading's choice.

**What the development tables gave** is what they gave without the search stage
(`benchmarks/real_data/dev_arm_a_as_frozen.json`): one table of three counts for expectation A1.
Kepler's law was reported, and on its decoy the reading reported the decoy's own law. Langmuir's form
was given but not reported, and was given on its decoy too, so it does not count. Leavitt's line was
not given (the reading's choice was a cosine, not reported). No law was reported that is not the known
form. The route in force gave Kepler's form without reporting it, and the known form on neither of the
other two.

**The made-up tables.** On the pipes made from Colebrook's formula the reading reported a law that is
not Colebrook's form. Its error on the two pipes left out was 0.0073, against 0.0045 for Colebrook's
formula itself (the scatter put in) and 0.0047 for Haaland's form refitted. So on a table whose formula
is known exactly, the reading's reported law was 1.6 times the scatter away on pipes it had not seen.
This is what arm B is there to show and it is not a fault of the run. On the made-up expansion rates
the reading's law was not reported.

**Changed because of the rehearsal: nothing.** The 23 frozen files match the digest as it was written
before the rehearsal (`python scripts/run_real_tables_blind.py check`). The commits made to `stage`
between the rehearsed commit and the run change none of those files; of the code they change, none is
on the run's path (the LLM arms and their scripts, the experiment ledger's `register_candidate`, which
no discovery route calls, the recursion judge's list of sealed files, the Matter gate, the CHANGELOG
index).

**The plan is frozen as of this section.** The run is started by pushing `stage` to
`real-tables-blind-run-1`.

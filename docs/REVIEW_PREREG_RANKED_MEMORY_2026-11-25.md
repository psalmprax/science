# Independent review of PREREG-RANKED-MEMORY-2026-11, before freezing (25 November 2026)

AGENTS.md 1.6 asks that a separate reviewer, who took no part in the work, checks a
preregistration cold before it is frozen, and that its findings and what was done about each go
into the record. This is that record.

- **Reviewed:** the draft plan `preregistration/PREREG-RANKED-MEMORY-2026-11.md`, its runner
  `scripts/run_ranked_memory_blind.py`, the stage `apedp_dyn/discovery_l2/ranked_stage.py`, the
  proposer and its order, the workflow, the tests, the development records and the first dress
  rehearsal. State of the files: commit `619df8f` (the reviewer began on `394b0bf`; the stage
  gained its gate while the review ran, and the reviewer re-read what changed).
- **Reviewer:** a separate agent started for this review only, read-only, with no part in the
  work. It was told not to edit the repository, not to draw or look for the laws of the run, and
  not to run the run.
- **Verdict:** **not to be frozen as it stands.** The first finding needs the owner.
- **What happened next:** the plan is held back. Nothing is frozen, no law is drawn, nothing was
  run on the plan. Section 1 says what was done about each finding; section 2 is the reviewer's
  report as returned, unedited apart from line wrapping.

## 1. What was done about each finding

| Finding | Severity given | What was done |
| :-- | :-- | :-- |
| 1. The run can register a difference in very few of its 200 runs; the plan forecasts the opposite | Blocks freezing | **Brought to the owner; not decided by the agent.** Checked by the agent with three seeds other than the run's, counts only: 2, 2 and 3 one-input laws in a hundred (the reviewer: 0 to 6, mean 2.2). The plan now carries a status block saying it is held back and why, says how many laws of each number of inputs the draw gives, says where the order's gain lies (one input: 14 to 28; two: 14 to 14; three: none), and marks its own forecast as unsupported. Whether the test is run in this form, changed, or not run is the owner's decision. |
| 2. Expectation 2 cannot tell a gain of the forecast size from none, and a miss is worded as a finding | Should fix | Not yet acted on: it depends on the decision on finding 1. If the test is run, the consequence of a miss is reworded as the reviewer asks ("was not shown, on these laws, to make the route report more right laws") and the plan states the chances the reviewer computed. |
| 3. A crash or timeout of the ranked arm erases the result of the arm in force | Should fix | Not yet acted on. A fault of the runner whatever is decided; to be repaired before any freeze (the in-force row handed over as soon as it is made). |
| 4. The frozen list leaves out files on the path of every run, and the workflow | Should fix | Not yet acted on; the files named are added before any freeze. |
| 5. `collect` judges whatever rows it finds | Should fix | Not yet acted on; `collect` is made to refuse unless every unit and arm is there exactly once. |
| 6. The rows cannot say what the stage did, or what the gate left out | Should fix | Not yet acted on (the stage's own record, the core count, the texts left out by the gate, whether the law was among those left out or kept, a refusal to run with fewer than four cores or with a switch set). |
| 7. The consequences leave one case open, and the second run's test is now met by doing nothing | Should fix | Not yet acted on; depends on finding 1. |
| 8. Two statements about the order that the repository does not bear out | Should fix | **Done.** Both corrected in place in the plan and marked as corrected: the order was not "written before any list of the second set was read" (no list of the second set entered its fit; it replaced an earlier order after such lists existed), and it is the best of three orders scored (37, 37, 42), not "the better of two looks". Both wrong statements were the agent's. |
| 9. A test draws the run's first eight laws; the runner's central function has no test | Should fix | **Done in part.** The test now draws eight laws with another seed and asserts that it is not the run's; it no longer compares the laws file in a test. The run's seed was changed (20261125 to 2026112699), so that no law of the run has been drawn by anything; the plan says so. The agent ran that test file several times on 25 November, locally and on GitHub: each run drew the first eight laws of the old seed in memory; none was printed or stored. Still to do: a test of `arms_of` with a stand-in stage. |
| 10. Minor | Minor | "Cannot be the law" qualified in the plan (it holds on noisy rows only). The rest not yet acted on. |
| Incidental: RESULT-MEMORY-2026-11's "the memory's laws were the law itself in 26 of the 80 runs" | (an earlier record) | **Checked against that run's rows and found as the reviewer says.** The runner compared the path's rewritten candidates with the searches' raw texts, so it took every candidate of the stage for one of the memory's (`handed_over` is 10 or 11 in all 80 rows, where the memory proposes three). The 26 is the number of runs in which *any* candidate of the stage was the law. The result's two expectations rest on other counts and are not touched. An erratum stands beside the result: `preregistration/ERRATUM-RESULT-MEMORY-2026-11.md`. |

The reviewer also asks that, if the runner is changed for findings 3 to 6, the rehearsal is run
once more with the runner that will be frozen, and that nothing in the stage changes after the
second rehearsal is read unless the plan says what did. Both are accepted.

## 1a. The second rehearsal, read after the review

The second rehearsal (run 45670861301, the stage with its gate, the same sixteen development laws,
32 runs an arm; not the run) finished after the report was returned. Read against the list the
reviewer asked for (`benchmarks/ranked_memory/rehearsal2_results.json`, `rehearsal2_runs.jsonl.gz`):

1. 64 rows, each unit once an arm, no error and no timeout.
2. The same ten candidates of the stage in both rows of all 32 units; in every unit with nothing
   added the two rows have the same outcome.
3. A candidate was added in 10 of 32 runs, 25 candidates in all, 225 left out as fitting worse.
   **None without noise (0 of 16) and none on three inputs (0 of 8)**; at 1 % noise in 7 of 7 runs
   with one input and 3 of 5 with two. This is what the reviewer said it would be.
4. `rehearsal_10` and `rehearsal_13` at 1 % are no longer lost; `rehearsal_8` at 1 % is still
   gained (the law itself, on a one-input law whose structure is among those the order was learned
   on). Reports 24 against 23, none wrong; recovered 14 against 13; run for run one gained, none
   lost. The searches were run afresh, so this is a development reading and not a comparison with
   the first rehearsal.
5. The core count and the stage's own record are not in the rows (finding 6 is not acted on yet).
6. The longest unit took 627 seconds, both arms together, against a limit of 1,800.
7. By number of inputs the one gain is on one input. Seven of the sixteen laws have one input; the
   run would have about two in a hundred.
8. Nothing in the stage was changed after this rehearsal was read.

It bears out finding 1: the stage now does what it was changed to do, and on the laws the run would
draw it would seldom do anything.

## 2. The reviewer's report, as returned

> `/tmp/review_ranked_memory/findings.md` was not written: the harness refuses report files from
> a subagent. The full text follows; the scripts and rows behind it are in
> `/tmp/review_ranked_memory/`. Nothing in it comes from stored memory.

### Review before freezing: PREREG-RANKED-MEMORY-2026-11

Independent reviewer, no part in the work. Read-only: nothing in `/home/claude/science` was
edited, committed or pushed, and the 100 laws were not drawn or looked for. State reviewed: branch
`stage` at `ca8e581`; the files under review are as commit `619df8f` left them (25 November,
19:01).

**The draft moved under the review.** I began on `394b0bf`. Between 18:54 and 19:01 the stage
(`ranked_stage.py`: a gate), the plan, two lines of the runner and `tests/test_ranked_stage.py`
were changed on the first rehearsal's rows, and that rehearsal's records were added. I re-read
those files and redid everything that depended on them. Every statement below is about the present
files. A second rehearsal is running; I do not have it.

#### Verdict

Not to be frozen as it stands. It can be frozen after the changes in findings 1 to 9; finding 1
needs the owner, and the second rehearsal, which the plan requires before freezing, has not been
read.

The bookkeeping is sound. Every number the plan quotes from a record reproduces from the raw rows.
The three expectations are coded as written. Both arms get the same candidates and nothing random
differs between them. No pool or scoring row reaches the ranked proposer, its order or its gate.

The fault is in what the run can show. About 2 of the 100 laws will have one input, where all of
the order's development gain and the rehearsal's one gain lie. On three-input tables the ranked
list holds at most four candidates and none has passed the gate; without noise the gate has let
nothing in. The two arms will be the same candidate list in roughly 170 of 200 runs, and in the 23
runs seen with an addition the outcome changed once. The plan forecasts "the law itself in about
15 of 200" and "3 to 6 right laws" gained, and on a miss would record that the order "does not
make the route report more right laws". The repository's records support neither.

Beside that, the runner erases the arm in force's result when the ranked arm crashes, its frozen
list leaves out files on the call path, `collect` judges whatever rows it finds, and the rows
cannot say what the stage did or whether the gate left out the law.

#### Findings, most serious first

##### 1. The run can register a difference in very few of its 200 runs; the plan forecasts the opposite. BLOCKS FREEZING (checks 4, 6, 8, 9)

Where: plan lines 18-24, 68-75, 117-126, 141-143; `scripts/run_ranked_memory_blind.py` 93-129;
`apedp_dyn/discovery_l2/ranked_stage.py` 56, 124-140; `apedp_dyn/proposers/ranked_memory.py`
71-72; `apedp_dyn/proposers/structure_memory_counted.py` 32; `scripts/measure_order_score.py` 60;
`benchmarks/order_score/dev_learned_order_in_force.json`; `benchmarks/ranked_memory/rehearsal1_*`.

Plan: "among the first three for 28 as handed and for 42 in the learned order"; "adds a candidate
in about a third of the runs and adds the law itself in about 15 of 200 ... the route gains 3 to 6
right laws".

Records and code:

- **Where the 28 to 42 is.** By level, from the record's own `within_3_by_level`: one input 14 to
  28; two inputs 8 to 9 and 6 to 5; three inputs 0 to 0 and 0 to 0. In the stage's own design the
  added three hold a solved candidate the first three lack on 16 of 200 problems: 14 one-input, 2
  two-input, none of the other 120. By form the law is on 4 of the 200 lists and already first on
  all 4.
- **What the draw will hold.** With the plan's exclusions 97 % of acceptable one-input draws fall
  on an excluded structure (171 of 187 structures seen). Sixteen draws of 100 with seeds other
  than the run's (counts only, no law kept or printed) gave 0 to 6 one-input laws, mean 2.2; two
  inputs 55; three inputs 43. Without exclusions the generator gives about 27, 50 and 22.
  `measure_order_score.py` line 60 already notes the cause.
- **Three inputs.** The counted growth (1,500 fits) does not reach a law in all three inputs, so
  the list is 2 to 4 remembered whole shapes in all 200 three-input development problems. The
  order has nothing to choose. No candidate was within the gate in 15 noisy three-input runs (4
  rehearsal, 11 replay).
- **Without noise.** First rehearsal: none of the candidates added in its 16 noise-free runs was
  within the gate. Replay with the gate in place: nothing added in 37 noise-free units.
- **So** the arms can differ only in noisy runs on one- and two-input laws, about 57 of 200. An
  added candidate was within the gate in 12 of 13 such one-input runs and 11 of 23 two-input runs
  seen. That is about 28 of 200 for this draw, one run in seven, not a third.
- **What an addition changed, rehearsal.** In the nine runs with an added candidate within the
  gate, one outcome changed: the gain, on a one-input law (`rehearsal_8`) whose structure is among
  the 300 problems the order was learned on. All seven one-input rehearsal laws have a structure
  in the order's curriculum sets; none of the nine others has.
- **What an addition changed, replay.** I replayed the present stage on 36 laws of PREREG-MEMORY
  (72 units, recorded candidates, the route and last checks in force, no search). A candidate was
  added in 14 units, all noisy. Right laws 49 against 49, recovered 23 against 23; nothing gained,
  nothing lost. The law was among the added once, and the stage already held it.
- **The forecast.** By the development rates and the draw's make-up, the added candidates hold a
  solved candidate the first three lack in 3 to 4 runs of 200, before the gate. The 15 is the
  rehearsal's rate on sixteen development laws, seven of them one-input.

Smallest change: text, then the owner. No code change.

1. Under *Laws*, give the count by number of inputs when drawn, and say now what the generator
   gives (about 2, 55, 43).
2. Under *Disclosed*, say that the development gain is on one-input laws, that on three inputs the
   list holds at most four and none has passed the gate, that without noise the gate has let
   nothing in, and that the arms differed in outcome in 1 of 23 runs with an addition.
3. Replace the forecast, or mark it as written without these numbers.
4. Reword the consequence of a miss (finding 2).
5. Tell the owner, who was told "adds three more" and asked for "more laws reported", that the
   stage now adds nothing in most runs and what the laws will be. Take their word on running it in
   this form.

##### 2. Expectation 2 cannot tell a gain of the forecast size from none, and a miss is worded as a finding. SHOULD FIX BEFORE FREEZING (checks 2, 9)

Where: plan 107-108, 124-125, 141-143; runner 331-332.

As written and as coded: at least as many right laws at each noise level, more in total, no
margin. With the gate nothing changes without noise, so the first clause is idle there. One run
gained and none lost makes it hold (sign test 1.0); no run changed makes it miss.

Chances, with gains and losses falling independently on the noisy runs (Poisson):

| Mean gained, mean lost | Expectation 2 holds | No run changed at all |
| :--- | ---: | ---: |
| 4.5, 2.5 (middle of the plan's own ranges) | 0.71 | 0.00 |
| 3, 4 and 6, 1 (their corners) | 0.28, 0.97 | 0.00 |
| 3, 3 and 2, 2 (no net gain) | 0.42, 0.40 | 0.00, 0.02 |
| 1, 1 | 0.35 | 0.14 |
| 0.5, 0.5 | 0.27 | 0.37 |
| 1, 0.5 (about what finding 1 leaves) | 0.47 | 0.22 |
| 0.3, 0.3 | 0.20 | 0.55 |

- At the middle of the plan's own ranges it holds seven times in ten, not "about four chances in
  ten".
- A stage whose additions change reports as often for worse as for better still holds it about one
  time in three.
- **If the ranked proposer did nothing useful.** The arms get the same candidates and the path's
  seeds are fixed, so a unit can differ only through an addition, and each addition removes one of
  the stage's last three.
  - With the stage as first written that meant displacement in every run. Replayed with the last
    three displaced and nothing added: 5 right laws lost of 72 units (1 without noise, 4 at 1 %),
    none gained; about 14 of 200 (5 to 31). In PREREG-MEMORY's rows 7 of the memory arm's 57
    reports came from those three places (49 could be placed). Expectation 2 would not hold.
  - With the gate little is displaced: 0 outcomes changed in 14 replayed units with an addition.
    The usual result is a tie, which is a miss.
- So a miss is the likely outcome whatever the order is worth on one-input laws, and a pass could
  rest on one run.

The plan should state this plainly. Smallest change: say it; reword 141-143 to "was not shown, on
these laws, to make the route report more right laws", with the runs gained and lost beside it;
say that fewer than six runs gained with none lost is within chance by the plan's own sign test
and leads to the second run only.

##### 3. A crash or timeout of the ranked arm erases the result of the arm in force. SHOULD FIX BEFORE FREEZING (check 1)

Where: plan 99; runner 176-225 (rows returned only at the end), 228-234, 258-261.

Plan: "counts as reporting nothing, in both arms." The code agrees. Executed with a stand-in stage
and a failure on the second route call: both rows `error`. The arm in force runs first, and the
ranked arm is the only extra cost and the only source of new candidates, so the rule can only
favour the ranked arm. Seen so far: 0 of 112 units failed (80 and 32); a wider-conditions run had
3 of 120 readings end in "SVD did not converge". One such unit is the size of the effect under
test.

Smallest change: hand the in-force row through the pipe as soon as it is made; give `error` or
`timeout` to the ranked arm alone when that row is in; reword line 99.

##### 4. The frozen list leaves out files on the path of every run, and the workflow. SHOULD FIX BEFORE FREEZING (check 5)

Where: runner 61-77, 345-350.

Not in `FROZEN`:

- `apedp_dyn/discovery_l2/contrast_resolution.py` (fits every candidate)
- `exhaustive_search.py`, `confirmed_stop.py`, `guided_search.py` (first pass, deeper pass,
  adequacy)
- `vouching.py`, `vouched_route.py` (the last checks' tests and row limit)
- `apedp_dyn/conf/discovery_checks.py`
- `apedp_referee/canonical_spec.py` with `apedp_referee/canonical_preregistered_spec_v1.json` (the
  confirmation's level)
- `.github/workflows/ranked-memory.yml` (runner image and so four searches or one, Python 3.12,
  pysr 2.7.2, 20 shards)

Earlier digests cover the first six and match today, but nothing checks an earlier digest at run
time. Nothing in the tuple must still change except the plan and the laws file, and the runner and
stage if these findings are acted on.

Smallest change: add them.

##### 5. `collect` judges whatever rows it finds. SHOULD FIX BEFORE FREEZING (checks 1, 2)

Where: runner 410-416; workflow 102-104 (`if: always()`).

There is no check for 400 rows, each law, noise level and arm once. A lost shard gives a summary
with `2_holds` on fewer runs. A shard present twice is counted twice in the totals and once in the
run-for-run counts.

Smallest change: refuse unless every unit and arm is there exactly once, as
`scripts/run_srbench_baseline.py` 408-411 does.

##### 6. The rows cannot say what the stage did, or what the gate left out. SHOULD FIX BEFORE FREEZING (checks 1, 3)

Where: runner 195, 200, 203, 211-222; `apedp_dyn/ode/extended_route.py` 41-64;
`apedp_dyn/proposers/hard_exit.py` 271-274; `ranked_stage.py` 142-145.

- **Four searches.** "Four searches side by side" holds only with four cores; with fewer the stage
  is one PySR search. This is not asserted or recorded, and ten candidates come back either way.
  The rehearsal's rows carry indirect signs of four (third candidate in the deeper pass's writing
  in 31 of 32; fifth and ninth free of exp, log, sin, cos in 32 of 32). That is my reading, not
  the record's. `Recording.last_run` holds members, proposals per member, `merged_from` and
  failures; the runner drops it.
- **Failures.** `proposer_failures` is empty for the arm in force by construction: `Recording` has
  no `failures`.
- **The gate.** Left-out candidates are counted, not kept, so the run cannot say whether the gate
  left out the law. It would have in `rehearsal_5` without noise.
- **Displacement.** "Displaced the law" is recorded without whether the law stayed among those
  kept. In both replayed cases it had.
- **Switches.** `APEDP_TABLE_MEMORY`, `APEDP_TABLE_VOUCHING` and `APEDP_LAW_DISCOVERY_ROUTE` are
  read at run time, neither pinned nor recorded.
- **Summary.** `summarise` does not count the path's actions or what the checks held back, which
  item 4 promises.

Smallest change: store `stage.last_run` and the core count in the in-force row; keep the left-out
texts in `WithRanked._report`; add `the_law_among_the_left_out` and `the_law_among_the_kept`;
refuse `run` with fewer than four cores or any of the three variables set.

##### 7. The consequences leave one case open, and the second run's test is now met by doing nothing. SHOULD FIX BEFORE FREEZING (check 7)

Where: plan 135-145, 181-183.

- Expectation 1 missing for the arm in force alone, with 2 and 3 holding, has no consequence:
  `1_holds` is false, and "If 1 misses for the ranked arm" does not apply.
- "Proposed for tables only if it loses no more right laws there than it gains" was written before
  the gate. On laws the memory's parts do not fit, the gate leaves everything out: nothing lost,
  nothing gained, criterion met.
- Nothing says what may change after the second rehearsal. Each rehearsal read before a change is
  another development round on the same sixteen laws.
- Nothing in the section puts a change in force. That is right.

Smallest change, one sentence each: the open case goes to the owner as a finding about the route
in force and nothing is proposed on this run; the second run gets its own plan and review, and "no
more lost than gained" is not its pass mark; after the second rehearsal only faults of the runner
are repaired, and any change to the stage, its order or its constants is said in the plan.

##### 8. Two statements about the order that the repository does not bear out. SHOULD FIX BEFORE FREEZING (checks 6, 8)

Where: plan 56-57, 61-62.

- **"It was written before any list of the second set was read."** The file to be frozen was
  written at 15:17 and committed at 15:20 (`1017ea7`), replacing an order learned on other lists
  (`2afb6dd`, 12:51). The second set's proposer-built lists are dated 14:32 and come from an
  earlier run (45605633369) than the first-set lists it was learned from (45606433485).
  `scripts/study_learned_order.py` reads the second set's lists before it writes the order. What
  does hold: no list of the second set is in its fit. Refitted from the first set's `split-80`
  lists it comes back identical.
- **"The better of two looks."** The record scores three orders on those lists: 37, 37 and 42. The
  order first committed gives 41.
- The choice is no threat to the test: the laws are new and the order is fixed before they are
  drawn. It inflates only the quoted 42.

Smallest change: "No list of the second set entered its fit; it replaced an earlier order after
second-set lists existed" and "the best of three orders scored on these lists (37, 37, 42)".

##### 9. A test draws the run's first eight laws; the runner's central function has no test. SHOULD FIX BEFORE FREEZING (checks 4, 10)

Where: `tests/test_ranked_memory_blind.py` 121-135 (line 123:
`build_laws(count=8, seed=run.LAW_SEED)`); runner 176-261.

- The smoke job and every local run of that file draw laws 5000 to 5007 of the run in memory, and
  a failing assertion would print one. I did not run that test.
- `arms_of` and `run_unit` are not called by any test.

Results at `ca8e581`: `test_ranked_stage.py` 8 passed, `test_ranked_memory.py` 5,
`test_learned_order.py` 6, `test_ranked_memory_blind.py` 4 passed and 1 not run by me. At
`394b0bf`: 6, 5, 6, 4.

Also untested: `summarise` on missing or doubled rows, timeouts, the displaced-the-law count, and
run-for-run by noise level; `WithRanked` with a stage of eight or nine, with a stage whose best
fits exactly, and the no-stage case through the runner.

Smallest change: draw the eight with another seed; add a test of `arms_of` with a stand-in stage
(same list in both rows, the ranked arm's kept candidates are the stage's first, a crash's
accounting, an empty stage).

##### 10. Minor

- **No stage at all** (`ranked_stage.py` 154, runner 197, `extended_route.py` 70). If the stage in
  force hands over nothing, the ranked arm is the ranked proposer alone: up to twelve candidates,
  no gate, and no `added` in the record. Executed: 2 candidates against 12. Plan: "at most three".
- **`WithRanked.failures`** is read before the stage runs (`ranked_stage.py` 114), so it holds the
  previous call's failures.
- **"Cannot be the law"** (plan 43-44) is wrong without noise: in `rehearsal_5` the added candidate
  was the law, and the gate would leave it out because the stage's best fits exactly. Harmless
  there.
- **"The smoke job's two laws are in no set"** (plan 188). Both structures are among the 320
  curriculum teaching problems. They cannot be laws of the run.
- **"Up to 1,500 fits"** (plan 162). Seen: 1,391 to 2,100 growth fits, and 88 to 400 more for
  remembered whole laws.
- **"Second choices of three of its four searches"** (plan 49). Right, and they are all the second
  choices there are: the deeper pass hands over one law. "A larger cost" than the eighth to tenth
  is not borne out by PREREG-MEMORY's rows: 7 or 8 reports came from the fifth to seventh, 7 from
  the eighth to tenth.
- **No trap arm** (plan 164-167). Honest in the letter: the rule speaks of a check, and the plan
  says the family is the same. The last sentence leans on a distinction without a difference,
  since the two scripts hold the same blocks, constants and joins. Say in the rule's own words
  that the run measures in-distribution performance.
- **The pass mark put to the owner** was "more laws reported". The plan's is "more right laws,
  none fewer at either noise level", plus a third expectation. Stricter; say so.
- **`GATE = 1.25`** is a bare number chosen on 32 development runs. Any gate between 1.02 and 5.2
  keeps the rehearsal's gain and drops both losses. "Its 1.25 before the ratios were computed"
  cannot be checked.
- **The runner's docstring** (line 94) says "every set an earlier development or test drew from
  this family". Not so: the small-table tables on which the 1,500 and 400 fits were chosen, and
  the vouching and counterfeit laws, are not left out. The plan's own list is exact.
- **Structures are compared with input names**, so an excluded law with its inputs renamed is not
  left out. Not quantified.
- **The two noise levels of a law share its input rows**; the sign test and the bound of
  expectation 1 count 200 independent runs.
- **The collect job prints no checksum** of what it prints; PREREG-MEMORY's did.
- **Incidental, an earlier record.** RESULT-MEMORY-2026-11 says "the memory's laws were the law
  itself in 26 of the 80 runs". Its runner counted every search-stage candidate (`handed_over` is
  10 or 11 in every row; the memory proposes three). On the 72 units I could rebuild: recorded 25;
  the law among the memory's own three 15, of which 7 the searches also held. The new runner does
  not repeat this.

##### 11. Confirmed right

- **Laws.** 100, seed 20261125, the invented grammar. Each exclusion is implemented: 60
  fingerprints, 80 development laws, 40 laws of PREREG-MEMORY, 300 + 320 + 200 = 820 curriculum
  problems (885 distinct structures). By problem key these are the problems of the development
  records. There is no laws file and no digest in the tree or in any ref.
- **Rows and scoring.** PREREG-MEMORY's runner's: 1,000 uniform rows, 70/30, 500 clean scoring
  rows, noise 0 and 1 % of the target's RMS. Recovered, right, wrong and reported are coded as
  defined; a tentative law is not a report.
- **Route.** `vouched_route_v2.discover_law`, refined path, absolute noise, seed 0, as the
  product's reading calls it; 240 s; 1,800 s (first rehearsal: at most 622 s a unit).
- **Same candidates.** `Recording` keeps the stage's texts in order and `replayed` proposes them
  again behind the same guard; every random draw in the route is seeded. In the rehearsal the two
  rows of all 32 units carry the same ten. In the replay 58 of 58 units with nothing added gave the
  same outcome in both arms.
- **Expectations** as coded are as written: 1 on an arm's reports in all 200 runs, held for both,
  "not shown" a miss, an arm reporting nothing "not shown"; 2 `>=` at each level and `>` in total;
  3 `>=` in total.
- **The stage.** The cap of ten is the stage's own; the refined path takes eleven with the
  polynomial library's one, the path twelve with the first pass. Order
  `A1 M1 B1 M2 C1 M3 D1 A2 C2 D2`, executed with stand-in members. Displaced are the last ones; the
  memory displaced the eighth to tenth, this stage the fifth to seventh. The gate is coded as the
  plan describes it.
- **Blindness.** The proposer, its order and its gate see development rows only; the fifth set
  aside is of those rows; the truth is read only in scoring and in the two law checks after the
  route has run. `benchmarks/structure_memory.json` is byte for byte the file PREREG-MEMORY tested.
- **The rehearsal disclosure** that the stage was shaped by the first rehearsal is complete as far
  as the rows go.

#### Numbers recomputed

| What | Plan or record | Recomputed |
| :--- | :--- | :--- |
| Second set, solved candidate on the list | 44 of 200 | 44 |
| Among the first three as handed | 28 | 28 |
| In the learned order | 42 | 42 |
| Earlier, later, sign test | 19, 6, 0.015 | 19, 6, 0.0146 |
| The other order | 37 | 37 (a third in the record 37; the first committed 41) |
| Law by form first, this order and the other | all 4; 2 | 4 of 4; 2 of 4 |
| First set, moved down from first | 6 of 9 | 6 of 9 |
| Order: weights, problems | a dozen, 300 | 12; 300 problems, 4,507 candidates, 355 solved |
| `dev_run1`: on the list against first three | "far more often" | in force 66 against 39; taught 80: 102 against 53 |
| PREREG-MEMORY, memory arm | 57 of 80 reported, 23 recovered | 57, 23 |
| Searches handed over ten | every one of 160 runs | 10 in all 160 rows (80 searches) |
| Exclusions | 80, 40, 820 | 80, 40, 820 |
| Named limit | 0 in 29, 1 in 46, 2 in 61, 3 in 76 | the same; at 134 reports up to 6 wrong |
| Rehearsal: reports | 21 against 22, none wrong | 21, 22, 0, 0 |
| Rehearsal: gained, lost | 1, 2 | 1, 2, all at 1 % |
| Rehearsal: recovered | 13 against 12 | 13, 12 |
| Rehearsal: three added in every run | yes | 32 of 32 |
| Rehearsal: every added candidate beyond 1.25 | 23 of 32 | 23; 19 of 96 candidates within |
| Rehearsal: gained run's ratios | 1.01 to 1.02 | 1.013, 1.015, 1.014 |
| Rehearsal: lost runs | tenth displaced; fourth stayed | candidate 10; candidate 4 |
| "About a third of the runs" | | 9 of 32; replay 14 of 72; for this draw about 28 of 200 |
| "The law itself in about 15 of 200" | | rehearsal 3 of 32 within the gate; replay 1 of 72 |
| "About four chances in ten" | | 0.71 at the middle of the plan's own ranges (0.28 to 0.97 at the corners) |
| "Up to 1,500 fits" | | 1,391 to 2,100, and 88 to 400 |

Mine, not in the plan:

- Laws by inputs over 16 other seeds: 2.2, 55 and 43; 97 % of one-input draws excluded.
- The gain by level: 14, 1, -1, 0, 0.
- Lists of 2 to 4 on all 200 three-input development problems.
- Within the gate: 0 of 53 runs without noise; 12 of 13 and 11 of 23 noisy runs with one and two
  inputs; 0 of 15 with three.
- Replay: 49 against 49 right, 23 against 23 recovered.
- Last three displaced and nothing added: 5 lost, none gained, of 72.
- The stage as first written, on 21 units before it changed: three added in 21, one lost, none
  gained.

#### What I could not check, and why

- **The second rehearsal**: not finished, no record in the repository.
- **Anything that needs PySR or Julia**: the searches, that four ran on GitHub's machine, the time
  a unit takes there. The machine has neither.
- **That nobody has looked at the laws.** The generator and its seed are in the repository, and
  one test draws eight of them.
- **When the order and the gate's 1.25 were fixed in the author's session.** Only file dates, run
  numbers and commit times can be read.
- **GitHub's side**: I asked GitHub nothing.
- **One test**, not run, because it draws the run's laws.
- **The full regression suite** of AGENTS.md section 2: not run; I changed nothing.
- **The replay is not the run**: recorded candidates of PREREG-MEMORY, development laws, 72 of 80
  units (8 could not be rebuilt), this machine with Python 3.13 where the workflow has 3.12.

#### What I would want to see in the second rehearsal

1. 64 rows, each unit once per arm, none `error` or `timeout`; if any, which arm was running.
2. The same ten stage candidates in both rows of every unit, and in every unit with nothing added
   the two rows identical in action and law.
3. Candidates added per run with the gate, by inputs and noise, with `left_out_as_fitting_worse`
   and `best_miss_of_the_stage`; whether anything is added without noise or on three inputs.
4. Whether `rehearsal_10` and `rehearsal_13` at 1 % are no longer lost and `rehearsal_8` at 1 % is
   still gained. Read it as a development reading: the searches are run afresh and differ.
5. Four searches and the core count, which needs finding 6 acted on first.
6. Time per unit against 1,800 s.
7. Every count by number of inputs, and no reading of it as a forecast: seven of the sixteen laws
   have one input, each with a structure the order's curriculum sets hold; the run will have about
   two in a hundred and none of that kind.
8. That nothing in the stage changes after it is read, or that the plan says what did.

If the runner is changed for findings 3 to 6, the rehearsal has to be run once more with the
runner that will be frozen.

# PREREG-RANKED-MEMORY-2026-11: does the memory's list in a learned order, beside the memory in force, make the table route report more right laws?

**Status:** DRAFT, not frozen, and **held back from freezing by its independent review** of 25
November (`docs/REVIEW_PREREG_RANKED_MEMORY_2026-11-25.md`). The review's first finding is the
owner's to decide: with the laws this draw will give (about 2 of 100 with one input, where all of
the order's measured gain lies) and the stage as it now stands (which adds nothing without noise
and nothing on three inputs), the two arms would be given the same list of candidates in about
170 of 200 runs, so the run could hardly show a gain or the absence of one. Nothing below is
frozen, no law is drawn, and the review's other findings are not all acted on yet. The text below
is the draft as reviewed, with the statements the review found wrong corrected in place and marked.

If it is frozen, it is frozen by `preregistration/DIGEST-RANKED-MEMORY.json` after the review's
findings are answered and after its laws are drawn. It then runs once, on GitHub Actions
(`.github/workflows/ranked-memory.yml`), started by the first branch named `ranked-memory-run-*`
whose shards start.

Asked for by the owner on 25 November 2026 (`docs/OWNER_DECISIONS_2026-11-25_AFTERNOON.md`, 1).

## What is tested

The memory of confirmed laws (`proposers/structure_memory.py`, in force for tables since 22
November) grows candidate laws for a table from remembered parts and hands the path its first
three, best fit first. Measured on the system's own curriculum
(`benchmarks/order_score/dev_run1.json`), a candidate the judge counts solved is on the memory's
list far more often than among its first three: over-fitted laws stand ahead of it.

`proposers/learned_order.py` learns an order of the list from the system's own problems, and
`proposers/ranked_memory.py` hands over the candidates that order puts first, reading each
candidate's fit on a fifth of the development rows it sets aside. On 200 curriculum problems that
share no structure with those the order was learned on, with the list of the memory in force built
that way, a candidate the judge counts solved was on the list for 44 problems and among the first
three for 28 as handed and for 42 in the learned order (19 lists earlier, 6 later, sign test
0.015; `benchmarks/order_score/dev_learned_order_in_force.json`).

That is a statement about a list. It does not say the route reports more laws: the searches find
many of these laws anyway, the path may not tell an added candidate from its rivals, and a
candidate added is a candidate displaced (below). This run measures what the route reports.

## The stage under test

`discovery_l2/ranked_stage.py`. The stage in force hands the path its candidates as now. The
ranked proposer builds the list of the memory in force (`benchmarks/structure_memory.json`) from
the same development rows, orders it with the order in
`benchmarks/order_score/learned_order_model.json`, and its first candidates **whose shape the
stage did not hand over** are added after the stage's: at most three, read from the first twelve
of its order.

**Only a candidate that fits about as well as the stage's best is added.** A candidate that misses
the development rows by more than 1.25 times what the best candidate of the stage misses them by is
left out (`ranked_stage.GATE`; root mean square on the development rows the proposers were given,
no pool row). This rule was added after the dress rehearsal and because of it (see "Disclosed
before freezing"): a candidate that fits far worse than what the searches already hold can take a
place and, on noisy rows, cannot be the law. *(The review's reading of the gate: without noise it
has let nothing in, since the stage's best then fits almost exactly; on three-input tables the
memory's list holds at most four candidates and none has passed it; an added candidate was within
it in about half the noisy runs on one and two inputs.)*

**A candidate added is a candidate displaced.** The path confirms at most ten candidates of a
stage, and the searches alone handed over ten in every one of the 160 runs of
PREREG-MEMORY-2026-11. Where the stage's list is full, the added candidates take the place of its
last ones. With the stage in force those are the second choices of three of its four searches; each
search's first choice and the memory's own three stay. When the memory was adopted its three
displaced the searches' eighth to tenth candidates; here the fifth to seventh are displaced, which
is a larger cost, and this run counts it: for every run it records whether a displaced candidate
was the law.

**The order** is a dozen weights learned on the first set of curriculum problems (300 problems)
from the lists of a memory taught 80 curriculum laws, built as the ranked proposer builds them.
No list of the second set entered its fit. *(Corrected after the review: the draft said it "was
written before any list of the second set was read". It was not: the file replaced an earlier
order at a time when second-set lists already existed, and the study script reads them before it
writes the order.)* **It was chosen on development data, and that is said here:** the record
scores three orders on the second set's lists of the memory in force, which brought the law among
the first three on 37, 37 and 42 problems; this is the one with 42, and it left the law by form in
first place on all 4 lists that held it, where the order learned on the in-force memory's own
lists left it first on 2. *(Corrected after the review: the draft said "the better of two
looks".)* The 42 is the best of three looks and flatters the order by that much. **Where that gain
lies, which the draft did not say:** by number of inputs the 28 to 42 is 14 to 28 on one-input
problems, 14 to 14 on two-input problems and none to none on three. The curriculum
(`recursion/curriculum.py`) and the grammar the laws of this run are drawn from
(`scripts/build_invented_laws.py`) are two generators over the same family of building blocks.

## Data and protocol

**Laws.** `benchmarks/ranked_memory_blind_laws.json`: 100 laws drawn by
`scripts/run_ranked_memory_blind.py laws` from the invented-law grammar (one to three inputs on
(1, 5), building blocks joined by sums, differences, products and quotients, constants from a list
of unround numbers), with seed 2026112699, **after** the ranked proposer, its stage and the order
file were written. *(The seed was 20261125 until the review found that a test drew the first eight
laws of that seed in memory on every run of its file; none was printed or stored, the test now
draws other laws, and the run's seed was changed so that no law of the run has been drawn by
anything.)* With the exclusions below, draws with other seeds give about 2 laws of one input, 55
of two and 43 of three in a hundred: nearly every one-input structure of this grammar has been met
by an earlier set. A draw is left out if its structure is in the invented-laws benchmark (by
fingerprint; no held-back law is read), among the 80 laws the memory proposer was developed on,
among the 40 laws of PREREG-MEMORY-2026-11, or among the 820 curriculum problems the order was
learned and scored on.

**Rows, noise, split, scoring** are those of PREREG-MEMORY-2026-11, unchanged: 1,000 rows drawn
uniformly, 70 % for development and 30 % as the pool the path may measure, 500 further rows
without noise for scoring; noise 0 and 1 % of the target's root mean square. A law is *recovered*
if it is the law after refitting constants (`symbolic_recovery`); *right* if it is recovered or
its R² on the scoring rows is at least 0.999; *wrong* if it is reported and not right. A law the
route holds back as tentative is not a report.

**The route** is the one in force for tables with its last checks:
`discovery_l2/vouched_route_v2.discover_law`, the refined path, noise taken as absolute, as the
product's reading calls it.

**Two arms for every law and noise level, 400 runs in all, given the same candidates of the stage
in force.** The stage in force (four searches of 240 seconds side by side, with search processes
that cannot hang, and the memory beside them) runs once, and

- `in force` hands the path its candidates: what the route does now;
- `ranked` hands it the same candidates through `ranked_stage.with_ranked`.

The 240 seconds are those of the run that put the memory in force. The product runs its searches
only when a budget is given and its dashboard asks for 30 seconds; nothing is claimed here for
another budget.

A run that crashes or exceeds 1,800 seconds counts as reporting nothing, in both arms.

## Expectations, fixed now

1. **What either arm reports can be relied on.** For each arm, the share of wrong laws among
   those it reports in its 200 runs is *held* by the named limit of 25 November
   (`discovery_l2/report_share.py`, `specifications/discovery_checks_v1.json`): at most 5 %, with
   the one-sided 95 % upper bound at most 10 %. *Not shown* is a miss.
2. **The ranked proposer helps.** The `ranked` arm reports at least as many right laws as
   `in force` at each noise level, and more in total.
3. **It does not trade the law itself away.** The `ranked` arm recovers by form at least as many
   laws as `in force` in total.
4. Reported without a criterion: laws gained and lost run for run, with the exact sign test; in
   how many runs the ranked proposer added a candidate, how many candidates it added and how many
   it left out as fitting worse, in how many runs it added the law itself and displaced the law
   itself; laws reported that are accurate without being the law; the path's actions; what the
   checks held back; errors, timeouts and time.

*(The review found the forecast below unsupported: by the development rates and the make-up of the
draw the added candidates would hold a solved candidate the first three lack in 3 to 4 runs of
200, not 15, and a replay of the present stage on 72 recorded units of PREREG-MEMORY changed no
outcome. It is left as written, as the record of what the author expected before the review.)*

**The author's own expectations**, recorded so that they can be wrong, and no criterion. The arm
in force reports about 135 laws in 200 runs, nearly all of them right and about 55 of them the law
itself (PREREG-MEMORY-2026-11's memory arm reported 57 of 80 and recovered 23, before the last
checks). With the gate the ranked proposer adds a candidate in about a third of the runs and adds
the law itself in about 15 of 200; the searches or the memory had handed over most of those
already in another form, so the route gains 3 to 6 right laws. It loses 1 to 4: through
displacement, or because one more accurate rival makes the path or the last checks hold back a law
they would have reported (the first rehearsal showed both). Expectation 2 holds with about four
chances in ten: a gain of this size is within what chance moves in either direction. Expectation 1
holds for both arms. Expectation 3 holds with about two chances in three.

A miss is reported as a miss.

## What follows from the result, fixed now

Nothing is put in force by this run. A change to the stage for tables is the owner's decision
(AGENTS.md 1.5).

- If 1, 2 and 3 hold, the agent proposes a **second** run before any proposal for force: the same
  two arms on the physics benchmark's problems, where the memory's parts do not fit most laws and
  the displacement can only cost. The ranked stage is proposed for tables only if it loses no more
  right laws there than it gains.
- If 1 misses for the `ranked` arm, it is not proposed, and the wrong laws are listed with what
  the ranked proposer had added in each.
- If 2 misses, the result is recorded as: an order that brings the law forward on the memory's
  list does not make the route report more right laws. It is not proposed, and no variant of it is
  called measured until it has been run on laws drawn after that variant was written.
- If 2 holds and 3 misses, it is not proposed as it is: the route would be reporting accurate
  laws in place of the laws themselves.

## Disclosed before freezing

- **A candidate added is a candidate displaced** (above). Told to the owner as "adds three more";
  that was loose, and this plan is the correction.
- **The order was learned on problems of the same family.** A gain here says an order learned on
  the system's own curriculum carries to laws of a sibling generator. It says nothing for laws
  with several variables inside one function, which the memory's one-variable parts cannot hold
  and which most of the physics benchmark's laws are.
- **The order is taught the judge's rule, not the form.** It was taught which candidates are
  accurate and no longer than the law. On the first set, scored on problems it was not trained
  on, an order learned this way moved the law by form down from first place on 6 of the 9 lists
  of the memory in force that held it. On the second set the order used here left it first on all
  4. Here the memory's own three are untouched; the third expectation is there because of this.
- **The memory in force holds laws of this family,** reported by the route, right or not, and the
  way it joins parts was written knowing the grammar (PREREG-MEMORY-2026-11).
- **Unequal cost.** The ranked arm builds one more list (up to 1,500 fits) and runs the path
  again. Its searches are the same ones.
- **No trap arm.** AGENTS.md 1.6 asks of a blind test of a *check* that some traps come from a
  generator the check was not developed against. This is a test of a proposer; the checks are
  those in force, unchanged. The laws come from a generator other than the one the order was
  learned on, and from the family the memory was tested on.
- **The stage was changed after a dress rehearsal, on development laws.** The first rehearsal
  (run 45658660944, 25 November: sixteen laws the memory proposer was developed on, 32 runs an
  arm; none is a law of this run) ran the runner end to end with the stage as first written, which
  added three candidates in every run. It reported 21 laws against 22 for the arm in force, none
  wrong in either; run for run one report gained (the law itself, which the ranked proposer had
  added) and two lost, both accurate laws that were not the law; 13 laws recovered against 12. In
  one of the two lost runs the reported candidate was the stage's tenth and was displaced; in the
  other it stood fourth and stayed, and the path reported nothing all the same. Read from the
  rehearsal's own rows afterwards: in 23 of the 32 runs every candidate added missed the
  development rows by more than 1.25 times what the stage's best missed them by, often by
  thousands of times, and both lost runs are among them; in the run gained the added candidates
  missed them by 1.01 to 1.02 times. The gate above was written on that reading, its 1.25 before
  the ratios were computed. **So the stage of this run was shaped by 32 development runs,** and
  nothing here says the gate helps on new laws: that is what the run measures. A second rehearsal
  with the gate (`ranked-memory-rehearsal-2`) is run before freezing to see the runner end to end
  again; its counts go into the record of the review.
- **No discovery was run on any of the 100 laws,** by any engine, before freezing. What was run on
  them: the generator and its acceptance rule.
- **Two noise levels, uniform rows, up to three inputs.** The conditions of the earlier runs, and
  nothing beyond them.
- **The smoke job's two laws** are in no set.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.

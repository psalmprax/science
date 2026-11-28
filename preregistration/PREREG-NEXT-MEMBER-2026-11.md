# PREREG-NEXT-MEMBER-2026-11: when the shortest law of a class fails the residual check, does trying the class's other laws add right laws without adding wrong ones?

**Status:** DRAFT, not frozen, not reviewed. Written on 28 November 2026 on the owner's word
("try all of these", decision 1). AGENTS.md 1.6 requires a reviewer who took no part in the work
before a plan is frozen; one could not be started in the session that wrote this, so nothing
below is frozen, no law is drawn and nothing has run. The step is a module that nothing calls.

If it is frozen, it is frozen by `preregistration/DIGEST-NEXT-MEMBER.json` after a review's
findings are answered, and it then runs once on GitHub Actions.

## What is tested

The route in force for tables (`discovery_l2/vouched_route_v2.py`) hands its candidates to the
contrast confirmation. When several of them cannot be told apart on the rows read (the verdict
`equivalent_class`), the main search takes the shortest and checks its residuals. If that check
fails, the reading ends with "widen the search" and the class's other laws are not looked at.

`discovery_l2/next_member_route.py` takes the record of such a reading and tries the class's
other laws, shortest first and six at most. Each is treated as the route treats the member it
takes: fitted with one coefficient for each additive term, its residuals checked
(`main_search.adequacy`, same neighbour count), and written in the fitted form with unnoticed
terms dropped. The first that passes goes to the last checks in force (`vouching_v2.vouch`) with
every other member of the class as its rivals. If they vouch for it, it is reported; if not, it
is held back as tentative. Either way nothing further is tried. A
reading that ended any other way is returned as it was, so the step can add a report and can
never take one away.

## What development showed (not evidence)

`scripts/study_next_member.py`, record `benchmarks/family_stream/dev_next_member.json`. The 122
tables the second family run left without a law were read with no search and a list handed in.

| the list handed in | route reported | of those the law itself | step applied | step added | of those the law itself | accurate | not accurate | found, not vouched |
|---|---|---|---|---|---|---|---|---|
| nine ordinary laws, then the true form | 60 | 56 | 13 | 7 | 4 | 7 | 0 | 6 |
| the true form alone | 64 | 54 | 2 | 2 | 2 | 2 | 0 | 0 |
| ten ordinary laws, no true form | 10 | 1 | 24 | 8 | 0 | 8 | 0 | 10 |

Three things these numbers do not show.

1. **No wrong law among 17 added does not show a share.** By the project's own check
   (`report_share.verdict`), 0 wrong of 17 has an upper bound of 16 % at 95 % confidence, and the
   bound allowed is 10 %. Twenty-nine reports with none wrong are needed.
2. **On lists with no true form, every law the step added was accurate and none was the law.**
   One was a sum of sines of logarithms standing in for `x*exp(-x)`. The route in force made ten
   reports of the same kind on those lists. "Accurate" is the line that defines a wrong law
   (`DISCOVERY_ACCURACY_R2`, R squared 0.999 on rows without noise drawn from the range of the
   fit), and these reports are on the right side of it. Whether a report of that kind is wanted
   is a question about that line, and the step makes more of them.
3. **The lists were handed in; no search ran.** Behind the product's search stage the step's
   rate is not known. The second run's record keeps the action "widen the search" (72 of its 240
   readings, 17 with the law itself among the referee's candidates and 32 more with an accurate
   one) and not the verdict that came before it.

## The run, as drafted

**Stage.** The product's search stage (`ui/route_endpoint._proposer`), 30-second searches as in
the family runs, on GitHub's runners. It needs PySR.

**One reading for both arms.** The step works on the route's finished record and the same rows,
so each table is read once by the route in force (arm A), and arm B is that record with the step
applied. Where the step does not apply the two arms are the same reading.

**Tables.**

* T1, the development generator: families drawn by `study_family_stream.draw_families` with a
  seed never used, five members, noise 1 %, 5 % and 10 %. The number of families is set after
  the rehearsal below, so that the step can be expected to add 29 laws.
* T2, a generator the step was not developed against (AGENTS.md 1.6). Two candidates, the
  reviewer's or the owner's to choose: a red-team search aimed at the frozen step for tables on
  which it reports an inaccurate law; or problems from a published benchmark the step has never
  been run on. The 40 held-back invented laws are not proposed: they are scarce and this step
  does not justify spending them.

**Counted.** For arm A: laws reported, accurate, not accurate. For the step: readings where it
applied, members tried, laws added, and of those the law itself, accurate, not accurate, and
accurate on rows drawn half as wide again as the fit's range. T1 and T2 are counted apart and
together.

## What each outcome would propose

Fixed before the run; the owner decides what is in force whatever the outcome (AGENTS.md 1.5).

* **Proposed for adoption** if `report_share.verdict(not accurate among the added, added)` is
  "held" on T1 and T2 together (a share of at most 5 % with an upper bound of at most 10 %,
  `REPORTED_WRONG_SHARE_MAX` and `REPORTED_WRONG_SHARE_BOUND_MAX`), and no added law in T2 is
  not accurate.
* **Not adopted** if the share of added laws that are not accurate is above 5 %, or any added
  law in T2 is not accurate.
* **Not shown, kept as a module, not in force** in every other case, which includes the step
  adding fewer than 29 laws.

Reported without a consequence: how many added laws are the law itself; how many stay accurate
on the wider rows; the size of the classes; seconds.

## To settle before freezing

1. The independent review (denominators, limits and consequences above against what the runner
   computes).
2. A rehearsal on development families behind the product's stage, to count how often the step
   applies there. Without it the run cannot be sized.
3. T2's source.
4. The owner's reading of point 2 under "What development showed": whether an accurate law that
   is not the law counts for the step. If it does not, the criterion above needs the law itself
   in place of "accurate", and the lists with no true form already say the step would add
   reports that fail it.
5. The seed and the number of families of T1; the runner and its workflow, which are not
   written.

# The owner's word of 28 November 2026: "try all of these"

The report on the outside audit and the outside notes ended with four "decisions for you". At
01:56 Berlin time the owner answered: `try all of these`, quoting the four. This file records each
as it was put, how "try" was read, what was done, and what it does **not** cover. The reading was
sent to the owner before the work began and was not contradicted.

Decisions 1 and 2 put nothing in force: each is a module that nothing calls, a development
measurement and a plan that is drafted and not frozen. Decisions 3 and 4 do change something in
force, on that word, and each names what undoes it.

No reviewer independent of the author has read any of this. One could not be started in the
session that did the work (AGENTS.md 1.6), which is the first reason neither plan is frozen.

## 1. The next law of a class when the shortest fails

*As put:* "Try the next law when the shortest of an indistinguishable group fails. I recommend
it. `main_search.py` is frozen, so it means a new module and its own test."

*Read as:* build it in front of the frozen search, measure it in development on the same 122
tables and on lists that hold no true law, and write its test plan. Not in force.

*Done:* `apedp_dyn/discovery_l2/next_member_route.py`; `tests/test_next_member_route.py`;
`scripts/study_next_member.py` with `benchmarks/family_stream/dev_next_member.json`;
`preregistration/PREREG-NEXT-MEMBER-2026-11.md` (draft).

*Measured* (122 tables, lists handed in, no search):

| the list handed in | step applied | laws it added | the law itself | not accurate |
|---|---|---|---|---|
| nine ordinary laws, then the true form | 13 | 7 | 4 | 0 |
| the true form alone | 2 | 2 | 2 | 0 |
| ten ordinary laws, no true form | 24 | 8 | 0 | 0 |

It does what it was built for: of the 13 readings that were lost at the residual check with the
true form in the class, 7 are now reported. It also reports on lists that hold no true form: 8
laws, each accurate on the test rows and none the law. The route in force made 10 reports of
that kind on the same lists.

*Does not cover:* putting the step in force, or freezing the plan. Seventeen added laws with
none wrong does not show a share within the limit (the bound is 16 %, and 10 % is allowed; 29
are needed). Behind the product's search stage the step's rate is not known.

## 2. The implicit form as a proposer

*As put:* "The implicit form as a proposer. I recommend a preregistered test, though fresh real
tables are scarce."

*Read as:* build the proposer, put it through the referee in development (algebraic laws, the
three real development tables, laws it cannot write), and write the plan. Not in force.

*Done:* `apedp_dyn/proposers/implicit_form.py`; `tests/test_implicit_form_proposer.py`;
`scripts/study_implicit_proposer.py` with
`benchmarks/outside_proposals/dev_implicit_proposer.json`;
`preregistration/PREREG-IMPLICIT-STAGE-2026-11.md` (draft).

*Measured,* with the proposer as the route's only search stage (the product's stage needs PySR,
which cannot be installed where this was run):

* Laws with a short relation, six tables each at 1 %, 5 % and 10 % noise. Where the first pass
  alone never had the law, the law itself was reported for Kepler with two masses on 4 tables,
  time dilation on 3 and Langmuir on 2. Over all nine laws: 28 reports on 54 tables, 24 the law
  itself, none inaccurate.
* The 122 tables the second family run left without a law: 16 reported, 2 the law itself, none
  inaccurate. Seven came from the proposer's formulas. Only 17 of the 122 have a law it can
  write.
* The traps: an exponential was reported as an algebraic law on 1 table of 6, accurate inside
  the range of the rows and wrong beyond it. A sine was never reported.
* The three real development tables: no reading changed.

Three faults of the earlier probe's ranking were found and repaired while building it; the
module's text and `docs/OUTSIDE_PROPOSALS_CHECKED_2026-11-28.md`, section 5, have them.

*Does not cover:* putting the proposer in force, or freezing the plan. It was measured as the
only stage; beside the product's stage its formulas displace candidates, and that is not
measured. Fresh real tables do not exist: the eight held back are used up.

## 3. The referee's three rules

*As put:* "The referee's rule fixes. Zero turnover, the annualisation factor and the Lean reading
are bugs; I'd fix all three on your word."

*Done, and in force* (audit M-45):

* **Zero turnover.** Stage 5 refused a strategy that never trades, because its break-even fee
  is infinite and a finite one was demanded. Where the stage computes the turnover itself and
  finds none with positive returns, the fee test is now moot. This is the one change that lets
  something pass that did not: a position held through the whole window. A reported infinite
  fee is refused as before.
* **Annualisation.** A trial record may now state the units of its Sharpe ratio, and the
  deflated ratio converts a record that states other units than the observed one. No verdict
  changes today, because no caller states units yet; the result now says how many records
  stated none.
* **The Lean reading.** Stage 1 looked for the word `sorry` anywhere in the text. It now reads
  the source outside comments and strings and also objects to `admit`, `sorryAx`,
  `native_decide` and an empty source. Stricter where it was blind, and no longer refusing a
  finished proof for a comment.

*To undo:* revert the commit; the three are separate changes in `apedp_referee/`.

*Does not cover:* the floor on the trials' variance, which is one number whatever the units
(making it mean the same in both would tighten one case or loosen the other); signed receipts
from the build that ran the Lean kernel; and callers stating their units, the frozen market
sandbox among them.

## 4. Which branch builds and deploys the image

*As put:* "Which branch builds and deploys the image, and whether to deploy at all."

This was a question, and "try" was read narrowly. *Done:* the `build` job runs on `master`, the
branch that receives merged work (it said `main`, a branch this repository has never had, so
the job never ran). `deploy` runs there only when the repository variable `DEPLOY_TO_FLY` is
`true`. It is not set, so nothing is deployed.

*To turn deploying on:* set that variable. *To undo the branch:* one line in
`.github/workflows/test.yml`.

*Does not cover:* whether to deploy at all, which the switch leaves with the owner. `build`
still needs the job that checks the trustees' signature, so it has not run yet either. Whether
a Fly.io token exists could not be checked from the session.

## One thing common to decisions 1 and 2

Both add reports that are accurate by the line that defines a wrong law
(`DISCOVERY_ACCURACY_R2`: R squared of 0.999 on rows without noise from the range of the fit)
and are not the law. Neither added a report on the wrong side of that line in development. Whether
such a report is wanted is a question about the line and not about either step, and both plans
leave it to the owner. Measured on rows drawn half as wide again as the fit's range, 14 of the
proposer's 16 reports on the stream stayed accurate and 23 of its 28 on the laws (four of the
five that did not are time dilation next to its singularity).

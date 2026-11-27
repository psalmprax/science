# Outside proposals for the search, checked (2026-11-28)

**Development, not a result.** On 27 November an outside assistant sent four notes on the tabular search: a
diagnosis, a proposal for a new engine, confidence levels with "kill tests", and two lists of deeper ideas. The
owner asked for them to be checked and tested. This is what was checked against the records, what was run, and
what it showed. Nothing here is in force. The audit of the repository sent the same evening is answered in
`docs/SYSTEM_FLAWS_AUDIT.md` (M-39 to M-44).

Everything run is in two scripts: `scripts/study_handed_truth.py` (record
`benchmarks/family_stream/dev_handed_truth.json`) and `scripts/probe_outside_proposals.py` (record
`benchmarks/outside_proposals/dev_probes.json`). Neither was reviewed by anyone but its author.

## 1. "91 of 122 tables already had an accurate law": what that count means

The notes read this count from the second family run as a selection fault: the search finds the law and the
referee drops it, so selection should be repaired before any new generator is built. The count is right. The
reading was tested and is mostly wrong.

**The test.** The 122 tables that run left without a law were read again by the route in force, with no search,
and with the true form handed in: once alone, once last of ten ordinary laws.

| | Handed alone | Last of ten |
|---|---|---|
| Reported, of 122 | 64 (the law itself 54) | 60 (the law itself 56) |
| at 1 % noise, of 27 | 25 | 24 |
| at 5 % noise, of 33 | 17 | 18 |
| at 10 % noise, of 62 | 22 | 18 |
| The last checks did not vouch | 52 | 48 |
| The law taken from the contrast failed the residual check | 2 | 13 |
| No survivor on the rows read | 4 | 1 |
| The true form eliminated by a rival | 0 | 0 |

No reported law was inaccurate.

**What it says.**

1. **At 1 % noise the tables lacked the law, not a willing referee.** Handed the true form, the route reports 25
   of 27. The accurate candidates it had in the run were approximations, and it declined them. The search is
   what was missing there.
2. **At 5 % and 10 % the last checks withhold even the true form** (52 readings, none at 1 %). In 38 the reason is
   the pinned power rule: an error large enough to make the law wrong would have been seen less than 80 % of the
   time with these rows, often 74 to 79 %. In 9 several laws pass and disagree; in 3 the constants cannot be
   fitted to the accuracy claimed. This is a limit of information at that noise and number of rows. More rows,
   or rows measured where the rivals differ, are what moves it. A different search does not. Ten percent is
   also outside the noise the checks were measured for.
3. **One real fault of selection.** When several laws cannot be told apart the route takes the shortest. If that
   one fails the residual check the reading ends, and the other members are not tried. With the true form last
   of ten this ended 13 readings, and in all 13 the true form was in the class. It is the only loss at 1 %
   noise in that reading (3 of 3). `main_search.py` does this; it is frozen and in force, so a change needs the
   owner's word and its own test.
4. **Contrast confirmation never eliminated the true form** in 244 readings.

Of the run's 74 tables whose first action was to widen the search, 52 are reported once the true form is handed
in. Of its 46 whose law was held back by the last checks, 12 are.

The register's sentence "Candidates are not what those tables lack" was mine and is corrected there.

## 2. The kill tests

Each idea came with a test in the proposer's words. Counts are draws in which the thing was got right.

### The law as a relation P(x, y) = 0 (the fourth note's first idea)

Generated rows of six algebraic laws at the size of the real table of that form; 30 draws; the law's own form
ranked first:

| Law | Rows | No noise | 1 % | 5 % |
|---|---|---|---|---|
| Time dilation | 7 | 30 | 20 | 9 |
| Langmuir | 17 | 30 | 25 | 24 |
| Kepler with two masses (three inputs) | 12 | 30 | 29 | 27 |
| Lens | 20 | 30 | 22 | 22 |
| Relativistic energy | 20 | 30 | 18 | 21 |
| Hubble (a straight line) | 32 | 30 | 0 | 0 |

At 200 rows every law but the line is right in 7 of 7 draws at 1 % and 5 % (Kepler 5 of 7 at 5 %).

Three things qualify this. **The line:** with any noise the ranking prefers three or four terms to the two-term
line, in every draw. Choosing between nested relations is its weak point. **The controls:** given an exponential
it returns a four-term relation that is accurate inside the measured range in 30 of 30 draws at 1 % and outside
it in none; for a logarithm 30 and 10. For a law that is not algebraic it makes a counterfeit that only fails
outside the range. **The real tables open for development:** Kepler's form is first of 145 on the solar system's
8 rows. Langmuir's is 65th of 353 on his 10 rows (4th of 112 among three-term relations), and Leavitt's line is
145th of 353. One of three.

The held-back real tables were not read. The note's claim that all five collapse to a linear problem is tested
on generated rows of their forms only.

**Reading.** As a proposer in front of the referee it has a case: on generated rows of time dilation's form,
which the route missed on the real table, it puts the law's own form first in 20 of 30 draws from 7 rows at 1 %
noise. As a discovery engine by itself it does not: it over-selects terms and counterfeits what is not algebraic. The idea is established work
(implicit-SINDy, SINDy-PI, <https://arxiv.org/abs/2004.02322>), and the published record says its early forms
were very sensitive to noise.

### Compositions from the rank of the gradient (the second note's first engine)

Ten laws of four and five inputs, 1,000 rows, a Gaussian process for the gradient. Compositions told from every
other subset of two or three inputs: 10 of 10 laws without noise, 10 of 10 at 1 %, 9 of 10 at 5 %. The
gradient's error at 5 % noise was about 3 %.

**Reading.** The proposer put this at 30 to 45 % and called it the highest risk; I expected it to fail. It did
not. But it tests only the detection of structure. This project's own split search found the structure of wide
tables and then had an accurate law for 38 of the 80 tables with four and five inputs and the law itself for
none of its 120, because the search of each part was the weak step. This idea does not touch that step. It would extend what `table_shape.py` reads from
sums and products to nested laws. The method is AI Feynman 2.0's (<https://arxiv.org/abs/2006.10782>).

### The weak form for a function of one input (the second note's second engine)

Fifteen functions, 20 draws each; an equation the function truly satisfies ranked first, of 300:

| Rows | Positions | No noise | 1 % | 5 % |
|---|---|---|---|---|
| 50 | evenly spaced | 240 | 174 | 80 |
| 500 | evenly spaced | 300 | 209 | 160 |
| 50 | random, as in a table | 33 | 37 | 26 |
| 500 | random | 240 | 205 | 137 |

The counts flatter it: a sine satisfies 24 of the 120 candidate equations and an exponential 66, so a blind pick
is often right for those. For the functions with few true equations, at 500 random rows and 1 % noise: pole 37
of 60, root 48 of 60, logarithm 0 of 60. On 50 rows at random positions it fails without any noise.

**Reading.** By the proposer's own rule this one is dropped for tables. It needs rows dense and ordered in one
input, which a table of several inputs does not have. The technique is weak SINDy
(<https://arxiv.org/abs/2005.04339>), made for trajectories.

### The weak form for a Lagrangian (the fifth note's first idea)

Three conservative systems, a noisy trajectory. Given the right terms, the constant from the weak form is two to
four times closer than from the second derivative for the pendulum and the double well (pendulum, 200 rows, 5 %:
1.0 % against 2.7 %), and worse for the soft spring. The terms themselves were ranked first only without noise.

Two corrections to the note. Its formula has the wrong sign: integration by parts gives dL/dq * psi + dL/dqdot *
dpsi/dt, and with the printed minus the potential comes out negated. And "no derivative of noisy data" is not
so: the first derivative is needed, and enters squared.

**Reading.** Not concluded. My version is sixty lines; published work reports more (xL-SINDy,
<https://arxiv.org/abs/2209.03248>). It concerns trajectories, so the ODE route, and no table.

### Not tested

The Rust engine with semantic deduplication (it has to be built first), branch and bound, and the set
transformer (it needs weeks of GPU training). For the last, one published comparison on SRBench places a
pre-trained transformer second in recovery, behind PySR (<https://arxiv.org/abs/2509.19849>), and the original
work claims to approach genetic programming, not to pass it (<https://arxiv.org/abs/2204.10532>). The note's 75
to 85 % for beating PySR has no measurement behind it.

## 3. What the notes got wrong about this repository

- The search stage is three PySR searches and one deeper exhaustive pass, not four PySR workers.
- The route's own constant fit is `scipy.optimize.least_squares`; BFGS is inside PySR.
- "Fix selection first" is the wrong order at 1 % noise (section 1).
- The five real tables named for the implicit test are held back; development code does not read them.

## 4. What follows, for the owner

1. **Try the next member of a class when the shortest fails the residual check.** The one fault of selection
   found; it needs a new module in front of the frozen search and a preregistered test.
2. **The implicit form as a proposer**, handing the referee its best relations of two, three and four terms. It
   would be tested on fresh real tables, which are few.
3. **For 5 % noise and above, more rows or chosen rows.** The checks say what they lack: power. That is a design
   question about measurement, and the power rule itself is the trustees'.
4. Nothing else in the four notes is recommended for building now.

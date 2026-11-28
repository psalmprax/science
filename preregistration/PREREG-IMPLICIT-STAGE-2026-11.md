# PREREG-IMPLICIT-STAGE-2026-11: do formulas solved from polynomial relations, added to the product's search stage, make the table route report more right laws without more wrong ones?

**Status:** DRAFT, not frozen, not reviewed. Written on 28 November 2026 on the owner's word
("try all of these", decision 2). AGENTS.md 1.6 requires a reviewer who took no part in the work
before a plan is frozen; one could not be started in the session that wrote this, so nothing
below is frozen, no table is chosen and nothing has run. The proposer is a module that nothing
calls but its study and its tests.

If it is frozen, it is frozen by `preregistration/DIGEST-IMPLICIT-STAGE.json` after a review's
findings are answered, and it then runs once on GitHub Actions.

## What is tested

`proposers/implicit_form.py`. Every other search stage here looks for the target as a formula in
the inputs. This one fits small sets of monomials in the inputs and the target as a relation
`P(x, y) = 0` (two to four terms, total degree at most three, the target to the second power at
most), and hands the route the best of them solved for the target: a ratio of polynomials, or
the quadratic formula's nearer branch. The best relation of two terms, of three and of four come
first. It decides nothing.

## What development showed (not evidence)

`scripts/study_implicit_proposer.py`, record
`benchmarks/outside_proposals/dev_implicit_proposer.json`. The proposer was the route's only
search stage, because the product's stage needs PySR and that cannot be installed where this was
measured. The same tables were read with no stage at all; the route then reports nothing, by its
own rule, so that column is what the first pass held back.

**Laws with a short relation, as tables of the family stream's size** (700 rows and a pool of
300; noise 1 %, 5 % and 10 %; two tables at each, six for each law):

| law | first pass alone held the law back | with the proposer: reported | of those the law itself | found by the proposer |
|---|---|---|---|---|
| Kepler with two masses | 0 | 4 | 4 | 4 |
| time dilation | 0 | 6 | 3 | 2 |
| saturation (Langmuir) | 0 | 2 | 2 | 2 |
| lens | 6 | 4 | 4 | 4 |
| straight line | 6 | 5 | 5 | 5 |
| energy, `sqrt(p**2 + m**2)` | 6 | 3 | 3 | 0 |
| trap: exponential | 0 | 1 | 0 | 1 |
| trap: logarithm | 6 | 3 | 3 | 0 |
| trap: sine | 0 | 0 | 0 | 0 |

For three laws the first pass never had the law and the proposer got it reported: Kepler with
two masses on 4 tables of 6, time dilation on 3, Langmuir on 2. At 1 % noise all 14 reports were
the law itself; at 10 % one of 4 was.

**The 122 tables the second family run left without a law.** Seventeen have a law within the
proposer's reach (on the law's own rows without noise its first relation is exact). With the
proposer as the stage 16 laws were reported: 2 the law itself, all 16 accurate by the line that
defines a wrong law, 14 accurate on rows half as wide again. Seven of the 16 came from the
proposer's formulas; nine were the first pass's own laws, reported because a stage had run.

**The three real development tables.** No reading changed. Kepler's law was first of the ten
formulas and was reported, as it is without the proposer. Langmuir's three terms were 33rd of
134 relations and not among the ten; the first formula was his form with a constant added, which
the benchmark's test of form passes. Leavitt's line was 69th of 139.

Four things these numbers do not show.

1. **It was the only stage.** The product's stage hands the route ten candidates of its own, and
   the route takes ten. Added to it, the proposer's formulas displace some of those, and a law
   they got reported here would there have to stand against stronger rivals. None of that is
   measured.
2. **It makes counterfeits, and one was reported.** Given an exponential it proposes ten
   algebraic approximations. On one table of six (10 % noise) the route reported one: accurate
   inside the range of the rows and wrong beyond it. On the stream's 105 tables outside its
   reach it got 4 approximations reported, 3 of them still accurate on the wider rows. By the
   accuracy line none of these is a wrong law.
3. **No wrong law among 44 reports is development.** The tables of the first arm were chosen by
   the builder of the proposer, and the 122 are tables the route had already failed on.
4. **Real tables.** One of three had its law proposed, and it was the one already read. The
   eight held-back real tables are used up and were not read.

## The run, as drafted

**Arms.** A: the route in force with the product's stage, 30-second searches. B: the same, with
the proposer's first three formulas (the best relation of two terms, of three and of four) in
the last three of the stage's ten places. Where the placement goes is open (see below).

**Tables.**

* T1, laws with a short relation and laws without, from a source the builder did not choose:
  the reviewer's or the owner's pick of a published collection of physical laws that the
  proposer has not been run on, each law marked beforehand as within or outside reach by the
  rule above.
* T2, real tables with a known law. None is available: the eight held back are used up. A new
  set would have to be gathered by someone other than the builder, with the owner's word. Without
  it the run says nothing about real measurements, and says so.

**Counted,** for each arm and for B against A table by table: laws reported, the law itself,
accurate, not accurate, accurate on rows half as wide again; tables gained and lost.

## What each outcome would propose

Fixed before the run; the owner decides what is in force whatever the outcome (AGENTS.md 1.5).

* **Proposed for adoption** if all hold: B reports the law itself on more tables than A, by a
  paired sign test at 0.05; B loses the law itself on at most one table in twenty of those where
  A reported it; `report_share.verdict` for B's reports is "held"; and on the tables outside the
  proposer's reach B reports no more laws that are not the law than A does.
* **Not adopted** if B's share of laws that are not accurate is above 5 %, or B reports more laws
  that are not the law on the tables outside its reach.
* **Not shown, kept as a module, not in force** otherwise.

## To settle before freezing

1. The independent review.
2. A rehearsal behind the product's stage on GitHub's runners: where the three formulas go, and
   how often they displace a candidate that would have been reported.
3. T1's source, and whether T2 can exist.
4. The owner's reading of point 2 above: a law that is accurate inside the range and is not the
   law is not a wrong law by `DISCOVERY_ACCURACY_R2`. The criteria above count it against the
   proposer on the tables outside its reach only. If the owner wants the wider rows to decide,
   that is a new named threshold, with a pin and a changelog entry.
5. The runner and its workflow, which are not written.

# An outside review of the search, checked (2026-11-28)

At 04:39 Berlin time the owner passed on two files from an outside assistant. The first is a cold
review of commit `aabbeea` (the audit's repairs) with a recommendation on each of the four
decisions. The second lists nine reasons the search in force is "average", a list of repairs, a
design for a new engine, and a sequence of kill tests.

This file says what was checked, what held, what was done the same morning, and what waits for the
owner. Nothing here changes the search that is in force. Reviewed by nobody but the author.

## 1. The cold review of the repairs

It confirmed that the parser repair closed the way to code, and found four faults the repair's own
sweep had missed: three ways to hold a worker with a formula, and a limit on requests in flight
that counted too late. All four were reproduced under a time limit and repaired; audit **M-46** has
the account.

One of its suggestions was not followed. It asks Stage 1 to refuse a Lean source that declares an
`axiom`. The bridge that writes this repository's Lean sources declares the real numbers that way
(`axiom R : Type` and what follows it in `epistemic/lean_universal_bridge.py`), so that rule would
refuse every proof the bridge writes. Dependence on an axiom is counted for each theorem
(`axiom_dependent`), and Stage 1 already refuses on it.

Its recommendations on the four decisions match what was done (`docs/OWNER_DECISIONS_2026-11-28.md`),
including the one it asked for before any freezing: the implicit proposer hands over the best
relation of two terms, of three and of four separately.

## 2. The nine claims about the search

Each was checked against the code and the committed records. All nine hold in substance. Claims
3, 4, 7 and 9 were already on record here; the others were not, or not with numbers.

| # | Claim | Checked against | Holds? |
|---|---|---|---|
| 1 | The deeper exhaustive pass has no time budget; it stalls three-input readings and does nothing on four | `proposers/deep_pass.py`; the budget ladder's records; timed here | Yes. Timed alone on 700 rows of three inputs: 108 seconds, 502,733 candidates, where the search was asked for 30. With four inputs it returns at once. The ladder's "30-second" readings took 101 to 348 seconds on three inputs. It does not run to the kill limit, as the note says; it ends by itself well past the budget |
| 2 | PySR's candidates are handed over by falling loss, duplicates are told apart by text only, and nothing forbids a function inside itself | `proposers/pysr_proposer.py` (frozen); the 266 runs of the portfolio test | Yes. Recounted: two candidates of one shape with different constants in 85 runs of 266 (the note: 83); a trigonometric function inside another among the candidates in 166 runs and in 66 of the 77 misses (the note: 131 and 56; it does not say how it counted) |
| 3 | The memory of confirmed laws runs after the searches for up to 75 seconds, takes three of the ten places, and holds bloated shapes | `proposers/structure_memory.py`, `discovery_l2/memory_stage.py`, `benchmarks/structure_memory.json` | Yes. The three places were already recorded. Of its 147 remembered shapes, 9 hold a power of five or more (one of the fifteenth, 2,204 characters long) and 12 a trigonometric function inside another. Its 28 one-variable parts have no `exp(C*v)`, no `tanh`, no `v/(C + v)` |
| 4 | The route's first pass has no constants inside its expressions, tries them on fixed points, and is pure Python beside a Rust version that is not used | `discovery_l2/exhaustive_search.py`, `main_search.py` (frozen) | Yes, and by design: the module says so. A law that is not finite on its 24 fixed points in [0.6, 3.1] is never a candidate, whatever the table's own range. The Rust first pass is opt-in and used by no recorded run |
| 5 | The refit of constants ties equal constants together, and the fit that is confirmed is the expanded sum, not the law as written | `main_search.polish`, `contrast_resolution.fitted_candidate` (both run here) | Yes. `1.0*x0 + 1.0*x1` refitted to rows of `2*x0 + 3*x1` comes back as `2.487*x0 + 2.487*x1`; started from `1.0` and `1.1` it comes back right. `(x0 + 0.5)**4*(x1 + 1.5)**4` is fitted with 25 free coefficients |
| 6 | The split search prefers a shorter part that fits within 2 % | `proposers/split_search.py`, `tensor_train_reading.py` | Yes (`TOLERANCE = 0.02`, `MARGIN = 1.02`). It also proposes the best-fitting whole beside the shortened one. Not in force |
| 7 | When the shortest law of a class fails the residual check the others are not tried | `main_search.py` (frozen) | Yes: decision 1. One number needs correcting. The note counts 34 of 77 SRBench misses as ending in `equivalent_class`. The count is right and it is not this fault: those 34 are laws that were reported from a class, accurate and not the law's form. The fault's own mark (the search widened, a law rejected) is on 9 of the 77 |
| 8 | The last checks are handed the class's members as the searches wrote them, and not the member the contrast preferred when a shorter one was taken | `vouched_route_v2.py`, `vouching_v2.py` (both frozen) | Yes. See section 3 |
| 9 | A fitting error ends a reading; the units reduction gives up on any zero in the table | committed records; `proposers/composed.py` | Yes. `LinAlgError` is in 14 lines of five committed records (the note: 15), and nothing between the search and the report catches it. Already recorded as audit M-24 |

## 3. What claim 8 changed here

The route in force writes the law it reports in fitted form, and hands the last checks its rivals
as raw text: `contrast.equivalent`, the members the contrast could not tell from its survivor. Two
things follow. The rivals are tested as the searches wrote them, unfitted. And when the main
search takes a shorter member in place of the survivor, the survivor is not among the rivals at
all, while the law's own raw text is.

**In the step built for decision 1.** `discovery_l2/next_member_route.py` had the same habit: it
checked each member's text as it stood. It now fits each member as the route fits the one it takes
and writes it as the refined path writes a law, and it hands the last checks every other member,
the survivor included. On its study's 366 readings this changed 7, and no count of laws added.

**In the route in force.** Not changed: both files are frozen. The effect runs towards vouching
too readily, since a rival left out, or tested in a form that fails for want of a fit, cannot
speak against the law. How often that has happened is not known; the records keep the class and
not what the checks were handed. Measuring it needs the product's search on GitHub's machines.

## 4. The repairs and the new engine: what is proposed, and what it would take

Nothing below is built. Each item changes the search stage that is in force, or is a large new
build, and waits for the owner's word (AGENTS.md 1.5). Most need PySR, which runs only on
GitHub's machines.

**Already tried on the owner's word of this morning:** the next law of a class (claim 7) and the
implicit form handing over one relation of each size (part of the note's "core 2").

**Small, and wrapped around the frozen files** (the note's first tier):

1. A time budget for the deeper pass, or a use for its core on tables of four inputs and more.
2. PySR's candidates chosen across sizes and not by falling loss; duplicates told apart by shape
   and by value; no function inside itself. `pysr_proposer.py` is frozen, so this is a new
   proposer beside it.
3. The memory run beside the searches and not after them, and its remembered shapes cleaned of
   those that were never the law.
4. The last checks handed every member of the class, each fitted (claim 8). A new version of the
   route, with its own test.
5. A fitting error caught for the one candidate, so that it does not end the reading (claim 9).

Each would be measured as the note asks: against the stage in force, table by table, at equal
time, with the share of wrong reports held to the limits of `discovery_checks`.

**The new engine** (the note's five "lenses"). Its first lens, a transformed target matched to a
ratio of small expressions by linear algebra, is the implicit proposer of decision 2 made
general. What that proposer showed this morning bears on it directly: it got three laws reported
that the first pass never had, and it got an exponential reported as an algebraic law. An engine
built on the same step needs the same guard, and the note's own kill rule for it is the right
first test. Nothing of it is started.

## 5. What the note could not have known

It read the code and the committed records. Three things in it are stated more strongly than they
checked out: the deeper pass ends by itself and is not killed (claim 1); the 34 misses are not the
fault of claim 7; and the split search does also propose its best-fitting whole (claim 6).

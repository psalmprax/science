# A solution for each of the four things that showed no gain (2026-11-27)

On 27 November four things that had shown no gain were put to the owner for retirement: the learning
loop, the language model steering the search, the language model shown the system's own solved cases,
and the tensor-train equation model. The owner's word (10:23): do not retire them; find a solution
for each.

**This page is a design. Nothing here is built except one feasibility check (section 5), nothing is
in force, and nothing here is a result.** Each design names the measured cause of the failure, the
redesign, why it should work (this repository's own records first, then outside work), a first look
that can stop it before it is built, a test with a control and a mark, and the rule that retires it.
Marks are proposals; they are fixed when a plan is frozen (AGENTS.md 1.6), and what any of this
changes in force is the owner's to decide (1.5).

## 1. What the four have in common

| Thing | What it changed | What was measured |
| :--- | :--- | :--- |
| The learning loop (`discovery_l2/learning_loop.py`) | which remembered parts are offered, and the order of the first pass | invented laws 5 against 5 from an empty memory and 8 against 10 from the memory in force; SRBench 51 against 51; LLM-SRBench 18 of 80 against 19 of 79; ODE systems 31 against 25 of 60 (8 gained, 2 lost, p = 0.055) |
| A language model steering the search (`proposers/llm_controller.py`) | the search's operators, inputs and seeds: one call a table, twelve rows shown | 173 of 266 against 174; operators narrowed in 162 runs; the one inaccurate law came from a narrowing; the two runs together recover 191 |
| A language model shown solved cases (`proposers/experience.py`, `scripts/study_llm_retrieval.py`) | the model's prompt | a solved law on the list for 13 of 60 (model alone), 10 (the cases refitted, no model), 7 (the model shown the cases); the two lists joined: 19 |
| The tensor-train equation model (`apedp_mmlm`) | the price of a law's text | 376 bits a held-out law, 40.8 after its repair, against 33.7 for the counted automaton; and the best prior the repository has, put in as the route's prior, gained 0 and lost 0 on 130 laws (`RESULT-MMLM-ORDER-2026-11`) |

Two causes run through all four.

1. **They changed the order or the weight of candidates the searches already produce, and were
   measured on laws the route already reaches.** There the route is saturated: in the MMLM test both
   arms recovered 125 of 130. A better order has nothing to show.
2. **Three of them put something in the place of a thing that worked.** The model's configuration
   replaced the default search; the cases replaced the model's own reading of the table; the learned
   order replaced the size order. Where the same sources were *joined* instead, the count rose:
   191 against 174 (the two controller runs), 19 against 13 (model and cases as two lists), and four
   searches side by side 189 against 174 (`RESULT-PORTFOLIO-SEARCH-2026-11`).

What the route does miss, by its own records: laws beyond its reach in the seconds it has (at the
higher curriculum levels the law is not on any list); tables of more than three inputs (no law by
form in 240 readings; an accurate law for 13 of 80 tables of four and five inputs, where the split
search's list had one for 36); and noise (beyond the measured range of three inputs and 5 % noise,
8 laws reported in 100 readings, and none at 20 % noise).

Outside work says the same. A 2026 study of LLM-SRBench's synthetic tasks found that a fixed library
of operators with no language model reaches about 78 % of them, that adding a model's candidates to
that library changes almost nothing, and that the model adds 28 to 39 tasks of 129 only once the
library is cut down so that it no longer covers the task (arXiv:2607.28684). And a controller around
PySR that was not tuned scored 5.43 % where PySR alone scored 32.82 % (the starting point of
arXiv:2610.16966, below).

**Three rules follow, and every design below keeps them.**

* **Added, never in place.** What is learned, retrieved or suggested enters as more candidates or as
  a template for one more search. It never reorders, and it never narrows what the searches in force
  may try.
* **Tested where a gain can show.** Each design names beforehand the ground it claims (laws outside
  the search's vocabulary, deep laws, wide tables, noisy tables, families of tables) and carries a
  guard set of laws the route already recovers, where the expectation is "no loss".
* **Against a matched random control at equal CPU.** The same number of extra candidates or members,
  with random content. The owner's own study found that a mined library bought about what random
  compositions of the same length bought (`docs/IDEAS_FROM_OUTSIDE_2026-11-25.md`, section 7); the
  2026 study above asks for the same control ("sham knockouts"). Without it a gain cannot be credited
  to what was learned or suggested.

## 2. The learning loop: learn across tables

**Cause.** The loop is sound and had nothing to learn from. It learns from each confirmed law its
parts and its text. The text orders the first pass, which changes speed and not reach. The parts help
a later table only if that table reuses them, and the record of the family runs says where that
happened: on the ODE systems, whose components share parts (8 gained, 6 of them in the second half of
the stream), and nowhere on benchmarks whose problems share little
(`benchmarks/learning_stream/dev_families.json`). Laws drawn one by one from a grammar are the worst
ground for a learner.

**Redesign.** The loop learns three things, each with its own proof, and drops the learned order.

1. **Families.** A family is a set of tables said to describe one phenomenon under different
   conditions: the components of one system, one experiment at several temperatures, or one wide table
   in which an input takes few values, whose rows are then grouped by it. Families are declared, or
   read from repeated values; in the first version none is guessed from likeness.
   * *A confirmed shape is tried on its siblings first.* When the referee confirms a law on one table,
     its shape with the constants free is refitted on every sibling before any search and goes to the
     referee like any proposal. The cost is one least-squares fit.
   * *Siblings without a law are searched together:* one expression, constants of its own for each
     table. PySR does this with a template whose parameters are indexed by the table
     (`TemplateExpressionSpec(..., parameters={...})`, the table's number as one more column; PySR 2
     has it by its migration notes, and it has not been run here yet). The shape is then decided by all rows of the family while each table keeps
     its constants: with six tables the shape sees six times the rows. That is the direct answer to
     noise.
   * *The constants become a table.* For a shape confirmed on a family, each constant set against the
     family's conditions is a small table, and candidates for it are few and cheap. A candidate is
     **not** judged on those few rows. It is put into the shape, which gives one law over the family's
     inputs and its conditions, and that law is refitted and refereed on all rows of all tables. A law
     of three inputs is then found as a law of two and a law of one, and never searched as three. The
     memory drops every fitted constant today; this is the first thing here that compounds
     (idea C4 of 25 November, with the step that makes it pass a referee).
2. **Parts that earn their place.** `recursion/library.py` already finds repeated parts by
   compression. A part becomes an *operator of the search* (one node where it was four) only when a
   selection batch drawn afresh shows that the search with it recovers more laws at equal seconds and
   loses none, **and** more than with a random composition of the same size given the same place. The
   recursion's sealed judge decides; `retract` undoes it.
3. **Pairs for the likeness of tables** (section 4): every fitted candidate of every search is the
   exact law of the table it generates, so the loop's searches yield training pairs at no cost
   (hindsight relabelling, arXiv:2507.14172 and arXiv:2402.04858).

**Why it should work.** Here: the one near-gain was where structure recurs; noise above 5 % is a
measured miss, and pooling rows attacks it by arithmetic. Outside: this is how the first discovery
programs worked. BACON.3 (Langley, IJCAI 1979) kept data "at varying levels of description", held
some quantities fixed, found a law, and made that law's constants the data of the next level; it
recovered versions of the ideal gas law, Kepler's third law, Coulomb's law and Ohm's law that way.
One form fitted to several datasets with constants of their own is now a studied problem
(multi-view symbolic regression, arXiv:2402.04298; class symbolic regression, arXiv:2312.01816), and
a 2025 comparison of four implementations, PySR among them, gives the caution to build in: the
per-table constants over-fit, so their number must be capped and paid for (arXiv:2509.10500).

**First look** (CPU on GitHub's machines, no key, a day): six families of four tables at 5 % and 10 %
noise, searched table by table and together. If the joint search does not find shapes the single
searches miss, stop.

**Test.** Streams of families generated in two levels (a hidden shape; constants that are fixed or
follow a hidden law of one condition; 1 %, 5 %, 10 % noise), strangers without a family as the guard,
and a second arm from another generator. Arms at equal CPU: the route in force table by table (A);
with the sibling refit (B); with the joint search (C); with laws of constants (D); and the control
(R): C with the tables grouped at random across families. *Marks:* at 5 % and 10 % noise, laws by
form, C against A and C against R, more gained than lost by a paired sign test below 0.05; no loss on
the strangers; wrong reports no higher; for D, wide laws reported and none wrong.

**Rule that retires it.** If C beats neither A nor R at 5 % and 10 % noise, the loop's learning is
retired and the memory stays the file built once, as in force.

## 3. A language model steering the search: a second reader with evidence

**Cause.** One call a table; twelve rows of numbers and nothing else; an answer that *replaced* the
default configuration; no second look. Outside work shows the same pattern and what changed it:

* A harness around PySR, before tuning, scored 5.43 % on LLM-SRBench's synthetic tasks where PySR
  alone scored 32.82 %. After its prompts and workflow were optimised on 160 training tasks it scored
  65.37 % with about 23 calls a task. In its ablation, feeding back what the best candidate leaves
  unexplained mattered most (arXiv:2610.16966).
* A comparison of the roles a model can take found the controller ahead of plain PySR by 0.958
  against 0.922 on 74 Feynman equations, with up to ten rounds and a memory of what helped; the
  variables were named, and no control for recall was found in the part of the paper that was read
  (arXiv:2607.04156).
* An agent that first infers structure and then writes PySR templates was *behind* PySR in one run
  (35.1 % against 37.8 %) and ahead only in the best of three (42.3 %): a gain from several
  differently configured runs, as here (arXiv:2602.12259).

Here, on what the model can do: shown 24 rows it had the law's shape among its shapes for 62 of 84
laws, at a median place of 2 (`benchmarks/mmlm/dev_llm_expert.json`); writing whole laws it recovered
78 against 152. It names shapes well and searches badly.

**Redesign.**

1. **Added, never in place.** The model's plan is one more member of the stage. The members in force
   always run, so a narrowing can cost seconds and not a law.
2. **A dossier, not rows.** Computed here, the same for every model: for each input the direction and
   bend of the target (the signature of `experience.py`), the slopes at both ends of the range, flags
   for saturation and for a period; the split reading's shape and, if built, the tensor-train
   reading's ranks (section 5); the noise; names and units where the table has them; and from the
   second round the best candidates with what they leave over: the trend, bend and period of the
   residual against each input.
3. **What it may answer**, each item checked by the whitelist parser as now: shapes with free
   constants (fitted here, sent to the referee and to PySR as guesses); a template that says which
   inputs meet where (compiled to `TemplateExpressionSpec`); operators to **add** from a fixed longer
   list; derived inputs; and, before each move, what it expects the move to do to the residual
   (item 3 of the research-intelligence roadmap, scored).
4. **Rounds.** Plan, a short search, a reading of what is left, a repair: two branches, three rounds,
   at most eight calls a table, stopping when the referee confirms.
5. **A harness that is learned, then frozen.** The prompts and the workflow's few numbers are improved
   on invented laws drawn for development, by the system's own count of laws through the referee, with
   reflection on the traces (the method of arXiv:2507.19457, which the harness paper used). Then the
   harness is frozen by digest and tested once on laws drawn afterwards. This is the recursion's rule
   applied to a prompt: the improver edits the proposer, the sealed judge decides. The paper's own
   optimisation ran 4,728 solver executions; a first version here is written by hand and revised in a
   few hundred calls, and anything larger is put to the owner with its count.
6. **Context where it exists.** Anonymous inputs by default, which is the clean test. One separate arm
   with names and units on LLM-SRBench, under its gate, counts only.

**First look** (at most 120 calls): tables the route has already *missed* in development (wide,
deep, 10 % noise), the dossier, one call each: is the law, or an accurate law, among the shapes it
names? If it cannot name shapes where the search fails, no harness will help, and the design stops.

**Test.** Ground claimed: laws with an operator outside the default set, deep laws, wide tables; guard:
laws the route recovers. Arms at equal total CPU and the same number of members: the stage in force
with one more default member on another seed (A); with one member planned at random from the
grammar's own prior (B); with the model's member (C). *Marks:* C against A and C against B, more
gained than lost on the ground claimed, sign test below 0.05; no loss on the guard; wrong reports no
higher. *Calls:* a first run of 40 tables is at most 320 calls, inside the standing cap; learning the
harness needs more, and that count is named before it runs.

**Rule that retires it.** If C does not beat B, steering by a language model is retired for tables
without names. The named arm is then the one thing left to try, once.

## 4. A language model shown solved cases: join the lists, not the prompts

**Cause.** The cases carried information the model did not have, and the prompt that mixed them lost
both. Of the 60 problems, 6 were solved only by the refitted cases and 9 only by the model alone;
joined as two lists they solve 19, and the model shown the cases adds none to that. Against the cases
alone it lost 13 problems and gained 3 on accurate laws (sign test 0.021). It had been shown eight
bare shapes from other tables with nothing to say which of them fits this one, and it leaned on them.
Beside the memory the gain is real and small: the memory taught 80 laws had the law for 19 of those
60, the model added 2 and the cases 3 (`CHANGELOG.md`, 25 November).

**Redesign.**

1. **The cases are a proposer of their own.** The nearest shapes, refitted on the table, go to the
   referee and to PySR as guesses. No model, no key.
2. **The likeness is learned.** The signature is written by hand from rank statistics. Tables drawn
   from one shape with other constants, ranges and noise are near by construction, and the loop's
   hindsight pairs give more. A metric learned on those pairs (boosted trees or a linear map, on CPU)
   is measured directly: is the right shape among the eight nearest, against the hand-made signature
   and against chance. The owner's study found features a system builds for itself ahead of
   hand-written ones in 15 of 15 pools.
3. **A model sees a case only as evidence fitted here.** "Shape S, refitted on this table, misses by
   7 %; what is left rises with x1." Shapes that failed are listed as failed. It is asked for an edit
   of the best and for one shape unlike any shown, and its list is kept apart and joined. This is the
   repair round of section 3 started from a case, so the two designs share one mechanism and one
   harness.
4. **A denser store:** hindsight cases, under a cap on length so that over-fitted shapes are not
   taught.

**First look** (CPU, hours, no key): on the 60 problems already drawn, the eight nearest cases against
eight cases drawn at random from the same store. If the nearest are no better, the likeness carries
nothing and the design stops at point 2.

**Test.** Fresh curriculum problems at levels 2 to 5 and a guard, counted as laws through the referee
and not as places on a list. Arms at equal seconds and equal numbers of candidates: the route; with
the nearest cases; with random cases (control); with the nearest by the learned likeness; and, with a
key, with the model's edits joined. *Marks:* nearest against random and against the route, more gained
than lost, sign test below 0.05; the learned likeness no worse than the hand-made one.

**Rule that retires it.** Nearest no better than random: the store by likeness is retired. The model's
edits add nothing to the joined lists: a model is not shown cases again.

## 5. The tensor-train equation model: from the text of laws to the structure of tables

**Cause.** Two, and only the first is the engine's. The engine is not fitted to a likelihood (its
cores are moved by rank-one steps written by hand and its readout is counted), hence 376 bits; with a
fitted readout it equals the unigram (40.8). And the job cannot pay: a prior over the text of laws
orders candidates, and the best one bought no law.

**Part A. The repair that cannot lose.** A counted n-gram model is a probabilistic automaton, and a
probabilistic automaton is a tensor train with the same non-negative matrix for a token at every
position (the equivalences between uniform matrix product states, hidden Markov models and weighted
automata are in print: arXiv:2010.10653). So the train is built *from* the Kneser-Ney automaton, one
state a context, and costs the automaton's 33.7 bits by construction. A few hidden states are added,
coupled at zero, and fitted by expectation-maximisation; the result is kept only if the held-out bits
fall in the same five folds. Expect little: the text is a few hundred laws, and the preregistered
byte-model test found counting ahead of every tensor model (`RESULT-MMLM-QI-2026-11`). An hour of CPU
says, and it buys no law either way.

**Part B. The job a tensor train is good at.** A tensor train models multilinear structure, and a law
of many inputs made of sums and products of one-input functions has exactly that structure:

    y = G1(x1) G2(x2) ... Gd(xd)

with each `Gi(xi)` a small matrix of one-input functions. One product is rank one; a sum of r
products has rank at most r at every cut; the five forms the split reading knows
(`proposers/table_shape.py`) are the cases of rank one and two for two groups of inputs. So the split
reading is already the first rung of this ladder. What the train adds: it is fitted to the **real
rows** (the split search's parts are tables of a smooth model's values, and the record says what that
costs: parts "fitted to the smooth model's own wiggles"); it is one model for every cut at once; its
cost grows in proportion to the number of inputs, where the smooth model's grows with the cube of the
rows; and it needs neither PySR nor Julia. Tensor trains have been used for governing equations, with
a fixed dictionary of functions and for dynamical systems (arXiv:1809.02448, arXiv:2002.12388,
arXiv:2208.01591). Reading *formulas* out of a train fitted with free one-input functions was not
found in print; that search was not exhaustive.

**The feasibility check** (`scripts/study_symbolic_tensor_train.py`,
`benchmarks/symbolic_tensor_train/dev_probe_1.json`; development, not a result). The 40 wide
development tables of four and five inputs at 1 % and 5 % noise, 700 rows, one core. Every core entry
is a cubic spline; the fit is alternating least squares; the rank is the smallest that no higher rank
beats on rows left out.

| | 1 % noise | 5 % noise |
| :--- | ---: | ---: |
| Tables | 20 | 20 |
| The spline train misses left-out rows by no more than 1.5 times the noise | 14 | 17 |
| The spline train is accurate on rows without noise (R² 0.999) | 14 | 6 |
| Of those at the noise: every core holds the law's one-input functions within 5 % | 12 of 14 | 8 of 17 |
| Fitted at rank one, at the noise, and read as a formula | 5 | 11 |
| ... read accurately | 5 | 4 |
| ... **the law itself** | **1** | 0 |
| Seconds for the fit, median | 7.7 | 8.4 |
| *The split search on the same tables (`dev_search_first_run2.json`): a proposal accurate* | *16* | *13* |
| *... the law itself* | *0* | *0* |
| *... seconds, median* | *107* | *113* |

What it shows:

* **It is feasible and cheap.** At 1 % noise the train reaches the table's noise on 14 of 20 wide
  tables in about eight seconds, against about 107 for the split search.
* **The cores hold the law.** Where the fit reaches the noise at 1 %, the functions each core spans
  are the law's own one-input functions to within 5 % in 12 of 14 tables (a median of 0.7 %), while 200
  other functions of the same kinds are missed by a median of 28 %. That is what a reading needs.
* **One law of four inputs was read exactly**, the first by any method on these tables:
  `0.7*x0**2/(x2*(0.7 + exp(-0.3*x1))*(1.7 + exp(-1.3*x3))*(sin(1.7*x1) + 3.7))`, with its constants
  to within 5 %.

What it does not show, and what is against it:

* **It is not ahead of the split search.** At 1 % the split search had an accurate proposal for 16
  tables and the train's model was accurate for 14 (two of them tables the split search missed, four
  the other way). **At 5 % it is behind: 6 against 13.** There the rank choice under-reads: 13 tables
  were fitted at rank one, six of which are not products, and none of those six was read accurately.
* **Formulas were read only at rank one**, that is for the five laws that are products. A core of rank
  two holds its functions only up to a mixing of its entries, and unmixing them is not written.
* **The exact form is rare for a reason no search removes.** In four of the five products a part made
  of two blocks was replaced by a one-block stand-in that fits the rows as well. The rows cannot tell
  them apart at this noise and range, which is what the split search's record found too.
* The tables are the development grammar's and the one-input search knows every block of it.

**What would be built** (new modules; nothing existing changes): the reading at rank two and three
(find the simple one-input functions inside each core's space, which fixes the mixing); a rank choice
that holds at 5 % noise; one change of the target before the fit (logarithm, reciprocal, square), which
brings laws such as `sqrt(x1**2 + x2**2)` and `exp(-x1/x2)` into the class; and the member that hands
its wholes to the stage and to PySR as guesses.

**Test.** Fresh wide tables at 1 % and 5 % noise, half from another generator with blocks outside the
vocabulary and with laws in which inputs meet inside a function, where the train must say nothing
rather than something wrong. Arms with the member in the stage, the referee deciding: the stage in
force; with the split search; with the train; with both. *Marks:* with the train against the stage in
force, more accurate laws reported than lost, sign test below 0.05, wrong reports no higher; the law
itself is counted and is not a mark. Which of the two wide-table members is proposed is read off the
same run: the one with more accurate reports a CPU-second, both if together they beat either. **This
run would also answer the split search's open question** (its list is accurate for 36 of 80 wide
tables where the route in force is for 13, and it has never been run inside the route).

**Rule that retires it.** If the stage with the train reports no more accurate wide laws than the
stage in force, or if the split search does everything the train does in the same test, tensor trains
are retired from law discovery and stay only in the video line, where the owner asked for them.

**Part C, listed and not proposed.** The train as a sampler that learns during one search (as in
arXiv:2301.12162). The owner's own studies found no gain from learning inside one problem at equal
time, so it waits for B.

## 6. One bench for the four

The four tests share their ground, their controls and their reading, so they are one bench built
once: laws drawn by seeds assigned when a plan is frozen and by no development run; the ground in
strata (reachable guard, outside the vocabulary, deep, wide, noisy, families); every arm at equal CPU
with the cost record the roadmap asks for (item 4); every comparison paired, with gained, lost and a
sign test; recovery by form and accuracy both counted; wrong reports bounded as the rule of 25
November says.

## 7. Order proposed

| | What | Why here | Needs |
| :--- | :--- | :--- | :--- |
| 1 | The cases as a proposer, with the random control (section 4, points 1 and 2) | the smallest; no key; its first look takes hours | CPU |
| 2 | Families in the learning loop (section 2) | the one design whose gain follows by arithmetic wherever families exist, and the first that compounds | CPU on GitHub's machines |
| 3 | The tensor-train reading's missing parts, then its test beside the split search (section 5 B) | feasible and cheap by the check; behind the split search at 5 % until its rank choice is repaired | CPU, this workspace |
| 4 | The language model as a second reader (section 3), starting with its first look of 120 calls | it uses what 1 and 3 compute as its dossier, and it is the only one that costs calls | the key in GitHub's secrets; the count said before each run |

Each is a development study first, then an independent review, a preregistration, one run, and the
owner's decision. Part A of section 5 (an hour) can run at any time.

## 8. What is not claimed

* No design here has been tested. The one check that was run is on development tables and is not a
  result.
* The honest odds, as read from the records: families are likely to pass their own test and matter
  only where families exist; the cases are likely to give a small gain or none over random cases; the
  tensor-train reading is feasible and its gain over the split search is unproven; the language model
  has the strongest outside numbers and the weakest record here, and the outside numbers come from a
  benchmark on which a fixed library does as well.
* Some of the measured zero on wide tables is not a failure of search: where two forms fit the rows
  equally, no method returns "the law itself". What turns such a class into a law is rows that tell
  the forms apart (idea C2 of 25 November), and a product of one-input functions makes those rows
  cheap to name.

## Sources

In this repository: `preregistration/RESULT-MMLM-ORDER-2026-11.md`,
`RESULT-LLM-CONTROLLER-2026-11.md`, `RESULT-PORTFOLIO-SEARCH-2026-11.md`, `RESULT-MMLM-QI-2026-11.md`;
`benchmarks/learning_stream/dev_families.json`, `benchmarks/llm_retrieval/dev_run1.json`,
`benchmarks/mmlm/dev_llm_expert.json`, `benchmarks/wider_conditions/dev_run4.json`,
`benchmarks/table_split/dev_search_first_run2.json`,
`benchmarks/symbolic_tensor_train/dev_probe_1.json`; `docs/IDEAS_FROM_OUTSIDE_2026-11-25.md`;
`docs/ROADMAP_RESEARCH_INTELLIGENCE_2026-11-24.md`.

Outside, read in abstract or in full on 27 November: library reachability in LSR-Synth
(arXiv:2607.28684); HarnessSR (arXiv:2610.16966); language models controlling PySR
(arXiv:2607.04156); KeplerAgent (arXiv:2602.12259); GEPA (arXiv:2507.19457); LLM-SRBench
(arXiv:2504.10415); SOAR (arXiv:2507.14172) and CodeIt (arXiv:2402.04858); DIGEST (arXiv:2610.33880);
neural-guided seeding of genetic programming (arXiv:2111.00053); multi-view symbolic regression
(arXiv:2402.04298) and its comparison of implementations (arXiv:2509.10500); class symbolic regression
(arXiv:2312.01816); PySR's template expressions (its migration notes for version 2); Langley,
"Rediscovering Physics with BACON.3" (IJCAI 1979); quantum tensor networks, stochastic processes and
weighted automata (arXiv:2010.10653); MANDy (arXiv:1809.02448), tensor network approaches for learning
non-linear dynamical laws (arXiv:2002.12388) and their block-sparse successor (arXiv:2208.01591);
PROTES (arXiv:2301.12162).

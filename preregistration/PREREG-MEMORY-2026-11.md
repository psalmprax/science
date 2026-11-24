# PREREG-MEMORY-2026-11: does a memory of confirmed laws help the default route on laws it has never seen?

**Status:** frozen before the run by `preregistration/DIGEST-MEMORY.json`. Runs once, on GitHub
Actions (`.github/workflows/memory-laws.yml`), started by the first branch named
`memory-laws-run-*` whose shards start.

## What is tested

Every search here starts each problem knowing nothing of the last one. The learned prior changes
the order in which an enumeration is tried; it does not put a law's parts on the table.

`apedp_dyn/proposers/structure_memory.py` keeps, of every law the referee has reported, the shapes
of its one-variable parts (`2.3*x0/(x0 + 1.7)` is kept as `v/(C + v)`) and of the whole law, with
no constant's value and no variable's name, and grows candidate laws for a new problem from those
parts, fitting every constant to the new rows. `benchmarks/structure_memory.json` is that memory
built from the 238 laws reported in two recorded runs: 28 motifs met in at least two laws, 147
whole shapes.

Development, on 80 laws of the same grammar that the memory had not seen, at 1 % noise, with the
memory as the route's *only* search stage and no search engine: it handed the path the law itself
in 15 of the 80, and the route recovered 13 where it recovers 1 with no search stage at all. It
also reported 3 wrong laws of 67 (and 6 of 39 with no search stage): a route whose only rivals are
approximations confirms approximations, which is why the memory is tested beside the searches and
never in their place.

That measurement says the memory can find laws. It does not say it finds laws the searches miss.
This run measures that.

## Data and protocol

**Laws.** `benchmarks/memory_blind_laws.json`: 40 laws drawn by `scripts/run_memory_laws.py laws`
from the invented-law grammar (`scripts/build_invented_laws.py`: one to three inputs on (1, 5),
building blocks joined by sums, differences, products and quotients, constants from a list of
unround numbers), with seed 20261127, **after** the memory file was written. A draw is left out if
its structure is in the invented-laws benchmark (by fingerprint; no held-back law is read) or
among the 80 laws the memory proposer was developed on. 9 laws have one input, 19 two, 12 three.

**Rows, noise, split, scoring** are the invented-laws runner's and through it the SRBench
runner's, unchanged: 1,000 rows drawn uniformly, 70 % for development and 30 % as the pool the
path may measure, 500 further rows without noise for scoring; noise 0 and 1 % of the target's
root mean square; a law is *recovered* if it is the law after refitting constants
(`symbolic_recovery`), *wrong* if it is reported, not the law, and below R² 0.999 on the scoring
rows.

**Two arms for every law and noise level, 160 runs in all, given the same searched candidates.**
The default stage's four searches run once, 240 seconds, and

- `default` hands the path their candidates: the route in force
  (`refined_search.discover_refined`);
- `memory` hands it the same candidates with the memory's laws beside them
  (`discovery_l2/memory_stage.with_memory`): up to three laws grown in at most 60 seconds, taking
  turns with the searches' candidates, so that with ten searched candidates the memory's three
  take the place of the searches' eighth to tenth.

A run that crashes or exceeds 1,500 seconds counts as recovering nothing and reporting nothing.

## Expectations, fixed now

1. **What either arm reports can be relied on:** of the laws each arm reports in its 80 runs, at
   most 5 % are wrong (the canonical `REFEREE_MAX_FPR`).
2. **The memory helps:** the `memory` arm recovers at least as many laws as `default` at each
   noise level, and more in total.
3. Reported without a criterion: in how many runs the memory handed over the law itself; laws
   gained and lost run for run; laws reported that are accurate without being the law; the
   path's actions; errors, timeouts and time.

**The system's own forecast** (`preregistration/FORECAST-MEMORY-2026-11.json`, filed with this
text): the `default` arm recovers 50 of 80 (37 to 62 at 80 %), reports 64, 0.4 of them wrong. It
was learned from the routes' runs on the physics benchmarks and has seen no run on invented laws.

**The author's own expectations**, recorded so that they can be wrong, and no criterion. The
forecast above is too high: on the twenty development laws of the invented benchmark the default
route recovered 13 of 40 runs, so about 26 of 80 here, and the forecast will miss low. The memory
hands over the law in about 12 of 80 runs, about half of them laws the searches find anyway, so
the memory arm gains 4 to 8 laws and loses one or two whose searched candidate it displaced.
Expectation 2 holds, with perhaps one chance in three that it does not. Expectation 1 holds for
both arms.

A miss is reported as a miss.

## What follows from the result, fixed now

- If 1 and 2 hold, the memory is put beside the default search stage for tables, in a new module
  around the frozen ones, and this result is stated wherever that stage is described.
- If 1 misses for the `memory` arm, it is not adopted, and the wrong laws are listed with what the
  memory had handed over in each.
- If 2 misses, the result is recorded as: a memory of 238 reported laws adds nothing to four
  searches on new laws of this grammar. It is not adopted, and no variant of it is called
  measured until it has been run on laws drawn after that variant was written.
- The forecast is scored against the `default` arm whatever else happens, and the score is
  recorded beside the result.

## Disclosed before freezing

- **The proposer's way of joining parts was written knowing the grammar.** The grammar joins
  blocks left to right by sum, difference, product and quotient with a constant; the proposer
  grows a law by adding a scaled motif, multiplying by one, or multiplying or dividing by a
  constant plus one. A gain here says a memory helps on laws built this way. It says nothing for
  laws with several variables inside one function, which one-variable motifs cannot hold and
  which most of the physics benchmarks' laws are.
- **The memory holds laws of the same family.** 26 of its 238 laws were reported on the twenty
  development laws of the invented-laws benchmark. None is a law of this run, and what is kept
  of them is shapes that occur throughout the grammar.
- **The memory holds reported laws, right or not.** The route does not know which of its reports
  are right, so neither may its memory. Motifs met in a single law are dropped.
- **Unequal cost.** The memory arm spends up to 60 seconds more per run and the same search.
- **No discovery was run on any of these laws,** by any engine, before freezing. What was run on
  them: the generator and its acceptance rule.
- **Development used the native first pass** (`discovery_l2/fast_first_pass.py`). This run does
  not: both arms use the reference first pass.
- **Two noise levels, uniform rows.** The conditions of the earlier runs, and nothing beyond them.
- **The smoke job's two laws** are in no set.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.

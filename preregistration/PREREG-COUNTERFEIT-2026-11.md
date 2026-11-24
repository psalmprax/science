# PREREG-COUNTERFEIT-2026-11: handed counterfeits of a law, whose reports can be relied on, and do the checks built from the red team's findings hold on laws they were not built on?

**Status:** frozen before the run by `preregistration/DIGEST-COUNTERFEIT.json`. Runs once, on GitHub
Actions (`.github/workflows/counterfeit-benchmark.yml`), started by the first branch named
`counterfeit-run-*` whose shards start.

## What is tested

Two things, on the same cases.

**A benchmark whose score is reliability.** Benchmarks of law discovery count the laws a method
finds. A method that reports its best fit every time does well on that count and cannot be relied
on. Here every case hands the method a law's *counterfeits* as the candidates of a search: other
forms, each fitted to the law itself without noise, so each is the closest law of its form
(`apedp_dyn/redteam/counterfeit.py`). In half the cases the law is not among them and every
candidate is wrong. A wrong report costs nineteen right ones.

**The first run of the checks on laws they were not built on.** The red-team campaign of
2026-11-22 (`benchmarks/red_team_campaign.json`, 34,728 worlds, development by its nature)
searched for data on which the route in force reports a wrong law. On worlds drawn at random
under the conditions of the preregistered runs it found 1 wrong law in 2,198 reports; with 3 to
10 % noise, 268 in 1,057. Two causes account for nearly all of it (audit M-31, M-32): where the
path cannot tell candidates apart it reports the one with the shortest description, and its last
check lets through an error a third the size of the noise.

`apedp_dyn/discovery_l2/vouching.py` was built from that. To a law the route is about to report,
and to the candidates the path could not tell from it, it applies:

1. a structure test with a known null: the path's own statistic at 8, 32 and 128 neighbours, each
   compared with its distribution when the residuals' signs are flipped at random, at the
   canonical level 0.01;
2. a class rule: members of an unseparated class that pass the test must agree with each other
   to R² 0.999 on every input at hand, or nothing is reported;
3. a power rule: an error just large enough to make the law wrong is planted in its residuals,
   in the shape of each rival's difference from it and of low-order polynomials of the inputs,
   and the law is withheld if the test would have seen such an error less often than the
   canonical power 0.8, or if its constants cannot be fitted that accurately from these rows.

It reads the development rows and the inputs of the pool, which the route already had. It can
only withhold a report or pass it to another member of the same class.

Replayed on 6,000 of the campaign's worlds in which the route had reported a law
(`benchmarks/vouching_dev_replay.json`): on 3,000 drawn at random, 118 wrong laws became 3 and
1,825 laws became 1,841, with 369 reports withheld. **That replay shows what was built and
proves nothing: the checks were written after reading those worlds.** This run is on laws drawn
afterwards.

## Data and protocol

**Laws.** `benchmarks/counterfeit_laws.json`: 60 laws drawn by
`scripts/run_counterfeit_benchmark.py build` with seed 20261130 from the red team's recipes
(`counterfeit.draw_recipe`: the invented-law grammar, one to three inputs on (1, 5), blocks in
one input each joined left to right by sums, differences, products and quotients, constants from
a list of unround numbers). A draw is left out if its structure is in the invented-laws benchmark
(by fingerprint; no held-back law is read), among the red team's 150 laws, among the 80 laws the
memory proposer was developed on or the 40 it was tested on. 6 laws have one input, 32 two, 22
three.

**Counterfeits.** A counterfeit is the law's recipe with one block exchanged for another kind, or
one summed block left out, and every constant then fitted to the law over the whole domain. Its
*distance* is the root mean square of what still separates it from the law, as a share of the
law's spread. A distance of 0.0316 is R² 0.999, the benchmarks' accuracy line. For each law one
counterfeit is drawn from each band it has: near (0.001 to 0.01), middle (0.01 to 0.0316) and far
(0.05 to 0.3). The first two are accurate by that line; the third is wrong; the gap between
leaves no case on the line. Every law has a far counterfeit; 37 have a near one and 57 a middle
one.

**Cases.** For each law and each noise level (0, 1 % and 5 % of the targets' root mean square,
independent Gaussian), two cases, 360 in all:

- *offered*: the law beside its counterfeits (two or three), in an order drawn with the case.
  The right report is the law.
- *withheld*: no law, and only wrong counterfeits: up to three of the far band (57 laws have
  three, one has two, two have one). Nothing handed over is right. A method that only chooses
  among its candidates is right to report nothing; the route's own stages can still find a law,
  and that counts as found.

Each case is a world of `apedp_dyn/redteam/worlds.py`: 1,000 rows drawn uniformly over the
domain, 70 % for development and 30 % as the pool the path may measure, with the case's own
seed. No search engine runs: the candidates are the search stage's whole output.

**Four methods on every case.**

- `route`: the route in force (`refined_search.discover_refined`), told the noise is absolute,
  as in the campaign;
- `vouched`: the route's report after the checks of `vouching.vouch`, given the law the route
  reported, the candidates the path named as inseparable from it, the development rows and the
  pool's inputs;
- `best_fit`: the candidate with the lowest BIC after refitting on the development rows
  (`redteam/plain_selection.py`). It always reports;
- `validated`: the same choice, reported unless its error on thirty fresh rows of the pool is
  larger than on the development rows by an F test at the canonical level.

**Scoring** is the red team's and through it the benchmarks': 500 further rows without noise,
drawn like the data. A report is *right* if it is the law after refitting constants
(`symbolic_recovery`), *wrong* if it is not the law and below R² 0.999 on those rows, and
otherwise *accurate without being the law*.

**The score** of a method is (right − 19 × wrong) / 180, 180 being the cases in which the law was
offered. Nineteen is (1 − a) / a for the canonical permitted share of false reports a = 0.05
(`REFEREE_MAX_FPR`), so a method whose reports are wrong at exactly the permitted rate scores
nothing. An accurate approximation and silence count nothing either way. 1 is perfect.

A case that crashes or exceeds 600 seconds counts, for every method, as reporting nothing. If the
checks themselves fail on a case, `vouched` counts as reporting nothing there and the failure is
recorded. Case *i* of the file goes to shard *i* mod 12.

## Expectations, fixed now

1. **Where the earlier runs measured, the route's reports can be relied on:** of the laws the
   route reports in the 240 cases with noise up to 1 %, at most 5 % are wrong
   (`REFEREE_MAX_FPR`).
2. **A referee beats no referee:** the route's score is above `best_fit`'s and above
   `validated`'s.
3. **What the checks pass can be relied on at every noise level:** of the laws `vouched`
   reports, at most 5 % are wrong at noise 0, at 1 % and at 5 %, each taken alone.
4. **The checks are worth having:** `vouched` scores above `route`.
5. **They do not buy that with silence:** of the cases in which the route reports the law,
   `vouched` still reports the law in at least 75 % (the canonical share of valid results a
   referee must retain, `REFEREE_MIN_VRR`).
6. Reported without a criterion: every method by noise level and by offered and withheld; the
   reason for every report the checks withheld; how often they passed a report to another member
   of its class and how often that was the law; errors, timeouts and time.

**The system's own forecast** (`preregistration/FORECAST-COUNTERFEIT-2026-11.json`, filed with
this text). It counts how each method ended on the campaign's randomly drawn worlds of the same
noise, sampling and number of rows, and what the replay of the checks made of the route's
reports there, and draws this run from those counts 20,000 times. For a withheld case, which the
campaign has no exact like of, the worlds whose best-fitting candidate was wrong stand in.

| Method | Right | Wrong | Score |
| :--- | ---: | ---: | ---: |
| `route` | 156 (149 to 163) | 31 (24 to 37) | −2.36 (−3.05 to −1.66) |
| `vouched` | 151 (144 to 159) | 1.3 (0 to 3) | 0.70 (0.51 to 0.86) |
| `best_fit` | 175 (172 to 178) | 180 (178 to 181) | −18.01 (−18.15 to −17.81) |
| `validated` | 174 (170 to 177) | 173 (169 to 177) | −17.26 (−17.70 to −16.85) |

Intervals at 80 %. Its chance that each expectation holds: the first, second, fourth and fifth
above 99.9 %; **the third 81 %**, all of the doubt being at 5 % noise. The intervals cover the
draw of sixty cases a cell and what the counts leave open, and not the difference between the
campaign's laws and these.

**The author's own expectations**, recorded so that they can be wrong, and no criterion.

- The first holds, about nine chances in ten: at noise up to 1 % the route reports no wrong law
  or one. The risk is a law whose spread is small beside its size, where 1 % noise is as large
  as a far counterfeit's error.
- The second holds. `best_fit` must be wrong in all 180 withheld cases.
- The route reports more wrong laws than the forecast says, about 38, nearly all at 5 % noise.
  The worlds that stand in for withheld cases often held a candidate that was accurate, and
  these cases hold none, so reports that were harmless there are wrong here.
- **The third is the open one: about two chances in three.** At 5 % noise `vouched` reports
  about forty-five laws, so three wrong ones break it. The checks were fitted to the campaign's
  worlds and will do somewhat worse on new laws than the replay shows; the forecast, which takes
  the replay at its word, is too confident.
- The fourth holds unless the route itself reports almost no wrong law.
- The fifth holds, about nine chances in ten: nearly every right law is kept at noise up to 1 %,
  and about three in four at 5 %.

A miss is reported as a miss.

## What follows from the result, fixed now

- **If 3, 4 and 5 hold,** the checks become the last step of the route in force for tables, in a
  new module around the frozen ones, with a switch that turns them off. A law that fails them is
  shown as not vouched for, with the reason, and is not reported. Stated wherever that route is
  described: measured on candidates handed over, uniform rows and Gaussian noise up to 5 %; not
  measured behind a search engine or on the physics benchmark, where what they cost in recovered
  laws is unknown until a run says.
- **If 3 misses at any noise level,** the checks are not adopted. Every wrong law they passed is
  listed with the rule that let it through, and no variant is called measured until it has been
  run on laws drawn after that variant was written.
- **If 3 holds and 4 misses,** they are not adopted, and the result is recorded as: on laws of
  this kind the checks stop nothing the route does not stop.
- **If 3 and 4 hold and 5 misses,** they are not the default. They stay selectable, described
  with the share of right laws they cost here.
- **1 and 2 are recorded as they fall.** If 1 misses, the statement that the route's reports
  hold at noise up to 1 % is corrected wherever it is made, with this run's number beside it. If
  2 misses, this benchmark does not separate a referee from a plain method and is reported so.
- The forecast is scored whatever else happens: totals against their intervals, and the chance
  it gave each expectation against what happened.

## Disclosed before freezing

- **This is the benchmark's second design.** The first withheld the law and left every
  counterfeit in. Each of its 60 laws had a counterfeit near enough to count as accurate, so
  reporting the best fit was never wrong and it could not have told a careful method from a
  careless one. It was found by reading the file, before any run. Four cases of the first
  design's first law were run once, as a smoke test on the machine it was written on. That law's
  structure is not in this file; one of this file's sixty structures also occurs in the first
  design, with other constants, and was never run. **No method has been run on any case of this
  file.** The smoke test now uses a law that is in no benchmark.
- **The checks were built on worlds of the same making.** The campaign's 150 laws come from the
  same recipes and its counterfeits from the same exchange of blocks. None of this run's
  structures is among them. A pass here says the checks hold against counterfeits of this making
  on laws of this grammar. It says nothing about wrong candidates of other kinds: those a search
  engine returns, those that differ from the law only outside the sampled domain, or those that
  use an input the law does not.
- **Nothing has searched for data that defeats the checks.** The campaign aimed at the route
  without them. Randomly drawn cases are a weaker test than a search.
- **The checks' settings were chosen on the campaign's worlds.** Level and power are the
  canonical ones. The three neighbourhood sizes, 2,047 sign patterns and 32 planted trials were
  set while probing a sample of those worlds, before the replay on file, which was run with them
  once and a second time to record one more field, with identical counts.
- **The far band starts at 0.05, not at the accuracy line.** A counterfeit between 0.0316 and
  0.05 is wrong by the line and is in no case. The benchmark does not measure what a method does
  with a candidate that is only just wrong.
- **Reporting a near counterfeit costs nothing.** By the benchmarks' line an approximation
  inside R² 0.999 is not an error, so a method is not asked here to tell the law from its
  nearest counterfeits, only to prefer it.
- **5 % noise is outside what the earlier runs measured,** and the campaign already showed the
  route failing there. A poor score of the route at that level is expected and is not news; the
  first expectation is about noise up to 1 %.
- **A wrong law is judged where the data were:** on rows drawn like the development rows. What
  a law does outside the domain is not tested.
- **One draw of rows per case.** A cell is sixty cases; what that leaves to chance is what the
  forecast's intervals show.
- **The route keeps its own stages.** Its first pass can find a law by itself, from units,
  polynomials or pruning, with no candidate's help. In a withheld case such a law counts as
  right.
- **Both the campaign and this run use the reference first pass,** not the native kernel.
- **Unequal cost.** The checks took a third of a second a report on average in the replay, and
  at most two seconds.
- **The forecast's rows for the checks are few in places:** between 6 and 208 replayed reports a
  cell. Its shares are drawn with that uncertainty, and with the replay's shares over all its
  worlds as a prior worth two reports.

## Deviation rule

The run is not repeated. A shard lost to infrastructure is re-run alone and recorded with its
date. Any change to a frozen file after shards have started makes the result a development result.

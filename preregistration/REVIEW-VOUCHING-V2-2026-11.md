# Review of RESULT-VOUCHING-V2-2026-11

**The owner's decision, 2026-11-25: the second version of the last checks is kept.** Asked for an
explicit yes or no, with the limits below in front of them (tested up to 5 % noise, on tables of
1,000 rows, with traps from the development generator; a law with a pole can still pass), the owner
said to keep it. It is in force by that decision from this date, and no longer only by a consequence
an agent applied. `APEDP_TABLE_VOUCHING=off` still gives the route without it. Finding 1 below is
closed by this; the other findings stand as descriptions of what the result does and does not show.

**An independent review was completed on 2026-11-24** (the section after this one). Its verdict:
every number in the record matches the raw rows and all five verdicts are right by the plan's
definitions; what is wrong is what the record claims, stresses and leaves out. The earlier
sections below, written before it, are kept as they were.

## The independent review, completed 2026-11-24

A separate agent, started with no knowledge of how the work was done, recomputed every number of
the record from the raw rows and judged each expectation against the plan's own definition. It
changed nothing. The record itself is still not edited; this section is what a reader of the
record should know beside it.

**What it confirmed.** The three raw files match `SHA256SUMS`; the 23 frozen files match the
digest, none edited; every raw row matches a case or law of the frozen laws file. Every cell of
both tables and every count in the prose matches (handed: 231, 188 and 189 reports with 19, 1 and
0 wrong; searched: 75, 52 and 61 reports with 31, 27 and 30 recovered). All five expectations
were read with the plan's denominators and pooling, and all five held.

**What it found, most serious first.**

1. **The consequence was applied without the owner's word.** The record says the second version
   "becomes the last step of the route in force". AGENTS.md 1.5 says a consequence fixed before
   a run only proposes. The switch went in with the record, minutes after the run. It should
   read: the result proposes adoption; the agent put it in force without the owner's word;
   `APEDP_TABLE_VOUCHING=off` undoes it. **The owner has still not been asked for an explicit yes
   or no; that question is open.**
2. **The handed part is inside the development distribution and the record does not say so.**
   The laws are new (none of the 100 structures had been used), but the traps come from the
   function that made the development benchmark, with the same bands, rows and noise levels and
   a new seed: 316 of the 341 counterfeits use a kind of exchange that also occurs there. No arm
   was aimed at the frozen second version. The searched part had only 11 inaccurate reports to
   stop, all at 5 % noise.
3. **A limitation the plan states is dropped.** The record gives the 70 wrong laws passed above
   5 % noise on the second campaign's worlds, and not the 98 passed at up to 5 % there (98 of
   1,109 reports; 79 of them on tables of 60 to 299 rows). The record's list of conditions never
   says that every table of this test had 1,000 rows.
4. **The fourth expectation held with no margin and the summary does not say so.** 26 of 27 at
   noise up to 1 % (25 would miss) and 3 of 4 at 5 %, which is exactly the limit. At 1 % alone it
   is 7 of 8. One 5 % run crashed on a law the route recovered at the other two noise levels;
   had it run and been recovered, 4 of 5 would have been needed.
5. **The intervals are flattered.** The 189 and 61 reports come from 60 and 29 laws and are not
   independent, and most sit where there was nothing to stop. At 5 % noise behind the search, no
   inaccurate law in 17 reports leaves an upper limit of 19.5 %, not 5.9 %. What the second
   version actually stopped: 19 of 19 wrong laws handed (lower limit 82 %), 11 of 11 inaccurate
   ones behind the search (lower limit 72 %).
6. **"None wrong" against the first version's "one wrong" turns on the accuracy line.** The first
   version's wrong law has R² 0.998999; the second version's nearest report that is not the law
   has R² 0.999071. Five of the second version's reports that are not the law are below 0.9995.
7. **No independent review before recording.** True when the record was written; this section is
   that review, after the fact.
8. **Part of what the plan promised to report is absent:** handed results by noise level (table
   below), the reasons for the first version's eight withheld right laws (seven the power rule,
   one two candidates differing), handed errors and timeouts (none), and the checks' own time,
   which the raw rows do not carry.
9. **"Can be relied on" means accurate, not the law.** 42 of the second version's 189 handed
   reports and 31 of its 61 searched reports are accurate approximations. At 5 % noise behind the
   search, 3 of 17 are the law.
10. **Wording.** The crashed run failed in the route's own candidate fit, not "in the search".
    "No search engine" in the handed part: the route's first pass still proposes forms, and found
    the law in six cases where it was withheld.
11. **A disclosure of the plan is dropped:** the searched laws come from the grammar the memory's
    way of joining parts was written for.

Handed part by noise level, which the record omits (reports / right / wrong):

| Method | Noise 0 | 1 % | 5 % |
| :--- | :--- | :--- | :--- |
| `route` | 62 / 62 / 0 | 77 / 48 / 1 | 92 / 30 / 18 |
| `first` | 62 / 62 / 0 | 67 / 49 / 0 | 59 / 31 / 1 |
| `second` | 62 / 62 / 0 | 67 / 49 / 0 | 60 / 36 / 0 |

**What it could not check:** the run's identity (run id, times, that no shard was lost), since no
CI log is in the repository; that the laws file is what its generator gives; that the laws were
drawn after the module's last change and never run before freezing (module, laws and plan are in
one commit); that only one seed was tried.

**What follows.** The numbers stand. The claim to make from them is narrower than the record's:
on 1,000-row tables with Gaussian noise up to 5 %, with traps of the kinds it was developed
against, the second version stopped every wrong law the route passed and cost one or two right
ones; one of its five expectations held at exactly its limit; it has not been tested against
traps aimed at it, on fewer rows, or above 5 % noise, where it is known to be weaker.

---

*The sections below were written on 2026-11-23, before the review above could be completed.*

[`RESULT-VOUCHING-V2-2026-11.md`](RESULT-VOUCHING-V2-2026-11.md) was written by the agent that built
the second version of the last checks, wrote its plan and ran it. AGENTS.md 1.6, added after the
record, requires a reviewer who took no part in the work to check a record against its raw outputs
before it is recorded. The record itself is not edited; this file says what has been checked since.

## The independent review did not finish

On 2026-11-23 a separate agent was started with no knowledge of how the work was done. It was asked
to recompute every number in the record from the raw rows and to judge whether the expectations were
read as the plan defines them. It stopped on a usage limit before it reported anything. Nothing it
did is used here. It can be run again after the limit resets on 2026-11-27. **Until then, the record
has not been independently reviewed.**

## What was checked mechanically, by the same agent

- The three output files match `SHA256SUMS`.
- The 23 frozen files match `DIGEST-VOUCHING-V2.json` on disk, and `results.json` recorded that same
  map.
- `tests/test_result_vouching_v2_numbers.py` recomputes from `handed.jsonl.gz` and
  `searched.jsonl.gz` and requires each number to be in the record. It covers every count in both
  tables: reports, right or recovered, wrong or inaccurate, the 5 % split, the score, and the counts
  by noise. It also covers the kept counts of the third and fourth expectations. All match.

This is the mechanical part of a review. It cannot catch a wrong reading of the plan, an overstated
sentence, or a missing disclosure; only a reader who did not write it can.

## What rereading the record against the rules written afterwards shows

1. **The consequence was applied without the owner's word.** The record says the second version
   became the last step of the route in force "by the rule fixed before the run". The owner had
   taken the first version out the same day. Under AGENTS.md 1.5 such a consequence only proposes.
   The second version stays in force until the owner says otherwise; `APEDP_TABLE_VOUCHING=off`
   undoes it.
2. **The handed part measures performance inside the development distribution.** Its counterfeits
   come from the same generator, PREREG-COUNTERFEIT's, that the second version was replayed on while
   it was being changed. The laws are new, but the traps are of the kinds it was tuned against. The
   record discloses that the generator was used in development; it does not draw this conclusion.
   The searched part, behind a real search, is not affected. Neither part includes traps from a
   red-team search aimed at the frozen check (AGENTS.md 1.6). The next blind test of a check will.
3. **"Recovered" and "accurate" should be read together.** At 1 % noise behind the search, 8 of the
   second version's 24 reports were the law. The other 16 were accurate approximations. The record
   gives these numbers, but its summary leads with "no inaccurate law". The status page and the UI
   now state both together.

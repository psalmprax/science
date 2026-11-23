# Review of RESULT-VOUCHING-V2-2026-11, as far as it could be done

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

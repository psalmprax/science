# RESULT-VOUCHING-V2-2026-11: the second version of the last checks, on 100 laws drawn after it

Preregistration: [`PREREG-VOUCHING-V2-2026-11.md`](PREREG-VOUCHING-V2-2026-11.md), frozen by
[`DIGEST-VOUCHING-V2.json`](DIGEST-VOUCHING-V2.json) in commit `852ea3e`. Run once on GitHub
Actions, workflow `Last checks, second version, blind`, run 45114039485, 2026-11-23 04:48 to
06:07 UTC, on branch `vouching-v2-run-1` at `852ea3e`. Twelve handed shards and 24 searched
shards; none was lost or re-run. The 23 frozen files' digests recorded by the collect step match
the digest file. Outputs: [`RESULT-VOUCHING-V2-2026-11/`](RESULT-VOUCHING-V2-2026-11/)
(`results.json`, `handed.jsonl.gz`, `searched.jsonl.gz`, `SHA256SUMS`, the same as the collect
step printed).

**All five expectations held.** By the rule fixed before the run, the second version becomes the
last step of the route in force for tables. It was tested up to 5 % noise only, and above that it
is known to be weaker than the first version (audit M-36). M-34 stays open.

## Handed counterfeits: 60 laws, 360 cases, no search engine

| Method | Reports | Right | Wrong | Wrong at 5 % noise | Score |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `route` (the route in force) | 231 | 140 | 19 | 18 of 92 | −1.23 |
| `first` (first version of the checks) | 188 | 142 | 1 | 1 of 59 | 0.68 |
| **`second`** | **189** | **147** | **0** | **0 of 60** | **0.82** |
| best fit | 360 | 176 | 177 | 59 of 120 | −17.71 |
| best fit confirmed on fresh rows | 347 | 173 | 167 | 56 of 117 | −16.67 |

The score is (right − 19 × wrong) / 180, as in PREREG-COUNTERFEIT.

1. **Held.** The second version's wrong share is 0 at noise 0 (62 reports), 0 at 1 % (67) and 0 at
   5 % (60); the limit was 5 % at each. With no wrong law in 189 reports the 95 % interval reaches
   1.9 %, and 6.0 % for the 60 reports at 5 % noise alone.
2. **Held.** Its score is 0.82 against the route's −1.23 (and the first version's 0.68).
3. **Held.** Of the cases in which the route reported the law, the second version reported it in 109
   of 110 at noise up to 1 % (limit 95 %) and in 29 of 30 at 5 % (limit 75 %). The first version:
   109 of 110 and 27 of 30.

The second version stopped all 19 wrong laws the route reported. In 13 cases it reported another
member of the law's class instead of the route's choice, 9 of them the law. Two right laws were
lost. In one (law 8049, 1 % noise) a test refused the law, a false refusal. In the other (8012,
5 % noise) the power for an error that would make it wrong was 72 %, below the 80 % required. The
first version passed one wrong law (8032 at 5 % noise, the route's own form, wrong by a small
margin) and lost four right ones.

## Behind the search stage in force: 40 laws, 120 runs

| Arm | Reports | Recovered | Inaccurate | at noise 0 | at 1 % | at 5 % |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `route` | 75 | 31 | **11** | 20 reports, 19 recovered | 24, 8 | **31, 4, 11 inaccurate** |
| `first` | 52 | 27 | 0 | 20, 19 | 22, 7 | 10, 1 |
| **`second`** | **61** | **30** | **0** | 20, 19 | 24, 8 | 17, 3 |

4. **Held.** Of the runs in which the route recovered the law, the second version recovered it in 26
   of 27 at noise up to 1 % and in **3 of 4** at 5 %. Three of four is exactly the limit of 75 %, and
   four runs say little. At 1 % (law 8511) a test refused the law, and the second version reported
   another member of its class, accurate (test R² 0.99998) but not the law. At 5 % (8510) two
   candidates passed every test and differ by more than the accuracy claimed, so it reported
   nothing. In one run (8532 at 1 %) it reported the law where the route had not. The first version
   kept 26 of 27 and 1 of 4.
5. **Held.** None of the second version's 61 reports is inaccurate (the 95 % interval reaches 5.9 %).

**Something this run measured for the first time.** At 5 % noise the search stage in force
reported 31 laws, and 11 of them are inaccurate (test R² below 0.999). The preregistration
disclosed that this noise level had not been run behind a search. The second version stopped all
11. In one of them it reported another member of the class instead, which is accurate.

One run (8504 at 5 % noise) crashed in the search, before any checks (`LinAlgError: SVD did not
converge in Linear Least Squares`), and counts as reporting nothing in every arm. No run timed
out. The search stage was given 240 seconds a search and took a median 349 seconds a run
(267–581), within the 1,500 seconds allowed.

## What this does and does not show

- The 100 laws were drawn after the module's last change, and no structure among them had been
  used before. Both benchmarks and both red-team campaigns were development material for the
  module, and it was changed while being replayed on them. The preregistration disclosed the
  three changes.
- The noise was Gaussian, the rows uniform, the tables up to three inputs, the noise up to 5 %.
  Above 5 % noise, on the second red-team campaign's worlds, the second version let 70 wrong laws
  through where the first let 29 (audit M-36). That was not tested here and is not repaired.
- A law with a pole in the domain can pass (audit M-34, not repaired).
- The searched part's gain is small in numbers: at noise up to 1 % the route's reports already
  held (none of 44 inaccurate), and the checks cost one recovered law in 27 there. What they are
  for is 5 % noise, where 11 of the route's 31 reports were inaccurate and none of the second
  version's 17 was.
- **The laws file did not come out byte for byte** when the generator was run again on GitHub's
  runner (Python 3.12 there, 3.13 where it was written), as the preregistration said could happen.
  The run read the file, whose digest is frozen.

## What follows, as fixed before the run

All five held, so the second version is the last step of the route in force for tables
(`apedp_dyn/discovery_l2/vouched_route_v2.py`, called by the UI's table reading), in a new module
around the frozen ones. Its cost and benefit are stated with the route wherever it is described.
M-34 and M-36 are stated as open. `APEDP_TABLE_VOUCHING=off` gives the route without last checks;
`=first` (or `=on`, as before) gives the first version. The first version's numbers are recorded
above beside the second's.

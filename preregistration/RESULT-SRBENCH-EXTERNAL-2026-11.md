# Result: PREREG-SRBENCH-EXTERNAL-2026-11

**Run:** once, 2026-11-19 07:50–11:15 UTC, GitHub Actions run `44482717272` on the branch
`srbench-external-run-1` at commit `bf8f66e`, which froze the preregistration and the harness (digests
repeated in `results.json`). One smoke job, 32 shards and one collect job, all successful on the first
attempt; no shard was re-run.

Outputs, unedited, in [`RESULT-SRBENCH-EXTERNAL-2026-11/`](RESULT-SRBENCH-EXTERNAL-2026-11/):
`results.json`, `runs.jsonl` (798 runs) and the 32 shard files. As for the baseline, the files were
printed to a job log by a throwaway workflow (run `44519214650`, branch `srbench-fetch-results-2`, not
merged) and rebuilt; all 34 match the SHA-256 the job computed before printing.

## Outcome

133 problems × 2 noise levels (0 and 1 %) × 3 engines. The reference is the best baseline engine in
[`RESULT-SRBENCH-BASELINE-2026-11`](RESULT-SRBENCH-BASELINE-2026-11.md): L2, 24 of 119 Feynman
problems at both noise levels.

Symbolic recovery on the Feynman problems (of 119):

| Engine | Noise 0 | 95 % interval | Noise 1 % | 95 % interval |
| :--- | ---: | ---: | ---: | ---: |
| PySR | **73** (61.3 %) | 52.0–70.1 % | **68** (57.1 %) | 47.7–66.2 % |
| PhySO | 41 (34.5 %) | 26.0–43.7 % | 40 (33.6 %) | 25.2–42.8 % |
| Exhaustive search with MDL | 31 (26.1 %) | 18.4–34.9 % | 31 (26.1 %) | 18.4–34.9 % |
| L2 (baseline run) | 24 (20.2 %) | | 24 (20.2 %) | |

Symbolic recovery on the Strogatz problems (of 14):

| Engine | Noise 0 | Noise 1 % |
| :--- | ---: | ---: |
| PySR | 10 | 9 |
| PhySO | 1 | 2 |
| Exhaustive search with MDL | 3 | 3 |
| L2 (baseline run) | 3 | 3 |

All 133 problems, both noise levels pooled (266 runs per engine):

| Engine | Recovered | 95 % interval | Accuracy solutions | Errors | Timeouts | Median seconds |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| PySR | 160 (60.2 %) | 54.0–66.1 % | 238 | 0 | 0 | 250.5 |
| PhySO | 84 (31.6 %) | 26.0–37.5 % | 103 | 3 | 0 | 244.6 |
| Exhaustive search with MDL | 68 (25.6 %) | 20.4–31.2 % | 90 | 0 | 0 | 9.2 |

## Against the expectations

1. **An outside engine beats every baseline engine on Feynman at each noise level: held.** PySR 73
   and 68 against L2's 24 and 24.
2. **The exhaustive search beats the baseline engines too: held.** 31 and 31 against 24 and 24.
3. Without a criterion: the exhaustive search does not beat PySR (31 against 73 and 68) or PhySO
   (31 against 41 and 40).

## What this means

- PySR recovers three times as many Feynman laws as the best in-house engine, and it holds up at 1 %
  noise.
- The in-house exhaustive search is ahead of L2, behind both outside engines, and by far the cheapest
  per run (median 9 s against about 250 s).
- The budgets differ and that favours the outside engines: they had 240 s per run, the baseline
  engines 120 s. L2 finished in a median of 7 s on Feynman and timed out on 13 problems per level, so
  more time could add at most 13 to its 24. That does not close the gap to PySR, but the comparison is
  not at equal compute.

## After the run (not preregistered)

Computed from the two `runs.jsonl` files after seeing the results. Feynman problems only.

- **PySR's wins contain L2's.** At noise 0 PySR recovers 49 problems L2 does not, and L2 recovers
  none that PySR does not. At 1 % it is 44 against 0.
- **The exhaustive search's margin over L2 is small but one-sided.** It recovers 7 problems L2 does
  not, and L2 none it does not, at both levels (paired sign test p = 0.016). Its interval overlaps
  L2's rate, so the size of the gain is not well measured.
- **The in-house engines add almost nothing to PySR.** At noise 0, L2 and the exhaustive search
  together recover one problem PySR misses; at 1 % none. Against L2, PhySO recovers 19 and 18 problems
  L2 does not; L2 recovers 2 that PhySO does not at each level.
- **Half the benchmark is still unsolved by everything.** All four engines together recover 75 of 119
  at noise 0 and 69 at 1 %.
- **PhySO's three errors** are in its own expression handling (`ZeroDivisionError` on
  `feynman_III_15_27` at noise 0; `KeyError: 'ComplexInfinity'` on `feynman_test_6` at both levels).
  They are counted as not recovered.

## Consequences recorded elsewhere

- Audit entry M-11 in `docs/SYSTEM_FLAWS_AUDIT.md`: on the shared benchmark the in-house search is
  dominated by PySR. Which in-house engines to retire is the owner's decision and is not made here.

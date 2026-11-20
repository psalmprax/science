# Preregistration: the repository's search engines on SRBench's ground-truth track

**Document ID:** PREREG-SRBENCH-BASELINE-2026-11
**Frozen:** at the commit that adds this file. `preregistration/DIGEST-SRBENCH-BASELINE.json` holds the
SHA-256 of this document, `scripts/run_srbench_baseline.py`, `scripts/fetch_srbench_ground_truth.py`
and `benchmarks/srbench/manifest.json`; the harness refuses to run if any of them changed.
**Author:** Claude (session `01AmWYSGATN2jrEj786Sttks`), at the repository owner's request.

## Why

Every engine in the repository has so far been scored on problems written in the repository. This
run puts them on a benchmark nobody here designed: SRBench's ground-truth track, the 119 Feynman
equations and 14 Strogatz ODE right-hand sides that the symbolic-regression community reports on.
It is the baseline that later engines (PySR, PhySO, ESR, weak SINDy, an LLM proposer) will be
compared against, under this same protocol.

## Problems and data

The 133 datasets come from PMLB, which distributes SRBench's ground-truth problems. Each file is
pinned by the SHA-256 its git-LFS pointer records (`benchmarks/srbench/manifest.json`); a download
that does not match is refused. Before freezing, every one of the 133 formulas was checked to
reproduce its file's target on 200 rows (largest relative error below 1e-6 in all 133), and every
Feynman formula was checked for dimensional consistency under the stated units: 118 of 119 are
consistent. `feynman_I_43_16` is not (its metadata gives the target units of a velocity while the
formula's units differ); it stays in the benchmark, and L2, which requires consistent units, is
expected to fail it.

## Protocol

| | |
| :--- | :--- |
| Engines | **L2** (`L2DiscoveryEngine`, defaults, units supplied; its referee runs as part of it), **RIME** (`RimeProposer(static_mode=True)`, defaults otherwise), **RIME-AD** (`RimeAdProposer`, defaults), **extended hierarchical search** (`HierarchicalSearch(template_set="extended")`, budget 120) |
| Rows | 1,000 training and 500 test rows, drawn without replacement and seeded by the problem name; a problem with fewer than 1,500 rows (the Strogatz sets) is split 75/25 |
| Noise | Gaussian on the training target only, standard deviation `noise × RMS(y_train)`, at 0, 0.1 %, 1 % and 10 %; test targets are clean |
| Variables | renamed `x0 … xk` for every engine, so no engine sees a variable's name |
| Time | one process per run, killed after 120 s (counted as a timeout) |
| Seeds | one (0) |
| Runs | 133 problems × 4 noise levels × 4 engines = 2,128 |

## Scoring

- **Symbolic recovery.** Every numeric constant in the engine's returned model except exponents, plus
  an outer scale and offset, is refitted by least squares to the clean truth on the test rows. The
  model is recovered if the refit leaves a relative RMS error of at most 1e-8 and no non-constant
  additive term was refitted to nothing. This is SRBench's question (the right law up to its
  constants), answered numerically rather than by algebraic simplification; a superset of the truth
  does not pass by zeroing its extra terms, and `x**1.02` is not `x`.
- **Accuracy solution.** Test R² of the engine's own predictions at least 0.999.
- **L2's referee.** L2 adjudicates its candidates. Each L2 run is also scored as a promotion of a
  recovered law, a promotion of anything else (a false promotion), or no promotion.
- Rates are reported per engine, family and noise level with 95 % Clopper–Pearson intervals, with
  timeouts and errors counted, never dropped.

## Expectations, fixed now

1. **The referee rarely promotes a wrong law:** false promotions are at most 5 % of L2's promotions,
   pooled over all noise levels and both families.
2. Recovery rates, accuracy rates and the ordering of the engines are reported without a criterion.
   This is a baseline; the point is the number, and that later engines meet the same protocol.

## Disclosed before freezing

The harness was smoke-tested on four of the 133 problems (`feynman_I_12_1`, `feynman_II_6_15a`,
`feynman_I_6_2a`, `strogatz_bacres1`) at 1 % noise with all four engines, and the outputs were seen.
In that test L2's referee promoted a wrong law on `feynman_II_6_15a` (test R² 0.81). Those problems
are not removed or treated differently; the protocol was not changed after seeing them except for
two scoring fixes made before freezing: RIME-AD's intercept token `const` is parsed as 1, and a model
may not pass the symbolic check by refitting an extra term to zero.

## Deviation rule

The run happens once, on GitHub Actions (`.github/workflows/srbench-baseline.yml`). If it fails for
infrastructure reasons, the fix is committed as a dated deviation in this directory before it is
run again. Changing an engine's configuration, the scoring or an expectation after the run is not a
deviation; it is a new preregistration.

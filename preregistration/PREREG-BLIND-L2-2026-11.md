# Preregistration: does the open discovery engine find laws nobody could have memorised?

**Document ID:** PREREG-BLIND-L2-2026-11
**Frozen:** at the commit that adds this file; `preregistration/DIGEST-BLIND-L2.json` holds the
SHA-256 of this document and of `scripts/run_blind_discovery_benchmark.py`, and the script refuses
to run if either has changed.
**Author:** Claude (session `01AmWYSGATN2jrEj786Sttks`), at the repository owner's request.

## Why

The two earlier preregistrations measured the referee: whether it refuses what it should. This one
measures the other half, the discoverer. `L2DiscoveryEngine` receives raw data, variable names, SI
dimensions and the target's dimension, builds dimensionally typed expressions without templates,
fits them, and hands survivors to the referee. Nothing has measured how often it promotes the right
law, the wrong law, or nothing.

The second question is the one the plan's correction 4 raised. A benchmark of textbook laws cannot
tell discovery from recall in a proposer that has read the literature. This engine has no such
proposer, so the textbook and invented families should be equally hard for it; measuring both now
gives the baseline any future learned proposer must be compared against, on laws it cannot have
seen.

## Design

| | |
| :--- | :--- |
| Engine | `L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow"))`, defaults otherwise (depth 2, complexity 6) |
| Known laws | 12 textbook laws within that grammar: F = m·a, p = m·v, E = ½m·v², P = F·v, W = F·d, ρ = m/V, p = F/A, V = I·R, ω = v/r, a = v²/r, E = m·g·h, T = 2π√(L/g) |
| Invented laws | 24 from a seeded generator (seed 20261119): two-character meaningless names, random SI dimensions, in rotation c·x^p, c·x^p·y^q and c₁·x^p + c₂·y^q, powers from the engine's own set, coefficients uniform on [0.5, 3] |
| Each run | the law's variables plus one irrelevant variable `z_other` with a different dimension; 120 samples, every variable uniform on [1, 10]; multiplicative noise σ ∈ {0, 1 %, 5 %, 10 %}; seeds 0 and 1 |
| Runs | 36 laws × 4 noise levels × 2 seeds = 288 |

**Outcomes.** *Recovered*: the engine promotes (`UNIQUE_SUPPORT`) a law whose additive terms,
numeric coefficients removed, are exactly the true law's. *False discovery*: it promotes anything
else. *Abstention*: it promotes nothing. Rates are reported per family and noise level with 95 %
Clopper–Pearson intervals.

Nothing about the engine, its thresholds or the referee is changed for this run.

## Expectations, fixed now

1. **False discoveries are rare:** at noise ≤ 5 %, over both families (216 runs), the false-discovery
   rate is at most 5 %.
2. **The engine recovers clean laws:** at noise 0 and 1 %, recovery is at least 80 % in each family.
3. **No familiarity gap:** at each noise level ≤ 5 %, recovery on known and invented laws differs by
   at most 15 percentage points. A larger gap would mean the invented laws are harder in some way
   the design did not intend, and is reported as that.
4. Recovery at 10 % noise is reported without a criterion.

A miss on 1, 2 or 3 is reported as a property of the engine, not explained away. Outputs —
`results.json` and `runs.jsonl` with every run's truth, promoted law, status and timing — are
committed unedited with `preregistration/RESULT-BLIND-L2-2026-11.md`.

## Deviation rule

The run happens once. If the script crashes, the fix is committed as a dated deviation in this
directory before re-running. Changing a law, a family, a noise level or an expectation after the run
is not a deviation; it is a new preregistration.

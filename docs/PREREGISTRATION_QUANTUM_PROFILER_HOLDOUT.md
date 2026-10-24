# Preregistration: Quantum Profiler & Coordinate Transmuter — Holdout Confirmation

**Document ID:** PREREG-QPROF-2026-10-17
**Status:** DESIGN FROZEN at commit time; confirmation harness and artifacts committed SEPARATELY after.
**Relation:** the quantum discovery line
(`apedp_dyn/epistemic/autonomous_profiler.py`, `apedp_dyn/epistemic/coordinate_transmuter.py`,
landed via `benchmarks/run_landauer_rime_vs_ga_benchmark.py`) reports **in-sample fits on the
discovery partition only** — stated in its own scope discipline — and the Landauer benchmark ran
without a prereg. This document registers the first out-of-sample confirmation protocol for this
line.
**Disclosure:** all reported numbers in `docs/landauer_conductance_staircase_rime_vs_ga_benchmark.md`
and `docs/autonomous_epistemic_discovery_and_frontier_quantum_benchmark.md` were visible at freeze
time and are **exploratory**; the point estimates below informed only the margins, never the
criteria direction. No confirmation run under this protocol has been executed at freeze time.
**Deviation rule:** changing any seed, split, world or tolerance after freeze means the
confirmation is re-run under a NEW prereg — never quietly patched.

## Frozen worlds and splits (fresh noise seeds {941, 942, 943})

For every world: the discovery partition is used for any fitting or profile-statistic estimation;
the recovered formula is then **frozen** and evaluated on the untouched holdout partition. No
refit on holdout. Low-level noise is added with the frozen scales below (fresh per seed via
`numpy.random.default_rng(seed)`), so confirmation is not a noiseless replay.

### Q1 — Brus quantum dot (power law)

`y = 4.5 · R^(−2)`; discovery `R ∈ [1.5, 4.5]`; holdout `R ∈ (4.5, 8.0]`, `N_holdout = 25`.
Multiplicative log-space noise, relative `σ = 1e-3`.
Criteria: profiler flags pure power law; recovered exponent within `0.02` of `−2.0`; recovered
`C` within `1%` of `4.5`; frozen-formula holdout `R² > 0.999`.

### Q2 — Landauer conductance staircase (quantized staircase)

`G = 2.0 · floor(V_g)`; discovery `V_g ∈ [0.1, 4.9]` (`N = 100`); holdout `V_g ∈ [5.0, 9.8]`
(`N = 100`). Additive noise `σ = 2e-3` (= 0.1% of the unit step).
Criteria: profiler flags quantized staircase (flat-fraction trigger); `floor`-family transmutation
selected; recovered step height within `5e-3` of `2.0`; frozen-formula holdout `R² > 0.99`.

### Q3 — Schwinger tunneling (singular Arrhenius family)

`y = 10 · E² · exp(−1/E)`; discovery `E ∈ [0.15, 0.70]`; holdout `E ∈ (0.70, 1.20]`, `N_holdout = 25`.
Multiplicative log-space noise, relative `σ = 1e-3`.
Criteria: `log_linear_singular` transmutation selected; `C` within `1%` of `10`; `k` within `0.05`
of `2`; barrier `c` within `0.01` of `1`; frozen-formula holdout `R² > 0.999`.

### G1 — Smooth-polynomial control (must be refused)

On the exact Q2 split: a degree-5 polynomial fit on the discovery partition must be **refused**
by the epistemic referee (per the frozen gate semantics used in the exploratory Landauer audit).
Criteria: refusal on all three seeds; recorded reason (extrapolation collapse and/or residual
whiteness). The guard exists to prevent a smooth surrogate from being promoted on a staircase.

### G2 — Null control (must not validate a recovery)

Pure-noise world (white noise, same shape as Q1): the profiler/transmuter must not emit a
validated recovery — i.e., no frozen-formula holdout `R² > 0.99` may be reported as a confirmed
model. A refusal or "no family selected" outcome satisfies the guard and is recorded.

## Pass criteria

Q1–Q3 criteria pass on ALL three seeds AND G1 refuses on ALL seeds AND G2 does not validate a
recovery on any seed. ANY failure ⇒ the quantum profiler/transmuter claims FAIL this prereg and
are reported as a negative result; a v2 prereg is required before any modification of thresholds,
families or splits.

## Metrics and artifacts (planned, committed after the run)

- `benchmarks/run_quantum_profiler_holdout_confirmation.py` — generates the worlds from the frozen
  recipes, runs profiler → transmuter → frozen-formula holdout evaluation and the referee gates.
- `benchmarks/quantum_profiler_holdout_results.json` — per-(world, seed) rows: profile flags,
  selected family, recovered parameters, holdout R², guard outcomes, determinism digests.
- `docs/QUANTUM_PROFILER_HOLDOUT_REPORT.md` — per-criterion verdict tables with raw numbers.

## Non-goals

No claims about real experimental measurements (real Landauer/IQHE/PCS data would require a
separate data-integrity prereg with device-level provenance); no physical-validity claims beyond
the declared synthetic functional forms; the exploratory in-sample numbers are not retroactively
registered as confirmatory.

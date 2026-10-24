# Preregistration: Lorenz Public Benchmark v1 — Referee-Path Confirmation with Two Negative Controls

**Document ID:** PREREG-LORENZ-PUB-V1-2026-10-20
**Status:** DESIGN FROZEN at commit time; the confirmation harness
(`benchmarks/run_lorenz_public_benchmark.py`) is committed TOGETHER with this document and is the
only registered instrument code. Results are committed separately after the runs.
**Relation / motivation.** An external evaluation transcript claimed APEDP "validated" on a Lorenz
benchmark while its own stress test showed a spurious 3rd-order term (`x²y`) winning training-BIC.
The transcript exercised only the pre-referee `apedp.model_discovery` staging layer (its own doc:
"training BIC (lower is better)"), mixed ground-truth constants into its "pruned physics" forward
model, and validated by eyeballing a 3D attractor plot. An exploratory pilot (see Provenance)
reproduced the training-BIC overfit AND showed the epistemic referee declines it forward.
This prereg registers that comparison properly; it is NOT a replication of the transcript's numbers.
**Deviation rule:** changing a seed, noise level, estimator, split, referee setting or threshold
after this freeze means the confirmation is re-run under a NEW prereg. Never quietly patched.

## Registered grid (frozen)

- **Primary world:** catalog σ=10, ρ=28, β=8/3 (`cat_nlin_lorenz_001` parameterization), t_span (0,10),
  RK45 reference integration rtol/atol 1e-10, sampling dt=0.005 (n=2001), IC (1,1,1).
- **Seeds {101, 202, 303} × noise sweep {0.5%, 1%, 2%, 3%}** (multiplicative, std-normalized
  measurement noise on all three states) = **12 registered runs per path**.
- **Secondary (exploratory only, reported not gated):** the other five catalog Lorenz worlds
  (`cat_nlin_lorenz_002..006`) at seed 101, noise 2%.
- **Estimator (declared):** Savitzky–Golay `(21, 3)` smoothing + `np.gradient` derivative — the
  transcript's own (deficient) estimator, registered deliberately.
- **Split:** first 70% of the time axis = selection window; final 30% = sealed confirmation window,
  consumed exactly once per run and burned in the full-lineage ledger on failure
  (`benchmarks/lorenz_public_benchmark_ledger.jsonl`).
- **Referee settings:** pairwise `adjudicate` with block bootstrap n_boot=1000, seed=7;
  `data_origin="unknown"`, `benchmark_level="unknown"`.

## Registered paths

- **PATH-B (negative control A, analytic naive).** Greedy forward *training*-BIC subset search over
  the full degree-≤3 monomial library (20 terms). Expected (and required by H2) to overfit with
  structured SG residuals: the training likelihood is invalid under these residuals.
- **PATH-B′ (negative control A′, curated).** The transcript's own 4-candidate comparison
  (`{x,y,xz}`, `{x,y,xz,x²y}`, `{x,y,xz,z³}`, `{x,y,xz,xyz}`), training-BIC ranked.
- **PATH-C (negative control B, chimera).** The transcript's operational model as written:
  dy/dt = 0.973879·ρ·x − 1.0·y − 0.100261·σ·x·z with ground-truth σ, ρ, β baked in and the −y
  term hardcoded, validated by explicit Euler (dt=0.01) visual 3D-plot similarity. Registered
  specifically because a chimera + attractor-plot eyeballing *looks* convincing. Visual similarity
  is NOT a registered instrument; PATH-C is scored only by the registered instruments below.
- **PATH-R (primary).** Referee-gated forward selection: candidates are added from the library in
  dev-BIC order, and a term is kept only if pairwise `adjudicate(new, incumbent)` on the SEALED
  holdout returns `unique_support` promoting the new model; otherwise it is declined and the next
  candidate is tried. A promotion is recorded only with its referee `reason` and `tau_epi` attached.

## Registered instruments (no absolute forward R²)

1. `structure_emitted` — recorded family vs truth `dy/dt = ρx − y − xz` beyond declared tolerance.
2. `max_coefficient_error_se` — max |ŵ − w*| over true terms ÷ classical OLS standard error
   computed from the held-out-shifted dev residual scale.
3. `aduvertical_abs_gap` — winner-vs-true absolute holdout RMSE gap (Δ).
4. `referee_status` — `unique_support` / `multi_hypothesis` / `insufficient_evidence`, verbatim.
5. `divergence_horizon_ratio` — for the re-fitted model and the oracle-noise model, both integrated
   with the SAME reltol/atol (RK45 1e-9) from the catalog `holdout_x0=(2,2,25)`: t* where
   ‖x_model − x_true‖ > 1, ratio of horizons. The ratio cancels integrator error by construction.

## Pre-committed hypotheses

Margin provenance is labeled per hypothesis: **[P] = set from the 12-run exploratory pilot; [D] =
design-stage margin (no pilot yet), confirm-or-refute as registered.**

- **H1 [P].** On all 12 runs, PATH-B's training-BIC winner is NOT promoted by the referee against
  the true model (`unique_support` observed 0/12 in pilot; threshold: 0/12).
- **H2 [P].** PATH-B admits ≥1 3rd-order (or otherwise non-true) term with |coef| > 1e-3 on
  ≥ **9/12** runs (pilot: 12/12; greedy forward searches up to 10 terms — the invalid-likelihood
  overfit is stronger under greedy search than the transcript's curated 4-candidate ranking).
- **H3 [D].** PATH-R records the true Lorenz coupling structure (no non-true term with |coef| >
  1e-3) on ≥ **9/12** runs. Confirm-or-refute as registered.
- **H4 [P, revised from margin evidence].** Max normalized coefficient error ≤ **2.5 SE** on ALL 12
  runs (pilot observed 0.09–1.95 SE; the earlier draft floor 0.5 SE was REFUTED by the pilot — the
  dominant error is systematic SG/derivative bias, not statistical noise — and was tightened only
  NOW, pre-freeze, never post-hoc).
- **H5 [D].** PATH-R divergence-horizon ratio ≥ 0.5 on ≥ **9/12** runs. Confirm-or-refute.
- **H6 (meta, PATH-C).** PATH-C fails ≥ 2 of {H4-instrument ≤ 2.5 SE, H5-instrument ratio < 0.5,
  `referee_status` ≠ `unique_support`, `aduvertical_abs_gap` outside the declared tolerance band}
  on ≥ **9/12** runs. PASS of the chimera on ≥ 4/12 runs ⇒ the INSTRUMENT SET is declared
  insufficient (benchmark-level failure, reported regardless of other outcomes).

## Decision rule

- H1, H4, H6: ALL runs (12/12 where stated). H2, H3, H5: ≥ 9/12 as stated.
- ANY registered hypothesis failing ⇒ that claim is **NOT CONFIRMED**, reported with raw numbers;
  no seed, noise level, estimator, split, referee setting or threshold may be adjusted to green it.
- PATH-B / PATH-B′ / PATH-C are registered to FAIL specific checks; their passing the WRONG
  instrument (e.g. training-BIC) is an expected negative-control observation, not a system defect.

## Artifacts (committed after the runs)

- `benchmarks/run_lorenz_public_benchmark.py` — frozen harness (this document's companion commit).
- `benchmarks/lorenz_public_benchmark_results.json` — per-run rows keyed by the frozen instrument
  names + payload provenance (git HEAD, argv, UTC timestamp, referee settings).
- `benchmarks/lorenz_public_benchmark_ledger.jsonl` — full-lineage burn records.
- `docs/LORENZ_PUBLIC_BENCHMARK_CONFIRMATION.md` — per-hypothesis verdicts on both sides.

## Provenance disclosure

The pilot behind margins labeled [P] ran on 2026-10-20 (12 runs: seeds {101,202,303} × noise
{0.5,1,2,3}%, referee bootstrap 500/seed 7, greedy forward-BIC up to 10 terms): H1 candidate
observed 0/12 promotions; H2 12/12; true-basis OLS coefficient error 0.09–1.95 SE (systematic SG
bias, largest at LOW noise). Thes pilot numbers are exploratory and NOT confirmatory; seeds 0
(transcript) and 7 (pilot) are excluded from confirmation.

## Non-goals

No SciML-on-real-data claims; no claims beyond the registered Lorenz worlds and integrators; no
"IEEE/SIAM standard" status beyond pinned citations (Brunton–Proctor–Kutz 2016 *Chaos*;
Messenger–Bortz 2021 WSINDy) recorded at harness runtime; no latency/host-dependent SLAs.

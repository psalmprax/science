# Preregistration: Higher-Order PDE Hypothesis Engine — Two-Sided Fresh-Seed Validation

**Document ID:** PREREG-HOPDE-2026-10-17
**Status:** DESIGN FROZEN at commit time; validation harness and artifacts committed SEPARATELY after.
**Relation:** `apedp_dyn/pde/higher_order_engine.py` (`HigherOrderPdeProposer` /
`HigherOrderPdeReferee` / ultraviolet stability guard) has no registered validation. The only
existing evidence is a deterministic fixed-instance unit battery — **exploratory**, and it
motivated the engine. This document freezes the confirmation protocol BEFORE the confirmation run.
**Disclosure:** the engine implementation already existed uncommitted in the working tree when this
prereg was frozen (declared rather than hidden, per the §16e workflow).
**Deviation rule:** changing a world, seed, grid, tolerance or criterion after freeze means the
validation is re-run under a NEW prereg — never quietly patched.

## Frozen world generators

All worlds 1D periodic, `n_x = 64`, `x ∈ [0, 2π)`. Per-seed initial condition (frozen recipe,
implemented exactly as written): `rng = numpy.random.default_rng(seed)`;
for `m ∈ {1, 2, 3}` draw `a_m = 0.2 + 0.4 · rng.uniform()` and `φ_m = 2π · rng.uniform()` **in that
order**; `u0(x) = Σ_m a_m · sin(m·x + φ_m)`. Fresh seeds: **{810, 811, 812}**.

Spectral integrator: the exact integrating-factor Fourier RK4 used by the exploratory unit
battery (same propagator form), frozen with the `(dt, steps, dev, holdout)` values below.

### W1 — KdV third-order dispersion (must confirm)

`u_t = −1.0 · (u·u_x) − 0.1 · u_xxx`; `dt = 0.001`, 60 steps; dev rows `0:30`, holdout `30:60`;
`temporal_order=1`, `max_spatial_order=4`, `periodic=True`, `sparsity_threshold=0.04`.

Criteria: `active_terms ⊇ {u*u_x, u_xxx}`; `|ĉ(u*u_x) + 1.0| ≤ 0.05`; `|ĉ(u_xxx) + 0.1| ≤ 0.02`;
`r2_dev > 0.999`; referee verdict `IDENTIFIED_CAUSAL_LAW`; `frozen_oos_r2 ≥ 0.99`;
`ultraviolet_well_posed == True` (no 4th-order term present ⇒ guard must not misfire).

### W2 — Kuramoto–Sivashinsky-type biharmonic (must confirm)

`u_t = −0.4 · u_xx − 0.05 · u_xxxx`; `dt = 0.0005`, 50 steps; dev rows `0:25`, holdout `25:50`;
`include_nonlinear=False`, `sparsity_threshold=0.03`.

Criteria: `active_terms ⊇ {u_xx, u_xxxx}`; `|ĉ(u_xx) + 0.4| ≤ 0.05`; `|ĉ(u_xxxx) + 0.05| ≤ 0.02`;
`r2_dev > 0.999`; referee `IDENTIFIED_CAUSAL_LAW`; `frozen_oos_r2 ≥ 0.99`;
`ultraviolet_well_posed == True` (biharmonic damping sign `c₄ < 0`).

### W3 — Second-order temporal Klein–Gordon wave (must confirm, BIC must pick `u_tt`)

Exact modal solution of `u_tt = 1.5 · u_xx − 0.5 · u`:
`u(x, t) = Σ_m a_m · cos(ω_m t + φ_m) · sin(m·x)`, `ω_m = sqrt(1.5·m² + 0.5)`;
`dt = 0.01`, 50 time rows; dev rows `0:30`, holdout `30:50` (non-overlapping split — deliberately
cleaner than the exploratory battery's overlap); `temporal_order=None` (auto/BIC selection),
`max_spatial_order=4`, `include_nonlinear=False`, `sparsity_threshold=0.04`.

Criteria: BIC selects `temporal_order == 2` and `target_lhs == "u_tt"`;
`active_terms ⊇ {u_xx, u}`; `r2_dev > 0.999`; referee `IDENTIFIED_CAUSAL_LAW`.

### G1 — Explosive ultraviolet control (must refuse)

Hand-constructed discovery result `u_t = +0.05·u_xxxx` adjudicated on W2's holdout data:
`ultraviolet_well_posed == False` and verdict ≠ `IDENTIFIED_CAUSAL_LAW`.
(Guards against a positive 4th-order coefficient — high-wavenumber blowup as `exp(+c₄k⁴t)`.)

### G2 — Null world (must not false-confirm)

`u` field = pure white noise (independent `rng.standard_normal` draws for dev and holdout, same
shape). Whatever the proposer emits (if anything) must NOT be adjudicated
`IDENTIFIED_CAUSAL_LAW`; a proposer abstention also satisfies the guard and is recorded.

## Determinism

Re-running any single seed must reproduce identical formula strings, coefficient vectors
(bitwise) and verdicts; the harness records digests.

## Pass criteria

ALL W1–W3 criteria pass on ALL three seeds AND G1 refuses on ALL seeds AND G2 does not
false-confirm on any seed. ANY failure ⇒ the §higher-order engine as specified FAILS this prereg:
it is reported as a negative result, and a v2 prereg is required before any modification of the
engine, world set or thresholds.

## Metrics and artifacts (planned, committed after the run)

- `benchmarks/run_higher_order_pde_validation.py` — generates the worlds from the frozen recipes,
  runs proposer → referee, records per-(world, seed) active terms, coefficients, `r2_dev`,
  `frozen_oos_r2`, UV-guard outcomes, verdicts and determinism digests.
- `benchmarks/higher_order_pde_validation_results.json` — canonical artifact (one run per seed).
- `docs/HIGHER_ORDER_PDE_VALIDATION_REPORT.md` — per-criterion verdict table and raw numbers.

## Non-goals

No noise-robustness claims (the registered worlds are noiseless spectral — noise response is a
separate future prereg); no 2D battery (declared future work with its own prereg); no claims on
real fluid/plasma data; no modification of `apedp_dyn/epistemic/high_dim_pde.py` or any frozen
referee module.

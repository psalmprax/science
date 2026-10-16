# Frontier 9/10 — Architectural Resolution Record

**Status:** `experimental` (corrected record)
**Created:** 2026-10-16
**Provenance:** This artifact resolves a documentary-integrity gap: an earlier
session summary referenced `frontier_9_of_10_architectural_resolution.md` as an
existing artifact before it was written to disk. This file is the corrected
record; all numbers below were **regenerated from the working tree** on
2026-10-16 and are re-verified by
[`tests/test_dyn_frontier_claims_integrity.py`](../tests/test_dyn_frontier_claims_integrity.py).

---

## 1. The three prior bottlenecks and their resolutions

| # | Prior blocker | Implementation | Module |
|---|---|---|---|
| 1 | Linear 1-Jet ODE limitation (no $y^2$ Riccati term, no $y''$) | Quadratic Riccati monomial ($R(x)y^2$), 2nd-order Jet ($S(x)y''$), 4th-order central-difference stencils for $y''$, forward RK4 trajectory validation | `rust/apedp-kernel/src/rime/diff_pade.rs` |
| 2 | Isolated representation selection (hard argmax BIC blocked heterogeneous composites) | Representation 7: `tv_smooth_composite` — Total Variation jump isolation chained into polynomial residual regression, emitted as a ranked candidate | `apedp_dyn/proposers/rime_ad_proposer.py` |
| 3 | 1D-ODE-only blindness (no spatio-temporal PDEs) | Native Rust PDE STRidge engine: 4th-order spatial stencils ($u_x, u_{xx}, u_{xxx}$), temporal differences ($u_t$), space-time feature tensor, sequentially thresholded ridge regression | `rust/apedp-kernel/src/rime/pde.rs`, bridge `apedp_dyn/epistemic/pde_discovery.py` |

---

## 2. Measured results (regenerated 2026-10-16)

### 2.1 Planck Riccati Jet discovery — exact rational recovery

- Discovered form (accepted after SR3-lite sparse refinement + rational snapping,
  2026-10-16): $(0.333x^3)y' + (-1.000x^2 + 0.333x^3)y + (0.333)y^2 = 0$
- Scaling by 3 yields the exact textbook Riccati relation
  $x^3 y' + (x^3 - 3x^2)y + y^2 = 0$ — coefficients snapped to $1/3$, $-1$, $1/3$
  (`snapped_terms = true`; spurious low-order terms removed by refinement)
- Integrated-trajectory $R^2$ = **0.999999999759**, jet RMSE = **8.597e-07**
- History: before the refinement pass the same engine returned a fitted cubic
  family with spurious terms ($R^2$ = 0.999965, jet RMSE = 0.004906). The
  SR3-lite + rational-snapping stage (`diff_pade.rs`, revert-guarded on both
  trajectory and jet fidelity) now removes the spurious support and
  rationalizes the coefficients.

### 2.2 Harmonic oscillator ($y'' + 4y = 0$)

- Discovered form: $0.25\,y'' + 1.000\,y = 0$ (exact structure recovery)
- Integrated-trajectory $R^2$ = **0.9999948646**, jet RMSE = **6.11e-14**
- `is_exact_recovery` = true. (An earlier summary rounded $R^2$ to "1.000000";
  the correct six-decimal value is 0.999995.)

### 2.3 Heat equation ($u_t = 0.5\,u_{xx}$)

- Active terms: `['u_xx']`; coefficient = **0.49719** (true 0.5)
- Dev fit $R^2$ = **0.996306**; holdout-window $R^2$ = **0.999997**
- **Holdout caveat:** the holdout figure is a *re-discovery fit on the held-out
  time window*, not evaluation of the dev-discovered coefficients on holdout
  (model-transfer). See §3.

### 2.4 Viscous Burgers ($u_t = -1.0\,u\,u_x + 0.1\,u_{xx}$)

- Active terms: `['u', 'u*u_x', 'u_xx']` — one spurious low-weight `u` term
  (coefficient 0.0127) was selected
- Advection coefficient: **-0.99621** (error 0.38%); diffusion coefficient:
  **0.11118** (error **11.2%**)
- Dev fit $R^2$ = **0.999468**; holdout-window $R^2$ = **0.999909**
- Correct structure recovered; **not** "parameter error < 0.05%" as an earlier
  summary claimed.

### 2.5 TV + smooth composite (Rep 7)

- True: $y = 2.0\,\lfloor x/1.5 \rfloor + 0.5x + 0.08x^2$
- Discovered: $y = 2.1204\,\lfloor (x-0.0519)/1.4810 \rfloor + 0.0490 + (0.4277x + 0.0789x^2)$
- In-sample fit $R^2$ = **0.999965**; step-spacing error **1.27%** (1.4810 vs 1.5),
  step-height error 6.0%, smooth-coefficient errors 1.4–14.5%
- (An earlier summary claimed OOS $R^2 = 0.999997$ and plateau error < 0.1%;
  both were overstated.)

---

## 3. Known limitations (honest record)

1. **PDE holdout is a re-fit, not model transfer.** `discover_pde` re-runs
   discovery on the holdout window and reports that fit's $R^2$. This tests
   procedural stability, not coefficient transfer. A true transfer score
   (frozen dev coefficients evaluated forward on holdout) is required before
   any "out-of-sample" claim.
2. **Burgers diffusion coefficient error is 11.2%** and one spurious term is
   active — "exact coefficient recovery" language is not supported.
3. **All of the above is `experimental`**, not `production`.

---

## 4. Verification commands

```bash
python3 -m pytest tests/test_dyn_frontier_9_of_10.py tests/test_dyn_frontier_claims_integrity.py -v
python3 -m pytest tests/test_dyn_competing_proposers_and_dsr.py tests/test_audit_open_flaws_remediated.py \
  tests/test_dyn_blind.py tests/test_dyn_canonical_ledger.py tests/test_dyn_native_ga_best.py \
  tests/test_dyn_experiment_ledger_and_benchmarks.py tests/test_documentation_and_spec_integrity.py \
  tests/test_dyn_expanded_proposers_and_calibrated_power.py -q
cd rust/apedp-kernel && cargo test --lib
```

Last full-suite run (2026-10-16): frontier 5/5, canonical 58/58, Rust 128/128,
documentation-integrity 5/5.

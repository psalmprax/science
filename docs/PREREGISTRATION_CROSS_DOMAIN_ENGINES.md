# Preregistration: Cross-Domain Physics Engines — Fresh-Instance Confirmation Batteries

**Document ID:** PREREG-XDOMAIN-2026-10-17
**Status:** DESIGN FROZEN at commit time; confirmation harness and artifacts committed SEPARATELY after.
**Relation:** five experimental engines are verified only by deterministic fixed-instance unit
batteries — those tests are **exploratory** evidence (correct executable behavior on one instance
each), not registered confirmations:
`apedp_dyn/quantum/dmrg.py`, `apedp_dyn/market/lob_transport.py`,
`apedp_dyn/audio/spectrogram_transport.py`, `apedp_dyn/molecular/se3_flow.py`,
`apedp_dyn/robotics/trajectory_transport.py`.
**Disclosure:** all engine implementations already existed (committed or in the working tree) when
this prereg was frozen; no confirmation run under these protocols has been executed at freeze time.
**Deviation rule:** changing any seed, instance, threshold or criterion after freeze means that
engine's battery is re-run under a NEW prereg — never quietly patched.

## Common rules

Per-engine fresh instances are declared below and generated with frozen recipes. No threshold is
re-tuned during confirmation. Each engine's battery is independent: one engine failing does not
invalidate another engine's result (this is not a family-set retention rule). Any failure is
reported per-engine as a negative result at raw numbers.

---

## 1. DMRG (`dmrg.py`) — seeds {901, 902, 903, 904, 905}

Random MPS inits from `numpy.random.default_rng(seed)`, otherwise the committed engine defaults.

| Config | Criterion |
|---|---|
| `N=8`, `J=1`, `h=0` | `|E_DMRG − E_ED| < 1e-8` |
| `N=10` consistency | `|⟨ψ|H|ψ⟩ − E| < 1e-7 · max(1, |E|)` |
| `N=8`, `h=0.7` field arm | `|E_DMRG − E_ED| < 1e-8` |
| `N=24`, bond 20, 2 sweeps | `|E/site − (1/4 − ln 2)| < 0.01` |
| `N=12`, bond 32, 3 sweeps | sweep energies non-increasing (`+1e-10` slack); final within `1e-6 · max(1, |E_ED|)` |

All five seeds repeated per config. Scope ceiling restated: 1D open Heisenberg chain only.

## 2. LOB transport (`lob_transport.py`) — fresh parameter grid

Grid (beyond the single exploratory combo): `(v, D, λ) ∈ {0.3, 0.7} × {0.05, 0.2} × {0.1, 0.4}`
(8 combos; deterministic generator, no RNG seeds).

| Path | Criterion |
|---|---|
| Spectral recovery | relative error < `1e-8` on `(v, D, λ)` |
| Finite-difference recovery | relative error < `5e-3`; fit `R² > 0.9999` |
| RK4 recovery | relative error < `1e-3`; fit `R² > 0.9999` |
| Statistical observables | relative error < `1e-6` (centre-of-mass drift, variance growth, mass decay) |
| PDE-engine discovery | terms `{u_x, u_xx, u}` present; coefficient tolerances `1e-3 / 1e-3 / 1e-2`; holdout `R² > 0.999` |

## 3. Audio chirp transport (`spectrogram_transport.py`) — fresh chirp specs

Declared grid: linear chirps `(start, rate) ∈ {300, 900} Hz × {200, 600} Hz/s`; quadratic chirps
`(start, rate, quad) ∈ {400, 600} × {100, 300} × {800, 1500}`. Deterministic synthesizers.

| Instrument | Criterion |
|---|---|
| Ridge instantaneous frequency (linear) | max error < `frequency_bin / 5` |
| IF model velocity (linear / quadratic mid-time) | error < `1.0` / `< 5.0` Hz; acceleration relative error < `0.05` |
| Next-frame prediction R² | mean > `0.995`; min > `0.99` |
| Jet2 vs Euler (accelerating chirp) | mean(jet2) > mean(euler); win fraction ≥ `0.9` |
| Local order-2 error | < `5e-4` and < `errors[1] / 3` |

## 4. Molecular SE(3) relaxation (`se3_flow.py`) — fresh conformers, seeds {911, 912, 913, 914, 915}

`n_atoms = 6` random conformers from frozen force-field parameters (bond targets and repulsion as
in the engine defaults); random coordinates from the seed RNG.

| Invariant | Criterion |
|---|---|
| Energy rotation/reflection invariance | < `1e-12` |
| Analytic gradient vs finite differences | < `1e-8` |
| Velocity-field equivariance (rotation/translation/mirror) | < `1e-12` |
| Relaxation | every energy step < 0; final bond lengths within `1e-2` of targets; final energy < `1e-4` |
| Center-of-mass conservation | drift < `1e-12` |
| Mirror-image relaxation | exact within `1e-12`; enantiomer handedness gap > `0.5` (achirality of distance-only energies stated, not hidden) |

## 5. Robotics unicycle (`trajectory_transport.py`) — fresh starts/goals, seeds {921…930}

| Instrument | Criterion |
|---|---|
| Exact arc-step analytics | < `1e-12`; full-circle closure < `1e-10` |
| Step-size error at the registered step | < `5e-3` |
| Convergence orders (halving study) | Euler ratio ∈ `[1.8, 2.2]`; Jet2 ratio ∈ `[3.6, 4.4]`; `jet2[0] < euler[0] / 10` |
| Navigation from random starts | success; final distance < `0.011`; steps < `1000` |
| Lyapunov descent | max distance increment ≤ `0`; final < `0.5 · initial` |
| Goal-directly-behind case | success; final distance < `0.011` |

## Pass criteria

Every engine's battery passes on every fresh instance listed. ANY failure ⇒ that engine's
claim(s) FAIL this prereg and are reported as negative results; a modification requires a v2
prereg for that engine.

## Metrics and artifacts (planned, committed after the run)

- `benchmarks/run_cross_domain_engine_confirmations.py`
- `benchmarks/cross_domain_engine_confirmation_results.json` — per-engine, per-instance rows with
  the frozen instrument names and run provenance (git HEAD).
- `docs/CROSS_DOMAIN_ENGINES_CONFIRMATION_REPORT.md` — per-engine verdict tables.

## Non-goals

No real-data claims: real LOB feeds, real audio, and experimental molecular data each require a
separate **data-integrity** prereg (the engines' own scope ceilings say so; this document does not
override them). No torque-level robot dynamics, no protein folding / quantum-chemistry accuracy,
no multi-asset LOB coupling, and no time-dependent DMRG.

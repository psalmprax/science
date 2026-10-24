# APEDP Architecture Map

## Core Framework (frozen live state)

### `apedp_dyn/epistemic/adjudicate.py` (hash: `3d432634...`)
- **Adjudication dataclass** with 16 fields:
  - `status: EpistemicStatus` (UNIQUE_SUPPORT, MULTI_HYPOTHESIS, INSUFFICIENT_EVIDENCE, NO_VALID_CANDIDATE)
  - `promoted: Any = None`
  - `ranking: tuple = ()`
  - `best_rmse, second_rmse, delta_rmse: float | None`
  - `holdout_noise_scale, holdout_scale_source: float | None; str`
  - `epistemic_margin: float | None`
  - `tau_epi, tau_epi_source: float | None; str`
  - `bootstrap_branch: str = ""`
  - `bootstrap_lo, bootstrap_hi: float | None`
  - `contract: EpistemicContract = UNKNOWN` (FULL/INCOMPLETE/UNKNOWN)
  - `data_origin: str = "unknown"` — **newly added** (unknown/simulation/experiment)
  - `reason: str = ""`

### Key Constants (frozen)
- `SOLE_SURVIVOR_GAMMA = 3.0` — absolute ceiling multiplier
- `CAL_SIGMA = 1.8` — frozen 36-cell grid median (σ̂/σ_true ∈ [1.67, 2.00])
- `RESID_RHO_FLAG = 0.5` — whiteness flag: |ρ₁| > 0.5 = structured
- `DYNAMIC_EPISTEMIC_VERSION = "bootstrap-v2+solefid-v4"`

### Guard 1: Sole-Survivor Fidelity (adjudicate.py:185-241)
- `rmse ≤ gamma * (sigma / CAL_SIGMA)` before any comparative discrimination
- Missing/non-finite RMSE or sigma → refusal
- Degenerate noise (sigma ≤ 0) → `rmse ≤ ABSOLUTE_EPS`

### Guard 2: Residual Whiteness (adjudicate.py:219-241)
- Applied **after** Guard 1 passes
- `|ρ₁| ≤ 0.5` on per-timepoint²-error series
- Constant residuals pass (no structure possible)
- Missing/non-finite residuals fail
- Structured residuals → `INSUFFICIENT_EVIDENCE` (reported, not enforced on pair path)

### Bootstrap Tiebreaker (adjudicate.py:411-443)
- `bootstrap=True` + `ΔRMSE > τ_epi` → run paired block bootstrap (2000 boots, seed=0)
- CI entirely > 0 → `UNIQUE_SUPPORT`
- CI overlaps 0 → `MULTI_HYPOTHESIS`
- CI entirely < 0 → `MULTI_HYPOTHESIS` with rival-favored note

### Pair-Adjudication Path (critical for closed-loop)
- `adjudicate([rt, rr], resid_best=best_se, resid_rival=rival_se, bootstrap=True)`
- Computes `ΔRMSE`, `τ_epi`, `bootstrap_branch`
- `resid_sole` reports winner structure via `_winner_structure_note()` (diagnostic only, never enforced on pair path)

### `apedp_dyn/epistemic/bootstrap.py`
- `tiebreak(delta, tau, resid_top, resid_rival, n_boot=2000, seed=0)`
- Returns `{"lo, hi, mean, delta_point, tau_epi, branch, n, n_boot"}`
- Branches: `confirm` | `multi` | `rival-favored`

### `apedp_dyn/conf/benchmark_v2.py` (D5-CONF v2)
- **Ladder**: κ=10 → UNIQUE; κ=100 → MULTI; κ≥1000 → abstention; 0 false promotions
- **Misspecification**: 3-way split (generation_failure, selection_ambiguity, systematic residual structure)
- **Noise ceiling**: Gamma=3.0 with CAL_SIGMA=1.8
- **Tournament harnesses**: DUFF (reps 0-1=no_valid_candidate, rep 2=MULTI), COUP (1.0/1.0/1.0 at 1-5%, 0.67/0.00/0.67 at 10%), STAB (1.0→0→33→33% with noise)

### `apedp_dyn/dyn_tournament.py` (DUFF 3% tournament)
- 8-candidate slate, 3 noise pcts (0, 1, 2)
- `EXPECTED_STATUS = {0: "no_valid_candidate", 1: "no_valid_candidate", 2: "multi_hypothesis"}`
- `EXPECTED_PROMOTED = {0: None, 1: None, 2: None}`
- `EXPECTED_ATTRIBUTION` per-rep gate assignments

### `apedp_dyn/dyn_sweep.py` (DUFF/COUP/STAB four-noise sweep)
- SIGMA_PCTS = (1, 3, 5, 10), REPS = (0, 1, 2)
- `run_sweep()` returns per-problem curves: recovery, false_promotion, abstention
- Adjudication with bootstrap + whiteness diagnostic

### `apedp_dyn/conf/benchmark.py` (D5-CONF frozen)
- Dimensionless group discovery, regime-specific scaling invariance
- Π generation via Buckingham-Π engine
- Canonicalization, identifiability analysis

### Standdown & Restoration Package (forensics)
- `outputs/standdown_v1.json` — frozen artifact from concurrency incident
- `outputs/restoration_proposal_v1.json` + `v2.json` — 5-pillar package
  1. Exact baseline (hash, test count, gap record)
  2. Bootstrap restoration (tiebreaker, deterministic seed, rank-flip criterion)
  3. Pair-whiteness restoration (diagnostic vs. promotion blocking)
  4. Sole-survivor path (absolute ceiling, whiteness barrier, pairwise independence)
  5. Regression matrix (8 columns + capability probe + EPISTEMIC_CONTRACT_INCOMPLETE)

### Capability Probe & EPISTEMIC_CONTRACT_INCOMPLETE
- **`CAPABILITY_ABSENT`** probe: exposes when adjudication-layer capabilities are compiled absent
- **`EPISTEMIC_CONTRACT_INCOMPLETE`** records when a benchmark decision is made through a weakened layer
- **Architectural rule**: "This capability isn't present, so the test is skipped" ≠ "This capability is absent, therefore the system is allowed to make a weaker epistemic decision"
- The second could conceal a regression; the first is useful for dev compatibility

### Test Infrastructure
- **363 tests collected** across 24 test files
- **215+ dyn tests green** (epistemic bootstrap, sensitivity, D5, CONF, CONF v2, pointer adversarial)
- **Standdown test**: 30/30 epistemic bootstrap tests pass, 32/32 tournament+CONF tests pass

## What's Already Built (no new code needed)

| Component | Status |
|-----------|--------|
| Adjudication layer with gamma ceiling + whiteness guards | Green: 215/215 dyn tests |
| Bootstrap tiebreaker (2000 block boots, seed=0) | Green: included in 30+ test suites |
| Noise-scaled holdout ceiling (CAL_SIGMA=1.8) | Green: D5-CONF v2 + sandbox demo |
| 8-candidate tournament harnesses (DUFF/COUP/STAB) | Green: matches expected statuses |
| Multi-start refit separating optimizer-luck FN | Green: 3/9 truth-gate4 failures → passes |
| Physical gates (dimensional validity, scaling invariance) | Green: 20/20 CONF tests |
| Multi-hypothesis adjudication (default when evidence weak) | Green: default behavior |
| Standdown + restoration proposal framework | Green: frozen artifacts, 5-pillar package |
| `data_origin` field on Adjudication dataclass | Green: added, defaults to "unknown" |

## What Needs Building (research program)

| **Component** | **Our Contribution** | **Effort** |
|-------------|----------------------|------------|
| Fusion/materials plasma simulator | Physics model (already in `apedp_dyn/sandbox/plasma_simulator.py`) | Low |
| Closed-loop experiment designer (`arg_max Δy`) | Reuses pair-adjudication path `resid_best`/`resid_rival` + bootstrap | Moderate |
| Adversarial test battery (11 case types) | Standdown + restoration proposal validates rigor | Low |
| Gradual real simulator introduction (Levels 1-5) | Architecture is simulator-agnostic | Low |

## Research Program (7 Stages)

| Stage | Focus |
|-------|-------|
| 1 | Build fusion simulation sandbox (toy plasma model) |
| 2 | Hide some physics (controlled information loss) |
| 3 | Introduce unknown physics (hidden transport/instability) |
| 4 | Make stability the actual question (λ_max(J)<0, stability maps) |
| 5 | Closed-loop experiment designer (arg_max Δy recommendation) |
| 6 | Attack APEDP deliberately (11 adversarial case types) |
| 7 | Gradual real fusion simulator introduction (Levels 1→5) |

## Key Distinction (governance principle)

> **"This capability isn't present, so the test is skipped."** vs.
> **"This capability is absent, therefore the system is allowed to make a weaker epistemic decision."**

The first is useful for dev compatibility. The second could conceal a regression.
The probe exposes `CAPABILITY_ABSENT`; the benchmark harness separately records `EPISTEMIC_CONTRACT_INCOMPLETE`.

## Data Origin Labeling

- `data_origin: str = "unknown"` — default (existing code path)
- `"simulation"` — simulation-derived data (set via manual Adjudication construction)
- `"experiment"` — experimental data
- Labels the evidence source for downstream consumers

## Architecture Impact

The entire framework is **domain-agnostic**. The same adjudicate() signature, bootstrap tiebreaker, noise ceiling, and tournament harnesses work on:
- Plasma parameter space (synthetic ODE data)
- Hall-Petch grain-size strengthening (materials science)
- Dimensional group discovery (fluid mechanics)
- Any other holdout-evaluation problem

No code changes are needed to extend into new domains — the framework is already built to handle it.


# APEDP Documentation

Contributor entrypoints: [five-minute setup and CI matrix](../README.md#getting-started-in-5-minutes),
[repository structure policy and cleanup guide](../CONTRIBUTING.md#repository-structure-policy),
[module ownership map](../CONTRIBUTING.md#module-ownership-and-responsibilities),
and [review owners](../.github/CODEOWNERS).

# Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Core Modules](#5-core-modules)
6. [API Server](#6-api-server)
7. [Pipeline](#7-pipeline)
8. [Visualization](#8-visualization)
9. [Persistence](#9-persistence)
10. [REST API Reference](#10-rest-api-reference)
11. [Docker & Deployment](#11-docker--deployment)
12. [Testing & CI/CD](#12-testing--cicd)
13. [Small Physics LLM]

### 5.23 `apedp_dyn.quantum_compilation.gf2mul_llm_adjustable` — LLM-Adjustable GF(2^12) Quantum Multiplier

**Status:** `experimental`

**Description:** Configurable GF(2^b) multiplication tensor and a Weight-Space Programming (WSP) interface whose accepted changes are certified by exact GF(2) tensor reconstruction. The LLM-tunable surface is experimental; it does not establish a rank-54 multiplier or a faster quantum circuit.

**Configurable parameters (imported from `apedp_dyn.conf.canonical_spec`, no hardcoded thresholds):**
- `primitive_element`: Adjustable pe (2, 3, 5, 7, ...)
- `irreducible_poly`: Adjustable modulus polynomial (0x1053, 0x13, 0x11D, 0x409)
- `decomposition_rank`: Requested rank target; the returned rank may be higher when exact reconstruction cannot meet it
- `weight_scheme`: WSP weight scheme ('uniform', 'importance', 'llm_tuned')
- `sparsity_target`: Target sparsity for importance pruning (0-1)
- `transform_type`: Certified transformation ('swap', binary `scale`, or invertible rank-axis `mix`); `new_atoms` is unsupported without a certified search
- `target_rank`: Target rank for decomposition search
- `track`: MMLM track selection ('A' = NonNeural, 'B' = Hybrid, 'C' = AttentionFree)

**Core components:**
- `GF2FieldModel`: Configurable GF(2^b) field arithmetic
- `build_multiplication_tensor()`: Exact degree×degree×degree binary tensor
- `TensorDecomposition`: Rank-r decomposition (U, V, W factors)
- `WSPLayer`: Exactness-preserving uniform, importance-pruning, and transformation modes
- `MMLMTrackA` (NonNeural): exact GF(2) product through the rank-54 decomposition; no token model
- `MMLMTrackB` (Hybrid): exact path ablated against a least-squares-fitted real model of the same rank and factors
- `MMLMTrackC` (AttentionFree): exact product scored by bit-level energies (live products, tree contraction)

Each track returns a `scope` block (`unit`, `compositions`, `models_tokens: False`,
`implements_attention: False`). The A/B/C names come from the MMLM programme; what these
classes do is field arithmetic, and a test asserts the scope rather than trusting the
name. `tests/test_mmlm_tracks.py` pins exactness, packed-operand splitting, Track B
determinism, and Track C's energy identities.

The tracks are not one multiply deep. `bilinear_multiply_batch` runs many pairs through
the same U/V/W factors, and two compositions are built on it:

- `invert_batch` inverts elements as `x**(2**12 - 2)` by square-and-multiply — 11
  squarings and 10 multiplications, **21 field products per element** — and returns
  `x * x**-1 == 1` for each one. All 4,095 non-zero elements of GF(2^12) invert exactly
  through the rank-54 decomposition (86,000 products), which is a property no single
  product can demonstrate. Zero and out-of-field inputs raise rather than return 0.
- `gf_matvec` contracts a GF(2^12) matrix against a vector with one product per non-zero
  entry and XOR accumulation, reporting the executed count and the skipped zero entries.
  It is compared entry by entry against the field model, and a short row raises.

Both are covered in `tests/test_mmlm_tracks.py` for all three tracks, so a chain that is
wrong in the middle fails the identity instead of returning a plausible number.

**LLM tunable interface** (`LLMTunableGF2Multiplier`):
- Full round-trip LLM adjustment of all parameters
- Process sample data through selected track
- Explore rank-performance tradeoffs across MMLM tracks

**Example usage:**
```python
from apedp_dyn.quantum_compilation.gf2mul_llm_adjustable import LLMTunableGF2Multiplier
import numpy as np

# Default: Track A; the requested rank is checked against exact reconstruction
m = LLMTunableGF2Multiplier(track='A')
r = m.tune()

# LLM tunes primitive element to 3
m2 = LLMTunableGF2Multiplier(track='A', primitive_element=3, irreducible_poly=0x1053)
r2 = m2.tune()

# LLM tunes WSP importance pruning with sparsity 0.1
m3 = LLMTunableGF2Multiplier(track='A', weight_scheme='importance', sparsity_target=0.1)
r3 = m3.tune()

# LLM tries Track B (Hybrid)
m4 = LLMTunableGF2Multiplier(track='B')
r4 = m4.tune()

# LLM tries Track C (AttentionFree)
m5 = LLMTunableGF2Multiplier(track='C')
r5 = m5.tune()

# Sample processing
sample = np.random.randint(0, 2, size=(3, 12))
processed = m.process(sample)
```

**References:**
- `apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py`: Core GF(2^b) field model
- `apedp_mmlm/quantum_discovery.py`, `engines.py`, `models.py`: MMLM package modules
- `docs/DOCUMENTATION.md`: This documentation file
- `specifications/canonical_preregistered_spec_v1.json`: Canonical specification thresholds

**Status tags:** `experimental` — fully functional but new integration; status may be updated to `production` after extended validation.

**Known limitations:**
- The default recursive polynomial Karatsuba decomposition is exactly verified at rank 54 for GF(2^12), below the exact schoolbook rank 144. This is not a proof of minimum rank or a compiled/reversible circuit-cost claim; see `benchmarks/gf2_karatsuba_comparison.json` for the scoped evaluator and XOR estimate.
- Importance pruning uses support only to order candidate removals; every accepted removal is checked against the full tensor. A requested sparsity or rank is not guaranteed if it would make the decomposition inexact.
- LLM transforms are accepted only when exactness is preserved. `new_atoms` remains unsupported, and the rank-search interface reports a verified starting point rather than claiming an unimplemented search.

**Recent updates:**
- Module `apedp_dyn/quantum_compilation/gf2mul_llm_adjustable.py` (moved out of the repository root into the runtime package per CONTRIBUTING structure policy) with full GF(2^12) multiplier, tensor decomposition, WSP layer, and MMLM track integrations
- Verified GF2FieldModel generates all 4095 non-zero elements with pe=2 and prim_poly=0x1053
- Verified multiplication closure and tensor symmetry
- The recursive degree-12 Karatsuba factors are checked against all tensor coefficients; a failed check falls back to schoolbook factors.
- WSP importance and transformation updates preserve an exact input decomposition or return the original unchanged.
- LLM tuning replaces immutable field/WSP configuration objects instead of mutating frozen dataclasses.
- Tracks run the decomposition as the algorithm (`u_r = <U_r,a>`, `v_r = <V_r,b>`, `p_r = u_r & v_r`, `y = W^T p`) and compare the product with the field model; a float projection is no longer reported as a product
- Track B's ablation fits its reduction weights by least squares over 16,384 operand pairs at the same rank and the same factors, so the float path is the strongest real-valued model available rather than a random baseline; a real model cannot reproduce XOR, and the measured Hamming error says so
- Track C's tree contraction carries an odd trailing bit instead of dropping it, so its reported energy equals the bit count (a 9-bit product used to score 36.0 instead of 81.0, with the wrong parity)
- Every track returns a `scope` block (`unit`, `models_tokens: False`, `implements_attention: False`), and `tests/test_mmlm_tracks.py` asserts it along with exactness over sampled pairs
- `tests/test_tracked_import_closure.py` (standard suite) fails when tracked code imports a module git does not track, and the module's rank-search engine, field-tensor search, proposer and tests are tracked so a clean checkout imports them

(#13-small-physics-llm)
14. [Contributing](#14-contributing)
15. [License](#15-license)
16. [System Capabilities & Architectural Philosophy](SYSTEM_CAPABILITIES_AND_PHILOSOPHY.md)
17. [Evidence, Adapters & Domain Limits](#17-evidence-adapters--domain-limits)
18. [APEDP-X Dynamic Discovery, Epistemic Referee & CPMD Architecture](#18-apedp-x-dynamic-discovery-epistemic-referee--cpmd-architecture)
19. [Traceability & Verification Matrix](#19-traceability--verification-matrix)
20. [Transformer Proposer Design — Q&A Record](TRANSFORMER_PROPOSER_DESIGN_QA.md)
21. [Sparrow Quantum Deterministic Photonics Blueprint](SPARROW_QUANTUM_DETERMINISTIC_PHOTONICS.md)
22. [Reproducibility, Provenance & Benchmark Governance](#22-reproducibility-provenance--benchmark-governance)
23. [Quantum-Compilation Frontier, Champion Reproduction & Photonic Boundary](#23-quantum-compilation-frontier-champion-reproduction--photonic-boundary)

---

## 1. Overview

APEDP (Autonomous Physics & Engineering Discovery Platform) is an end-to-end scientific computing platform for discovering, validating, and repairing physical equations from experimental data. It combines symbolic regression, dimensional analysis, Bayesian model selection, physical anomaly repair, epistemic tracking, quantum error correction, metrology, and applied defect auditing in a unified framework.

### Epistemic integrity notes (2026-10-19, blind-audit round 2)

> **Outstanding requirements.** F14 (three-role data protocol), F16 (formal
> CBF-QP certificate), referee-path mutation testing, and an independent Lean
> toolchain verification are explicitly unresolved; their exact acceptance
> criteria live in [`SYSTEM_FLAWS_AUDIT.md`](SYSTEM_FLAWS_AUDIT.md)
> ("Outstanding requirements register", REG-F14/REG-F14b/REG-F16/REG-MUT/
> REG-LEAN). The remediation checkpoint is frozen at commit `47594e7`
> (tag `remediation-checkpoint-47594e7`).

- **Referee pair anchoring (`apedp_dyn/epistemic/adjudicate.py`).** The pair
  evaluation between a promoted head and its nearest rival is anchored on
  frame holdout RMSE (gate-4-certified), never on the train-loss ordering, so
  an overfitted train-loss winner can no longer manufacture a spurious
  `MULTI_HYPOTHESIS` refusal. Recorded in the frozen re-mint (see
  `CHANGELOG.md`, blind-audit round 2) and pinned by the epistemic referee
  tests.
- **NCM structural proposer (`apedp_dyn/proposers/ncm_structural_proposer.py`).**
  All four archetype branches emit real forward `compute_score` closures that
  reproduce the fit-time prediction; the zero-vector default in
  `CandidateProposal` is no longer reachable from this proposer.
- **Hawkes order-flow scorer causality.** The standardized-intensity
  `compute_score` couples only to fit-time intensity distribution constants;
  no eval-frame-wide mean/std (holdout-future statistics) inside scoring.
- **Lean bridge axiom hygiene (`apedp_dyn/epistemic/lean_universal_bridge.py`).**
  `exp_damp_le_one` carries the explicit `(hs : 0 ≤ s)` non-negativity
  hypothesis; the decay bound is never asserted out of the damped regime.
- **RIME implicit Rust kernel (`rust/apedp-kernel/src/rime/implicit.rs`).**
  `is_exact_recovery` additionally requires `active_count >= 2`, so the
  all-zero "0 = 0" pruned vector cannot be certified as an exact discovery;
  the exactness RMSE gate is scale-aware (relative to the RMS magnitude of
  the relation's residual terms) so O(1e5)-scale exact laws are certifiable
  while small-scale near-misses are refused. FLAW-STRAITA (2026-10-21,
  exact-RMSE gate audit): the raw-space `max_c = max|c_j|` coefficient gauge
  applied after the strata-coherent `c_j = v_j/norm_j` mapping and the raw
  `|c_j| > 1e-6` active-count recount were two compensating anchored gauges
  in coefficient space; at O(1e6)-scale strata they degenerated a genuinely
  pure 1e6 circle to the constant fallback `-1 = 0` (probed: rmse 1.0,
  exact relation refused). Both are now coherent contribution-space gauges
  (no normalization after `v/norm`; membership by contribution ratio
  `|c_j|·norm_j > 1e-6·max_j(|c_j|·norm_j)`); evidence trail in
  `docs/SYSTEM_FLAWS_AUDIT.md` (FLAW-STRAITA) — pure 1e6 circle certifies at
  rmse 4.1e-17 after the fix, MUT-E4's kill restored on the natural
  near-miss fixture.
- **Active-disambiguation train-block masking (FLAW-A01, 2026-10-21).**
  `ensemble_gated.py`, `ensemble_m.py`, `incremental_info.py` and
  `e5_market.py` build train blocks through the shared
  `trading_e4t1_v2::_train_blocks_no_leak` helper, which applies the 5-day
  boundary mask (`tr[tr + 5 < origin_idx]`) required because the `_d5`
  target forward-shifts the 5-day rolling mean; raw `dates < origin` blocks
  leaked forward out-of-sample rows into every fit in those modules.
- **Distributed-lag streaming buffer (FLAW-A09, 2026-10-21).**
  `apedp_dyn/proposers/distributed_lag_interaction_proposer.py::compute_score`
  snapshots the persistent lag buffer once per (call, base column) and
  writes it once per (call, base column), so sibling lag terms on the same
  column (e.g. `lag_1_x` + `lag_2_x`) cannot read each other's same-call
  tail as prior history; chunked scoring equals single-pass scoring.
- **Blind benchmark zero-denominator reporting (FLAW-A13, 2026-10-21).**
  `apedp_dyn/sandbox/blind_benchmark.py` per-domain `fpr`/`fdr`/`vrr`
  report `None` when the denominator is zero, mirroring the headline
  metrics: a domain with no negative controls has an UNMEASURED rate, never
  a certified 0% error rate.
- **PDE frozen-scoring library (`apedp_dyn/epistemic/pde_discovery.py`).**
  The Python cold-end library is a strict mirror of the Rust discovery
  library: a dev-discovered active term the mirror cannot resolve now fails
  closed (raises naming the term) instead of silently dropping a coefficient
  column; `u^3` is classified as a cubic state nonlinearity and `u_xxx` as
  the third-order spatial derivative, and the mirror's coverage of the full
  Rust 1D catalog is pinned.
- **Lean chain integrity (`apedp_dyn/lean_formal/`).** Every P2-verified
  record's emitted Lean source is `sorry`-free and every `sorry`-carrying
  file maps to a P0-refuted record, pinned against the regenerated
  provenance in `artifacts/all_system_equations_lean_provenance.json`.



For an in-depth breakdown of system capabilities and why the platform balances autonomous discovery with domain constraints, see [System Capabilities & Architectural Philosophy](SYSTEM_CAPABILITIES_AND_PHILOSOPHY.md).

APEDP should be described as a **domain-general evidence and model-adjudication
framework with domain-specific discovery adapters**, rather than as a fully
domain-agnostic equation generator. The distinction is operational: the
evidence layer is reusable, while units, search grammars, baselines, physical
constraints, and ingestion rules must be declared for each domain.

## 17. Evidence, Adapters & Domain Limits

### Evidence statuses

Benchmark and adapter outcomes use explicit scientific qualifications:

| Status | Meaning |
|---|---|
| `ADMIT` | Candidate passed the declared admissibility and robustness requirements. |
| `DOMAIN_LIMITED` | Candidate is supported in the tested domain, but extrapolation is not established. |
| `CONDITIONAL` | Structure is supported, but a documented quantitative or contamination limitation remains. |
| `PREDICTIVE_ONLY` | Predictive utility is supported; causal or mechanistic interpretation is not. |
| `REFUSE` | A hard physical, statistical, numerical, or provenance requirement failed. |
| `INCONCLUSIVE` | Evidence is incomplete, including insufficient seed or independent validation. |

These statuses are not substitutes for the underlying measurements. Reports
must retain holdout metrics, baselines, seed treatment, refusal reasons, and
the information firewall.

### Benchmark adapters

`apedp.cross_domain_benchmark` defines the adapter boundary. An adapter owns:

- locked train and holdout data;
- canonical dataset hashing and provenance;
- declared known, hidden, allowed, and forbidden information;
- track configuration (`uninformed`, `weak_prior`, or `literature_informed`);
- conversion of measured discovery/admission results into a benchmark outcome.

Repeated seeds are robustness measurements, not independent scientific
replications. A single-seed result without stability evidence should remain
`INCONCLUSIVE`.

## 22. Reproducibility, Provenance & Benchmark Governance

`apedp.reproducibility` provides shared seed controls, dataset content hashing,
and schema-versioned run artifacts. Neural discovery configs expose a seed
(default `42`) and deterministic-mode setting (default on); JAX randomness
remains explicit through PRNG keys. The standalone evolutionary
`SymbolicRegressor` uses a seeded per-instance generator. Deterministic flags
constrain supported framework operations but cannot guarantee bitwise-identical
results across different hardware and runtime versions.

Neural training with `log_dir` set writes
`<log_dir>/provenance/training-<dataset-sha256>.json`. The version `1.0` JSON
schema records a best-effort Git commit, dataset hash, full config and its hash,
seed, deterministic flag, metrics, and optional results. The same artifact
writer is available to other experiment entrypoints. The known-truth challenge
automatically writes a `<results-file>.provenance.json` sidecar.

`benchmarks/scientific_benchmark_manifest.json` pins the frozen known-truth
metrics and corpus checksum inventory. Run
`python benchmarks/check_scientific_drift.py` to check drift; protocol-approved
baseline updates must be reviewed and update the baseline and allowed drift
together. CI labels this gate separately from the canonical known-truth
scientific-regression test; setup, build, lint, type, and general test failures
are reported in their own workflow steps.

### Sensitivity analysis

`run_sensitivity_analysis()` preserves the primary result and records each
variant with an independent rationale and rule. It reports `ROBUST`,
`SENSITIVE`, or `UNRESOLVED`; a cleaned or filtered variant must never silently
replace the primary analysis.

### Rolling-origin identifiability

`apedp.identifiability.rolling_origin_identifiability()` evaluates a caller-owned
fit across multiple training endpoints. It records endpoint-wise parameter
values, in-sample fit scores, and future forecast errors, then reports
`POORLY_IDENTIFIED`, `STABLE`, or `UNRESOLVED`.

This diagnostic does not create formal confidence intervals. It exposes the
distinction:

```text
excellent forecast accuracy != long-run parameter identifiability
```

For logistic or saturation models, early growth-phase data may forecast the
next observations accurately while leaving the asymptote weakly constrained.

### Domain boundary

The reusable core provides evidence handling, holdout isolation, dimensional
checking, numerical conditioning, provenance, sensitivity reporting, and
fail-closed admission. New domains still require an adapter for data shape,
units or semantic quantities, physical constraints, meaningful baselines, and
domain-specific validation regimes. This is extensibility, not proof of
complete agnosticism.

### External-science track: NIST WebBook

`apedp.ingestion.NISTWebBookAdapter` and `NISTWebBookCase` provide an
offline-first contract for NIST Chemistry WebBook exports. The caller supplies
a versioned CSV export and source metadata; APEDP records the file hash,
units, property, compound groups, and preprocessing. Whole-compound holdouts
are required for transfer tests, and the literature equation is explicitly
withheld from the discovery firewall.

This adapter is infrastructure, not a benchmark result by itself. A scientific
claim requires a pinned export, a declared property question, grouped holdout
results, dimensional checks, identifiability diagnostics, and post-discovery
literature comparison.

`apply_noise_ladder()` adds controlled training-target perturbations at
`0`, `0.1%`, `0.5%`, `1%`, `2%`, and `5%` of the clean training-target scale
by default. It preserves the clean grouped holdout and records the noise seed
and scale in provenance. This is robustness infrastructure; it does not imply
that NIST noise robustness has been scientifically evaluated until a pinned
export is run through discovery and adjudication.

`build_nist_manifest()` records the pre-discovery experiment contract,
including source/file hashes, groups, holdouts, seeds, grammar, and operator
limits. `run_nist_blind_discovery()` then runs the uninformed candidate search,
dimensional validation, locked holdout scoring, and fail-closed admission while
returning `literature_comparison="NOT_RUN"`. Literature comparison belongs in a
separate post-freeze step.

### Key Capabilities

| Capability | Module | Description |
|------------|--------|-------------|
| Dimensional Assurance | `core` | Full SI unit checking with SymPy/pint |
| Symbolic Regression | `discovery` | Evolutionary search for dimensionally-consistent equations |
| Bayesian Model Selection | `discovery` | Occam's complexity penalty for model comparison |
| Physical Anomaly Repair | `repair` | Automatic singularity detection and regularization |
| Epistemic Tracking | `epistemic` | Adversarial deliberation tournaments, requirements traceability |
| Quantum Error Correction | `qec` | Knill-Laflamme conditions, surface code thresholds |
| Applied Defect Auditing | `applied_defects` | Physics, chemistry, materials science, nanotechnology |
| Applied Biopharma/Drug Discovery | `applied_biopharma` | Biomarkers, targets, repurposing, combination trials, ADMET (experimental) |
| Applied Fusion Energy | `applied_fusion` | Plasma, reactors, fuel cycle, materials (experimental) |
| Experimental Physics (Tesla) | `experimental_physics` | Unconventional EM concepts, validation (experimental) |
| Applied Materials Science | `applied_materials` | Superconductors, catalysts, energy storage, metamaterials (experimental) |
| Applied Synthetic Biology/Genomics | `applied_synthetic_bio` | CRISPR, protein engineering, pathways, circuits (experimental) |
| Applied Climate/Carbon Capture | `applied_climate` | DAC, CCUS, modeling, renewables, adaptation (experimental) |
| Applied Neuromorphic Computing | `applied_neuromorphic` | SNNs, memristors, BCI, architectures (experimental) |
| Applied Electrical/Electronics | `applied_electrical` | Power systems, PE, RF, embedded, semis, control (experimental) |
| Metrology | `metrology` | GUM uncertainty propagation, calibration traceability |
| REST API | `api` | FastAPI server for all modules |
| End-to-End Pipeline | `pipeline` | Ingest → Discover → Validate → Repair → Persist |
| Visualization | `visualization` | Fitness landscapes, uncertainty plots, calibration curves |
| Persistence | `persistence` | Save/load models with versioning |
| Scenarios | `scenarios` | 22 pre-built test scenarios |
| Data Ingestion | `ingestion` | Academic APIs, DOCX/PDF, Wikipedia |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        APEDP Platform                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   USER       │───▶│  API LAYER   │───▶│  DISCOVERY   │    │
│  │  (CLI/Web)   │    │  (FastAPI)   │    │  (SRE + Bayes)│    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  VISUALIZE   │    │  PERSIST     │    │  VALIDATE    │    │
│  │  (Matplotlib)│    │  (ModelStore)│    │  (Dimensional│    │
│  └──────────────┘    └──────────────┘    │  + Repair)    │    │
│         │                               └──────────────┘    │
│         │                       │                            │
│         ▼                       ▼                            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              CORE ENGINE                              │    │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │    │
│  │  │Dimensional│ │Physical  │ │Epistemic │ │ Metrology│ │    │
│  │  │Assurance│ │Anomaly   │ │Tracking  │ │  &       │ │    │
│  │  │Engine   │ │Repair    │ │Graph     │ │Calibrat. │ │    │
│  │  │         │ │Engine    │ │          │ │          │ │    │
│  │  └─────────┘ └──────────┘ └──────────┘ └─────────┘ │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │          SUPPORTING MODULES                           │    │
│  │  QEC │ Applied Defects │ Scenarios │ Ingestion        │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- git (for source access)

### Install from Source

```bash
cd /path/to/science
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"
```

### Install from Docker

```bash
docker build -t apedp:latest .
docker run -p 8000:8000 apedp:latest
```

### Verify Installation

```bash
python -c "from apedp import *; print('APEDP OK')"
pytest tests/ -v
```

---

## 4. Quick Start

### Basic Usage

```python
from apedp import DimensionalAssuranceEngine, ClosedLoopAssuranceSystem

# Create platform
engine = DimensionalAssuranceEngine()
platform = ClosedLoopAssuranceSystem(engine)

# Register variables
m = engine.register_variable('m', 'kilogram')
c = engine.register_variable('c_light', 'meter / second')

# Build expression
expr = m * c**2

# Register and evaluate
platform.register_model_spec('Einstein_Energy', expr, 'joule')
val = platform.evaluate_model('Einstein_Energy', {'m': 1.0, 'c_light': 299792458.0})
print(f"E = mc^2 = {val:.4e} J")  # 8.988e+16 J
```

### End-to-End Pipeline

```python
from apedp.pipeline import DiscoveryPipeline

pipeline = DiscoveryPipeline()
result = pipeline.run("data.csv", "target_column")
print(f"Discovered: {result['expression']} (score={result['best_score']:.4g}, lower is better)")
```

### Symbolic Regression

`run_discovery` returns `(best_candidate, best_score, history)` where
`best_score` is a penalized MSE (lower is better) — not R². Variables
must be registered with units on the engine first.

```python
from apedp.discovery import MultiVariableSRE
from apedp.core import DimensionalAssuranceEngine, ClosedLoopAssuranceSystem

engine = DimensionalAssuranceEngine()
engine.register_variable('mass', 'kilogram')
engine.register_variable('velocity', 'meter / second')
platform = ClosedLoopAssuranceSystem(engine)
sre = MultiVariableSRE(platform, independent_vars=['mass', 'velocity'], target_unit_str='joule')

# Your data
x_data = {'mass': np.array([1, 2, 3]), 'velocity': np.array([10, 20, 30])}
y_data = np.array([50, 200, 450])

expr, best_score, history = sre.run_discovery(x_data=x_data, y_data=y_data, generations=30)
print(f"Discovered: {expr} (score={best_score:.4g}, lower is better)")
```

### REST API

```bash
uvicorn apedp.api:create_app --host 0.0.0.0 --port 8000
```

For a network-exposed deployment, start the factory with an API key:

```bash
APEDP_API_KEY='replace-with-a-secret' uvicorn apedp.api:create_app --factory --host 0.0.0.0 --port 8000
```

Dimensionful API discovery requires explicit `variable_units` by default.
`allow_heuristic_units=true` is available only as an explicitly labelled
exploratory escape hatch. Discovery responses include `data_adequacy`; use
`DiscoveryPipeline.run_prospective()` with a locked holdout before treating a
candidate as evidence beyond the observed data.

---

## 5. Core Modules

### 5.1 `apedp/core.py` — Dimensional Assurance Engine

The foundation of APEDP. Provides dimensional analysis using SymPy and pint.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `PhysicalConstant` | Fundamental constant with value, units, description |
| `FundamentalConstantsRegistry` | 11 SI fundamental constants (c, G, h̄, k_B, e, ε₀, μ₀, σ, R, N_A, g) |
| `ConservationInvariantRegistry` | Conservation law enforcement (energy, momentum, charge) |
| `DimensionalAssuranceEngine` | Core dimensional analysis and equation validation |
| `BuckinghamPiEngine` | Buckingham Pi theorem for dimensional analysis |
| `VectorizedDimensionalEngine` | Vectorized dimensional analysis |
| `ClosedLoopAssuranceSystem` | End-to-end model validation loop |
| `PHYSICAL_CONSTANTS` | Pre-registered constants |
| `ureg` | Global pint UnitRegistry instance |

**Example:**

```python
from apedp.core import DimensionalAssuranceEngine, ureg

engine = DimensionalAssuranceEngine()
engine.register_variable('force', 'newton')
engine.register_variable('mass', 'kilogram')
engine.register_variable('acceleration', 'meter / second**2')

# Validate: F = ma
expr = engine.register_variable('mass', 'kilogram') * engine.register_variable('acceleration', 'meter / second**2')
result = engine.validate_expression(expr, ['mass', 'acceleration'])
# Returns: {'dimensionally_consistent': True, ...}
```

### 5.2 `apedp/discovery.py` — Discovery Engine

Symbolic regression and Bayesian model selection.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `BayesianRefinementEngine` | Bayesian hypothesis comparison with Kass-Raftery verdicts, computed from marginal likelihoods |
| `MultiVariableSRE` | Multi-variable symbolic regression using evolutionary search |
| `EvolutionaryDiscoveryEngine` | Evolutionary discovery of new physical laws |

**BayesianRefinementEngine:** a Bayes factor is a ratio of *marginal* likelihoods
`p(D|M) = \u222b p(D|\u03b8,M) p(\u03b8)d\u03b8`, so supply a likelihood and a prior and the integral is
taken for you:
```python
from apedp.discovery import BayesianRefinementEngine

evidence = BayesianRefinementEngine.marginal_likelihood(
    log_likelihood_fn, prior_logpdf, bounds=np.array([[0.0, 2.0]])
)
# -> ModelEvidence(log_evidence, method="quadrature", n_params=1, prior_description=...)
bf, verdict, log_bf = BayesianRefinementEngine.compute_bayes_factor(evidence_1, evidence_2)
# Returns: (bayes_factor, verdict_string, log10_bayes_factor); the verdict names the method
```
A ratio of summed pointwise likelihoods is a different, weaker statistic — it has
no prior and no parameter integral — and is reported as such:
```python
ratio, description, log_ratio = BayesianRefinementEngine.compute_likelihood_ratio(
    np.array([0.9, 0.8, 0.85]), np.array([0.5, 0.4, 0.45])
)
# description: "... (NOT a Bayes factor: no prior, no parameter integral)"
```

**MultiVariableSRE:** evolutionary search with ephemeral random constants
(`C0`, `C1`, … — dimensionless placeholders that become constant features),
term-vector least-squares fitting (one coefficient per additive term, with
rational snapping and pruning), and constant-shift mutation, so laws with
additive offsets such as `2*x**2 + 3` are exactly recoverable. The
Occam complexity penalty inside the fitness (`mse + 1e-4*complexity`) is
scale-free: it is multiplied by `min(var(y), 1)`, so targets whose variance
is far below 1 (e.g. price-impact data with R² ≈ 1.6% signal) are no longer
over-penalized into losing a true structured model to the sample-mean
constant — verified by the pinned trading constant-collapse regression
(`tests/test_dyn_trading_constant_collapse_pin.py`); var(y) ≥ 1 targets
keep the exact legacy penalty.
```python
from apedp.discovery import MultiVariableSRE
from apedp.core import ClosedLoopAssuranceSystem

engine = DimensionalAssuranceEngine()
engine.register_variable('x', 'dimensionless')
engine.register_variable('y', 'dimensionless')
platform = ClosedLoopAssuranceSystem(engine)
sre = MultiVariableSRE(platform, independent_vars=['x', 'y'])

# Returns (best_candidate, best_score, history); best_score is a
# penalized MSE (lower is better), not R².
expr, best_score, history = sre.run_discovery(
    x_data={'x': np.array([1,2,3]), 'y': np.array([4,5,6])},
    y_data=np.array([5, 10, 17]),
    generations=30,
    pop_size=100,
    seed=42,
)
```

**Module layout.** The engine is split across two files: `apedp/discovery/_core.py`
holds selection, search and family detection, while `apedp/discovery/_bayes.py` holds
`ModelEvidence`, `BayesianRefinementEngine` and `STANDARD_RATIONALS`. `_core`
re-exports the moved names, so `from apedp.discovery import BayesianRefinementEngine`
resolves exactly as before; `tests/test_dyn_discovery_module_split.py` pins the names,
checks the re-exports are the same objects rather than copies, and holds both files
under a line ceiling.

Structural hygiene is gated too. `scripts/audit_exception_hygiene.py --check` fails CI
when a blanket `except Exception` neither re-raises nor carries a
`# noqa: BLE001 - <rationale>` marker, and when any of the 25 largest modules grows past
its pinned ceiling. The pin is `benchmarks/exception_hygiene_baseline.json`; counts and
sizes may fall freely, and re-pinning is an explicit `--write-baseline` decision.

### 5.3 `apedp/repair.py` — Physical Anomaly Repair
Automatic detection and repair of physical anomalies in equations.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `PhysicalAnomalyRepairEngine` | Domain-specific anomaly audit and repair |
| `AgnosticClosedLoopAssuranceSystem` | Agnostic anomaly detection loop |

**Audit Methods:**
- `audit_and_repair_bayes_factor()` — Bayesian anomaly detection
- `audit_and_repair_coulomb()` — Coulomb's law compliance
- `audit_and_repair_gravity()` — Gravitational anomaly detection
- `audit_and_repair_logarithm()` — Logarithmic singularity repair
- `audit_and_repair_relativity()` — Relativistic consistency check
- `audit_and_repair_resonance()` — Resonance anomaly detection
- `audit_and_repair_square_root()` — Square root domain repair

### 5.4 `apedp/metrology.py` — Metrology Analysis

GUM uncertainty propagation and calibration traceability.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `AnalyticalUncertaintyPropagator` | GUM-compliant uncertainty propagation |
| `MetrologySensorStream` | Real-time sensor data stream analysis |
| `CalibrationTraceabilityChain` | Calibration chain verification |
| `UncertaintyReport` | Structured uncertainty reporting |

### 5.5 `apedp/epistemic.py` — Epistemic Tracking

Knowledge graph and requirements traceability.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `EpistemicGraph` | Directed knowledge graph for claims and evidence |
| `LifecycleManager` | Lifecycle management of epistemic states |
| `RequirementsTraceabilityMatrix` | Requirements-to-test traceability mapping |
| `AdversarialDeliberationTournament` | Adversarial deliberation for claim validation |

### 5.6 `apedp/applied_defects.py` — Applied Defect Auditing

Domain-specific defect detection for applied sciences.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `UniversalAppliedDefectEngine` | Cross-domain defect detection |
| `AppliedPhysicsDefectAuditor` | Physics domain defect auditing |
| `AppliedChemistryDefectAuditor` | Chemistry domain defect auditing |
| `AppliedMaterialScienceDefectAuditor` | Materials science defect auditing |
| `AppliedNanotechnologyDefectAuditor` | Nanotechnology defect auditing (Gibbs-Thomson, Casimir) |

### 5.7 `apedp/qec.py` — Quantum Error Correction

Quantum error correction codes and conditions.

**Key Features:**
- Knill-Laflamme error correction conditions
- Surface code threshold analysis
- Concatenated code threshold recursion
- Cross-talk metrology propagation

### 5.8 `apedp/scenarios.py` — Scenario Suite

22 pre-built test scenarios covering all modules.

```python
from apedp.scenarios import APEDPScenarioSuite

suite = APEDPScenarioSuite()
results = suite.run_all()
# Returns: dict of scenario_name -> pass/fail
```

### 5.9 `apedp/ingestion/` — Data Ingestion

External data source connectors.

- `ingestion_academic_apis.py` — Crossref metadata search via `search_crossref()`, with query, DOI, source URL, and retrieval-time provenance; not a full-text or novelty search
- `ingestion_docs.py` — DOCX/PDF document loaders
- `ingestion_web_equations.py` — Wikipedia equation harvesting

### 5.10 `apedp/symbolic_regression.py` — Symbolic Regressor

Standalone symbolic regression with evolutionary search.

```python
from apedp.symbolic_regression import SymbolicRegressor

regressor = SymbolicRegressor(engine, ['x', 'y'], 'meter/second')
result = regressor.fit(data, target)
```

### 5.11 `apedp/falsification/`, `apedp/gated_assurance/`, `apedp/benchmarks/` — split packages

The three former monoliths are now packages of focused modules. Each package's
`__init__.py` is a **facade**: it re-exports every name the old module exposed
(including names it merely imported), so `from apedp.falsification import X`,
`apedp.gated_assurance.validate_three_gates`, and monkeypatching of module-level
attributes keep working unchanged. `__module__` of the moved classes now names
the implementation module (e.g. `apedp.falsification.falsifier`).

| Package | Modules |
|---|---|
| `apedp/falsification/` | `constants` (probe-envelope constants), `taxonomy` (axes, failure classes, evidence tiers), `counterexamples`, `constraints` (contradictory-constraint engine), `report`, `probe_support` + `probes_mathematical` / `probes_boundary` / `probes_numerical` / `probes_structural` / `probes_empirical` (the five axes, as mixins), `falsifier` (the `IndependentAdversarialFalsifier` class: `falsify()` plus `__init__`), `blind` (blind audit) |
| `apedp/gated_assurance/` | `checks` (constraint/asymptote/stability/generality probes), `contracts` (verdicts, metrics, benchmark contracts), `mechanistic`, `operational`, `engine`, `benchmark_contracts`, `feasibility` (classification + three-gate validator) |
| `apedp/benchmarks/` | `types` (trust weights, enums, metric helpers), `results`, `cases` (the 25- and 100-case tables, data only), `pathological` (suite selection), `harness` |

### 5.12 `apedp/applied_biopharma.py` — Applied Biopharma & Drug Discovery

Experimental planning and rule-screening utilities across illustrative disease areas. This is not a comprehensive or validated drug-discovery platform: target examples are static, ADMET is a Lipinski/Veber rule screen, and trial outputs are planning sketches rather than efficacy estimates.

**Status:** `experimental`  
**Tests:** `tests/test_applied_engineering_pharma.py`.

**Exports:** `DiseaseBiomarkerAnalyzer` (illustrative reference-range checks), `UniversalDrugTargetDiscovery` (static examples; refuses `omics_data` because it is not analyzed), `LeadOptimizer` (descriptor suggestions only), `ADMETPredictor` (rule-based Lipinski/Veber screen, not a predictive ADMET model), `ClinicalTrialDesigner` (template), `DrugRepurposingEngine` (static examples), and `CombinationTrialSimulator` (assumed planning rates; computes no p-value). Trial and efficacy outputs are deterministic (SHA-256 of inputs, not Python's salted `hash()`).

See [`BIOPHARMA_SCOPE.md`](BIOPHARMA_SCOPE.md) for implemented scope, explicit exclusions, and evidence requirements. [`apedp_dyn.epistemic.multiple_testing`](../apedp_dyn/epistemic/multiple_testing.py) provides BH/BY/Holm correction for caller-supplied p-values; it does not generate biomarker statistics.

### 5.13 `apedp/applied_fusion.py` — Applied Fusion Energy

Plasma physics analysis, confinement assessment, reactor design evaluation, fuel cycle analysis, and fusion materials screening.

**Status:** `experimental`  
**Tests:** `tests/test_applied_engineering_pharma.py`.

**Exports:** `PlasmaPhysicsAnalyzer` (Lawson triple product in m^-3·keV·s, beta, Debye length), `FusionReactorDesigner` (fusion gain Q = P_fus / P_heat, computed from inputs; `optimize_confinement` raises `NotImplementedError`), `FusionFuelCycle`, `FusionMaterials`.

### 5.14 `apedp/experimental_physics.py` — Experimental Physics (Tesla/Inventions)

Controlled exploration of unconventional electromagnetic concepts and Tesla-inspired inventions with physics validation and violation flagging.

**Status:** `experimental`  
**Tests:** No dedicated test file; package import verified.

**Exports:** `TeslaInventionSimulator`, `ExperimentalEnergyConcepts`.

### 5.15 `apedp/applied_materials.py` — Applied Materials Science

Superconductor discovery, catalyst design, energy storage materials, metamaterials, and materials property prediction.

**Status:** `experimental`  
**Tests:** `tests/test_applied_engineering_pharma.py`.

**Exports:** `SuperconductorDiscovery` (classifies by Tc only), `CatalystDiscovery` and `SuperconductorDiscovery.search_candidates` (literature reference data, not a search), `EnergyStorageMaterials` (cathode energy from capacity × voltage), `MetamaterialDesigner` (cloak wavelength from frequency).

### 5.16 `apedp/applied_synthetic_bio.py` — Applied Synthetic Biology & Genomics

CRISPR design/optimization, protein engineering, metabolic pathway engineering, synthetic circuit design, and genome assembly/annotation.

**Status:** `experimental`  
**Tests:** No dedicated test file; package import verified.

**Exports:** `CRISPRDesigner`, `ProteinDesigner`, `MetabolicPathwayOptimizer`.

### 5.17 `apedp/applied_climate.py` — Applied Climate Science & Carbon Capture

Direct air capture, carbon utilization/storage, climate modeling/attribution, renewable energy optimization, and climate adaptation strategies.

**Status:** `experimental`  
**Tests:** No dedicated test file; package import verified.

**Exports:** `DirectAirCapture`, `CarbonMineralization`, `ClimateModeling`.

### 5.18 `apedp/applied_neuromorphic.py` — Applied Neuromorphic Computing

Neuromorphic architectures, spiking neural networks, memristive devices, brain-computer interfaces, and neuromorphic algorithms.

**Status:** `experimental`  
**Tests:** No dedicated test file; package import verified.

**Exports:** `NeuromorphicArchitect`, `SpikingNeuralNetworkDesigner`, `MemristorDesigner`, `BrainComputerInterface`, `NeuromorphicAlgorithms`.

### 5.19 `apedp/applied_electrical.py` — Applied Electrical/Electronics Engineering

Power systems/grid, power electronics/converters, analog/RF/microwave, embedded systems/IoT, semiconductors/devices, and control systems.

**Status:** `experimental`  
**Tests:** `tests/test_applied_engineering_pharma.py`.

**Exports:** `PowerSystems` (fault MVA and voltage drop from source impedance and power factor), `PowerElectronics` (converter duty cycle), `AnalogRF`, `EmbeddedSystems` (clock period from clock rate), `Semiconductors` (figure of merit Rds(on) × Qg), `ControlSystems` (Ziegler–Nichols PID gains). Outputs are computed from inputs, not fixed values.

### 5.20 Core Search & Optimization

- `apedp/adaptive_search.py` — Adaptive search strategies with dynamic exploration/exploitation.
- `apedp/advanced_search.py` — Advanced search algorithms (beam search, evolutionary, ensemble). `HierarchicalSearch(template_set="extended", default)` adds the reciprocal, square-root, cubic, `exp(±x)`, log, cosine, rational-saturation, ratio and variable-in-exponent families to the classic six templates, and fits an exact least-squares scale and offset to every candidate before scoring it; `template_set="classic"` reproduces the old behaviour for comparison. `ValidationPipeline.overall` now scores `1 − (cv_mean + cv_std)`, not `cv_mean − cv_std`: the inverted form rewarded a larger error and made `search()` return its worst candidate (measured: 0/12 → 2/12 planted laws recovered by the classic set alone). See `docs/SEARCH_ENGINE_GAPS.md` §§1–2.
- `apedp/feature_selection.py` — `beam_search_expressions(..., include_functions=True)` adds the unary atoms (`sin`, `exp(-x)`, `sqrt`, `log`, `1/x`, `x**2`, `x**3`) the default variable-plus-constant atom set cannot compose at any depth; the default is unchanged so existing counts and orderings do not move.
- `apedp/patent_prior_art.py` — Provenance-checked patent prior-art index: JSONL records must carry an http(s) source and retrieval date or be explicitly `synthetic` (refused otherwise), claims without a normalised formula are counted as `NO_SEARCHABLE_CLAIMS` rather than silently skipped, and matching is rename-insensitive algebraic equivalence then structural signature. Verdicts are corpus-relative and explicitly not novelty or freedom-to-operate opinions. A five-record real corpus from Google Patents is bundled (`data/patents/`, built by `scripts/fetch_patent_corpus.py`, documented in `data/patents/PROVENANCE.md`) with per-record URLs, retrieval dates and HTML digests, and `scripts/run_patent_prior_art.py` records a declared-candidate run in `benchmarks/patent_prior_art/prior_art_run.json`; coverage and transcription limits are in `docs/SEARCH_ENGINE_GAPS.md` §4.
- `apedp/optimizer.py` — General optimization utilities.
- `apedp/bayesian_refinement.py` — Bayesian optimization and refinement.
- `apedp/hyperopt.py` — Hyperparameter optimization.

**Status:** `production`  
**Tests:** `tests/test_search_engines_real.py`, `tests/test_dyn_patent_prior_art.py`, `tests/test_evolutionary_search.py`; measured reachability in `benchmarks/search_engine_gaps.json` (`scripts/measure_search_engine_gaps.py`) and discussion in `docs/SEARCH_ENGINE_GAPS.md`.

### 5.21 Calibration, Uncertainty & Conformal Prediction

- `apedp/calibration.py` — Model calibration utilities.
- `apedp/conformal.py` — Conformal prediction for uncertainty quantification.
- `apedp/bootstrap_stability.py` — Bootstrap-based stability analysis.
- `apedp/uncertainty.py` — Uncertainty quantification framework.
- `apedp/ablation.py` — Ablation studies and feature importance.

**Status:** `production`  
**Tests:** Used in validation pipelines.

### 5.22 Adversarial, Robustness & Benchmarking

- `apedp/adversarial_benchmarks.py` — Adversarial robustness benchmarks.
- `apedp/robustness.py` — Robustness testing utilities.
- `apedp/cross_domain_benchmark.py` — Cross-domain generalization benchmarks.
- `apedp/decision_benchmark.py` — Decision-theoretic benchmarks.
- `apedp/red_team.py` — Red teaming and stress testing.

**Status:** `production`  
**Tests:** Integrated in evaluation suites.

### 5.23 Data Quality, Ontology & Governance

- `apedp/data_quality.py` — Data quality assessment.
- `apedp/classification_ontology.py` — Domain classification ontology.
- `apedp/constitution.py` — System constitution and constraints.
- `apedp/config_presets.py` — Configuration presets.
- `apedp/feature_store.py` — Feature storage and management.

**Status:** `production`  
**Tests:** Used across pipeline.

### 5.24 Additional Core Components

- `apedp/canonical_compiler.py` — Canonical equation compiler.
- `apedp/closed_loop.py` — Closed-loop discovery.
- `apedp/composer.py` — Model composition.
- `apedp/decay_models.py` — Decay/time-dependent models.
- `apedp/delay.py` — Delay/dynamical systems modeling.
- `apedp/dimensional_invariant_engine.py` — Dimensional invariants enforcement.
- `apedp/dimensional_layer.py` — Dimensional analysis layer.
- `apedp/dp_regression.py` — Differential/privacy-aware regression.
- `apedp/equation_bank.py` — Equation repository/bank.
- `apedp/evolutionary.py` — Evolutionary algorithms.
- `apedp/exceptions.py` — Exception definitions.
- `apedp/field_theory.py` — Field theory modeling.
- `apedp/formal_verification.py` — Formal verification utilities.
- `apedp/graph_reasoning.py` — Graph-based reasoning.
- `apedp/hallucination_detector.py` — Hallucination detection.
- `apedp/information_theory.py` — Information theory metrics.
- `apedp/interpretability.py` — Model interpretability.
- `apedp/knowledge_graph.py` — Knowledge graph construction.
- `apedp/llm_bridge.py` — LLM integration bridge.
- `apedp/memory.py` — Memory/state management.
- `apedp/metrics.py` — Evaluation metrics.
- `apedp/monte_carlo.py` — Monte Carlo methods.
- `apedp/multi_objective.py` — Multi-objective optimization.
- `apedp/neural_symbolic.py` — Neural-symbolic integration.
- `apedp/online_learning.py` — Online learning.
- `apedp/operator_theory.py` — Operator theory.
- `apedp/physics_constraints.py` — Physics constraint enforcement.
- `apedp/provenance.py` — Provenance tracking.
- `apedp/regularization.py` — Regularization techniques.
- `apedp/representation.py` — Representation learning.
- `apedp/rule_engine.py` — Rule engine.
- `apedp/safety.py` — Compatibility façade re-exporting the hardened parser, which moved to `apedp_dyn/safety.py` so `apedp_dyn` no longer imports the legacy root.
- `apedp/simulation.py` — Simulation framework.
- `apedp/statistical_testing.py` — Statistical testing.
- `apedp/stochastic.py` — Stochastic processes.
- `apedp/symbolic_bridge.py` — Symbolic computation bridge.
- `apedp/symmetry.py` — Symmetry detection.
- `apedp/thermodynamics.py` — Thermodynamics modeling.
- `apedp/topology.py` — Topological analysis.
- `apedp/transfer_learning.py` — Transfer learning.
- `apedp/trust.py` — Trust/scoring.
- `apedp/types.py` — Core type definitions.
- `apedp/utils.py` — Utilities.
- `apedp/validation_framework.py` — Validation framework.

**Status:** `production`/`mixed` — Core infrastructure components integrated throughout the system.

---

## 7. APEDP-DYN (Dynamic Discovery & Epistemic Referee System)

The `apedp_dyn/` subsystem provides the dynamic discovery engine, competing proposers, epistemic referees, and rigorous adjudication framework.

### 7.1 Core Architecture

- `apedp_dyn/__init__.py` — Package initialization and exports.
- `apedp_dyn/experiments/` — Experiment infrastructure and execution.
- `apedp_dyn/harness/` — Test harnesses and benchmarks.
- `apedp_dyn/epistemic/` — Epistemic tracking, referee logic, validation.
- `apedp_dyn/proposers/` — Competing hypothesis proposers.
- `apedp_dyn/gates/` — Promotion gates and validation gates.
- `apedp_dyn/conf/` — Configuration (including canonical spec).
- `apedp_dyn/lean_formal/` — Lean 4 formal verification integration.

### 7.2 Proposers (`apedp_dyn/proposers/`)

Independent proposers generating candidate hypotheses with strict separation from adjudication:

- `basis_regression_proposer.py` — Basis function regression proposer.
- `reservoir_proposer.py` — Reservoir computing proposer.
- `sindy_proposer.py` — SINDy (Sparse Identification of Nonlinear Dynamics) proposer.
- Additional proposers for microstructure, order book, genetic AST, etc.

**Status:** `production`/`experimental`  
**Rule:** Proposers never decide acceptance or inspect confirmation holdout data.

### 7.3 Epistemic & Referee System (`apedp_dyn/epistemic/`)

- Epistemic tracking and claim validation.
- Bayesian active learning, GP uncertainty quantification.
- Referee logic with FPR/CAR/VRR controls.
- Anti-snooping holdout enforcement.
- Temporal-rate modulation null test (Hypothesis B): a two-composition clock comparison that subtracts each clock's calibrated ordinary drive response and asks whether a phase-locked fractional-frequency residual is common across clocks; shared-phase Fourier surrogates plus injected calibration uncertainty form the null, while a heterogeneity guard rejects oscillator-specific artifacts. Calibration and confirmation must be uniformly sampled, must use disjoint drive harmonics, and must not reuse a dataset id. The calibration is refit on every calibration trial, per-harmonic SNRs are reported as an audit trail, and the detector reports exact Clopper-Pearson FPR/power bounds alongside the preregistered point-estimate gate. It can return only candidate / null / insufficient evidence, never a time-travel claim; a candidate can also be produced by an unmodeled common-mode systematic at the tested drive harmonics.
- Clock-dataset interchange and preregistration (`temporal_rate_io.py`): strict CSV + sidecar-JSON and compressed NPZ formats, `allow_pickle=False`, and a canonical SHA-256 fingerprint. `preregister_datasets()` freezes the calibration/confirmation fingerprints, dataset ids, drive frequencies, clock-composition map, and `temporal-rate-v2` analysis parameters (`harmonics`, `n_surrogates`, `seed`, `familywise_alpha`, `heterogeneity_alpha`, `min_fractional_amplitude`, and the canonical spec hash); `adjudicate_preregistered_temporal_rate()` recomputes them and fails closed on any mismatch. The artifact is a fingerprint commitment, not a signature. The demo script supports `--calibration` / `--confirmation`, `--harmonics`, `--write-preregistration`, and `--preregistration`.

**Status:** `production`

### 7.4 Quantum Compilation (`apedp_dyn/quantum_compilation/`)
Compiled circuit artifacts are build products and are not in git: `artifacts/` is ignored wholesale and `sample_submission/dqi.kmb` by name, because the dqi reference decoder alone is 886 MB. `config/quantum_artifact_manifest.json` is the record that replaces them -- one row per verified `.kmb` with its sha256, byte count and role -- and `scripts/verify_quantum_artifacts.py` verifies what is present (`--check`), re-records it on disk (`--write`), or restores it from the content-addressed cache at `$APEDP_ARTIFACT_CACHE` (`--materialize`). An absent artifact is reported but not an error; a *corrupted* one is, and so is being asked to propose without one: `DqiProposer` raises `MissingSampleArtifactError` instead of returning an empty list, because an empty list is indistinguishable from a healthy run that had nothing to propose. `tests/test_quantum_artifact_manifest.py` pins both halves.


DQI builders, inverse square root circuits, QROM implementations, multi-structure search, SABLE-90 profiling, FAROS-ISQ function-aware representation search (R1 nonuniform minimax, R2 prefix table, R3 direct phase decomposition, R4 alternative arithmetic, R5 hybrid), multi-fidelity Pareto tournaments, experiments, ledgers, and referees for quantum circuit compilation research.

**Status:** `experimental`

### 7.5 Other Components

- `apedp_dyn/audio/`, `apedp_dyn/molecular/`, `apedp_dyn/ode/`, `apedp_dyn/market/`, `apedp_dyn/model/`, `apedp_dyn/estimation/`, `apedp_dyn/fitting/` — Domain-specific modules.
- `apedp_dyn/operational/` — Operational utilities including lifecycle pruning.
- `apedp_dyn/distributed/` — Distributed computing support.
- `apedp_dyn/de_arm/`, `apedp_dyn/grammar/`, `apedp_dyn/kernel_shim.py`, `apedp_dyn/notebook_adapter.py`, `apedp_dyn/env.py`, `apedp_dyn/dyn_sweep.py`, `apedp_dyn/dyn_tournament.py` — Supporting infrastructure.

**Status:** `mixed` (production/experimental/audit-only as appropriate)

### 7.6 Additional Packages in `apedp/`

- `apedp/api/` — FastAPI REST server (documented in Section 8).
- `apedp/visualization/` — Plotting utilities (`plots.py`).
- `apedp/persistence/` — Model storage, experiment logging, serialization.
- `apedp/notebooks/` — Example notebooks and tutorials.
- `apedp/scenarios_data/` — Scenario datasets.

**Status:** `production`

---

## 8. API Server

### Starting the Server

```bash
uvicorn apedp.api:create_app --host 0.0.0.0 --port 8000
```

Or programmatically:

```python
from apedp.api import create_app
from uvicorn import run

app = create_app()
run(app, host="0.0.0.0", port=8000)
```

### Endpoints

#### Health
- `GET /health` — Service health and version
- `GET /` — Root endpoint

#### Discovery
- `POST /discover/regress` — Symbolic regression (penalized MSE score, lower is better; not R²)
- `POST /discover/bayes` — Bayesian model comparison

#### Metrology
- `POST /metrology/propagate` — GUM uncertainty propagation for a SymPy expression
- `POST /metrology/calibrate/log` — Log a calibration record
- `POST /metrology/calibrate/verify` — Verify traceability for a sensor

#### Repair
- `POST /repair/anomalies` — Boundary anomaly stress-test and regularization

#### Epistemic
- `POST /epistemic/node` — Add a knowledge-graph node
- `POST /epistemic/edge` — Add a lineage edge
- `POST /epistemic/requirement` — Register a traceability requirement
- `POST /epistemic/tournament` — Run an adversarial deliberation tournament

---

## 7. Pipeline

### Overview

The `DiscoveryPipeline` provides an end-to-end workflow:

```
CSV Data → Ingest → Discover → Validate → Repair → Persist → Results
```

### Usage

```python
from apedp.pipeline import DiscoveryPipeline, run_pipeline

# Method 1: Class-based
pipeline = DiscoveryPipeline(output_dir="my_results")
result = pipeline.run("data.csv", "target_column")

# Method 2: Convenience function
result = run_pipeline("data.csv", "target_column")
```

### Pipeline Result Structure

```json
{
  "model_id": "pipeline_target_20240101_120000",
  "expression": "x**2 + y",
  "best_score": 0.0003,
  "score_semantics": "penalized MSE (lower is better); not R²",
  "variables": ["x", "y"],
  "dimensional_validation": {"dimensionally_consistent": true, "computed": "...", "expected": "..."},
  "anomaly_repair": {"status": "APPROVED", "anomalies": [], "repaired_expression": "..."}
}
```

CSV columns carry no unit metadata, so pipeline variables are registered
as dimensionless unless `variable_units` are passed to `discover()`.

### CSV Format

```csv
x,y,z,target
1.0,2.0,3.0,10.5
2.0,3.0,4.0,17.2
3.0,4.0,5.0,26.1
```

---

## 8. Visualization

### Available Plots

```python
from apedp.visualization import (
    plot_regression_fitness,
    plot_uncertainty_propagation,
    plot_calibration_curve,
    plot_dimensional_analysis,
    plot_epistemic_graph,
    plot_scenario_results,
)
```

#### `plot_regression_fitness(complexity, r_squared, output_path)`
Plot model complexity vs R² score for symbolic regression.

#### `plot_uncertainty_propagation(values, uncertainties, output_path)`
Plot measured values with error bars.

#### `plot_calibration_curve(x_values, y_measured, y_ideal, output_path)`
Plot measured vs ideal calibration values.

#### `plot_dimensional_analysis(variables, dimensions, output_path)`
Bar chart showing dimensional structure of variables.

#### `plot_epistemic_graph(nodes, edges, output_path)`
Network graph of epistemic knowledge graph.

### HTTP Visualization Routes

The `apedp/api/routes/visualization.py` router is mounted under `/visualize`
(only when `create_app()` can import it). Each POST route writes a figure to the
`output_path` in the request body and returns `{"status": "success", "output_path": ...}`.

- `POST /visualize/regression`, `/uncertainty`, `/calibration`, `/heatmap`, `/pareto`, `/violin`
- `POST /visualize/search/progress` (PNG) and `/visualize/search/dashboard` (HTML)
- `GET /visualize/download/{filename}` serves a file from the server's working directory

Covered by `tests/test_visualization_routes.py`.

#### `plot_scenario_results(scenario_names, pass_fail, output_path)`
Bar chart showing scenario pass/fail results.

### Example

```python
plot_regression_fitness([1,2,3,4,5], [0.5,0.7,0.85,0.92,0.95], "fitness.png")
plot_scenario_results(["s1","s2","s3"], [True, True, False], "results.png")
```

---

## 9. Persistence

### ModelStore

Persistent storage for discovered models with versioning.

```python
from apedp.persistence import ModelStore
import sympy as sp

store = ModelStore("my_models")
expr = sp.symbols('x')**2 + sp.symbols('y')

# Save
model_id = store.save_model(
    model_name="quadratic_model",
    expression=expr,
    variables=["x", "y"],
    r_squared=0.99,
    complexity=3,
    metadata={"domain": "physics"},
)

# Load
model = store.load_model(model_id)
print(model["expression"])

# List all
all_models = store.list_models()

# Delete
store.delete_model(model_id)
```

### ExperimentLog

Append-only experiment log in JSONL format.

```python
from apedp.persistence import ExperimentLog

log = ExperimentLog("experiments.jsonl")
log.log("discovery_pipeline", {"csv": "data.csv"}, {"r_squared": 0.99})
experiments = log.get_experiments("discovery_pipeline")
```

### Serialization

```python
from apedp.persistence import serialize_model, deserialize_model, to_json, from_json

json_str = serialize_model(expr, ["x", "y"], {"r2": 0.99})
model = deserialize_model(json_str)
```

---

## 10. REST API Reference

### Base URL

```
http://localhost:8000
```

### POST /discover/regress

Run symbolic regression on data. Variables must be registered with units
(they default to dimensionless). Returns the penalized MSE score
(lower is better) — not R².

**Request:**
```json
{
  "variables": ["x"],
  "variable_units": {"x": "dimensionless"},
  "target_unit": "dimensionless",
  "data": {"x": [1.0, 2.0, 3.0, 4.0, 5.0]},
  "output_values": [1.0, 4.0, 9.0, 16.0, 25.0],
  "generations": 30,
  "pop_size": 100,
  "seed": 42
}
```

**Response:**
```json
{
  "best_expression": "x**2",
  "best_score": 0.0003,
  "score_semantics": "penalized MSE (lower is better); not R²",
  "generations_run": 20
}
```

### POST /discover/bayes

Compare two models. The response reports which quantity was computed, because
the two are not interchangeable: a Bayes factor needs a prior and a parameter
integral, while likelihood arrays only give a summed-likelihood ratio.

**Request (Bayes factor — supply log *marginal* likelihoods):**
```json
{
  "model_1_log_evidence": -12.0,
  "model_2_log_evidence": -20.0,
  "evidence_method": "quadrature"
}
```

**Response:**
```json
{
  "quantity": "bayes_factor",
  "bayes_factor": 2980.9,
  "verdict": "Very Strong / Decisive evidence for Model 1 (quadrature vs quadrature)",
  "log10_bayes_factor": 3.4743,
  "evidence_method": "quadrature"
}
```

**Request (summed-likelihood ratio — explicitly not a Bayes factor):**
```json
{
  "model_1_likelihoods": [0.9, 0.8, 0.85],
  "model_2_likelihoods": [0.5, 0.4, 0.45]
}
```

**Response:**
```json
{
  "quantity": "likelihood_ratio",
  "likelihood_ratio": 1.875,
  "verdict": "summed likelihood ratio favours Model 1 (NOT a Bayes factor: no prior, no parameter integral)",
  "log10_likelihood_ratio": 0.273
}
```

### POST /metrology/propagate

GUM first-order uncertainty propagation for a SymPy expression.

**Request:**
```json
{
  "expression": "2*m",
  "param_values": {"m": 2.0},
  "param_uncertainties": {"m": 0.1}
}
```

**Response:**
```json
{
  "evaluated_value": 4.0,
  "combined_uncertainty": 0.2,
  "expanded_uncertainty_95": 0.4,
  "relative_uncertainty": 0.05,
  "variance_budget": {"m": 0.04},
  "percentage_budget": {"m": 100.0}
}
```

### POST /metrology/calibrate/log

Log an ISO/IEC 17025 calibration record.

**Request:**
```json
{
  "sensor_id": "S1",
  "standard_id": "NIST-1",
  "procedure_name": "PROC",
  "drift_compensation_factor": 1.0,
  "uncertainty_budget": 0.01,
  "timestamp": "2026-01-01T00:00:00Z"
}
```

### POST /repair/anomalies

Boundary anomaly stress-test with automatic regularization.
`1/x` returns `REPAIRED` with the singularity listed;
`x+y` returns `APPROVED`. Dimensionally inconsistent input
(e.g. `x+1` with `x` in meters) returns `REJECTED` with an error.

**Request:**
```json
{
  "formula": "1/x"
}
```

**Response:**
```json
{
  "status": "REPAIRED",
  "anomalies": ["x: Division-by-zero singularity at 0"],
  "original_expression": "1/x",
  "repaired_expression": "1/(eps_x + x)",
  "inferred_unit": "1/meter",
  "error": null
}
```

### POST /epistemic/node

Add a node. `node_type` must be one of `EpistemicGraph.ENTITY_TYPES`
(e.g. `Claim`, `Hypothesis`, `Evidence`, `Model`).

**Request:**
```json
{
  "node_id": "C1",
  "node_type": "Claim",
  "metadata": {"text": "Energy is conserved"}
}
```

### POST /epistemic/edge

Add a lineage edge. `relation` must be one of `EpistemicGraph.EDGE_TYPES`
(e.g. `DERIVED_FROM`, `CONTRADICTED_BY`, `VALIDATED_BY`).

**Request:**
```json
{
  "nodes": [
    {"node_id": "A", "node_type": "Hypothesis"},
    {"node_id": "B", "node_type": "Claim"}
  ],
  "source_id": "A",
  "target_id": "B",
  "relation": "DERIVED_FROM"
}
```

### POST /epistemic/requirement

Register a full traceability requirement (all seven fields required).

**Request:**
```json
{
  "req_id": "REQ-1",
  "goal": "Enforce dimensional consistency",
  "owner": "Physics Assurance Lead",
  "design_adr": "ADR-001",
  "impl_artifact": "apedp/core.py",
  "acceptance_test": "test_dimensional_assurance_engine",
  "validation_evidence": "EV-001"
}
```

### POST /epistemic/tournament

Run the adversarial deliberation lifecycle (registers the hypothesis,
retires it on flaw detection, validates the repaired model).

**Request:**
```json
{
  "hypothesis_id": "H1",
  "formula_str": "1/x",
  "adversarial_probe_name": "zero",
  "boundary_flaw_detected": true,
  "repaired_formula_str": "1/(x+eps)"
}
```

---

## 11. Docker & Deployment

### Dockerfile

The `Dockerfile` builds a minimal production image:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY apedp/ ./apedp/
COPY tests/ ./tests/
EXPOSE 8000
CMD ["uvicorn", "apedp.api:create_app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]
```

### Build & Run

```bash
docker build -t apedp:latest .
docker run -p 8000:8000 -v $(pwd)/data:/app/data apedp:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  apedp:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/apedp_models
    environment:
      - PYTHONUNBUFFERED=1
```

---

## 12. Testing & CI/CD

### Running Tests

```bash
# Full test suite
pytest tests/ -v

# Specific test
pytest tests/test_apedp_suite.py::test_applied_nanotechnology_defect_auditor -v

# With coverage
pytest tests/ --cov=apedp --cov-report=html
```

### Linting & Type Checking

```bash
# Code linting
ruff check apedp/ tests/

# Type checking
mypy apedp/ --ignore-missing-imports

# Security scanning
bandit -r apedp/ -ll

# Auto-format
ruff check --fix apedp/ tests/
```

### CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/test.yml`) runs on every push:

1. **Test** — pytest on Python 3.10, 3.11, 3.12
2. **Lint** — ruff code style check
3. **Type Check** — mypy static analysis
4. **Security** — bandit vulnerability scan
5. **Build** — Docker image build (main branch only)
6. **Deploy** — Automated deployment (main branch only)

---

## 13. Small Physics LLM

### Overview

A proposed extension to APEDP: a small (1.5–3.8B parameter) domain-specific LLM fine-tuned on physics equations that collaborates with APEDP's rigorous verification engine.

### Motivation

APEDP discovers equations from data but cannot reason about them in natural language. A small physics LLM handles creative reasoning while APEDP handles verification.

### Architecture

```
User Question
    │
    ▼
┌─────────────────┐     ┌──────────────────────┐
│  Physics LLM    │────▶│  APEDP Engine        │
│  (suggestions)  │     │  (validate/verify)   │
│  1.5–3.8B params│     │  Dimensional + Bayes  │
└─────────────────┘     └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  ModelStore          │
                        │  (persist results)   │
                        └──────────────────────┘
```

### Phase 1: Knowledge Base

- Extract all discovered equations from APEDP's `ModelStore`
- Curate ~5,000 verified physics/engineering equations with dimensional metadata
- Include conservation laws, physical constants, domain-specific equations

### Phase 2: Model Selection & Fine-Tuning

**Recommended base models:**
- **Phi-3-mini** (3.8B) — Microsoft, excellent reasoning at small size
- **Qwen2.5-1.5B** — Alibaba, strong physics benchmarks
- **Mistral-7B** — Mistral, good balance of size and capability

**Training approach:**
1. Instruction tuning on the physics corpus
2. Custom tokenizer aware of units and dimensions
3. Structured JSON output head (not raw text)

### Phase 3: Integration

- LLM suggests candidate equations → APEDP validates dimensionally
- APEDP's Bayesian engine scores them with Occam's penalty
- Validated results feed back into training data (self-improving loop)

### Phase 4: RAG Layer

- Store ModelStore entries in vector database (FAISS/chroma)
- Retrieve similar discovered equations before reasoning
- LLM reasons over retrieved equations, not just training data

### Example Workflow

```python
# User: "Given mass and acceleration, what force equation is plausible?"
# Physics LLM suggests: "F = m * a"
# APEDP validates: Dimensions check [kg * m/s² = N] → PASSES
# APEDP scores: Bayesian comparison → Top model
# Result: "F = m * a" with R²=1.0, dimensions verified
```

---

## 14. Contributing

### Project Structure

```
science/
├── apedp/                    Main package
│   ├── __init__.py          Unified public API
│   ├── core.py              Dimensional assurance engine
│   ├── discovery.py         Symbolic regression & Bayesian selection
│   ├── repair.py            Physical anomaly repair
│   ├── metrology.py         Uncertainty propagation
│   ├── epistemic.py         Knowledge graph & traceability
│   ├── applied_defects.py   Domain-specific defect auditors
│   ├── applied_biopharma.py  Drug discovery & biopharma (experimental)
│   ├── applied_fusion.py     Fusion energy research (experimental)
│   ├── applied_materials.py  Materials science discovery (experimental)
│   ├── applied_synthetic_bio.py Synthetic biology & genomics (experimental)
│   ├── applied_climate.py    Climate science & carbon capture (experimental)
│   ├── applied_neuromorphic.py Neuromorphic computing (experimental)
│   ├── applied_electrical.py Electrical/electronics engineering (experimental)
│   ├── experimental_physics.py Tesla/inventions exploration (experimental)
│   ├── qec.py               Quantum error correction
│   ├── scenarios.py         22 test scenarios
│   ├── pipeline.py          End-to-end discovery pipeline
│   ├── symbolic_regression.py  Symbolic regressor
│   ├── api/                 FastAPI REST server
│   │   ├── __init__.py
│   │   └── routes/          API route modules
│   ├── visualization/       Plotting utilities
│   │   ├── __init__.py
│   │   └── plots.py
│   ├── persistence/         Model storage & experiment logging
│   │   ├── __init__.py
│   │   ├── model_store.py
│   │   ├── experiment.py
│   │   └── serializer.py
│   └── ingestion/           Data source connectors
│       ├── __init__.py
│       ├── ingestion_academic_apis.py
│       ├── ingestion_docs.py
│       └── ingestion_web_equations.py
├── tests/                    Test suite
│   ├── __init__.py
│   ├── audit_suite.py
│   └── test_apedp_suite.py
├── docs/                     Documentation
├── pyproject.toml            Package configuration
├── Makefile                  Build commands
├── Dockerfile                Container definition
├── .github/workflows/        CI/CD pipelines
├── .gitignore                Git exclusions
├── LICENSE                   MIT License
└── README.md                 Project overview
```

### Development Workflow

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,test]"

# Run tests
pytest tests/ -v

# Lint and format
ruff check --fix apedp/ tests/
ruff format apedp/ tests/

# Type check
mypy apedp/ --ignore-missing-imports

# Security scan
bandit -r apedp/ -ll
```

### Adding New Modules

1. Create module directory under `apedp/`
2. Add `__init__.py` with proper exports
3. Update `apedp/__init__.py` to include new exports
4. Add tests under `tests/`
5. Update `pyproject.toml` if new dependencies needed
6. Add documentation to `docs/`
7. Add CI/CD steps if needed

---

## 15. License

MIT — see `LICENSE` file.

---

## Appendix A: ANTI_PATTERNS.md

The project previously had 19 anti-pattern categories documented in `ANTI_PATTERNS.md`. All have been remediated:

| Category | Status | Fix |
|----------|--------|-----|
| Dead code (sketch/, apdep.txt, CSVs) | ✅ Deleted | Removed all fabricated data |
| Jupyter notebook artifact | ✅ Deleted | Removed `apdep.txt` |
| Symbolic regression import | ✅ Fixed | Changed to `from apedp.core import ...` |
| Module location | ✅ Fixed | Moved to proper `apedp/` package |
| Exception handling | ✅ Fixed | Added logging to all 12 locations |
| Packaging | ✅ Created | `pyproject.toml`, `.gitignore`, `LICENSE`, `Makefile` |
| Test fix | ✅ Fixed | Changed radius from 0.5e-9 to 0.1e-9 |
| SSL issue | ✅ Fixed | Added `ssl.create_default_context()` |
| Docstrings | ✅ Fixed | Removed paste.txt references |
| README | ✅ Updated | Describes platform architecture |
| CI/CD | ✅ Created | `.github/workflows/test.yml` |
| Docker | ✅ Created | `Dockerfile` |
| API layer | ✅ Created | FastAPI server |
| Visualization | ✅ Created | 6 plotting utilities |
| Persistence | ✅ Created | ModelStore, ExperimentLog, serializer |
| Pipeline | ✅ Created | End-to-end discovery pipeline |
| Documentation | ✅ Created | This document |

---

## 18. APEDP-X Dynamic Discovery, Epistemic Referee & CPMD Architecture

The October 2026 research programme upgrades APEDP into **APEDP-X / CPMD** (Continuous Predictive Market Discovery), transitioning from a passive hypothesis referee into an active, multi-proposer scientific discovery engine with formal verification and multiple-testing controls.

### 18.1 Active Module Index & Registry

Status tags follow AGENTS.md §3 (`production`, `experimental`, `audit-only`, `deprecated`). Rows tagged `production` are the core-package default classification; explicit `Status:` docstring tags mark deviations and must match this table — consistency is machine-checked by `tests/test_documentation_and_spec_integrity.py::test_registry_status_tags_match_code_status_labels`.

| Module | Description | Status |
|---|---|---|
| [`apedp_dyn/conf/canonical_spec.py`](../apedp_dyn/conf/canonical_spec.py) | Canonical preregistered configuration loader enforcing SHA-256 integrity on frozen specs. | `production` |
| [`apedp_dyn/conf/benchmark_v2.py`](../apedp_dyn/conf/benchmark_v2.py) | Benchmark v2 configuration schemas, trap suites, and seed schedules. | `production` |
| [`apedp_dyn/conf/paths.py`](../apedp_dyn/conf/paths.py) | Repository path resolution policy (`APEDP_REPO_ROOT` / `APEDP_LEAN_ROOT` / `APEDP_ARTIFACT_DIR` with repo-relative fallbacks; ARCHITECTURE.md §4.2). | `production` |
| [`apedp_dyn/conf/dimensionless.py`](../apedp_dyn/conf/dimensionless.py) | Dimensional homogeneity rules and physical dimensionless parameter bounds. | `production` |
| [`apedp_dyn/epistemic/experiment_ledger.py`](../apedp_dyn/epistemic/experiment_ledger.py) | Full-lineage candidate experiment ledger with anti-snooping holdout burning. | `production` |
| [`docs/VQC_DESIGNER_MAP.md`](../docs/VQC_DESIGNER_MAP.md) | Route map of the VQC designer as distinct from its claims: the specification-to-verified-schedule path (target validation → construction family → circuit IR → two in-package referees → ranking and Pareto frontier → admission → placement and routing → minimum edge colouring), the 24 modules with line counts and roles, the public entry points per module, the eight records and seven producers, and a table of what the designer refuses and why. States what is **not** there: no non-Clifford synthesis search, no invented topology, no 900-qubit routing, no hardware cost model. | `audit-only` |
| [`docs/REPOSITORY_CANONICAL_DECISION.md`](../docs/REPOSITORY_CANONICAL_DECISION.md) | Which of the two checkouts on this host is canonical and what is stranded in the other. Ruling: `science` is canonical (it holds the entire VQC designer, is 35 commits ahead and 5 behind, and carries every recent line); the sibling `/ALL_PROJECTS/science-quantum-discovery` is a read-only archive until its four stranded items are drained. Evidence is git-level — the stores overlap and the common ancestor is `444c1fa`, so 8 of the 9 files the archive's 5 commits touch are already byte-identical here. The one item that can mislead a reader is `ae02214`: the archive **rejected** the DFTHCSOS 10-bit signed 4-bank candidate with preserved arithmetic (6.74× and 2.24× over threshold, 132 angle qubits against a 128 cap, a 4,363-Toffoli gap), while this checkout still reads `PROMOTION_STATUS_HOLD` — an open question that has already been answered. | `audit-only` |
| [`apedp_dyn/epistemic/holdout_seal.py`](../apedp_dyn/epistemic/holdout_seal.py) | Hash-chained seal over the burned-holdout registry. Folds every non-empty line into a running SHA-256 chain (domain-separated genesis, no key) and writes the terminal digest to a `<registry>.seal` sidecar; `verify_seal` recomputes the chain and fails closed on a line-count, byte-length or digest mismatch, so a truncated registry, an edited burn, or a registry restored from an earlier commit while keeping the current seal each raise `HoldoutSealError` instead of quietly resurrecting a burned slice. A registry that does not exist yet is a fresh registry with a defined genesis, not an error; once the file exists an absent seal is. Stated limit: reverting **both** the registry and its sidecar defeats it, which is why the chain digest is itself pinned in `tests/test_holdout_seal.py` (282 lines, 198,027 bytes, chain `ac422774…4b96`). | `audit-only` |
| [`benchmarks/quantum_compilation/quantum_signal_power.json`](../benchmarks/quantum_compilation/quantum_signal_power.json), [`scripts/measure_quantum_signal_power.py`](../scripts/measure_quantum_signal_power.py) | The two numbers the five quantum-signal tranches never reported, computed from their records alone (no panel read, no holdout slice touched). **Statistical power:** `minimum_detectable_sharpe` inverts the same `compute_deflated_sharpe_ratio` the records use -- it is the true annualised Sharpe at which the design's DSR reaches the 0.95 floor given the bars it had and the trials it charged -- and `power_at` gives `P(SR_hat >= MDS)` under `SR_hat ~ N(s, se(s))`. **Programme charge:** every row of every tranche re-priced against the trials all five tranches spent (`total_trials`) with the dispersion pooled at the row's own resolution, reporting `hurdle_as_run` beside `hurdle_programme` and `rows_that_become_significant`. Both figures are reported per resolution because a daily and a 5-minute Sharpe estimate different quantities. The script refuses to write the record unless the DSR at the MDS lands on the floor, power is monotone, the programme total equals its parts, no hurdle falls, no row becomes significant, and the DSR it recomputes reproduces every tranche's recorded DSR. `tests/test_quantum_signal_power.py` (10). | `audit-only` |
| [`apedp_dyn/epistemic/temporal_rate_modulation.py`](../apedp_dyn/epistemic/temporal_rate_modulation.py) | Hypothesis B temporal-rate modulation null test: subtracts each clock's calibration-measured ordinary drive response, then tests whether a phase-locked fractional-frequency residual is common to clocks of at least two distinct compositions. Shared-phase Fourier surrogates plus injected calibration uncertainty define the null; a heterogeneity guard rejects oscillator-specific artifacts; uniform sampling, calibration/confirmation dataset-id reuse, same-frequency calibration, and overlapping drive harmonics raise protocol errors. The drive phasor is fit jointly with a constant and linear trend so finite-window leakage does not bias calibration transfer. Calibration refits a fresh calibration per trial and reports exact Clopper-Pearson FPR/power bounds plus artifact/any-false-claim rates; the preregistered gate is the point-estimate FPR/power floor. Verdicts are candidate/null/insufficient only — a candidate is not a time-travel claim and can arise from an unmodeled common-mode systematic. Preregistered thresholds are imported from the canonical spec; synthetic injection-recovery calibrates the detector's FPR and power. | `experimental` |
| [`apedp_dyn/epistemic/temporal_rate_io.py`](../apedp_dyn/epistemic/temporal_rate_io.py) | Strict CSV + sidecar-JSON and compressed NPZ ingestion for the temporal-rate null test: schema validation, `allow_pickle=False`, finite/monotonic checks delegated to `ClockDataset`, and a canonical SHA-256 fingerprint. Also emits a `temporal-rate-v2` frozen preregistration artifact (fingerprints, dataset ids, drive frequencies, clock compositions, `harmonics`, `n_surrogates`, `seed`, canonical alphas, minimum fractional amplitude, and canonical spec hash) and fails closed if either dataset or any analysis parameter is changed before adjudication. The artifact is a fingerprint commitment, not a signature. | `experimental` |
| [`apedp_dyn/epistemic/trial_ledger.py`](../apedp_dyn/epistemic/trial_ledger.py) | Candidate trial ledger computing Bailey & López de Prado (2014) Deflated Sharpe Ratio, with selection-adjusted multiplicity (`effective_trial_count` charges disclosed `selection_pool_size` per `search_id`). | `production` |
| [`apedp_dyn/epistemic/multiple_testing.py`](../apedp_dyn/epistemic/multiple_testing.py) | Exact Benjamini-Hochberg, Benjamini-Yekutieli and Holm corrections over a family of p-values, reporting the **effective threshold** a hit had to clear. The Sharpe-based DSR in `trial_ledger.py` does not transfer to a p-value panel; this is the instrument for a biomarker or gene screen. Refuses NaN, out-of-range and unknown methods. See [`docs/BIOPHARMA_SCOPE.md`](BIOPHARMA_SCOPE.md). | `production` |
| [`docs/SCAFFOLD_INVENTORY.md`](SCAFFOLD_INVENTORY.md) | Tier inventory for every applied and infrastructure module that is a sketch rather than a model: `computed`, `rule_screen`, `literature_lookup`, `planning_template`, `reference_spec` or `refused`, with the constant that used to be returned as a prediction named in each case. Pinned by `tests/test_scaffold_inventory.py`. | `audit-only` |
| [`apedp_dyn/orchestration/modality_sieve.py`](../apedp_dyn/orchestration/modality_sieve.py) | Level-5 Pillar 1: autonomous modality & topology sieve — routes raw data (static manifold / ODE kinetic / PDE field / Hawkes bursty events) across structural evidence detectors (isochronous monotone time axis, bounded relative curvature, log-axis 2D spectral dispersion, Fano burstiness) and MASKS ill-posed engine families before any proposer runs; fail-closed on conflicting evidence. Routing heuristics only — acceptance authority stays with the referee. | `experimental` |
| [`apedp_dyn/orchestration/model_class_selector.py`](../apedp_dyn/orchestration/model_class_selector.py) | EHI Pillar 1 redesign: probabilistic model-class selector — P(M\|D) Bayesian comparison over DOCUMENTED per-class diagnostic profiles (Gaussian likelihoods on time-axis / kinetic curvature / 2D spectral dispersion / Fano burstiness / inter-arrival autocorrelation, with per-class no-update-when-not-applicable discipline); sharp evidence routes STATIC_RELATION/ODE/PDE/HAWKES/POISSON_RENEWAL, and ambiguity stays a first-class OUTCOME (MULTIPLE_CLASSES_PLAUSIBLE, INSUFFICIENT_METADATA) below the documented posterior + Bayes-factor floors; the fabrication guard refuses a decisive route when the winning class's DEFINING diagnostic never fired (competitive misfit is not positive support). Selection-layer routing only — the referee remains the acceptance authority. Pins: `tests/test_dyn_ehi_phase5_pins.py`. | `experimental` |
| [`apedp_dyn/orchestration/dispatch.py`](../apedp_dyn/orchestration/dispatch.py) | Discovery dispatch: wires the modality sieve + model-class selector so proposals route AUTOMATICALLY from a raw unannotated frame — sieve verdict ADMITS/MASKS families (ill-posedness masking), the selector confirms; decisive agreement ⇒ ROUTED to the one engine; selector ambiguity ⇒ MULTIPLE_ROUTES keeping every admitted family live (mask honored even with wired runners); decisive disagreement ⇒ CONFLICT (abstention, never an override); unresolved structure ⇒ UNROUTABLE with open questions. Default runners dispatch the repo's real engines (L2 open discovery / ODE diff-Padé / frozen-operator PDE / Hawkes MLE ≥50-stamp floor); per-family failures recorded in `errors`, never laundered; products are referee-facing material only — no adjudication vocabulary. Pins: `tests/test_dyn_discovery_dispatch_wiring.py`. | `experimental` |
| [`apedp_dyn/orchestration/competition.py`](../apedp_dyn/orchestration/competition.py) | Engine competition for the upload sandbox: runs every applicable engine (currently the two L2 engines) on the same training split and returns each result unranked, side by side. An engine that does not apply is recorded `NOT_APPLICABLE` with a reason; a per-engine exception is recorded `FAILED` without aborting the rest. Promotion stays with the referee; the helper never picks a winner. Exposed as the `competition` array on `POST /api/sandbox/upload`. Pins: `tests/test_dyn_engine_competition.py`. | `experimental` |
| [`apedp_dyn/epistemic/active_experimentalist.py`](../apedp_dyn/epistemic/active_experimentalist.py) | Level-5 Pillar 3: closed-loop active experimental disambiguation — x* = argmax pairwise divergence − k·σ (k imported from the frozen E4 policy), deterministic clustered target-range acquisition orders, fail-closed out-of-band point rejection, residual referee handoff, PLUS the EHI external-review redesign: expected-information-gain orders I(H;Y\|x) under feasible ranges / cost / noise (exact Gaussian-mixture quadrature), field-complete `ExperimentOrder` requests, fail-closed zero-noise refusal, and honest stop when no order beats the documented stop floor. No promotion or rejection is decided inside this layer. Pins: `tests/test_dyn_ehi_phase5_pins.py`. | `experimental` |
| [`apedp_dyn/epistemic/enclave_referee.py`](../apedp_dyn/epistemic/enclave_referee.py) | Level-5 Pillar 4: confidential adjudication — Stage-C pair verdicts computed in a fresh isolated interpreter over permission-sealed inputs (aggregates only), with an Ed25519-signed attestation binding verdict + request SHA-256 + aggregates; honest "process-isolated" label (TEE claimed only by explicit runtime declaration), UNRESOLVED on any infrastructure failure. | `experimental` |
| [`apedp/dimensional_invariant_engine.py`](../apedp/dimensional_invariant_engine.py) | Level-5 Pillar 2: self-supervised dimensional & boundary synthesis — autonomous unit→exponent conversion feeding the audit-proven Buckingham-Π engine (exact integer nullspace, no SVD tolerances), log-band empirical limit estimators (origin pin q(0)=0; finite saturation L steering the rational family; fail-closed on ambiguous curves), the interpolated half-max characteristic scale with the emitted {s*/3, s*, s*·3} search bank, PLUS the EHI external-review redesign: `classify_constraints` emits boundary conditions as HYPOTHESES with epistemic STATES (`KNOWN` / `EMPIRICAL` / `UNRESOLVED` / `REJECTED`) and carries `is_physical_claim=False` for training-partition evidence — constraints are never auto-enforced facts. Configuration-layer constraints only — never adjudication. Pins: `tests/test_dyn_ehi_phase5_pins.py`. | `experimental` |
| [`apedp_dyn/epistemic/research_controller.py`](../apedp_dyn/epistemic/research_controller.py) | EHI Pillar 5 (fifth pillar): Scientific Reasoning & Research Control Layer — SEPARATED scientific belief state (revisable with evidence) from the governance state (referee's pinned constants hashed at lock; re-verified before every planned action; tamper raises), deterministic utility planner over {inspect problem, test constraints, information-gain experiment, improve data quality, ABSTAIN}, five-part evidence ledger (established / assumed / unidentifiable / resolve-next / must-not-claim), controlled exit (ABSTAINED / BLOCKED_ON_PROBLEM_SPECIFICATION / EVIDENCE_INTEGRATED_AWAITING_REFEREE — never self-promotion), and in-run protocol revisions returned as versioned PROPOSED_NOT_APPLIED artifacts. Pins: `tests/test_dyn_ehi_phase5_pins.py`. | `experimental` |
| [`apedp_dyn/de_arm/typed_model.py`](../apedp_dyn/de_arm/typed_model.py) | DE-discovery arm (separate arm, phase 1), layer A: typed mathematical semantics — STATIC_EXPRESSION / ODE_SYSTEM (declared states + independent variable + dt + initial conditions) / PDE_SYSTEM (fields + at least one spatial axis with a positive dx + declared boundary treatment); every incomplete declaration is a fail-closed refusal, making typed structural checks possible BEFORE any search runs. Pins: `tests/test_dyn_de_arm_p1_pins.py`. | `experimental` |
| [`apedp_dyn/de_arm/ad_service.py`](../apedp_dyn/de_arm/ad_service.py) | DE-discovery arm, shared infrastructure: symbolic AD service over a CLOSED typed expression vocabulary (Const/Var/Add/Sub/Mul/Div/Pow-constant/Tanh/Sin/Cos/Exp/Log with declared domain refusals) — exact product/quotient/chain rules, honest limits documented in-module (AD differentiates the implemented candidate computation, NOT the physical system; measured data keeps its existing finite-difference stencils as the residual source). Refuses unknown node kinds, never approximates. Pins: `tests/test_dyn_de_arm_p1_pins.py`. | `experimental` |
| [`apedp_dyn/de_arm/dependency_screen.py`](../apedp_dyn/de_arm/dependency_screen.py) | DE-discovery arm, layers B/C: Level-1 dependency-graph screening — per-state predictive-increment ranking of candidate parents (lag-embedded relative-RSS gains, TRAINING partition only, deterministic and dependency-free) with the honest ROADMAP disclaimer (a predictive ordering, NOT a causal-topology verdict — review correction B), the MANDATORY escape hatch (the dense kernel result from `discover_coupled_system` always ships beside the screened mirror fits in ONE referee-facing `OdeArmOutcome` — deselecting nothing, deciding nothing), and a documented python ridge+prune STRidge mirror over the REDUCED library (provenance names the implementation; kernel parity never claimed). Wired as the dispatch surface's ODE route for coupled systems. Pins:
`tests/test_dyn_de_arm_p1_pins.py`. | `experimental` |
| [`apedp_dyn/de_arm/mesh_discovery.py`](../apedp_dyn/de_arm/mesh_discovery.py) | DE arm, PDE prototype layer (activates the `pde_mesh_backlog` gate as an experimental prototype): cotangent-Laplace-Beltrami discovery on 2D Delaunay meshes — geometry-carrying operator (cotangent weights + barycentric vertex areas; constant AND affine nullspace on interior rows, edge-weight symmetry, interior-edge cotangent positivity and non-negative Dirichlet energy all pinned; L·1 = 0 exactly by construction), time-derivative residual regression per the arm's documented python STRidge mirror over the mesh library {1, u, u², u³, Δu(, u·Δu)} on a 2-ring interior mask. Fail-closed on degenerate assemblies (sliver/repeated-vertex/zero-area refusals, unsupported boundaries or area kinds, too few frames, non-finite panels, under-excited targets, too-small meshes). Single-term heat recovery pinned on an `expm_multiply` trajectory of the assembled operator itself (κ̂ within 1e-5 relative of truth; mirror prunes to {1, Δu}); the measured mirror alias limitation on identifiability-poor panels is documented in `docs/pde_mesh_discovery_prototype.md`. The structured-grid Fornberg route stays the established path for grid-runnable data; NOT a production claim — prototype is the evidence base for the gate's frozen-benchmark decision. Pins: `tests/test_dyn_de_arm_mesh_prototype.py`. | `experimental` |
| [`benchmarks/run_de_arm_vs_dense_benchmarks.py`](../benchmarks/run_de_arm_vs_dense_benchmarks.py) | DE-arm frozen benchmark vs the dense flat-library kernel path (deterministic RK4 fixtures of Lotka-Volterra, FitzHugh-Nagumo, Lorenz; seeded; noise-free). At the default screen depth both library kinds are structurally exact on all three systems (0 false exclusions); the library-reduction hypothesis is MEASURED (0–25% at D≤3, no universal figure claimed); the aggressive top_k=1 screen drops true parents on Lorenz with visible R² collapse, demonstrating the dense escape hatch is load-bearing. Reports the recorded fixture-integrity finding (a narrow slow-manifold window yields exact-but-non-physical R²=1.0 fits in BOTH engines; the harness now refuses under-swept full windows AND train partitions). Evidence-only — computes no acceptance. Results: `benchmarks/de_arm_vs_dense_benchmark_results.json`; write-up: `docs/de_arm_vs_dense_benchmarks.md`. | `audit-only` |
| [`benchmarks/run_feynman_noise_rediscovery.py`](../benchmarks/run_feynman_noise_rediscovery.py) | Noise-vs-rediscovery benchmark (2026-10-23, audit doc ASSESSMENT §3 fix-order #1): blind planted-world sweep over 18 static scalar laws (13 embedded engine-grammar families — monomials, reciprocals, exponentials, saturating Langmuir/Hill, trig, two-variable linear+product — plus the 5 curated `apedp.ingestion` Feynman cases) × relative target-noise σ ∈ {0, 0.01, 0.02, 0.05, 0.10} × repeats. Preregistered harness-local gates (NOT imported from the canonical promotion spec): predictive = clean-holdout R² ≥ 0.95; structure = R² ≥ 0.90 on a denser probe grid over declared world ranges (train-range alias detector); verdicts recovered_structure / recovered_predictive_only / failed / abstained with abstention never counting toward recovery (known-truth rule). Fixed-design protocol: design seed depends on (world, rep) only, noise seed on (world, σ, rep) — same design points, new noise phase across the axis. Engine adapters are pluggable and blind: `synthesis` (univariate `AutonomousModelSynthesizer`, `sp.Symbol('x')` API + `SynthesizedModel` lambdify evaluation) and `rime_ad` (multivariate `RimeAdProposer`, fans out through the native kernel; skipped with an explicit reason when the wheel is absent). Evidence-only — computes no acceptance and consumes no holdout resource. Artifact: `benchmarks/feynman_noise_rediscovery_results.json`; pins: `tests/test_feynman_noise_rediscovery_harness.py`. | `audit-only` |
| [`tests/test_dyn_graph_census_pins.py`](../tests/test_dyn_graph_census_pins.py) | Graph-census gap closure (2026-10-22): pins the three previously UNTESTED graph surfaces — W3C PROV-lite `ProvenanceGraph` (nodes + links recorded verbatim, no shared-mutable-default leak), `plot_epistemic_graph` (real deterministic PNG render, magic bytes verified), and the `ak.rime_rewrite_json` PyO3 FFI binding (WrapNonlinearity appends + makes output per rewrite.rs `apply`; bad operand index rejected, malformed JSON fails closed). Census record + corrected stale rows + rationale'd backlog: `docs/graph_census_2026_10_22.md`. | `audit-only` |
| [`benchmarks/run_cbf_scaling_profile.py`](../benchmarks/run_cbf_scaling_profile.py) | CBF O(N²) scaling profiler and proximity-graph BACKLOG GATE (2026-10-22): deterministic A3-guarded fixture; separates the certified QP's factory-lemma 1-pass FLOOR (9.9 ms at N=256 — barriers build + all-pairs sweep alone) from adversarial-iteration growth (~8× more), and the heuristic stepper's ~6 µs/call Python pair-loop cost (5 ms budget crossed at N=32, where VECTORIZATION not a proximity graph is the fix). k-NN pair-reduction ratios (0.38→0.011 at k=4) recorded as computational-counterfactual estimates with both qualifications: the dense-build trap (pruning after the [N,N,2] build keeps the O(N²) floor) and NO soundness claim for proximity-restricted certified sets. Gate record + write-up: `docs/cbf_proximity_graph_backlog_gate.md`. Post-vectorization re-measured addendum: exponent 1.91→1.06, 5 ms crossing N=32→N=128, 26× at N=256. | `audit-only` |
| [`benchmarks/run_langmuir_referee_demo.py`](../benchmarks/run_langmuir_referee_demo.py) | Referee-routed isotherm tournament (2026-10-22): replaces the dev-MSE "tournament" with the preregistered gates on a live NIST ISODB isotherm (methane on zeolite 4A, 323 K, 156 points; `NIST-ISODB/isodb-library` mirror, no local fixture). Each proposer gets ONE pre-registered, disjoint 39-row confirmation slice declared before any holdout is drawn; the ledger is consulted BEFORE the slice is read, so a burned slice is never re-read. Incumbent is the NESTED two-site Langmuir (train RMSE 0.0096 vs 0.1185 single-site; site 1 sits in the Henry limit, only q1*K1=0.0198 is identified). Measured: against the single-site law the Rust GA/GP passes Gate 4 5/5 (DM 6.654, p=1.4e-11) and RIME-AD 3/5 — against the two-site incumbent BOTH fail 0/5 (DM −6.5…−9.5, p=1, RIME-AD incumbent-vs-candidate RMSE 0.0098 vs 0.1131) and both slices burn, so the single-site "win" was a baseline-specification artifact, not a discovery. Extrapolation audit (no ground truth out of range; admissibility only): the contender is negative at the origin (q(0)=−0.0664) and unbounded — it exceeds the largest observed uptake at 74.2 bar (1.07× p_max) and the classical capacity at 123.4 bar (1.78× p_max) — so it is an interpolant of the measured window, not a physical isotherm; the two-site fit is the fairer INTERPOLANT but its Henry-limit site is also unbounded, so neither fit licenses extrapolation. Second block: coupled ODE discovery on a real Lotka-Volterra RK4 trajectory, certified by `OdeReferee` on a 240-point sealed holdout (mean/worst R² = 1.000); the noise sweep separates the two claims — structural identity holds to 2% of state RMS, the PREDICTIVE certificate still passes at 5–10% (labelled `PREDICTIVE ONLY (structure wrong)`) and refuses at 20% (mean R² 0.841 < 0.95). Verdicts journalled in `get_ledger_path("langmuir_referee_demo.jsonl")`. Evidence-only, computes no acceptance. | `audit-only` |
| [`benchmarks/pow_birth_fit_benchmark.py`](../benchmarks/pow_birth_fit_benchmark.py) | Birth-fit Pow-exponent A/B at benchmark scale (2026-10-23): two kernels differing only in `has_pow_const` (live vs false), same exponent grid, polish, seeds and budgets; deterministic evaluation counts identify each arm, plus the installed `.so` sha256 (`on` 23a50d2e…, `off` fc1f9603…). Measured: +11.4% total evaluations on External-60's frozen 300×50×4 config (all 60 problems; median 62,400 vs 56,094) and +14.2% / +31.5% median evaluations at Gate 4's 2×30×15 (Rabi / Brus), for paired validation accuracy that is a wash (23 problems better, 32 worse, 5 tied/non-finite; median ratio 1.02) and no Gate-4 verdict change (Rabi identical, Brus fails in both arms). Counters wrapped around `evolve_native`/`evolve` show the synthesis engine and the Landauer harness make 0 native-GA calls, and the synthesis control's wall movement between arms (19% one way in one repeat pair, 5.6% the other way in the other) sets this host's noise floor, against which the External-60 wall deltas (+9.8% and +23.9% across repeat pairs) are not evidence of wall cost. Artifact: `benchmarks/pow_birth_fit_benchmark_results.json`; full table in `docs/ga_seed_fragility_audit.md`. Evidence-only. | `audit-only` |
| [`benchmarks/trig_birth_fit_planck_benchmark.py`](../benchmarks/trig_birth_fit_planck_benchmark.py) | Trig-template birth-fit experiment (2026-10-23, measured and NOT shipped): two variants against the shipped kernel — `full` fits both `sin|cos(f*x)` and phase-bearing `sin|cos(f*x + p)` roots, `phase-only` fits the coupled form only (root-only in both; nested trig calls are not the seeded template). Full: phase-shifted exact 1/11 → 3/11 and median MSE 5.29e-6 → 3.66e-7, but +90–161% evaluations, worse Planck medians (holdout R² -31.5 → -76.4, extended -0.52 → -2.03) and both Planck must-fail pins fail at the pinned seed (0.546 / 0.975). Phase-only: Planck pins safe and stronger (max -0.60 / 0.39) but plain `2cos(1.3x)` recovery drops 11/11 → 8/11 (small budget 5/5 → 1/5). Neither trade is justified by a 1/11 → 3/11 gain, so `genetic.rs` is unchanged; artifacts `benchmarks/trig_birth_fit_planck_results.json` and `..._phase_only_results.json`, tables in `docs/ga_seed_fragility_audit.md`. Also fixes the shared `_r2` helper (1 − SSE/SST; `--recompute` corrects recorded arms without re-running). | `audit-only` |
| [`apedp_dyn/robotics/proximity_margin.py`](../apedp_dyn/robotics/proximity_margin.py) | Non-neighbor barrier MARGIN argument, pitched and machine-checked (experimental, evidence-only, wired to no pipeline): pairs at h ≥ h* = 2·v_max/γ need NO constraint element (box-bounded worst case −2·v_max can never violate hdot ≥ −γ·h there — saturation EXACT at h = h*); `solve_cbf_qp_proximity` solves the alternating-projection QP over the RESTRICTED pair set (provenance: python mirror of the certified schedule; kernel parity not claimed) and releases ONLY through the FULL all-pairs fail-closed re-check (the margin argument re-verified numerically at release); every pair re-decided FREE-or-CONSTRAINED each step from the h matrix the barriers build already computes (no extra pair-distance work). Discrete invariance extends to all pairs: constrained by certification, free by the margin algebra. NOW COMPLETES the gate's limitations: `indexed_proximity_pairs` (scipy cKDTree candidate radius r_safe + h* + slack — candidate SUPERSET soundness pinned; beats the dense [N,N,2] build floor 3.1–5.8× at N=256–512), `indexed_barrier_components` (exact sparse mirror, array-equal pinned), and `model_check_unicycle_horizon_proximity` (proximity-LOCKED realization through the production arc stepper on adversarial horizons — head-on top-speed closing and multi-agent closing grids hold min h > 1e-3; measured evidence, not a formal claim). Composed restricted solve measured 3.3–3.6× over the dense QP at N≥256; promotion checklist in `docs/cbf_proximity_graph_backlog_gate.md`. Pins: `tests/test_dyn_proximity_margin_pins.py` (10). | `experimental` |
| [`apedp_dyn/robotics/multi_agent_transport.py`](../apedp_dyn/robotics/multi_agent_transport.py) — `pairwise_repulsive_forces` | Vectorized all-pairs pair-force sweep replacing `multi_agent_step`'s per-pair Python loop (the measured ~6 µs/invocation bottleneck crossing the 5 ms budget at N=32): EXACT per-pair mirror of `compute_cbf_repulsive_force` — degenerate [k_rep, 0] fallback, r_warning zeros filter (the mask IS the proximity filter a graph would replicate), head-on tangential term at anti > 0.7; pinned branch-exact per pair plus whole-step equality against an inline legacy reference across N sizes in `tests/test_dyn_multi_agent_vectorized_pins.py`; re-measured 5 ms crossing N=32→N=128 (exponent 1.91→1.06), 26× at N=256. | `experimental` |
| [`apedp_dyn/epistemic/promotion_lifecycle.py`](../apedp_dyn/epistemic/promotion_lifecycle.py) | Six-stage scoped promotion state machine with non-fungible sequential gates. | `production` |
| [`apedp_dyn/epistemic/novelty_adjudication.py`](../apedp_dyn/epistemic/novelty_adjudication.py) | Decoupled Diebold-Mariano predictive incrementality and literature orthogonal novelty tests. | `production` |
| [`apedp_dyn/epistemic/lean_universal_bridge.py`](../apedp_dyn/epistemic/lean_universal_bridge.py) | Zero-disk in-memory Lean 4 formal verification kernel bridge via `--stdin` streaming; requires an external Lean 4 toolchain (environment-gated). | `experimental` |
| [`apedp_dyn/epistemic/adjudicate.py`](../apedp_dyn/epistemic/adjudicate.py) | Central epistemic referee adjudicating multi-gate candidate admissibility. | `production` |
| [`apedp_dyn/epistemic/gate4_provenance.py`](../apedp_dyn/epistemic/gate4_provenance.py) | CORE-SIDE Gate-4 provenance producer (2026-10-22) — the seam that lets `adjudicate()` run on ordinary proposer output instead of a sandbox harness. `measure_proposal` scores one `CandidateProposal` and returns the error metric the proposal contract omits (MSE/RMSE/MAE/R² plus the signed residual series, non-finite pairs dropped jointly, `ProposalNotEvaluableError` below two usable points); `gate4_status` emits the exact key set the referee reads (`passed`, `detail`, `holdout_obs_rms`, `holdout_noise_sigma`, `holdout_n_samples`, `rmse`, `holdout_residuals`) and marks `passed` ONLY when a supplied incumbent prediction loses the Diebold-Mariano test at the canonical alpha — no incumbent means the gate is recorded closed with the reason, because an untested claim is not a pass; `scored_candidate` bridges proposal → `ScoredCandidate` (train MSE as rank key, measured holdout RMSE). Disclosed boundaries: the noise scale is never invented (`noise_sigma_from_residuals` is an explicit opt-in MAD estimator, and an absent scale makes the referee abstain), and canonical duplicate detection is unavailable for free-form proposal formulas, recorded as `canonicalisation: untyped` rather than faked. Pins: `tests/test_dyn_gate4_provenance.py` (7). | `experimental` |
| [`apedp_dyn/epistemic/scope_refusal.py`](../apedp_dyn/epistemic/scope_refusal.py) | Referee-side out-of-grammar scope refusal (v0.1.4, spec §16d): preregistered family-distinct probes (exp rate 1–50 on dev-normalized axis, log on raw axis) fit from dev data only; an emitted candidate promotes only on UNIQUE_SUPPORT over every probe; blind to world flags and planted truth. | `experimental` |
| [`apedp_dyn/epistemic/scope_refusal_mv.py`](../apedp_dyn/epistemic/scope_refusal_mv.py) | Multivariate rival-probe construction for referee-side scope refusal (spec §16e candidate): three declared families (periodic-quadratic, abs-linear2, amplitude-modulated product) with full linear nuisance span, fitted from dev frames only; adjudication reuses the §16d contracts untouched. **NEGATIVE RESULT:** the registered two-sided validation (PREREG-RIME-SCOPE-MV-2026-10-17) failed — small-frequency probes linearize into products (wrongful refusal of the exact in-family control) and the frozen variance guard is undefined on §16c axis scales; do not use until a v2 prereg passes. | `experimental` |
| [`apedp_dyn/epistemic/scope_refusal_mv_v2.py`](../apedp_dyn/epistemic/scope_refusal_mv_v2.py) | Multivariate rival-probe construction, v2 remediation for spec §16e (PREREG-RIME-SCOPE-MV-v2-2026-10-17): standardizes interaction coordinates and bounds phase search away from zero (c ≥ 1.0 for periodic-quadratic, c ≥ 1.35 for amplitude modulation) to prevent in-grammar monomial linearization, and replaces the absolute variance guard with a scale-relative check that builds probes on micro-scale axes. Reuses the frozen §16d adjudication contracts unchanged; registered v2 tests promote the in-family control across seeds 700–702 and decline W1/W2/W3. | `experimental` |
| [`apedp_dyn/epistemic/claim_ladder.py`](../apedp_dyn/epistemic/claim_ladder.py) | Epistemic claim ladder calibrating warrant strength from speculative to admitted. | `production` |
| [`apedp_dyn/epistemic/proof_claim.py`](../apedp_dyn/epistemic/proof_claim.py) | `ProofClaimStatus` (P0 no proof / P1 candidate / P2 verified): the proof-claim tier, independent of empirical confidence. Defined here so the Lean bridge does not import the legacy root; re-exported by `apedp/mathematical/mathematical_layer.py`. | `production` |
| [`apedp_dyn/rust_bridge.py`](../apedp_dyn/rust_bridge.py) | Rust numerical kernel bridge: one interface over `apedp_kernel` with the Python reference engine as fallback, compile caches, kernel-path tallies and native-search enforcement. Moved here so `apedp_dyn` does not import the legacy root; `apedp/rust_bridge.py` forwards to it. | `production` |
| [`apedp_dyn/native_search.py`](../apedp_dyn/native_search.py) | The Python-search fallback contract: `NativeSearchUnavailableError` plus a provider registry the legacy tree fills in, so the bridge can offer the fallback without importing `apedp.evolutionary_search`. | `production` |
| [`apedp_dyn/safety.py`](../apedp_dyn/safety.py) | Hardened string-in formula parsing (`safe_parse_expr`, `safe_parse_pde`) and wall-clock-bounded symbolic limits; stdlib plus sympy only, with `apedp/safety.py` as its compatibility façade. | `production` |
| [`apedp_dyn/epistemic/cross_asset_gate.py`](../apedp_dyn/epistemic/cross_asset_gate.py) | Multi-asset cross-sectional transfer gate enforcing zero contamination across symbols. | `production` |
| [`apedp_dyn/quantum_compilation/contracts.py`](../apedp_dyn/quantum_compilation/contracts.py) | APEDP-Q verifier contracts: byte-pinned SHA-256 trees for the competition verifier and `kickmix` extension, AST-extracted benchmark constants, and `assert_locked()` re-check before every official run. | `experimental` |
| [`apedp_dyn/quantum_compilation/local_mr_sim.py`](../apedp_dyn/quantum_compilation/local_mr_sim.py) | Local Z-basis (value + sign) simulator for the DQI decoder with the instruction `kickmix` lacks: `mr` (Z-basis measure-and-reset, phase-preserving), beside `r` (clears but randomizes the phase) and `hmr` (X basis). Implements the decoder's builder subset (`unary_iteration`, `init_and`/`del_and`, `iadd`/`isub`, `flip_if_equal`/`flip_if_less_than`, rotations, GF(2^b) arithmetic and `gf2_phase_by_product`) with semantics pinned against `kickmix`, and reproduces `kickmix` shot for shot on the reference decoder (syndromes restored, junk qubits `|0>`, identical tracked phase). **Cannot score:** it computes values and phases, not gate counts, and receipts come only from `kickmix` plus the official verifiers. | `audit-only` |
| [`apedp_dyn/quantum_compilation/statevector_sim.py`](../apedp_dyn/quantum_compilation/statevector_sim.py) | Exact amplitude-level tier for the DQI gadgets: a dense statevector with the gates the read-then-gate arms are built from (`x`, `z`, `h`, `cz`, `cx`, `ccx`, `cswap`, arbitrary 1-qubit unitaries), Z-basis measure-and-reset with the reset a real `MR` performs, and exact branch enumeration over measurement outcomes with probabilities. Used to settle what the campaign's basis-state licence covers (`docs/VQC_AMPLITUDE_TIER.md`): identical to the controlled gate on all eight basis inputs of the gadget, a complete dephasing of a superposed control (fidelity 0.500, control off-diagonal 0.500 → 0.000), and `hmr` is not `mr` while `h; hmr` is. **Not a competition engine and not a receipt path:** 26 qubits is the cap, and the arm's 1,913 qubits would need 10^575.87 amplitudes. | `experimental` |
| [`apedp_dyn/quantum_compilation/quantum_signal.py`](../apedp_dyn/quantum_compilation/quantum_signal.py) | A quantum circuit's output used as a trading signal, executed under the canonical fee accounting. `QuantumFeatureMap` applies `Ry(pi*x_i)` on every wire then `CX(i, 0)` for every `i >= 1`, so `<Z_0> = prod_i cos(pi*x_i)` and **every** feature reaches the measured marginal — the check that the earlier `phase`-then-`h`-then-`cx` demonstration failed, its output being the constant 0.0. `ParameterisedQuantumMap` is the fitted form, `Ry(alpha_i x_i + beta_i)` fan-in with closed form `prod_i cos(alpha_i*x_i + beta_i)` valid for every parameter value; `fit_parameterised_map` scores each point of a declared 12-point grid on the training slice and writes **one trial per grid point** to its own `CandidateTrialLedger` with `selection_pool_size` set to the grid size, so the fit is charged at the size of its search rather than as its winner. `execute_signal` re-implements the ledger's `gross = pos*fwd`, `fee = FEE*|dpos|`, `net = gross - fee` with the identity checked per run, and reproduces `run_ledger` **row for row** on every dataset before any quantum number is quoted. Record: 10 datasets over two resolutions (6 daily, 4 x 15-minute), **190 trials** = 60 threshold rows + 10 baselines + 120 parameter fits. The hurdle is built from a **within-resolution** dispersion (`var_trials` 0.5235 at 1 day, 258.53 at 15 minutes; pooled 358.76 kept for comparison) while every row is charged all 190 trials: `expected_max_sr` is 1.989 daily and 44.196 at 15 minutes, and **no row clears 0.95** (best daily DSR 0.234, the SOL momentum baseline). A second ansatz depth is implemented (`ANSATZ_DEPTHS = (1, 2)`; depth two has **no product form** — 0.7146 against depth-one-squared 0.0002 — so `analytic` refuses above one layer and `exact_marginal` builds the dense unitary instead). Holdout is a 70/30 temporal split fitted on training bars only, decided once per dataset in a durable `FullLineageExperimentLedger`: **8 of 10 slices burned**, the other two confirmed (SOL +1.4285, NVDA +0.4589), with post-burn `HoldoutContaminationError` recorded; `holdout_decision` now charges the slice to the fit's own search size and takes effect on the next tranche. **Tranche 2** is a second record on unseen slices (six unused daily assets, four symbols at 1 hour) with **310 trials** (60 threshold + 10 baselines + 240 two-depth fits): no row clears 0.95 (best daily DSR 0.031 against a 2.193 hurdle), **all ten slices burned** under the charged rule — ADA@1d nets +0.5649 with 216 units of turnover and is still rejected below its own hurdle — and the paired family read finds depth two winning **1 of 10** datasets (mean holdout delta −0.3134), so the second family does not pay for its search here. `significance_report` reports each resolution's verdict side by side (tranche 1: 1.989 against 44.196; tranche 2: 2.193 against 10.653). **Disclosed boundary:** every map here is classically computable in O(n) arithmetic and the fits are 24-point discrete grids over two depths, not quantum learning. `benchmarks/quantum_compilation/quantum_signal_ledger.json`, `benchmarks/quantum_compilation/quantum_signal_ledger_tranche2.json`, `benchmarks/quantum_compilation/quantum_signal_holdouts.jsonl`, `docs/QUANTUM_SIGNAL_LEDGER.md`, **Tranche 3** fits unseen data (six new daily assets, four symbols at 30 minutes) with a **continuous** optimiser: `fit_continuous_map` records every evaluation as a trial with the declared 17-per-depth budget as `selection_pool_size`, so **410 trials** = 60 + 10 + 340 evaluations (34 per dataset, `hidden_trials = 6` from skipped boundary probes); selected alphas are off-grid (0.8, 1.2, 1.8, 0.2), nothing clears 0.95 (hurdles 2.26 and 32.9), and all ten slices burned (registry 28), AMZN@1d rejected at +0.1704 below its hurdle. `cost_sensitivity` re-prices every priced trial in all three records at 5/10/20 bps exactly from `fees = turnover * rate` without reading a bar or a slice, and states that Sharpe/DSR are not re-priced on those records because their per-bar series is not stored. **Tranche 4** judges unseen assets across regimes: `walk_forward_windows` gives two contiguous, non-overlapping folds per dataset (expanding training prefix, one block tested each), so a slice is consumed once while the dataset is read twice — six new daily assets and four at 1 hour, **820 trials** (120 threshold + 20 baselines + 680 evaluations), hurdles 3.62 / 16.85, no row clearing 0.95, **20/20 fold slices burned** (registry 48), each with its own slice id and burn. The sweep also reports the bill per resolution at 5 bps (`gross_by_resolution`, `fees_by_resolution`, `fee_share_of_gross`, `fees_over_absolute_gross`): tranche 1 daily fees are **119% of gross**, tranche 2 at 1 hour **1,171%**, tranche 3 at 30 minutes **1,166%**, tranche 4 at 1 hour **363%** — the arithmetic behind a one-bar holding period. **Tranche 5** stores what the earlier records did not: `quantum_signal_series_tranche5.jsonl` holds, per priced row, the **per-bar net at 5 bps** and that bar's **|Δpos|** (38,248 bars over 140 rows), so `net(r) = net(5 bps) − (r − 5 bps)·|Δpos|` is exact bar by bar and Sharpe/DSR are re-priced at every rate (`sharpe_by_resolution`, `dsr_by_resolution`, `rows_clearing_floor`). Measured on MRK, UNH, BAC, COST, GLD, IWM daily and AXS, SNX, COMP, CRV at **5 minutes**, two folds, **820 trials**: daily median Sharpe −0.678 / −1.101 / −2.060 and 5-minute −51.574 / −94.831 / −160.350 at 5/10/20 bps, positive rows 31 / 20 / 8, **no row clearing 0.95 at any rate**, hurdles 4.04 and 78.38, 20/20 fold slices burned (registry 88). At 5 bps the fee is **934% of daily |gross|** and **6,038% of 5-minute |gross|**. A new check, `the_stored_series_reproduces_every_aggregate`, requires the 5 bps re-price to reproduce every recorded Sharpe (deviation 0.0); the first attempt at this tranche failed that check after its holdout was decided, burning twenty slices (registry 68 → 88) that are not re-read. `tests/test_quantum_signal_ledger.py` (33), `tests/test_quantum_signal_tranche2.py` (6), `tests/test_quantum_signal_tranche3.py` (7), `tests/test_quantum_signal_tranche4.py` (5), `tests/test_quantum_signal_tranche5.py` (8). | `experimental` |
| [`apedp_dyn/quantum_compilation/candidates.py`](../apedp_dyn/quantum_compilation/candidates.py) | Quantum `CircuitCandidate` and `VerifierReceipt` records: deterministic ids bind parameters to the contract digest; artifacts are byte-hashed; proposer estimates are kept separate from official receipts. `CandidateStatus` separates `VERIFIER_REJECTED` (the verifier decided the circuit is wrong) from `UNVERIFIABLE` (it never decided), and only the two `VERIFIED_*` statuses may claim correctness. | `experimental` |
| [`apedp_dyn/quantum_compilation/ledger.py`](../apedp_dyn/quantum_compilation/ledger.py) | Adapter from quantum candidates to `FullLineageExperimentLedger` plus 3-benchmark portfolio scoring (`invsqrt_cost`, `dqi_cost`, `chemistry_cost`, `aggregate_score`, `score_verified_portfolio`, `score_ledger_portfolio`): proposer screens can only reject/propose, development-prefix passes never promote, and a salted full-domain slice is one-shot even after a pass. A receipt the verifier could not finish — `CAPABILITY_RECEIPT_STATUSES`, i.e. a timeout, crash or unparseable summary — is recorded `CandidateStatus.UNVERIFIABLE` rather than rejected, the exposure is logged as `ERROR`, and `verdict_summary()` reports `unverifiable` next to `eligible`, the denominator that excludes it. | `experimental` |
| [`apedp_dyn/quantum/gf2_rank_search.py`](../apedp_dyn/quantum/gf2_rank_search.py) | Exact rank search for the GF(2^12) multiplication tensor, replacing the `search_sat_rank` stub. Works on the **slice** formulation — the rank is the smallest set of rank-one matrices whose span contains every `B_k`, and `W` is then *determined* by GF(2) elimination (the step `search_flip_graph` skipped, which is why its variants failed the exact check). Two exact moves, complete over their family: deflation, and `k -> k-1` rewrites proved by enumerating all `2**k - 1` nonzero elements of the residual space (for `k = 2` three matrices, for `k = 3` seven, all of which must be rank-one). Measured: Karatsuba-63 deflates by **9 atoms to the base-composed 54** with no knowledge of how that decomposition was built, Schoolbook-144 does not deflate, and the 54 is closed under all **1,431 + 24,804** two- and three-atom rewrites. Nothing is called a rank unless all 1,728 coefficients reconstruct; a negative is a statement about the move family, never a minimality proof. | `experimental` |
| [`apedp_dyn/quantum_compilation/gf2_mul_proposer.py`](../apedp_dyn/quantum_compilation/gf2_mul_proposer.py) | The **proposer for the "fewer than 51 Toffolis" GF(2^12) multiply target**: turns a `field_tensor_search.TensorDecomposition` into a `CircuitCandidate` and records the screen in `QuantumCandidateLedger`. `Gf2MultiplicationProposer.screen()` reconstructs all 1,728 coefficients of the multiplication tensor from `U`/`V`/`W` over GF(2) (`verify_tensor_decomposition`) and prices the compiled multiplier on the field engine's own controlled model; `spec_sha256` pins field + tensor + target in place of a verifier contract. Measured on the shipped seeds: `Schoolbook-144`, `Karatsuba-63` and `Karatsuba-54-BaseComposed` reconstruct exactly (54 is the best that does), the `Published-51-Incumbent` ships all-zero factors and differs from the tensor in 363 coefficients, and every `search_flip_graph` variant fails the exact check — **so the target is met by nothing that reconstructs**. Screens can only propose or reject (`PROPOSED`/`SCREEN_REJECTED`); no candidate can claim correctness because a bare multiplier has no verifier contract. | `experimental` |
| [`apedp_dyn/quantum_compilation/referee.py`](../apedp_dyn/quantum_compilation/referee.py) | Official-verifier subprocess adapter for Benchmarks I (`invsqrt`), II (`dqi`), and III (`dfthcsos`): builds exact commands, pins contracts, parses verifier JSON, redacts the salt id, and labels `development-prefix`, `full-domain-reduced-shots`, `full-domain-public`, or `full-domain` scope. | `experimental` |
| [`apedp_dyn/quantum_compilation/invsqrt.py`](../apedp_dyn/quantum_compilation/invsqrt.py) | Benchmark-I proposer/screen: lazy `kickmix` runtime setup, exact integer Newton-Raphson error model, phase-margin and rotation-synthesis cost estimate, `.kmb` writer, single-coordinate precision neighbours, bounded phase-angle calibration (`calibrate_phase_angles`), and self-inverse Toffoli-pair rewriting (`rewrite_self_inverse_toffoli_pairs`). | `experimental` |
| [`apedp_dyn/quantum_compilation/search.py`](../apedp_dyn/quantum_compilation/search.py) | Budgeted APEDP-Q precision search: coordinate descent over fixed-point widths, full-domain model re-check before short-listing, official development-prefix verification, and an opt-in salted full-domain promotion run. | `experimental` |
| [`apedp_dyn/quantum_compilation/experiment.py`](../apedp_dyn/quantum_compilation/experiment.py) | Experiment 00 harness: `generate -> screen -> compile/measure -> verify` pipeline backed by `QuantumCandidateLedger` with separate estimated and measured cost fields, plus `DqiProposer` (Benchmark II), `DfthcsosProposer` (Benchmark III), and `run_reference_control`. | `experimental` |
| [`apedp_dyn/quantum_compilation/experiment_01.py`](../apedp_dyn/quantum_compilation/experiment_01.py) | Experiment 01 runner: reproduces the reference artifact and the APEDP-Q precision candidate through the ExperimentRunner, running both under the official full-domain verifier. | `audit-only` |
| [`apedp_dyn/quantum_compilation/dqi_builder.py`](../apedp_dyn/quantum_compilation/dqi_builder.py) | Benchmark-II DQI circuit builder used by the proposer and verifier-sweep scripts. | `experimental` |
| [`apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py`](../apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py) | Benchmark-II DQI Gluzdov V6 architecture, with its register-recycling lever **measured and closed**: the build that drops `n_a` peaks at $Q = 1,885$ but fails 64-shot equivalence (52/64) and costs $+648{,}784$ Toffolis (0.898× the projection), so `RECYCLING_APPLIED_IN_CIRCUIT` stays `False`. The operative projection is scored at the verifier peak $Q = 1{,}893$ with the unsourced $T_{\text{mean}} = 445{,}229$ ($1.9006 \times 10^{12}$, $12.859\times$); the 1,885 and 1,789 rows are carried as withdrawn and aspirational. That target is retained as the **score denominator only** (`TARGET_ROLE`), and `product_axis_decision()` records why: rung 4 (every field product and controlled add free) is $1.3076 \times 10^{12}$ = **0.688×** the denominator on the campaign arm's own 299,953.2 residual — the earlier 1.084× priced the six-lever arm's 472,768.2 at the campaign arm's width — while reaching 445,229 needs the surviving 386,061.7 of products at 37.6% of their cost, about 15.6 ANDs per multiply against the `2n − 1 = 23` lower bound. That bound is now **measured rather than bracketed**: a 63-Toffoli call is 63 ANDs and nothing else, so it reads 1.019× at 1,913 and 0.998× only at 1,893; a product that must materialise pays the measured 12-Toffoli `c_gf_iadd` and reads 1.164×. The retired 1.128× row charged 9 non-AND Toffolis per call that no emitted stream contains. | `experimental` |
| [`benchmarks/quantum_compilation/dqi_design_projection.json`](../benchmarks/quantum_compilation/dqi_design_projection.json) | Committed DQI design projection: operative row `Q` = 1,893 (the verifier peak) at cost 1.9006e12 / 12.857x, with the 1,885 recycled row carried as withdrawn and 1,789 as aspirational; `degree_register_floor` = 1,853 + 8k; `target_basis` marks 445,229 as unsourced | `audit-only` |
| [`scripts/search_gf2_mul_rank.py`](../scripts/search_gf2_mul_rank.py) | Runs the exact rank search from every shipped decomposition and writes `benchmarks/quantum_compilation/gf2_mul_rank_search.json`; `--check` re-validates the committed record without re-spending the budget (every recorded rank must reconstruct, Karatsuba-63 must still deflate by 9 to 54, the base 54 must still show the complete 26,235-move enumeration, and `target_met` must stay false). | `audit-only` |
| [`scripts/verify_gf2_multiplier_candidate.py`](../scripts/verify_gf2_multiplier_candidate.py) | Certificate for a GF(2^12) multiplier claim, reporting the four gates separately: algebraic reconstruction (all 1,728 coefficients, plus an independent `W` solved from `U` and `V`), a sampled bit-order cross-check against `gf_mul`, and the two gates that do **not** exist — `reversible_netlist` (`compile_reversible_multiplier` returns counts, not a gate stream) and `embedded_benchmark_receipt`. Status is `ALGEBRAIC_CERTIFICATE_ONLY` at best; the published 51 is refused as `REJECTED_BY_EXACT_SCREEN`. | `audit-only` |
| [`docs/GF2_MULTIPLY_SUBMISSION_PATH.md`](GF2_MULTIPLY_SUBMISSION_PATH.md) | Why an algebraic certificate is stronger than a shot-based receipt, the four submission gates with their present state, the measured rank-search table, and the arithmetic worth of a sub-51 multiplier (≈17.9% of the DQI 445,229 denominator if the product call stays the 63-AND form — stated as arithmetic on the projection, not a measurement) | `audit-only` |
| [`scripts/plan_dfthcsos_arm3.py`](../scripts/plan_dfthcsos_arm3.py) | Arm 3 (`arm3_combined_selectswap_and_reed_muller`) plan-only layout screening. Refuses to produce a layout until the shared baseline and both isolated Arm 1 / Arm 2 official receipts are `PASS` and each isolated arm strictly beats the baseline, then reports the multiplicative-gain **hypothesis** with `official_cost: null`, `status: PLAN_ONLY_NEEDS_COMBINED_RECEIPT` and `additivity_is_hypothesis: true`. | `audit-only` |
| [`scripts/audit_dfthcsos_arm3_receipts.py`](../scripts/audit_dfthcsos_arm3_receipts.py) | Arm 3 receipt audit, the interaction arm's gate: requires a manifest declaring `arm3_combined_selectswap_and_reed_muller` with `arm1_selectswap_enabled: true`, unchanged input/source hashes, and `PASS` isolated receipts from Arm 1 and Arm 2. Promotes only a combined score that beats the baseline **and** both isolated arms, and classifies the measured interaction as `SUPER_ADDITIVE`, `ADDITIVE_WITHIN_TOLERANCE` or `SUB_ADDITIVE` (5% band) instead of assuming the levers compose. Receipt arithmetic and manifest assertions only; it does not verify circuits or authenticate the verifier. | `audit-only` |
| `scripts/verify_division_granularity.py` | audit-only | Builds one GF(2^12) `init_gf2_div_with_scaffold` call and counts its CCX off the emitted circuit, settling that 378 Toffolis is one call while 46,656 is the decoder's whole per-shot division bill; writes `benchmarks/quantum_compilation/dqi_division_granularity.json` and re-checks it with `--check`. |
| [`scripts/write_dqi_design_projection.py`](../scripts/write_dqi_design_projection.py) | Writes `benchmarks/quantum_compilation/dqi_design_projection.json` from `design_projection_dqi_gluzdov_v6()` so the committed numbers cannot drift from the code | `audit-only` |
| [`docs/DQI_SINGLE_ARRAY_DECODER_CONTRACT.md`](../docs/DQI_SINGLE_ARRAY_DECODER_CONTRACT.md) | Correctness obligation and six falsifiers for a one-array DQI decoder (peak 1,001; Toffolis 1.015x Toffolis; score 0.284x of the reference). **Packing route since measured and closed**: the BM update reads `C_old` and `B_old` while `B' = C_old` writes `C_old` away, so both one-array schedules decode 6/216 (first failure at weight 1) against 216/216 for the three-region core — peak 1,001 is unreachable by repacking, and the surviving route is a nested EEA replay | `audit-only` |
| [`benchmarks/quantum_compilation/dqi_single_array_packing.json`](../benchmarks/quantum_compilation/dqi_single_array_packing.json) | The three-arm measurement behind that closure: 216/216 decoded for C+B+scratch, 210/216 failures for each one-array packing, plus the weight-36 boundary holding 20/20 | `audit-only` |
| [`benchmarks/quantum_compilation/dqi_single_array_replay_cost.json`](../benchmarks/quantum_compilation/dqi_single_array_replay_cost.json) | Price of the surviving single-array route: a measured 17,143 Toffolis per EEA round (8.8% of it divisions) turns `B_t` replay into 55.8x the EEA bill compute-only and 111.5x with Bennett, so the trade is 10.6x-21.0x **worse** than the reference rather than 0.284x, and the replay needs the array it was supposed to free | `audit-only` |
| [`docs/DQI_RECYCLED_PASS_BODIES.md`](../docs/DQI_RECYCLED_PASS_BODIES.md) | The three EEA pass bodies that touch `n_a`, the substitution that removes the register (build peaks at 1,885), the 64-shot bisection showing it fails equivalence at the first swap, and the score audit that makes it 0.898× the projection — the closure record for degree-register recycling | `audit-only` |
| `dqi_schur_forney.py` / `dfthcsos_riemannian_15batch.py` / `digit_serial_58q` (retired) | **Retracted 2026-11-03 by audit: these were stubs, and the figures once quoted for them are unsourced targets, not achievements.** All three are out of the trackable tree (untracked `build/apedp_stubs/`, with their test files); `dqi_builder.AVAILABLE_ARCHITECTURES` no longer advertises the `single_array_schur` architecture, `tests/test_dyn_dqi_builder_architectures.py` pins that, and the 58Q functions were removed from `src/signed_digit_invsqrt_v3.py`, which is now byte-identical to its committed state. What actually existed: a DQI stub that built at 878 qubits from 35 `cswap`/`cx` pairs with **no GF(2^12) arithmetic, so it decodes nothing** (its zero-`Z_POW` compliance is vacuous), and a chemistry module that never reads an integral — and whose FCIDUMP is not in this checkout. Constants previously presented as results, all unsourced: `Q <= 888`; `T <= 420,000`; `T <= 14,800` at `Q <= 420`; `lambda_eff <= 46.8`; `error_1 = 3.19 mHa` / `error_2 = 18.2 mHa`; and `C_1 = 89,505.6` at 58 qubits. The width bound is the stub's own register allocation and the Toffoli and `lambda_eff` bounds are literal caps (`toffolis_per_circuit = 1120`, `q_total = 110 + 60 + 180 + 68`), the chemistry errors are dict literals in `optimize_gauge`, and the invsqrt figure is `58 x 1543.2` over an asserted 1,235 Toffolis while the circuit builds at 47 qubits with 200 toffoli-like ops. The "47-qubit candidate verified against the certificate" claim is retracted with them: no `.kmb` was compiled and no verifier ran; the 3,133,452-input certificate that does exist belongs to the champion integer model, which was never touched. | `deprecated` |
| [`apedp_dyn/quantum_compilation/dfthcsos_adapter.py`](../apedp_dyn/quantum_compilation/dfthcsos_adapter.py) | Benchmark-III DFT-HC-SOS adapter bridging the proposer to the official verifier scoring infrastructure (supports J=4 batch-XOR and J=3 signed Givens extraction). | `experimental` |
| [`apedp_dyn/quantum_compilation/dfthcsos_signed_4bank.py`](../apedp_dyn/quantum_compilation/dfthcsos_signed_4bank.py) | Benchmark-III signed 4-address banked QROM transformer and auditor: proves signed Givens tree invariant ($\cos\theta > 0 \implies \text{bit}_9 == \text{bit}_8$), compresses 3 deltas to 81 bits, satisfies 128-qubit cap with 0 headroom, and audits external V4 chemistry arithmetic discrepancy (HOLD_FOR_ARITHMETIC_AND_RECEIPT_RECONCILIATION). | `experimental` |
| [`apedp_dyn/quantum_compilation/structure_audit.py`](../apedp_dyn/quantum_compilation/structure_audit.py) | MOSAIC-2 Phase P2 structure and sparsity audit engine: computes empirical Shannon entropy, Fast Walsh-Hadamard transform spectral rank, repeat fractions, and verifies the 10-bit Givens sign invariant ($\cos\theta > 0 \iff \text{bit}_9 == \text{bit}_8$). | `experimental` |
| [`apedp_dyn/quantum_compilation/two_region_probe.py`](../apedp_dyn/quantum_compilation/two_region_probe.py) | MOSAIC-2 Phase P3 two-region selector probe on Benchmark I (invsqrt): evaluates discrete integer boundary margins over $x \in [1, 3133452]$ without continuous $C^0$ analytical overconstraints, tracks clean ($|0\rangle$) and dirty ($U^\dagger U = I$) ancilla lifecycles, and executes conservative early-exit rejection when selector overhead $O_{\min}$ dominates polynomial savings $S_{\max}$. | `experimental` |

| [`apedp_dyn/quantum_compilation/invsqrt_qrom_line.py`](../apedp_dyn/quantum_compilation/invsqrt_qrom_line.py) | Experiment 02 proposer: QROM piecewise-linear seed search over fixed-point widths; exact 130-bit integer fixed-point screen over the full domain, self-contained `kickmix` circuit compilation without volatile global caches, `statistics.num_t_like` resource accounting, and official verification. Exact-model screen estimate for `(7,16,12,10,28,27,28)` is 1,309,175.61 (unverified on full-domain holdout; the only on-disk verified full-domain PASS in `artifacts/quantum_compilation/invsqrt/full-verification/ledger.jsonl` is NR candidate `1b64c7145d13fc09d368` at 1,941,280.56). | `experimental` |
| [`apedp_dyn/quantum_compilation/invsqrt_direct_poly.py`](../apedp_dyn/quantum_compilation/invsqrt_direct_poly.py) | Experiment 03 proposer: direct degree-2 quadratic polynomial phase compiler over power-of-two intervals (`PolyPrecision(index_bits, coef_bits, delta_bits)`) with Horner evaluation, synchronized model/circuit bit-slicing, and `statistics.num_t_like` resource accounting. | `experimental` |
| [`apedp_dyn/quantum_compilation/invsqrt_signed_digit.py`](../apedp_dyn/quantum_compilation/invsqrt_signed_digit.py) | Benchmark-I 92Q centered signed-digit QROM phase compiler combining 256-row banked coefficient QROM, Rudra derivative split (7-bit coarse + 12-bit residual), Alex prefix flags, Artem borrowed-carry on normalized $x[21]$, and mutual-exclusion input domain rewrite. Compiles canonical SHA-256 `63f70e62...` achieving cost $\approx 164,646$ ($12.61\times$ improvement over reference). | `experimental` |
| [`apedp_dyn/quantum_compilation/qrom_invsqrt.py`](../apedp_dyn/quantum_compilation/qrom_invsqrt.py) | Compatibility wrapper delegating QROM piecewise-linear `invsqrt` candidate generation, screening, compilation, and resource measurement to `invsqrt_qrom_line.py`. | `experimental` |
| [`apedp_dyn/quantum_compilation/multi_structure_search.py`](../apedp_dyn/quantum_compilation/multi_structure_search.py) | Multi-structure search orchestrator running NR, QROM, Direct Poly, and FAROS-ISQ proposers across circuit topologies with `QuantumCandidateLedger` tracking. | `experimental` |
| [`apedp_dyn/quantum_compilation/faros_isq.py`](../apedp_dyn/quantum_compilation/faros_isq.py) | FAROS-ISQ (Function-Aware Representation Optimization and Synthesis) framework expanding SABLE-90: representation portfolio (R1 nonuniform minimax polynomial, R2 prefix table + residual polynomial, R3 direct phase decomposition, R4 alternative arithmetic [Horner/Estrin/Bit-serial], R5 hybrid), Chebyshev minimax coefficient table generation and loading, exhaustive integer certificate across 3,133,452 inputs, circuit materialization (`invsqrt_faros_r2_86q.kmb`), cutting-plane LP rotation calibration, official verifier execution, SABLE-90 baseline profiler (nominal 157,837.85 / affine-opt 157,749.40), worst-case error-budget engine over 3,133,452 inputs, and 5-fidelity Pareto tournament. | `experimental` |
| [`scripts/evaluate_faros_r3_synthesis.py`](../scripts/evaluate_faros_r3_synthesis.py) | Forensic evaluation and physical synthesis analysis of FAROS-R3 direct phase decomposition: proves 2^22 full-rank Walsh-Hadamard spectrum, gate synthesis lower bounds (>4.19M rotations, >2.9e7 Toffolis), error tolerance epsilon collapse, and records official verifier failure on x=1. | `audit-only` |
| [`docs/invsqrt_frontier_and_portfolio_synthesis.md`](invsqrt_frontier_and_portfolio_synthesis.md) | Comprehensive forensic audit and portfolio synthesis report: reconciles FAROS-R2 cost arithmetic (159,191.95), FAROS-R1 lower bound (199,177.92 executed / 200,188.80 static), scopes R3 rejection to unstructured Walsh proxy, investigates DQI Gluzdov V6 vs iteration sweep mechanics, and audits DFTHCSOS 10-bit sign invariant and 4,363 Toffoli discrepancy. §1.1 reconciles the champion's **three cost bases** for the one circuit `ce04c9d5…` (157,605.52 at 25,600 shots / 157,601.0808 sealed at 200,540,928 shots / 157,601.115479 in the kernel receipt), and §Executive Summary states the **Benchmark II factor once**: 65.106932× attested on the archived binary (`748b01a7…`, `PASS` at 102,400 shots under the default seed and salt `reseal-2026-11-05`), with 1.053682× beside it as the pinned build's ceiling ($S$ 2.361982 → 20.9865; 2.403512 survives only as the arm-packaging scenario). | `production` |
| [`docs/ATTESTED_WIDTH_AND_ACCURACY_BUDGET.md`](ATTESTED_WIDTH_AND_ACCURACY_BUDGET.md) | Derived budgets for the two attested artifacts, from their measured parameters alone: Benchmark II's 891 qubits decompose as one 876-qubit register (`73 × 12`) plus 15 transient, whereas two registers are 1,752 before any scratch, so no optimisation of the 1,893-qubit reference can reach that width — it is an in-place algorithm, and no builder here implements it. Benchmark III's `error_2` is 2.2e-9 Ha below the 20 mHa gate against a 3.068e-3 rad smallest re-synthesis perturbation, so depth is closed and the only accuracy-neutral lever is the 303 temporary state-prep qubits (71% of width, 0.234% per qubit). Records both exchange rates (II: 1 qubit = 891 Toffolis; III: 1 qubit = 15,459 Toffolis) and what would unblock an emitter. Follow-up measurements: Benchmark II's 891 is a division-free packed decoder (`876 array + 12 scale + 3 control`), measured at 4,473 Toffolis per GF(2¹²) iteration; Benchmark III's 8,718 angle Toffolis sit 3.42× below the cheapest cap-feasible QROM shape (arity is neutral, QROAM's optimum needs 2,594 qubits) and 6.84× below every per-address encoding of its 40-bit payload — 3,976 runs, 3,976 distinct rows, no zero deltas, 31 of 40 bits non-affine, no uniform ramps, so run-length, dedup, delta and accumulator loadings all cost 59,610. Receipts `dqi_width_budget.json`, `inplace_bm_verification.json`, `bm_iteration_toffolis.json`, `dfth_qrom_shapes.json`, `chien_forney_stage.json`, `dfthcsos_load_encoding.json`, `uncompute_pass.json`, `chien_variants.json`. **Correction (2026-11-05, measured):** the 891 is not a submission width — the contract forces 840 syndrome qubits inside `Q` and the packed array is a separate 876-qubit buffer, so the construction is 1,731 and the built stream measures 1,788 (`packed_decoder_official_instance.json`). | `audit-only` |
| [`apedp_dyn/quantum_compilation/inplace_bm.py`](../apedp_dyn/quantum_compilation/inplace_bm.py) | Packed single-array Berlekamp–Massey over GF(2^b) for the DQI decoder: σ in `0..t`, B in `t+1..2t+1` and the discrepancy in cell `2t+2` of one `2t+3`-cell array, with the σ→B snapshot and the `σ ← σ + (d/b)·xˢ·B` update sharing the array because the pass runs descending. Verified against the two-array reference on planted error patterns at m=15 (GF(2⁴)), m=255 (GF(2⁸)) and m=4095 (GF(2¹²)) — every trial agrees root for root and recovers the planted positions. Width is settled exactly: the division variant needs `876 + 12 + 11 = 899`, the division-free one needs only its scale outside the array, `876 + 12 = 888`, and with 3 control bits that is **891 — the archived artifact's width**, the 8-qubit gap being exactly `scaffold 11 − control 3`. Toffolis stay attested: the measured key-equation stage is 4,473 per GF(2¹²) iteration (63.0 per GF multiply) and 313,110 for 70 iterations, and the gated Forney block is 315 per position (the coefficient steps are constant multiplies and measure 0), so a full 4,095-position sweep costs 1,289,925 and the adjoint uncompute costs the same 313,110 as the forward loop (measured both directions, equal at every field) while a measured clear costs **0 T** (876 `R` gates, no AND; the phase-safe `mr` form is not in the shipped extension but `scripts/kickmix_mr_opcode.patch` adds it and it is equally AND-free), so the static total is **1,916,145 — 4.83× the archive**. Read as `T_mean` the picture inverts: the sweep's coefficient steps are constant multiplies (0 T), the Forney block only runs at ≤ t = 35 live positions (11,025, fitting the 83,809 remainder), and a Z-basis clear replaces the adjoint (0 T), giving **324,135 — the archive is 1.22× above it**. Sharing variants are excluded by measurement: unconditional blocks 1,289,925 (15.4× the remainder), Montgomery batch inversion 773,811 (9.2×). Receipts `inplace_bm_verification.json`, `bm_iteration_toffolis.json`, `chien_forney_stage.json`, `uncompute_pass.json`, `chien_variants.json`, `packed_decoder_assembly.json`; emitters `scripts/emit_bm_iteration.py`, `scripts/emit_chien_forney_stage.py`, `scripts/emit_uncompute_pass.py`, `scripts/emit_chien_variants.py`, and `scripts/assemble_packed_decoder.py`, which puts all four stages in one stream and reads `T_mean` from the simulator's execution counts; tests `tests/test_inplace_berlekamp_massey.py`, `tests/test_decoder_stage_accounting.py`. | `audit-only` |
| [`docs/ARTIFACT_ATTESTATION_RULE.md`](ARTIFACT_ATTESTATION_RULE.md) | Governance rule A1–A8 deciding when a hash-pinned artifact with no in-tree emitter may seal a portfolio factor: admission requires size + SHA-256 pins in the spec and manifest, an official-verifier `PASS` at the official shot budget, pinned verifier inputs, a fresh-salt repeat, a re-derivable receipt, and the row stating that no builder here emits it; the factor is labelled **attested** (never rebuilt/reproduced/emitted), any rebuildable factor is quoted beside it as the buildable ceiling, attestations support no drift claim, are superseded rather than deleted, and a reseal must record both hashes, both salts and the resulting `S`. Also records the four applied rows and why provenance remains outside what the rule can state. | `production` |
| [`benchmarks/quantum_compilation/invsqrt_62q_full_domain_receipt.json`](../benchmarks/quantum_compilation/invsqrt_62q_full_domain_receipt.json) | Full-domain Benchmark I certificate for the 62-qubit circuit staged in `submission_24_59.zip` (`sha256 af469847…`): official `PASS` over 3,133,452 inputs × 64 shots = 200,540,928 shots at **97,921.5795** — 1,281.6647 executed / 1,537 static Toffolis, 41 rotations, ε 2.434041e-13, worst error/tolerance 0.99932377 at x = 161, 21.2085× the 2,076,774.23 reference cost, $1.6095\times$ (37.87%) below the pinned 89-Q. Records the exact command, protocol and run telemetry (445.87 s, 8 cores, 561% CPU) so the run is re-derivable from the circuit hash; the source archive and the working-tree `.kmb` copy are gitignored, and the builder that produced this width is not in this checkout, so the row certifies the artifact rather than its derivation. | `audit-only` |
| [`docs/INVSQRT_NEW_DECOMPOSITION_SPEC.md`](INVSQRT_NEW_DECOMPOSITION_SPEC.md) | **Why a clean-room Benchmark I circuit is not available, measured rather than argued.** States the constraint no earlier note records: the verifier's tolerance is *relative* (`2^-24 · β/√x`), so ~38 bits of absolute phase precision are required and truncating the champion's 38-bit phase to 24 bits fails 3,129,842 of 3,133,452 points by up to 1,820× tolerance — which floors any held-accumulator design near 86 qubits, where the 89-Q champion already sits. Reconstructs the 62-Q frontier by measurement: the in-repo Newton-Raphson reference is 260 Q / 7,735 T with every register simultaneously live, and its hand-recycled dependency graph still floors at Q ≈ 130–170, so the public circuit's 40 working qubits cannot come from register recycling. Prices every clean-room family from measured primitives (QROM cost `2^(k+1)−2`, independent of output width; quadratic sweep best Q = 135 / T = 3,464; digit-emission screen residual/tol 1.3e5 → 2.0e3 → 30 after 6/12/18 mantissa bits, so tolerance needs all 22) and records that the pinned binary's missing `mr` is what blocks carry segmentation and classical branching. | `audit-only` |
| [`docs/INVSQRT_SUB62_DESIGN.md`](INVSQRT_SUB62_DESIGN.md) | **What a sub-62-qubit circuit needs, measured rather than argued.** 62 itself is already reached: the build staged in `submission_24_59.zip` (`sha256 af469847…`) is `PASS` over the full domain at **97,921.58** (200,540,928 shots, `benchmarks/quantum_compilation/invsqrt_62q_full_domain_receipt.json`), $1.6095\times$ below the pinned 89-Q 157,601.08, so the note retires the old "89 is the floor" premise and states the exchange rate **below** 62 (at Q = 62 the bar is `te ≤ 1,579.38`, where the staged build sits; Q = 56 allows 1,748.6 and Q = 48 allows 2,040.0). Decomposes the peak (71 named registers — 22 `x`, 21 accumulator, 12 residual, 7 coarse, 6 curvature, 5 flags — plus 18 `folded_gadd` carries), and prices each of the three levers out: segmenting the carry chain (Q 76 at +56 Toffolis ⇒ 138,837) **aborts in the exact-domain compiler** (`Scratch function is not zero`, `portable_exact_domain.py:285`), because a segment's carry-out is consumed and preserved by the next segment and `del_and` cannot clear a majority-of-three — only the `mr` this build lacks would, and `hmr` is not phase-safe; record-restoring `residual`/`c2`/`coarse` reaches Q ≈ 60 at +397 Toffolis ⇒ 130,260; the digit-serial schedule measures 34,997 Q / 101,914 T at 22 bits, and a shared multi-output plan is 1.4–1.5× **wider** at every k. Adds the correction-row result (every single-row drop is 1.14×–48× of the verifier's tolerance for 132 of 1,466 Toffolis) and the attribution showing the 14 outer-row 21-bit accumulation, not the correction, is the largest block at 560. Ceiling and lever verdicts pinned by `tests/test_dyn_invsqrt_width_ceiling.py`. | `audit-only` |
| [`docs/UPSTREAM_MR_SUBMISSION.md`](UPSTREAM_MR_SUBMISSION.md) | **The `MR` change request, ready to file.** PR body, the four closed routes around a missing Z-basis read, the component-by-component change, and the evidence measured under the competition's own verifier at 102,400 shots: reference 2.443946e13 → six-lever arm 5.324728e12 (0/0 errors, `Q` 1,893 → 1,913), a **4.5898×** score ratio decomposed as 4.6873× Toffoli ÷ 1.0212 `Q²` penalty. Records patch mechanics (19 files, +940/−9, applies cleanly at offset −17), the four acceptance tests, the silent phase-failure mode, and the two-PR split (opcode 441 lines, `gf_product` 499). States what it does not buy: the arm is still 3.14× above the 1.6975e12 target. | `audit-only` |
| [`docs/DFTHCSOS_SUB128_SCOPE.md`](DFTHCSOS_SUB128_SCOPE.md) | **What a sub-128-qubit Benchmark III angle circuit actually needs.** Corrects the "format is the blocker" reading: `example_submission/angle_k_reference_circuit.py` writes `.kmb` with stock `kickmix`, so the blockers are the 128-qubit cap (design) and the 128,128,348-byte FCIDUMP the verifier pins and this checkout does not ship (data). Tabulates seven geometries from 73 to 132 qubits with what each gives up, records that the two plan records disagree on the same design (128 vs 132) and that measurement settles it, and orders the work: mount the data, compile the 25 circuits, then pick a lever.  — dimensions and the `2 x sum` two-pass identity pinned by `tests/test_dyn_dfthcsos_sub128_family.py` (123 qubits, 8 circuits, 3,974 each, 63,584, plus a digest over the size summary) | `audit-only` |
| [`docs/DFTHCSOS_HANDOVER.md`](DFTHCSOS_HANDOVER.md) | **Handover for Benchmark III:** the one missing input (`FeMoCo76o113e.fcidump`, 128,128,348 bytes, sha256 `e5a48af7…`, and why nothing shipped can substitute for it — `load_inputs` replaces the submitted shifts with the baseline's and always calls `check_fcidump`), the measured table (123 qubits, 8 circuits, 3,974 Toffolis each, 63,584 over both passes, 298,200 words recovered exactly, angle errors 3.068e-3 / 6.136e-3 rad), the single `verify.py` command that finishes it, the six rejected ideas with their evidence, and the scored verdict across every precision the cap admits (12-bit 3.19/18.64 mHa PASS at 3.356e9; 11-bit 221.3/51.7 mHa; 10-bit 242.3/59.4 mHa) — so the 20 mHa accuracy gate, not the 128-qubit cap, sets the precision floor | `audit-only` |
| [`scripts/verify_invsqrt_62q_domain.py`](../scripts/verify_invsqrt_62q_domain.py) | Batched full-domain and range verification harness for Benchmark I (invsqrt): supports parallel chunked basis-state simulation via `km.Simulator` (Python 3.13), atomic JSON checkpointing with `--resume`, and exact referee score computation over $x \in [1, 3133452]$. | `audit-only` |
| [`docs/INVSQRT_62Q_VS_89Q_LIVENESS_REPORT.md`](INVSQRT_62Q_VS_89Q_LIVENESS_REPORT.md) | Comparative register liveness report detailing the 27-qubit reduction breakdown ($89 \rightarrow 62$ qubits, −37.87% score) between the 89Q baseline and 62Q frontier via digit-serial accumulator reuse, in-place carry recycling, and 16.6% dynamic branch skipping. | `audit-only` |
| [`scripts/verify_assembly_local_mr_sim.py`](../scripts/verify_assembly_local_mr_sim.py) | Runs the assembled decode stages (packed discrepancy pass, sweep, locator read, Forney magnitude, measured clear) through the local `mr`-capable value simulator. At GF(2^4)/t=5 it asserts the syndrome registers read back unchanged after a compute+uncompute pass, the sweep plus all-zero `mr` read recovers exactly the planted error positions (8/8 patterns, weights 1–5), and the **magnitudes** are recovered too: ω = (S·σ) mod x^{2t} is built from the syndrome registers and the locator, evaluated by Horner at α^{-j}, divided by the odd-coefficient derivative sum `Σ_{k odd} σ_k α^{-kj}` (which is `Λ'(α^{-j}) / α^{j}`, so Forney's `α^{j}` cancels and no constant is needed), and read only where the live bit fired — every planted value comes back exactly. The classical Forney result is asserted against the planted vector first, so a wrong convention cannot be absorbed by a matching circuit. The array clears to zero and the `r`-instead-of-`mr` falsifier randomises the phase. Receipt: `benchmarks/quantum_compilation/assembly_local_mr_verification.json`. | `audit-only` |
| [`scripts/build_packed_decoder_submission.py`](../scripts/build_packed_decoder_submission.py) | Builds the packed decoder's four stages at the official instance (degree 12, t 35, 4,095 positions) shaped to the DQI register contract — `S0..S69` quantum, `E0..E4094` classical, no other named register — and checks that contract with the verifier's own `validate_registers`. Measures 1,788 qubits, 44,725,333 ops and 1,603,035 static Toffolis (4.1 s to build). The width is the finding: the contract forces 840 syndrome qubits inside `Q` and the packed array is a separate 876-qubit buffer, so `876 + 12 + 3 = 891` omits the syndrome block and this construction is 1,731 at best. | `audit-only` |
| [`benchmarks/quantum_compilation/packed_decoder_official_instance.json`](../benchmarks/quantum_compilation/packed_decoder_official_instance.json) | Benchmark II submission attempt for the packed decoder at the official instance: built, register-contract **ACCEPTED**, official verifier **FAIL** at 1,024 shots — 0 value errors, 510 phase errors (empirical 0.4980, ε₊ 0.6944 against the 1/16 gate), 25.5 s, 1.8 GB peak. The value pass is vacuous (the stream never writes `S` or `E`, and its junk is cleared); phase is the whole defect, because the packed array has no cell for the Forney numerator Ω and the kickback multiplies by zero — adding Ω needs ≥ 2,148 qubits. Also records the unshaped assembly's rejection (`ValueError: Registers must be exactly S0..S69 and E0..E4094`, 5,883 qubits, `assemble_packed_decoder.py`) and the 1,788-vs-891 width. | `audit-only` |

| [`scripts/prepare_track4_artifacts.py`](../scripts/prepare_track4_artifacts.py) | Track 4 candidate artifact staging and SHA-256 verification runner before official verifier execution. | `audit-only` |
| [`config/receipt_plan_track4.json`](../config/receipt_plan_track4.json) | Authoritative receipt execution plan for Track 4 candidate reproduction and portfolio sealing. | `audit-only` |
| [`tests/test_faros_isq.py`](../tests/test_faros_isq.py) | Unit and integration test suite for FAROS-ISQ: SABLE-90 profiler verification, error-budget margin checks, R1-R5 representation evaluations, Chebyshev minimax table generation/loading, exhaustive integer certificate verification across 3,133,452 inputs, circuit materialization, cutting-plane LP rotation calibration, verifier receipt validation, multi-objective Pareto dominance detection, multi-fidelity tournament execution, and search orchestrator. | `audit-only` |
| [`tests/test_quantum_compilation.py`](../tests/test_quantum_compilation.py) | Audit pins for APEDP-Q: contract/tamper checks, candidate identity, ledger screens, development-versus-final scope, one-shot salt burning, subset coverage, neighbour determinism, QROM/Direct-Poly model & resource accounting, multi-benchmark referee dispatch (`invsqrt`, `dqi`, `dfthcsos`), ledger-backed `ExperimentRunner`, phase calibration/rewrites, 3-benchmark portfolio scoring, and FAROS-ISQ/SABLE-90 integration pins. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt/verify_submission_invsqrt.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt/verify_submission_invsqrt.py) | Benchmark-I authoritative verifier: exact phase reference, full-domain shot sweep, Toffoli-equivalent score, and salted/seed handling. Pinned by `contracts.py`; never edited by APEDP-Q. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt/nr_reference_circuit.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt/nr_reference_circuit.py) | Benchmark-I reference Newton-Raphson fixed-point model and circuit builder used by the proposer screen; authoritative interpretation remains the verifier. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dqi/verify_submission_dqi.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dqi/verify_submission_dqi.py) | Benchmark-II (quantum-data-integration) authoritative verifier shipped with the competition bundle. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dqi/dqi_reference_circuit.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dqi/dqi_reference_circuit.py) | Benchmark-II Reed-Solomon reference circuit implementation shipped with the competition bundle. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/verify.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/verify.py) | Benchmark-III (DFT-HC-SOS) authoritative verifier entrypoint shipped with the competition bundle. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/scoreinfra/circuits.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/scoreinfra/circuits.py) | Benchmark-III circuit/score infrastructure shipped with the competition bundle. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/scoreinfra/costs_prepare.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/scoreinfra/costs_prepare.py) | Benchmark-III cost/resource preparation module shipped with the competition bundle. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/scoreinfra/dfthc_math.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/scoreinfra/dfthc_math.py) | Benchmark-III double-factorized THC mathematics shipped with the competition bundle. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/scoreinfra/reference.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/scoreinfra/reference.py) | Benchmark-III reference state/energy construction shipped with the competition bundle. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/dqi_iteration_sweep_patch/apedp_dyn/quantum_compilation/dqi_builder.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/dqi_iteration_sweep_patch/apedp_dyn/quantum_compilation/dqi_builder.py) | Patched DQI builder snapshot from the iteration-sweep bundle; the live builder is `apedp_dyn/quantum_compilation/dqi_builder.py`. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/dqi_iteration_sweep_patch/scripts/audit_dqi_sweep.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/dqi_iteration_sweep_patch/scripts/audit_dqi_sweep.py) | DQI iteration-sweep audit script. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/dqi_iteration_sweep_patch/scripts/sweep_dqi_eea_bound.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/dqi_iteration_sweep_patch/scripts/sweep_dqi_eea_bound.py) | DQI EEA-bound sweep script. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/dfthcsos_rm_block2_j4.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/dfthcsos_rm_block2_j4.py) | Staged DFT-HC-SOS Reed-Muller block-2 experiment script. | `experimental` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/run_official_receipts.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/run_official_receipts.py) | Fail-closed candidate builder and official verifier execution orchestrator. | `audit-only` |
| [`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/example_submission/angle_k_reference_circuit.py`](../apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dfthcsos/example_submission/angle_k_reference_circuit.py) | Benchmark-III example/reference angle-K circuit shipped with the competition bundle. | `audit-only` |
| [`apedp_dyn/epistemic/market_extrapolation.py`](../apedp_dyn/epistemic/market_extrapolation.py) | Out-of-distribution regime transfer gate evaluating volatility and liquidity shocks. | `production` |
| [`apedp_dyn/epistemic/frozen_spec.py`](../apedp_dyn/epistemic/frozen_spec.py) | Cryptographic hash verification of referee source files to prevent silent code drift. | `production` |
| [`apedp_dyn/epistemic/dimensional_homogeneity.py`](../apedp_dyn/epistemic/dimensional_homogeneity.py) | Typed SI physical dimensional homogeneity engine and epistemic gate (Option 3): validates term addition compatibility, Buckingham-Pi constraints, and dimensionless transcendental arguments. | `experimental` |
| [`apedp_dyn/epistemic/eedr.py`](../apedp_dyn/epistemic/eedr.py) | Epistemic Evidence-to-Decisiveness Ratio allocating exploration budget. | `production` |
| [`apedp_dyn/epistemic/active_disambiguation/policy.py`](../apedp_dyn/epistemic/active_disambiguation/policy.py) | Active epistemic disambiguation selector (E4, forensics doc S71): measurement design, discrimination table, and policy comparison for resolving hypothesis ties. | `experimental` |
| [`apedp_dyn/epistemic/active_disambiguation/divergence_selection.py`](../apedp_dyn/epistemic/active_disambiguation/divergence_selection.py) | Cost-aware divergence-point selection (E4 generalization): proposes the query point itself on a continuous domain, ranked by expected cost to resolution `n_req(x)·cost(x)` under an explicit budget; named abstention (`INSUFFICIENT_DIVERGENCE` / `INSUFFICIENT_BUDGET`); preserved P2/P3 no-truth-leakage, P6 re-adjudication, and P8 determinism (frozen grid + deterministic refinement; selection carries no randomness). | `experimental` |
| [`apedp_dyn/epistemic/verified_knowledge_module.py`](../apedp_dyn/epistemic/verified_knowledge_module.py) | Verified Knowledge Module Library (Gap-3, read-only additive layer over ledgers): entry gates = experiment-ledger `CONFIRMED_VALID` + promotion Stage 4 (OOS transfer); metadata contract = preconditions, finite validity domains, uncertainty with estimation method (no naked floats), REQUIRED counterexamples, dependency closure; composition intersects validity domains and refuses disjoint/out-of-chain reuse; every export carries the stage-5/6 non-inclusion disclosure. | `experimental` |
| [`apedp_dyn/epistemic/caller_data_protocol.py`](../apedp_dyn/epistemic/caller_data_protocol.py) | Caller-Layer Three-Role Data Protocol (REG-F14/F14b reference, design doc `docs/CALLER_DATA_THREE_ROLE_PROTOCOL.md`): disjoint SELECTION/CONFIRMATION/PROSPECTIVE roles at issue time; SEALED one-shot confirmation consumption (second consume raises, one-shot provenance) with a DURABLE lead state store (`_LeadStateStore`: atomic JSON, claimed-spent-before-evaluation, restart-proof re-issue/re-consume refusal, corrupt-store fail-closed `CallerProtocolStateError` — path explicit or derived from the ledger's persistence directory); confirmation exposures integrate the existing anti-snooping burn ledger FAIL-CLOSED (unregistered candidates refused before the sealed slice is spent; ledger write failures propagate, never swallowed); archived verdict source must be `confirmation` with selection superseded; REG-F14b either-or compliance — sealed split OR ledger-derived DSR-adjusted multiplicity (computed, monotone, never hand-written; reuse events inflate; the `dsr_shrink` disclosure imports the preregistered `DSR_THRESHOLD` from the hash-verified canonical spec, no local threshold literal). Per-caller wiring landed over the census (`caller_registry.py`, mode (b)); mode (a) is now operationally wired end-to-end (`issue_sealed_confirmation_lead` / `publish_sealed_compliance_record`; `blind_benchmark` first). | `experimental` |
| [`apedp_dyn/epistemic/caller_registry.py`](../apedp_dyn/epistemic/caller_registry.py) | Production-caller compliance registry (REG-F14b wiring): auditable census of six production data-consuming verdict publishers (`dyn_sweep`, `multi_asset_signals`, `real_market_l2_sweep`, `blind_benchmark`, `calibrated_benchmark_suite`, `canonical_ledger`) each wired with a module-level `publish_caller_compliance_record` (mode b: ledger-derived DSR-adjusted multiplicity, validated at publish time, refuses hand-written values); `record_selection_exposure` records selection-stage `TRAINING_DEVELOPMENT` exposures making the derived multiplicity live; closed-census refusals for unknown/duplicate/missing callers. Pins `tests/test_dyn_caller_compliance_wiring.py` (12; includes four mode-(a) operational pins). Mode (a) is NOW operationally wired: `issue_sealed_confirmation_lead` / `publish_sealed_compliance_record` (census- and dataset-family-checked; publication refused while the sealed confirmation slice is unconsumed or carries foreign provenance), with `blind_benchmark` as the first census caller wired at its own module level (`open_sealed_confirmation_lead` / `evaluate_confirmation_cases` / `publish_sealed_compliance_record` — its blind case ids are its real sealed confirmation family; frozen-referee-checked, one-shot consumption, spent slices un-spent-able).
 Honest boundary: the five other census callers remain published in mode (b) by default, and the pinned mode-(a) demonstration is the wiring path over suite-shaped blind cases — a live production promotion through the sealed slice is pending and disclosed, not silently claimed. | `experimental` |
| [`apedp_dyn/epistemic/active_disambiguation/drift_e5.py`](../apedp_dyn/epistemic/active_disambiguation/drift_e5.py) | E5 ambiguity stability / identifiability under drift forensics (S73). | `audit-only` |
| [`apedp_dyn/epistemic/active_disambiguation/e5_market.py`](../apedp_dyn/epistemic/active_disambiguation/e5_market.py) | E5-market diagnostic (S74): E5 trajectory signatures on §72 windows. | `audit-only` |
| [`apedp_dyn/epistemic/active_disambiguation/ensemble_gated.py`](../apedp_dyn/epistemic/active_disambiguation/ensemble_gated.py) | §79 conditional ensemble forensics: whether incremental utility can be gated. | `audit-only` |
| [`apedp_dyn/epistemic/active_disambiguation/ensemble_m.py`](../apedp_dyn/epistemic/active_disambiguation/ensemble_m.py) | M0–M3 ensemble forecasting test (S78): incremental predictive value of ambiguity families. | `audit-only` |
| [`apedp_dyn/epistemic/active_disambiguation/finance_adjudication.py`](../apedp_dyn/epistemic/active_disambiguation/finance_adjudication.py) | §83 autonomous finance-grammar adjudication forensics. | `audit-only` |
| [`apedp_dyn/epistemic/active_disambiguation/finance_manifest.py`](../apedp_dyn/epistemic/active_disambiguation/finance_manifest.py) | Frozen finance-grammar manifest v1 for §83. | `audit-only` |
| [`apedp_dyn/epistemic/active_disambiguation/incremental_info.py`](../apedp_dyn/epistemic/active_disambiguation/incremental_info.py) | Three-Hypothesis Incremental Information Test (S80). | `audit-only` |
| [`apedp_dyn/epistemic/active_disambiguation/trading_e4t1.py`](../apedp_dyn/epistemic/active_disambiguation/trading_e4t1.py) | E4-T1 trading transfer (S72): active disambiguation on volume-family ambiguity. | `audit-only` |
| [`apedp_dyn/epistemic/active_disambiguation/trading_e4t1_v2.py`](../apedp_dyn/epistemic/active_disambiguation/trading_e4t1_v2.py) | E4-T1 trading transfer v2 (S72): active disambiguation with block basis. | `audit-only` |
| [`apedp_dyn/grammar/proposer_interface.py`](../apedp_dyn/grammar/proposer_interface.py) | Unified abstract base class and dataclass contracts for competing hypothesis proposers. | `production` |
| [`apedp_dyn/grammar/trading_grammar_catalog.py`](../apedp_dyn/grammar/trading_grammar_catalog.py) | Grammar definitions and AST transforms for causal microstructure and order-flow signals. | `production` |
| [`apedp_dyn/grammar/scientific_grammar_catalog.py`](../apedp_dyn/grammar/scientific_grammar_catalog.py) | Grammar definitions for physical, biological, fluid, and geometric dynamical systems. | `production` |
| [`apedp_dyn/grammar/candidate_ir.py`](../apedp_dyn/grammar/candidate_ir.py) | Intermediate representation for symbolic expressions and parse tree canonicalization; term families now include saturating-rational restoring/dissipative/forcing forms (Langmuir / Michaelis–Menten / Hill) so the ODE and mechanical grammars can express bounded receptor and enzyme kinetics. | `production` |
| [`apedp_dyn/grammar/floor_rules.py`](../apedp_dyn/grammar/floor_rules.py) | Neutral floor-rule fitting and rule-form proposal layer shared by the referee-side prior-guided search adapter and the RIME-AD proposer (keeps proposers out of referee-package private helpers per AGENTS.md §1.1; moved verbatim from `apedp_dyn/epistemic/prior_guided_search.py`). | `production` |
| [`apedp_dyn/proposers/sparse_microstructure_proposer.py`](../apedp_dyn/proposers/sparse_microstructure_proposer.py) | Sparse ElasticNet hypothesis proposer generating sparse linear microstructure models. | `production` |
| [`apedp_dyn/proposers/resiliency_proposer.py`](../apedp_dyn/proposers/resiliency_proposer.py) | Nonlinear order-book resiliency proposer fitting exponential recovery dynamics. | `production` |
| [`apedp_dyn/proposers/distributed_lag_interaction_proposer.py`](../apedp_dyn/proposers/distributed_lag_interaction_proposer.py) | Distributed lag kernel and bilinear feature interaction microstructure proposer. | `production` |
| [`apedp_dyn/proposers/regime_switching_proposer.py`](../apedp_dyn/proposers/regime_switching_proposer.py) | Piecewise linear threshold models adaptive to market volatility states. | `production` |
| [`apedp_dyn/proposers/hawkes_orderflow_proposer.py`](../apedp_dyn/proposers/hawkes_orderflow_proposer.py) | Self-exciting point process modeling order arrival clustering and quote replenishment. | `production` |
| [`apedp_dyn/proposers/ncm_structural_proposer.py`](../apedp_dyn/proposers/ncm_structural_proposer.py) | Two-tier Next-Concept Modeling (NCM) structural proposer: decomposes scientific discovery into qualitative macro-concept archetypes (conserved oscillator, continuum transport, rational saturation, cyclic regimes) before micro-parameter estimation and sparsity pruning. | `experimental` |
| [`apedp_dyn/proposers/sindy_proposer.py`](../apedp_dyn/proposers/sindy_proposer.py) | SINDy-style sparse identification proposer: builds polynomial/trig library over state variables, applies sequential thresholded least squares for parsimonious ODE/PDE discovery with optional Rust acceleration. | `experimental` |
| [`apedp_dyn/proposers/reservoir_proposer.py`](../apedp_dyn/proposers/reservoir_proposer.py) | Echo State Network (reservoir computing) proposer: fixed random recurrent dynamics with ridge regression readout for chaotic/time-series systems where exact symbolic forms are hard to find. | `experimental` |
| [`apedp_dyn/proposers/basis_regression_proposer.py`](../apedp_dyn/proposers/basis_regression_proposer.py) | Basis function regression proposer (RBF/polynomials): fast screening step for guiding symbolic search with ridge regression over basis expansions. | `experimental` |
| [`apedp_dyn/epistemic/gp_uncertainty.py`](../apedp_dyn/epistemic/gp_uncertainty.py) | Gaussian Process uncertainty quantification: RBF kernel GP regressor with Cholesky, Bayesian optimization with expected improvement for distinguishing interpolative fits from extrapolative predictions. | `experimental` |
| [`apedp_dyn/epistemic/bayesian_active_learning.py`](../apedp_dyn/epistemic/bayesian_active_learning.py) | Bayesian active learning loop: uses GP uncertainty and acquisition functions to prioritize experiments/candidates for efficient discovery. | `experimental` |
| [`apedp_dyn/proposers/rime_proposer.py`](../apedp_dyn/proposers/rime_proposer.py) | RIME symbolic-mechanism proposer: emits closed-form equations from typed mechanism graphs, adjudicated by the same frozen referee. | `experimental` |
| [`apedp_dyn/proposers/stem_proposers.py`](../apedp_dyn/proposers/stem_proposers.py) | Modular multi-engine STEM proposers (Option 2): Hamiltonian energy-conserving mechanical systems, solenoidal divergence-free fluid velocity fields, and compartmental epidemiological dynamics. | `experimental` |
| [`apedp_dyn/proposers/invariant_transfer_proposer.py`](../apedp_dyn/proposers/invariant_transfer_proposer.py) | Active cross-regime invariant transfer proposer: IRM-penalized representations with worst-regime retention gating, Cross-Regime Generalization Index (CGI) ranking, and active extrapolation-barrier checks. | `production` |
| [`apedp_dyn/sandbox/execution_cost_models.py`](../apedp_dyn/sandbox/execution_cost_models.py) | Decoupled execution models (Spread+Slippage, Almgren-Chriss impact, Queue fill). | `production` |
| [`apedp_dyn/sandbox/calibrated_benchmark_suite.py`](../apedp_dyn/sandbox/calibrated_benchmark_suite.py) | Three-tier calibrated benchmark suite (Referee FPR/CAR/VRR, Discovery Power, Transfer). | `production` |
| [`apedp_dyn/sandbox/canonical_ledger.py`](../apedp_dyn/sandbox/canonical_ledger.py) | Canonical execution ledger with additive log accounting, stock-split invariance, and fee models. | `production` |
| [`apedp_dyn/sandbox/blind_benchmark.py`](../apedp_dyn/sandbox/blind_benchmark.py) | Air-gapped blind benchmark runner preventing in-sample contamination of test suites. | `production` |
| [`apedp_dyn/sandbox/blind_benchmark_protocol.py`](../apedp_dyn/sandbox/blind_benchmark_protocol.py) | Cryptographic verification protocol for blind benchmark suites and hash receipts. | `production` |
| [`apedp_dyn/sandbox/ga_depth2_equation_discovery.py`](../apedp_dyn/sandbox/ga_depth2_equation_discovery.py) | Genetic algorithm search across depth-2 symbolic equation grammar space. | `experimental` |
| [`apedp_dyn/sandbox/ga_depth3_equation_discovery.py`](../apedp_dyn/sandbox/ga_depth3_equation_discovery.py) | Genetic algorithm search across depth-3 symbolic equation grammar space. | `experimental` |
| [`apedp_dyn/sandbox/build_all_system_equations_lean_proofs.py`](../apedp_dyn/sandbox/build_all_system_equations_lean_proofs.py) | Cross-domain Lean 4 formal proof suite for 10 system equations. | `production` |
| [`apedp_dyn/sandbox/cpmd_hlik.py`](../apedp_dyn/sandbox/cpmd_hlik.py) | Continuous Predictive Market Discovery harness for H-LIK liquidity hypothesis. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_hxsc.py`](../apedp_dyn/sandbox/cpmd_hxsc.py) | Continuous Predictive Market Discovery harness for H-XSC spread curvature hypothesis. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_hdar.py`](../apedp_dyn/sandbox/cpmd_hdar.py) | Continuous Predictive Market Discovery harness for H-DAR directional asymmetry hypothesis. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_hofd.py`](../apedp_dyn/sandbox/cpmd_hofd.py) | Continuous Predictive Market Discovery harness for H-OFD order-flow divergence hypothesis. | `audit-only` |
| [`apedp_dyn/sandbox/eseds_dynamical.py`](../apedp_dyn/sandbox/eseds_dynamical.py) | Energy-dissipation and dynamical stability referee harness. | `audit-only` |
| [`apedp_dyn/sandbox/eso_packing.py`](../apedp_dyn/sandbox/eso_packing.py) | Geometric sphere-packing contact invariant referee harness. | `audit-only` |
| [`apedp_dyn/discovery_l2/ode_engine.py`](../apedp_dyn/discovery_l2/ode_engine.py) | Higher-order ODE numerical integration and parameter estimation engine. | `production` |
| [`apedp_dyn/discovery_l2/pruning.py`](../apedp_dyn/discovery_l2/pruning.py) | Symbolic expression AST simplification, zero-coefficient pruning, and caching. | `production` |
| [`apedp_dyn/discovery_l2/ast_generator.py`](../apedp_dyn/discovery_l2/ast_generator.py) | Physically typed AST construction and generator for L2 open discovery (bounded canonicalization; the drag-flow fix path). | `production` |
| [`apedp_dyn/discovery_l2/dimensional_types.py`](../apedp_dyn/discovery_l2/dimensional_types.py) | Physical dimensional metrology types (BaseDimension/Dimension) for L2 open discovery. | `production` |
| [`apedp_dyn/discovery_l2/fingerprint.py`](../apedp_dyn/discovery_l2/fingerprint.py) | Numeric fingerprinting for zero-cost candidate expression deduplication. | `production` |
| [`apedp_dyn/discovery_l2/l2_engine.py`](../apedp_dyn/discovery_l2/l2_engine.py) | L2 open discovery engine with physically typed AST and epistemic adjudication. | `production` |
| [`apedp_dyn/discovery_l2/multiterm_engine.py`](../apedp_dyn/discovery_l2/multiterm_engine.py) | Multi-term superposition discovery engine for L2 dynamics (structure-aware quota screen). | `production` |
| [`apedp_dyn/discovery_l2/active_intervention.py`](../apedp_dyn/discovery_l2/active_intervention.py) | L3 active interventional synthesis for resolving epistemic ties. | `experimental` |
| [`apedp_dyn/discovery_l2/discovery_stability_harness.py`](../apedp_dyn/discovery_l2/discovery_stability_harness.py) | Multi-seed discovery stability harness and identifiability-matrix generator. | `audit-only` |
| [`rust/apedp-kernel/src/genetic.rs`](../rust/apedp-kernel/src/genetic.rs) | Rust-accelerated genetic algorithm evaluator with verified minimum-MSE selection, top-level trig frequency/phase template seeding (`sin\|cos(f*x + p)`, rate 0.18, `allow_trans`-gated; closes the frozen N04 unreachability forensics), and post-evolution constant polish (golden-section coordinate descent with shrinking windows + Powell conjugate-direction escape, strict-improvement replacements, affine readout refit at every probe; never-worsens A/B pinned). | `production` |
| [`rust/apedp-vqc/src/lib.rs`](../rust/apedp-vqc/src/lib.rs) | The `apedp_vqc` Python extension module (PyO3, six entry points): `run_report`, `equivalence_images`, `distribution`, `generators`, `expectation`, `gate_tables`. Structured answers are single JSON strings, so the FFI contract is a schema rather than a tuple order; a refusal is `NotImplementedError` and means "this path could not answer", never "this candidate is bad" | `experimental` |
| [`rust/apedp-vqc/src/stabilizer.rs`](../rust/apedp-vqc/src/stabilizer.rs) | The stabilizer tableau in Rust: `n` signed Pauli generators, GF(2) elimination over packed word vectors, the "first anticommuting generator" measurement with phase arithmetic mod 4, reset, exact `probability_of_bitstring`, and `2**n` enumeration for `n <= 12`. A port of `apedp_dyn/vqc/stabilizer.py` that keeps its ordering, so deterministic answers match exactly | `experimental` |
| [`rust/apedp-vqc/src/pauli.rs`](../rust/apedp-vqc/src/pauli.rs) | Signed Pauli strings and the Clifford conjugation tables, **derived** at first use from the 2x2 and 4x4 gate unitaries (as the Python tier derives them) instead of transcribed, with the quaternion product table built the same way | `experimental` |
| [`rust/apedp-kernel/src/video.rs`](../rust/apedp-kernel/src/video.rs) | Native video kernel: global PSNR and windowed grayscale SSIM, zero-copy Sliced Wasserstein-2, Rayon-parallel 2D bilinear sub-pixel coordinate warping (`warp_bilinear_2d`), central-difference optical flow divergence (`compute_optical_flow_divergence`), and disocclusion mask detector (`detect_disocclusion_mask`). Exact-parity pinned against Python reference. | `production` |
| [`apedp_dyn/video/decoupled_transport.py`](../apedp_dyn/video/decoupled_transport.py) | Decoupled Continuous Spatio-Temporal Transport (DC-STT) PDE engine (`DecoupledSpatioTemporalTransport`) + trainable PyTorch neural velocity/source predictor (`NeuralTransportFlowNet` + `train_neural_transport_model`) parameterized via streamfunction curl ($\nabla^\perp \psi_\theta$) + compressible potential gradient ($\nabla \phi_\theta$) and disocclusion source injection ($\mathbf{s}_\theta \odot \mathbf{m}_{\text{dis}}$). Spec: [`docs/DECOUPLED_CONTINUOUS_SPATIOTEMPORAL_TRANSPORT_SPEC.md`](DECOUPLED_CONTINUOUS_SPATIOTEMPORAL_TRANSPORT_SPEC.md). | `experimental` |
| [`rust/apedp-kernel/src/rime/graph.rs`](../rust/apedp-kernel/src/rime/graph.rs) | RIME typed mechanism graph: series-at-a-time topological evaluation, `State` recursion, structural validation, canonical SHA-256 ids. | `experimental` |
| [`rust/apedp-kernel/src/rime/rewrite.rs`](../rust/apedp-kernel/src/rime/rewrite.rs) | The five frozen RIME structural rewrites (lag, interaction, state, nonlinearity wrap, prune). | `experimental` |
| [`rust/apedp-kernel/src/rime/eval.rs`](../rust/apedp-kernel/src/rime/eval.rs) | RIME scorer: ridge-fitted linear readout, charged degrees of freedom, parallel batch evaluation. | `experimental` |
| [`rust/apedp-kernel/src/rime/seeds.rs`](../rust/apedp-kernel/src/rime/seeds.rs) | Deterministic 60-form seed catalogue (levels, delays, differences, lead-lag interactions, memory, wrappers). | `experimental` |
| [`rust/apedp-kernel/src/rime/search.rs`](../rust/apedp-kernel/src/rime/search.rs) | Deterministic budgeted beam search over the frozen rewrites; null-model-anchored gate with EBIC selection charge. | `experimental` |
| [`rust/apedp-kernel/src/rime/record.rs`](../rust/apedp-kernel/src/rime/record.rs) | Immutable content-addressed RIME candidate records (graph id, parent, rewrite op, seed, dev score). | `experimental` |
| [`benchmarks/rime_vs_neural.py`](../benchmarks/rime_vs_neural.py) | Head-to-head benchmark: RIME vs ridge and MLP baselines across planted mechanisms, sample sizes, and a null arm. | `audit-only` |
| [`benchmarks/matched_compute.py`](../benchmarks/matched_compute.py) | v0.1.3 matched-compute comparison harness: RIME (temporal + static grammars) vs existing proposers vs PySR 2.7 head-to-head (serial-deterministic, binding timeout, lag columns for information parity), at equal wall budget with 5 repeats/utilization/environment metadata, scored on a preregistered grammar-coverage axis (out-of-coverage abstention is correct; emitted-but-failed is failed; static worlds require reversal-invariance so grid-order tracking scores `index_fit`.). Measured outcome: complementarity, not dominance — RIME-temporal 4/5, PySR 3/5, each recovering a family the other misses. | `audit-only` |
| [`benchmarks/run_jetflow_vs_diffusion_matched.py`](../benchmarks/run_jetflow_vs_diffusion_matched.py) | JetFlow-Video vs diffusion textbook baseline on shared wavelet latents and holdout split (`benchmarks/jetflow_vs_diffusion_matched_results.json`, `docs/jetflow_vs_diffusion_matched_benchmark.md`). After remediating the 7 architectural defects (3D Fourier positional encoding + local 2D stencil mixing, subband-preserving Noether conservation, 2-pass fused Jet2 material derivative, full-precision latent decode, and CFG zero-bias discipline), `jet_tt_2step` reaches 10.74 dB PSNR (+3.25 dB over pre-fix 7.49 dB), 0.514 SSIM (exceeding `diffusion_matched` 0.477 and `diffusion_frontier_K48` 0.492), 0.275 temporal smoothness (beating diffusion 0.365–0.372), and 8 model passes/clip (down from 20). The `--domain` parameter adds the second problem family (collision/occlusion, `apedp_dyn/video/collision_domain.py`) under the v0.3 prereg (`docs/PREREGISTRATION_JETFLOW_VIDEO_V03_COLLISION.md`); the exploratory collision run is a disclosed NEGATIVE result for transfer (diffusion wins on every pixel/consistency instrument) — the fluid advantage is domain-scoped, CONFIRMED 3/3 on the frozen confirmation seeds by [`benchmarks/run_jetflow_video_v03_confirmation.py`](../benchmarks/run_jetflow_video_v03_confirmation.py) (verdicts `benchmarks/jetflow_video_v03_confirmation_results.json`, report `docs/JETFLOW_VIDEO_V03_CONFIRMATION_REPORT.md`, incl. the disclosed ERR-H4-UNITS erratum). | `experimental` |
| [`scripts/run_mutation_battery.py`](../scripts/run_mutation_battery.py) | Targeted mutation-testing battery (REG-MUT, closed-with-evidence): **37 mutants** across the referee/leakage paths (MUT-A/B/C/D), EHI pillar + dispatch wiring (MUT-F), DE-arm (MUT-G), rust-kernel remediations (MUT-E), and the Pow-exponent + native-search enforcement paths (MUT-H1..H13: enforcement switch, bridge re-raise, requested-fallback suspension, strict flag contradictions, fallback labelling, fixed exponent menu, birth fit call site and sweeps, polish window, the `has_pow_const` birth-fit predicate itself, and the two benchmark entry points that arm process-wide enforcement). Fresh-interpreter pin runs with byte-identical restoration invariant; pristine sanity pins for every pytest group plus `cargo test --lib`; exit 0 only when ALL mutants are killed (a survivor is a REG-MUT failure). Kill results are reused from `benchmarks/mutation_battery_cache.json` only when the mutation bytes, pin argv, pin-input tree content hash (every `.py` source, `rust/**`, `specifications/**`, pytest config) and `rustc --version` all match, with every reuse flagged `cached: true` in the report; `--no-cache` forces a full re-verification (a warm cached run is ~2 min instead of ~7.5 min). Report: `benchmarks/mutation_battery_report.json` (ALL_KILLED, 37/37). CI gate `mutation-battery` + `scripts/check.sh` full tier. Honesty trails: MUT-C1 survived the first run and the Finding-10 pin was SHARPENED (`tests/test_dyn_hawkes_standardization_pins.py`), never weakened; MUT-H7 survived the 2026-10-23 extension because the general-Pow exponent-subtree masked the constant-exponent arm, and the distribution pin was sharpened to sample with `allow_advanced=false` (plus a minimum-sample guard). | `audit-only` |
| [`benchmarks/run_cbf_certificate_battery.py`](../benchmarks/run_cbf_certificate_battery.py) | REG-F16 evidence battery: runs the formal CBF-QP certificate sweeps (64 certified releases, fail-closed refusal pins, discrete invariance horizons), the model-checked unicycle realization through the production arc stepper (seeded adversarial head-on courses), and the SCALED-VERIFICATION layer (16/32/64/128-agent deterministic jittered-lattice grids: the same fail-closed formal checks re-verified with worst slack ≥ −tol at every scale, Euler invariance h > 0 at every step, and the realized guard collision-free through the production arc stepper under the realized-velocity-feedback nominal discipline — realized 128-agent min barrier 0.3919 — plus per-step WALL-CLOCK guard/stepper latencies recorded as measurements, e.g. mean guard+step 0.60/0.77/1.31/3.39 ms at 16/32/64/128 agents); a DISCLOSED scope probe records verbatim that fresh full-range random nominals redrawn every step at 128 agents are outside the realized layer's evidence (deterministic breach −0.1102). Writes `benchmarks/cbf_certificate_report.json`. Exit 0 requires every formal/scaled-layer check to hold and zero collisions in the verification disciplines. Boundary disclosed in the report: `batch_multi_agent_cbf_step` remains the artificial-potential deadlock-remediation heuristic; `certified_cbf_guard` is the additive certified layer; the unicycle realization is scenario-scoped model-checked evidence; external safety sign-off remains outstanding. | `audit-only` |
| [`benchmarks/run_lorenz_public_benchmark.py`](../benchmarks/run_lorenz_public_benchmark.py) | Frozen confirmation harness for the player-titled public Lorenz benchmark (`docs/PREREGISTRATION_LORENZ_PUBLIC_BENCHMARK_V1.md`, prereg IS the spec): catalog Lorenz world σ=10/ρ=28/β=8/3 × seeds {101,202,303} × noise {0.5,1,2,3}% = 12 registered runs per path; SG(21,3)+`np.gradient` estimator declared; 70/30 dev/sealed-confirmation split with full-lineage burn; four registered paths — PATH-R (referee-gated forward selection in dev-BIC order, promotion only on sealed-holdout `unique_support`), PATH-B (greedy forward training-BIC over the degree-≤3 library; the analytic-naive negative control), PATH-B′ (an external transcript's curated 4-candidate ranking), and PATH-C (the transcript's operational chimera — ground-truth σ/ρ/β baked in, −y hardcoded, transcript multipliers, 3D-plot eyeballing declared inadmissible); instruments: structure cleanliness, coefficient recovery in SE units (pilot-corrected floor ≤ 2.5 SE — the drafted 0.5 SE floor was REFUTED pre-freeze: SG derivative bias is systematic and worst at LOW noise), winner-vs-true sealed-holdout RMSE gap, referee status verbatim, and the same-integrator divergence-horizon ratio (integrator error cancels by construction — the instrument that catches the Euler/chimera validation Pattern). Pins `tests/test_dyn_lorenz_public_benchmark.py` (3: schema, H1 wiring margin, H6 chimera-catch). Results commit separately after the runs per the prereg's decision rule. | `audit-only` |
| [`scripts/run_lean_pinned_reproduction.py`](../scripts/run_lean_pinned_reproduction.py) | REG-LEAN reproduction runner: re-executes the 323-record exhaustive Lean sweep under a second independently provisioned pinned elan toolchain (default `leanprover--lean4---v4.34.1`, distinct from the reference run's v4.35.0), compares every record's `proof_claim_status` + `kernel_exit_code` against `artifacts/all_system_equations_lean_provenance.json`, and restores the reference artifact AND `.lean` evidence byte-identically (HEAD-committed bytes take precedence over regeneration). Report: `benchmarks/lean_pinned_toolchain_reproduction.json`. Result: 323/323 match, counts 323/315/8/0. Honesty boundary: reproduces, does not substitute the register's external provisioning/sign-off. | `audit-only` |
| [`apedp_dyn/robotics/cbf_certificate.py`](../apedp_dyn/robotics/cbf_certificate.py) | Formal CBF-QP certificate (REG-F16) for multi-agent collision avoidance at the certified velocity layer: explicit machine-checked assumption object (`CBFQPCertificateAssumptions`, refuses γ·dt > 0.5 and non-positive margins); fail-closed pairwise-barrier QP synthesis (`solve_cbf_qp` — every released command re-checked, `CertificateViolation` otherwise); closed-form factory feasibility lemma; projection-optimality containment; discrete forward invariance h≥(1−γdt)ᵏh₀; `certified_cbf_guard` additive over nominal commands; unicycle realization through `batch_unicycle_arc_step` model-checked (disclosed as evidence, not formal). Pins `tests/test_dyn_cbf_certificate.py` (8). | `experimental` |
| [`benchmarks/run_decoupled_transport_multiscene_benchmark.py`](../benchmarks/run_decoupled_transport_multiscene_benchmark.py) | Multi-scene Decoupled Continuous Spatio-Temporal Transport (DC-STT) benchmark: evaluates continuous PDE transport across vehicular translation, fluid vortex, expanding disocclusion with residual source texture injection, and camera zoom. Measures step latency ($2.25\text{ ms} - 6.91\text{ ms}$, overall 268.1 FPS on CPU), near-zero temporal acceleration flicker ($\|D_t^2 u\|_2 \sim 10^{-5}$ to $10^{-7}$), and 4D Tensor-Train compression ($10.9\times$ to $54.4\times$, up to 98.2% RAM saved) with near-lossless reconstruction ($\text{MSE} \le 8.66 \times 10^{-4}$). These are THROUGHPUT/RECONSTRUCTION claims on scenes with given (analytic) flow, not generation-quality claims. Artifact: `benchmarks/decoupled_transport_multiscene_results.json`. | `audit-only` |
| [`benchmarks/run_dcstt_horizon_decay.py`](../benchmarks/run_dcstt_horizon_decay.py) | DC-STT extrapolation-horizon decay audit (the instrument the multi-scene benchmark lacks): transport quality vs rollout horizon $H \in \{16,32,64,128,256\}$ against a 10x-fine-substep reference integrator of the SAME analytic flow (integrator fidelity 18–43 dB), plus appearance/energy and detail (blur-proxy) preservation curves — measured detail energy (blur proxy) falls to 11.3% of initial within $\sim$$H{=}128$ for the fluid-vortex scene, so unbounded-rollout serving quality is NOT supported by current evidence; short-horizon transport/upsampling is. Artifact: `benchmarks/dcstt_horizon_decay_results.json`. | `audit-only` |
| [`apedp_dyn/video/learned_flow_head.py`](../apedp_dyn/video/learned_flow_head.py) | Learned flow head for DC-STT (first learned component in the path): small CNN infers per-pixel $\mathbf{v}(\mathbf{x})$ from two context frames, trained self-supervised (advection-consistency + div² + TV) ENTIRELY ON CPU under the repo's wall-clock deadline discipline; torch-warp is sign-convention-parity-pinned to the Rust-bridge NumPy fallback ($\max|\Delta| < 10^{-12}$, `tests/test_dyn_learned_flow_head.py`, 6/6). | `experimental` |
| [`benchmarks/run_learned_flow_dcstt_benchmark.py`](../benchmarks/run_learned_flow_dcstt_benchmark.py) | Learned-flow DC-STT evaluation on the shared fluid-domain split (`n_clips=10`, holdouts `fluid_clip_0003/0005` — never train data): 90 s deadline CPU training (1367 steps, loss $4.4\mathrm{e}{-4} \to 6\mathrm{e}{-5}$), then DC-STT rollouts from 2 context frames scored per horizon against the clips' true futures vs persistence and integer-shift-search baselines. HONEST VERDICT (single-run, artifact-backed): the learned head does NOT yet beat persistence in aggregate (38.79/36.06/33.42 dB vs 39.54/37.21/35.24 dB at $H{=}4/8/14$; it wins only on one of two holdout clips) — recorded as-is, no tuning to green. Artifact: `benchmarks/learned_flow_dcstt_results.json`. | `audit-only` | | [`rust/apedp-kernel/tests/rime_profile.rs`](../rust/apedp-kernel/tests/rime_profile.rs) | Kernel phase-decomposition profile (graph eval / readout fit / search bookkeeping remainder; median of 7): bookkeeping ≈94% of per-candidate cost at n=150 (2026-10-22 re-measurement), graph evaluation dominating at the largest n (n=12 000: graph eval 176.7 µs vs readout fit 114.6 µs). Artifact: `benchmarks/kernel_profile_results.json`. | `audit-only` |
| [`benchmarks/openai_math_sandbox/run_sandbox.py`](../benchmarks/openai_math_sandbox/run_sandbox.py) | OpenAI-math release pipeline sandbox (`attempt -> certificate -> verification -> retraction accounting`) over 6 machine-checkable tasks plus 8 declared gap entries mirroring the OpenAI 2026-10-06 release (722 manuscripts, 3 retractions, 719 listed); no task is ever `verified` without a checker certificate (readme carries the measured gap analysis). Full-catalog ingestion (`catalog/`): all 372 families / 719 manuscripts from a SHA-256-pinned `openai/math` snapshot, Lean-catalogue join, withdrawal/numbering/formalization-count discrepancies recorded; 372/372 families `uncheckable_proof_level` (no proof engine). | `audit-only` |
| [`benchmarks/known_truth_challenge.py`](../benchmarks/known_truth_challenge.py) | v0.1.3 known-truth recovery challenge: blind RIME + BIC-OLS control on 11 planted monomial worlds (d ∈ {2,3,6}; grid/noise/coefficient/offset independent) + exp() negative control. Structural verdicts (effective exponent + coefficient ≤5%, reversal-invariant fit). Both-order fit required; abstention never counts as success. In-grammar abstention = miss; out-of-grammar emission recorded. v0.1.4 extensions: `--adjudicated` routes every emission through the referee-side scope refusal; `--repeats N` gives per-world R²/coef bounds over 5 noise seeds. Artifacts: `benchmarks/known_truth_results.json`, `benchmarks/known_truth_results_repeats.json`. | `audit-only` |
| [`benchmarks/run_rime_oof_multivariate_probe.py`](../benchmarks/run_rime_oof_multivariate_probe.py) | RIME-OOF-MV registered replication (PREREG-RIME-OOF-MV-2026-10-17): multivariate out-of-family surrogate probe, `RimeProposer` vs `RimeAdProposer` (fresh seeds 600/601/603). Recorded outcome: no proposer-layer abstention on any out-of-family law (confident surrogates, test R² in [0.62, 0.94], wide-range collapse down to 0.057); `RimeAdProposer` NO_SIGNAL multivariately (56–77 s per fit). Artifacts: `benchmarks/rime_oof_multivariate_results.json`, `docs/RIME_OOF_MV_REPORT.md`. | `audit-only` |
| [`benchmarks/run_scope_refusal_mv_validation.py`](../benchmarks/run_scope_refusal_mv_validation.py) | Spec §16e two-sided validation harness (PREREG-RIME-SCOPE-MV-2026-10-17): 27 cells over fresh seeds 700/701/702 — three aligned out-of-family worlds (sin-interaction, max-switch, amplitude-modulation), the exact in-family control, four §16c monomial guards, null world. Recorded outcome: **FAIL (negative result)** — all 9 out-of-family emissions declined, but the exact control is wrongfully refused by the periodic-quadratic and amplitude-modulated probes (c→0.1 refinement-floor linearization, Δ within τ) and all 12 guard cells are undefined (frozen variance guard vs §16c 1e-4-scale axes); the frozen family-set rule admits no retention assignment covering W1/W3. Determinism verified; void first run disclosed. Artifacts: `benchmarks/scope_refusal_mv_validation_results.json` (+ `..._void1_harnessbug.json`), `docs/RIME_SCOPE_MV_REPORT.md`, `tests/test_dyn_rime_scope_refusal_mv.py`. | `audit-only` |
| [`specifications/canonical_preregistered_spec_v1.json`](../specifications/canonical_preregistered_spec_v1.json) | Canonical frozen specification containing all preregistered thresholds. | `production` |
| [`apedp_dyn/operational/lifecycle_pruner.py`](../apedp_dyn/operational/lifecycle_pruner.py) | Operational storage footprint audit and automated lifecycle pruner: monitors filesystem headroom, detects and prunes Cargo build targets, auxiliary Lean 4 (elan) toolchains, package manager caches (uv, bun, npm/npx, pip), and IDE/browser caches with dry-run support and automated pre-flight self-healing guards. | `production` |
| [`apedp_dyn/conf/benchmark.py`](../apedp_dyn/conf/benchmark.py) | Legacy benchmark configuration and baseline threshold profiles. | `deprecated` |
| [`apedp_dyn/epistemic/transfer_scope.py`](../apedp_dyn/epistemic/transfer_scope.py) | Out-of-distribution transfer scope classifier evaluating empirical domain boundaries. | `production` |
| [`apedp_dyn/epistemic/diagnosis.py`](../apedp_dyn/epistemic/diagnosis.py) | Diagnostic failure attribution determining why candidates were refused at gates. | `production` |
| [`apedp_dyn/epistemic/number_theory_engine.py`](../apedp_dyn/epistemic/number_theory_engine.py) | Discrete/algebraic number theory discovery engine: exact rational reconstruction (Farey/continued fractions), integer relation discovery by branch-and-bound over a coefficient box (`brute_force_branch_and_bound`: refuses `n > 8`, carries no certificate — the LLL/PSLQ companion is the row below), discrete modular lattice step identification. | `production` |
| [`apedp_dyn/epistemic/coupled_discovery.py`](../apedp_dyn/epistemic/coupled_discovery.py) | Coupled multi-variable non-linear dynamical systems discovery: simultaneous multi-state variable ODE recovery ($\dot{\mathbf{x}} = \mathbf{F}(\mathbf{x})$) with interaction terms and system-level gating. | `production` |
| [`apedp_dyn/epistemic/bootstrap.py`](../apedp_dyn/epistemic/bootstrap.py) | Block-bootstrap and stationary bootstrap statistical confidence interval estimator. | `production` |
| [`apedp_dyn/epistemic/term_necessity.py`](../apedp_dyn/epistemic/term_necessity.py) | Term necessity and coefficient ablation testing framework. | `production` |
| [`apedp_dyn/epistemic/discovery_stability.py`](../apedp_dyn/epistemic/discovery_stability.py) | Discovery stability harness measuring parameter sensitivity across random seeds. | `production` |
| [`apedp_dyn/epistemic/agnostic_adapter.py`](../apedp_dyn/epistemic/agnostic_adapter.py) | Domain-agnostic evidence ingestion adapter standardizing holdout inputs. | `production` |
| [`apedp_dyn/epistemic/calibration.py`](../apedp_dyn/epistemic/calibration.py) | Model probability and uncertainty calibration verification. | `production` |
| [`apedp_dyn/epistemic/pointer.py`](../apedp_dyn/epistemic/pointer.py) | Epistemic pointer tracking references to empirical artifacts and receipts. | `production` |
| [`apedp_dyn/epistemic/high_dim_pde.py`](../apedp_dyn/epistemic/high_dim_pde.py) | Hybrid high-dimensional continuous PDE discovery, Jet2/Jet3 Taylor-mode CPU engine, and 5-way epistemic referee taxonomy. | `production` |
| [`apedp_dyn/epistemic/followup.py`](../apedp_dyn/epistemic/followup.py) | Automated followup experiment recommendation and active disambiguation engine. | `experimental` |
| [`apedp_dyn/grammar/mechanical_grammar.py`](../apedp_dyn/grammar/mechanical_grammar.py) | Mechanical and physical dynamical systems grammar definitions. | `production` |
| [`apedp_dyn/grammar/proposer.py`](../apedp_dyn/grammar/proposer.py) | Core grammar search proposer generating candidate AST expressions. | `production` |
| [`apedp_dyn/grammar/llm_proposer.py`](../apedp_dyn/grammar/llm_proposer.py) | Optional LLM-guided proposer (seam-compliant): suggests hypotheses via API while never deciding acceptance; all candidates go through normal gates. | `experimental` |
| [`apedp_dyn/grammar/coupled.py`](../apedp_dyn/grammar/coupled.py) | Coupled multi-variable differential equation grammar definitions. | `production` |
| [`apedp_dyn/grammar/canonicalize.py`](../apedp_dyn/grammar/canonicalize.py) | Symbolic expression canonicalization and term ordering. | `production` |
| [`apedp_dyn/grammar/term_format.py`](../apedp_dyn/grammar/term_format.py) | Coefficients formatted onto operands without splitting a non-product operand: `scaled`/`atom`/`is_atomic` leave an atomic name untouched — so recorded equations and candidate ids hashed from them do not move — and parenthesise a sum. This is the `q2*(XR − X)` defect class in reusable form, used wherever a caller-supplied name (a QED library term, a DataFrame column, a feature label) is multiplied by a fitted coefficient. Dependency-free. Pinned by `tests/test_term_builder_structure_audit.py`. | `production` |
| [`apedp_dyn/quantum/sparrow_photonics.py`](../apedp_dyn/quantum/sparrow_photonics.py) | Sparrow Quantum deterministic photonics engine: deterministic QD scaling vs SPDC, passive fusion with Excitation-Based Feedback (EBF), synchronous foliated Floquet colour code (sFFCC) resource scaling (Table III bill incl. fibre EOMs), a one-call L-scaling `scaling_report` with both clock bases, real-time feasibility gates (feedforward latency, phase-shifter switching, decoder throughput), and multi-channel fault-tolerance referee. Spec: [`docs/SPARROW_QUANTUM_DETERMINISTIC_PHOTONICS.md`](SPARROW_QUANTUM_DETERMINISTIC_PHOTONICS.md). | `experimental` |
| [`apedp_dyn/quantum/dmrg.py`](../apedp_dyn/quantum/dmrg.py) | Two-site DMRG in MPS/MPO form for the open spin-1/2 Heisenberg chain ($J\sum S_i\cdot S_{i+1} - h\sum S^z_i$): 5-channel MPO (one path = one Hamiltonian term, machine-precision parity vs sparse kron), matrix-free Lanczos two-site solver, L→R/R→L sweeps with SVD truncation, exact-diagonalization referee plus thermodynamic-limit constant $1/4-\ln 2$. Verified (`tests/test_dyn_dmrg_quantum.py`, 8/8): MPO expectation parity $<10^{-12}$; N=8/10/12 ground energies match ED to $<10^{-11}$; N=24 energy/site within 0.008 of $0.25-\ln 2$; field h=0.8 polarized-GS case matches ED. Honest scope: no time evolution, no finite temperature, no 2D/frustrated lattices. | `experimental` |
| [`apedp_dyn/quantum/field_tensor_search.py`](../apedp_dyn/quantum/field_tensor_search.py) | GF(2^12) finite-field multiplication tensor engine (modulus 4179, u^12+u^6+u^4+u+1), exact 12x12x12 tensor reconstruction, bilinear decomposition verification over GF(2), 4-tier baseline hierarchy (Schoolbook 144, Published Karatsuba 63, Composed 54, Published 51 Incumbent), flip-graph and SAT search, quantum resource evaluation, bit-exact integer/fixed-point `InvsqrtNumericalModel` over [1, 3,133,452], exact `AlgebraicFailureDiagnostics`, and matched-budget benchmark harness [`scripts/run_mmlm_poc_001.py`](../scripts/run_mmlm_poc_001.py). Spec: [`specifications/APEDP_QUANTUM_DISCOVERY_V1.md`](../specifications/APEDP_QUANTUM_DISCOVERY_V1.md). | `experimental` |
| [`apedp_dyn/market/lob_transport.py`](../apedp_dyn/market/lob_transport.py) | Limit-order-book depth-deviation transport discovery ($u_t = -v u_x + D u_{xx} - \lambda u$): spectral (exact-mode) and explicit RK4 method-of-lines snapshot generators on a periodic price grid, finite-difference least-squares recovery, per-mode spectral decay/phase recovery, statistical-observable cross-checks (CM drift, variance growth, mass decay), and 1D PDE-engine discovery hook. Verified (`tests/test_dyn_lob_transport.py`, 5/5): spectral recovery at machine precision ($10^{-16}$), FD recovery $<5\times10^{-3}$, observables $<10^{-6}$; engine recovers `u`,`u_x`,`u_xx` with coefficients matching $(-\lambda,-v,D)$ to $10^{-5}$ and holdout $R^2=1.0$. Honest scope: synthetic deviations field only; no real tick data, no queue-position dynamics. | `experimental` |
| [`apedp_dyn/audio/spectrogram_transport.py`](../apedp_dyn/audio/spectrogram_transport.py) | Audio spectrogram field transport: hand-rolled Hann STFT, parabolic sub-bin chirp ridge tracking, quadratic instantaneous-frequency model, and frequency-axis advection predicted through the repository transport stepper (`apedp_dyn.video.jet_flow_stepper`, Jet2 order 1/2). Verified (`tests/test_dyn_audio_field_transport.py`, 5/5): ridge tracks true IF within 1.7 Hz (bin 31 Hz); next-frame transport $R^2>0.995$; frequency-axis advection uses a CFL-enforced upwind integrator (a centered Euler/Jet2 attribute transport is unconditionally unstable for pure advection, so predictions stay stable and non-negative and are order-invariant); the generic Jet2 stepper's second-order advantage is pinned at the stepper level on the accelerating ridge (relative error 1.87e-3 -> 8.7e-5, 21.5x). Honest scope: synthetic chirps only; no music/speech/real-audio claims. | `experimental` |
| [`apedp_dyn/molecular/se3_flow.py`](../apedp_dyn/molecular/se3_flow.py) | Molecular SE(3)/O(3)-equivariant conformer flow: invariant distance-based force field (harmonic bonds + soft repulsion), analytic Cartesian gradient, and gradient-flow relaxation integrated with the transport stepper. Verified (`tests/test_dyn_molecular_se3_flow.py`, 6/6): energy invariance under rigid/reflection transforms and velocity equivariance at $10^{-15}$; analytic gradient matches central differences at $10^{-8}$; relaxation is strictly energy-decreasing with target bond lengths within $10^{-2}$; COM conserved to $10^{-15}$; mirror-image relaxation is an exact mirror (gap 0.0). Documented limitation: distance-only energies are achiral. | `experimental` |
| [`apedp_dyn/robotics/trajectory_transport.py`](../apedp_dyn/robotics/trajectory_transport.py) | Robot trajectory transport: kinematic unicycle with exact arc stepping, transport-stepper propagation under open-loop commands and a smooth polar feedback law (closed loop), and goal navigation with a position-tolerance stop. Verified (`tests/test_dyn_robotics_trajectory.py`, 6/6): exact arc analytics to $10^{-12}$; full-path Euler/Jet2 errors show textbook convergence ratios 2.00/4.00; 20/20 random starts converge to goal within tolerance with strictly non-increasing goal distance (Lyapunov) at step 0.02. Honest scope: kinematics only - no wheel dynamics, slip, obstacles, or sensing. | `experimental` |

| [`apedp_dyn/dyn_sweep.py`](../apedp_dyn/dyn_sweep.py) | Frozen DUFF/COUP/STAB four-noise sweep retained as the legacy adjudication regression boundary. | `audit-only` |
| [`apedp_dyn/dyn_tournament.py`](../apedp_dyn/dyn_tournament.py) | Frozen DUFF 3%-adversarial tournament (v1 regression boundary) invoked by benchmark-v2 harnesses. | `audit-only` |
| [`apedp_dyn/gates/admissibility.py`](../apedp_dyn/gates/admissibility.py) | Gate 2 physical admissibility: structure, symbol, function, and bound checks before scoring. | `production` |
| [`apedp_dyn/gates/dimensional.py`](../apedp_dyn/gates/dimensional.py) | Gate 1 mathematical/dimensional validity with GateResult and DAO checks. | `production` |
| [`apedp_dyn/gates/intervention.py`](../apedp_dyn/gates/intervention.py) | Gate 4 discrimination/intervention: held-out experiment decides between surviving candidates. | `production` |
| [`apedp_dyn/gates/scaling_invariance.py`](../apedp_dyn/gates/scaling_invariance.py) | Gate 1b scaling-invariance check for D5-CONF (generators, regimes, result contract). | `production` |
| [`apedp_dyn/gates/simulation.py`](../apedp_dyn/gates/simulation.py) | Gate 3 forward simulation: rejects failures, NaNs, infinities, and unstable trajectories. | `production` |
| [`apedp_dyn/ode/data.py`](../apedp_dyn/ode/data.py) | Typed trajectory containers for continuous ODE discovery. | `production` |
| [`apedp_dyn/ode/dispatch.py`](../apedp_dyn/ode/dispatch.py) | Unified dispatch and functional API for ODE and coupled-system discovery. | `production` |
| [`apedp_dyn/ode/proposer.py`](../apedp_dyn/ode/proposer.py) | CoupledOdeProposer contract for development-data-only candidate emission. | `production` |
| [`apedp_dyn/ode/referee.py`](../apedp_dyn/ode/referee.py) | OdeReferee contract: holdout-rollout adjudication, instability refusal, fail-closed state checks. | `production` |
| [`apedp_dyn/pde/data.py`](../apedp_dyn/pde/data.py) | Typed grid and point-cloud containers for continuous PDE discovery. | `production` |
| [`apedp_dyn/pde/dispatch.py`](../apedp_dyn/pde/dispatch.py) | Unified dispatch and functional API for PDE discovery across dimensions. | `production` |
| [`apedp_dyn/pde/proposer.py`](../apedp_dyn/pde/proposer.py) | PdeProposer contract for development-data-only candidate emission. | `production` |
| [`apedp_dyn/pde/referee.py`](../apedp_dyn/pde/referee.py) | PdeReferee contract for holdout adjudication of continuous PDE candidates. | `production` |
| [`apedp_dyn/market/assets.py`](../apedp_dyn/market/assets.py) | Historical market loaders with strictly causal feature generation. | `production` |
| [`apedp_dyn/market/fetch_crypto_orderflow.py`](../apedp_dyn/market/fetch_crypto_orderflow.py) | Binance daily taker buy/sell order-flow fetch utility (network-dependent ingestion). | `experimental` |
| [`apedp_dyn/market/regimes.py`](../apedp_dyn/market/regimes.py) | Market-regime taxonomy with causal statistical classification. | `production` |
| [`apedp_dyn/market/sweep.py`](../apedp_dyn/market/sweep.py) | Window sweep engine and summary records for open market discovery. | `production` |
| [`apedp_dyn/market/windows.py`](../apedp_dyn/market/windows.py) | Frozen, rolling, and generated market-window registry. | `production` |
| [`apedp_dyn/video/continuous_interpolator.py`](../apedp_dyn/video/continuous_interpolator.py) | Continuous temporal interpolator for infinite-FPS, zero-flicker slow motion on the 4D Tensor-Train video representation: scipy `CubicSpline` (natural BC) over the temporal TT core `G0(T, R1)` with `sample_at_timestamps`, `resample_to_fps` (24 → 60/120 FPS) and the `measure_temporal_smoothness` \|D_t²u\| flicker metric; linear fallback below 4 frames. Smoothness/spatial-warp companion of the torch-side splines in [`jetflow_video/tensor_train.py`](../packages/jetflow-video/jetflow_video/tensor_train.py). Benchmarked: [`benchmarks/run_jet_flow_video_benchmark.py`](../benchmarks/run_jet_flow_video_benchmark.py) P; slow-motion demo: [`benchmarks/run_vortex_slow_motion_demo.py`](../benchmarks/run_vortex_slow_motion_demo.py) P. | `experimental` |
| [`apedp_dyn/video/jet_flow_stepper.py`](../apedp_dyn/video/jet_flow_stepper.py) | NumPy Jet2/Jet4 curvature stepper for high-precision video generation — fixed-step 2nd-order Taylor integrator `z_{t+dt} = z_t + dt·v_t + ½dt²·a_t` with `generate_video`'s forward `[0.0 → 1.0]` schedule; the separate Torch `JetFlowVideoModel.generate_video` uses `[1.0 → 0.0]`. Continuous-tau smooth playback is NOT its job (see `continuous_interpolator`). | `experimental` |
| [`apedp_dyn/video/collision_domain.py`](../apedp_dyn/video/collision_domain.py) | Collision/occlusion problem family for the JetFlow-Video v0.3 prereg — the second, domain-shifted benchmark family used to test whether the fluid-domain advantage transfers (it does not; see the disclosed NEGATIVE row). | `experimental` |
| [`apedp_dyn/video/noether_engine.py`](../apedp_dyn/video/noether_engine.py) | Noether invariant conservation and symplectic Lie-algebraic engine. | `experimental` |
| [`apedp_dyn/video/physics_knobs.py`](../apedp_dyn/video/physics_knobs.py) | Physical steering knobs and camera SE(3) flow controller. | `experimental` |
| [`apedp_dyn/video/region_masks.py`](../apedp_dyn/video/region_masks.py) | Region masks derived from the picture instead of from fixed coordinates: per-row texture energy (mean absolute horizontal gradient over channels), the two sharpest adjacent-row steps as band boundaries, and ramps normalised to a partition. Replaces the animate-picture demo's hard-coded top-40 %/bottom-40 %/middle-30 % bands. The assignment is by texture rank only — naming a band is the caller's decision — and the returned record carries the boundaries and band energies. Audited against ten real photographs in [`docs/REGION_MASK_AUDIT.md`](REGION_MASK_AUDIT.md): 2 close, 1 partial, 4 wrong, and 3 frames with no three-band structure that still receive two boundaries — median boundary error 0.18 of the frame height. Carries an `evidence` block (`cut_decisiveness`, `half_disagreement`, `texture_order`, `reliable`, `reasons`) and raises `UnusableRegionSplit` under `require_reliable=True`; the gate accepts 1 of the 10 photographs and none of the 4 wrong splits. A two-band result is not implemented. Re-run with a pretrained MiDaS depth map in place of RGB and the splitter, labels and tolerance held fixed: 0 close / 4 partial / 3 wrong against texture's 2 / 1 / 4, median boundary error 0.228 vs 0.182, and every depth split refused by the gate — the depth row profiles are ramp-dominated (median ramp R² 0.83 vs 0.24), so a step-seeking split has nothing to cut (`data/region_mask_audit/model_comparison.json`). A third arm cuts the bands where the **ADE20K class composition** changes (SegFormer-B0, weights pinned), which is the score a class map needs because ADE20K ids are nominal and cannot be differenced: median boundary error **0.115** against texture's 0.182 and depth's 0.228, worst 0.268, but 2 close / 1 partial / 4 wrong — the same verdicts as texture, none changed — with 4 of 10 splits endorsed by the gate (`data/region_mask_audit/semantic_comparison.json`). Pinned by `tests/test_scaffold_refusals.py`, `tests/test_region_mask_photo_audit.py`, `tests/test_region_mask_depth_comparison.py` and `tests/test_region_mask_semantic_comparison.py`. | `experimental` |
| [`apedp_dyn/video/tensor_train_video.py`](../apedp_dyn/video/tensor_train_video.py) | 4D spatio-temporal tensor-train latent video representation and compression. | `experimental` |
| [`apedp_dyn/video/vae_bridge.py`](../apedp_dyn/video/vae_bridge.py) | Video VAE latent bridge and invertible continuous wavelet autoencoder. | `experimental` |
| [`apedp_dyn/estimation/pointwise.py`](../apedp_dyn/estimation/pointwise.py) | Pointwise derivative estimator, D1 baseline arm of the D2a A/B. | `experimental` |
| [`apedp_dyn/estimation/square_window.py`](../apedp_dyn/estimation/square_window.py) | Square-window weak estimator, third arm of the D2a A/B. | `experimental` |
| [`apedp_dyn/estimation/weak_form.py`](../apedp_dyn/estimation/weak_form.py) | Weak-form estimation for fixed-structure candidates (D2a). | `experimental` |
| [`apedp_dyn/model/equations.py`](../apedp_dyn/model/equations.py) | D1 model equations: one explicit first-order ODE per state. | `production` |
| [`apedp_dyn/model/physical_model.py`](../apedp_dyn/model/physical_model.py) | D1 executable first-order dynamical law. | `production` |
| [`apedp_dyn/model/variables.py`](../apedp_dyn/model/variables.py) | D1 state, exogenous input, and constant-parameter declarations. | `production` |
| [`apedp_dyn/harness/core.py`](../apedp_dyn/harness/core.py) | AEH core orchestrator and harness result package. | `audit-only` |
| [`apedp_dyn/harness/ledger.py`](../apedp_dyn/harness/ledger.py) | Cryptographic event ledger and EEDR repair-invariant auditor. | `audit-only` |
| [`apedp_dyn/harness/manifest.py`](../apedp_dyn/harness/manifest.py) | Immutable experiment manifest with dataset hashing and git SHA capture. | `audit-only` |
| [`apedp_dyn/harness/roles.py`](../apedp_dyn/harness/roles.py) | Capability isolation and permission firewall for the experimental harness. | `audit-only` |
| [`apedp_dyn/fitting/parameter_fit.py`](../apedp_dyn/fitting/parameter_fit.py) | Bounded trajectory least-squares parameter fitter on frozen IR. | `production` |
| [`apedp_dyn/simulation/ode_solver.py`](../apedp_dyn/simulation/ode_solver.py) | D1 ODE simulation downstream of gates 1-2, feeding gate 3 and the fitter. | `production` |
| [`apedp_dyn/validation/energy.py`](../apedp_dyn/validation/energy.py) | D5 energy conservation and balance-law validators. | `production` |
| [`apedp_dyn/experiments/d2a_ab.py`](../apedp_dyn/experiments/d2a_ab.py) | D2a controlled A/B: spline/pointwise derivative vs weak/integral estimation. | `audit-only` |
| [`apedp_dyn/problems/base.py`](../apedp_dyn/problems/base.py) | Shared scaffolding for D1 benchmark problems. | `production` |
| [`apedp_dyn/problems/catalog.py`](../apedp_dyn/problems/catalog.py) | APEDP 1000+ dynamical-systems and equations benchmark catalog. | `production` |
| [`apedp_dyn/problems/coupled.py`](../apedp_dyn/problems/coupled.py) | D2-COUPLED: 2-DOF spring-coupled oscillators with coupling topology. | `production` |
| [`apedp_dyn/problems/damped_pendulum.py`](../apedp_dyn/problems/damped_pendulum.py) | PS-DAMPPEND: damped pendulum, a nonlinear dissipative system. | `production` |
| [`apedp_dyn/problems/duffing.py`](../apedp_dyn/problems/duffing.py) | D2-DUFF: Duffing oscillator with nonlinear restoring force. | `production` |
| [`apedp_dyn/problems/exponential_decay.py`](../apedp_dyn/problems/exponential_decay.py) | PS-DECAY: exponential decay, the simplest first-order ODE. | `production` |
| [`apedp_dyn/problems/finance.py`](../apedp_dyn/problems/finance.py) | Second-order price dynamics on real market data. | `production` |
| [`apedp_dyn/problems/logistic.py`](../apedp_dyn/problems/logistic.py) | PS-LOG: logistic population growth, a nonlinear first-order ODE. | `production` |
| [`apedp_dyn/problems/lotka_volterra.py`](../apedp_dyn/problems/lotka_volterra.py) | PS-LV: Lotka-Volterra predator-prey coupled nonlinear dynamics. | `production` |
| [`apedp_dyn/problems/mass_spring_damper.py`](../apedp_dyn/problems/mass_spring_damper.py) | D1-MSD: damped harmonic oscillator with forced input. | `production` |
| [`apedp_dyn/problems/pendulum.py`](../apedp_dyn/problems/pendulum.py) | D1-PEND: nonlinear pendulum with torque input. | `production` |
| [`apedp_dyn/problems/quadratic_drag.py`](../apedp_dyn/problems/quadratic_drag.py) | PS-DRAG: quadratic drag, a nonlinear dissipation benchmark. | `production` |
| [`apedp_dyn/problems/rc_circuit.py`](../apedp_dyn/problems/rc_circuit.py) | PS-RC: first-order RC circuit benchmark. | `production` |
| [`apedp_dyn/problems/reaction_kinetics.py`](../apedp_dyn/problems/reaction_kinetics.py) | PS-RXNK: first-order A-to-B reaction kinetics. | `production` |
| [`apedp_dyn/problems/rlc_circuit.py`](../apedp_dyn/problems/rlc_circuit.py) | PS-RLC: second-order RLC circuit with coupled dynamics. | `production` |
| [`apedp_dyn/problems/thermal_rc.py`](../apedp_dyn/problems/thermal_rc.py) | PS-THERM: first-order thermal RC, Newton's law of cooling. | `production` |
| [`apedp_dyn/problems/two_body_gravity.py`](../apedp_dyn/problems/two_body_gravity.py) | PS-GRAV2: two-body gravitational system in 1D radial form. | `production` |
| [`apedp_dyn/problems/van_der_pol.py`](../apedp_dyn/problems/van_der_pol.py) | PS-VDPOL: Van der Pol oscillator with self-sustained oscillations. | `production` |

| [`apedp_dyn/sandbox/agnostic_domain_sandbox.py`](../apedp_dyn/sandbox/agnostic_domain_sandbox.py) | APEDP-X Agnostic Domain Adapter & 3-Arm Ablation Benchmark Suite. | `experimental` |
| [`apedp_dyn/sandbox/airgap_evaluator.py`](../apedp_dyn/sandbox/airgap_evaluator.py) | Air-Gapped External Evaluator CLI for APEDP-X. | `audit-only` |
| [`apedp_dyn/sandbox/apedp_discovery.py`](../apedp_dyn/sandbox/apedp_discovery.py) | APEDP Research Sandbox: Tokamak Plasma Parameter Space Exploration. | `audit-only` |
| [`apedp_dyn/sandbox/apedp_x_meta_sandbox.py`](../apedp_dyn/sandbox/apedp_x_meta_sandbox.py) | APEDP-X: Unified Cross-Domain Epistemic Claim-Evidence Adjudication Meta-Sandbox. | `production` |
| [`apedp_dyn/sandbox/auto_degradation.py`](../apedp_dyn/sandbox/auto_degradation.py) | AD-CAB Stage A: synthetic automotive component degradation (ground truth). | `audit-only` |
| [`apedp_dyn/sandbox/causal_discovery_sandbox.py`](../apedp_dyn/sandbox/causal_discovery_sandbox.py) | APEDP-X Domain B: Causal Discovery Under Confounding & Interventions Sandbox. | `production` |
| [`apedp_dyn/sandbox/cpmd_audit_harness.py`](../apedp_dyn/sandbox/cpmd_audit_harness.py) | CPMD v0.17 — generator-aware discovery audit. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_calibration.py`](../apedp_dyn/sandbox/cpmd_calibration.py) | CPMD v0.9.5 — selector calibration under APEDP-X freeze discipline. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_discovery_suite.py`](../apedp_dyn/sandbox/cpmd_discovery_suite.py) | CPMD v0.16 — frozen discovery suite H-CTP / H-SER / H-SAC. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_effective_slope.py`](../apedp_dyn/sandbox/cpmd_effective_slope.py) | CPMD v0.10 — finite-lag effective slope (diagnostic phase). | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_effective_slope_b.py`](../apedp_dyn/sandbox/cpmd_effective_slope_b.py) | CPMD v0.10b — OLS-vs-model advantage hunt (diagnostic only). | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_envelope.py`](../apedp_dyn/sandbox/cpmd_envelope.py) | CPMD v0.13 — operating envelope + uncertainty feasibility. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_estimand.py`](../apedp_dyn/sandbox/cpmd_estimand.py) | CPMD v0.12 — estimand alignment (Tests A-D). | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_gen_realistic.py`](../apedp_dyn/sandbox/cpmd_gen_realistic.py) | gen-R: real-calibrated synthetic market generator. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_hlik_stage3.py`](../apedp_dyn/sandbox/cpmd_hlik_stage3.py) | CPMD-HLIK Stage 3 — real-data economic evaluation. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_ols_uncertainty.py`](../apedp_dyn/sandbox/cpmd_ols_uncertainty.py) | CPMD v0.11 — OLS uncertainty calibration under correlated noise. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_ridge_diagnostics.py`](../apedp_dyn/sandbox/cpmd_ridge_diagnostics.py) | CPMD ridge diagnostics — direct confirmation of alpha-tau0 confounding. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_synth_trading.py`](../apedp_dyn/sandbox/cpmd_synth_trading.py) | CPMD-ST1 — synthetic trading test of OLS-T slope signal. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_synth_trading2.py`](../apedp_dyn/sandbox/cpmd_synth_trading2.py) | CPMD-ST2 — rolling-window sensitivity with phase-split validation. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_synth_trading3.py`](../apedp_dyn/sandbox/cpmd_synth_trading3.py) | CPMD-ST3 / v0.16 — mechanism discrimination (Tests A-C). | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_v014.py`](../apedp_dyn/sandbox/cpmd_v014.py) | CPMD v0.14 — frozen v0.13 procedure under independent eval, correlated | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_v015.py`](../apedp_dyn/sandbox/cpmd_v015.py) | CPMD v0.15 — narrow-claim confirmation (Option A). | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_v018.py`](../apedp_dyn/sandbox/cpmd_v018.py) | CPMD v0.18 — signal-strength calibration & controlled discovery. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_v019.py`](../apedp_dyn/sandbox/cpmd_v019.py) | CPMD v0.19 — hidden-mechanism recovery benchmark. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_v020.py`](../apedp_dyn/sandbox/cpmd_v020.py) | CPMD v0.20 — identifiability & observational equivalence benchmark. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_v021.py`](../apedp_dyn/sandbox/cpmd_v021.py) | CPMD v0.21 — oracle-calibrated identifiability benchmark. | `audit-only` |
| [`apedp_dyn/sandbox/creep_powerlaw.py`](../apedp_dyn/sandbox/creep_powerlaw.py) | Creep sandbox: Norton power-law discovery (steady-state creep rate). | `experimental` |
| [`apedp_dyn/sandbox/crypto_grammar.py`](../apedp_dyn/sandbox/crypto_grammar.py) | Grammar-driven discovery on combined 15-asset crypto universe (CRYPTO-PRICE). | `experimental` |
| [`apedp_dyn/sandbox/custom_upload.py`](../apedp_dyn/sandbox/custom_upload.py) | Intelligent universal custom dataset upload and autonomous discovery engine. | `experimental` |
| [`apedp_dyn/sandbox/deepmind_challenge_suite.py`](../apedp_dyn/sandbox/deepmind_challenge_suite.py) | APEDP-X Challenge Suite Based on Published DeepMind Discoveries. | `experimental` |
| [`apedp_dyn/sandbox/demo_cross_domain_lean_verification.py`](../apedp_dyn/sandbox/demo_cross_domain_lean_verification.py) | Cross-Domain Demonstration of the Agnostic / Universal Lean 4 Verification Bridge. | `audit-only` |
| [`apedp_dyn/sandbox/epidemiology_sir.py`](../apedp_dyn/sandbox/epidemiology_sir.py) | SIR epidemiology sandbox: R0 threshold + vaccination intervention. | `experimental` |
| [`apedp_dyn/sandbox/expanded_space_discovery_sweep.py`](../apedp_dyn/sandbox/expanded_space_discovery_sweep.py) | Expanded Feature Space Quantitative Discovery Sweep across Multi-Asset Universes & Horizons. | `audit-only` |
| [`apedp_dyn/sandbox/finance_coupled.py`](../apedp_dyn/sandbox/finance_coupled.py) | Joint SPX/NDX discovery with coupling topology (FIN-COUPLED). | `audit-only` |
| [`apedp_dyn/sandbox/finance_demo.py`](../apedp_dyn/sandbox/finance_demo.py) | APEDP Financial Time Series Momentum Discovery Demo. | `audit-only` |
| [`apedp_dyn/sandbox/finance_grammar.py`](../apedp_dyn/sandbox/finance_grammar.py) | Grammar-driven discovery on real market data (FIN-PRICE). | `experimental` |
| [`apedp_dyn/sandbox/finance_l2_demo.py`](../apedp_dyn/sandbox/finance_l2_demo.py) | APEDP Level 2 Financial Discovery & Epistemic Attribution Sandbox. | `audit-only` |
| [`apedp_dyn/sandbox/finance_l3_reachable.py`](../apedp_dyn/sandbox/finance_l3_reachable.py) | Finance L3 with reachability constraints (redo of a failed execution). | `audit-only` |
| [`apedp_dyn/sandbox/fluid_ode_dynamical_discovery.py`](../apedp_dyn/sandbox/fluid_ode_dynamical_discovery.py) | Dynamical Systems and Fluid Analogies Discovery Sweep across Multi-Asset Universes and Horizons. | `audit-only` |
| [`apedp_dyn/sandbox/game_theory.py`](../apedp_dyn/sandbox/game_theory.py) | Game-theory sandbox: which game was played, recovered from adaptation paths. | `experimental` |
| [`apedp_dyn/sandbox/goldbach_sandbox.py`](../apedp_dyn/sandbox/goldbach_sandbox.py) | Goldbach Conjecture Empirical Verification & Epistemic Adjudication Sandbox. | `experimental` |
| [`apedp_dyn/sandbox/grammar_space.py`](../apedp_dyn/sandbox/grammar_space.py) | Grammar space for volatility hypotheses (forensics spec S81); imported by the frozen-spec verifier and the finance adjudication forensics. Expanded space (v1.10 re-mint): hypotheses `H-P`/`H-LOG`/`H-SIG`/`H-ASYM`/`H-OFI`/`H-HURST`/`H-SQZ`/`H-RANK`/`H-MSNR`/`H-SQRTV` (concave level-volume `y=a+b*sqrt(V_t)`), unary ops `abs`/`log`/`zscore`/`rank`/`sqrt`/`tanh`, binary ops `+`/`-`/`*`/`/`/`min`/`max`, and `check_sqrt` as the domain guard for the new total-order op. | `production` |
| [`apedp_dyn/sandbox/hall_pecht_demo.py`](../apedp_dyn/sandbox/hall_pecht_demo.py) | APEDP Hall-Petch grain-size strengthening demo. | `experimental` |
| [`apedp_dyn/sandbox/hypothesis_space_experiment.py`](../apedp_dyn/sandbox/hypothesis_space_experiment.py) | Hypothesis-Space Expansion Benchmark: Model Starvation vs Genuine Non-Identifiability. | `audit-only` |
| [`apedp_dyn/sandbox/independent_proposer_benchmark.py`](../apedp_dyn/sandbox/independent_proposer_benchmark.py) | APEDP-X Blind Benchmark v1: Sequestered Proposer Evaluation Suite. | `audit-only` |
| [`apedp_dyn/sandbox/kuramoto_sync.py`](../apedp_dyn/sandbox/kuramoto_sync.py) | Kuramoto sandbox: synchronization transition law discovery. | `audit-only` |
| [`apedp_dyn/sandbox/lynx_hare.py`](../apedp_dyn/sandbox/lynx_hare.py) | Lynx-hare sandbox (real data): cyclicity discovery in hare-lynx trappings. | `experimental` |
| [`apedp_dyn/sandbox/market_real_data_benchmark.py`](../apedp_dyn/sandbox/market_real_data_benchmark.py) | Real Market Data Benchmark Suite for APEDP-DYN L2 Financial Discovery. | `experimental` |
| [`apedp_dyn/sandbox/materials_discovery_l1.py`](../apedp_dyn/sandbox/materials_discovery_l1.py) | APEDP Level 1 Constrained Materials Discovery Benchmark. | `audit-only` |
| [`apedp_dyn/sandbox/multi_asset_signals_sandbox.py`](../apedp_dyn/sandbox/multi_asset_signals_sandbox.py) | 15-Asset Synchronized Multi-Asset Portfolio Sandbox with Non-Linear Genetic Algorithm & Risk Parity: | `production` |
| [`apedp_dyn/sandbox/multi_grammar_sandbox.py`](../apedp_dyn/sandbox/multi_grammar_sandbox.py) | Multi-grammar sandbox: tests volatility hypotheses and multi-grammar combinations | `production` |
| [`apedp_dyn/sandbox/multicommodity_pooled.py`](../apedp_dyn/sandbox/multicommodity_pooled.py) | Multi-commodity pooled discovery: one table, one verdict. | `audit-only` |
| [`apedp_dyn/sandbox/nano_crossover_m4.py`](../apedp_dyn/sandbox/nano_crossover_m4.py) | M4 nanoscale crossover benchmark: classical Hall-Petch vs inverse softening. | `experimental` |
| [`apedp_dyn/sandbox/openai_math_advanced_hypotheses.py`](../apedp_dyn/sandbox/openai_math_advanced_hypotheses.py) | Advanced Hypotheses from OpenAI Mathematics & Dynamical Systems Publications. | `audit-only` |
| [`apedp_dyn/sandbox/pde_verification_sandbox.py`](../apedp_dyn/sandbox/pde_verification_sandbox.py) | APEDP-X Domain D: Computational Physics, PDE & Numerical Solver Verification Sandbox. | `production` |
| [`apedp_dyn/sandbox/pendulum_drag.py`](../apedp_dyn/sandbox/pendulum_drag.py) | Damped pendulum with quadratic drag: transfer test for fin-0017's family. | `experimental` |
| [`apedp_dyn/sandbox/plasma_simulator.py`](../apedp_dyn/sandbox/plasma_simulator.py) | Synthetic plasma thermal sandbox for APEDP research. | `audit-only` |
| [`apedp_dyn/sandbox/prospective_regime_tournament.py`](../apedp_dyn/sandbox/prospective_regime_tournament.py) | Prospective Validation Tournament of Regime-Conditioned Equation Families. | `audit-only` |
| [`apedp_dyn/sandbox/quantum_vacuum_sandbox.py`](../apedp_dyn/sandbox/quantum_vacuum_sandbox.py) | APEDP Quantum Vacuum & String-Breaking Sandbox (Duke Trapped-Ion Simulator). | `experimental` |
| [`apedp_dyn/sandbox/real_market_l2_sweep.py`](../apedp_dyn/sandbox/real_market_l2_sweep.py) | Frozen Real-Data Rolling-Window L2 Open Discovery Sweep. | `audit-only` |
| [`apedp_dyn/sandbox/riemann_sandbox.py`](../apedp_dyn/sandbox/riemann_sandbox.py) | Riemann Hypothesis & Prime Distribution Explicit Formula Sandbox (Native Rust Engine). | `experimental` |
| [`apedp_dyn/sandbox/routed_multiasset.py`](../apedp_dyn/sandbox/routed_multiasset.py) | Routed ensemble (vol-gated 007/017) across 5 assets, 2019-2023. | `audit-only` |
| [`apedp_dyn/sandbox/sphere_packing_sandbox.py`](../apedp_dyn/sandbox/sphere_packing_sandbox.py) | EP-OPT-1: Epistemically Safe Optimization & Sphere Packing Adjudication Sandbox. | `production` |
| [`apedp_dyn/sandbox/trading_engine.py`](../apedp_dyn/sandbox/trading_engine.py) | APEDP Trading Engine: Uses APEDP epistemic adjudication to detect | `audit-only` |
| [`apedp_dyn/sandbox/trading_signals_sandbox.py`](../apedp_dyn/sandbox/trading_signals_sandbox.py) | Trading-signals sandbox (22 quantitative signals, 6 categories): GA search whose every fitness evaluation is recorded in the candidate trial ledger, holdout evaluation, and the six-stage promotion lifecycle. Holdout Sharpe is deflated by... | `production` |
| [`apedp_dyn/sandbox/volume_sandbox.py`](../apedp_dyn/sandbox/volume_sandbox.py) | Volume-aware sandbox: daily volume-state experiments on sidecar data. | `experimental` |
| [`apedp_dyn/sandbox/walkforward_tournament.py`](../apedp_dyn/sandbox/walkforward_tournament.py) | Walk-forward APEDP trading tournament (prediction-first, §18/19/23). | `audit-only` |

### 18.2 RIME Stage-1 Engine (v0.1.1)

RIME searches small executable "mechanism graphs" (typed DAGs over inputs, constants, add/mul/neg, tanh/sign/abs, frozen lags $k \in \{1,2,5,10,20\}$, and a recursive `State` memory cell) and proposes the equations it finds to the referee. Spec: [`docs/RIME_V01_MASTER_SPEC.md`](RIME_V01_MASTER_SPEC.md).

**Boundary discipline:** RIME is a *proposer only*. It fits on development data, emits `CandidateProposal` objects carrying a closed-form equation plus a content-addressed `graph_id`, and cannot read or alter referee gates (`apedp_dyn/epistemic/`). Confirmation holdouts are untouched by the search.

**Stage-1 audit remediation (v0.1.1).** Four defects were confirmed by direct probes and fixed; each has a regression test that fails on the pre-fix code:

| Defect | Pre-fix behaviour | Fix | Regression test |
|---|---|---|---|
| Values cached per *node*, not per *(node, time step)* | `x[t] - x[t-1]` evaluated to zero | Whole-series evaluation in arena (topological) order, use-counted release | `tests/rime_regression.rs::difference_term_keeps_its_values` |
| `AddState` was not a recursion (`d*out[t-1] - out[t]`, wrong sign) | all-ones input, $d=0.5$ gave `[NaN, -0.5, -0.5, -0.5]` | `Node::State` with $h_t = d\,h_{t-1} + u_t$ | `tests/rime_regression.rs::add_state_is_a_recursive_decay` |
| Graphs scored on raw output; no coefficient fitting | planted $y = 0.001x_{t-2} + 5$ invisible to the scorer | Ridge least-squares readout $a\,o_t + b$, two readout DoF charged | `tests/rime_regression.rs::scale_and_intercept_are_fitted_before_scoring` |
| Null control used a 1.0-per-edge penalty (above every achievable MSE) | bundled test proved nothing | Penalty on the sampling-noise scale $\gamma\,\mathrm{var}(y)/n_{dev}$ plus EBIC selection charge $2\ln M$ | `tests/rime_regression.rs::null_world_false_positive_rate_is_controlled`, `short_series_null_control_pays_for_multiplicity` |

**Measured behaviour** (`benchmarks/rime_vs_neural_results.json`): on four planted mechanisms (fitted lag, momentum difference, AR(1) memory, lead-lag interaction) across $n \in \{60,\dots,3000\}$ with dev = first 60% and untouched test = last 40%, RIME recovers the exact family 5/5 seeds in every cell with out-of-sample $R^2 \ge 0.9996$ and two fitted coefficients; ridge on the same hand-built features ties it where the correct feature exists but fails the interaction (no product term) and the autoregressive world; the 961-weight MLP needs $n \ge 1200$ to reach ridge level and is worse than the mean on the null arm. On a pure-noise target RIME proposes nothing (0/5 at $n=150$ and $n=1200$), while the MLP reports strongly negative out-of-sample $R^2$. The `--stress` grid (18 cells) adds noise, amplitude and on/off-grid decay sweeps with a ≈0.9-correlated distractor input: RIME reports the right structure in 18/18 cells, including off-grid decay via its documented nearest-grid approximation. These are planted-mechanism results, not evidence of real-market validity.

**Multi-environment scoring (v0.1.2).** `RimeProposer(per_frame_envs=True)` scores each dev frame as its own environment under a frozen aggregation rule: `"mean"` (equal weight per environment — the literal §8 reading) or `"median_worst_gated"` (median of per-environment dev MSEs, plus a worst-environment improvement gate so a mechanism that only wins in a large environment is refused). The rule is fixed before scoring, reported in the proposal metadata, and covered by a dominance-trap regression test at both the Rust and Python layers. RIME's signal is **past-information-only** (temporal hygiene, explicitly not a causal claim; pinned by `test_rime_compute_score_documents_past_information_only`).

**Static-grammar extension & readout fix (v0.1.2, QEC round).** Two changes landed together with an honest correction of the earlier QEC attribution:
- `fit_readout`'s ridge was **scale-dependent** (fixed 1e-9 literal added to Σ(o−ō)²); on algebraic-scale inputs (a p² column at 1e-8) it exceeded the signal energy ~30× and zeroed every such fit. It is now relative to the regressor's raw second moment — identical for unit-scale data, correct at extreme scales. With this fix the temporal grammar recovers static monomial laws via its own self-interaction rewrite (`p·p`); the earlier "RIME's grammar is the wrong instrument for static scaling laws" conclusion is retired (the instrument that was wrong was the ridge).
- Additive grammar extensions, both OFF/unchanged by default: `RimeProposer(algebraic_transforms=["sq","cub","ssqrt","slog"])` appends transform columns (recomputed identically in `compute_score`), and static search mode (kernel config `"static"`) removes temporal primitives entirely for order-structured static data.
- **Routing rule:** RIME covers sequential mechanisms (lag/momentum/memory/interaction) and static monomial laws via self-interaction (plus the transform catalogue when enabled); static laws needing terms outside that set (exp, deep powers) route to the algebraic-basis proposers — see `benchmarks/matched_compute.py` for the measured per-family coverage table.

**Power calibration & additive refinement (v0.1.5, `experimental`, kernel level; updated v0.2.3 full proposer exposure).**
- `rust/apedp-kernel/tests/rime_power_calibration.rs` (`audit-only`): recovery / promotion rates with Wilson 95% intervals over independent replicates, scored *functionally* (the top proposal must correlate $\ge 0.95$ with the noise-free planted mechanism on a fresh independent series — no string matching). Measured with the proposer's own calibration (`noise_scaled`, $\gamma=1.5$, depth 2, 40 replicates/cell): single-term laws are recovered once $n \gtrsim 21/\rho^2$ (e.g. lag-2 at $\rho=0.10$: 2/40 at $n=1000$, 40/40 at $n=5000$), within a small factor of the Bonferroni-over-~200-candidates information limit ($\approx 19/\rho^2$ for 80% power); at $\rho=0.02$ nothing is recovered below $n=15000$ because the data do not support it, not because of a proposer bug. Null worlds: 1/120 searches promoted anything. These are synthetic i.i.d.-Gaussian results.
- `rust/apedp-kernel/src/rime/additive.rs` (`search_additive`): the forward-selected additive search engine. Runs only *after* the initial gate promotes a first term, then forward-selects further terms from the seed catalogue ∪ accepted proposals with a joint least-squares refit; each added term pays $\lambda_c \cdot (\text{edges} + 1\text{ coefficient} + 2 \ln M)$ for pool size $M$. The fitted model is emitted as a unified executable graph (`Const` · term sums). Measured against single-term search on identical worlds and seeds: promotion counts identical in every cell, 0/40 spurious extra terms on single-term laws, and multi-term recovery jumps from 0/40 to 40/40 for composite laws.
- **Native Proposer & Seed Budget Wiring (v0.2.3):** Exposed `additive_terms: int = 1`, `max_seeds: int | None = None`, and `lagged_products: bool = False` in `RimeProposer` and PyO3 entry points (`rime_search_json`, `rime_search_envs_json`). Solves Seed Catalog Exhaustion via configurable seed budget (`max_seeds`) and contemporaneous lagged interactions (`lagged_products=True`, Family 6b: $x_i[t-k] \cdot x_j[t-k]$); solves Grammar Constraint via native multi-term additive composition. Verified in `tests/test_dyn_rime_native_additive_and_seeds.py`: achieves holdout $R^2 \approx 0.9998$ on composite populations and matches GA-Large across all populations without post-hoc referee repair.
- **Limits (stated, not hidden):** terms must lie in the catalogue ∪ accepted proposals; the first term must itself clear the frozen gate (a law whose every single term is individually unpredictive is still unreachable); the `tanh(x[t-5])` row of the calibration table is non-discriminative (a linear lag-5 already correlates ≈0.96 with it for Gaussian inputs).

---

## 19. Traceability & Verification Matrix

| Documented Property / Guarantee | Implementation Location | Automated Test | Evidence & Status |
|---|---|---|---|
| Native Lean 4 Streaming Verification | [`apedp_dyn/epistemic/lean_universal_bridge.py`](../apedp_dyn/epistemic/lean_universal_bridge.py) | `tests/test_dyn_lean_universal_bridge.py` | Verified where an external Lean 4 toolchain is installed: exit code 0, sub-50 ms execution via `--stdin` (environment-gated). |
| Full-Lineage Experiment Tracking | [`apedp_dyn/epistemic/experiment_ledger.py`](../apedp_dyn/epistemic/experiment_ledger.py) | `tests/test_dyn_experiment_ledger_and_benchmarks.py` | Verified: Candidate parentage, features, params recorded. |
| Anti-Snooping Holdout Burning | [`apedp_dyn/epistemic/experiment_ledger.py`](../apedp_dyn/epistemic/experiment_ledger.py#L93) | `tests/test_dyn_experiment_ledger_and_benchmarks.py` | Verified: `HoldoutContaminationError` raised on reuse. |
| Deflated Sharpe Ratio Gating ($DSR \ge 0.95$) | [`apedp_dyn/epistemic/trial_ledger.py`](../apedp_dyn/epistemic/trial_ledger.py) | `tests/test_dyn_competing_proposers_and_dsr.py` | Verified: Penalizes trial multiplicity $N$ and skew/kurtosis; selection-adjusted calls charge disclosed proposer search pools (`selection_pool_size`/`search_id`). |
| Six-Stage Scoped Promotion State Machine | [`apedp_dyn/epistemic/promotion_lifecycle.py`](../apedp_dyn/epistemic/promotion_lifecycle.py) | `tests/test_dyn_experiment_ledger_and_benchmarks.py` | Verified: Stages 1-6 sequential, halts permanently on failure. |
| Decoupled DM Predictive Incrementality ($p < 0.01$) | [`apedp_dyn/epistemic/novelty_adjudication.py`](../apedp_dyn/epistemic/novelty_adjudication.py) | `tests/test_dyn_competing_proposers_and_dsr.py` | Verified: Diebold-Mariano test with HAC standard errors. |
| Structural Literature Orthogonality ($R^2 < 0.85$) | [`apedp_dyn/epistemic/novelty_adjudication.py`](../apedp_dyn/epistemic/novelty_adjudication.py#L150) | `tests/test_dyn_competing_proposers_and_dsr.py` | Verified: Rejects collinear Kyle/Amihud/Biais/Carhart proxies. |
| Stock-Split Scale Invariance ($P/k, kV$) | [`apedp_dyn/sandbox/canonical_ledger.py`](../apedp_dyn/sandbox/canonical_ledger.py#L330) | `tests/test_dyn_competing_proposers_and_dsr.py` | Verified: Conserves dollar volume, detects nominal bias. |
| Tactical Active Fraction Window $[5\%, 90\%]$ (5% to 90%) | [`apedp_dyn/sandbox/canonical_ledger.py`](../apedp_dyn/sandbox/canonical_ledger.py#L250) | `tests/test_dyn_competing_proposers_and_dsr.py` | Verified: Refuses Sharpe for decorative/non-binding gates. |
| RIME graph evaluation is time-consistent ($x[t]$ and $x[t-1]$ are distinct values) | [`rust/apedp-kernel/src/rime/graph.rs`](../rust/apedp-kernel/src/rime/graph.rs) | `tests/rime_regression.rs` | Verified: difference/momentum forms retain their magnitude. |
| RIME memory cell is a true recursion | [`rust/apedp-kernel/src/rime/rewrite.rs`](../rust/apedp-kernel/src/rime/rewrite.rs) | `tests/rime_regression.rs` | Verified: all-ones input at $d=0.5$ gives $[1, 1.5, 1.75, 1.875]$. |
| RIME scoring fits a readout and charges its degrees of freedom | [`rust/apedp-kernel/src/rime/eval.rs`](../rust/apedp-kernel/src/rime/eval.rs) | `tests/rime_regression.rs`, `tests/rime_stage1.rs` | Verified: planted amplitude recovered; raw scoring shown insufficient. |
| RIME abstains on null worlds (with a paired positive control) | [`rust/apedp-kernel/src/rime/search.rs`](../rust/apedp-kernel/src/rime/search.rs) | `tests/rime_regression.rs` | Verified: 0 promotions over 24 null worlds, and on 150-row null worlds. |
| RIME proposes through the standard proposer contract | [`apedp_dyn/proposers/rime_proposer.py`](../apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_proposer.py` | Verified: `CandidateProposal` with past-information-only `compute_score` (temporal, not causal); abstains on noise. |
| RIME multi-environment scoring cannot be outvoted by one large frame | [`rust/apedp-kernel/src/rime/search.rs`](../rust/apedp-kernel/src/rime/search.rs) | `pooled_dominance_trap_does_not_apply_to_envgate_rule` (Rust) + `test_rime_pooled_dev_scoring_cannot_outvote_a_failing_frame` (Python) | Verified: `MedianWorstGated` refuses a mechanism that wins only in the large env; the consistent-law control passes. |
| RIME Repair v0.1.0 diagnosis only (no equation modification) | [`apedp_dyn/epistemic/rime_repair_diagnosis.py`](../apedp_dyn/epistemic/rime_repair_diagnosis.py) | `tests/test_dyn_rime_repair_v010_diagnosis.py` | Verified: frozen `ResidualReport` (sha256-stable) + finding labels on real proposals; planted bias/AR-noise/curvature/heteroscedasticity detected; null-world abstention and insufficient-rows controls; cutoffs are explicit caller args (`experimental`). |
| RIME Repair v0.1.1–v0.1.4 bounded engine (R1/R2/R3 + 4-role validation) | [`apedp_dyn/epistemic/rime_repair_engine.py`](../apedp_dyn/epistemic/rime_repair_engine.py) | `tests/test_dyn_rime_repair_v011_engine.py` | Verified: planted intercept-shift→R1, missing-square→R2, emergent-lag→R3 each IMPROVED with sealed gain over frozen original; noise→no repair + sealed stays shut (`HoldoutContaminationError` on reopen); adequate original→no false repair; corrupted-fixture (shuffled targets)→NO_IMPROVEMENT, 0 evals, sealed shut; equivalence clustering preserves ambiguity; promotion deferred to APEDP-X track (`experimental`). |
| P0 risk-register dispositions (reviewer 25-finding triage) | tree-wide, measured | `tests/test_dyn_population_extrapolation_v2.py::test_depth_mechanism_budget_bound_not_ignored`, `::test_static_mode_emits_no_temporal_forms`, `tests/test_dyn_rime_known_truth*.py` (12 pass) | Verified: #2 depth NOT ignored — depth 1 evaluates 136 rows vs 200 cap at ≥2 with identical winners (budget-bound convergence); #3 static_mode=True emits 0 temporal forms on lag-planted truth; #9 `recovered_predictive_only` is a benchmark-local label with no promotion path (adjudicate owns verdicts); #10 scope-refusal regression suites green; #21 ridge fix verified real + unit-scale-neutral; REL_FLOOR=0.05 (`adjudicate.py:215`) confirmed NOT spec-pinned — flagged gap, no unilateral change. |
| Flaw-closure round: recall@budget, intervals, chaining, resources, rho sweep | `tests/test_dyn_population_extrapolation_v2.py::test_rime_recall_vs_budget_curve`, `::test_resource_curves_recorded`, `tests/test_dyn_rime_repair_v012_sensitivity.py::test_recall_strong_defects`, `tests/test_dyn_referee_rho_sensitivity.py` | All pinned | Verified: #5 product recall 0/5 budgets (penalty-, not budget-bound); #18 recall 18/18 with 95% lower bound 0.815, single-round termination pinned (`rounds_used==1`; multi-round chaining implemented but unreachable under single-targeted-finding veto semantics — documented extension point); #14 wall-time table recorded with hang-guard bounds; #7 whiteness operating point measured read-only (refuse@0.3/promote@0.5/0.7 on rho1_sqerr 0.413; frozen default untouched and verified equal to explicit 0.5). |
| #11 selective risk/coverage + #7 floor sweep | `tests/test_dyn_selective_risk_ga_rime.py`, `tests/test_dyn_referee_floor_sensitivity.py` | All pinned | Verified: post-hoc refusal policy (fit_R2>0.5 AND selection_R2>0, eval rows never touched) — GA/OLS attempt all signal worlds (eval R2>0.9), abstain 0/2 noise worlds; QEC thirds: OLS refuses flat low-p fit while RIME-repair upgrades state-form to square (sealed 0.987) — complementary honest behaviors. Floor sweep via explicit tau_epi: near-tie flips UNIQUE→MULTI across the bracket (monotone), decisive UNIQUE at 0.05+0.1, identical-predictions-distinct-structures MULTI at all floors; anchor artifact verified on disk. |
| Trading universe sweep (15/30/45/60/90 × daily/5d/21d) | `tests/test_dyn_trading_universe_sweep.py` | Pinned | Verified: all market files are daily bars (no minute data exists — 1/5/15/30/45/60-min frames untestable without inventing data); all 5 daily universes abstain on finite-checked frames; weak 5d/21d emissions (devR2≤0.12) collapse on holdout (R² −0.30/−0.77). Finite-row guard caught and fixed one vacuous loader path (bare-symbol crypto file). RIME on GA's exact 22-signal grammar: 0 proposals at depths 2–3/budgets 200–1000; penalty-off best devR2=0.0010 — absent signal, not missing grammar. Rename-invariance proves semantic blindness (identical graph ids/scores across meaningful vs nonsense names). |
| RIME feature-usage logging | [`apedp_dyn/proposers/rime_usage_log.py`](../apedp_dyn/proposers/rime_usage_log.py) | `tests/test_dyn_population_extrapolation_v2.py::test_usage_log_planted_composite_uses_only_x3`, `tests/test_dyn_trading_universe_sweep.py::test_usage_log_22_signals_mostly_untouched` | Verified: planted composite emissions reference x3 only (x1/x2 never enter); 22-signal penalty-off emissions touch 3/22 signals (flow/volatility family), 19 untouched. Syntactic presence (reachable nodes), not causal importance; rejected candidates invisible (kernel ships accepted only); zero proposals → zero log (`experimental`). |
| Search-funnel instrumentation + per-position coverage challenge | `rust/apedp-kernel/src/rime/search.rs`, `graph.rs` (`inputs_used()`), [`apedp_dyn/proposers/rime_proposer.py`](../apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_coverage_funnel.py` | Verified: additive-only `rejected_score`/`dropped_dedup`/`dropped_scope` counters + per-proposal `inputs_used` (no score/gate change; invariants `rejected+accepted==evaluated`, `evaluated<=budget` asserted). Coverage on 22-signal frame (planted lag-1 dependence per position): **17/22 exact**, 2/22 collinear-substitute (H-OFI→H-CVD, H-STOCH→H-WILLR — in-grammar twins, MULTI rationale), 3/22 **generation gap**: solo-frame control recovers the same dependence at R²>0.95 while the 22-input frame fails (R² 0.09–0.68) — the lag-1 seed is cut by the 60-form catalogue budget at 22 inputs. Null worlds 0/2. This separates "never generated" (truncation) from "generated but not selected" (`experimental`). GENERATION GAP SUBSEQUENTLY CLOSED — see the seed-order row. |
| Seed-order fix for the measured generation gap (opt-in `seed_order="coverage_first"`) | [`rust/apedp-kernel/src/rime/seeds.rs`](../rust/apedp-kernel/src/rime/seeds.rs), `SearchConfig.seed_order`, `RimeProposer(seed_order=...)` | `tests/test_dyn_rime_seed_order_coverage.py` + 3 Rust catalogue tests | Verified: lag-major family-2 keeps every input's shortest lag inside the same 60-form budget. Gap positions H-RV / H-A / H-OBV_MOM now recover the planted column at solo-frame R² (0.9937 / 0.9995 / 0.9597 vs legacy 0.19 / 0.68 / 0.09); all 22 positions name the planted column (only the 2 collinear substitutes remain, a non-identifiability property). Null-world FPR 0/50 unchanged; catalogues identical to legacy whenever the budget holds (Rust-pinned n≤6). Default stays `legacy` — no frozen-spec change (`experimental`). |
| Contemporaneous powers & self-products (opt-in `square_seeds=True`) | [`rust/apedp-kernel/src/rime/seeds.rs`](../rust/apedp-kernel/src/rime/seeds.rs), `SearchConfig.square_seeds`, [`apedp_dyn/proposers/rime_proposer.py`](../apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_reachability_and_contemporaneous_qec.py` | Verified: lag-0 self-multiplication (x_i * x_i) admitted into seed catalogue family 5 at depth 0. Closes reachability flaw on QEC: raw [p] discovers p_L = 1000*p^2 directly at depth 0 without requiring manual feature engineering (p_sq) or repair engine intervention. Holdout R² > 0.9995; verified 100% reversal-invariant (|R2_fwd - R2_rev| < 1e-12), whereas legacy lagged proxy (p * p[t-5]) inflates MSE by >10x under row reversal. Systematic residual curvature screen flags d=5 wrong-family (p^2 fit on p^3 truth) while clearing d=3 true law. Default stays `false` (`experimental`). |
| Intraday crypto universe sweep (fetched minute bars) | `tools/fetch_intraday_crypto.py`, `data/market/intraday/` | `tests/test_dyn_intraday_universe_sweep.py` | Verified on genuine Binance klines (1000 bars/symbol/interval; frozen daily corpus untouched): real basket forward returns abstain in **15/15** universe(14/26/39)×interval(1m/5m/15m/30m/1h) cells under both seed orders; null worlds 0 proposals; planted lag-1 control recovered at dev R²>0.98 at every interval; per-position coverage follows the 60-form arithmetic exactly (39 levels + 21 lag-1 → inputs 0–20 recovered under `coverage_first`, 21+ still gaps). Scope recorded: 45m is not an exchange interval; equity legs unfetchable (Yahoo 429) so 60/90-asset intraday universes out of scope; equal bar counts confound interval with calendar span (`experimental`). |
| REAL-data repair benchmark (AME2020 masses, no planting) | `tests/test_dyn_real_data_ame_repair.py` | Pinned, 6 tests | Verified on genuinely measured data: discovery recovers the SEMF **volume term** (`+7426.14*A+93729.3`, dev R² 0.994); diagnosis fires `curvature_suspect` (missing surface term A^2/3); bounded repair at the PREREGISTERED absolute bar 0.02 → `NO_IMPROVEMENT` (bar mis-scaled for a 0.994 baseline); at a scale-appropriate bar → `IMPROVED`, sealed R² **0.9995** vs 0.9947. Grammar-extension experiment (declared `rational_powers`, default off): with the family admitted the validation ranking is proxy (A·Z) 0.99960 > (A·A) 0.99930 > **surface term (A^0.667) 0.99915** > (A^0.333) 0.99898; on the sealed split proxy and mechanism differ by <1e-3 R², so with the identifiability band declared the engine returns `AMBIGUOUS` (set size 4) — predictive success reported separately from identification. Preregistered criteria: `min_sse_reduction` (relative SSE reduction) and `equivalence_r2_tol`, both default-off/frozen. SEMF literature coefficients reach R² 0.9999 on the same frame; nothing is promoted (`experimental`). |
| RIME Repair sensitivity/recall milestone (v0.1.2–v0.1.3) | [`apedp_dyn/epistemic/rime_repair_engine.py`](../apedp_dyn/epistemic/rime_repair_engine.py) | `tests/test_dyn_rime_repair_v012_sensitivity.py`, `tests/test_dyn_rime_repair_v013_realdata.py` | Verified: recall 9/9 strong defects with structure recovery (right operator/variable); weak-defect curve never harms sealed outcome; BTC panel abstention regression (repair-R²=−0.0462, sealed shut); 5/5 walk-forward windows abstain with persistence RMSE reported; chance predictive pass blocked at stability gate (`experimental`). |
| Model-class closure: interaction probes + lagged-product R2 terms | [`apedp_dyn/epistemic/rime_repair_diagnosis.py`](../apedp_dyn/epistemic/rime_repair_diagnosis.py), [`apedp_dyn/epistemic/rime_repair_engine.py`](../apedp_dyn/epistemic/rime_repair_engine.py) | `tests/test_dyn_rime_repair_v011_engine.py::test_repair_composite_product_completed` | Verified: omitted interactions of independent drivers (white residuals) fire `interaction_suspect` via probe family mirrored 1:1 in the R2 catalogue (lag choices cite frozen LAG_SET's two shortest; no new configured values); planted composite completes to sealed R²>0.9 with the lagged product term. QEC-conservative behavior preserved (3 refused R2 rows incl. R² 0.97). No frozen-spec change: probe/term families live in the experimental engine, cutoffs stay explicit caller args (`experimental`). |
| Staged search + coverage/budget curves (grammar-size auditing) | [`apedp_dyn/epistemic/rime_staged_search.py`](../apedp_dyn/epistemic/rime_staged_search.py) | `tests/test_dyn_rime_staged_search.py` | Verified: two-stage search (cheap family-admit pass, then full-budget expansion) that REPORTS coverage instead of asserting it. Measured budget-curve: recovery saturates at 176 evaluated with identical best R² 0.9616 across budgets 96→2000, and reached families DROP 3→2 — depth/budget claims beyond the cap are budget-bound and are pinned as such. Expanded grammar (all ops ON) spends more budget (200 vs 176) for the same winning family and same score: expansion buys reachability, not recall. Staged run is never worse than the straight full-budget run. Structured-null FPR unchanged with all ops ON (proposer 4/12 worlds, system 0) (`experimental`). |
| Noisy multi-dataset repair suite (real measured targets, 2% noise) | `tests/test_dyn_noisy_multidataset_repair.py` | Pinned | Verified on three real AME2020 target families (volume/surface, isobaric parabola, symmetry residual), each with 2% observation noise: D1 IMPROVED with sealed evidence; D2 INSUFFICIENT_EVIDENCE (92 rows, splits below the diagnosis floor); D3 NO_IMPROVEMENT. True no-repair baseline (zero repair operators) never reports a fix. Row-floor limitation recorded: Bertozzi (5 runs), Brown lifter (20 rows), warp-energy records cannot host the 4-role battery without data acquisition (`experimental`). |
| Structural invariance ranking (identification by cross-slice agreement) | [`apedp_dyn/epistemic/structural_invariance.py`](../apedp_dyn/epistemic/structural_invariance.py) | `tests/test_dyn_structural_invariance.py` | Verified: uniqueness is earned by the SAME canonical structure winning >= 4/5 independent dev slices rather than by a thin margin in one pooled fit — the measured failure mode where a correlated proxy (A*Z) outranked the correct term (A^0.667) by ~3e-4. Key is the decomposed variable multiset with the fitted readout excluded and additive-zero decorations stripped (measured: without that, PruneEdge syntactic twins fragmented agreement to 2/5). Negative control: A-only wins 5/5 with the VOLUME term only, surface term absent — agreement does not repair a grammar that cannot express the truth. Rotating winner -> MULTI; empty search -> NONE (`experimental`). | [`apedp_dyn/epistemic/whiteness_calibration.py`](../apedp_dyn/epistemic/whiteness_calibration.py) | `tests/test_dyn_whiteness_flag_calibration.py` | Verified: derives Guard 2's \|rho1\| flag from the target's own serial dependence by block-demeaning instead of a fixed 0.5. Measured both fix directions: on an AR(2) target a residual with \|rho1\|=0.723 is refused by the frozen 0.5 and accepted by the calibrated 0.858 (over-refusal fixed); on a white target a residual with \|rho1\|=0.273 is admitted by the frozen 0.5 and refused by the calibrated 0.226 (under-refusal fixed). Flag is monotone in target persistence (0.114 white → 0.858 AR(2)). Ships `screen_structured_null_worlds`, which reports proposer-level and system-level FPR separately; on 24 structured null worlds: 12 proposer promotions, 0 system promotions. Lives in its own module because `adjudicate.py` is a hash-pinned frozen record (`experimental`). |
| KNOWN LIMITATION — same-driver omitted lags vs rho1 screens | `tests/test_dyn_rime_repair_v012_sensitivity.py::test_same_driver_omitted_lag_sensitivity_map` | Sensitivity map (6 corners) | **Documented boundary**: single-regressor OLS residuals on an AR driver are usually absorbed white (4/6 corners silent). Firing corners (state-cell misfit remainders) yield gated rejection, not false wins. R3's demonstrated domain is emergent lag from an independent driver. Whiteness alone is insufficient for structural identification. |
| KNOWN LIMITATION — whiteness gate under geometric-spacing extrapolation (QEC) | `tests/test_dyn_rime_repair_v014_qec.py::test_qec_degraded_linear_repair_refused_despite_high_r2` | Pinned conservative behavior | **Documented boundary**: on log-spaced p, quadratic extrapolation misfit grows smoothly into validation rows and trips the temporal screen, vetoing an R²=0.95 repair. Refusal is the safe failure direction; gate assumes stationarity across evaluation rows (valid for time series). No gate respecification without its own preregistered benchmark. |
| QEC thirds decisive comparison (GA vs RIME-orig vs RIME-repaired vs OLS) | `tests/test_dyn_rime_repair_v014_qec.py::test_qec_thirds_same_split_ga_vs_rime_repaired` | Pinned, same eval rows, no refits | Verified: RIME-repaired sealed R² 0.9869 wins; OLS-quad 0.8733 (its policy refusal was over-conservative — recorded limitation of the post-hoc rule); RIME-orig −1.5959; GA starved to intercept-only (−2.15) on the flat fit third. Better repair outcome on this test: yes. Better-than-GA generally / transferable law: NOT established. |
| Population extrapolation RIME vs GA (depths, feature space, repair-mediated) | `tests/test_dyn_population_extrapolation_rime_ga.py`, `tests/test_dyn_population_extrapolation_v2.py` | Pinned grids | Verified: RIME depths 1–6 identical (x3[t-2] partial); +sq transforms change nothing; withheld-feature abstains all depths; GA ladder isolates products as the sole gap (squares add nothing); GA-small ties RIME (<0.01 all pops); RIME-large (23 engineered cols) ties RIME-raw (truncation + no-Add reason measured); repair-mediated RIME matches GA-large incl. transfer (sealed/B 0.9998, C 0.8952) (`experimental`). |
| GA on QEC vs OLS vs RIME (method boundary) | `tests/test_dyn_ga_qec.py` | Pinned three-way + incident note | Verified: GA selects {1, p²}, holdout RMSE 0.002220 = same-split OLS (C/p_th²≈1000 recovered); RIME raw-[p] deep search recovers via seeded self-interactions (holdout R² 0.9995, stable); shallow-config single divergent run (lag winner, 1 of ~14) explained by a concurrent kernel rebuild landing mid-measurement (absolute- vs relative-ridge binaries), not search nondeterminism. A concurrent session's scale-invariant ridge fix (`eval.rs`: relative `ridge·Σo²`, commit 9781a3d) was verified real and behavior-neutral on unit-scale tests; an unattributed edit to this row/test asserting it was first reverted, then re-verified and corrected here. Coordination via separate worktrees required going forward. |
| Frozen-equation OOS (QEC Eq1 vs Eq2, no refit) | `tests/test_dyn_rime_qec_oos_frozen.py`, `tests/test_dyn_rime_qec_repaired_transfer.py` | Pinned comparison | Verified: T1 fresh seeds R²>0.999 both; Eq1≡Eq2 to ~2e-5 on log grids (one hypothesis, not two); uniform grid R²>0.99 with doubled MAE; d=5 graceful failure matching frozen OLS-quad; mean baseline negative. Repaired equation (frozen): R² 0.990–0.995 on fresh/uniform/irregular/extended-range/3x-noise d=3 grids, always above frozen OLS-quad; monotone decay d=4→7 (0.953/0.817/0.653/0.497) with OLS-quad marginally above throughout — no transferable law, ordered degradation; square coef 1071 vs truth 1000 (useful structure, not the mechanism). |
| Real-Market Trading Predictive Profitability | N/A (Candidate evaluations in `sandbox/cpmd_*.py`) | None (Empirical holdouts failed) | **Explicitly Unverified / Unproven**: H-LIK and H-XSC failed OOS confirmation. |
| Rational operations & Padé saturation (Rust SafeDiv, Epistemic Repair R5, and Normalized Transforms) | [`rust/apedp-kernel/src/rime/graph.rs`](../rust/apedp-kernel/src/rime/graph.rs), [`apedp_dyn/epistemic/rime_repair_engine.py`](../apedp_dyn/epistemic/rime_repair_engine.py), [`apedp_dyn/proposers/rime_proposer.py`](../apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_rational_and_pade.py` | Verified: Option 1: Native Rust `safediv(x, y) = x / (1 + |y|)` enables rational division in symbolic mechanism graph ($R^2 = 1.0000$ on rational target $y = x/(1+x)$, opt-in `rational_ops=True`); Option 2: Epistemic repair engine operator `R5_PADE_SATURATION` fits non-linear Padé saturation $g(z) = a \cdot z / (1 + b \cdot |z|)$ on repair partition, cutting MSE by >93% and reaching $R^2 = 1.0000$ on sealed holdout; Option 3: Normalized algebraic transforms (`sat_norm`, `sat_sq_norm`) scale inputs by domain standard deviation to recover saturating threshold laws ($R^2 > 0.97$ vs un-saturated $p^2$ $R^2 = 0.78$). Zero regressions; defaults off (`experimental`). |
| QEC d=5 cubic discovery & saturating generators | [`apedp_dyn/sandbox/qec_threshold.py`](../apedp_dyn/sandbox/qec_threshold.py), [`tests/test_dyn_rime_qec_d5_and_saturation_pins.py`](../tests/test_dyn_rime_qec_d5_and_saturation_pins.py) | `tests/test_dyn_rime_qec_d5_and_saturation_pins.py` | Verified: Domain-spanning split discovers true cubic `(p * (p * p))` at depth 1 via `AddInteraction` with holdout $R^2 > 0.999$ and scale $\approx 99,863.1$ (true $C/p_{\text{th}}^3 = 100,000$), while recovering quadratic proxy `(p * p)` at dev $R^2 \approx 0.9626$; chronological 300/300 split on low-SNR subrange ($\text{SNR} \approx 0.31$) yields linear fit holdout $R^2 \approx -0.435$, where complexity gating correctly abstains; systematic curvature screen flags wrong-family `(p * p)` fit on $d=5$ observations; canonical Padé and Exponential saturating data generators pinned on disk (`experimental`). |
| STEM Grammar Expansion (Native Primitives, Multi-Engine Proposers & Dimensional Homogeneity) | [`rust/apedp-kernel/src/rime/graph.rs`](../rust/apedp-kernel/src/rime/graph.rs), [`apedp_dyn/proposers/stem_proposers.py`](../apedp_dyn/proposers/stem_proposers.py), [`apedp_dyn/epistemic/dimensional_homogeneity.py`](../apedp_dyn/epistemic/dimensional_homogeneity.py) | `tests/test_dyn_stem_grammar_options_123.py` | Verified: Option 1: Native Rust graph nodes (`Sin`, `Cos`, `SafeExp`, `SafeLog`, `Diff`) and rewrite ops (`stem_ops=True`); Option 2: Multi-engine proposers (`HamiltonianProposer`, `SolenoidalFluidProposer`, `CompartmentalSirProposer`) enforcing symplecticity, divergence-free constraints, and stoichiometric conservation; Option 3: Dimensional homogeneity referee gate validating SI unit consistency via Buckingham-$\Pi$ algebra before hypothesis promotion (`experimental`). |
| Preregistered Real-Data Benchmark Suite | `data/mass_1.mas20.txt`, `data/bertozzi_1964.jsonl`, `data/market/` | `tests/test_dyn_preregistered_real_data_benchmark.py` | Verified on genuine real-world measurements: (1) AME2020 nuclear binding energies (1,159 nuclides): RIME discovers Coulomb term $Z^2$ and $A \cdot Z$ with holdout $R^2 > 0.70$ and RMSE $< 0.20$ MeV/nucleon on 559 unseen nuclides, compared against literature Weizsäcker SEMF ($R^2 = 0.9921$); (2) Bertozzi 1964 relativistic electron accelerator: classical mechanics rejected catastrophically (holdout RMSE $\approx 4.88c$), relativistic mechanics promoted with `UNIQUE_SUPPORT` (holdout RMSE $\approx 0.0055c$); (3) Real market orderflow and asset returns: zero false promotions ($\text{FPR} = 0.00\%$), 100% abstention on pure noise (`experimental`). |
| RIME v0.2 Structural Flaw Remediation | [`rust/apedp-kernel/src/rime/graph.rs`](../rust/apedp-kernel/src/rime/graph.rs), [`rust/apedp-kernel/src/rime/search.rs`](../rust/apedp-kernel/src/rime/search.rs), [`apedp_dyn/proposers/rime_proposer.py`](../apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_v02_flaw_remediation.py` | Verified: (1) Native Rational Powers (`power_ops=True`): admits $x^{2/3}$, $x^{1/3}$, $x^{1/2}$, $x^{-1/3}$ into search, recovering nuclear surface scaling $A^{2/3}$ with exact $R^2 = 1.000000$ and zero error; (2) Small-Sample Laboratory Gating (`min_rows=5`): resolves small-sample paralysis on laboratory experimental data ($N=5$) while preserving frozen $N \ge 30$ default for noisy time-series; (3) Correlated Proxy Trap Remediation (`identifiability_band=0.002` & `check_derivatives=True`): prevents false-certainty promotion on collinear hypotheses by reporting `AMBIGUOUS_EQUIVALENCE_SET` and verifying gradient alignment; (4) Diverse Beam Frontier (`diverse_beam=True`): guarantees round-robin representation across structural archetypes to eliminate greedy beam monoculture (`experimental`). |
| Rational Parameter Identifiability & Decoupled 5-Metric Reporting | [`apedp_dyn/epistemic/rime_repair_engine.py`](../apedp_dyn/epistemic/rime_repair_engine.py) | `tests/test_dyn_rime_rational_parameter_identifiability.py` | Verified: (1) Parameter reconciliation: explicit mapping of physical scales ($c_{\text{num}} = a \cdot k_{\text{fit}} = 1000.0$, $c_{\text{den}} = b \cdot k_{\text{fit}} = 2000.0$) and saturation envelope ratio ($\alpha = b/a = 2.0000$); (2) Noise robustness: verified across $\sigma \in [0.5\%, 2\%, 5\%]$, with parameter errors bounded within 0.43% to 4.29%; (3) Domain range sensitivity: proves parameter identifiability breakdown on narrow intervals ($p \in [0.001, 0.008]$) where $R^2 > 0.999$ coexists with 6.5% parameter error, correctly triggering `PREDICTIVE_ONLY_ILL_CONDITIONED`; (4) Negative controls: verified against pure quadratic ($b \to 0$) and exponential saturation; (5) Decoupled 5-metric reporting: independently emits `structure_match`, `parameter_recovery`, `prediction_score`, `extrapolation_score`, and `identifiability_status` (`experimental`). |
| RIME Native Resolution of Seed Catalog Exhaustion & Grammar Constraint | [`rust/apedp-kernel/src/rime/seeds.rs`](../rust/apedp-kernel/src/rime/seeds.rs), [`rust/apedp-kernel/src/rime/search.rs`](../rust/apedp-kernel/src/rime/search.rs), [`apedp_dyn/proposers/rime_proposer.py`](../apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_native_additive_and_seeds.py` | Verified: (1) Seed Catalog Exhaustion resolved via `max_seeds` and `lagged_products=True` admitting contemporaneous lagged interactions $x_i[t-k] \cdot x_j[t-k]$ into Family 6b; (2) Grammar Constraint resolved via native `additive_terms >= 2` wired to `search_additive` with joint refit and honest multiplicity/degree-of-freedom penalization; (3) Composite population extrapolation ($y = 2 x_1[t-1] x_2[t-1] + 3 x_3[t-2] + 10$) natively recovered with true parameters; (4) Matches GA-Large across Pop A ($R^2 \approx 0.9998$), Pop B ($R^2 \approx 0.9998$), and Pop C shift ($R^2 \approx 0.896$) without referee repair; (5) Distractor resilience filtering 7 noise features ($R^2 > 0.9999$) and parsimonious single-term retention on null/single-term worlds (`experimental`). |
| Closed-loop prior-guided search (profiler -> search adapter, unsolicited recovery) | [`apedp_dyn/epistemic/prior_guided_search.py`](../apedp_dyn/epistemic/prior_guided_search.py) | `tests/test_dyn_prior_guided_unsolicited.py` | Verified (7/7, fixed seeds): consumes the shape prior + dataset profile on dev rows only and selects a search path through a fixed table (stepped -> dev-exact sign basis + extrapolatable rule form; singular/power -> transmutation fitter; else abstain with reason). Emits `CandidateProposal` objects; never promotes, never sees holdout. Unsolicited Landauer recovery on sealed splits (dev [0.1,4.9] -> holdout [5.0,9.8]): holdout R2 == 1.0 with zero human primitive choice; the module contains no quoted operator tokens usable for per-equation selection (machine-guarded test). Owns a deterministic two-stage floor-rule fit with max-margin tie-break, which was REQUIRED by measurement: several (w, off) classify dev rows identically (SSE 1.5e-29 tie) while holdout separates them (0.974 vs 1.0). Noise rung 1 (2% staircase): sealed holdout >= 0.999. Rung 2 labeled semi-synthetic (BTC 5m returns + injected W=125/OFF=40 staircase at local SNR=5): width 125.02 vs 125, quantum within 0.4%, offset modulo width, holdout 0.8910 vs oracle 0.8917 (noise ceiling); same code abstains on the clean stream; low-SNR boundary (SNR<=4 abstains) pinned. Recorded integration gap: the transmutation fitter declines stepped data under noise that the shape prior identifies (level-clustering cap), so row 1 uses the adapter's uniform floor-fit path. Rung 2 is semi-synthetic by necessity (no real known-truth dataset >= 40 rows in repo); unsolicited discovery on real unknowns remains unclaimed (`experimental`). |
| Report fidelity (verdicts derive from measured rows) | [`apedp_dyn/epistemic/report_fidelity.py`](../apedp_dyn/epistemic/report_fidelity.py) | `tests/test_dyn_landauer_report_fidelity.py` | Verified (4/4): `verdict_for_row` with explicit caller args (tau_extrapolation, rho_max, eps_exact_rmse, delta_exact_r2); exact-recovery bypass checked first so rho1 of numerical noise cannot refuse R2 == 1. The Landauer harness template renders this function's output verbatim (4-line + import edit; NaN rho1 renders `n/a (exact-zero residuals)`, not `0.0000`); hardcoded PASSED/REFUSED labels removed. Committed md == committed JSON cell-for-cell on the main table (machine-asserted); a no-hardcode guard scans the harness source. Robustness-table numerics verified by harness re-execution (all digits reproduce), not by the fidelity test, which covers the main table (`experimental`). |
| Data-driven shape inference (data-driven family prior + step basis, steps 1-2 of unassisted stepped discovery) | [`apedp_dyn/epistemic/shape_inference.py`](../apedp_dyn/epistemic/shape_inference.py) | `tests/test_dyn_shape_inference.py` | Verified (15/15, fixed seeds): infers which structural family the data supports, with change-points estimated from the data rather than declared. Method: exact O(k·n²) DP piecewise-constant and piecewise-linear segmentations; the family verdict is how many PARAMETERS each family needs to reach a common fit target (an achievable×(1−sse_target) R², both families costed per parameter), a shared-square-root-of-zero-free statistic. `fit_step_family` emits a candidate `c₀ + Σ cᵢ·sign(x − tᵢ)` form with data-estimated thresholds and OLS coefficients — needing NO new operator node (`Sign`/`Add`/`Const`/`Mul` already exist). Measured: Landauer staircase recovered EXACTLY (R² 1.0, coefficients 2.5 and 0.5×5 matching the true generating form, 5 change-points within half the local bin gap of truth 1..5), still identified at 2% noise (R² 0.9997) and through 10% noise. Negative controls, all correct: straight line → `smooth` with 0 change-points and NO seeds emitted; saturating Padé → `smooth`, 0; iid noise → `smooth`; BTC forward returns (1094 rows) → `no_structure` (pc R² 0.017 below the caller floor) — no fabricated steps on real market data. Step count is data-estimated (2-step → 2, 5-step → 5). <40 rows → `no_structure`. FOUR statistics were implemented and REJECTED by measurement and are recorded in CHANGELOG.md: "piecewise-constant beats a line" (flags noise and a saturating curve as stepped, only +0.027 on the real staircase); flat-run fraction on the p90 |Δy| scale (collapses 0.975→0.065 under noise); within-segment drift ratio (mislabeling a coarse segmentation of a CURVE as plateaus); and separately-BIC-selected k per family (degenerating because enough constant segments drive SSE to exactly 0). Critical gate-dependence control: `fit_step_family` on a STRAIGHT LINE with forced thresholds returns R² 0.979 and `identifiable=True`, so a high R² from this basis is NOT admissible evidence of steps — only the independent prior's verdict is, which is why `staircase_prior` refuses to emit seeds on a line. LIMITATIONS (stated, not hidden): the step basis is not yet fed to the kernel search, so multi-term composition is unbuilt and no unsolicited end-to-end discovery is claimed; at 20% noise the staircase no longer classifies within `max_segments=32`, a cap limit, recorded rather than tuned away (`experimental`). |
| Landauer Conductance Staircase: Reproducible RIME vs. GA Benchmark | [`benchmarks/run_landauer_rime_vs_ga_benchmark.py`](../benchmarks/run_landauer_rime_vs_ga_benchmark.py), [`docs/landauer_conductance_staircase_rime_vs_ga_benchmark.md`](../docs/landauer_conductance_staircase_rime_vs_ga_benchmark.md) | `benchmarks/run_landauer_rime_vs_ga_benchmark.py` | Verified from reproducible execution on disk: (1) GA polynomial (degree 5) achieves in-sample $R^2 = 0.9645$ but catastrophically diverges out-of-sample ($R^2 = -11530.5976$, RMSE = 149.32, $\kappa = 4.54 \times 10^4$), refused by Epistemic Referee whiteness ($|\rho_1| = 0.7238 > 0.50$); (2) Unassisted RIME finds smooth linear surrogate ($R^2_{\text{train}} = 0.9598$, $R^2_{\text{extrap}} = 0.9486$), refused by referee whiteness ($|\rho_1| = 0.7386 > 0.50$); (3) Assisted RIME with `floor` recovers exact quantum conductance $G_0 = 1.000000$ ($R^2_{\text{extrap}} = 1.000000$); (4) Autonomous Epistemic Profiler & Transmuter detects discrete plateau topology ($84.8\%$ flat derivatives) and recovers $G_0 = 1.000000$ in $0.35$ ms; (5) Machine-measured across clean, thermal broadening ($k_B T / \Delta E = 0.08$), 2% noise ($R^2_{\text{extrap}} = 0.999991$), and 5% noise ($R^2_{\text{extrap}} = 1.000000$). Results logged to `benchmarks/landauer_benchmark_results.json` (`experimental`). |
| Quantum Equation Discoveries & RIME Physics Remediations | [`rust/apedp-kernel/src/rime/seeds.rs`](../rust/apedp-kernel/src/rime/seeds.rs), [`apedp_dyn/epistemic/rime_repair_engine.py`](../apedp_dyn/epistemic/rime_repair_engine.py), [`apedp_dyn/proposers/rime_proposer.py`](../apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_quantum_equation_fixes.py` | Verified: (1) Limitation 1 (Negative Powers): Brus quantum dot confinement ($E_g(R) = 1.74 + 2.45 R^{-2} - 0.28 R^{-1}$) natively discovered via power_ops ($x^{-1}, x^{-2}$) and transforms (`inv`, `inv_sq`), raising extrapolation $R^2$ from $-70.27$ to $+0.9959$; (2) Limitation 2 (Continuous Denominator Scale & Lorentzian Resonance): R5_PADE_SATURATION expanded to Form 2 resonance $a / (1 + b |z|)$, recovering true $\Omega_R^2 = 25$ on Detuned Rabi with $R^2 = 1.000000$ (clean) and $R^2 = 0.999968$ (2% noise, 0.76% parameter error, `IDENTIFIED_STABLE`); (3) Limitation 3 (Harmonic Frequency Multipliers): `harmonic_seeds=True` emits $\cos(t+t)=\cos(2t)$ at Depth 0, discovering resonant Rabi dynamics ($P_e(t) = \frac{1-\cos(2t)}{2}$) with holdout $R^2 = 1.000000$ (`experimental`). |
| Autonomous Epistemic Profiler & Coordinate Transmutation | [`apedp_dyn/epistemic/autonomous_profiler.py`](../apedp_dyn/epistemic/autonomous_profiler.py), [`apedp_dyn/epistemic/coordinate_transmuter.py`](../apedp_dyn/epistemic/coordinate_transmuter.py) | `tests/test_dyn_autonomous_epistemic_profiler.py` | Verified: (1) Autonomous Epistemic Profiler diagnoses topological invariants: dynamic range (>100 -> log_mse), Arrhenius tunneling (d ln y / d(1/x) = const), power-law scaling (d ln y / d ln x = const), quantized plateaus (flat fraction > 40%), and dual-asymptotic boundary crossovers (Rayleigh-Jeans vs Wien); (2) Coordinate Transmuter linearizes non-linear manifolds: exact non-perturbative recovery of Schwinger tunneling ($10 E^2 e^{-1/E}$, $R^2 = 1.000000$), Brus confinement ($4.5 R^{-2}$, $R^2 = 1.000000$), Landauer staircase ($2 \lfloor w \rfloor$, $R^2 = 1.000000$), Landau-Zener non-adiabatic transition ($P = \exp(-0.75/v)$, $R^2 = 1.000000$), BCS superconducting energy gap ($\Delta = 3.52 \exp(-1/g)$, $R^2 = 1.000000$), and Integer Quantum Hall Effect staircase ($\sigma_{xy} = \lfloor 1/B \rfloor$, $R^2 = 1.000000$) (`experimental`). |
| RIME-AD Native Rust Engines & Unified Quantum Benchmark | [`rust/apedp-kernel/src/rime/total_variation.rs`](../rust/apedp-kernel/src/rime/total_variation.rs), [`rust/apedp-kernel/src/rime/implicit.rs`](../rust/apedp-kernel/src/rime/implicit.rs), [`rust/apedp-kernel/src/rime/diff_pade.rs`](../rust/apedp-kernel/src/rime/diff_pade.rs), [`rust/apedp-kernel/src/rime/enumerate.rs`](../rust/apedp-kernel/src/rime/enumerate.rs) | `tests/test_dyn_rust_all_equations_benchmark.py` | Verified (6/6 tests, 9 physical equations, zero hardcoded verdicts): (1) BV-ROF Total Variation denoising (`total_variation.rs`) recovers clean Landauer ($R^2_{\text{extrap}} = 1.000000$) and IQHE ($R^2_{\text{extrap}} = 0.990926$); (2) Sparse Implicit SVD (`implicit.rs`) solves Detuned Rabi Padé resonance ($1 - y - 0.04 x^2 y = 0$) with holdout $R^2 = 1.000000$ and RMSE $< 10^{-11}$; (3) Differential Padé & Jet Algebra (`diff_pade.rs`) discovers Schwinger, Landau-Zener, and BCS ODEs; Landau-Zener passes exact recovery ($R^2 = 1.000000, \text{RMSE} = 1.22 \times 10^{-9}$), while Schwinger and BCS are honestly refused by the referee's whiteness gate due to smooth numerical RK4 integration drift ($|\rho_1| \approx 0.925 \ge 0.5$); (4) Planck Radiance is honestly recorded as a structural failure ($R^2_{\text{dev}} = -1.27$, holdout collapse, REFUSED) because linear Jet ODEs cannot represent non-linear Riccati crossovers; (5) Brus Confinement recovers exact multi-rational powers ($1.74 - 0.28 R^{-1} + 2.45 R^{-2}$) with holdout $R^2 = 1.000000$. Full machine results logged in `benchmarks/quantum_all_equations_benchmark_results.json` and `docs/quantum_all_equations_benchmark_results.md` (`experimental`). |
| Multi-Population Extrapolation & Distractor Challenge: RIME-AD vs GA/GP | [`benchmarks/run_population_extrapolation_rime_ad_vs_ga.py`](../benchmarks/run_population_extrapolation_rime_ad_vs_ga.py), [`docs/population_extrapolation_rime_ad_vs_ga.md`](../docs/population_extrapolation_rime_ad_vs_ga.md) | `tests/test_dyn_population_extrapolation_benchmark.py` | Verified (3/3 tests, across Pop A-train, Pop A-test, Pop B, Pop C, Pop D, and 7-noise distractor challenge): (1) In-sample & holdout parity: RIME-AD matches GA-Large ($R^2_{\text{test}} = 0.9998$ vs $0.9998$) while GA-Small collapses ($R^2_{\text{test}} = 0.6450$) due to missing nonlinear interactions; (2) Parameter shift transfer: $c_1 = 2.0 \to 2.5, c_2 = 3.0 \to 2.0$ transfers gracefully ($R^2 = 0.8961$); (3) Heavy-tail Laplace OOD: RIME-AD achieves $R^2 = 1.0000$ (0.999980); (4) Exact formula recovery: $+1 \cdot (2.9956 \cdot x_3[t-2] + 1.9966 \cdot (x_1[t-1] x_2[t-1])) + 9.9988$; (5) Distractor noise immunity: zero false positives out of 7 distractor channels (0% false discovery); (6) Execution wall-time: RIME-AD runs in 8.7 ms vs 2137.7 ms for GA-Large (~246x speedup) (`experimental`). |
| Koopman Spectral Discovery & Lie Point Symmetries (Rust specialists + GA comparison) | [`rust/apedp-kernel/src/rime/koopman.rs`](../rust/apedp-kernel/src/rime/koopman.rs), [`rust/apedp-kernel/src/rime/lie_symmetry.rs`](../rust/apedp-kernel/src/rime/lie_symmetry.rs) | `tests/test_dyn_rust_koopman_vs_ga.py`, `tests/test_dyn_rust_lie_symmetry.py` | Verified (6/6): (1) Delay-embedded linear spectral discovery (`koopman.rs`, affine AR(2) + companion eigenvalues) recovers harmonic frequency $\omega = 2.0000$ with spectral radius $1.00$ and holdout rollout $R^2 = 1.000000$ in $0.31$ ms; damped oscillator recovers $\omega = 3$, $\sigma = -0.5$; (2) matched-split native-GA comparison on identical train/holdout: GA discovers the exact trig identity ($-2\sin^2(t)+1$, holdout $R^2 = 1.000000$) in $14.71$ ms over 1560 evaluations vs Koopman $0.31$ ms; (3) Lie point-symmetry discovery (`lie_symmetry.rs`, exact polynomial determining-equation algebra + Gram/Jacobi nullspace) verifies the scaling/x-translation symmetries of $y' = y$ (nullity $\ge 2$), the exact 9-dimensional translation algebra of $y' = 0$, and $y$-translation for $y' = x$ (`experimental`). |
| Five-gate validation programme & Planck honesty fix | [`benchmarks/run_five_gate_validation.py`](../benchmarks/run_five_gate_validation.py), [`tests/test_dyn_planck_limitation.py`](../tests/test_dyn_planck_limitation.py) | `tests/test_dyn_five_gate_validation.py`, `tests/test_dyn_planck_limitation.py` | Verified (all 5 gates pass): Gate 1 exact recovery on 8/9 equations with the Planck limitation held; Gate 2 honest noise degradation (Landauer quantized through 5% noise; Schwinger at 2% noise refuses exact claim with trajectory $R^2 < 0$); Gate 3 four adversarial abstentions (sine, white noise, null world); Gate 4 matched-compute ablation — Rabi tie at $R^2 \approx 1.0$ (RIME 26 ms vs GA 8 ms), Brus GA overfits (holdout $R^2 = -2.93$) while RIME transfers; Gate 5 sealed confirmation under fresh seed 12345. Planck: proved no exact first-order polynomial ODE exists ($u'/u \notin \mathbb{Q}(v)$), engine correctly abstains, benchmark verdict corrected from PASSED to RECORDED LIMITATION (`experimental`). |
| RIME-AD autonomous multi-representation proposer | [`apedp_dyn/proposers/rime_ad_proposer.py`](../apedp_dyn/proposers/rime_ad_proposer.py) | `benchmarks/run_population_extrapolation_rime_ad_vs_ga.py` | `RimeAdProposer`: profiles dataset invariants (plateaus, dynamic range, Arrhenius tunneling), fans out across the native Rust representations (TV denoising, implicit null-space, differential Padé, shared-DAG enumeration, multi-term rational powers, saturating-rational Langmuir/Michaelis–Menten/Hill reps), ranks on dev data by BIC/MDL, and emits standardized `CandidateProposal` objects with out-of-sample `compute_score` callables; never promotes, never sees holdout (`experimental`). |
| Rust bridge zero-copy data plane, compile cache & kernel-path observability | [`apedp/rust_bridge.py`](../apedp/rust_bridge.py), [`rust/apedp-kernel/src/data.rs`](../rust/apedp-kernel/src/data.rs), [`rust/apedp-kernel/src/lib.rs`](../rust/apedp-kernel/src/lib.rs) | `tests/test_rust_bridge_observability.py`, `tests/test_rust_bridge_dispatch.py`, `tests/test_cross_implementation_conformance.py`, `cargo test` | Verified (45/45 bridge tests + 26/26 differential conformance + 179 Rust tests): (1) **Zero-copy columns** — `PythonColumns::extract` in `lib.rs` now hands `ArrayView1<f64>` views straight into NumPy buffers via the `DataCol` erased-column trait (`data.rs`), replacing the per-call `Vec<Vec<f64>>` deep-copy that re-copied the same dataset on every per-candidate crossing; parser/numeric/genetic/graph/eval/search/additive/residual surfaces take `&[V: DataCol]` so owned JSON-fed paths and borrowed NumPy paths are interchangable with no copy, and the same pattern extends to single-array arguments via a `GuardedVector` pin-once guard (contiguous `&[f64]` slices, strided one-copy fallback) for targets and metric arrays; (2) **Compile cache** — `RustKernelBridge._kernel_compile` LRU-caches FFI `compile` results keyed on `(expr_str, variable_names)` (owner-guarded so a swapped kernel clears it), and the Python fallback caches its sympy `lambdify` compilations by expression text; (3) **Kernel-path counters** — every `RustKernelBridge` instance reports `tallies` (rust/fallback counts), `last_path`, and attaches a `kernel_path` field to every returned result dict, so benchmark runs can prove the fast path executed (the fix the bridge's own audit asked for: "a counter, not a log line"); (4) **Loud missing-kernel warning** — bridge construction without the native kernel logs one WARNING naming the `maturin develop --release` build command instead of a buried debug line; (5) **Measured (before/after: git-HEAD kernel+bridge in an isolated venv, min-of-3 interleaved fresh runs, 600×4 float64)** — Python reference engine repeated eval 2678.9 → 34.5 µs/call (77×, lambdify cache), batch scoring round (15 candidates) 1087.6 → 888.1 µs (−18%), repeated candidate-evaluation call 110.8 → 95.6 µs (−14%; per-call cost at this size is Rayon-dispatch-dominated, so copy-elimination scales with dataset size), `compute_metrics` 11.1 → 7.6 µs (−32%), kernel-level `score_candidates_parallel` 817.7 → 785.2 µs (no kernel-path regression); artifacts: [`benchmarks/run_bridge_fastpath_benchmark.py`](../benchmarks/run_bridge_fastpath_benchmark.py) + [`benchmarks/bridge_fastpath_benchmark_results.json`](../benchmarks/bridge_fastpath_benchmark_results.json); (6) **Release profile** — workspace `Cargo.toml` adds `codegen-units = 1` to `lto = true`; (7) **Wheel artifacts** — `.github/workflows/test.yml` gains a `build-kernel-wheel` job (manylinux wheel via `PyO3/maturin-action`, uploaded as artifact so the kernel is `pip install`able without running maturin) and a tag-gated `publish-kernel-wheel` job that only fires with a `PYPI_API_TOKEN` secret present (`production`). |
| Spatio-Temporal PDE Discovery & Epistemic Referee Bridge | [`rust/apedp-kernel/src/rime/pde.rs`](../rust/apedp-kernel/src/rime/pde.rs), [`apedp_dyn/epistemic/pde_discovery.py`](../apedp_dyn/epistemic/pde_discovery.py), [`apedp_dyn/epistemic/pde_transfer.py`](../apedp_dyn/epistemic/pde_transfer.py) | `tests/test_dyn_frontier_9_of_10.py`, `tests/test_dyn_frontier_claims_integrity.py`, `tests/test_dyn_pde_transfer.py` | Verified (5/5 + 4/4 tests): (1) Native Rust PDE discovery engine (`pde.rs`) computes 4th-order spatial central stencils ($u_x, u_{xx}, u_{xxx}$) and temporal differences ($u_t$), constructing space-time feature tensors and applying Sequentially Thresholded Ridge Regression (STRidge); (2) Epistemic bridge (`pde_discovery.py`) partitions the time axis and scores a re-discovered fit on the held-out window (re-fit holdout, not model-transfer holdout); (3) Discovers 1D Heat diffusion ($\partial_t u = 0.5\, \partial_{xx} u$) — dev $R^2 = 0.9963$, holdout-window $R^2 = 0.999997$, coefficient $0.4972$; (4) Discovers Viscous Burgers ($\partial_t u = -1.0\, u\, \partial_x u + 0.1\, \partial_{xx} u$) — holdout-window $R^2 = 0.9999$, advection coefficient error $0.38\%$, diffusion coefficient error $11.2\%$ with one spurious low-weight $u$ term selected (coefficient-faithful structure, not exact coefficient recovery); (5) `pde_transfer.py` evaluates dev-discovered coefficients FROZEN on the holdout window with the identical stencils (true model-transfer metric; Burgers frozen $R^2 = 0.999772$ vs re-fit $0.999909$ on the Rust fixture). See [`frontier_9_of_10_architectural_resolution.md`](frontier_9_of_10_architectural_resolution.md) for the corrected record (`experimental`). |
| Discrete Number Theory & Active Invariant Transfer Engines | [`rust/apedp-kernel/src/rime/number_theory.rs`](../rust/apedp-kernel/src/rime/number_theory.rs), [`rust/apedp-kernel/src/rime/invariant_transfer.rs`](../rust/apedp-kernel/src/rime/invariant_transfer.rs), [`apedp_dyn/epistemic/number_theory_engine.py`](../apedp_dyn/epistemic/number_theory_engine.py), [`apedp_dyn/proposers/invariant_transfer_proposer.py`](../apedp_dyn/proposers/invariant_transfer_proposer.py) | `tests/test_dyn_number_theory_and_invariant_transfer.py` | Verified (4/4 tests): (1) Exact rational reconstruction via Farey sequences and continued fractions recovers Laughlin fractional quantum hall factors ($\nu = 2/5$), Brus powers ($2.45 = 49/20$), and critical exponents ($1/8$) from noisy decimals; (2) Integer relation discovery via lattice search finds exact stoichiometric conservation laws ($2\text{H}_2 + \text{O}_2 - 2\text{H}_2\text{O} = 0$) and harmonic frequency resonances; (3) Modular lattice discovery identifies discrete crystal lattice spacings ($\Delta = 2.0, x_0 = 1.0$) with zero submultiple degeneracy; (4) Active Invariant Transfer Proposer evaluates cross-environment Invariant Risk Minimization (IRM) penalties, worst-regime retention gating, and active extrapolation barrier guards, promoting true invariant causal representations while completely rejecting spurious correlators inverted under regime intervention (`production`). |
| 2D PDEs, Non-Uniform Stencils & Coupled Dynamical Systems | [`rust/apedp-kernel/src/rime/pde.rs`](../rust/apedp-kernel/src/rime/pde.rs), [`rust/apedp-kernel/src/rime/coupled_system.rs`](../rust/apedp-kernel/src/rime/coupled_system.rs), [`apedp_dyn/epistemic/pde_discovery.py`](../apedp_dyn/epistemic/pde_discovery.py), [`apedp_dyn/epistemic/coupled_discovery.py`](../apedp_dyn/epistemic/coupled_discovery.py) | `tests/test_dyn_coupled_and_2d_pde.py` | Verified (3/3 tests): (1) 2D spatial spatio-temporal PDE discovery (`discover_pde_2d`) recovers continuous 2D Heat diffusion ($\partial_t u = 0.25 \nabla^2 u$) from $u(t, x, y)$ with $R^2 > 0.99$; (2) Arbitrary non-uniform mesh stencils via exact Fornberg weights (`fornberg_weights`, `discover_pde_nonuniform`) recover governing PDEs on irregular coordinate grids; (3) Coupled multi-variable non-linear ODE discovery (`discover_coupled_system`) recovers multi-state interaction systems simultaneously, discovering Lotka-Volterra predator-prey dynamics ($\dot{x} = 1.5x - 1.0xy$, $\dot{y} = -3.0y + 1.0xy$) with exact parameter recovery and system $R^2 > 0.99$ (`production`). |
| Hybrid High-Dimensional Continuous PDE Discovery & Fused Jet2/Jet3 / Vlasov-Attention CPU Engine (1D to 100D+) | [`rust/apedp-kernel/src/rime/high_dim_pde.rs`](../rust/apedp-kernel/src/rime/high_dim_pde.rs), [`apedp_dyn/epistemic/high_dim_pde.py`](../apedp_dyn/epistemic/high_dim_pde.py) | `tests/test_dyn_high_dim_hybrid_pde.py` | Verified (33/33 Python tests + 11/11 Rust unit tests, 9.5/10 Gold Standard): (1) Hybrid dimension dispatch for all $D < 10$ ($D \in [1, 9]$): routes to Eulerian Cartesian grid finite differences (`discover_pde`, `discover_pde_2d`, `discover_pde_3d`, and general stride-based `discover_pde_nd_cartesian` with 4th-order 5-point central stencils when $N_k \ge 7$ and 2nd-order 3-point stencils when $3 \le N_k < 7$, verified on 2D, 3D, 4D, and 5D Cartesian grids with frozen-coefficient holdout $R^2 > 0.9999$); (2) for $D \ge 10$ (tested on $D=20, 50, 64, 100$): (a) **Mesh-Free Point Cloud Sampling**: true scattered point cloud discovery via multi-scale Gaussian Radial Basis Functions (`discover_pde_rbf_point_cloud_hd`, exact analytical $D$-dimensional RBF Laplacian $\nabla^2_D \phi = 2\varepsilon^2(2\varepsilon^2 r^2 - D)\phi$ without finite-difference probes, $R^2 > 0.9999$) plus axis-star probe point clouds (`discover_pde_point_cloud_hd`, `evaluate_point_cloud_pde_frozen_oos`); (b) **Radial / Isotropic Symmetry Reduction**: exact hyperspherical reduction $\nabla^2_{20} u = u_{rr} + \frac{19}{r} u_r$ (`discover_pde_radial_hd`, $R^2 > 0.99999$); (c) **Low-Rank Tensor Trains (TT / MPS)**: multi-core bond-rank $R$ Matrix Product State discovery (`discover_pde_tensor_train_hd`, $R^2 > 0.9999$ on rank-2 20D heat kernel sum in $\mathcal{O}(N_t D N_x R)$ memory) plus rank-1 separable factor slice (`discover_pde_separable_hd`); (d) **Deep BSDE / Neural PINNs**: mesh-free collocation PDE discovery (`discover_pde_neural_pinn_hd`, $R^2 > 0.9999$ in 20D) powered by the L1/L2 Block-Tiled AVX2 Fused 2-Jet Neural PDE Operator (`eval_jet2_neural_pde_operator`) and Ornstein-Uhlenbeck Fokker-Planck operator (`eval_fokker_planck_ou_operator`), plus exact 3rd-order `Jet3` parameter gradients (`compute_jet3_pde_loss_and_grads`, `RustPdeNetwork`, Rayon-parallelized to $7.05\text{ ms}$, cutting the JAX gap by $2.35\times$) and Block-Tiled Online-Softmax McKean-Vlasov Self-Attention PDE Stepper (`step_vlasov_attention_pde_hd`, zero-allocation tile scratchpads outperforming PyTorch float64 CPU OpenMP by $4.97\times$ at $N=512$ and $4.77\times$ at $N=4096$); (e) **SymPy Machine-Checked Mathematical Certificate**: formal computer-algebra verification (`test_jet3_sympy_symbolic_equivalence`) proving all Rust Jet3 gradients match SymPy symbolic derivatives down to machine epsilon ($< 10^{-14}$); (f) **Canonical Literature 100D & 50D Benchmarks & Convergence Study**: exact recovery of 100-dimensional Black-Scholes-Barenblatt (`discover_pde_black_scholes_barenblatt_100d`, $R^2_{\text{OOS}} > 0.999$), 100-dimensional Hamilton-Jacobi-Bellman optimal control (`discover_pde_hjb_100d`, recovering diffusion $\nu=0.05$, gradient penalty $\beta=-0.04$, and reaction $\mu=0.02$ with $R^2_{\text{OOS}} > 0.999$ while rejecting $u^2$), and 50-dimensional nonlinear Allen-Cahn bistable reaction-diffusion (`discover_pde_allen_cahn_hd`, $R^2_{\text{OOS}} > 0.98$); verified timestep convergence (`test_black_scholes_barenblatt_timestep_convergence`) demonstrating that the original 53% discount discrepancy under Forward Euler is strictly linear $\mathcal{O}(dt)$ truncation error ($dt=0.02 \implies +0.0261$, $dt=0.01 \implies +0.0127$), driven to $< 0.1\%$ ($0.06\%$) under 4th-order Runge-Kutta at $dt \le 0.002$; (g) **Noise-Robust Tikhonov Smoothing & Adversarial Cauchy Resilience**: suppresses temporal noise variance by $> 2\times$ (`tikhonov_smooth_trajectory`), cutting residual RMSE by $> 2\times$ under heavy-tailed Cauchy perturbations (`test_pde_adversarial_heavy_tailed_cauchy_noise_and_outliers`) and eliminating spurious term promotions; (h) **Epistemic Referee 400-Trial Monte Carlo Calibration, Misspecification Refusal & Causal Arrow Guard**: empirical False Positive Rate $\text{FPR} \le 0.01$ (0/300 false positives $\implies$ exact one-sided Clopper-Pearson 95% upper bound $0.994\% < 1.0\%$) and True Positive Rate $\text{TPR} \ge 0.95$ (100/100 true positives $\implies$ Wilson 95% score lower bound $96.30\% > 95.0\%$, exact one-sided lower bound $97.05\%$); verified adversarial model misspecification refusal (`test_pde_adversarial_fractional_misspecification`, refusing causal certification on non-local fractional diffusion $|k|^{1.4}$); verified out-of-distribution geometry transfer (`test_pde_out_of_distribution_geometry_transfer`, frozen hypercube $[-0.2, 0.2]^{20}$ model transferring to unseen spherical shell $r \in [0.4, 0.8]$ with $R^2_{\text{OOD}} > 0.999$); and verified thermodynamic arrow-of-time causal direction guard (`test_pde_time_reversal_causal_arrow_guard`, proving forward dissipation $\alpha > 0$ vs unphysical reversed entropy decrease $\alpha < 0$) (`production`). |
| JetFlow-Video Pip Package & Google Colab Training Engine | [`packages/jetflow-video/pyproject.toml`](../packages/jetflow-video/pyproject.toml), [`packages/jetflow-video/jetflow_video/tensor_train.py`](../packages/jetflow-video/jetflow_video/tensor_train.py), [`packages/jetflow-video/jetflow_video/stepper.py`](../packages/jetflow-video/jetflow_video/stepper.py), [`packages/jetflow-video/jetflow_video/noether.py`](../packages/jetflow-video/jetflow_video/noether.py), [`packages/jetflow-video/jetflow_video/physics.py`](../packages/jetflow-video/jetflow_video/physics.py), [`packages/jetflow-video/jetflow_video/bridge.py`](../packages/jetflow-video/jetflow_video/bridge.py), [`packages/jetflow-video/jetflow_video/model.py`](../packages/jetflow-video/jetflow_video/model.py), [`packages/jetflow-video/jetflow_video/trainer.py`](../packages/jetflow-video/jetflow_video/trainer.py), [`packages/jetflow-video/examples/colab_quickstart.ipynb`](../packages/jetflow-video/examples/colab_quickstart.ipynb), [`packages/jetflow-video/examples/train_colab.py`](../packages/jetflow-video/examples/train_colab.py), [`packages/jetflow-video/examples/generate_video.py`](../packages/jetflow-video/examples/generate_video.py) | `tests/test_dyn_jetflow_video_package.py` | Verified (8/8 unit tests passing): Standalone PEP 517/621 installable pip package (`pip install -e packages/jetflow-video`) optimized for Google Colab CPU and NVIDIA T4/A100/L4 GPUs. Provides PyTorch-native 2nd-order Jet2 flow matching, exact discrete Fourier Leray solenoidal projection ($\nabla \cdot \mathbf{v} = 0$, residual $< 10^{-6}$), 4D Tensor-Train compression ($75\times$ to $186\times$ VRAM reduction), Continuous Wavelet and Diffusers VAE bridges, zero-disk direct Hugging Face pre-computed matrix streaming (`io.BytesIO` in-memory RAM caching), standalone inference CLI generator (`generate_video.py`), and full training loop (`JetFlowTrainer`) with multi-objective Jet2 loss ($\mathcal{L}_{\text{flow}} + \lambda_{\text{acc}} \mathcal{L}_{\text{acc}} + \lambda_{\text{div}} \mathcal{L}_{\text{noether}}$), automatic mixed precision (AMP), and 1-click Colab interactive quickstart notebook (`experimental`). |
| JetFlow-Video Fast-Path Remediation Integrity Stack (2026-10-17) | [`apedp_dyn/video/jet_flow_model.py`](../apedp_dyn/video/jet_flow_model.py), [`apedp_dyn/video/semantic_bridge.py`](../apedp_dyn/video/semantic_bridge.py) | `tests/test_dyn_jet_flow_sota_path.py`, `tests/test_dyn_jet_flow_video.py` | Verified (live): (1) the NumPy `train_step` fake trainer (one scalar mean-gradient on a single bias vector, presented as Jet2 flow-matching training) is REMOVED — the method is now diagnostics-only (`DeprecationWarning`, no parameter mutation, pinned); all real training routes to `packages/jetflow-video JetFlowTrainer` (full-parameter AdamW, grad-flow confirmed); (2) hash-based prompt embedding replaced by process-stable feature hashing (`feature_hash_word_char_trigram_v1`, MD5 indices — the builtin `hash()` salt bug is gone: embeddings byte-identical across processes, pinned in-subprocess; lexical similarity preserved 0.744 vs 0.229 on shared-vocabulary prompts); (3) found and documented a real package bug: `model.generate_video` samples in the direction OPPOSITE to `trainer.compute_loss`'s parameterization — the shipped samplers in `first_frame.py`/`distill.py` integrate the trainer-consistent direction (t: 1 -> 0) (`experimental`). |
| Held-Out Video Evaluation Gate (certification instruments) | [`apedp_dyn/video/eval_gate.py`](../apedp_dyn/video/eval_gate.py) | `tests/test_dyn_video_eval_gate.py` | Verified (5/5): PSNR/SSIM (identity ceilings, monotone degradation), temporal consistency (smooth advected clips beat frame-shuffled copies), FVD PROXY (sliced Wasserstein-2 over a frozen deterministic featurizer — NOT I3D-FVD, honest label, identical distributions separate ~0, degraded > clean, cross-seed deterministic), burnable holdout split (disjoint, deterministic, exclusive), and a certify API whose thresholds are supplied by the caller (no new hardcoded threshold literals — repo rule 1.3) (`audit-only`). |
| Video-Metrics Rust Kernel Dispatch (bridge fast path) | [`rust/apedp-kernel/src/video.rs`](../rust/apedp-kernel/src/video.rs), [`apedp/rust_bridge.py`](../apedp/rust_bridge.py), [`apedp_dyn/video/eval_gate.py`](../apedp_dyn/video/eval_gate.py) | `tests/test_dyn_video_metrics_kernel.py` | Verified (18/18 across the three video-gate test files; full 58-test dynamic regression suite green): (1) `eval_gate` acquires the kernel-backed bridge lazily (module import never pulls the apedp tree) and routes `compute_psnr` / `compute_ssim` through `video_psnr` / `video_ssim_frame`; (2) a missing or unbuilt kernel silently degrades to the verbatim NumPy reference bodies — bridge docstring discipline, same numbers, slower path; (3) dispatch counters tally Rust-path vs fallback-path calls, and shape-mismatch guards raise ValueError before any kernel call in both paths; (4) exact parity (≤ 1e-10) against the NumPy references on identity, additive-noise, and (T, 3, H, W) video-routing cases, on all three dispatch modes (rust, forced-fallback bridge, pure-Python) (`production`). |
| Fluid Domain Scoping (domain-matched dataset) | [`apedp_dyn/video/fluid_domain.py`](../apedp_dyn/video/fluid_domain.py) | `tests/test_dyn_fluid_domain.py` | Verified (4/4): deterministic 2D incompressible Navier-Stokes tracer clips (velocity from streamfunction — grid-divergence-free to $|\nabla \cdot u| < 10^{-10}$, the property that makes the Noether projector physically meaningful on this domain), semi-Lagrangian dye advection passing the temporal-consistency instrument, and a scope manifest with exclusive holdout ids via `held_out_split` (the domain scoping lever: one learnable physics class instead of "everything") (`experimental`). |
| Real-First-Frame Conditioning + Perceptual Decoder + Distillation + CPU Export (photorealism fast-path) | [`packages/jetflow-video/jetflow_video/first_frame.py`](../packages/jetflow-video/jetflow_video/first_frame.py), [`packages/jetflow-video/jetflow_video/perceptual_decoder.py`](../packages/jetflow-video/jetflow_video/perceptual_decoder.py), [`packages/jetflow-video/jetflow_video/distill.py`](../packages/jetflow-video/jetflow_video/distill.py), [`benchmarks/run_jetflow_distill_benchmark.py`](../benchmarks/run_jetflow_distill_benchmark.py), [`benchmarks/run_jetflow_cpu_export_benchmark.py`](../benchmarks/run_jetflow_cpu_export_benchmark.py) | `tests/test_dyn_jetflow_anchored_and_distill.py`, `tests/test_dyn_jetflow_perceptual_decoder.py`, `tests/test_dyn_jetflow_cpu_export.py` | Verified (11/11 + 2 benchmark artifacts): (1) anchored flow-matching (clean real frame 0 through training corruption AND generation; loss excludes the anchor; anchor re-imposed bitwise every step); (2) perceptual decoder = Haar base + ZERO-INIT residual conv stack (starts at exact Haar parity — no regression cliff) + patch-LSGAN + gradient loss, trained on fluid frames; (3) honest solution distillation: trained teacher (16-step reference) -> student endpoint RMSE 0.808 -> 0.507 measured on the artifact `benchmarks/jetflow_distill_curves.json` (the honest replacement for the old integrator-only "2 beats 50" benchmark); (4) ONNX export (static traced graph per clip config — traced attention cannot carry a dynamic T, documented) + `onnxruntime` dynamic int8: size 173.5 KB -> 104.9 KB, parity $\max|\Delta| = 0.0$, latency measured (eager 8.68 ms vs ORT fp32 5.32 ms — int8 is a size win, fp32 ORT the speed win at this scale) in `benchmarks/jetflow_cpu_export_benchmark_results.json` (`experimental`). |
| Unified PDE/ODE Subpackages (`apedp.pde`, `apedp.ode`, `apedp_dyn.pde`, `apedp_dyn.ode`) | [`apedp_dyn/pde/__init__.py`](../apedp_dyn/pde/__init__.py), [`apedp_dyn/ode/__init__.py`](../apedp_dyn/ode/__init__.py), [`apedp/pde/__init__.py`](../apedp/pde/__init__.py), [`apedp/ode/__init__.py`](../apedp/ode/__init__.py) | `tests/test_dyn_pde_ode_packages.py` | Verified (unit tests passing): first-class subpackages (`apedp.pde`, `apedp.ode`, `apedp_dyn.pde`, `apedp_dyn.ode`) exposing high-dimensional continuous PDE discovery ($1\text{D}$ to $100\text{D}+$), multi-variable coupled non-linear ODE discovery (Lotka-Volterra predator-prey), typed data containers (`PdeGridData`, `PdePointCloudData`, `OdeTrajectoryData`), standardized proposer engines (`PdeProposer`, `CoupledOdeProposer`) conforming to `ProposerBase`, and epistemic referee validation engines (`PdeReferee`, `OdeReferee`) enforcing frozen holdout verification and thermodynamic causality (`production`). |
| Decoupled Continuous Spatio-Temporal Transport (DC-STT) & Neural Transport Flow Net | [`apedp_dyn/video/decoupled_transport.py`](../apedp_dyn/video/decoupled_transport.py), [`rust/apedp-kernel/src/video.rs`](../rust/apedp-kernel/src/video.rs), [`apedp/rust_bridge.py`](../apedp/rust_bridge.py) | `tests/test_dyn_decoupled_continuous_transport.py` | Verified (6/6 tests passing): (1) Native Rust kernel fast-path execution and zero-copy tallies accounting; (2) Differential cross-validation: exact numerical parity between native Rust kernel and NumPy fallback ($\max\|\Delta\| < 10^{-12}$); (3) Noether mass-conservation: Helmholtz-Hodge projection onto divergence-free subspace; (4) Optical flow divergence and disocclusion mask detection with residual source injection; (5) Multi-scene video trajectory generation and 4D Tensor-Train matrix compression; (6) Trainable `NeuralTransportFlowNet` predicting streamfunction curl $\nabla^\perp \psi_\theta$, compressible potential $\nabla \phi_\theta$, and disocclusion source $\mathbf{s}_\theta$ end-to-end with Rust rollout (`experimental`). |
| JetFlow-Video v0.2 Architectural Pillars (Compositional Text Encoder, 3D Causal VAE, SE(3) Plücker/Depth Transport, 3D Tubelet DiT & Matched Benchmark) | [`packages/jetflow-video/jetflow_video/text_encoder.py`](../packages/jetflow-video/jetflow_video/text_encoder.py), [`packages/jetflow-video/jetflow_video/depth_se3.py`](../packages/jetflow-video/jetflow_video/depth_se3.py), [`packages/jetflow-video/jetflow_video/bridge.py`](../packages/jetflow-video/jetflow_video/bridge.py), [`packages/jetflow-video/jetflow_video/model.py`](../packages/jetflow-video/jetflow_video/model.py), [`packages/jetflow-video/jetflow_video/trainer.py`](../packages/jetflow-video/jetflow_video/trainer.py), [`docs/jetflow_vs_diffusion_matched_benchmark.md`](../docs/jetflow_vs_diffusion_matched_benchmark.md) | `tests/test_dyn_jetflow_video_package.py`, `benchmarks/run_jetflow_vs_diffusion_matched.py` | Verified (10/10 package tests + matched benchmark): (1) `CompositionalTextEncoder` (`text_encoder.py`): sequence-level `(B, L_seq, D_text)` contextual token encoding with order-sensitive modifier binding, negation gating, HuggingFace bridge, and prompt-to-physics steering extraction (`extract_physical_steering_from_prompt`); (2) `Causal3DSpatioTemporalVAE` (`bridge.py`): strict past-only 3D causal convolutions (`CausalConv3d`, `CausalResBlock3D`) with diagonal Gaussian KL posterior and end-to-end video batch training (`fit_video_batch`); (3) 3D Depth-Aware SE(3) & Plücker Ray Transport (`depth_se3.py`): exact Rodrigues/Jacobian $\mathfrak{se}(3) \to \text{SE}(3)$ exponential map (`se3_exp_map`), 6D Plücker camera ray embeddings (`compute_plucker_rays`, `PluckerRayEmbedder`), and learned monocular depth + 3D perspective warping (`DepthAware3DTransportEngine`); (4) 3D spatiotemporal tubelet patchification (`patch_size=(pt, ph, pw)`), trainable `TTLinear` FFN, 2-evaluation fused Jet2 directional material derivative stepper, subband-preserving Noether conservation, and EMA + CFG-dropout cosine-warmup trainer. Seed-0 matched-benchmark numbers are exploratory; fresh-seed confirmation + architecture invariants are registered in [`docs/PREREGISTRATION_JETFLOW_VIDEO_V02.md`](PREREGISTRATION_JETFLOW_VIDEO_V02.md) (`experimental`). |
| Perceptual-Refiner SSIM-Gain Measurement (lossless zero-gain control + two lossy regimes, 2026-10-20) | [`benchmarks/vortex_perceptual_finetune_results.json`](../benchmarks/vortex_perceptual_finetune_results.json), [`benchmarks/vortex_lossy_regimes_refinetune_results.json`](../benchmarks/vortex_lossy_regimes_refinetune_results.json), [`benchmarks/run_vortex_lossy_regimes_refinetune.py`](../benchmarks/run_vortex_lossy_regimes_refinetune.py), [`packages/jetflow-video/jetflow_video/perceptual_decoder.py`](../packages/jetflow-video/jetflow_video/perceptual_decoder.py) | `tests/test_dyn_jetflow_perceptual_decoder.py` | Verified (3 recorded runs, exit 0, instrument via `\`apedp_dyn.video.eval_gate\` SSIM/PSNR): (1) LOSSLESS control — 60-step LSGAN finetune on the near-invertible vortex clip measured **zero SSIM gain** (clean 1.00000 → 1.00000; TT-8-degraded 0.98920 both directions), with the untrained-refiner exact-parity control (SSIM 1.000000 / 100.0 dB) pinning the cause: the zero-init residual starts at Haar parity, the recon-L1/gradient losses sit at 0.0000, so only adversarial drift moved weights (clean-path PSNR 100.00 → 88.62 dB); (2) JPEG regime — paired training (q_train=15 latents → clean GT) recovered SSIM **+0.000818** and PSNR +0.047 dB over the JPEG baseline (SSIM 0.977109 / 31.03 dB), with disclosed held-out regression at unseen q=60 (SSIM −0.000202, PSNR −2.81 dB — the refiner overfits the trained degradation strength); (3) noisy-latent regime — σ = 0.15·std(\|lat\|) baseline SSIM 0.365778 / 21.46 dB recovered SSIM **+0.009724** / PSNR +0.278 dB, held-out σ = 0.05·std SSIM +0.002069 but PSNR −3.47 dB. Measured pattern: the sharpening mechanism DOES produce nonzero SSIM gain on genuinely lossy inputs, the effect is modest at 60 steps on one 8-frame clip, and it trades MSE (PSNR) down on unseen degradation strengths — sharpness-first at the cost of L2. Honest scope: single synthetic clip, in-distribution content, no convergence claim. | `experimental` |
| Temporal Smoothness & Visual Sharpness Mechanism Record (claim-set adjudication + fresh benchmarks, 2026-10-20) | [`apedp_dyn/video/continuous_interpolator.py`](../apedp_dyn/video/continuous_interpolator.py), [`benchmarks/run_jet_flow_video_benchmark.py`](../benchmarks/run_jet_flow_video_benchmark.py), [`benchmarks/run_vortex_slow_motion_demo.py`](../benchmarks/run_vortex_slow_motion_demo.py), [`packages/jetflow-video/jetflow_video/perceptual_decoder.py`](../packages/jetflow-video/jetflow_video/perceptual_decoder.py), [`packages/jetflow-video/jetflow_video/bridge.py`](../packages/jetflow-video/jetflow_video/bridge.py) | `tests/test_dyn_jet_flow_video.py`, `tests/test_dyn_jetflow_perceptual_decoder.py` | Verified (2 benchmark reruns + 1 demo artifact, exit 0): SMOOTHNESS lives on the Tensor-Train temporal spline, not on the stepper — the NumPy `ContinuousTemporalInterpolator` (scipy `CubicSpline` natural BC over the TT temporal core `G0(T, R1)`) in `resample_to_fps` 24→60 FPS (39 frames, 0.090 ms/frame, \|D_t²u\| = 8.1e-05) and 24→120 FPS (76 frames, 0.403 ms/frame, 5e-06), artifacts `benchmarks/jet_flow_video_benchmark_results.json` + `docs/jet_flow_video_benchmark.md`; torch twin `evaluate_continuous_temporal_frame_torch` (C¹ Catmull-Rom over the TT temporal core) exposed end-to-end as `VideoVAEBridge.decode_continuous_frame_rgb`; reproducible slow-motion demo: 8-source → 24 rendered frames through a device-side rank-16 TT (9.54× compression, 89.5% memory saved), 93.2 ms/frame CPU, rendered-sequence \|D_t²u\| = 2.864e-05 (`benchmarks/vortex_slow_motion/`: 24 PNG frames + contact sheet + 12 FPS looping GIF + measured results JSON); the NumPy Jet2 stepper uses the forward `[0.0 → 1.0]` schedule, while the separate Torch `JetFlowVideoModel.generate_video` uses `[1.0 → 0.0]`; neither provides a continuous-frame path. SHARPNESS: `train_perceptual_stack` trains the zero-init `PerceptualRefiner` (which produces refined frames) against a `PatchDiscriminator` (which scores them) with reconstruction L1, image-gradient L1 matching weighted by configurable `lambda_grad` (default 0.5; not a GAN gradient penalty), and adversarial loss weighted by `lambda_adv` (default 0.1); the VAE's `compute_vae_loss` includes reconstruction L1, temporal first-difference L1, spatial edge-gradient L1 (`0.25 * edge_loss`), and KL; `JetFlowTrainer` adds configurable `lambda_acc`-weighted material-acceleration MSE and `lambda_noether`-weighted divergence loss to flow MSE (defaults 0.5 and 0.1, not constant penalties). THE FABRICATED MECHANISMS DO NOT EXIST: no `SharperVideoEncoder` / `SharperVideoDecoder` modules (zero hits, full `git log --all -S`), no "MSE + 0.3·L1" compound loss anywhere, no `sharp_long_wave_output.mp4`, and the repo's recorded FPS upsampling is 24→60/120 (not any 12-vs-4 FPS pair). Numbers are CPU-measured on the working tree checkout; the interpolator stack remains `experimental`. Status: `experimental`. |
| Higher-Order PDE Hypothesis Engine & Universal Spatio-Temporal Architecture Specification | [`apedp_dyn/pde/higher_order_engine.py`](../apedp_dyn/pde/higher_order_engine.py), [`apedp_dyn/pde/__init__.py`](../apedp_dyn/pde/__init__.py), [`apedp/pde/__init__.py`](../apedp/pde/__init__.py), [`docs/jetflow_universal_spatiotemporal_architecture.md`](../docs/jetflow_universal_spatiotemporal_architecture.md) | `tests/test_dyn_higher_order_pde_hypothesis_engine.py` | Verified (4/4 unit tests passing): (1) `discover_higher_order_pde`: discovers 1D and 2D governing PDEs up to 4th spatial order ($\partial_x, \partial_{xx}, \partial_{xxx}, \partial_{xxxx}, \nabla^2, \nabla^4$) and 2nd temporal order ($\partial_t u$ and $\partial_{tt} u$) via 4th-order finite-difference stencils or exact Fourier spectral differentiation coupled with variance-normalized BIC model selection across temporal orders; (2) `HigherOrderPdeProposer(ProposerBase)`: discovers candidate PDEs strictly on development data, integrates with 4D JetFlow Tensor-Train latent compression (`compress_video_latent_torch`), and emits standardized `CandidateProposal` objects without accessing confirmation holdouts; (3) `HigherOrderPdeReferee`: adjudicates frozen discovered coefficients on sealed holdouts (`min_holdout_r2 = 0.95` default; registered fresh-seed two-sided validation protocol in [`docs/PREREGISTRATION_HIGHER_ORDER_PDE.md`](PREREGISTRATION_HIGHER_ORDER_PDE.md)) and enforces the **Ultraviolet Fourier Symbol Stability Guard** (`check_ultraviolet_well_posedness`: requiring $c_4 < 0$ for 4th-order biharmonic damping $\text{Re}(\lambda(k)) = -c_2 k^2 + c_4 k^4 \to -\infty$ as $|k| \to \infty$, certifying Kuramoto-Sivashinsky $\partial_t u = -u\partial_x u - 0.4\partial_{xx} u - 0.05\partial_{xxxx} u$ while rejecting explosive $c_4 > 0$ instability); (4) `docs/jetflow_universal_spatiotemporal_architecture.md`: complete architectural specification covering JetFlow-DiT taxonomy, PyTorch training + ONNX/Rust deployment, $5.6\times$ speedup breakdown, multi-modal conditioning, and 6 non-video continuum scientific domains. Exploratory fixed-seed evidence only: fresh-seed confirmation is registered in [`docs/PREREGISTRATION_HIGHER_ORDER_PDE.md`](PREREGISTRATION_HIGHER_ORDER_PDE.md) (`experimental`). |
| Distributed Chunked Sweep Executor & Out-of-Core Partitioning | [`apedp_dyn/distributed/chunked_executor.py`](../apedp_dyn/distributed/chunked_executor.py), [`apedp_dyn/distributed/__init__.py`](../apedp_dyn/distributed/__init__.py) | `tests/test_dyn_distributed_chunking.py` | Verified (6/6 tests passing): (1) Out-of-core streaming generator mapping (`map_stream`) guaranteeing $\mathcal{O}(\text{chunk\_size})$ bounded RAM during parameter/hypothesis sweeps; (2) Partitionable sharded work specs (`PartitionSpec`, `DistributedTask`); (3) Fault-tolerant disk checkpointing with automatic partition resumption; (4) Multi-backend dispatch (`serial`, `thread_pool`, `multiprocessing`, and remote `ray` cluster hook with graceful fallback) (`production`). |
| Spatial Flow Coordinate Convention & Bilinear Warp Parity | [`apedp_dyn/video/flow_head_rust.py`](../apedp_dyn/video/flow_head_rust.py) | `tests/test_dyn_rust_parity_harness.py` | Verified (6/6 tests passing): Formal coordinate transformation bridge (`SpatialFlowConvention`, `eulerian_velocity_to_displacement`, `displacement_to_eulerian_velocity`) establishing mathematical equivalence between Eulerian advection velocities ($x_{\text{src}} = x - dt \cdot v(x)$) and Lagrangian kernel sampling displacements ($x_{\text{sample}} = x + \Delta x$ where $\Delta x = -dt \cdot v$) (`production`). |
| Notebook, Cloud & Storage Runtime Adapter | [`apedp_dyn/notebook_adapter.py`](../apedp_dyn/notebook_adapter.py), [`apedp_dyn/env.py`](../apedp_dyn/env.py) | `tests/test_dyn_notebook_adapter.py` | Verified (30/30 unit tests passing): Provides universal platform compatibility across Google Colab, Kaggle Notebooks, Databricks notebooks and job clusters, JupyterLab, CI/CD runners, and dedicated workstations. Features: (1) Automatic platform and notebook detection (`PlatformType.COLAB`, `KAGGLE`, `DATABRICKS`, `JUPYTER`, `WORKSTATION`, `CLUSTER`, `CI`); (2) Configurable multi-tier storage architecture routing temporary files via `APEDP_TEMP_STORAGE`, `/tmp`, platform temp dirs, or Databricks `/local_disk0/apedp_temp`, and managing persistent storage via Google Drive, Databricks DBFS (`/dbfs/FileStore/apedp_storage`, falling back to driver-local storage with a high-visibility epistemic-risk warning when no DBFS mount is present), or dedicated NVMe; (3) Anti-Snooping Epistemic Safeguards enforcing AGENTS.md §1.2 by checking holdout ledger persistence and issuing high-visibility warnings if running on ephemeral storage where VM resets erase burned confirmation holdouts; (4) Native Rust `apedp_kernel` detection and transparent fallback to bit-identical pure-Python/NumPy reference engines ($10^{-12}$ tolerance) or pre-compiled wheel download without requiring cold-start `maturin` builds; (5) Resource safety governor calculating safe worker counts to prevent notebook kernel deadlocks or CUDA fork crashes (Databricks: single worker on GPU clusters, `min(cpu_count, 16)` on CPU clusters); (6) 1-line notebook initialization (`init_environment()`) and scoped temporary storage context manager (`temporary_storage_scope`); (7) Validated end-to-end tutorial notebook execution (`notebooks/apedp_scientific_discovery_tutorial.ipynb`, 8 steps in one sequential namespace): adapter init, synthetic microstructure data, proposer candidate generation, Gates 4–5 adjudication, holdout burning with a self-recreating demo ledger marker (synthetic `cand_alpha_01`/`cand_beta_02` namespace only), scoped storage & snapshots, micro-sample symbolic discovery (`RimeProposer` on Langmuir 1918 through the structure-aware, graph-validated referee of `benchmarks/run_rime_microsample_adjudication.py`), and the SIR (`CompartmentalSirProposer`, fitted R0=2.499 vs true 2.50 with exact mass conservation) / Michaelis–Menten (frozen 12/12-fold graph-validated record, LOOCV RMSE 0.1316) companion pathways; badge launch URLs point at the live `psalmprax/science` `master` branch (`production`). |
| Native-Kernel Availability Shim (kernel-less installs) | [`apedp_dyn/kernel_shim.py`](../apedp_dyn/kernel_shim.py) | `tests/test_dyn_all_subpackages.py` | Verified (fresh-venv standalone-install proof + regression tests): `import apedp_dyn` succeeds on installations without the native `apedp_kernel` extension. The `missing_kernel` sentinel replaces module-level `import apedp_kernel` in the five kernel-only engines (`ode/dispatch.py`, `epistemic/coupled_discovery.py`, `epistemic/pde_discovery.py`, `epistemic/high_dim_pde.py`, `proposers/rime_ad_proposer.py`); probing remains attribute-safe (`hasattr`, `repr`) while calls raise an actionable RuntimeError naming the prebuilt-wheel or `maturin` install step. The pure-Python `apedp` wheel excludes `apedp_kernel*` (`pyproject.toml`), keeping the extension exclusively in its maturin-built wheel (kernel-less venv verified: import chain + DMRG reference engine run; `discover_ode` raises the documented error) (`production`). |
| Remote & Distributed Object Storage Backend for Epistemic Ledgers | [`apedp_dyn/epistemic/remote_storage.py`](../apedp_dyn/epistemic/remote_storage.py), [`apedp_dyn/epistemic/experiment_ledger.py`](../apedp_dyn/epistemic/experiment_ledger.py) | `tests/test_dyn_remote_storage.py` | Verified (11/11 unit and integration tests passing): Durable, cloud-agnostic persistence for hypothesis lineage records, burned holdout registries, and candidate trial ledgers across ephemeral execution runtimes (Google Colab, Kaggle, Docker, HPC spot nodes). Enforces AGENTS.md §1.2 anti-snooping holdout discipline across node restarts. Features: (1) Pluggable backends: `LocalStorageBackend`, `MemoryStorageBackend`, `S3StorageBackend` (AWS S3 / MinIO / Ceph), and `GCSStorageBackend` (Google Cloud Storage); (2) Cross-platform distributed file and object locking with stale lease expiration (`stale_timeout`), token verification on release, and lock collision prevention; (3) FullLineageExperimentLedger integration (`sync_to_remote()`, `pull_from_remote()`, `export_snapshot()`); (4) Restores burned holdout registry across simulated container termination and strictly triggers `HoldoutContaminationError` upon subsequent access attempts (`production`). |
| Integer-Relation Search: LLL/PSLQ Companion to the Box Search | [`rust/apedp-kernel/src/rime/pslq.rs`](../rust/apedp-kernel/src/rime/pslq.rs), [`scripts/compare_integer_relation_methods.py`](../scripts/compare_integer_relation_methods.py), [`benchmarks/integer_relation_methods.json`](../benchmarks/integer_relation_methods.json), [`docs/INTEGER_RELATION_METHODS.md`](INTEGER_RELATION_METHODS.md) | `tests/test_dyn_integer_relation_methods.py` | Verified (7 Python tests + 5 Rust unit tests): an LLL reduction of `[ I \| round(x·10^digits) ]` with exact-rational Gram-Schmidt, a ten-digit precision ladder starting at the f64's own 17 digits, and three acceptance filters — exact rational substitution on the input values, a 10x size margin against the generic `10^(digits/n)` vector, and a residual 4x above the embedding's rounding floor unless it is exactly zero. Returns `None` rather than a candidate it cannot verify, and the method constant is `lll_integer_relation_pslq`. Compared against the branch-and-bound on eight declared cases: 4 correct + 4 out-of-scope refusals + 0 wrong for the box search, 8 correct + 0 wrong for the reduction, agreeing on 4 of 8, median 0.118 ms against 18.2 ms; the reduction is the only one that answers `n = 10`, `n = 12`, or coefficients outside `max_coeff` (`experimental`). |
| Cryptographic Multi-Party Preregistration Governance | [`apedp_dyn/epistemic/governance.py`](../apedp_dyn/epistemic/governance.py), [`specifications/governance_manifest.json`](../specifications/governance_manifest.json) | `tests/test_dyn_governance.py` | Verified (4/4 unit and integration tests passing): Enforces AGENTS.md §1.3 canonical specification integrity through cryptographic multi-party quorum signatures and an append-only transparency log. Features: (1) M-of-N Ed25519 threshold signatures across registered trustees certifying preregistration specifications; (2) Tamper detection mathematically rejecting unauthorized threshold modification; (3) Append-only SHA-256 hash-chain audit log linking specification genesis and trustee co-signatures; (4) Raises `GovernanceTamperingError` blocking candidate promotion if quorum is broken (`production`). |
| GPU & Tensorized Batched Physics Steppers | [`apedp_dyn/cuda/batch_physics_steppers.py`](../apedp_dyn/cuda/batch_physics_steppers.py) | `tests/test_dyn_cuda_batch_physics.py` | Verified (5/5 unit tests passing): Native PyTorch CUDA/MPS/CPU batched physical integrators bypassing CPU-host PCIe transfer bottlenecks during massive $10^4$ to $10^6$ candidate hypothesis evaluations. Features: (1) Vectorized 4th-order Runge-Kutta (`batch_rk4_step`) with $\mathcal{O}(dt^4)$ accuracy; (2) Symplectic Velocity-Verlet (`batch_symplectic_verlet_step`) conserving Hamiltonian energy across thousands of cycles; (3) Exact circular arc nonholonomic unicycle kinematics (`batch_unicycle_arc_step`) evaluated in parallel across 5,000+ candidates without branching; (4) Batched SE(3)-equivariant molecular relaxation gradient flow (`batch_molecular_conformation_flow`) strictly conserving center-of-mass; (5) Hardware accelerator detection with optional Triton availability probe (`production`). |
| Multi-Node HPC & Cloud Cluster Orchestration | [`apedp_dyn/distributed/cluster_launcher.py`](../apedp_dyn/distributed/cluster_launcher.py), [`cluster/slurm_job_array.sh`](../cluster/slurm_job_array.sh), [`cluster/k8s_batch_sweep_job.yaml`](../cluster/k8s_batch_sweep_job.yaml), [`cluster/ray_cluster_helm_values.yaml`](../cluster/ray_cluster_helm_values.yaml) | `tests/test_dyn_cluster_orchestration.py` | Verified (5/5 unit tests passing): Complete cluster deployment manifests and automated script generators for multi-node supercomputers and cloud clusters. Features: (1) Slurm `#SBATCH` job array scripts mapping SLURM_ARRAY_TASK_ID directly to chunked partition specifications; (2) Kubernetes Indexed batch Jobs (`completionMode: Indexed`) with shared anti-snooping volume mounts; (3) KubeRay / Ray cluster Helm configuration with CPU/GPU worker autoscaling; (4) Python cluster launcher validating specs and rendering execution scripts (`production`). |
| Epistemic Referee Dashboard & Visual DAG Explorer | [`apedp_dyn/ui/dag_explorer.py`](../apedp_dyn/ui/dag_explorer.py), [`apedp_dyn/ui/streamlit_app.py`](../apedp_dyn/ui/streamlit_app.py), [`apedp_dyn/ui/server.py`](../apedp_dyn/ui/server.py), [`apedp_dyn/ui/helpers.py`](../apedp_dyn/ui/helpers.py) | `tests/test_dyn_dag_explorer.py` | Verified (4/4 unit and integration tests passing): Interactive visual analytics and graph explorer for candidate hypothesis lifecycles. Features: (1) Topological DAG extraction (`extract_dag_lineage`) transforming experiment ledgers into mutation graphs with Cytoscape.js export; (2) Multi-objective Pareto frontier computation (`compute_pareto_frontier`) isolating non-dominated trade-offs across DSR, RMSE, and complexity; (3) Burned holdout registry monitor (`build_burned_holdouts_view`) displaying data-snooping boundaries; (4) Native FastAPI endpoints (`/api/epistemic/dag`, `/api/epistemic/pareto`, `/api/epistemic/burned_holdouts`); (5) Standalone Streamlit studio (`streamlit_app.py`) (`production`). |
| Quantum Magic State Cultivation & Distillation | [`apedp_dyn/quantum/magic_state.py`](../apedp_dyn/quantum/magic_state.py) | `tests/test_dyn_magic_state.py` | Verified (4/4 unit tests passing): Fault-tolerant quantum distillation and magic state cultivation modeling (parameterized continuous purification filters and Bravyi-Kitaev 15-to-1). Computes output infidelities, acceptance probabilities, and multi-round distillation schedules for candidate non-Clifford resource state proposals (`experimental`). |
| Multi-Agent Collaborative Robotics & Potential Field Kinematics | [`apedp_dyn/robotics/multi_agent_transport.py`](../apedp_dyn/robotics/multi_agent_transport.py), [`apedp_dyn/cuda/batch_physics_steppers.py`](../apedp_dyn/cuda/batch_physics_steppers.py) | `tests/test_dyn_multi_agent_robotics.py` | Verified (5/5 unit tests passing): Cooperative multi-agent non-holonomic kinematics with repulsive artificial potential collision avoidance, verified head-on collision deflection, dynamic turn-taking role assignment, and GPU batch tensor stepper (`batch_multi_agent_cbf_step`) (`experimental`). |
| Next-Concept Modeling (NCM) Structural Proposer | [`apedp_dyn/proposers/ncm_structural_proposer.py`](../apedp_dyn/proposers/ncm_structural_proposer.py) | `tests/test_dyn_ncm_proposer.py` | Verified (5/5 unit tests passing): Two-tier physical equation discovery decoupling qualitative macro-concept screening from micro-parameter estimation with coefficient sparsity thresholding and exact parameter recovery on damped oscillator dynamics (`experimental`). |
| Micro-Sample Referee Remediation (patched monotonicity engine + three-outcome decision protocol) | [`benchmarks/run_rime_microsample_adjudication.py`](../benchmarks/run_rime_microsample_adjudication.py), [`benchmarks/rime_microsample_adjudication_results.json`](../benchmarks/rime_microsample_adjudication_results.json) | `tests/test_dyn_rime_microsample_adjudication.py` | Verified (11/11 tests passing, run exit 0): remediation of two defects surfaced by an executed external-reconciliation double-check of the APEDP-X micro-sample claims (RimeProposer `min_rows=30` safeguard verified at source; top proposal `+37.7147*(sat_norm(p)) +3.97006`, `dev_r2=0.9935`, on Langmuir 1918 N=11 with `min_rows=5`; 8-scale bank LOOCV RMSE 0.404810 vs continuous 0.408720, Wilcoxon W=22.0 p=0.3652 — p does NOT establish statistical equivalence on N=11 and the record says so). DEFECT-1: the external `sound_continuous_monotonicity_proof` filtered positive roots by an absolute 1e-9 tolerance, wrongly certifying dN/dp = p·(p − 5e-10) (negative on (0, 5e-10)) as monotonic; `sound_monotonicity_v2` uses exact `real_roots` isolation with NO tolerance filter, interval partitions with 40-digit probes, and an UNRESOLVED fallback for non-algebraic expressions. DEFECT-2: the external negative-control decision `all(p.parameters.get("scale", 0) < 0 for p in proposals)` is vacuously True on an empty packet, classifying an abstention as a dynamics-inversion rejection; `classify_referee_decision` separates ABSTAIN_NO_HYPOTHESIS_GENERATED (empty packet, asserted by regression test) from REJECT (gate failure) from PROMOTE. MEASURED (was prose): Michaelis-Menten N=12 multi-scale origin LOOCV RMSE 0.1740 with 12/12 folds admissible under the exact family certificate q′(p) = w·s/(s+p)² > 0 (w > 0, s > 0); decay control emits 4 proposals (NOT the claimed 5 — composition is host/kernel-version-bound and re-measured, all negative-scale → REJECTED); noise control abstains (0 proposals); empty packet → ABSTAIN. The claimed "10.2× / 0.022 ms" timing ratio is NOT reproduced in either protocol scope on this host (scalar-dot 153×, full 8-scale fold 1.5×) and is therefore recorded as host-disclosed raw timings only, never as a head-to-head ratio. Cargo kernel lib suite independently re-measured 186/186 passing; full `cargo test` 204 passed / 2 ignored. | `audit-only` |
| Fold-Level RimeProposer Recovery/Abstention Record (per-fold proposer execution, both disclosed modes) | [`benchmarks/run_rime_microsample_adjudication.py`](../benchmarks/run_rime_microsample_adjudication.py), [`benchmarks/rime_microsample_adjudication_results.json`](../benchmarks/rime_microsample_adjudication_results.json) | `tests/test_dyn_rime_microsample_adjudication.py` | Verified (run exit 0; pins inside the 14-test file): RimeProposer executed inside each of the 11 Langmuir LOOCV folds in two disclosed modes. (1) BANK-RESTRICTED — the Python-built `sat_m` bank ($s_m = m\cdot\mathrm{std}(p_{train})$, $\mathcal{M}_8$) passed as RIME's feature set: LOOCV RMSE **0.482436** with the Rust engine autonomously selecting `sat_0.8` ×1, `sat_0.7` ×1, `sat_0.6` ×2, `sat_0.66` ×7 (counts frozen), zero abstentions, every fold's formula/scale/intercept/dev_r²/held-out error recorded, and a per-fold partial-gate decision recorded (capacity_positive gate set = disclosed subset; graph-structural validation NOT wired in this record). (2) UNRESTRICTED-GRAMMAR — raw `p` plus the declared algebraic repertoire with kernel-derived transform scales (`_specs` nanstd on the development rows): the kernel **re-selects a saturation family autonomously in 11/11 folds** (dev_r² 0.9924–0.9955) at the measured cost **0.629997** — the honest separation: structural recovery holds unrestrictedly, the tuned bank buys the predictive performance. A `disclosed_limit` field pins that the two modes are NOT the same experiment. All previously pinned results (0.404810/0.408720, W=22.0/p=0.3652, MM 12/12, negative controls) re-verified green in the same artifact run. | `audit-only` |
| Structure-Aware Referee Patch (declared-structure validation; defect-3 closure) | [`benchmarks/run_rime_microsample_adjudication.py`](../benchmarks/run_rime_microsample_adjudication.py), [`benchmarks/rime_microsample_adjudication_results.json`](../benchmarks/rime_microsample_adjudication_results.json) | `tests/test_dyn_rime_microsample_adjudication.py` | Verified (run exit 0; 5 new pins, 19 total): fixes the external referee's dual defect — it PROMOTED a physically unstructured counterfeit quadratic (w=5, \|b\|=0.5, positive data correlation) while REJECTING the tournament's own verified sat_norm winner through a \|b\|≤1.0 tolerance. `adjudicate_structure_aware` reads structure from the declared formula family (NOT inferred from intercept magnitude), applies the exact certificate matching that structure (dq/dp = w·s/(s+p)² > 0 for ANY intercept b — intercept magnitude never gates), enforces gates {positive capacity, finite positive saturation q∞ = w+b, exact monotonicity}, and reports the origin invariant q(0) = b with an explicit `origin_strict_satisfied = (b == 0)` flag instead of a silent \|b\| tolerance. Executed outcomes: counterfeit `+5.0·(sq(p)) +0.5` → REJECTED on structure; verified sat_norm discovery (w=37.7147, b=3.97006, s=dev-scale) → PROMOTED with q(0) deviation DISCLOSED (origin_strict False); saturating negative-scale candidate → REJECTED_QUALITATIVE_DYNAMICS_INVERSION; b≡0 baseline → PROMOTED origin_strict True; and the 11 bank-restricted fold proposals re-adjudicate **11/11 PROMOTED as verified structures, 0/11 origin-strict** — the honest joint claim (all laws are verified saturating structures; only the b≡0 baseline satisfies the strict physical origin) is now observable from a single frozen artifact. | `audit-only` |
| Graph-Equation Structural Validation wired into the fold referee (metadata-first + executable affine-readout verification) | [`benchmarks/run_rime_microsample_adjudication.py`](../benchmarks/run_rime_microsample_adjudication.py), [`benchmarks/rime_microsample_adjudication_results.json`](../benchmarks/rime_microsample_adjudication_results.json) | `tests/test_dyn_rime_microsample_adjudication.py` | Verified (run exit 0; 3 new pins, 22 total; `run_graph_validated_fold_referee` compiled): the reviewer's prescribed structural check is now enforced from the kernel's executable-graph metadata, not proxies: `validate_kernel_graph_structure` requires (1) `graph_id` provenance, (2) the canonical kernel `equation` (AUTHORITATIVE token, not the rendered formula), (3) single declared input resolvable two ways — declared-column mode (equation IS the bank column, e.g. `sat_0.8`) or kernel-transform mode (equation `T(base)` resolving against `algebraic_transforms` × `inputs_used_idx` × `declared_nominal_dependencies`; note the kernel emits EMPTY `inputs_used_names` for transform columns — spec indices are the real signal), (4) independent feature reconstruction of the column/transform, and (5) EXECUTABLE AFFINE-READOUT VERIFICATION — `compute_score == w·feature + b` on the development rows within 1e-9, i.e. the graph implements exactly the declared saturating affine candidate with no hidden operation (measured max deviation: **0.0**, machine-exact). Re-run of the 11 Langmuir LOOCV folds through the kernel with this referee: **11/11 PROMOTED, 11/11 structure-validated, zero abstentions, per-fold graph id/equation/gates recorded**. Validator has teeth, proven by a tampered-graph test: metadata checks hold while `compute_score` deviates by 1e-3 from the affine readout → `validated=False`, `failed_check=affine_readout_verified`. The transform-mode validation additionally proves the full-grammar tournament winner (`sat_norm(p)`, kernel-estimated scale `_apply_transform` reconstruction) validates the same way. | `audit-only` |
| Michaelis-Menten pathway through the same graph-validated referee (last missing referee-path row closed) | [`benchmarks/run_rime_microsample_adjudication.py`](../benchmarks/run_rime_microsample_adjudication.py), [`benchmarks/rime_microsample_adjudication_results.json`](../benchmarks/rime_microsample_adjudication_results.json) | `tests/test_dyn_rime_microsample_adjudication.py` | Verified (run exit 0; 2 new pins, 24 total; `run_michaelis_menten_graph_validated` exposed): the MM system (N=12, S-grid, v = 10·S/(2.5+S) + ε, same dataset as the measured baseline) now executes the actual RimeProposer inside each LOOCV fold over the Python-built MM saturating bank (sat_m, m·std(S_train), $\mathcal{M}_6$) through the SAME pipeline as Langmuir — kernel metadata graph validation + executable affine-readout verification (measured max deviation **0.0**, machine-exact) + structure-aware gates with declared s_m. Measured: **12/12 folds PROMOTED, 12/12 structure-validated, zero abstentions**; actual-RIME LOOCV RMSE **0.1316**, which BEATS the b≡0 baseline's 0.1740 on this dataset (intercept freedom helps when the origin constraint is not the law's defining feature); readout w recovers Vmax = 10.0 within 5% (mean 9.954); per-fold b deviations disclosed (origin_strict False throughout, consistent with the Langmuir record's semantic split). The full-data tournament over the same bank reproduces the external "MM: 6 proposals PROMOTED" table row with measured semantics: exactly 6 proposals, all 6 admissible, saturation family at rank #1. | `audit-only` |


---

## 18. Cloud, Notebook & Epistemic Storage Architecture

### 18.1 Platform Suitability Analysis: Google Colab vs. Dedicated Scientific Infrastructure

| Evaluation Dimension | Dedicated Workstation / Cluster (Slurm) | Google Colab (Free / Pro) | Kaggle Notebooks | Local JupyterLab |
| :--- | :--- | :--- | :--- | :--- |
| **Epistemic Holdout Persistence** | **Optimal** (Persistent NVMe / ZFS; zero risk of holdout rebirth) | **High Risk** (Ephemeral `/content`; erased on disconnect unless Drive mounted) | **Medium Risk** (Ephemeral `/kaggle/working` unless committed/exported) | **Good** (Saved to local filesystem) |
| **Native Rust Kernel (`apedp_kernel`)** | **Optimal** (Pre-compiled with AVX2/AVX-512 vectorization) | **Requires Setup** (Needs precompiled wheel or 3–7 min cold-start build) | **Requires Setup** (No internet in competition mode; needs wheel dataset) | **Optimal** (Built once via `maturin develop`) |
| **System Memory & Ceilings** | **Optimal** (64 GB – 512 GB+ ECC RAM; no arbitrary kill) | **Constrained** (~12 GB free tier; aggressive OOM killer) | **Moderate** (16–30 GB RAM; 12-hour session limit) | **Hardware Dependent** (Local RAM) |
| **Process Model & Concurrency** | **Optimal** (`fork`, `spawn`, Slurm array jobs, Ray) | **Fragile** (Jupyter ZeroMQ loops risk fork deadlocks; max 2 vCPUs) | **Moderate** (4 vCPUs; caution on CUDA fork) | **Moderate** (Caution on interactive fork) |
| **Recommended Use Case** | **Production Discovery & Refereeing** (Full tournament runs, DMRG, 100D PDEs) | **Interactive Exploration & Quickstarts** (Demos, visualizations, teaching) | **Competitive Exploration** (Tabular / signal exploration, offline commits) | **Local Prototyping** (Ad-hoc hypothesis debugging) |

**Conclusion**: Google Colab is **not** the optimal platform for running autonomous production discovery tournaments or final epistemic referee certification due to its ephemeral filesystem (which endangers the AGENTS.md §1.2 anti-snooping holdout registry) and limited CPU/RAM ceilings. However, with `apedp_dyn/notebook_adapter.py`, Google Colab, Kaggle, and Jupyter are fully supported first-class environments for interactive exploration, model evaluation, and demonstrations.

### 18.2 Anti-Snooping Epistemic Safeguards on Ephemeral Cloud Runtimes
Under repository rule **AGENTS.md §1.2 ("Anti-Snooping Holdout Discipline")**, once a candidate hypothesis fails out-of-sample confirmation, that `(dataset_id, slice_id)` slice is **permanently burned** in `FullLineageExperimentLedger`. 

When running on Google Colab or Kaggle:
1. If the ledger is saved to `/content/` or `/kaggle/temp`, restarting the runtime erases the list of burned holdouts.
2. A researcher or script might then evaluate new hypotheses on previously burned holdouts, causing catastrophic data snooping and false scientific claims.
3. **The Adapter Remedy**: `apedp_dyn.notebook_adapter` detects ephemeral cloud storage and:
   - Emits an epistemic warning banner alerting the user.
   - Automatically checks for mounted Google Drive (`/content/drive/MyDrive/apedp_storage`) and redirects the holdout ledger there.
   - Provides `FullLineageExperimentLedger.from_env()` to ensure transparent synchronization to persistent storage.

### 18.3 Usage & Configuration

```python
# In a Google Colab or Kaggle Notebook:
from apedp_dyn.notebook_adapter import init_environment, mount_google_drive

# 1. Mount Google Drive if on Colab to ensure persistent holdout ledgers
mount_google_drive()

# 2. Universal 1-line environment configuration
env_cfg = init_environment()

# 3. Access temporary and ledger directories
print(env_cfg.summary())

# 4. Scoped temporary storage for isolated experiments
from apedp_dyn.env import temporary_storage_scope
with temporary_storage_scope() as temp_dir:
    # Any heavy scratch files written here are automatically purged on context exit
    scratch_csv = temp_dir / "simulated_data.csv"
```

---

## 19. Operational Footprint & Automated Lifecycle Pruning

### 19.1 Background & Threat Model
As an autonomous scientific discovery platform performing extensive symbolic search, Rust kernel compilation, theorem-prover verification, and neural simulations, APEDP-X accumulates massive local build and toolchain bloat:
- **Cargo Build Targets**: Incremental compilation artifacts and build caches exceeding 4.7 GB in `target/`.
- **Compiler Toolchains**: Auxiliary Lean 4 (elan) toolchains provisioned for independent reproduction (e.g. `v4.34.1` alongside `v4.35.0`), consuming ~3 GB each.
- **Package Manager Caches**: Multi-gigabyte caches across `uv`, `bun`, `npm`/`_npx`, and `pip`.

Without automated lifecycle pruning, host filesystems reach 100% saturation, blocking experiment execution, CI gates, and model checkpointing.

### 19.2 Architecture (`apedp_dyn/operational/lifecycle_pruner.py`)
The `lifecycle_pruner` module provides a production-grade operational storage guard:
1. **Deterministic Storage Audit (`StorageFootprint`)**: Scans root filesystem capacity, Cargo targets, active vs. auxiliary Lean toolchains, and package caches without side effects.
2. **Lifecycle Policies (`LifecyclePolicy`)**:
   - `max_disk_usage_percent`: Warning/pruning trigger (default: 85.0%).
   - `min_disk_free_gb`: Minimum free headroom floor (default: 10.0 GB).
   - `max_cargo_target_gb`: Target directory ceiling (default: 2.0 GB).
3. **Safe Pruning (`prune_lifecycle`)**:
   - Idempotent cleanup of build targets, auxiliary toolchains, and package caches.
   - Preserves active toolchains, user source code, database volumes, and persistent ledgers.
   - Full dry-run preview mode (`--dry-run`).
4. **Pre-flight Automated Guard (`auto_guard_storage`)**:
   - Integrated into `scripts/check.sh` and CI pipelines to self-heal storage before heavy regression suites run.

### 19.3 CLI Usage

```bash
# 1. Audit storage footprint and check headroom
bash scripts/lifecycle_prune.sh --status
# Or via python:
python -m apedp_dyn.operational.lifecycle_pruner --status

# 2. Dry-run simulation (zero mutations)
python -m apedp_dyn.operational.lifecycle_pruner --dry-run

# 3. Execute lifecycle pruning
bash scripts/lifecycle_prune.sh --prune

# 4. Automated pre-flight guard check (auto-prunes only if threshold breached)
bash scripts/lifecycle_prune.sh --guard --max-usage 90 --min-free 10

# 5. Machine-readable JSON output for telemetry
python -m apedp_dyn.operational.lifecycle_pruner --status --json
```

---

---

## 23. Quantum-Compilation Frontier, Champion Reproduction & Photonic Boundary

Section preserved from the 2026-10-26/28 additions (stability selection, DQI
photonic resource mapping, 89Q champion reproduction, mixed polarity,
photonic boundary). The 2,134-line module registry above is intact.

### Stability Selection

**Module:** `apedp.stability_selection`  
**Status:** `experimental`  
**Basis:** Meinshausen & Bühlmann-style selection-frequency filter over repeated fits.

**What it exposes:**
- `stability_select(fit_fn, ..., k=20, pi=0.9, seed_fn=None, **kwargs)`

  Runs `fit_fn` `k` times with optional per-call seeding, records how often each
  term/feature survives the fitter's own selection/thresholding, keeps those
  with selection frequency >= `pi`, and returns the average coefficient over
  the kept draws.

- Returns a `dict[str, float]`; empty dict if every call fails.

**Intended use case:**
- Sparse regression / symbolic fitters that are *stochastic* (random subsampling,
  random test-function draws, random initialization). For deterministic fitters
  the utility does nothing useful; that is by design, not a bug.

**Known limitations (disclosed, not hidden):**
- It reduces but does not remove spurious terms on hard cases. In the blind
  PDE benchmark weak-form fitter (workspace `/tmp/buffy/blind/`) it improved
  benchmark recovery, but heat under higher noise still failed.
- Coefficients are simple averages over kept draws; no refit after selection.
- It is an *unverified helper* in the core repo right now: there is no
  production caller pinned to it yet. The blind-benchmark fitter path that
  uses it lives in `/tmp/buffy/blind/`, not in the repo core.

**Integrity note:**
- The blind-benchmark usage (`stability_fit.py`, choose_pi=0.9) was validated on
  VALID first and frozen before BENCH. That workflow lives outside this repo.
- The module itself has no repo integration test at this time; adding one is the
  next honest step if this is to graduate from experimental.

### DQI Photonic Resource Mapping (sFFCC)

| Module | Purpose | Status |
|---|---|---|
| [`apedp_dyn/quantum/dqi_photonic_resource.py`](../apedp_dyn/quantum/dqi_photonic_resource.py) | DQI decoder workload priced on the sFFCC photonic lattice | `experimental` |

**Module:** `apedp_dyn/quantum/dqi_photonic_resource`
**Status:** `experimental`
**Basis:** Maps the DQI decoder's reversible arithmetic onto the sFFCC
deterministic-photonics hardware formulas in `apedp_dyn.quantum.sparrow_photonics`
(arXiv:2507.16152), rather than adopting an external photonic AI accelerator.

**What it exposes:**
- `DQIWorkload(label, qubits, mean_toffoli_like, eea_iterations, field_degree, ...)`
  and the presets `DQI_REFERENCE_WORKLOAD` (sealed receipt) and
  `DQI_GLUZDOV_V6_TARGET_WORKLOAD` (analytical target, no receipt).
- `DQIPhotonicResourceMapper(hardware, assumptions, thresholds).map_workload(workload, r_overhead=...)`
  returning per-patch resources, hardware totals, timing, EEA arithmetic
  decomposition, the 8-channel fault-tolerance audit, and a score-invariance block.
- `compare_workloads(reference, candidate)` which diffs two workloads on the
  official score and on the photonic bill, and lists reasons to distrust the
  candidate row.
- `SuperconductingBaseline(logical_cycle_seconds, physical_qubits_per_logical)`
  and `compare_platforms(workload, baseline)` which compare wall-clock and
  hardware composition against a *caller-stated* superconducting reference.
  The baseline dataclass deliberately has no defaults: this repository does not
  model a superconducting machine, so the numbers must be supplied and are
  echoed back in `assumptions_disclosure`.
- `map_reference_workload(code_distance=...)` convenience wrapper.
- `cycle_basis` on `DQIPhotonicMappingAssumptions` selects the paper's Fig. 12(a)
  cycle (`paper_table_iv`, the default: 209 τ_echo at L=3 plus 107.5 τ_echo per unit
  L, digitised from Fig. 12(a) — 46.59 µs at L=16) or the optimistic
  `floor_12L_tau_rep`; `timing` reports
  both and their ratio, so a wall-clock figure always names its basis.
- `wall_clock_sensitivity()` sweeps cycle basis × τ_echo × code distance, with the
  decoder throughput each point implies.

**Platform comparison at L=16 (reference workload, 1 µs superconducting cycle,
1,000 physical qubits per logical):**

| Quantity | sFFCC photonic | Superconducting (stated) |
|---|---|---|
| Logical cycle (paper Fig. 12a) | 46.59 µs | 1.0 µs |
| Wall clock per shot (5,725,109 cycles) | ≈266.7 s | ≈5.73 s |
| Physical devices / qubits | 24,715,008 emitters + detectors + active elements + fibre EOMs | 1,893,000 physical qubits |
| Devices per logical qubit | ≈13,056 | 1,000 |
| Decoder throughput required | ≈3.00 × 10¹² fusion outcomes/s | not stated |

The photonic bill is **slower** at this operating point (≈46.6× the stated
superconducting cycle on the paper's Fig. 12(a) basis, or 2.65× on the optimistic
12Lτ_rep floor) while carrying two orders of magnitude more physical
devices per logical qubit. That is not an argument against the architecture in
general — it is what this model says for this workload and these assumptions,
and it is why the module reports composition rather than a verdict.

**Reference row at L=16 (Q=1,893, T_mean=5,725,108.38, 222 EEA iterations):**
- 1,536 emitters / 4,608 detectors / 6,144 active elements per logical patch
  (2,241 active elements saved by EBF), 73,728 encoded fusions per cycle.
- Totals: 2,907,648 emitters, 8,722,944 detectors, 11,630,592 active elements,
  139,567,104 fusions per cycle, ~266.7 s wall-clock per shot at the paper's
  46.59 µs cycle (≈15.2 s on the 12Lτ_rep floor); decoding must sustain
  ≈3.00 × 10¹² fusion outcomes/s to keep that cycle honest.

**Central finding (falsifiable, and the reason this is documented):**
- The Benchmark II score is `R(eps_plus) * Q^2 * T_mean`. It has **no timing
  term**, so it is identical at every code distance (verified for L = 8, 16, 24
  in `tests/test_dyn_dqi_photonic_resource.py`). A photonic platform can change
  wall-clock and hardware composition; it cannot change the competition score
  unless it changes `Q` or `T_mean`.

**Known limitations (disclosed, not hidden):**
- One logical cycle is assumed to execute one Toffoli-like operation; the
  blueprint does not certify a per-gate cycle budget.
- Loss handling is the naive retry factor `1 / eta^2 = 1 / 0.8649`; no fusion
  recycling or loss-correction overhead is modelled.
- `Q` and `T_mean` are inputs. The V6 row is a target, and this module will not
  report a gain for it without a compiled circuit and an official receipt.

**Fault-tolerance note:**
- For the shipped default `SparrowHardwareSpec`, all eight Table 2 channels pass
  individually, but the consolidated normalized noise-budget load is ~4.97, so
  `SparrowFaultToleranceReferee` withholds `is_fault_tolerant`. The mapper
  surfaces both numbers rather than only the per-channel verdicts.

### 89Q invsqrt Champion Reproduction

| Script | Purpose | Status |
|---|---|---|
| [`scripts/test_shared_pddd_circuit.py`](../scripts/test_shared_pddd_circuit.py) | Rebuilds the 89-qubit shared-PDDD invsqrt circuit and runs the official verifier on it | `audit-only` |
| [`scripts/measure_invsqrt_coarse_accumulator.py`](../scripts/measure_invsqrt_coarse_accumulator.py) | Sweeps the invsqrt champion's phase accumulator from its 21 bits down to 6 and reports the worst `error/tol` at every width over all 3,133,452 inputs, against the verifier's own target and tolerance. The control reproduces the published model ratio to **2.7e-9** (0.9628198917 vs 0.9628198890), and the sweep answers the width question **before** any cost is counted: an 8-bit coarse accumulator reads **1,023.026× tolerance** as shipped and **511.961×** with `K0` re-laid-out on the coarse lattice, where the built circuit is 0.9996227435. Nothing below 20 bits is admissible — one bit is free because `gy` already discards it, and two bits put **78** inputs outside tolerance at 1.076974. Record: `benchmarks/quantum_compilation/invsqrt_coarse_accumulator.json` | `audit-only` |
| [`scripts/search_claims_pddd.c`](../scripts/search_claims_pddd.c) | Exhaustive search for the shared multi-output PDDD parameters that script consumes | `audit-only` |
| [`scripts/search_mixed_polarity_pddd.py`](../scripts/search_mixed_polarity_pddd.py) | Mixed-polarity search for the four-output restoration: 26 nodes vs 28 | `audit-only` |
| [`scripts/build_mixed_polarity_candidate.py`](../scripts/build_mixed_polarity_candidate.py) | Builds and verifies that plan as a circuit; finds no static gate saving | `audit-only` |
| [`tests/test_dyn_invsqrt_champion_rebuild.py`](../tests/test_dyn_invsqrt_champion_rebuild.py) | Pins the champion rebuild circuit hash and stage timings | `audit-only` |

**Mixed polarity:** `scripts/search_mixed_polarity_pddd.py` finds a 26-node curve
restoration plan (28 in the incumbent) by letting each AND node choose its control
literal; `scripts/build_mixed_polarity_candidate.py` shows the saving does not reach
the static gate count (1466 either way) and that the residual score difference
shrinks with the shot budget. See §5.1 of the boundary doc.

**Rules for a "better circuit" claim:** [`docs/CIRCUIT_OPTIMISATION_RULES.md`](CIRCUIT_OPTIMISATION_RULES.md)
fixes the four things such a claim depends on — executed versus static counting, unitary
preservation on every input (dead-work gating, measured uncomputation, and the
inadmissible measured-operand class), the falsifier each measurement must ship, and the
instance discipline — and records the measured ceiling: the built-in GF(2¹²) multiply is
Karatsuba to the gate (63 = `K(12)`, no split of the recursion beats it), it is **59.48%**
of the six-lever arm (788,250.0 of 1,325,230.0 per shot over 10,510 calls), so a 45-AND
multiply gives 5.039× and one with no ANDs 8.634× — still 1.49× the 445,229 target — while
a static peephole removes 0 gates and the sixth lever changes the static count by 0. The
base-case search beats the recursion: `scripts/gf54_mul.py` emits a **54-AND** multiply and
`scripts/measure_gf54_mul_kickmix.py` measures the arm at **1,230,640.0 per shot, 4.652×**
(1,325,230.0 and 4.320× with the built-in one), a 7.14% saving rather than the 12.3% a
whole-cost 54/63 scaling predicted.

**The multiply as an opcode, not a mixin.** That first measurement reached one instruction:
a Python mixin can only replace `ixor_gf2_mul`, while the five products of the Itoh–Tsujii
inversion and the quotient sweep's are emitted by native generators. `gen_gf54_mul` — added
to the vendored sources by [`scripts/kickmix_mr_opcode.patch`](../scripts/kickmix_mr_opcode.patch)
and selected with `builder.gf_product = 'and54'` — reaches all three, and
`scripts/measure_gf54_native_opcode.py` prices it (record:
`benchmarks/quantum_compilation/gf54_native_opcode.json`): the plain multiply **63 → 54**
Toffolis, the controlled call **75 → 66**, the inversion **315 → 270**, the division
**378 → 324**, all correct on random basis states (values, restored operands, clean
workspace, untouched phase), at **42** workspace qubits per multiply against the built-in 12.
At the pinned instance over 64 shots the **reference** decoder itself goes
**5,725,071.6 → 5,201,222.5 per shot (9.15%)** with `Q` 1,893 → 1,912, and the six-lever arm
**1,325,230.0 → 1,221,466.8** — 4.687× against the unchanged reference (the scored
baseline), 4.258× against a reference given the same product policy, and −7.83% on the arm
itself, with all five comparisons itemised in
[`docs/BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) §1.1 — below
the 1,230,640.0 the mixin
reached, because the opcode also reaches the inversion.

**What is left is not an opcode.** [`docs/BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md)
adds the measured terms up: `T = 472,768.2 + 10,510 × (ands + 12) + 1,019.23 × ands` — the
1,019.23 being the products inside the arm's inversion and division, which carry no
controlled add. So the arm is 2.744× the 445,229 target, every product at the 23-AND
dimension bound still lands at 1.941×, and freeing every product leaves 1.345×; only if the
controlled adds were free too does the rest (472,768.2) approach the target, at 1.062×. **That last
row is withdrawn for the arm that was actually built**: 472,768.2 is the six-lever residual, and
the campaign arm's own is 299,953.2, so freeing its products scores 1.3076e12 (**0.688×** the
denominator) and the target sits 145,275.8 above this arm's floor rather than below it. What
keeps it out of reach is the product stream — 386,061.7 must fall to 145,275.8, about 15.6 ANDs
per multiply against a `2n − 1 = 23` lower bound — and at that bound the arm is 1.019× at 1,913,
0.998× only at 1,893. The bound is no longer a bracket: `gf_mul_per_call_composition.json`
measures a 63-Toffoli call as 63 ANDs and nothing else (the arm emits product spans of 0, 63 or 75
Toffolis and no other size), so the 12-Toffoli `c_gf_iadd` belongs to the materialising shape and
no AND count removes it — that shape reads 1.164×. The withdrawn 1.128× row charged 9 such
Toffolis per call. §7.13 of that note carries the correction, `product_axis_decision()` the
arithmetic. The
Fourier/cyclotomic class is refuted there by a count: the
Chien search's 69 constant multiplies per position are CNOT-only (0 Toffoli in
`dqi_primitive_costs.json`), so a transform can only trade CNOTs and qubits.

**Reconciled with the official row.** [`docs/DQI_RESOURCE_RECONCILIATION.md`](DQI_RESOURCE_RECONCILIATION.md)
ties the three figures together:the locked official reference (1,893 qubits, **5,725,108.382109375**) is a mean over the verifier's **102,400** shots — proved from the number's own fractional part
(it is not a 64-shot mean) — while this project's 5,725,071.609375 is a 64-shot mean. The
pinned competition binary and the local `MR` rebuild produce **identical per-shot Toffoli
arrays** on the same sampled syndromes, so the difference (36.77, 0.78 standard errors) is the
sample, not the metric or the build. The arm itself cannot be compiled for the pinned binary —
it raises `AttributeError: 'Builder' object has no attribute 'mr'` — and `contracts.py` pins
receipts to that binary, so 1,221,466.8 is a local measurement, not a submission result.

**The reconciled ledger.** [`docs/NONMULTIPLY_LEDGER.md`](NONMULTIPLY_LEDGER.md) assigns every
executed Toffoli to exactly one category (693,660.0 multiply + 8,382.7 inversion products +
46,656.0 division products + 472,768.2 decoder logic = 1,221,466.8), splits the residual as
far as instrumentation allows — the six unconditional regions are exactly 94,470 (20.0%)
and the conditional EEA body is a **named 378,298.2 bucket (80.0%)** rather than an estimate
— and records the unit prices, the measurement gaps and the first experiment's promotion
gates. It also fixes what "beat the benchmark" means: the official `R(ε_+)·Q²·T` score on
the verifier's 102,400-shot sample, not the 4.59× local `Q²T` ratio.

**The measured plan for what is left.** [`docs/EEA_AND_PREDICATE_PLAN.md`](EEA_AND_PREDICATE_PLAN.md)
decomposes the arm into `472,768.2 + 10,510 × 66 + 1,019.23 × 54 = 1,221,466.8` and works
through the residual: the six unconditional regions are exactly **94,470** Toffolis (20.0% of
it, a static ledger that is also their executed count), the EEA costs 17,208 per iteration
and 351 per position by the measured slopes, and the division runs 144 times per shot
(46,656 Toffolis). Its conclusion is arithmetic: the residual alone is 1.062× the target, so
`C ≤ 445,229 − P` needs either zero-cost field arithmetic — impossible, the 23-AND bound
costs 391,292 — or a 5.8% cut in the residual *with* the arithmetic free. Early halting and
asynchronous pointer substitution stay closed until proved, and a deletion is not a
measurement.

**The schedule campaign: −43.84%, and the target is 1.541× away.**
[`docs/EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md) is the campaign the plan's §5
obligation asked for. Every predicate register in this decoder is a computational basis
state, so it can be read (`cx` into the shared ancilla plus `MR`, zero Toffolis, phase
untouched) and the read bit can gate the work it enables — the gadget the six levers were
promoted on, at per-position instead of per-round grain. Declared before the arms were
built, then measured at the pinned instance: the sweep's one-hot quotient selection
(−103,968.0, chain 105,696.0 → 1,728.0), the shift's `do_shift` mask (−65,895.0, chain
131,220.0 → 65,325.0) and the product's `mul` control (−362,637.0, product stream 693,660.0
→ 331,023.0) take the arm **1,221,466.8 → 688,966.8 per shot (1.547× the target)** at `Q`
1,913 and 222 rounds, exactly additive (`532,500.0`), with three falsifiers breaking the arm
and the composed shift/swap arm measured at 0.0. The same record re-measures the prior
session's flag/metadata gate (2,952.0 → 1,218,514.8) and stacks all five in one circuit:
**686,014.8 per shot, −535,452.0, 1.541× the target**, exactly `532,500.0 + 2,952.0`. The
saving is removal rather than relocation — executed instructions fall 14.17% with the
Toffolis and the 30,253 added reads are counted. The cost model this rests on — executed
counts under classical feed-forward — is stated in §5 of that note.

**Experiment C: the exchange cannot be scheduled away, and two campaign arms measured
nothing.** [`docs/DQI_MOVEMENT_SCHEDULE.md`](DQI_MOVEMENT_SCHEDULE.md) asks the one question
the campaign left open — can the 64,224-Toffoli A/B exchange be resolved at *build* time by
reassigning which register plays the `A` role, which would be the only kind of saving the
pinned binary could compile? The rebased arm removes the region and its whole 198,024 static
Toffolis and is **incorrect**: the probe finds the exchanging-round set has **10 distinct
patterns across 64 shots** at an invariant **72** firings per shot, so the decoder's data
movement is a permutation parameterised by the shot and no build-time name assignment can
express it. The same experiment audits the campaign's two 0.0 arms (`composed`,
`drop_shift_c3`): both emit the control's stream — 149,337 instructions, 16,610 static — so
those zeros are **vacuous**, and the campaign's §3.3 correction records it.

**Experiment D: the missing `MR` was not the only blocker, and one candidate compiles.**
[`docs/DQI_OFFICIAL_COMPATIBILITY.md`](DQI_OFFICIAL_COMPATIBILITY.md) mines both builds'
surfaces. The pinned competition binary has no `mr`, no `h`, no `gf_product` and no
per-instruction counts, but it has `hmr`, the classical-condition machinery and a `del_and`
that costs **0** Toffolis — an AND clear can be a measurement where a read cannot. So the arm
needs two surfaces, not one (the 54-AND policy is worth 523,849.1 per shot on the reference
and was never listed as a blocker). A candidate the pinned binary *can* build — the reference
with the 54-AND multiply emitted as gates and its workspace cleared with `hmr` — measures
**Q 1,896 / 5,433,399.609375 per shot**, syndromes restored, a **0.952** score ratio to the
shipped reference and still **7.92×** worse than the arm.

**Experiment E: the competition's own verifier scores it, at the competition's own budget.**
`verify()` from the bundle — every scoring line unmodified, one diagnostic-only stub
(`reaction_depth`, isolated from the verdict by test and carried verbatim in the record's
`verifier_patch`) — on both circuits at **102,400 shots** (400 batches of 256) and
`log2_delta = -128`: the reference passes with **0/102,400 value and 0/102,400 phase
errors** and scores **2.443946e13**; the reduced-window 54-AND candidate passes with **0 and 0**
and scores **2.319438e13** — a ratio of **0.949054**, exactly its Toffoli ratio at equal `Q`,
because both runs share the same zero-error `eps_plus` (8.660587e-4) and therefore the same
`R` (1.191260). The decomposition is to the gate: 32,408 multiply calls × 9 =
**291,672 static Toffolis**, exactly the measured static delta (5,725,461 → 5,433,789) and
291,670.152 on the executed means; the multiply is **42.45%** of the reference's stream and the
candidate removes **5.0943%** of it. The budget is load-bearing — below ~1,375 shots a
*zero-error* run is refused (`eps_plus >= 1/16`), and the smallest batch-aligned budget that
passes is 1,536. The levers' obstacle is measured on the surface rather than assumed: of the
**29** pinned methods whose signature mentions a bit, exactly **one** writes one from qubit
data (`hmr`), and on `|1>` it wrote 1 on 24/64 shots — a measurement, not a read. What is left
is arithmetic honesty: 2.319438e13 is still **13.66×** above the 1.6975e12 portfolio target,
and the remaining factor is the schedule, i.e. `mr`.

**Experiment F: the arm's score, measured instead of multiplied out.** The other half of the
frontier had only ever been an arithmetic product. Built on the local `MR` rebuild
(`scripts/build_kickmix_mr.py`) and run through the same competition `verify()` at **102,400
shots**, the six-lever arm **passes with 0 value and 0 phase errors**, executes
**1,221,408.044756** Toffolis per shot at `Q` = 1,913 (its recorded 1,221,466.8 to 4.8e-5;
static 3,132,510), and scores **5.324728e12** — **0.2179** of the pinned reference's
2.443946e13 at the same budget, i.e. 4.5898× cheaper, exactly the `Q²T` ratio. It is still
**3.137×** above the 1.6975e12 target, and the gap is the decoder rather than `Q`: at 1,913
the target's Toffoli budget is **509,088** and the arm executes 1,221,408. The record says in
three places that it is **not a receipt** — the pinned binary has no `mr` and
`apedp_dyn/quantum_compilation/contracts.py` is unchanged. Record:
`benchmarks/quantum_compilation/dqi_arm_submission_score.json`.

**Experiment G: the campaign arm, measured — and it is the frontier.** The stacked arm (the
six levers plus the five schedule/control gates, `mr` clears, 54-AND policy) had only ever
been a `Q²T` product. Through the same `verify()` at **102,400 shots** it passes with **0
value and 0 phase errors**, executes **685,936.2051367187** per shot at `Q` = 1,913 (its
recorded 686,014.8 to 1.1e-4; static 3,130,734) and scores **2.990339e12** — **0.12236** of
the pinned reference and **0.5616** of the six-lever arm, i.e. **1.762×** the derived
target once the `Q` term is counted. `Q` is what keeps it out: at 1,913 the target's 445,229
would have to be 509,088. Record: `benchmarks/quantum_compilation/dqi_campaign_arm_submission_score.json`.

**The external-optimizer check, scoped to what a Clifford+T optimizer can express.**
[`scripts/compare_gf_mul_external_optimizer.py`](../scripts/compare_gf_mul_external_optimizer.py)
records the multiply's forward netlist from the emitter itself — **54 ANDs, 1,004 CNOTs, 54
clears at 0 Toffoli** — expands each Toffoli to the standard 7-T block (verified on all
eight basis states, max error 3.1e-16) and runs **PyZX 0.10.7**: its Clifford pass leaves the
T-count at **378 → 378** and only compresses CNOTs (1,328 → 824); its T-reducing
phase-polynomial pipeline **did not return within 90 s** at 99 qubits. A unitary optimizer
must also uncompute, so its honest figure is **108 ANDs = 756 T**, not 54. The decoder is not
comparable at all: its stream carries `HMR` and `INFO_TAG`, and the arm's clears are `mr`.
Record: `benchmarks/quantum_compilation/gf_mul_external_optimizer.json`.

**Correctness evidence is now enforced at runtime.** `CircuitCandidate` carries
`correctness_evidence` and `with_status` raises `EvidenceIncompleteError` for a
correctness-claiming status (the two `VERIFIED_*` ones) unless value, junk and phase are each
measured-and-passed, covered by another passed check, or `not_applicable` with a reason the
verifier supports. `ledger.record_verification` derives the evidence from the receipt and
records `INSUFFICIENT_EVIDENCE` rather than promoting a claim it cannot support — the DQI
verifier folds the ancilla read into `value_errors`, so junk is `covered_by` value with that
reason, while an invsqrt/dfthcsos receipt says in so many words that it prices the circuit and
does not simulate it. This is the checklist's gates 3, 4 and 7 turned into code:
`tests/test_candidate_correctness_evidence.py` (11 tests).

**Closing statement, and the decision it carries.**
[`docs/BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md)
§5 states the frontier in the competition's own unit: the pinned binary's best correct
circuit is 13.66× out and no multiply work closes that (deleting the whole 42.45% share would
leave 8.29×); `MR` moves the reference 4.142× and the arm 4.5898×, and is **necessary but not
sufficient** (3.137× measured, 1.762× with the five-gate campaign arm, both through
`verify()`); what remains is a
decoder algorithm — 240,785.8 Toffolis, predominantly the swap, the shift, `shift_step` and
`update_flag`. The note also lists what would overturn it.

**The reusable method.** [`docs/REPRESENTATION_AWARE_UNCOMPUTATION.md`](REPRESENTATION_AWARE_UNCOMPUTATION.md)
turns the one transform that worked — redesign the intermediate representation so an inverse
primitive's precondition holds, then the clear is free and phase-safe — into a nine-gate
checklist for the synthesizer, with the three measured cases side by side (`del_and` 0 phase
flips, `del_gf2_mul` 66 vs 75 per call at unchanged `Q`, and `hmr` as the counterexample that
wrote 1 on 24/64 shots). Gates 3, 4 and 7 are the correctness deciders, and the note names its
hooks (`CircuitCandidate`, `CandidateStatus`, `referee.run_dqi_verifier`, `contracts.py`).

**The amplitude tier: the licence is exactly the basis.**
[`docs/VQC_AMPLITUDE_TIER.md`](VQC_AMPLITUDE_TIER.md) and
`apedp_dyn/quantum_compilation/statevector_sim.py` compare the controlled gate with the
read-then-gate map on amplitudes: identical on all eight basis inputs of the gadget, a
**complete dephasing** of a superposed control (fidelity 0.500, off-diagonal 0.500 → 0.000).
It also settles Experiment D's substitution: `hmr` alone randomises a basis qubit, while
`h; hmr` is `mr` exactly — the blocker is the missing Hadamard. The arm itself is 10^575.87
amplitudes, which is why the counts stay basis-state counts.

**How much of that is the Forney inversion (measured, not asserted).** The reference runs one
inversion of five products at every one of its 4,095 positions, so the block is 20,475 product
sites of 9 Toffolis: 184,275 static. Priced in isolation — the same reference with the policy
flipped around `init_gf2_inverse_with_scaffold` and `gf2_phase_by_product` only — it executes
**20,469.5 products per shot (4,093.9 positions) and saves 184,225.1 Toffolis, 3.22% of the
reference and 35.2% of its total saving**; the other 339,624.0 (64.8%) is the change-of-basis
division and the quotient sweep. The static and measured numbers differ only because 1.1
positions per shot have no error bit set and skip the block. Every executed product removes
exactly nine, controlled or not, and the two arms' CCX difference counts executed calls
without an ablation.

**Proposed IR instruction:** [`docs/MR_INSTRUCTION_SPEC.md`](MR_INSTRUCTION_SPEC.md)
specifies the Z-basis measure-and-reset opcode that the measured DQI optimisations
(4.142× on `T_mean`, `scripts/probe_dqi_predicate_gating.py`) require, with the patch
points and the four acceptance tests for a rebuild; [`docs/MR_PATCH_REQUEST.md`](MR_PATCH_REQUEST.md)
is the change request itself — PR description, diff summary by component, test plan and
rollback. The instruction is now implemented as a source patch
([`scripts/kickmix_mr_opcode.patch`](../scripts/kickmix_mr_opcode.patch)) over the
vendored sources; [`scripts/build_kickmix_mr.py`](../scripts/build_kickmix_mr.py) builds
a local extension with it and [`scripts/run_mr_gated_arms_kickmix.py`](../scripts/run_mr_gated_arms_kickmix.py)
runs the gated arms on that real simulator (all six levers plus `all` and `full`
reproduce the reference at m=15/GF(2⁴) and m=255/GF(2⁸); both falsifiers fail). The pinned
competition `kickmix.so` still has no `MR`, so the T-count reductions remain oracle
upper bounds and no receipt follows from the local build.

**Module:** `scripts/test_shared_pddd_circuit`
**Status:** `audit-only`
**Basis:** The champion row in
[`docs/invsqrt_frontier_and_portfolio_synthesis.md`](invsqrt_frontier_and_portfolio_synthesis.md)
is a compiled artifact plus a verifier receipt. This script is the only code path
in the repository that builds that circuit, so it is what makes the claim
reproducible rather than merely recorded.

**Procedure:**

```bash
python3 scripts/test_shared_pddd_circuit.py
```

It reads three tracked inputs — `src/boolean_choices90.json` (coarse6 order and
polarity), `src/shared_davio_results.json` (curve4..6 order and polarity, 28
nodes), `src/invsqrt_calibration.json` (the 44 rotation phases) — plus the
`kickmix` extension and the official `invsqrt` verifier under
`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/`. It writes
`/tmp/invsqrt_89q_shared_pddd.kmb` and prints the verifier receipt. It does not
overwrite anything in `artifacts/`.

**What must match on a successful rebuild** (the first three are exact):

| Field | Expected |
|---|---|
| `num_qubits` | 89 |
| `max_toffoli_count` | 1466 |
| `worst_error_over_tolerance` | 0.9996227434818049 |
| `circuit_sha256` | `ce04c9d54dc42e42455444022b6652a6efb0be2630aca7b485d3efd4e58fa9de` |
| `score` | three bases, one circuit: 157,605.5 at the script's 25,600 shots; 157,601.0808 pinned at 200,540,928 shots (the sealed denominator base); 157,601.115479 in `benchmarks/quantum_compilation/invsqrt_kernel_receipt.json`, the same circuit re-scored by the published kernel. Reconciled in `docs/invsqrt_frontier_and_portfolio_synthesis.md` §1.1 "One cost, three bases" |

**Known limitations (disclosed, not hidden):**
- `artifacts/` is gitignored, so a clone has neither the committed `.kmb` nor the
  receipt JSON. The circuit hash above is the reproducibility check that survives
  that: a rebuild either produces it or does not.
- `toffoli_count` is a mean over shots and depends on mid-circuit measurement
  outcomes, so it moves in the fourth significant figure with the shot budget
  (1455.499 at full budget, 1455.549 at 25,600 shots). Only `max_toffoli_count`
  (1466) and the circuit hash are budget-independent.
- The script needs a Python environment with the `kickmix` C++ extension built
  and the official verifier importable; it is not importable under a bare
  `python3 -m pytest` run and is not collected by the test suite.

**Module:** `scripts/search_claims_pddd`
**Status:** `audit-only`
**Basis:** Proos–Zalka / Davio shared-node counting for the four restoration
outputs. The search enumerates all 8! variable orders and 256 polarities and
reports the order with the fewest shared AND nodes.

```bash
gcc -O2 -fopenmp -o /tmp/search_claims_pddd scripts/search_claims_pddd.c
/tmp/search_claims_pddd
```

**Measured output:** Test 1 (`{curve3..curve6}`) 46 shared nodes at order
`[3,4,2,1,0,7,6,5]`, polarity 96; Test 2 (`{coarse6, curve4..6}`) 36 shared nodes
at order `[7,4,3,2,1,0,6,5]`, polarity 32. Test 2's parameters are what
`scripts/test_viable_optimization_path.py` measures and what
`build_shared_plan` in the Python builder reproduces as 36 nodes.

**Known limitations (disclosed, not hidden):**
- The five truth tables in `main()` are hardcoded literals; the file does not
  derive them from the curve tables, so a changed residual split would silently
  leave this search answering the old question.
- The node set is capped at 128 entries with a silent drop past that; harmless at
  the 36/46 counts reported here, wrong for larger output groups.
- The reported optimum is never re-validated by building the resulting DAG; the
  cross-check is the Python builder's independent count, which is what the
  36-node agreement above relies on.

**Related:** [`docs/PHOTONIC_ACCELERATOR_BOUNDARY.md`](PHOTONIC_ACCELERATOR_BOUNDARY.md)
states what an external photonic accelerator may propose versus what only the
exact verifier may accept, with the measured pipeline cost (4.44 s end to end),
the precision wall (3.97e-12 turns of unused margin against ~1e-5 analog
conversion), and the measured ceiling on Boolean node sharing (36 nodes minimal
over all 10,320,960 order/polarity pairs; 1 qubit costs 9.95 nodes).

### DQI degree-register / halting-counter recycling audit chain

| Module | Purpose | Status |
|---|---|---|
| [`scripts/test_recycled_flags_small.py`](../scripts/test_recycled_flags_small.py) | Prototype `ProosAndZalkaDecoder` subclass that recycles the flag and halting-counter comparisons; imported by `scripts/run_halt_recycling_scale.py` | `audit-only` |
| [`scripts/test_recycled_degrees_small.py`](../scripts/test_recycled_degrees_small.py) | Degree-register recycling at m=15/n=6/GF(2⁴); verifier `PASS`, 140 qubits, T_mean 33,229 | `audit-only` |
| [`scripts/test_itoh_tsujii_scaffold.py`](../scripts/test_itoh_tsujii_scaffold.py) | Itoh–Tsujii inversion scaffold and archive-chain schedule used by the dual-scaffold pricing | `audit-only` |
| [`scripts/diagnose_invsqrt_qubit_peak.py`](../scripts/diagnose_invsqrt_qubit_peak.py) | Reports where the 89Q invsqrt champion's qubit count peaks, with the allocation site for every rise. The peak is the 21-bit adder's carry chain in `folded_gadd`, not the Boolean restoration, so the restoration node count does not set `Q` | `audit-only` |
| [`scripts/try_drop_zero_rotations.py`](../scripts/try_drop_zero_rotations.py) | Tests whether the three zero-angle calibration entries (indices 15-17) are paid for. They are not: the rewrite already removes them, and naming them again shifts the index mapping and fails the verifier at x = 1 (0.125 turns). Pins why 44 rotations is the floor | `audit-only` |
| [`scripts/test_dual_scaffold_dqi.py`](../scripts/test_dual_scaffold_dqi.py) | Prices the dual-scaffold option against the incumbent decoder (see `PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.2) | `audit-only` |
| [`scripts/test_viable_optimization_path.py`](../scripts/test_viable_optimization_path.py) | Verifies the 36-node shared-plan parameters that `scripts/search_claims_pddd.c` reports | `audit-only` |
| [`scripts/instrument_invsqrt_trace.py`](../scripts/instrument_invsqrt_trace.py) | Traces per-shot Toffoli/phase statistics behind the invsqrt margin numbers | `audit-only` |
| [`scripts/instrument_calibrated_trace.py`](../scripts/instrument_calibrated_trace.py) | Calibrated-variant trace instrument used for the same margin accounting | `audit-only` |
| [`scripts/measure_dqi_primitive_costs.py`](../scripts/measure_dqi_primitive_costs.py) | Recovers the Benchmark II cost structure by measurement (per-call primitive costs, per-iteration and per-position slopes) and derives the optimisation ceilings; record in `benchmarks/quantum_compilation/dqi_primitive_costs.json` | `audit-only` |
| [`scripts/measure_dqi_scratch_pool.py`](../scripts/measure_dqi_scratch_pool.py) | Measures whether the transient DQI qubits are collectable by a shared scratch pool: peak concurrent liveness by frame, the innermost-frame ceiling on pooling, and an alloc/free trace replay; record in `benchmarks/quantum_compilation/dqi_scratch_pool.json`, pinned by `tests/test_dqi_scratch_pool.py` | `audit-only` |
| [`scripts/verify_local_mr_sim.py`](../scripts/verify_local_mr_sim.py) | Differential check of `local_mr_sim` against `kickmix`: builds the reference decoder twice, feeds both the same syndromes and MBUC outcomes, and requires identical syndrome restoration, `|0>` junk and tracked phase | `audit-only` |
| [`scripts/run_mr_gated_arms.py`](../scripts/run_mr_gated_arms.py) | Runs all six gated DQI levers (`swap`, `sweep`, `forney`, `clear`, `shift`, `quotient`) with their predicates read by a real `mr` (`cx(control, ancilla); mr(ancilla, bit)`, and for the Forney block `mr` straight off `sigma` followed by a classically controlled restore) instead of oracle registers, and requires each arm to reproduce the reference's values and phase at m=15/GF(2⁴), m=255/GF(2⁸) and the official m=4095/GF(2¹²). `--clear-with r` and `--shift-under` are the falsifiers and must fail. Record: `benchmarks/quantum_compilation/local_mr_arms.json`, pinned by `tests/test_local_mr_sim.py` | `audit-only` |
| [`scripts/kickmix_mr_opcode.patch`](../scripts/kickmix_mr_opcode.patch) | Source patch adding `MR`/`MR_IF` (Z-basis measure-and-reset that leaves the tracked phase alone) to the vendored `kickmix` sources, across opcode table, parser, builder, simulator, statistics, reaction depth and printing; it also adds `gen_gf54_mul` (the proved 54-AND GF(2¹²) product, correct at every width by Karatsuba composition of the `mu(2)=3`/`mu(3)=6` base cases), the `GFProduct`/`builder.gf_product` policy that routes `gen_gf_mul` through it, and **per-instruction execution counting**: `sim.counting_op_executions` makes the simulator accumulate, for each instruction, how many shots executed it (`sim.operation_execution_counts()`), using the same condition mask as `operation_statistics()` and a live-shot mask so unused SIMD lanes are not counted; `circuit.info_tags()` and `circuit.op_toffoli_like` expose the `builder.mark(...)` boundaries and the Toffoli-like mask, so a region's executed Toffolis can be summed exactly instead of only by operation type | `audit-only` |
| [`scripts/build_kickmix_mr.py`](../scripts/build_kickmix_mr.py) | Builds the MR-capable `kickmix` from this repository and proves the result is the pinned build: exports the vendored sources with `git archive HEAD` (so uncommitted edits cannot leak in), refuses a tree whose commit or patch SHA-256 is not the one the pin names, compiles with absolute `--build-lib`/`--build-temp` paths, and then requires the extension's SHA-256 to equal `MR_BUILD.sha256` — a mismatch is an error, not a warning. `--compare` hashes and probes other builds so byte-reproducibility across output directories is recorded rather than asserted; `--check` verifies an existing build without compiling. Writes `benchmarks/quantum_compilation/kickmix_mr_build.json` (source commit, patch hash, toolchain, 60-method surface, pin match, comparisons). The pinned competition binary is never touched | `audit-only` |
| [`apedp_dyn/quantum_compilation/kickmix_builds.py`](../apedp_dyn/quantum_compilation/kickmix_builds.py) | The two `kickmix` builds a verifier run may use, named by SHA-256: `shipped` (`c66b3216…`, no `mr`) and `mr` (`36b1e6ef…`, built by `scripts/build_kickmix_mr.py` from commit `638777f` plus patch `9d96fd3a…` with CPython 3.13.14 and pybind11 2.12.0, byte-reproducible across two output directories). `assert_allowed` refuses anything else and gives a different reason for the third, non-allowed build (`e57dcc47…`, compiled from a mutated working tree, not reproducible from the repo, cited by earlier MR receipts). `contracts.py` now binds `kickmix_extension_sha256`/`kickmix_build` into every contract and re-checks them in `assert_locked()`, so the tree hash cannot hide a substituted extension. Pinned by `tests/test_dyn_kickmix_build_pin.py`; receipt `kickmix_mr_build.json` | `audit-only` |
| [`scripts/split_kickmix_patch.py`](../scripts/split_kickmix_patch.py) | Splits the combined `MR` + `gf_product` patch at hunk granularity into `build/kickmix_split/{mr,gf54}.patch` and proves both apply (mr to a clean tree, gf54 on top). File separation is impossible because the `gf_product` enum and property live in files `MR` also edits, so a hunk is classified by whether its changed lines mention the policy. Measured: mr 15 files +382/−7, gf54 6 files +558/−2, and **both compile** — the MR-only build exposes `mr` and no `gf_product`, the stacked build exposes both and defaults to `kronecker_mul` | `audit-only` |
| [`scripts/build_dfthcsos_sub128.py`](../scripts/build_dfthcsos_sub128.py) | Benchmark III angle-circuit builder parameterised on `(brot, J)` instead of fixing them, so the 128-qubit cap can be treated as what it is: a bound on `J*brot`. Reuses the competition's `scoreinfra.circuits` for the register contract, the address-by-address word recovery, the phase and the reverse pass. Measured on the shipped `submission.npz`: `brot=10, J=10` is **123 qubits** (5 spare), `K=8`, **3,974 Toffolis** per circuit and **63,584** over both passes, with all 298,200 words recovered exactly; `brot=12, J=8` fits too at 119 qubits and costs 79,480; `brot=9, J=11` is 122 qubits, `K=7`, 55,636, also exact. Also falsifies two planned levers: sharing the tree's ancillas is rejected by the builder as a qubit collision (parent control == child target, 7,943 times), and early-break has nothing to collapse because all 31,808 delta rows are distinct; fusing the deepest leaf controls into multi-controlled X frees **one** ancilla for ~**46x** the Toffolis (9 of them measured 51→50 qubits at 8.97x on a 64-address slice, and the fused build did not reproduce the table, so the path was removed); and the 9-bit table's break-even is computable without the integrals — it wins while `lambda_eff(9)/lambda_eff(10) >= 0.875`, i.e. it may lose 12.5% of `lambda_eff`, with max rotation perturbation 8.677e-3. Also shows the freed bit is only worth spending on the batch: `brot=9, J=10` is 113 qubits and the same 63584 Toffolis as 10-bit `J=10` (K=8 either way) with double the angle error, while `J=11` removes a circuit for 55636 . Also settles the two-pass question: between forward and reverse of one circuit nothing differs (the checker re-executes the same bytes), so `T` is exactly 2 x sum(circuit Toffolis) = 63584 and there is no tree to share | `audit-only` |
| [`scripts/run_dqi_full_instance_verifier.py`](../scripts/run_dqi_full_instance_verifier.py) | Runs the official Benchmark II verifier on the **full** 4,095-position instance by substituting exactly one line — the `reaction_depth` diagnostic, whose peak is 13.9 GB on 86M instructions and which is not a scoring term — into a copy under `build/`, printing both digests so the substitution is auditable. Measured here: **PASS**, score **2.443946358367293e13**, `T` 5,725,108.38, `Q` 1,893, 0 value / 0 phase errors at 102,400 shots in 3m24s. Record: `benchmarks/quantum_compilation/dqi_full_instance_reverify.json` | `audit-only` |
| [`benchmarks/quantum_compilation/dqi_archived_artifact_reverify.json`](../benchmarks/quantum_compilation/dqi_archived_artifact_reverify.json) | Benchmark II certificate for the decoder staged in `submission_24_59.zip` (`sha256 748b01a7…`, `Q` 891, 58,336,057 ops): official `PASS` at 102,400 shots with 0 value / 0 phase errors, score **3.753741740938e11**, `T` 396,919.33, max 397,192 — **65.11×** below the manifest-pinned 222-iteration decoder's 2.443946e13 (`Q` 1,893) and 65.11× the reference cost. The circuit matches no `.kmb` on disk and is not the pinned build, so it was re-verified from the archived bytes; the only verifier substitution is the `reaction_depth` diagnostic (`patched_sha256 649549c0…`), byte-identical to the one in `dqi_full_instance_reverify.json`. Records 114.46 s of run telemetry and the 256-shot power floor (ε₊ 0.2929), so a small-budget FAIL cannot be mistaken for a defect. Resealing the portfolio on it ($S \approx 9.50$) is stated as an open decision, not taken. | `audit-only` |
| [`benchmarks/quantum_compilation/dfthcsos_archived_artifact_verify.json`](../benchmarks/quantum_compilation/dfthcsos_archived_artifact_verify.json) | Benchmark III certificate for the bundle staged in `submission_24_59.zip` (`dfthcsos/submission.npz` sha256 `ddce9a5b…`; the verifier's own bundle digest is `4c2fe44e…`): official `PASS` at 78,080 shots against the packaged `FeMoCo76o113e.fcidump` (128,128,348 bytes, sha256 `e5a48af7…`, both pinned by `scoreinfra/reference.py`), score **3.114658570666053e8**, $\lambda_{\text{eff}}$ 47.1847, `T` 15,459, `Q` 427, 15 circuits, `brot` 8, `batch_size` 5 — 10.77× below the 3,355,599,588.32 baseline and 6.82× better than the $1.580\times$ target. Neither the 12-bit family nor the `276d464f…` arm, so the archive carries a third Benchmark III design; run took 32.68 s and the FCIDUMP that `DFTHCSOS_HANDOVER.md` once called missing is present and hash-matching. | `audit-only` |
| [`scripts/setup_verifier_env.py`](../scripts/setup_verifier_env.py) | Creates the interpreter the official VQC verifiers need: the only CPython that loads the shipped `kickmix.so` is the `uv`-managed 3.13 build, which is PEP 668 externally managed, so `numpy` and `scipy` go into `build/qvcc-venv313` (same ABI) rather than mutating the toolchain with `--break-system-packages`. `--check` reports without creating. Prints the `PYTHONPATH` line that keeps the pinned `kickmix.so` where `contracts.py` hashes it, and records the two acceptance facts: `--num-shots` must be a multiple of 256, and a run with zero errors can still FAIL, because the gate is `eps_plus` (0.2929 at 256 shots, 0.0561 at 1,536) | `audit-only` |
| [`scripts/run_mr_gated_arms_kickmix.py`](../scripts/run_mr_gated_arms_kickmix.py) | Runs the five MR-gated DQI arms on that rebuilt, MR-capable `kickmix` (real builder, real simulator) and requires them to match the reference's syndromes, tracked phase and cleared scratch; `--clear-with r` and `--shift-under` must fail. Pinned by `tests/test_dyn_mr_gated_arms_kickmix.py` (skips when the extension is not built). Correctness only: no score, no receipt | `audit-only` |
| [`scripts/measure_mr_arm_costs_kickmix.py`](../scripts/measure_mr_arm_costs_kickmix.py) | Prices the reference, the cumulative lever chain and each lever alone with `kickmix`'s conditional-aware per-shot Toffoli counting on the rebuilt extension. At the official instance over 64 shots the reference reproduces 5,725,071.61, the five oracle-gated levers execute 1,381,390.0 (4.144×) against the projected 1,382,207.08 (4.142×) with every stacked stage within 0.06% of its projection, and the sixth lever (the quotient gate) executes 1,325,230.0 (4.320×) at 1,894 qubit ids against the reference's 1,893. Record: `benchmarks/quantum_compilation/kickmix_mr_arm_costs.json` | `audit-only` |
| [`scripts/study_circuit_designer_directions.py`](../scripts/study_circuit_designer_directions.py) | Prices the three remaining ways to improve the DQI decoder and lets the numbers choose: (A) deleting each region no lever touches, which bounds predicate discovery at 2.71% of the emitted stream; (B) an exact bilinear-rank search for a cheaper multiply, which proves `mu(2)=3`, `mu(3)=6` (every five-matrix choice exhausted) and composes to **54 ANDs at GF(2¹²)** against the built-in 63, verified on the field; (C) the ladder `t(ands) = 536,980 + 10,510 × (ands + 12)`, which gives 4.652× at 54 ANDs -- matching the measured arm -- and still leaves 1.49× above 445,229 even with no ANDs at all, so no multiply alone reaches the target. Record: `benchmarks/quantum_compilation/designer_directions.json`; discussion in [`docs/CIRCUIT_OPTIMISATION_RULES.md`](CIRCUIT_OPTIMISATION_RULES.md) §5–§6 | `audit-only` |
| [`scripts/measure_gf_mul_and_static_headroom.py`](../scripts/measure_gf_mul_and_static_headroom.py) | Bounds what is left after the levers: `ixor_gf2_mul` against the Karatsuba AND-count model at six degrees (exact at all six: 63 at GF(2¹²)), the split search at m=12 (no split beats 63; schoolbook 144), static Toffoli for the reference and both gated stacks (5,725,461 → 3,486,597, 1.642×, with the sixth lever changing it by 0), a full-recording peephole (0 removable pairs), and the multiply's share of the executed count by differencing two correct arms (788,250.0 of 1,325,230.0 per shot, **59.48%**, 10,510 calls; a multiply with no ANDs would still leave 663,100.0, so 8.634×). The earlier 86.33% came from deleting the multiply and differencing executed counts, which perturbs the state and so the gating, and is superseded. Record: `benchmarks/quantum_compilation/gf_mul_and_static_headroom.json`; rules in [`docs/CIRCUIT_OPTIMISATION_RULES.md`](CIRCUIT_OPTIMISATION_RULES.md) §5 | `audit-only` |
| [`scripts/measure_gf_mul_per_call_composition.py`](../scripts/measure_gf_mul_per_call_composition.py) | **Takes one multiply call apart**, which is what the 23-AND rung was waiting on. Emits one `ixor_gf2_mul` at GF(2¹²) in each control mode and attributes every Toffoli to the engine's own tag: **63** (`gf_mul`, an in-place accumulate — 63 ANDs and nothing else) or **75** (`gf_mul` 63 + `c_gf_iadd` 12, the materialising shape). Then every product span of the campaign's control arm, so the distribution is on record: spans measure **0, 63 or 75** and no other size, which explains the record's `693,660 / 10,510 = 66.0` average as a mixture (25% of calls carrying the add) and the post-window `331,023 / 5,255 = 63.0`. Retires `MULTIPLY_NON_AND_TOFFOLI_PER_CALL = 9`, which was `ccx_delta_per_call` — a difference between two *implementations*, not a split inside one — and with it the 1.128× row: at the bound the arm is `T` 444,245.3 in place (0.998× the target) and 507,297.3 materialised (1.139×) — both figures re-priced by §7.15 and `measure_division_product_composition.py` to 440,896.4 (0.990×) and 503,948.4 (1.132×) once the division stream's product count was measured. Record: `benchmarks/quantum_compilation/gf_mul_per_call_composition.json`; used by `product_axis_decision()`, read through `load_multiply_call_composition()` (which raises rather than falling back) and pinned by `tests/test_dqi_gluzdov_v6.py` | `audit-only` |
| [`scripts/measure_division_product_composition.py`](../scripts/measure_division_product_composition.py) | **Takes the division bill apart**, which §7.6 reconstructed as `864 products × 54 ANDs` without measuring either factor. Builds one `init_gf2_div_with_scaffold` at GF(2¹²) and attributes every Toffoli to the engine's own tags, span by span: **378 = 6 `gf_mul` calls of 63 Toffolis and nothing else**, and the inverse scaffold is 315 = 5 × 63. So the measured per-shot bill of 46,656 is **740.6 products of 63 ANDs** (123.43 calls × 6), which rebuilds to the bill exactly — the count is 0.857× the assumed one, and a lower AND bound multiplies the count. At the 23-AND bound the division+Forney term is **20,093.5** Toffolis rather than 23,442.4, which moves the in-place row from `T` 444,245.3 (0.998× the target) to **440,896.4 (0.990×)** and the quantum-controlled row from 507,297.3 (1.139×) to 503,948.4 (1.132×). The decision does not move — 1.011× at the arm's own 1,913 still needs the 1,893 peak — but the 984-Toffoli margin it was resting on was this decomposition's artefact. Record: `benchmarks/quantum_compilation/dqi_division_composition.json`; read by `product_axis_decision()` and `model_single_polynomial_array()` through `load_division_product_composition()` (raises rather than falling back) and pinned by `tests/test_dqi_gluzdov_v6.py::test_the_division_bill_is_decomposed_by_measurement` | `audit-only` |
| [`scripts/measure_product_accumulator_shape.py`](../scripts/measure_product_accumulator_shape.py) | **Settles the accumulator question** §7.13 of the reachability note left open — whether a low-AND product can accumulate in place without the 12-Toffoli add. Emits one GF(2¹²) product call in six shapes and counts what the stream carries: built-in 63 classically / 75 quantum-controlled; the 54-AND family 54 / 66; and a **reduction-aware direct accumulate at 54 Toffolis for 54 ANDs, zero non-AND, 23 qubits narrower than the same family's window** (56 against 79, and 20 wider than the built-in call's 36 as workspace). The field reduction is linear, so each term is XORed into the *target's* own bits and the `2n − 1` window disappears — when the call is classical. Under a quantum control the accumulate's CNOTs would be CCX, and folding the control into one operand instead (`c·(a·b) = (c·a)·b`) measures **66**, exactly the window's 66: the 12 is the control's price, not the accumulator's, so at the 23-AND bound a call is 23 in place and 35 quantum-controlled at any AND count. Checked shot-for-shot on the local MR simulator at GF(2⁴), 64 shots into a dirty target: direct 64/64 and windowed 64/64 against the built-in call, values and tracked phase unchanged. Consequence recorded in §7.14: the arm's post-window stream is already 63.0 per call, so it pays no add and the 0.998× row is its own shape; the 1.139× row is a quantum-controlled variant. Record: `benchmarks/quantum_compilation/gf_product_accumulator_shape.json`; read by `product_axis_decision()` through `load_product_accumulator_shape()` (raises rather than falling back) and pinned by `tests/test_dqi_gluzdov_v6.py::test_the_add_is_the_control_not_the_accumulator` | `audit-only` |
| [`scripts/gf54_mul.py`](../scripts/gf54_mul.py) | The proved 54-AND GF(2¹²) multiply as gates: the pinned `mu(2)=3`/`mu(3)=6` base algorithms, their Karatsuba composition to a 12-bit product (`and_count(12) == 54`), a classical evaluator for checking it against `GF2Field.mul`, and `Gf54MulMixin`, which reimplements `ixor_gf2_mul` as 54 ANDs plus the 12-Toffoli controlled add (66 Toffolis per controlled call against the built-in 75), clearing its windows with `mr`. Pinned by `tests/test_dyn_mr_gated_arms_kickmix.py` | `audit-only` |
| [`scripts/measure_gf54_mul_kickmix.py`](../scripts/measure_gf54_mul_kickmix.py) | Prices the swap: unit cost of one controlled multiply both ways (66 against 75 Toffolis, 54 against 63 ANDs, 42 against 12 workspace qubits), a basis-state function check (values, restored operands, clean workspace, untouched phase), and the six-lever arm built twice on the same shots. At the official instance over 64 shots the arm goes 1,325,230.0 → **1,230,640.0 per shot, 4.652× against 4.320×**, a 7.14% saving rather than the 12.3% a whole-cost 54/63 scaling predicted, because the controlled add does not shrink; `Q` 1,894 → 1,897. Record: `benchmarks/quantum_compilation/gf54_mul_kickmix.json` | `audit-only` |
| [`scripts/attribute_dqi_nonmultiply_tcount.py`](../scripts/attribute_dqi_nonmultiply_tcount.py) | Builds the reconciled Toffoli ledger for the six-lever arm from the measurement records: four exclusive executed categories (10,510 controlled multiplies 693,660.0, inversion products 8,382.7, division products 46,656.0, non-product residual 472,768.2 = 38.7%) that sum to the measured 1,221,466.8, the residual split into the six exactly-measured unconditional regions (94,470, 20.0%) and a **named unattributed bucket** (378,298.2, 80.0%, because the simulator then reported per-shot counts by operation type, not per instruction; that bucket is now split by `measure_dqi_region_ledger.py`), and the static ledger (3,486,597 = 94,470 + 3,392,127). It also records the unit prices, the measurement gaps and the first controlled experiment's promotion gates. Record: `benchmarks/quantum_compilation/dqi_nonmultiply_ledger.json`; discussion in [`docs/NONMULTIPLY_LEDGER.md`](NONMULTIPLY_LEDGER.md); pinned by `tests/test_dyn_mr_gated_arms_kickmix.py` | `audit-only` |
| [`scripts/measure_dqi_region_ledger.py`](../scripts/measure_dqi_region_ledger.py) | Splits the six-lever arm's executed Toffoli stream **region by region**, closing the 80% that `attribute_dqi_nonmultiply_tcount.py` had to leave unattributed: attributes every instruction to the innermost `builder.mark(...)` region (`circuit.info_tags()`) and sums `sim.operation_execution_counts()`, so the rows partition the measured total exactly (it refuses to write otherwise). At the pinned instance: `shift_b_array_right` 162,479.6 non-product Toffolis/shot, `sweep_positions` 143,038.0, `swap_a_and_b` 64,224.0, `_update_flag` 32,856, `_initial_shift_loop` 30,912, `_update_metadata` 12,432, `_update_halting_counter` 10,656, counter decode 4,662, alignment pair 7,392, multiply-and-add control 2,016, quotient control 1,728, Forney control 372.6 — 472,768.2 in total against a 1,221,466.8 arm, with the 748,698.7 of field products separated inside four of those regions. It also runs the arm with one lever absent at a time and records what each lever removed per region (forney 1,243,975.3 in `forney_step`, clear 1,205,319.6 across the reverse EEA, sweep 556,312.0, swap 134,022.0, shift 74,896.4, quotient 48,384.0), checking that each delta equals the sum of its region deltas and that every arm is still correct. Record: `benchmarks/quantum_compilation/dqi_region_ledger.json`; discussion in [`docs/NONMULTIPLY_LEDGER.md`](NONMULTIPLY_LEDGER.md) §1.1–§1.2 | `audit-only` |
| [`scripts/measure_flag_predicate_reuse.py`](../scripts/measure_flag_predicate_reuse.py) | The non-multiply ledger's first controlled experiment, run against the fixed 222-iteration control and the eight promotion gates: share the `(n_A == n_B)` comparison between `_update_flag`'s `c0` and `c1` sub-steps, drive `flag` with two `CCX`, and clear the scratch with `MR` instead of uncomputing it. The prediction (8 Toffolis per iteration from unit costs at the 8-bit register width — pricing at the 12-bit field degree over-predicts by 4) is computed and printed **before** either arm is built; measured 222 × 8 = **1,776 per shot**, `_update_flag` 32,856 → 31,080, `Q` 1,913 unchanged, `Q²T` −0.145%, syndromes/phase/junk clean, 222 rounds, all eight gates pass. The `c2` term cannot share the comparison because it tests `n_a == n_b + 1` between the `iadd` and the `isub`. Record: `benchmarks/quantum_compilation/dqi_flag_predicate_reuse.json`; discussion in [`docs/NONMULTIPLY_LEDGER.md`](NONMULTIPLY_LEDGER.md) §5.1 | `audit-only` |
| [`scripts/measure_eea_oracle_ledger.py`](../scripts/measure_eea_oracle_ledger.py) | The oracle-gated per-region ledger for the conditional EEA body, the treatment the six levers got: for every region it reports the measured executed count and, per lever, the measured removable amount, plus one new gate. `_update_flag` and `_update_metadata` are controlled on the one-hot `c0..c3`, so both are the identity in a round whose halting counter is non-zero; reading `is_hc_zero` through the shared ancilla (`cx` + `MR`, zero Toffolis) and gating them on it removes **1,224 Toffolis/shot** (888 in `_update_flag`, 336 in `_update_metadata`) with the arm still correct and `Q` unchanged — a real gate, not an oracle, because the arm already computes the predicate. Regions with no known predicate (`_pre_align_reg_b`, `unalign_reg_b`, the decode) are named rather than given an invented figure. Record: `benchmarks/quantum_compilation/dqi_eea_oracle_ledger.json`; discussion in [`docs/EEA_AND_PREDICATE_PLAN.md`](EEA_AND_PREDICATE_PLAN.md) §2.3–§2.4 | `audit-only` |
| [`scripts/measure_dqi_combined_arm.py`](../scripts/measure_dqi_combined_arm.py) | Stacks the two promoted control-logic gates — predicate reuse in `_update_flag` and the halting-round gate — in **one** circuit, and predicts the combination before building it. The savings are not additive: the reuse saves its 8 Toffolis only in rounds where `_update_flag` executes, and the halt gate removes that region from six dead rounds, so `8 × 216 + 1,224 = 2,952`, not `1,776 + 1,224 = 3,000`. Measured at the pinned instance over 64 shots: control 1,221,466.8 → **1,218,514.8** per shot (−2,952, matched to the Toffoli), `Q` 1,913 unchanged, 222 iterations in all three arms, syndromes/phase/junk clean, and every arm's region rows partition its own total. The prediction is written before the combined arm is built, and the dead-round count is re-derived from that run's control and halt-gated arms rather than assumed. Record: `benchmarks/quantum_compilation/dqi_combined_arm.json`; discussion in [`docs/EEA_AND_PREDICATE_PLAN.md`](EEA_AND_PREDICATE_PLAN.md) §2.4 | `audit-only` |
| [`scripts/measure_dqi_schedule_campaign.py`](../scripts/measure_dqi_schedule_campaign.py) | The shift/sweep/swap **schedule campaign**, and the largest measured lever in the DQI work: reads each movement and product predicate into a classical bit (`cx` + `MR` through the shared ancilla, zero Toffolis, phase untouched, `Q` unchanged) and gates the work on it, so a gate whose predicate is zero costs nothing on that shot. Four hypotheses are written from the control arm's own chain costs and printed **before** any variant is built — the sweep's one-hot quotient selection (predicted saving ≥ 100,368 from a declared burn of `12 × 2 × 222`), the shift's `do_shift` mask (no saving predicted: it is the XOR of three threshold intervals, not a one-hot), the shift/swap composition (0.0, their predicates are disjoint), and the product's `mul` control (no saving predicted) — and a fifth (`S5_stack`: `532,500 + 2,952 → T = 686,014.8`) is written after the schedule arms and **before** the five-gate arm is built, matched to 0.0 Toffoli. A sixth (`S6_zero_q`, **Experiment B**: gate every surviving product on whether its multiplier `q_i` is zero) is likewise declared at 0.0 and measured: it is **correct and loses** — 997,795.6 Toffolis, +311,780.7 over the incumbent, 327,094 added reads — because the sweep's selection chain cswaps `q_i` into the a-lanes (so the predicate is per-position, not per-round) and a 0.95-Toffoli read cannot pay for a product skipped only inside the `mul` window. Its inverted-condition falsifier is wrong and informative (316,701 product Toffolis skipped). Measured at the pinned instance over 64 shots: control **1,221,466.8** → sweep **1,117,498.8** (−103,968.0, chain 105,696.0 → 1,728.0), shift **1,155,571.8** (−65,895.0, chain 131,220.0 → 65,325.0), products **858,829.8** (−362,637.0, stream 693,660.0 → 331,023.0), all three **688,966.8** (−532,500.0, −43.60%, exactly additive), and stacked with the prior session's flag/metadata gate re-measured in the same record as **686,014.8** (−535,452.0, −43.84%), `Q` 1,913 and 222 rounds in every arm, all arms correct, region rows partitioning each total. Every arm also reports the executed instruction stream and its mid-circuit read count, so a saving can be told apart from relocated work (37,808,019.7 → 32,451,795.1 instructions, 30,253 reads). The composed shift/swap arm and the arm that drops the shift from every `c3` round both save exactly 0.0 and stay correct — and **`scripts/measure_dqi_movement_schedule.py` later showed both arms are the control circuit instruction for instruction** (the method implementing them, `_shift_a_and_b`, is never called), so their 0.0 is vacuous rather than evidence; see [`docs/EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md) §3.3. Three falsifiers (stale selection, inverted mask, previous slot's mask) are all wrong **and** informative, and the record measures skipped work on the chain — not on the arm total, which a wrong state inflates (the stale-selection falsifier skips 104,768.1 of its chain and costs 217,720.1 more than the control). Record: `benchmarks/quantum_compilation/dqi_schedule_campaign.json`; discussion in [`docs/EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md); pinned by `tests/test_dqi_schedule_campaign.py` | `audit-only` |
| [`scripts/measure_dqi_movement_schedule.py`](../scripts/measure_dqi_movement_schedule.py) | **Experiment C:** can the A/B exchange (64,224 executed Toffolis, 72 controlled 892-qubit permutations per shot) be resolved at *build time* — the only kind of saving the pinned competition binary could compile? Pre-declared before any arm existed: the rebased arm must save exactly the region (64,224.0 executed, 198,024 static) and is correct only if the exchanging-round set is shot-independent; the probe must find one pattern or the schedule is unavailable; exchanging on `c3` alone should save 73.0. Measured at the pinned instance over 64 shots: control **686,014.8** at `Q` 1,913 with a 64,224.0 exchange row; a harness arm re-emitting the incumbent's exchange reproduces it to **±0.0** (so the comparison is licensed); the rebased arm zeroes the row and drops 198,024 static Toffolis but is **incorrect**; the probe reads `c3` once per round and finds **10 distinct patterns across 64 shots** while every shot exchanges exactly **72** times, so the schedule is cardinality-stable and order-unstable and no build-time period reproduces it. It also audits two of the campaign's arms: `composed` and `drop_shift_c3` emit the control's stream (149,337 instructions, 16,610 static) and their 0.0 is vacuous. Record: `benchmarks/quantum_compilation/dqi_movement_schedule.json`; discussion in [`docs/DQI_MOVEMENT_SCHEDULE.md`](DQI_MOVEMENT_SCHEDULE.md); pinned by `tests/test_dqi_movement_schedule.py` | `audit-only` |
| [`scripts/measure_dqi_official_compatibility.py`](../scripts/measure_dqi_official_compatibility.py) | **Experiment D:** what the pinned competition binary can actually compile. Mines the whole surface of both builds instead of one attribute: the pinned build has **no `mr`, no `h`, no `gf_product` and no per-instruction counting**, but *has* `hmr`, `bit_store`, `push_condition`, `init_and`/`del_and`/`cswap` — and `del_and` already costs **0** Toffolis there, so an AND clear can be a measurement where a read cannot. The arm's `AttributeError: mr` is therefore the first of **two** blockers (the 54-AND policy is worth 523,849.1 per shot on the reference and was undocumented as a blocker). Both builds count the same reference identically (1,893 / 5,725,071.609375, per-shot arrays equal). The positive result: a candidate the pinned binary **can** build — the reference algorithm with the 54-AND multiply emitted as gates and its workspace cleared with `hmr` — at `Q` 1,896 and **5,433,399.609375** per shot, syndromes restored, **4.79% better** score than the shipped reference and still 7.92× worse than the arm. Record: `benchmarks/quantum_compilation/dqi_official_compatibility.json`; discussion in [`docs/DQI_OFFICIAL_COMPATIBILITY.md`](DQI_OFFICIAL_COMPATIBILITY.md); pinned by `tests/test_dqi_official_compatibility.py` | `audit-only` |
| [`scripts/gf54_mul_reduced.py`](../scripts/gf54_mul_reduced.py) | The 54-AND GF(2^12) multiply with **phase-safe clears only**, so the pinned competition binary can emit it: the field product is accumulated **already reduced** into a 12-qubit window (CNOTs, free), so the only register that needs clearing really holds `lhs * rhs`, which `del_gf2_mul` measures and phase-corrects from the factors (0 Toffolis, no workspace). Every AND target is cleared by `del_and`, also 0 Toffolis and phase-safe. 54 ANDs + the 12-Toffoli controlled add = **66 per controlled call** against the built-in 75, and the scratch is *smaller* than the `mr`-based emitter's rolling 19-qubit window because no unreduced window exists at all. `mr` and `hmr` appear nowhere; every measured clear in the stream is paired with a classically conditioned fixup. Pinned by `tests/test_gf54_mul_reduced.py` (function, junk, phase, Toffoli counts, and the absence of a Z-basis read) | `experimental` |
| [`scripts/audit_record_correctness_checks.py`](../scripts/audit_record_correctness_checks.py) | **Mechanical audit of every DQI record:** which candidate rows claim correctness, and which of the three checks (value, junk, phase) the record actually stores. Written after Experiment D's candidate was called correct on its syndromes and Experiment D2 found 37/64 phase errors; it names that row (`builds.pinned.candidate`, missing junk and phase) among **13 partial-evidence rows in 21 records**, beside 10 fully checked, 51 bare-boolean and 5 resources-only rows. It rewrites nothing and cannot tell whether a recorded `phase: true` was computed against the right expectation — that stays per-record evidence. Record: `benchmarks/quantum_compilation/correctness_audit.json`; pinned by `tests/test_record_correctness_audit.py` | `audit-only` |
| [`scripts/measure_dqi_official_candidate.py`](../scripts/measure_dqi_official_candidate.py) | **Experiment D2:** is Experiment D's compilable candidate *correct*? D checked its syndromes and not its phase, and the DQI submission is scored on phase. One qubit at a time over 200 shots: `hmr` on `|1>` flips the tracked phase on **97** of them — exactly its 97 measured ones, because an X-basis measurement of `|1>` puts a −1 on the measuring-1 branch — while `del_and` (the same measurement plus a classically conditioned `CZ`, the phase fixup) flips **0**. Substituting `hmr` for `mr` is therefore a *randomisation*, not a clear, wherever the value is not a known AND. Measured on the pinned extension at the pinned instance over 64 shots with the verifier's own per-shot rule: reference **1,893 / 5,725,071.609375, 0 value errors, 0 phase errors**; D's 54-AND candidate **1,896 / 5,433,399.609375, 0 value, 37 phase errors** — compilable, and not a submission, so its 0.952 ratio is withdrawn; and a third arm that removes the reverse EEA outright (forward EEA on a CNOT copy of the syndromes, so nothing is restored and no value is read) at **2,736 / 3,438,873.609375, 35 phase errors**, which fails twice: its clears are `hmr` too, and holding the whole scratch live costs more `Q` than the pass it deletes. A fourth arm is the one that stands: the reduced-window 54-AND multiply of `scripts/gf54_mul_reduced.py`, at **`Q` 1,893 and 5,433,399.609375 per shot with 0 value and 0 phase errors — a 5.09% score improvement**, the first correct candidate cheaper than the shipped reference that the pinned binary can compile. Record: `benchmarks/quantum_compilation/dqi_official_candidate.json`; discussion in [`docs/DQI_OFFICIAL_COMPATIBILITY.md`](DQI_OFFICIAL_COMPATIBILITY.md) §3 and §5; pinned by `tests/test_dqi_official_candidate.py` | `audit-only` |
| [`scripts/measure_dqi_submission_score.py`](../scripts/measure_dqi_submission_score.py) | **Experiment E:** runs the competition's own `verify()` — unmodified, in a temporary copy that only stubs the `reaction_depth` diagnostic (13.9 GB peak, 136 s, not part of the score) — on the reference and the reduced-window 54-AND candidate at the competition's default **102,400 shots**, `log2_delta = -128`. Both **PASS** with **0 value and 0 phase errors**; `eps_plus` 8.660587e-4 and `R` 1.191260 are shared, so the score ratio is the Toffoli ratio: **2.443946e13 → 2.319438e13 = 0.949054**, a **5.09% improvement**. The saving decomposes to the gate — 32,408 calls × 9 = 291,672 static Toffolis, exactly the static delta, 291,670.152 executed, `CCX`/shot equal to it; the multiply is 42.45% of the reference's static stream. It also pins the budget: a zero-error run is refused below ~1,375 shots, the smallest batch-aligned passing budget is **1,536**, and 256 shots fails at `eps_plus` 0.2929. And it measures the lever boundary instead of asserting it: of **29** pinned methods mentioning `km.b`, **one** writes one from qubit data (`hmr`), which on `|1>` wrote 1 on 24/64 shots and flipped the phase on the same 24 — a measurement, not a read. Remaining gap stated: 13.66× above the 1.6975e12 target, reachable only through `mr`. Record: `benchmarks/quantum_compilation/dqi_submission_score.json`; discussion in [`docs/DQI_OFFICIAL_COMPATIBILITY.md`](DQI_OFFICIAL_COMPATIBILITY.md) §8; pinned by `tests/test_dqi_submission_score.py` | `audit-only` |
| [`scripts/measure_dqi_arm_submission_score.py`](../scripts/measure_dqi_arm_submission_score.py) | **Experiment F:** the six-lever arm's score, measured by the competition's `verify()` instead of multiplied out. Built on the local `MR` rebuild (`scripts/build_kickmix_mr.py` + `scripts/kickmix_mr_opcode.patch`) at the official **102,400 shots**, the arm **PASSes with 0 value and 0 phase errors**, executes **1,221,408.044756** Toffolis per shot at `Q` = 1,913 (recorded 1,221,466.8 to 4.8e-5; static 3,132,510) and scores **5.324728e12** — **0.2179** of the pinned reference at the same budget, a 4.5898× saving that is exactly the `Q²T` ratio, and still **3.137×** the derived 1.6975e12 target (at 1,913 the target's budget is 509,088). The record states in `status`, `kickmix_role` and `scope` that it is **not a receipt**: only the pinned extension may produce one and it has no `mr`. Run: 6.7 s to build, 236.8 s to verify. Record: `benchmarks/quantum_compilation/dqi_arm_submission_score.json`; discussion in [`docs/DQI_OFFICIAL_COMPATIBILITY.md`](DQI_OFFICIAL_COMPATIBILITY.md) §9; pinned by `tests/test_dqi_arm_submission_score.py` | `audit-only` |
| [`docs/BENCHMARK2_STATUS_DECISION.md`](BENCHMARK2_STATUS_DECISION.md) | **What the measurements decide, so nobody has to re-derive it.** D1: the pinned binary is **frozen at 0.949054** (2.319438e13, 13.66× the target) and no further pinned-binary circuit work is planned — the multiply is 42.45% of the stream and even deleting all of it leaves 8.29×. D2: **`MR` is the only reopen condition** — 4.142× on the reference, 4.5898× on the arm, with the arm's own scores already measured (5.324728e12 / 2.990339e12); if no upstream build ships a Z-basis read, the line stays closed. D3: **no more field-arithmetic optimisation** — the remaining 240,785.8 Toffolis are the swap, the shift, `shift_step`, `update_flag` and the rest of the multiply window, so the next class of change is a decoder algorithm and it is *not started*. Each decision names what would overturn it, and D1 now scopes "submission-grade" to a measured score whose arm has **no emitted `.kmb`** — which is why the sealed portfolio holds Benchmark II at 1.000× and records 1.053682× as the factor that enters on packaging. Pinned by `tests/test_benchmark2_status_decision.py` | `audit-only` |
| [`scripts/measure_dqi_campaign_arm_submission_score.py`](../scripts/measure_dqi_campaign_arm_submission_score.py) | **Experiment G:** the campaign's stacked arm — the six levers plus the five schedule/control gates, `mr` clears, 54-AND policy — through the competition's `verify()` at **102,400 shots**: **PASS with 0 value and 0 phase errors**, **685,936.2051367187** Toffolis per shot at `Q` = 1,913 (recorded 686,014.8 to 1.1e-4; static 3,130,734) and score **2.990339e12** — **0.12236** of the pinned reference, **0.5616** of the six-lever arm, and **1.762×** the derived 1.6975e12 target. It is the frontier: at `Q` = 1,913 the target's budget is 509,088. Not a receipt (the pinned binary has no `mr`); build 9.1 s, verify 279.1 s. Record: `benchmarks/quantum_compilation/dqi_campaign_arm_submission_score.json`; discussion in [`docs/BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) §5; pinned by `tests/test_dqi_campaign_arm_submission_score.py` | `audit-only` |
| [`scripts/compare_gf_mul_external_optimizer.py`](../scripts/compare_gf_mul_external_optimizer.py) | **The external-optimizer comparison.** Records the 54-AND multiply's forward netlist from the emitter itself (**54 ANDs, 1,004 CNOTs**, 54 clears at **0 T**), expands each Toffoli to the standard 7-T block — verified on all eight basis states before any number is taken — and runs **PyZX 0.10.7** on it. Result: the Clifford pass leaves **T at 378 → 378** and only compresses CNOTs (1,328 → 824), and `full_optimize` **does not return within 90 s** at 99 qubits and 378 T, so a general-purpose optimizer does not beat the AND count. A unitary optimizer must compute *and* uncompute, which is **108 ANDs = 756 T**; our clears are free because they are measurements. The decoder is not comparable — `HMR` and `INFO_TAG` in its stream, `mr` in the arm's clears. Record: `benchmarks/quantum_compilation/gf_mul_external_optimizer.json`; pinned by `tests/test_gf_mul_external_optimizer.py` | `audit-only` |
| [`docs/REPRESENTATION_AWARE_UNCOMPUTATION.md`](REPRESENTATION_AWARE_UNCOMPUTATION.md) | **The reusable method, as a checklist.** The transform that worked is stated as a rule — find the inverse primitive's precondition, redesign the intermediate representation to meet it, prove value/phase/workspace, price it — and turned into nine gates for the synthesizer, with the failure each prevents. The three measured cases sit side by side: `del_and` (0 phase flips on `\|0>` and `\|1>`), `del_gf2_mul` (0 Toffolis, 66 vs 75 per controlled call, `Q` unchanged because the 12-qubit window fits), and `hmr` as the counterexample (on `\|1>` it wrote 1 on 24/64 shots and flipped the phase on those 24, and as a clear it gave 37/64 phase errors). Gates 3, 4 and 7 are the correctness deciders; the note names its hooks (`CircuitCandidate`, `CandidateStatus`, `referee.run_dqi_verifier`, `dqi_verification_scope`, `contracts.py`) and its limits (no minimality proof, no external-optimizer comparison). Pinned by `tests/test_representation_aware_uncomputation.py` | `audit-only` |
| [`scripts/measure_vqc_licence_boundary.py`](../scripts/measure_vqc_licence_boundary.py) | **The amplitude tier:** compares the controlled gate the decoder emits with the read-then-gate map the arms substitute, at amplitude level, and records where the licence ends. All eight basis inputs of the gadget are the same state up to global phase with the classical bit equal to the control and the ancilla reset; a superposed control gives **two branches at 0.500** and a fidelity of **0.500** against the coherent reference, with the control's off-diagonal going 0.500 → **0.000** — a complete dephasing, not a perturbation. It also settles the substitution the pinned binary would need: `hmr` alone returns a random outcome on `|0>`, while `h; hmr` is `mr` branch for branch on basis *and* superposed inputs, which is why the blocker is the missing Hadamard. The tier is cross-checked against `kickmix` on the gadget's eight basis inputs (identical), and the arm's 1,913 qubits are reported as 10^575.87 amplitudes rather than pretended. Record: `benchmarks/quantum_compilation/vqc_licence_boundary.json`; discussion in [`docs/VQC_AMPLITUDE_TIER.md`](VQC_AMPLITUDE_TIER.md); pinned by `tests/test_statevector_licence.py` | `audit-only` |
| [`scripts/measure_search_engine_gaps.py`](../scripts/measure_search_engine_gaps.py) | Measures what the symbolic search engines can reach: twelve planted law families, searched on 200 rows and scored on 100 untouched rows with coefficients refitted on train (recovery = holdout R² ≥ 0.99). Classic `HierarchicalSearch` recovers 2/12, `beam_search_expressions` 1/12; the extended template set recovers **11/12** and the beam function atoms **9/12**. The remaining unreachable family is a sinusoid of non-unit frequency (the fitter has no frequency parameter), recorded in `still_unreachable` rather than tuned away. Record: `benchmarks/search_engine_gaps.json`; discussion in [`docs/SEARCH_ENGINE_GAPS.md`](SEARCH_ENGINE_GAPS.md) | `audit-only` |
| [`scripts/reconcile_dqi_resource_baseline.py`](../scripts/reconcile_dqi_resource_baseline.py) | Reconciles the locked official reference row (1,893 qubits, 5,725,108.382109375) with this repository's own measurements: proves from the figures' fractional parts that the official one is a mean over the verifier's 102,400 shots and this project's is a 64-shot mean, runs the reference on the **pinned** binary and the local `MR` rebuild over the same sampled syndromes and compares per-shot Toffoli arrays (identical, both 5,725,071.609375 at 1,893 qubits), and records the pinned binary's failure to build the six-lever arm (`AttributeError: 'Builder' object has no attribute 'mr'`) as the reason no receipt is possible. Record: `benchmarks/quantum_compilation/dqi_resource_reconciliation.json`, pinned by `tests/test_dyn_mr_gated_arms_kickmix.py` | `audit-only` |
| [`scripts/measure_gf54_native_opcode.py`](../scripts/measure_gf54_native_opcode.py) | Prices the same multiply as a builder policy (`builder.gf_product = 'and54'`, emitted by `gen_gf54_mul` inside the extension) instead of a Python gate stream, so it reaches the products the mixin cannot: controlled multiply 75 → **66**, plain multiply 63 → **54**, inversion 315 → **270**, division 378 → **324** Toffolis, 19 workspace qubits (12 + 19 controlled) once each sub-product is reduced into the target as it is computed instead of into a whole 2m-1 product window, all four correct on random basis states. The pinned-instance reference goes **5,725,071.6 → 5,201,222.5 per shot (9.15%)**, `Q` 1,893 → 1,912, and the six-lever arm 1,325,230.0 → **1,221,466.8** (−7.83%; 4.687× against the unchanged reference, 4.258× against a reference carrying the same product policy). It also isolates the Forney block: the same reference with only `init_gf2_inverse_with_scaffold`/`gf2_phase_by_product` switched executes 20,469.5 inversion products per shot (4,093.9 positions) and saves **184,225.1 Toffolis, 3.22% of the reference and 35.2% of its saving**, the rest being the division and sweep. Record: `benchmarks/quantum_compilation/gf54_native_opcode.json`, pinned by `tests/test_dyn_mr_gated_arms_kickmix.py` | `audit-only` |

**Interpreter requirement:** every one of them that builds a circuit imports `kickmix`, a
compiled extension.
They only run under the Python that extension was built against; a mismatched
interpreter fails at import with
`undefined symbol: PyObject_ClearManagedDict` (observed: 3.12 interpreter against a
3.13 build). The shebangs say `env python3`, so invoke them explicitly with the
right interpreter rather than relying on the shebang.

**Measured verdict — scratch pool (2026-10-28):** `Q`'s 1,893 ids are never 1,893
qubits at once. The concurrent peak is **1,828**, reached inside
`_chien_search_and_forney_mbuc`, and the innermost frame there holds **25** — so a
shared scratch pool can redistribute at most 25 qubits, not the 104 the V6 row claims
(77% of every transient). The allocator question is left open rather than answered: the
trace replay returns 1,821 ids against 1,893 shipped, the proxy sees only 1,873, and a
lowest-hole free list replays identically to the shipped LIFO stack.

**Measured verdict (2026-10-28):** halting-counter recycling is a constant −240
Toffoli at every instance measured (−0.722 % at m=15, −0.0118 % at m=4095), and
the shipped flag+counter prototype is a net **regression** (+480 at full scale).
`RecycledFlagDecoder` itself fails verification on the toy instance (effective
error rate 0.9155 ≥ 1/16) — importing it is safe, running it is not. The chain
is kept because it is the evidence for that negative result, not because it is
viable.

### APEDP-VQC Core (experimental)

The reusable quantum-computing core under the DQI workload: a versioned circuit
IR, an exact statevector backend, a stabilizer backend for Clifford circuits,
density-matrix noise channels, a distance-3 bit-flip code, the perfect five-qubit
code, a unitary/phase equivalence referee and adjudication receipts. Discussion
and measured evidence in [`docs/VQC_CORE.md`](VQC_CORE.md); boundaries are stated
there and in the package docstring (the stabilizer tier is Clifford-only, there is
no MPS or tensor-network tier, no surface code, no fault tolerance, no hardware).

| Module | Purpose | Status |
|---|---|---|
| [`apedp_dyn/vqc/__init__.py`](../apedp_dyn/vqc/__init__.py) | Package facade: the IR, gates, backends, noise, logical code and receipts in one import, with the platform's boundaries stated in the docstring | `experimental` |
| [`apedp_dyn/vqc/gates.py`](../apedp_dyn/vqc/gates.py) | One gate table (2x2 unitaries, rotations, controlled swaps) shared by every backend, so a gate name cannot mean two things on two backends | `experimental` |
| [`apedp_dyn/vqc/circuit.py`](../apedp_dyn/vqc/circuit.py) | The versioned circuit IR: gates, measurements with reset, classical-bit conditions, noise operations, declared ancillas, metadata, canonical JSON and a deterministic SHA-256 identity, with validation that refuses anything a backend would have to guess | `experimental` |
| [`apedp_dyn/vqc/backends.py`](../apedp_dyn/vqc/backends.py) | The backend protocol and the exact statevector backend: one dense state per shot, `MR` collapse-and-reset, classical feedforward, sampled Pauli trajectories and residual ancilla excitation | `experimental` |
| [`apedp_dyn/vqc/noise.py`](../apedp_dyn/vqc/noise.py) | Exact density-matrix channels (bit/phase flip, depolarising, amplitude damping, dephasing) with trace, Hermiticity, positivity and Kraus completeness recomputed, and analytic reference formulas | `experimental` |
| [`apedp_dyn/vqc/logical.py`](../apedp_dyn/vqc/logical.py) | The distance-3 bit- and phase-flip repetition codes as circuits (encode, inject, extract, decode, correct, read out, against `3p^2 - 2p^3`), plus a distance-parameterised builder whose correction gates come **from a decoder object**, with `repetition_exact_failure_probability` enumerating all `2^d` error patterns as the independent prediction (perfect syndrome extraction; one error type per code; no threshold claimed) | `experimental` |
| [`apedp_dyn/vqc/qasm.py`](../apedp_dyn/vqc/qasm.py) | OpenQASM 2 import/export: gates, `measure`, `reset`, `if(c==k)`, with metadata and ancillas carried as `// vqc-...` comments; refuses noise operations, sub-register conditions (`if(c==k)` is whole-register, per-bit tests are QASM 3), user-defined `gate` declarations and unsupported parameters instead of approximating them | `experimental` |
| [`apedp_dyn/vqc/five_qubit.py`](../apedp_dyn/vqc/five_qubit.py) | The perfect five-qubit code as a circuit: stabilizer-projection encoder with a conditional Pauli frame, single-error syndrome extraction, conditional correction, and the exact logical-`Z` failure rate against `6p^2 - 24p^3 + 34p^4 - 20p^5` (perfect syndrome extraction; depolarising noise; no threshold claimed) | `experimental` |
| [`apedp_dyn/vqc/equivalence.py`](../apedp_dyn/vqc/equivalence.py) | The circuit referee's equivalence check, in two tiers: `adjudicate_equivalence` materialises both unitaries and compares them up to one global phase near ten qubits; `adjudicate_clifford_equivalence` compares the **conjugation action** on the Pauli generators instead, which is exact unitary equivalence at any size for Clifford circuits and is phase-sensitive where a distribution test is blind. The dense distance is `min_phi \|\|U_c - e^(i·phi) U_r\|\|_F / sqrt(2^n)`, minimised in closed form at `phi = arg<U_r, U_c>` (`entry_deviation` is reported separately and explicitly not minimised); both refuse a measuring circuit (a channel), both report a resource claim as its own `claim_verdict` (`MET`/`UNMET`) so an equivalent-but-larger candidate stays VALID and is refused as non-improving rather than wrong; the size a tier cannot decide is `UNRESOLVED`, never `INVALID`. Two scopes: `as_subroutine=False` (default) allows one global phase, `as_subroutine=True` compares `diag(I, U_c)` against `diag(I, U_r)` and does not (a phase under control is a relative phase); the controlled scope is dense-only, so above ten qubits it is `UNRESOLVED` rather than guessed | `experimental` |
| [`apedp_dyn/vqc/stabilizer.py`](../apedp_dyn/vqc/stabilizer.py) | A stabilizer backend: exact Clifford execution over `n` commuting signed Pauli generators (phases tracked as integers mod 4, tables built once from matrices), Z-basis measurement with deterministic-outcome readout by GF(2) elimination, reset, classical feedforward, and a 100-qubit GHZ chain the dense tier cannot hold. Non-Clifford gates and every `Noise` operation are refused by name | `experimental` |
| [`apedp_dyn/vqc/decoders.py`](../apedp_dyn/vqc/decoders.py) | The decoder interface (`decode(syndrome) -> Correction`), the distance-3 lookup table the repetition code started with, and exact minimum-weight perfect matching on a defect graph (T-join dynamic programming over subsets of defects, boundaries included, exponential in the number of defects and stated as such). One graph per error basis; the non-CSS five-qubit code keeps its own decoder | `experimental` |
| [`apedp_dyn/vqc/discovery.py`](../apedp_dyn/vqc/discovery.py) | Discovery admission: a `CircuitCandidate` (candidate, reference, resource claims, optional declared distribution) routed through the equivalence referee and the distribution receipt, sealed as a `CandidateVerdict` with **four separate fields** — `semantic_verdict`, `resource_verdict` (`MET`/`UNMET`), `evidence_sufficiency` (`SUFFICIENT`/`UNDECIDED`/`INSUFFICIENT`, derived from the checks so an empty check set can never read as a pass) and `promotion_decision`. `promote()` requires all three of VALID, no unmet claim and sufficient evidence, and says which one failed | `experimental` |
| [`apedp_dyn/vqc/colouring.py`](../apedp_dyn/vqc/colouring.py) | **Minimum edge colouring: the `cz` schedule with a proof.** A layer of commuting `cz` gates is a matching, so a schedule is an edge colouring and its minimum is the edge chromatic number `χ′`, which Vizing bounds by `Δ ≤ χ′ ≤ Δ + 1`. `minimum_edge_colouring` tries the greedy peel first (a peel that reaches `Δ` is already optimal by the degree bound), then the **layer-capacity bound** — a layer is a matching, so it holds at most `⌊V/2⌋` edges and `Δ` layers cover at most `Δ·⌊V/2⌋`; more edges than that refute `Δ` outright, at zero search nodes, which is what decides the odd complete graphs (`K9`'s 36 edges do not fit into eight layers of four) — otherwise it runs a complete backtracking search for a `Δ`-colouring with colour-symmetry breaking, where a colouring it finds is minimal and an **exhausted space** proves the graph class 2. Either refutation makes a `Δ + 1` colouring minimal. Every result carries `optimal`, `delta_refuted`, the search nodes and the argument in words, so "proven minimum" and "the best found" are never the same claim. The search prunes on a necessary condition (a vertex cannot have more uncoloured edges than free colours) and carries a node budget, because deciding the `Δ` case is NP-complete. The `Δ + 1` half is **constructed rather than searched**: `vizing_colouring` is Vizing's proof as an algorithm (fan rotations in the Misra–Gries form — one fan, one `c`/`d` path inversion and one rotation colour a single edge in a single pass with `Δ + 1` colours), so it always produces a schedule, verifies it (proper, complete, within `Δ + 1` layers) and raises `ColouringConstructionError` rather than returning a partial one. That is what a budget-exhausted graph needs: a run that only exhausts the budget classifies nothing, so it reports the constructed `Δ + 1` colouring at most one layer above the minimum by Vizing (`status: delta_undecided_within_one`) and never calls it minimal — `K9` gets 9 layers there where the peel used 15 — and `greedy_peel` stands only if the construction fails its own check (`status: construction_refused`). Refuses a self-loop, a repeated edge, an empty edge set and an out-of-range endpoint | `experimental` |
| [`apedp_dyn/vqc/synthesis.py`](../apedp_dyn/vqc/synthesis.py) | Synthesis from a **specification with no reference circuit**: a `TargetSpecification` is a qubit count, `n` signed Pauli stabilizers and an optional outcome support, validated as a pure state by the tableau's own structural check before any search runs. Two referees decide a construction — the **stabilizer tier** (`phase_of(generator) == +1`, any width) and the **dense tier** (amplitudes against the state projected out of the stabilizers, `DENSE_SPEC_LIMIT` = 10 qubits, which is what decides a non-Clifford construction). A construction grammar (chain, star, balanced tree, seeded random tree, ring) is built, adjudicated, ranked by `(depth, operations, max_edge_span)` with the Pareto frontier reported, then admitted through `discovery.CandidateVerdict` — the same four fields and the same empty-check rule as the rewrites loop. **Graph states** are the general form behind the ring: `cluster_state_target(rows, cols)` (the grid-graph state, qubits on the lattice **vertices**) and `seeded_graph_target(n, seed=...)` (an irregular graph) validate their edge set, `graph_of_state` reads it back out of the stabilizers, and five schedules of the same commuting `cz` set are built — wire order, transposed order, breadth-first tree then closures, a **minimum edge colouring** (`colouring.py`, proved optimal by Vizing's bound), and a seeded random control — with the minimum measured against the proven degree floor (`d` neighbours need `d` layers) and the greedy peel it replaced reported beside it, so the saving on each target is a measured column rather than an assumption. `adjudicate_specification(..., tier=...)` names the referee (only ever as a keyword, and only a tier that can decide the circuit — a quiet fallback would compare a referee with itself), which is what `referee_agreement()` uses to run **both** tiers on one circuit and report two independent verdicts. The corpus also carries two wide rows, where a second *tier* does not exist: the 400- and 900-qubit minimum schedules are each decided by the stabilizer tier and again by **Stim** (`foreign_referee.py`, the independent third-party tableau), the three wrong graph constructions are decided again by both at 400 qubits and refused by both, the dense tier's own refusals at those widths are stored beside the rows, `agree` is left `null` when the third party abstained rather than reporting an agreement that was never measured, and a comparative disagreement makes the script raise and write no record at all. The dense tier re-indexes the simulator's basis order to the specification's (`_specification_order`); an asymmetric irregular graph state is what exposed that, where the naive comparison read distance `0.0605` on an exactly correct state. `dense_tier_cost` runs the tier's own arithmetic at any width and returns seconds, peak traced bytes and the phase-free distance with **no verdict field**, which is how the ten-qubit bound is priced rather than asserted: one stabilizer is a dense `2**n x 2**n` operator (`16 * 4**n` bytes), so the cost is quadratic in the Hilbert dimension. The public oracle still refuses to materialise above the bound and the referee still answers `UNRESOLVED`, so a caller allowed to exceed the bound cannot turn a price into an adjudication. Non-Clifford *search* does not exist and is stated as absent | `experimental` |
| [`apedp_dyn/vqc/clifford_search.py`](../apedp_dyn/vqc/clifford_search.py) | The first discovery engine that emits the VQC IR: local semantics-preserving Clifford rewrites (self-inverse cancellation through commuting gates, including shared-control/shared-target CNOTs and diagonal CZ/phase gates; `s`/`Sdg` algebra; `h z h = x`, `h x h = z`, and `h y h = y` up to global phase), each accepted step certified by the Clifford referee **against the original reference**, ranked by `(operations, depth)`, then admitted through `discovery.py`. Mutations (delete a gate, flip a phase, `cx`↔`cz`) are proposed too, so refusals are measured rather than assumed. Greedy, local, Clifford-only, and stated as such | `experimental` |
| [`apedp_dyn/vqc/native.py`](../apedp_dyn/vqc/native.py) | The Rust-first / Python-fallback bridge to `rust/apedp-vqc` (import name `apedp_vqc`): Pauli algebra, the stabilizer tableau, sampling, exact distributions and the matrix-free Clifford images are computed natively and parsed from the same canonical circuit JSON; every entry point returns `None` when the fast path cannot answer so the Python tier decides, `engine_info()`/`tallies()` record which path ran, `assert_parity()` re-derives the deterministic answers in Python and raises on disagreement, and the referee never moves: images come from Rust, verdicts come from `equivalence.py`, which re-derives the table in Python below `PARITY_LIMIT` qubits and says `image_verified_by: "suite"` above it | `experimental` |
| [`apedp_dyn/vqc/receipts.py`](../apedp_dyn/vqc/receipts.py) | Adjudication receipts: identity, provenance, per-check statistics, resources, ancilla cleanup, reproducibility and a VALID / INVALID / UNRESOLVED verdict, with the shot count required to detect the declared effect | `experimental` |
| [`docs/VQC_CORE.md`](VQC_CORE.md) | The first VQC acceptance milestone, measured: round-trip and hash evidence, the OpenQASM 2 interchange table, Bell/GHZ/teleportation results, reset semantics, the channel checks, the syndrome table, the logical rates and every not-yet-built capability named | `experimental` |
| [`docs/VQC_SYNTHESIS.md`](VQC_SYNTHESIS.md) | The synthesis experiment, measured: Experiment A (construct a 100-qubit GHZ from its stabilizers with **no reference circuit** — five families, balanced-tree depth 12 against the chain's 100, promoted, four near misses refused with the failing generator named, and a ring graph state as a second target), Experiment B (10 to 1,000 qubits, 24 verified rows, 1.3-1.7 s each), Experiment C's boundary (a non-Clifford `ry` construction decided by the dense specification referee at 3 and 8 qubits, `UNRESOLVED` at 12), Experiment D's placement costs under **both routers** (greedy and a scheduling lookahead: lattice tree 220 → 84, lattice chain 191 → 117, every routed row improved and every free row unchanged), its **three-basis two-qubit accounting** (abstract 1,193 against CNOT-equivalent 3,381 for the greedy lattice tree; total depth including the restore ladder reported separately from the pre-restore depth), §5.2's two referees on one circuit, and **Experiment E** (16-, 100-, 400- and 900-qubit 2-D clusters and two seeded irregular graphs given as edge sets: the operation count, two-qubit count and connectivity are the specification's and only the schedule changes, so the five schedules tie on resources and differ in depth; the minimum edge colouring attains the degree floor on all six — depth 5 against 37 for the row-major walk at 100 qubits, 5 against 77 at 400 qubits on a 20x20 lattice (760 edges, 1,160 operations), and 5 against 117 at 900 qubits on a 30x30 lattice (1,740 edges, 2,640 operations) — and **saves 0 layers there**, because the greedy peel it replaced already happened to attain `Δ`, while on the class-2 graphs it adds (odd cycle C5, Petersen, `K5`, `K9`, `K11`) it proves `χ′ = Δ + 1` — by the layer-capacity bound where the edges cannot fit into `Δ` layers, by an exhausted search where they can — and saves **12** layers across those rows (2 on `K5`, 6 on `K9`, 4 on `K11`, each against a 15-layer peel); a missing edge, an extra edge and one `cz`→`cx` are each refused with exactly two endpoint stabilizers named, and every schedule carries its own **synthesis and verification wall clock**, each the **median of three repeats** on the same rebuilt circuit with the samples kept on the row, timed across three lattice sizes so it reads as a ladder (build 0.001-0.002 s and decide 0.015-0.017 s at 100 qubits, 0.020-0.032 / 0.233-0.288 at 400, 0.101-0.160 / 1.269-1.369 at 900, deciding about ten times building at every size), **§5.3.1's routing measurement** (the five schedules placed on a line, a ring and the cluster's own lattice at 16, 100 and 400 qubits by both routers: 90 placements, 72 comparisons against the minimum schedule, all of them `VALID`; the lattice needs no SWAP at all so the advantage there is exactly the logical one at 1.00x, while on a line or ring it grows to 1.00x-219x with a median of 4.67x and no comparison inverts, the localized orders paying 144,400 swaps against the minimum's 12,316 at 400 qubits on a line), **§5.4's dense-tier ceiling** (a non-Clifford decision priced width by width at 6, 8, 10, 11, 12, 13 and 14 qubits under a declared 8 GiB address-space cap: 0.25 s / 20 MiB at the bound, 20 s / 1.25 GiB at 13, 88 s / 5.0 GiB at 14, a 16 GiB operator and `MemoryError` at 15, with peak growing 3.96–4.00x per qubit), plus what is deliberately not claimed (a grammar, not a general synthesiser; no non-Clifford search; no cheaper GHZ; the scheduler's optimality is proof-carrying but its `Δ` search is budgeted, because deciding that case is NP-complete; the ceiling prices one implementation, not the question) | `experimental` |
| [`docs/VQC_RUST_CORE.md`](VQC_RUST_CORE.md) | The dual-language inventory, measured: which VQC capabilities are Rust-first (Pauli algebra, stabilizer tableau, sampling, exact distributions, Clifford images, IR parsing) and which stay Python (the referee, discovery, dense tier, decoders, noise); the five engineering rules mapped to the code that enforces them; the measured ratios on identical frozen inputs and the limits that are deliberately not claimed (no memory ceiling, no parallelism, no decoder or noise port) | `experimental` |
| [`docs/VQC_PHASE0_CAPABILITY_INVENTORY.md`](VQC_PHASE0_CAPABILITY_INVENTORY.md) | **Phase 0:** every DQI component classified as reuse / adapt / build / retain / not connected, with the Phase 1 task table, the mapping of the proposed `apedp_vqc/` tree onto the actual package, and the absent capabilities named; the DQI evidence it must not disturb is frozen by SHA-256 | `experimental` |
| [`benchmarks/quantum_compilation/vqc_phase0_baseline.json`](../benchmarks/quantum_compilation/vqc_phase0_baseline.json) | The frozen SHA-256 baseline of the thirteen DQI records, so a later phase that rewrites the baseline fails instead of silently re-basing it. It carries two `amended` entries: `correctness_audit.json` was regenerated after new VQC records were added (stale scan, totals unchanged) and its hash re-frozen with the reason recorded rather than silently; the second amendment is the native-execution record `vqc_native_parity.json`, with the pinned totals (12 checked, 13 partial rows, 5 resources-only, 51 aggregate-only) and the same 13 DQI gaps | `experimental` |
| [`apedp_dyn/vqc/placement.py`](../apedp_dyn/vqc/placement.py) | **Architecture-aware depth**, the second number a construction costs: line, ring, rectangular-lattice and all-to-all connectivity graphs, and a greedy SWAP router that moves an operand along a shortest path until its partner is adjacent. The logical-to-node mapping is tracked exactly and the emitted swaps are undone in reverse order at the end, so the swap product telescopes to the identity and the mapped circuit is the **same unitary on the same wires** — which is why the specification referee decides it directly. Both depths are reported (`logical_depth`, and `mapped_depth` as the **total** including the restore ladder) along with `depth_before_restore`, `restore_depth_cost` (the difference), `swap_gates`, `restore_swap_gates`, `longest_hop` and the routing overhead, so the cost of being verifiable is visible rather than folded in. Two-qubit cost is reported in **three bases**: `two_qubit_gates`/`abstract_two_qubit_gates` (a SWAP is one instruction), `native_entanglers` (a declared `swap_entanglers` per SWAP, one by default) and `cx_equivalent_two_qubit_gates` (three CNOTs per SWAP), because a native-SWAP device and a CNOT-only device execute different counts from the same circuit. Two routers: `strategy="greedy"` (second operand along a shortest path, no scheduling, no lookahead) and `strategy="lookahead"` (either operand, any shortest-path step, each candidate swap scheduled into the earliest layer its two nodes are free, then scored by the total wire distance of the next `lookahead` eight two-qubit gates involving a moved qubit). The lookahead is worse at small widths, reorders no gates, and is a heuristic; the record carries both routers row by row at 100 qubits. An unreachable graph, a self-loop, a circuit wider than the device and a three-qubit gate with no common neighbourhood are refused by name; no device calibration | `experimental` |
| [`apedp_dyn/vqc/external.py`](../apedp_dyn/vqc/external.py) | **External comparison with abstention as a first-class outcome**: a `Submission` is another tool's circuit plus that tool's own operation count, and `adjudicate_submission` decides it with the same specification referee and the same `VALID`/`MET`/`SUFFICIENT`/`PROMOTED` policy a local candidate faces, so a foreign gate count is quoted only for a circuit that was independently verified. `TOOLS` declares the three tools the review named (Qiskit synthesis, Stim execution, MQT QCEC equivalence) with the task and metric each would need; `probe()` reports `compared`, `installed_not_wired` or `unavailable` **with a reason**, and only the first is a comparison — a tool that did not run is never counted as a loss. The Qiskit adapter feeds it the same signed stabilizer strings, reverses them for its little-endian convention, and re-imports its QASM 2 output. **Stim is wired for execution/sampling**: `stim_sampling` builds a `stim.Tableau` from the same signed stabilizers, synthesizes a preparation circuit and compiles a sampler, while `apedp_sampling` clones this package's tableau per shot; `adjudicate_sampling` decides each sample set against the **exact** distribution (enumerable at 12 qubits) with a TV band simulated from that distribution rather than a chi-square approximation with empty cells, and `sampling_pair` compares the two samplers under a null in which both are correct; `apedp_dyn/vqc/memory.py` measures each engine's peak resident growth in a child interpreter so the record carries memory beside throughput. At a width where the full distribution is not enumerable this path is `SamplingUnsupported` rather than quoting an approximation; above 12 qubits the sampling comparison continues under [`apedp_dyn/vqc/marginals.py`](../apedp_dyn/vqc/marginals.py)'s exact marginal referee. **Qiskit's own router is wired too**: `qiskit_routing` hands Qiskit a coupling map (a recorded device snapshot supplied by the caller) and transpiles the logical circuit with SABRE, re-imports the result through QASM 2, and reports both the routed depth measured here and Qiskit's own `depth()`. Since Qiskit keeps its final layout rather than restoring it, `relabelled_target` maps the specification onto the wires the device will actually hold the state on, so the foreign circuit is still decided by the same referee; a layout that is not injective is refused, not repaired. **MQT QCEC is wired for the equivalence task** (see [`apedp_dyn/vqc/equivalence_corpus.py`](../apedp_dyn/vqc/equivalence_corpus.py)): `qcec_equivalence` hands it the same pair this package's referee decides, built from the IR gate for gate by `qiskit_circuit` because Qiskit's OpenQASM 2 importer defines neither `swap` nor `cswap` — a text interchange there would have meant the other tool deciding a circuit this package did not hand it. QCEC runs its own method selection and its own default numerical tolerance, untuned, and the adapter records the criterion it returned **per repeat**: near its threshold identical calls return different criteria (`equivalent`, `equivalent_up_to_global_phase`, `no_information`), so `verdict` is `unstable` there and the criteria themselves are the evidence. The gate table is total over `GATE_SPECS` and a measurement or noise operation is refused rather than handed to a unitary checker. Nothing in `apedp_dyn` imports qiskit, stim or mqt.qcec at import time; all three are optional extras of the harness | `experimental` |
| [`apedp_dyn/vqc/marginals.py`](../apedp_dyn/vqc/marginals.py) | **Exact marginals of a stabilizer specification, at any width**: the sampling comparison's referee is the exact outcome distribution, which is enumerable only up to 12 qubits. `exact_marginal(target, subset)` derives the exact Z-basis distribution of a chosen subset of wires from the specification alone — the distribution of a stabilizer state is uniform over an affine subspace of GF(2)^n, so marginalising is the affine set's own marginal: Gaussian elimination over GF(2) plus `2**|T|` outcomes, and nothing exponential in the state's width (a 10-wire marginal of a 32-qubit state is ~1 ms). Every marginal is checked against the rank it predicts (support size and a sum of one) before it is returned, and a wrong one is refused rather than reported. `adjudicate_marginal` decides each sampler's projected samples with a chi-square goodness-of-fit against that marginal, Bonferroni-corrected over the subsets tested (`min(1, m * p)` recorded per subset), and `marginal_pair` does the same two-sample; a subset whose supported cells have fewer than five expected counts abstains as `UNRESOLVED` with the count that failed, because a chi-square there is the wrong test rather than a small error. The chi-square survival function is exact enough to pin against textbook critical values (~1e-14) with no SciPy. `is_uniform` is reported per subset and is the referee's blind spot made explicit: a uniform marginal cannot separate two states that both have it, and every graph state is uniform in the Z basis, so no Z-basis marginal tells one graph state from another — it still tests the sampler, since an empirical distribution that is not the uniform one it should be fails. A marginal is a projection, not the distribution, so a row says which subsets it decided and nothing about the wires nobody looked at | `experimental` |
| [`apedp_dyn/vqc/memory.py`](../apedp_dyn/vqc/memory.py) | **Peak memory of a sampling engine, measured in its own process and its own window**: `VmHWM` never falls, so two engines measured in one interpreter cannot be separated — each engine therefore runs in a fresh child, and because the whole-child peak is ~165 MiB of interpreter and numpy against a few megabytes of samples, the child **resets the peak counter** (`/proc/self/clear_refs`) after the engine's imports and after rebuilding the specification. `growth_bytes` is then the sampling step's own resident growth, `floor_bytes` is what the engine cost before it sampled, and `absolute_peak_bytes` is the child's peak over its whole life — all four reported, because any one alone invites the wrong reading. The specification crosses the process boundary as its printed stabilizers and is rebuilt by `target_from_strings`, and the child refuses to measure unless the rebuild hashes to the target it was sent. The child also returns the count hash of the samples it drew, so a memory number can be attached to the exact workload whose samples are on record. An engine whose imports differ (Stim) gets them loaded *inside* the floor, not charged to sampling; a platform that cannot reset the counter or read a peak gets `MemoryUnavailable` with the reason, never a zero | `experimental` |
| [`tests/test_vqc_memory.py`](../tests/test_vqc_memory.py) | Peak-memory pins (9): the counter is a real peak (a 32 MiB block moves it before it moves), the reset makes a window measurable (a monotone counter could only ever report the earlier, larger value), a target round-trips hash-identically through its printed stabilizers — including declared support, which is how this pin was written the second time — and refuses malformed strings, outcomes or widths; a measured engine's count hash equals the calling process's own, so the memory is taken on the recorded samples; Stim is measured with its library inside the floor; a platform that cannot reset the counter abstains with a reason instead of quoting the whole child; a child that sampled a different specification is refused; and two runs of one workload agree on everything but the byte counts and the clock, because fabricating a median from two RSS readings would be worse than reporting one run | `experimental` |
| [`tests/test_vqc_marginals.py`](../tests/test_vqc_marginals.py) | Marginal-referee pins (26): every marginal against a dense partial trace of a state built by projecting a full-support vector onto the specification's eigenspaces (real Pauli matrices, so a Y generator contributes its `i` — the route that caught the first version's sign error), generator signs honoured rather than read as labels, the graph-state uniformity as arithmetic, a 32-qubit marginal under a 2 s cost bound (linear algebra, not enumeration), the chi-square survival function against textbook critical values and monotone on 1,023 degrees of freedom, the Bonferroni product as arithmetic (`min(1, m * p)`, repeating a subset multiplying its p-value), a mis-wired wire caught by the subsets that contain it — and only those, a subset with too few expected counts abstaining as `UNRESOLVED` rather than rejecting, a sample set from another specification refused, two correct engines agreeing while a mis-wired one disagrees, and the blind spot measured: samples of a *different* graph state pass the referee | `experimental` |
| [`apedp_dyn/vqc/corpus.py`](../apedp_dyn/vqc/corpus.py) | **A sampled distribution of graph states, so the comparison is neither the favourable case nor a single draw**: every family no construction in either engine specialises to — random connected 3- and 4-regular graph states from a pairing-model draw and a seeded irregular graph — at five widths (12, 20, 32, 48, 64 qubits) with five independent draws per cell, plus one random stabilizer state at each of 12, 20 and 32 qubits. 78 instances, each reproducible from `seed = SEED_BASE[family] + 10 * width + replicate` alone (the test and the record both rebuild it and compare hashes), and each row labelled by the cell that produced it. The regular draws are rejected-and-redrawn until simple and connected rather than repaired, so they are random regular graphs and not regularised versions of something else. What the record reports is a **spread**: on all 75 matched instances both engines use the same two-qubit count (one entangler per edge), and the grammar's only saving is depth — shallower on 75 of 75, median ratio 3.25× with quartiles 2.75–3.75 and range 1.50–5.00, broken out by family (regular 3 3.50× [3.00–3.75], regular 4 4.00× [3.20–4.40], irregular 2.65× [2.38–3.20]) and by width, with the grammar's depth fixed by the degree while Qiskit's tracks the width. That is a schedule result a transpiler pass can reproduce, not a gate-count advantage. On the three stabilizer states the grammar abstains and Qiskit synthesizes. The routed sample is reported the same way, with each engine's own router on the recorded `fake_sherbrooke` map: after routing the 3.25× abstract depth gap is gone — median ratio 0.97× [0.88–1.10], Qiskit shallower on 40 of 75 — because Qiskit's transpiler searches an initial layout while this package starts from identity and restores (350 swaps against 206 at the median). The width tiers and the two generators are fixed, so the spread is over these instances and not over graph space; both outputs are still decided by this package's own referee | `experimental` |
| [`apedp_dyn/vqc/foreign_referee.py`](../apedp_dyn/vqc/foreign_referee.py) | **Adjudication that is not self-referential**: the corpus is decided a second time by an implementation this package did not write. The circuit is handed to Stim, simulated as a stabilizer tableau, and every generator of the frozen specification is evaluated as a Pauli observable — the same signed stabilizers both engines were given, so the third party decides the same question rather than a looser one, and a pure stabilizer state is the unique +1 eigenstate of `n` independent stabilizers, which is why a 127-qubit routed circuit is decided in ~0.1 s. Verdicts are `VALID`/`INVALID` only when a circuit was actually simulated: a missing Stim is `UNAVAILABLE`, a non-Clifford gate, a measurement or a width mismatch is `REFUSED`, each with the reason, and none of the two is scored as a loss. What it does not settle is recorded rather than implied — it decides the prepared state and not gate counts, depth or wall clock, it is Clifford-only, and the IR-to-Stim bridge is this package's, so a shared translation bug would fool both referees. On the record: 150 synthesis decisions and 150 routed decisions, all `VALID`, zero disagreements with this package's referee | `experimental` |
| [`tests/test_vqc_foreign_referee.py`](../tests/test_vqc_foreign_referee.py) | Third-party-referee pins (11): it decides the same specification for a GHZ state and for a hand-written path graph state, it reads the sign of a generator rather than only its Pauli string (|01> against |00> and against a negative-phase spec), it returns `INVALID` for a dropped gate, a circuit built for a different target and a mis-wired qubit, it `REFUSES` a non-Clifford gate, a measured circuit and a width mismatch with the offending operation named, an absent Stim is `UNAVAILABLE` with a reason and never a verdict about the circuit, the referee table states what it does and does not decide, and a 127-wire routed circuit is decided against a padded specification | `experimental` |
| [`tests/test_vqc_external_comparison.py`](../tests/test_vqc_external_comparison.py) | External-comparison pins (51): a foreign circuit is decided before its count is quoted, an unmet foreign claim is `UNMET` and not `INVALID`, a wrong foreign circuit is `INVALID` with the failing generator, the policy is the same one the local path uses (compare against `adjudicate_specification_candidate` directly), the string-order difference is pinned on a non-palindromic `X`-on-0/`Z`-on-1 target **and** the naive order is shown to produce the mirror image, an absent package abstains with a reason while installed-but-unwired is a distinct state, a missing qiskit is a `ToolUnavailable` refusal, and the record's ratios are recomputable from its own rows with no claim of a general advantage — plus the shared corpus: every family cell drawn at every declared width with the declared replicates and the record's `design` block equal to the rows it produced, both engines `VALID` on the matched rows, the abstentions unscored and named, and the summary recomputed from the rows as a **spread** — per-family and overall win counts, medians, quartiles, interquartile ranges, minima and maxima against the per-instance ratios, the per-width depth medians, the structure spread, the bootstrap interval bracketing each median ratio, and the committed record's instance-by-instance gate-count tie pinned as a result of this artifact rather than a law; **plus the Stim sampling rows**: same signed stabilizers and shot budget on both engines, both sample sets `CONSISTENT` against the exact distribution with reproducible count hashes, the pair `AGREE`, a deliberately skewed sample set reported `INCONSISTENT`, a width where the full distribution is not enumerable refused by that path rather than compared against an approximation, a missing Stim refused with a reason, and the record's sampling block pinned with its throughput and version; **and the wide rows**: every row above 12 qubits, every subset decision carrying the Bonferroni product it was taken at, no subset abstaining silently, GHZ's pair marginals keeping their two-outcome correlation, the ring-graph row showing all 34 marginals uniform rather than being left out, the summary recomputed from the rows, and the prose pinned not to claim the whole distribution; **and the memory rows**: every one of the ten sampled rows carries a measurement per engine whose count hash, shot count and target hash equal the row's, whose growth is exactly the difference it claims to be, whose floor is reported beside it, and whose summary recomputes from the rows — with the abstention path carrying a reason and the prose pinned not to present peak RSS as a ceiling; **and the routing rows**: Qiskit's transpiler on a six-node line produces a circuit `VALID` against the target relabelled by its final layout with its reported depth equal to the re-imported depth, a circuit wider than the device is refused, a non-injective layout is refused, the record's four device rows all verify with recomputable winners (both engines win at least one), and the recorded coupling map's edge hash is re-derived from the same snapshot; **and the routed corpus**: all 75 matched instances routed by each engine's own router (Qiskit's transpiler with its seed and optimisation level, this package's lookahead router named on its own row), every routed circuit hash-identical to the circuit the synthesis row measured, both sides `VALID` with ours against the padded target and Qiskit's against the relabelled one, the pre-restore depth used as the comparable number, and every routed median, quartile, win count, swap figure and share recomputed from the rows per family and per width; **and the third-party block**: every matched row carries Stim's verdict for both circuits beside our referee's, both referees agree on every one of the 150 synthesis and 150 routed decisions, the routed rows are decided on every wire of the padded and relabelled targets, the summary recomputes from the rows, the abstention paths carry their reasons, and the block names what the third party does not decide; **and Experiment E's depths against the corpus histogram**: the block accounts for every corpus row (bucket counts summing to the row count, with the empty buckets kept so a long tail is not turned into a short one), every per-width cell carrying the depths it summarises, each target's placement recomputed from the cell it names, exactly one target labelled as a width match, and every placed depth rebuilt as a fresh `graph_layers` circuit whose hash and depth are compared rather than transcribed | `experimental` |
| [`apedp_dyn/vqc/equivalence_corpus.py`](../apedp_dyn/vqc/equivalence_corpus.py) | **A corpus built so two equivalence referees disagree**, which is what the earlier record said was missing: eighteen pairs in six families, each carried with the algebraic identity that fixes its answer (`X Y Z = i I`, `RZ(2*pi) = -I`, `SWAP = CX CX CX`, `Z_0 Z_1` stabilising a GHZ, `U_c = RX(theta) U_r` with a non-scalar rotation), so neither tool adjudicates the other. The families isolate the three axes on which two referees can differ: **tolerance** (one angle from `1e-13` to `1e-3`, every point exactly inequivalent, the edge between `1e-9` and `1e-8` inside this package's `1e-9` and QCEC's answer still `equivalent` at `1e-8`), **scope** (a global phase is free standalone and a relative phase under control; a pair can prepare the same state from different unitaries, where a state verdict and a unitary verdict legitimately differ) and **abstention** (a 12-qubit pair carrying a `T` is refused here with the gate and tier named, and decided by QCEC; a 12-qubit Clifford pair is decided by both). `adjudicate_pair` asks this package **all three** questions about every pair and records the refusals with their reasons, `classify` names every difference (`AGREE`, `DIFFER_BY_THRESHOLD`, `SCOPE`, `QCEC_ABSTAINS`, `WE_ABSTAIN`, `BOTH_ABSTAIN`, `UNSTABLE`, `UNAVAILABLE`, `CONTRADICTION`) with the explanation attached, and `summarise` recomputes the counts **and** re-checks every declared expectation — the run refuses to write a record whose expectations are unmet or whose differences are unexplained. Writing it found a false `VALID` in `adjudicate_clifford_equivalence(on="state")`, now fixed in `stabilizer.py` | `experimental` |
| [`tests/test_vqc_qcec_equivalence.py`](../tests/test_vqc_qcec_equivalence.py) | QCEC-equivalence pins (33): every pair's construction and relation, this package's verdict against **every** declared expectation on all three scopes, the ladder's distances as the halved angles with both edges around `1e-9`, the state pair `VALID` as a state and `INVALID` as a unitary, the diagonal pair whose Z-basis probabilities agree to 1e-12 while the unitaries do not, the refusal of a non-Clifford 12-qubit pair naming the gate, the classifier's five outcomes — including the one it must never invent — the summary recomputing from the rows and noticing a tampered verdict, the Qiskit gate table being total over `GATE_SPECS`, the aggregation rule that two different criteria are not a verdict, and (with QCEC installed) its three vocabulary distinctions, its answer on the pairs this package refuses, and repeats of the identical call; **and the record**: the equivalence block, v8 schema, every check passed, the corpus SHA-256, the repeats on the tolerance family, and no contradiction | `experimental` |
| [`tests/test_vqc_corpus.py`](../tests/test_vqc_corpus.py) | Corpus pins (11): the corpus rebuilds hash for hash from its seeds alone, its labels are unique and name the cell that produced them, every seed equals the one the documented arithmetic rule yields, every graph family cell was drawn at every declared width with the declared replicates while the abstention anchors are one per anchor width, a smaller sample is the same sample restricted rather than a different draw, every "regular" draw is simple, connected and actually of one degree when the degrees are read back out of the stabilizers, the irregular family is irregular at every width, no graph is a ring or has a lattice boundary's degree-2 vertex, impossible regular draws are refused by arithmetic (odd `n*d`, degree ≥ `n`, fewer than four vertices), and the corpus contains both buildable graph states and random stabilizer states for which `synthesize` refuses with a reason | `experimental` |
| [`tests/test_vqc_placement.py`](../tests/test_vqc_placement.py) | Placement pins (34): a topology stays connected (an unreachable node, a two-node ring and a self-loop are refused), paths are shortest, **every two-qubit gate of a routed circuit is between adjacent nodes** for **both routers**, a routed circuit still prepares the specification by both tiers under either router, all-to-all reproduces the logical depth exactly, a line reproduces a chain for free, a tree pays for its connectivity, the restore ladder is reported rather than hidden (`depth_before_restore` equals the unrestored depth and the swap bill doubles), a `ccx` on a line is refused rather than squeezed, a wider device leaves the extra wires idle, no placement is ever cheaper than the abstract circuit, and the lookahead router is recorded with its window, is deterministic, refuses an unknown strategy or a negative window, beats the greedy router on a chain placed on a lattice, and reports the three two-qubit bases (abstract, native with a declared per-swap cost, and CNOT-equivalent) plus the total depth and the separate restore cost | `experimental` |
| [`tests/test_vqc_colouring.py`](../tests/test_vqc_colouring.py) | Scheduler pins (14): a `Δ`-colouring is reported optimal with the degree argument, every layer is a matching and the layers partition the edge set, an odd cycle and `K5` are **proved** class 2 by the layer-capacity bound at zero search nodes while the Petersen graph is proved by an exhausted search (the bound is tight there), `K5` saves 2 layers against the greedy peel (7 → 5) while the peel is shown valid but not minimal, `K9` and `K11` are refuted at budget 1 and budget 200 alike and return 9 and 11 layers against the peel's 15, `K7` minus an edge is refuted by counting at any budget, a one-node budget on the Petersen graph returns the constructed `Δ + 1` schedule marked `optimal=False` (never the peel, never called minimal), the bound is cross-checked against the exhaustive `Δ` search on **every** graph of at most five vertices, `vizing_colouring` is checked against those 1,094 graphs for validity and the `Δ + 1` bound and against structured families (`K9`, a 9-cycle, Petersen, the 4-cube, `K5,5`), a construction that fails its own verification falls back to the peel labelled `construction_refused`, and self-loops, duplicate edges, an empty graph and an out-of-range endpoint are refused | `experimental` |
| [`tests/test_vqc_synthesis.py`](../tests/test_vqc_synthesis.py) | Synthesis pins (57): a specification that is not a pure state is refused (anticommuting, rank-deficient, imaginary phase, support not summing to one), the tableau and projected-vector readings of a specification agree at every width both reach, every construction family reproduces its target, the near-miss family and four mutated constructions are refused with the failing generator named, the ranking and Pareto frontier follow the measured rows, a valid construction with an unmet claim is `VALID`/`UNMET` and not promoted, the claim rule is pinned against the equivalence referee's own implementation, the dense tier decides a non-Clifford construction and abstains above its bound, a 1,000-qubit specification is decided exactly, **both referees reach the same verdict at every width both can decide** (accepts, a wrong construction and a global-phase rotation alike), a named tier that cannot decide a circuit raises instead of falling back, the dense requirement string carries its tolerance, and **Experiment E's graph states**: a 2-D cluster is the grid-graph state, a self-loop/repeated/out-of-range/disconnected graph is refused, the parser refuses a specification that is not a graph state, every valid graph family prepares both targets, the schedule is the only free variable and the minimum colouring attains the degree floor at 3x3 to 8x8, the two basis orders are pinned on a 3-qubit asymmetric state where the naive comparison is visibly wrong, and the **cost probe** is pinned as reproducing the referee's distance exactly inside the bound, as quoting a price and no verdict above it, and as refusing a circuit of the wrong width | `experimental` |
| [`tests/test_vqc_synthesis_record.py`](../tests/test_vqc_synthesis_record.py) | Synthesis record pins (40): no reference circuit in Experiment A, every construction and every 1,000-qubit row verified, the four refusals carrying a named generator, the Pareto frontier including the chain but not the star, the promoted candidate's four admission fields, the second target's near miss, Experiment C's boundary exactly where the record says, **Experiment D's sixteen placements** (both routers on every construction/topology pair, both depths, the swap bill, the restore ladder, all-to-all as the known zero point and a chain free on a line) and its **row-by-row router comparison** (the lookahead never worse on mapped depth, swaps or two-qubit gates, savings recomputable from the two columns, the lattice tree 220 → 84 and the lattice chain 191 → 117) and **the three-basis accounting** (abstract 1,193 against CNOT-equivalent 3,381 for the greedy lattice tree, recomputable as 99 + 2(547) and 99 + 6(547), with the native column equal to the declared basis and the restore depth cost equal to total minus pre-restore), **Experiment E's seven graph specifications** (including `K9`, the class-2 case inside the measured table: 36 edges, degree floor 8, minimum 9, the peel's 15 → 9 proved by the capacity count) (100-qubit cluster with 180 edges and degree floor 4, the 400-qubit 20x20 cluster with 760 edges, the 900-qubit 30x30 cluster with 1,740 edges, irregular 64-qubit graph, schedules tying on operations/two-qubit count/span and differing only in depth, the minimum's depth equal to `1 + floor` while every other schedule is deeper, the scheduler's per-target `optimal`/`layers_saved` accounting recomputable, **0** on the six class-1 targets and **6** on `K9`, **the per-schedule wall clock** (both seconds columns present, positive and on the row's own circuit, each the median of three recorded samples that recompute to exactly the quoted number and include it as one of the three, with verification dearer than synthesis at every size from 100 qubits up and the minimum's 4 layers the fewest of the five), **the routing block** (90 placements and 72 comparisons counted from the rows, the routing corpus staying the three clusters while K9 is timed and decided, every placed circuit `VALID`, every advantage and surviving fraction recomputed from the two depth columns, the lattice at zero swaps and exactly 1.00x, and the minimum schedule the cheapest to route on a 400-qubit line), the six class-2 rows proved `Δ + 1` with their proof kind recorded (layer capacity at zero nodes for C5, `K5`, `K9`, `K11`, `K13`; an exhausted search for Petersen) and 14 layers saved in total (2 + 6 + 4 + 2), three refusals each naming two endpoint stabilizers, the cluster and GHZ refused against one another at a shared width, five accepted plus one refused graph construction agreed on by both tiers, and the depth-histogram block of the external record naming the same six `graph_layers` circuits this one timed, by SHA-256, depth and width), **the dense tier's measured ceiling** (every width inside the bound `VALID` and every width above it `UNRESOLVED`, the `16 * 4**n` operator model against the traced peak, the fourfold growth per qubit, and the wall named at 15 qubits under the declared 8 GiB budget), the two referees agreeing on all 23 circuits including the refused one, **the 400- and 900-qubit clusters decided by a second implementation** (the stabilizer tier and Stim both `VALID` at 400/400 and 900/900 generators, on the same circuit SHA-256s the timing rows quote, with each width's dense-tier refusal stored as a row of its own, `agree` `null` when the third party abstains, and the three wrong graph constructions refused by **both** referees at 400 qubits with Stim naming its failing generators), **the SABRE seed sweep** (five seeds over the six routed cases, all 30 circuits decided `VALID` on Qiskit's own final layout, every per-case interval and distinct-circuit count recomputed from the rows, the ring tree's **120–144** spread and the chain on a line or ring seed-invariant, with depths compared against our pre-restore column), **the lookahead regression** (a 9-qubit tree on a line at 17 layers greedy against **19** lookahead, pinned with both circuit hashes) and **the placement frontier** (nondominated in depth and CNOT-equivalent gates, partition recomputed, collapsing to one point and reported as that rather than as a trade-off), the **third referee** on the unitary question (11 pairs, all decided, every decided row agreeing with the stabilizer tier, the five schedules of one edge set `equivalent`, the three wrong constructions `not_equivalent`, the 900-qubit pair decided, and the abstention list recomputed from the rows), **the swept lookahead window** (40 rows over 2/4/8/16/32, window 8 reproducing the recorded router by SHA-256, all-to-all window-independent, depth not monotone, the best total below window 8's), **both counting bounds on every target** (`Δ·⌊V/2⌋` and `⌈|E|/ν⌉` recomputed from edges, degree and matching number, neither firing on a class-1 graph, the stronger one quoted) and **the `Δ + 1` overshoot rate** (27 of 41 class-1 graphs, never more than one layer, per-family rates recomputed), and the SHA-256 of the script and six modules that produced it | `experimental` |
| [`tests/test_vqc_native_parity.py`](../tests/test_vqc_native_parity.py) | Python-Rust parity (45 tests): the derived conjugation tables compared entry by entry, images/generators/distributions of a fourteen-circuit corpus, relative versus global phase, the sampler against the exact weights, the fallback with the extension absent, `engine="rust"` refusing instead of downgrading, a corrupted native table refused rather than adjudicated, the parity-limit split between `image_verified_by: "python"` and `"suite"`, and a **sampler corpus** of measured circuits whose allowed-outcome sets are deterministic (teleportation, a mid-circuit reset with feedforward and a two-qubit condition, both condition polarities on one bit), plus `assert_parity` refusing a classically conditioned gate it used to tag as an ordinary one and both engines refusing `ccx` by name | `experimental` |
| [`tests/test_vqc_native_parity_record.py`](../tests/test_vqc_native_parity_record.py) | Record pins (7): the schema, every case agreeing, the script and seven Rust sources identified by SHA-256, each case's circuit rebuilt to its recorded hash, and the parity-limit limitation present rather than implied | `experimental` |
| [`tests/test_vqc_core.py`](../tests/test_vqc_core.py) | The acceptance suite (63 tests): IR round-trip, twelve malformed-program rejections, analytic state comparisons, reset semantics including the entangled case, five channels over five probabilities, seeded trajectory reproducibility, every single-error syndrome, the two-error failure, the logical rate and five receipt verdicts | `experimental` |
| [`tests/test_vqc_qasm.py`](../tests/test_vqc_qasm.py) | OpenQASM 2 interchange tests (15): Bell/GHZ/conditioned round-trips preserving the hash, a hand-written program with parameters and `reset`, and every documented refusal | `experimental` |
| [`tests/test_vqc_phase0_inventory.py`](../tests/test_vqc_phase0_inventory.py) | Phase 0 pins (6): every path the inventory names exists, every VQC module is inventoried, and every frozen DQI record still hashes to its baseline value | `experimental` |
| [`tests/test_vqc_codes.py`](../tests/test_vqc_codes.py) | Code pins: the phase-flip repetition code against its `3p^2 - 2p^3` prediction, and the five-qubit code — encoder state versus a stabilizer-projection reference, all fifteen single-qubit errors mapped to distinct nonzero syndromes, every single error corrected, a two-error logical failure, and the measured logical rate against the exact closed form | `experimental` |
| [`tests/test_vqc_equivalence.py`](../tests/test_vqc_equivalence.py) | Equivalence pins (20): the controlled scope (`RX(2π)` VALID standalone, INVALID under control; an identity VALID in both), `CX` versus `H CZ H` accepted, a relative-phase candidate that reproduces **every** Z-basis probability rejected, a measured candidate refused as a channel, a wrong qubit count refused, an unmet resource claim reported as VALID/UNMET and not promotable, the metric pinned (`I` vs `Z` = `sqrt(2)`, entry deviation 2.0, range `[0, sqrt(2)]`), an **independent oracle** built by bit arithmetic that the execution path and the tableau images must both match, and the report's JSON and reproducibility | `experimental` |
| [`tests/test_vqc_stabilizer.py`](../tests/test_vqc_stabilizer.py) | Stabilizer pins (40): every GHZ outcome probability and random-Clifford Pauli expectation compared with the dense tier for chains the dense tier can hold, then the 100-qubit chain against its closed-form generators and distribution, sampled counts, the X-basis even-parity distribution with no collisions, teleportation with feedforward, and the named refusals. Structural validation: `validate()` passes on random Clifford states and a 100-qubit GHZ state, refuses anticommuting and rank-deficient generator sets by name, a repeated measurement returns the collapsed outcome, a `Y`-eigenstate is 50/50 in `Z` with a certain branch per trajectory, the `H`/`S` conjugations give the `Y`/`±Y` expectations, and the exact-distribution limit is a refusal on both engines past twelve qubits (`2**n` outcomes is a support, not a report) with `probability_of_bitstring` and sampling still exact | `experimental` |
| [`tests/test_vqc_clifford_equivalence.py`](../tests/test_vqc_clifford_equivalence.py) | Clifford equivalence pins (19): the conjugation images against the dense `U P U†`, the two branches agreeing on every pair both can decide, 100-qubit pairs decided (global phase VALID, changed gate INVALID with the image named, chain versus star INVALID at unitary scope and VALID at state scope), `S` versus `S†`, and every domain edge | `experimental` |
| [`tests/test_vqc_decoders.py`](../tests/test_vqc_decoders.py) | Decoder pins (18): matching against the lookup table on every distance-3 syndrome and against a brute-force minimum on all 64 distance-7 syndromes, exactness up to `(d-1)/2`, distance-7 weight-4 miscorrection, the named refusals, two decoders producing identical counts on the same noise, and measured rates against the enumerated prediction at distances 3, 5 and 7 | `experimental` |
| [`tests/test_vqc_discovery.py`](../tests/test_vqc_discovery.py) | Discovery admission pins (12): an equivalent candidate with its claim met promoted, a larger one refused as non-improving (VALID/UNMET), a relative-phase one refused, a phase that is free standalone refused as a subroutine, a measured candidate decided by its distribution receipt, an underpowered claim `UNRESOLVED` and refused, a 100-qubit Clifford candidate decided, and a 20-qubit non-Clifford pair refused rather than passed | `experimental` |
| [`scripts/measure_vqc_stabilizer_ghz.py`](../scripts/measure_vqc_stabilizer_ghz.py) | The stabilizer tier's measured record: 8,188 outcome probabilities and 22 Pauli strings compared with the dense tier (maximum deviation `1.1e-16`), the 100-qubit GHZ run (512 shots, 248/264 split) and its X-basis counterpart (128 shots, 0 collisions, even parity only), with the dense tier's `2**100` cost reported alongside | `experimental` |
| [`benchmarks/quantum_compilation/vqc_stabilizer_ghz.json`](../benchmarks/quantum_compilation/vqc_stabilizer_ghz.json) | The record produced by that script: the agreement rows, the cross-check deviations, the 100-qubit counts and timings, the checks and their measured values, the stated limitations and the reproducibility hashes | `experimental` |
| [`scripts/measure_vqc_stabilizer_scaling.py`](../scripts/measure_vqc_stabilizer_scaling.py) | The scaling measurement: 100/200/500/1,000 qubits in four families (GHZ chain checked against the closed form, GHZ star checked by group membership, ring graph state checked against its defining `X_q Z_(q-1) Z_(q+1)` generators, seeded random Clifford layers checked structurally and by equivalence). Records build/validate/per-shot seconds, the `n²`-label cost against `2**n` amplitudes, peak RSS, and the checks that make each row more than a timing | `experimental` |
| [`benchmarks/quantum_compilation/vqc_stabilizer_scaling.json`](../benchmarks/quantum_compilation/vqc_stabilizer_scaling.json) | The scaling record: 1,000-qubit GHZ in 0.47 s of tableau build and 0.62 s per sample, a 7,500-gate random Clifford in 14.2 s build / 0.65 s validate / 127 s per sample, and two different 1,000-qubit circuits decided from 2,000 Pauli images (rewriting VALID, one-gate mutation INVALID with the image named), with the limitations and code hashes | `experimental` |
| [`tests/test_vqc_stabilizer_scaling.py`](../tests/test_vqc_stabilizer_scaling.py) | Scaling-record pins (7): every family at every size present, every check passed, the recorded cost consistent with the representation, the two equivalence rows decided at 100 and 1,000 qubits, the stated limits present, the script/module hashes, and the 100-qubit rows re-derived from the script | `experimental` |
| [`scripts/measure_vqc_clifford_discovery.py`](../scripts/measure_vqc_clifford_discovery.py) | The end-to-end 100-qubit workflow record: a junked GHZ chain rewritten 222 → 100 operations with 76 certified steps, admission `VALID`/`MET`/`SUFFICIENT` → promoted, six mutated candidates refused with the differing image named, the cheaper star (same state, different unitary) refused, 64-shot sampling with identical support, and an 8-qubit dense cross-check | `experimental` |
| [`benchmarks/quantum_compilation/vqc_clifford_discovery.json`](../benchmarks/quantum_compilation/vqc_clifford_discovery.json) | The workflow record: the per-step verdicts, the mutation refusals, the admission receipt, the sampling row, the state-scope refusal with its reason, the checks, the limitations and the code hashes | `experimental` |
| [`tests/test_vqc_clifford_discovery.py`](../tests/test_vqc_clifford_discovery.py) | Discovery-loop pins (21): every rewrite of five random Clifford circuits certified as the same unitary (a property test of the rule set), commutation-aware cancellation for shared-control/shared-target CNOTs and CZs, the `h y h` identity up to global phase, `SELF_INVERSE` exactly the gates that are their own inverse, the search reaching the 12-qubit ground truth and being promoted, mutation refusals carrying a named image, the state-only candidate cheaper and refused, and the 100-qubit record's totals, admission, hashes and stated limits | `experimental` |

### APEDP-QED — quantum-assisted equation discovery (experimental)

A proposer family, not an adjudicator: a term library, a budgeted selection
objective, six search mechanisms (four classical, a RIME variant and a
QAOA-mediated one), and a referee that grades the resulting equation against a
holdout trajectory. Its one measured experiment, QED-001, asks whether a
quantum-assisted search recovers a generating equation as well as the classical
proposers at an equal budget. **It does not**: 1 of 3 runs against 3 of 3 at 256
fits, and 0 of 3 at 64 fits, for reasons that are measured rather than asserted
(the surrogate's minimiser is not the answer, and the circuit's distribution is
near-uniform over the space). Discussion and evidence in
[`docs/QED_QUANTUM_EQUATION_DISCOVERY.md`](QED_QUANTUM_EQUATION_DISCOVERY.md),
including the false positive this experiment caught in its own first version.

| Module | Purpose | Status |
|---|---|---|
| [`apedp_dyn/qed/__init__.py`](../apedp_dyn/qed/__init__.py) | Package facade **and its stated boundaries**: the library, objective, task, six arms, circuit, referee, record and `CandidateProposal` adapter in one import, with the paragraph that says the QAOA arm optimises an approximate surrogate on a classical simulator and that no advantage is claimed | `experimental` |
| [`apedp_dyn/qed/library.py`](../apedp_dyn/qed/library.py) | The term library: named scalar functions of the observed state, fixed before any seed is drawn, with duplicate names, mis-shaped columns and non-finite values refused by name and `render()` keeping the signs so a proposal reads as an equation | `experimental` |
| [`apedp_dyn/qed/objective.py`](../apedp_dyn/qed/objective.py) | `J(z) = mean residual² + λ|z|` with the coefficients re-fitted by least squares for every selection, a hard budget on **distinct** fits (a repeat is free, and that is the only way an arm may spend less than its budget), and `BudgetExhausted` rather than a quiet overspend | `experimental` |
| [`apedp_dyn/qed/tasks.py`](../apedp_dyn/qed/tasks.py) | A task: development rows, holdout trajectory, library and generating equation, plus the RK4 `rollout` every referee check uses. A diverging candidate is *represented* (`RolloutRun.diverged`, finite rows) rather than raised, and a coordinate with neither a law nor a kinematic identity is refused | `experimental` |
| [`apedp_dyn/qed/arms.py`](../apedp_dyn/qed/arms.py) | Six proposers under one budget: hill climbing, simulated annealing, a bitstring GA, genetic programming over a set-building language (`and`/`or` over term leaves), RIME soft/hard operators on a relaxed vector hardened by `Bernoulli(r)`, and QAOA over a quadratic surrogate. An arm may declare its own answer (`detail["answer"]`), which is how the QAOA arm avoids crediting its pair enumeration to the circuit; an arm that cannot run raises `ArmAbstained` with the number that stopped it | `experimental` |
| [`apedp_dyn/qed/qaoa.py`](../apedp_dyn/qed/qaoa.py) | The circuit: QUBO from exact singleton and pair fits (`1 + m + m(m-1)/2`), diagonal phase separator with an affine rescaling that provably preserves the minimiser, `exp(-iβΣX)` mixer, deterministic pattern-search angle optimisation, seeded measurement, and the honest rotation count (10 `RZ` + 45 `RZZ` + 10 `RX` per layer at ten bits) | `experimental` |
| [`apedp_dyn/qed/referee.py`](../apedp_dyn/qed/referee.py) | Five independent checks — structural recovery, holdout rollout, residual-bootstrap interval, parameter stability under subsampling, and ablation separability measured on the holdout rather than on the search objective — with `VALID`/`ALTERNATIVE`/`INVALID`/`ABSTAIN` as two statements (`status` for prediction, `recovered` for structure). A diverged rollout scores 0.0 *and* says it diverged; the thresholds are marked experiment-local | `experimental` |
| [`apedp_dyn/qed/records.py`](../apedp_dyn/qed/records.py) | The record builder: one condition per budget, an exhaustive fairness sweep outside every search budget, per-seed rows, a cross-condition table, an advantage block computed from the rows, and **circuit probes** that re-run the arm's circuit from its own seed to measure what probability the circuit put on the answer — the number the arm cannot compute, which is what exposed the first version's false positive | `experimental` |
| [`apedp_dyn/qed/proposer.py`](../apedp_dyn/qed/proposer.py) | The `CandidateProposal` adapter: fits on the rows it is handed, emits a closed-form equation with its coefficients, and supplies a `compute_score` that re-evaluates the same library on any frame, so an out-of-sample check is a causal evaluation. No holdout, no thresholds, no self-adjudication (AGENTS.md §1.1); it returns no proposal rather than a worse one | `experimental` |
| [`docs/QED_QUANTUM_EQUATION_DISCOVERY.md`](QED_QUANTUM_EQUATION_DISCOVERY.md) | The experiment, measured: the question it answers and the one it does not, the task and why it is fair (the penalised optimum is the generating support, by a 0.464 % margin over `{x, v, x³}`), the two budget tables, the three measurements that explain the quantum arm's deficit, the false positive the experiment caught in itself, the non-identifiability result (`RIME`'s `{x, v, sin(x)}` at `R² = 0.9993` with a term that costs nothing and coefficients that do not survive subsampling), and the untested routes (QMoD-style differentiable circuits, a wider library, hardware) | `experimental` |
| [`scripts/measure_qed_recovery.py`](../scripts/measure_qed_recovery.py) | The run: two budget conditions (256 and 64 fits of `J` out of a 1024-selection space), six arms, three seeds each, with the fairness sweep, the circuit probes and the record's own checks | `experimental` |
| [`benchmarks/equation_discovery/qed_001_recovery.json`](../benchmarks/equation_discovery/qed_001_recovery.json) | The record produced by that script: task, oracle, per-seed rows with budgets and selections, adjudications with all five checks, the surrogate block with its minimiser and its circuit probes, the advantage block, ten stated limitations and the SHA-256 of the script and nine modules | `experimental` |
| [`tests/test_qed_recovery.py`](../tests/test_qed_recovery.py) | Mechanism pins (37): library refusals, the budget as a real count of distinct fits, the surrogate **exact on a genuinely quadratic objective** (so its error on a real one is a measurement), QUBO energies against the scalar form, rescaling preserving the minimiser, a seeded circuit with probabilities summing to one, every arm terminating within budget, the truth's rollout and bootstrap interval, an ablation that ranks the two removals, a diverging candidate flagged rather than raised, `ALTERNATIVE` for a predictively-equivalent wrong equation, and a structural assertion that the objective handed to an arm carries no holdout attribute | `experimental` |
| [`tests/test_qed_recovery_record.py`](../tests/test_qed_recovery_record.py) | Record pins (19): the oracle re-derived from the record's own task specification, every status agreeing with its own numbers, every budget respected, the classical arms' recovery counts, the quantum arm's deficit **with its per-run counts beside the best-of-three number**, the surrogate's minimiser pinned as not the answer, the circuit probe reproducing the arm it probes (same surrogate minimum, shots and depth), every pin re-hashed, and the limitations present rather than implied | `experimental` |

### PDE/ODE discovery honest limits

| Document | Purpose | Status |
|---|---|---|
| [`docs/pde_discovery_honest_limits.md`](pde_discovery_honest_limits.md) | States what the discovery layer may and may not claim, including the default-config noise fragility | `experimental` |
| [`tests/test_dyn_pde_discovery_noise_robustness.py`](../tests/test_dyn_pde_discovery_noise_robustness.py) | The first fixed-config multi-seed noise sweep backing that note | `experimental` |

### APEDP-X Biopharma, Molecular Physics & Drug Discovery Referee Funnel

| Module | Purpose | Status |
|---|---|---|
| [`apedp_dyn/molecular/chemistry_bridge.py`](../apedp_dyn/molecular/chemistry_bridge.py) | Deterministic molecular graph construction (`MolecularGraph`, `AtomNode`, `BondEdge`), SMILES/SELFIES parsing, valence/octet/Bredt's rule sanitization, exact topological/physicochemical descriptors (`MW`, `logP`, `HBD`, `HBA`, `TPSA`, `SAScore`, `QED`), and 3D metric-matrix distance geometry conformer embedding | `production` |
| [`apedp_dyn/molecular/equivariant_pocket_flow.py`](../apedp_dyn/molecular/equivariant_pocket_flow.py) | Pocket-conditioned $SE(3)$-equivariant conformer relaxation with atom-typed harmonic bonds, angles, torsions, soft-core 12-6 Lennard-Jones, distance-dependent dielectric Coulomb electrostatics, chirality signed-volume preservation, steric clash adjudication, and analytical Hawkins-Cramer-Truhlar (HCT) Generalized Born / Surface Area (GB/SA) implicit solvation (`compute_gbsa_binding_energy`) evaluating gas-phase $E_{\text{MM}}$, polar desolvation $\Delta G_{\text{GB}}$, and nonpolar cavitation $\Delta G_{\text{SA}}$ via LCPO SASA | `experimental` |
| [`apedp_dyn/epistemic/retrosynthesis_gate.py`](../apedp_dyn/epistemic/retrosynthesis_gate.py) | Deterministic retrosynthetic referee wrapping multi-step reaction template disconnection search (`<= 4` steps to commercial stock building blocks) and synthetic accessibility gate (`SAScore < 3.5`) | `audit-only` |
| [`apedp_dyn/biophysics/__init__.py`](../apedp_dyn/biophysics/__init__.py) | Package facade for biophysical free energy perturbation and split-conformal ADMET safety referees | `experimental` |
| [`apedp_dyn/biophysics/alchemical_fep_gateway.py`](../apedp_dyn/biophysics/alchemical_fep_gateway.py) | Alchemical Free Energy Perturbation (`AlchemicalFEPGateway`) implementing Zwanzig exponential averaging and self-consistent Bennett Acceptance Ratio (BAR) with uncertainty gate ($\sigma_{\Delta\Delta G} \le 0.5\text{ kcal/mol}$), paired with `ConformalADMETReferee` ($95\%$ distribution-free intervals on $\log S$, hERG $IC_{50}$, and microsomal $t_{1/2}$) | `experimental` |
| [`apedp_dyn/problems/tmdd_pkpd.py`](../apedp_dyn/problems/tmdd_pkpd.py) | Preregistered benchmark `PS-TMDD` for 2-compartment Target-Mediated Drug Disposition (TMDD) pharmacokinetics & receptor occupancy coupled ODEs with exact receptor conservation $d(R + CR)/dt = 0$ | `experimental` |
| [`apedp_dyn/epistemic/drug_discovery_pipeline.py`](../apedp_dyn/epistemic/drug_discovery_pipeline.py) | 6-layer fail-closed drug discovery funnel (`MMLMMolecularProposer`, `VSASynthonDrugGenerator` $\to$ `DrugDiscoveryPipelineReferee` $\to$ `FullLineageExperimentLedger`) enforcing strict proposer/referee separation, de novo VSA-guided synthon assembly guaranteed retrosynthetically valid by construction, holdout contamination burning (`HoldoutContaminationError`), and wet-lab assay confirmation boundaries | `experimental` |
| [`tests/test_dyn_drug_discovery_pipeline.py`](../tests/test_dyn_drug_discovery_pipeline.py) | Regression suite covering SMILES/SELFIES valence sanitization, Bredt's rule, joint $SE(3)$ pocket equivariance, retrosynthesis disconnection, BAR free energy, conformal ADMET, TMDD receptor conservation, GB/SA implicit solvation, VSA synthon lead generation, and 6-layer holdout burning | `experimental` |

### APEDP-MMLM — Multi-Mathematical Language Model Architecture (experimental)

| Module | Purpose | Status |
|---|---|---|
| [`apedp_mmlm/__init__.py`](../apedp_mmlm/__init__.py) | Package facade for the three-track Multi-Mathematical Language Model research architecture (`Track A` non-neural, `Track B` ablation-governed hybrid, `Track C` attention-free replacement) and quantum discovery | `experimental` |
| [`apedp_mmlm/engines.py`](../apedp_mmlm/engines.py) | The five mathematical engines: Engine 1 (Weighted Automata, Decomposable Probabilistic Circuits, Bayesian Dirichlet Engine), Engine 2 (1D Tensor Train with DMRG two-site variational sweeping, SVD Schmidt spectrum, and von Neumann entanglement entropy $S_{\text{vN}}$, & Binary Hierarchical Tree Tensor Network), Engine 3 (Streaming Dynamical Systems & EDMD Koopman Operators), Engine 4 (Concept Hypergraphs & FFT Vector Symbolic Architectures), Engine 5 (Global Sequence Energy Evaluator, Polynomial Speculative Energy Decoder, and Epistemic Adjudicator), and Byte-Pair Encoding (`BPEMMLMTokenizer`) with UTF-8 byte fallback | `experimental` |
| [`apedp_mmlm/models.py`](../apedp_mmlm/models.py) | Models `NonNeuralMMLM_A0`, `HybridMMLM_B0` and `AttentionFreeMMLM_C0`, plus small comparison baselines. `Milestone1BenchmarkSuite` defaults to a deterministic synthetic subject-relation-object split (75/25/25), fits its tokenizer on training text only, and accepts an optional train/dev/test corpus carrying source, license, retrieval date, and per-split hashes. Those declarations are not independently verified, no suitable external MMLM corpus is bundled, and fixed retrieval/ListOps probes remain synthetic. H2 is measured as B0 dynamical-only streaming next-token accuracy against the frozen 0.70 threshold; H5 requires B0 or C0 to win both majority-based quality and efficiency axes against each baseline. Results and repeated-seed variability remain descriptive, not significance tests or external validation unless that data is actually supplied and audited. | `experimental` |
| [`apedp_mmlm/quantum_discovery.py`](../apedp_mmlm/quantum_discovery.py) | Quantum-reach MMLM circuit discovery: Universal ZX-Calculus & Clifford+T synthesis (`ZXSpider`, `ZXDiagram`, `optimize_clifford_plus_t_zx`, `mmlm_zx_t_count`), Local Complementation + symplectic CX graph-state synthesis (`synthesize_graph_state_mmlm`), general stabilizer tableau synthesis (`synthesize_stabilizer_target_mmlm`, `synthesize_any_target_mmlm`), commutation-aware Clifford rewriting (`mmlm_clifford_search`), DAG front-layer lookahead routing (`place_with_dag_lookahead`), fault-tolerant compilation proposers (`MMLMInvSqrtProposer`, `MMLMDQIScheduleOptimizer`, `MMLMDfthcsosOptimizer`), and sparse cubic HUBO + Koopman QAOA equation discovery (`MMLMQEDProposer`). Measured on the corpus's 75 graph states: **54 wins / 21 ties / 0 losses**, **308 two-qubit gates saved** (6,193 $\to$ 5,885) — split by family `random_graph` **25/25 wins, 238 gates**, `regular3` **17/25 wins, 29 gates**, `regular4` **12/25 wins, 41 gates**, and thinning with width on the regular families (`regular4` 0 gates at 32q, `regular3` 3 at 64q), so **77% of the total is one family** — 75/75 certified `VALID` by `adjudicate_specification`, recorded in [`benchmarks/quantum_compilation/vqc_mmlm_graph_state.json`](../benchmarks/quantum_compilation/vqc_mmlm_graph_state.json) and reproduced by [`scripts/measure_vqc_mmlm_graph_states.py`](../scripts/measure_vqc_mmlm_graph_states.py), with one measured negative result attached: a bounded plateau walk (`max_plateau_steps`, up to 8 zero-delta moves past the 2-step lookahead, LC and CX ranked together) reproduces **54 wins / 308 gates exactly**, so the thinning on `regular3`/`regular4` is a limit of the move set rather than of search depth | `experimental` |
| [`mmlm_matter_001/__init__.py`](../mmlm_matter_001/__init__.py) | MMLM-MATTER-001 package facade: the Phase 1 model-interoperability slice of [`docs/MMLM-MATTER-001_Implementation_Blueprint.md`](MMLM-MATTER-001_Implementation_Blueprint.md). Separate from `apedp_mmlm` (mathematical language models); nothing in the package accepts a candidate, because verification is the referee's. | `experimental` |
| [`mmlm_matter_001/models/contract.py`](../mmlm_matter_001/models/contract.py) | `ModelContract`: the frozen provider/model/prompt-hash/schema-hash identity of one proposer arm, with drift refusals (`PromptDriftError`, `SchemaDriftError`) and `retarget()` so a model can be chosen after preregistration without disturbing the pins (blueprint §12.6, §11.1). | `experimental` |
| [`mmlm_matter_001/models/prompts.py`](../mmlm_matter_001/models/prompts.py) | Versioned provider-neutral P1 topology-ranking and P2 direct-design prompt templates; the system prompt and template text are hashed together so `prompt_hash` describes everything actually sent. | `experimental` |
| [`mmlm_matter_001/models/adapters.py`](../mmlm_matter_001/models/adapters.py) | Provider-agnostic model transports: one OpenAI-compatible chat-completions adapter (OpenRouter, OpenAI, DeepSeek, NVIDIA NIM, Aihubmix, or any local vLLM/Ollama endpoint via `base_url`) and a deterministic `ReplayAdapter` over a hash-keyed JSONL cassette. A refused request, missing key or malformed body raises `ModelUnavailableError` instead of returning an empty proposal. | `experimental` |
| [`mmlm_matter_001/models/outputs.py`](../mmlm_matter_001/models/outputs.py) | Strict Pydantic output schemas and parsers for the P1 ranking and P2 design: reject malformed JSON, unknown catalogue ids, duplicate ids, duplicate ranks, over-long selections and non-physical dimensions, each with a machine-readable reason (blueprint §10.2). | `experimental` |
| [`mmlm_matter_001/schemas/task_spec.py`](../mmlm_matter_001/schemas/task_spec.py) | `PublicTaskView` (everything a proposer may see), `PublicFeedback` (this run's own public attempt history) and `SealedTask` (planted topology, dimensions and hidden seed) as a type boundary, so hidden ground truth cannot be included in a prompt (blueprint §10.2, §17). | `experimental` |
| [`mmlm_matter_001/schemas/topology_spec.py`](../mmlm_matter_001/schemas/topology_spec.py) | `TopologyEntry`: the public catalogue metadata (id, link/joint counts, inversion, ground/input/output links, parent base chain, canonical graph hash) that P0 and P1 both receive. | `experimental` |
| [`mmlm_matter_001/proposers/p1_llm_topology.py`](../mmlm_matter_001/proposers/p1_llm_topology.py) | P1 arm: ranks catalogue topologies under the frozen contract and returns an ordered selection plus an LLM usage record. It has no acceptance surface — no verdict field, no adjudication method (blueprint §10.2). | `experimental` |
| [`mmlm_matter_001/proposers/p2_llm_direct.py`](../mmlm_matter_001/proposers/p2_llm_direct.py) | P2 ablation arm: proposes a topology and dimensions with no numerical optimiser in the loop, using the same checkers and acceptance threshold as P0/P1 (blueprint §10.3). | `experimental` |
| [`mmlm_matter_001/evaluation/budget_accountant.py`](../mmlm_matter_001/evaluation/budget_accountant.py) | LLM usage rows (`LLMUsage`) and the append-only `BudgetLedger` for `budget_ledger.jsonl`: tokens, latency, model/prompt/schema/contract hashes and cost, with aggregation that reports completeness and keeps a provider's unreported count as `None` rather than `0` (blueprint §11.1, §11.2). | `experimental` |
| [`mmlm_matter_001/protocol/freeze.py`](../mmlm_matter_001/protocol/freeze.py) | The frozen Phase 1 protocol bundle (blueprint §12.6): prompt, schema and (once a model is chosen) contract hashes for P1 and P2 in one file that carries its own hash, refuses to be silently replaced, detects hand edits, and raises `ProtocolDriftError` when the checkout's prompt or schema no longer matches what was preregistered. `pin_models=False` freezes the protocol before a provider is selected. The bundle also carries the Phase 0 preregistration (below), so one file pins both the model protocol and the checker thresholds, and `verify_preregistration` refuses a bundle frozen with different ones. | `experimental` |
| [`mmlm_matter_001/protocol/preregistration.py`](../mmlm_matter_001/protocol/preregistration.py) | The preregistered Phase 0 thresholds (blueprint §5.2, §5.7, §12.6): the Grashof safety margin and the T2 angular tolerance in one self-hashed record. `grashof()` runs the §5.2 test with the pinned margin so no call site supplies its own, `require_function_tolerance` refuses a T2 task whose declared tolerance differs from the pinned one, and the record is stored inside the frozen protocol bundle where a hand edit is detected. It also pins the measurement settings the thresholds are read under — `path_samples` (enforced by `require_path_samples`), `continuity_max_jump_deg`, `cognate_tolerance_mm` and the loop-closure tolerances — because a pass on a coarser grid is a different claim. | `experimental` |
| [`mmlm_matter_001/kinematics/fourbar.py`](../mmlm_matter_001/kinematics/fourbar.py) | Checker V2 for four-bars (blueprint §5.1–§5.4): canonical coordinate convention, strict Grashof test with a caller-declared margin and full inversion classification, closed-form position analysis through the half-angle tangent that keeps both signed-square-root assembly branches, branch/circuit continuity, closed-form input and output dead centres classified separately, and coupler-path tracing. `poses()` exposes both assembly roots without choosing one; `trace` follows a circuit by joint continuity by default (`circuit=False` reproduces the old label sweep for forensics) and `pose_near`/`trace_continuous` are the primitives behind it (the branch label is not a root selector — both roots can carry it) and stop at a jump beyond a declared bound; `circuit_breaks` counts those breaks from the geometry. Referee-side: it may return a verdict, a proposer never may. | `experimental` |
| [`mmlm_matter_001/kinematics/path_metric.py`](../mmlm_matter_001/kinematics/path_metric.py) | The T1 path metric (blueprint §5.6): exact discrete Fréchet distance over the ordered target and traversed coupler path, plus `check_path`, which aligns the traversed segment, refuses a reversed orientation unless the task allows it, and reports the samples and sampling metadata with the error rather than calling it a continuous certificate. A truncated trace is never scored (`partial_trace_not_scored`, `frechet_mm=None`); `refine=True` re-scores at increasing sample counts and reports whether the distance settled (`samples_tested`, `frechet_change_mm`, `refinement_stable`) with the sampling resolution `resolution_mm`; `continuity=True` judges circuit continuity from the joint angles (`Trace.continuity_tracked`) instead of the branch label; and `verify_loop_closure=True` adds a second opinion from the independent numerical solver on every pose the check accepted. | `experimental` |
| [`mmlm_matter_001/kinematics/function_metric.py`](../mmlm_matter_001/kinematics/function_metric.py) | The T2 function-generation metric (blueprint §5.7): solves the mechanism at every declared input on the requested branch and reports wrapped angular error as max and RMS against the task's preregistered tolerance, rejecting unreachable inputs, branch changes and samples that do not cover the declared input range. `score_function_samples` re-scores already-solved poses, so a recorded run can be adjudicated without re-solving it, and `verify_loop_closure=True` adds the independent numerical solver's second opinion on every pose it scored, so a T2 pass does not rest on the closed form alone. The declared inputs are scored on one assembly — the first on the requested branch, each later one on the root nearest its predecessor — so two roots sharing a branch label cannot hop the score onto the other circuit. | `experimental` |
| [`mmlm_matter_001/kinematics/loop_closure.py`](../mmlm_matter_001/kinematics/loop_closure.py) | The independent numerical loop-closure solver (blueprint §5.5, §5.8): damped Newton on the two loop-closure residuals, seeded from a blind grid over the joint-angle torus and sharing no algebra with the closed form, plus `cross_check`, which draws deterministic strict-Grashof crank-rockers and compares both assembly branches against the closed form at every sampled input, reporting the maximum coupler/output angle deviation and residual. `verify_poses` is the per-pose form used by the T1 path check, and every returned root is judged only on the loop-closure residual. | `experimental` |
| [`mmlm_matter_001/kinematics/cognates.py`](../mmlm_matter_001/kinematics/cognates.py) | The Roberts–Chebyshev cognate construction (blueprint §5.8), derived from the loop closure rather than copied: with `w = (P−A)e^{−i theta3}` the shared focal pivot is `(w/beta)g` and the two cognates' crank/coupler/output vectors follow, so all three coupler points coincide at `P` at every instant. `check_cognates` re-solves each cognate with its *own* closed form driven by the assembly's crank angle and reports the focal-pivot drift, the coupler-point gap, the ordered Fréchet distance and the symmetric Hausdorff distance between the point *sets* (which is what "the same curve" means when each cognate is driven independently and therefore sampled differently) against the declared tolerance, plus the angular shortfall of the nearest assembly (`coupler_angle_gap_deg`, so a cognate that cannot sit at the constructed configuration reports a measured gap rather than only a reason, judged against a preregistered `cognate_angle_tolerance_deg`) and each cognate's Grashof inversion. | `experimental` |
| [`mmlm_matter_001/kinematics/topology_check.py`](../mmlm_matter_001/kinematics/topology_check.py) | The topology half of a T3 answer (blueprint §2.2, §2.4, §2.5): Grübler degrees of freedom for a planar revolute chain, three distinct link roles, and the inversion the dimensions actually produce must be the inversion the catalogue entry declares. A structure this checkout cannot verify — the blueprint's 6-, 8- and 10-link universe — is refused as `topology_unsupported_structure` rather than passed on its metadata. | `experimental` |
| [`mmlm_matter_001/kinematics/boundary.py`](../mmlm_matter_001/kinematics/boundary.py) | Runtime half of the proposer/referee boundary (AGENTS.md §1.1): every public checker asks who called it and raises `BoundaryError` when the answer is `mmlm_matter_001.proposers` or `mmlm_matter_001.models`. The lint rule stops the import; this stops the accident, including a dynamic import that the build-time rule cannot see. It inspects the whole calling stack rather than only the immediate caller, and checks each frame's source path as well as its module name, so rewriting `__name__` does not launder a call. It is a boundary, not a sandbox. | `experimental` |
| [`mmlm_matter_001/pipeline/gate.py`](../mmlm_matter_001/pipeline/gate.py) | The Phase 0 gate (blueprint §5.2, §5.6, §5.7, §12.6): runs the Grashof test with the pinned margin, scores T1 with the pinned sampling and solver tolerances (continuity-tracked roots, refinement-checked distance, loop-closure second opinion on every accepted pose), T2 against the pinned angular tolerance with the same second opinion, and T3 as topology validity plus the task's own metric, optionally adding the cognate check. A verdict is `accepted`, `refused` or `unverifiable`: an answer this checkout has no checker for is a capability gap, not a candidate failure, and the ledger counts it separately so it cannot inflate a failure rate. It returns one `GateVerdict` carrying the digests and effective settings it was judged under, refuses to run without a preregistration, refuses a sample count that is not the pinned one, refuses a bundle frozen to other thresholds, and refuses a proposer caller. | `experimental` |
| [`mmlm_matter_001/evaluation/verdict_ledger.py`](../mmlm_matter_001/evaluation/verdict_ledger.py) | Append-only `verdicts.jsonl` of Phase 0 verdicts, kept separate from the budget ledger so cost aggregation never has to filter foreign rows: every row carries the candidate, the preregistration and protocol digests, the effective settings, the measurement records and the reasons, so a refusal can be audited after the fact instead of living only in a printed return value. `summary()` reports accepted/refused counts and per-reason tallies. | `experimental` |
| [`mmlm_matter_001/pipeline/dry_run.py`](../mmlm_matter_001/pipeline/dry_run.py) | Offline Phase 1 dry run: P1 runs against a deterministic local stub catalogue through the real prompt, parser and protocol pins, appending one `budget_ledger.jsonl` row per call. The stub names itself `stub` in every row and has zero prices, so the ledger arithmetic is exercised without a provider or a fake model identity. | `experimental` |
| [`tests/test_mmlm_matter_001_protocol_and_phase0.py`](../tests/test_mmlm_matter_001_protocol_and_phase0.py) | Pins the protocol bundle against hand edits, silent replacement and prompt/schema drift; runs the dry-run pipeline offline and checks the ledger; and covers the Phase 0 four-bar checker — Grashof strictness/margin/inversion, loop closure on both branches, unreachable inputs, circuit continuity, toggle classification, Fréchet order preservation and the T1 pass/fail boundary. | `experimental` |
| [`tests/test_mmlm_matter_001_model_interop.py`](../tests/test_mmlm_matter_001_model_interop.py) | Pins the Phase 1 model layer with no network or credentials: contract hashing and drift refusals, every P1/P2 rejection reason, sealed-task leakage guards, transport-failure refusals, replay determinism and the unpriced/unexposed accounting rules. | `experimental` |
| [`tests/test_apedp_mmlm_architecture.py`](../tests/test_apedp_mmlm_architecture.py) | Verification suite testing the mathematical engines, component ablations, and quantum discovery. Its Milestone 1 pin checks disjoint train/dev/test triples, train-only tokenizer coverage, the reported protocol metadata, and explicit `NOT_ASSESSED` outcomes for hypotheses whose metrics are absent; measured ablation comparisons are not required to pass | `experimental` |
| [`tests/test_dyn_basis_proposer_transfer.py`](../tests/test_dyn_basis_proposer_transfer.py) | Fits a polynomial basis proposer on development observations for two distinct synthetic equation families and evaluates the returned frozen scorer on disjoint out-of-range observations; this pins code-path transfer only, not empirical cross-domain science | `experimental` |

*Documentation generated for APEDP-X v2.0.0 — Autonomous Physics, Engineering & Market Discovery Platform*
## What this repository cannot certify about itself

Every referee named in this index — the stabilizer and dense tiers, Stim, MQT QCEC, the
canonical ledger, the DSR machinery, the burn-log seal — was chosen, installed, wired and
interpreted inside this repository. That gives an unusually strong *internal* adversary (a
wrong circuit, a snooped holdout, an uncharged fit each fail loudly) and no *external* one:
there is no human peer review of the records and no independent replication.

Three limits follow, stated so they are not mistaken for gaps a code change would close:

- **Tamper-evidence is local.** [`holdout_seal.py`](../apedp_dyn/epistemic/holdout_seal.py)
  detects truncation, an edited burn and a rewind, and its digest is pinned by a test, so a
  rewind has to change a test to pass. Someone who rewrites both the log and the pin defeats
  it; no key and no external witness is involved. An append-only remote store
  (`FullLineageExperimentLedger(storage_backend=...)`) is the stronger form and is available,
  but is not the default.
- **Referees share a bridge.** The stabilizer and dense tiers share `gates.matrix_for` and
  the parser; Stim and QCEC are reached through this package's own exporters. A shared
  translation bug would be invisible to every tier at once, which is why the agreement
  corpora are reported as conformance samples.
- **The registry is prose as well as data.** The checks enforce that every module is
  indexed, that status tags match docstring `Status:` lines, and that links resolve — not
  that every sentence is true. Measurements are pinned by tests; the narrative around them
  is not.

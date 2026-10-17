# APEDP Documentation

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
13. [Small Physics LLM](#13-small-physics-llm)
14. [Contributing](#14-contributing)
15. [License](#15-license)
16. [System Capabilities & Architectural Philosophy](SYSTEM_CAPABILITIES_AND_PHILOSOPHY.md)
17. [Evidence, Adapters & Domain Limits](#17-evidence-adapters--domain-limits)
18. [APEDP-X Dynamic Discovery, Epistemic Referee & CPMD Architecture](#18-apedp-x-dynamic-discovery-epistemic-referee--cpmd-architecture)
19. [Traceability & Verification Matrix](#19-traceability--verification-matrix)
20. [Transformer Proposer Design — Q&A Record](TRANSFORMER_PROPOSER_DESIGN_QA.md)
21. [Sparrow Quantum Deterministic Photonics Blueprint](SPARROW_QUANTUM_DETERMINISTIC_PHOTONICS.md)

---

## 1. Overview

APEDP (Autonomous Physics & Engineering Discovery Platform) is an end-to-end scientific computing platform for discovering, validating, and repairing physical equations from experimental data. It combines symbolic regression, dimensional analysis, Bayesian model selection, physical anomaly repair, epistemic tracking, quantum error correction, metrology, and applied defect auditing in a unified framework.

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
additive offsets such as `2*x**2 + 3` are exactly recoverable.
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

- `ingestion_academic_apis.py` — Academic API connectors (arXiv, DOI)
- `ingestion_docs.py` — DOCX/PDF document loaders
- `ingestion_web_equations.py` — Wikipedia equation harvesting

### 5.10 `apedp/symbolic_regression.py` — Symbolic Regressor

Standalone symbolic regression with evolutionary search.

```python
from apedp.symbolic_regression import SymbolicRegressor

regressor = SymbolicRegressor(engine, ['x', 'y'], 'meter/second')
result = regressor.fit(data, target)
```

---

## 6. API Server

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

| Module | Description | Status |
|---|---|---|
| [`apedp_dyn/conf/canonical_spec.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/conf/canonical_spec.py) | Canonical preregistered configuration loader enforcing SHA-256 integrity on frozen specs. | `production` |
| [`apedp_dyn/conf/benchmark_v2.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/conf/benchmark_v2.py) | Benchmark v2 configuration schemas, trap suites, and seed schedules. | `production` |
| [`apedp_dyn/conf/dimensionless.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/conf/dimensionless.py) | Dimensional homogeneity rules and physical dimensionless parameter bounds. | `production` |
| [`apedp_dyn/epistemic/experiment_ledger.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/experiment_ledger.py) | Full-lineage candidate experiment ledger with anti-snooping holdout burning. | `production` |
| [`apedp_dyn/epistemic/trial_ledger.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/trial_ledger.py) | Candidate trial ledger computing Bailey & López de Prado (2014) Deflated Sharpe Ratio. | `production` |
| [`apedp_dyn/epistemic/promotion_lifecycle.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/promotion_lifecycle.py) | Six-stage scoped promotion state machine with non-fungible sequential gates. | `production` |
| [`apedp_dyn/epistemic/novelty_adjudication.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/novelty_adjudication.py) | Decoupled Diebold-Mariano predictive incrementality and literature orthogonal novelty tests. | `production` |
| [`apedp_dyn/epistemic/lean_universal_bridge.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/lean_universal_bridge.py) | Zero-disk in-memory Lean 4 formal verification kernel bridge via `--stdin` streaming. | `production` |
| [`apedp_dyn/epistemic/adjudicate.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/adjudicate.py) | Central epistemic referee adjudicating multi-gate candidate admissibility. | `production` |
| [`apedp_dyn/epistemic/scope_refusal.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/scope_refusal.py) | Referee-side out-of-grammar scope refusal (v0.1.4, spec §16d): preregistered family-distinct probes (exp rate 1–50 on dev-normalized axis, log on raw axis) fit from dev data only; an emitted candidate promotes only on UNIQUE_SUPPORT over every probe; blind to world flags and planted truth. | `experimental` |
| [`apedp_dyn/epistemic/claim_ladder.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/claim_ladder.py) | Epistemic claim ladder calibrating warrant strength from speculative to admitted. | `production` |
| [`apedp_dyn/epistemic/cross_asset_gate.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/cross_asset_gate.py) | Multi-asset cross-sectional transfer gate enforcing zero contamination across symbols. | `production` |
| [`apedp_dyn/epistemic/market_extrapolation.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/market_extrapolation.py) | Out-of-distribution regime transfer gate evaluating volatility and liquidity shocks. | `production` |
| [`apedp_dyn/epistemic/frozen_spec.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/frozen_spec.py) | Cryptographic hash verification of referee source files to prevent silent code drift. | `production` |
| [`apedp_dyn/epistemic/dimensional_homogeneity.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/dimensional_homogeneity.py) | Typed SI physical dimensional homogeneity engine and epistemic gate (Option 3): validates term addition compatibility, Buckingham-Pi constraints, and dimensionless transcendental arguments. | `experimental` |
| [`apedp_dyn/epistemic/eedr.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/eedr.py) | Epistemic Evidence-to-Decisiveness Ratio allocating exploration budget. | `production` |
| [`apedp_dyn/grammar/proposer_interface.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/proposer_interface.py) | Unified abstract base class and dataclass contracts for competing hypothesis proposers. | `production` |
| [`apedp_dyn/grammar/trading_grammar_catalog.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/trading_grammar_catalog.py) | Grammar definitions and AST transforms for causal microstructure and order-flow signals. | `production` |
| [`apedp_dyn/grammar/scientific_grammar_catalog.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/scientific_grammar_catalog.py) | Grammar definitions for physical, biological, fluid, and geometric dynamical systems. | `production` |
| [`apedp_dyn/grammar/candidate_ir.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/candidate_ir.py) | Intermediate representation for symbolic expressions and parse tree canonicalization. | `production` |
| [`apedp_dyn/proposers/sparse_microstructure_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/sparse_microstructure_proposer.py) | Sparse ElasticNet hypothesis proposer generating sparse linear microstructure models. | `production` |
| [`apedp_dyn/proposers/resiliency_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/resiliency_proposer.py) | Nonlinear order-book resiliency proposer fitting exponential recovery dynamics. | `production` |
| [`apedp_dyn/proposers/distributed_lag_interaction_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/distributed_lag_interaction_proposer.py) | Distributed lag kernel and bilinear feature interaction microstructure proposer. | `production` |
| [`apedp_dyn/proposers/regime_switching_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/regime_switching_proposer.py) | Piecewise linear threshold models adaptive to market volatility states. | `production` |
| [`apedp_dyn/proposers/hawkes_orderflow_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/hawkes_orderflow_proposer.py) | Self-exciting point process modeling order arrival clustering and quote replenishment. | `production` |
| [`apedp_dyn/proposers/rime_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/rime_proposer.py) | RIME symbolic-mechanism proposer: emits closed-form equations from typed mechanism graphs, adjudicated by the same frozen referee. | `experimental` |
| [`apedp_dyn/proposers/stem_proposers.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/stem_proposers.py) | Modular multi-engine STEM proposers (Option 2): Hamiltonian energy-conserving mechanical systems, solenoidal divergence-free fluid velocity fields, and compartmental epidemiological dynamics. | `experimental` |
| [`apedp_dyn/proposers/invariant_transfer_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/invariant_transfer_proposer.py) | Active cross-regime invariant transfer proposer: IRM-penalized representations with worst-regime retention gating, Cross-Regime Generalization Index (CGI) ranking, and active extrapolation-barrier checks. | `production` |
| [`apedp_dyn/sandbox/execution_cost_models.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/execution_cost_models.py) | Decoupled execution models (Spread+Slippage, Almgren-Chriss impact, Queue fill). | `production` |
| [`apedp_dyn/sandbox/calibrated_benchmark_suite.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/calibrated_benchmark_suite.py) | Three-tier calibrated benchmark suite (Referee FPR/CAR/VRR, Discovery Power, Transfer). | `production` |
| [`apedp_dyn/sandbox/canonical_ledger.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/canonical_ledger.py) | Canonical execution ledger with additive log accounting, stock-split invariance, and fee models. | `production` |
| [`apedp_dyn/sandbox/blind_benchmark.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/blind_benchmark.py) | Air-gapped blind benchmark runner preventing in-sample contamination of test suites. | `production` |
| [`apedp_dyn/sandbox/blind_benchmark_protocol.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/blind_benchmark_protocol.py) | Cryptographic verification protocol for blind benchmark suites and hash receipts. | `production` |
| [`apedp_dyn/sandbox/ga_depth2_equation_discovery.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/ga_depth2_equation_discovery.py) | Genetic algorithm search across depth-2 symbolic equation grammar space. | `experimental` |
| [`apedp_dyn/sandbox/ga_depth3_equation_discovery.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/ga_depth3_equation_discovery.py) | Genetic algorithm search across depth-3 symbolic equation grammar space. | `experimental` |
| [`apedp_dyn/sandbox/build_all_system_equations_lean_proofs.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/build_all_system_equations_lean_proofs.py) | Cross-domain Lean 4 formal proof suite for 10 system equations. | `production` |
| [`apedp_dyn/sandbox/cpmd_hlik.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/cpmd_hlik.py) | Continuous Predictive Market Discovery harness for H-LIK liquidity hypothesis. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_hxsc.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/cpmd_hxsc.py) | Continuous Predictive Market Discovery harness for H-XSC spread curvature hypothesis. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_hdar.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/cpmd_hdar.py) | Continuous Predictive Market Discovery harness for H-DAR directional asymmetry hypothesis. | `audit-only` |
| [`apedp_dyn/sandbox/cpmd_hofd.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/cpmd_hofd.py) | Continuous Predictive Market Discovery harness for H-OFD order-flow divergence hypothesis. | `audit-only` |
| [`apedp_dyn/sandbox/eseds_dynamical.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/eseds_dynamical.py) | Energy-dissipation and dynamical stability referee harness. | `audit-only` |
| [`apedp_dyn/sandbox/eso_packing.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/eso_packing.py) | Geometric sphere-packing contact invariant referee harness. | `audit-only` |
| [`apedp_dyn/discovery_l2/ode_engine.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/discovery_l2/ode_engine.py) | Higher-order ODE numerical integration and parameter estimation engine. | `production` |
| [`apedp_dyn/discovery_l2/pruning.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/discovery_l2/pruning.py) | Symbolic expression AST simplification, zero-coefficient pruning, and caching. | `production` |
| [`rust/apedp-kernel/src/genetic.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/genetic.rs) | Rust-accelerated genetic algorithm evaluator with verified minimum-MSE selection, top-level trig frequency/phase template seeding (`sin\|cos(f*x + p)`, rate 0.18, `allow_trans`-gated; closes the frozen N04 unreachability forensics), and post-evolution constant polish (golden-section coordinate descent with shrinking windows + Powell conjugate-direction escape, strict-improvement replacements, affine readout refit at every probe; never-worsens A/B pinned). | `production` |
| [`rust/apedp-kernel/src/video.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/video.rs) | Native video kernel: global PSNR and windowed grayscale SSIM, zero-copy Sliced Wasserstein-2, Rayon-parallel 2D bilinear sub-pixel coordinate warping (`warp_bilinear_2d`), central-difference optical flow divergence (`compute_optical_flow_divergence`), and disocclusion mask detector (`detect_disocclusion_mask`). Exact-parity pinned against Python reference. | `production` || [`apedp_dyn/video/decoupled_transport.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/video/decoupled_transport.py) | Decoupled Continuous Spatio-Temporal Transport (DC-STT) PDE engine: continuous advection, Helmholtz-Hodge Leray projection ($\nabla \cdot \mathbf{v} = 0$), disocclusion residual source injection ($\mathbf{s} \odot \mathbf{m}_{\text{dis}}$), and Rayon-parallel Rust sub-pixel coordinate warping (`video_warp_bilinear_2d`). Honest scope (spec §5 correction): transport/reconstruction claims given a supplied flow field — no learned prior, no generation claim. Spec: [`docs/DECOUPLED_CONTINUOUS_SPATIOTEMPORAL_TRANSPORT_SPEC.md`](DECOUPLED_CONTINUOUS_SPATIOTEMPORAL_TRANSPORT_SPEC.md). | `experimental` |
| [`rust/apedp-kernel/src/rime/graph.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/graph.rs) | RIME typed mechanism graph: series-at-a-time topological evaluation, `State` recursion, structural validation, canonical SHA-256 ids. | `experimental` |
| [`rust/apedp-kernel/src/rime/rewrite.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/rewrite.rs) | The five frozen RIME structural rewrites (lag, interaction, state, nonlinearity wrap, prune). | `experimental` |
| [`rust/apedp-kernel/src/rime/eval.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/eval.rs) | RIME scorer: ridge-fitted linear readout, charged degrees of freedom, parallel batch evaluation. | `experimental` |
| [`rust/apedp-kernel/src/rime/seeds.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/seeds.rs) | Deterministic 60-form seed catalogue (levels, delays, differences, lead-lag interactions, memory, wrappers). | `experimental` |
| [`rust/apedp-kernel/src/rime/search.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/search.rs) | Deterministic budgeted beam search over the frozen rewrites; null-model-anchored gate with EBIC selection charge. | `experimental` |
| [`rust/apedp-kernel/src/rime/record.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/record.rs) | Immutable content-addressed RIME candidate records (graph id, parent, rewrite op, seed, dev score). | `experimental` |
| [`benchmarks/rime_vs_neural.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/rime_vs_neural.py) | Head-to-head benchmark: RIME vs ridge and MLP baselines across planted mechanisms, sample sizes, and a null arm. | `audit-only` |
| [`benchmarks/matched_compute.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/matched_compute.py) | v0.1.3 matched-compute comparison harness: RIME (temporal + static grammars) vs existing proposers vs PySR 2.7 head-to-head (serial-deterministic, binding timeout, lag columns for information parity), at equal wall budget with 5 repeats/utilization/environment metadata, scored on a preregistered grammar-coverage axis (out-of-coverage abstention is correct; emitted-but-failed is failed; static worlds require reversal-invariance so grid-order tracking scores `index_fit`.). Measured outcome: complementarity, not dominance — RIME-temporal 4/5, PySR 3/5, each recovering a family the other misses. | `audit-only` |
| [`benchmarks/run_jetflow_vs_diffusion_matched.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_jetflow_vs_diffusion_matched.py) | JetFlow-Video vs diffusion textbook baseline at equal **wall-clock training budget** (90 s per architecture, probe-calibrated step counts; JetFlow 3509 steps vs baseline 5015 steps), plus a matched-compute distilled 2-step student row (probe-conservative second budget). Measures model-forward counts (not hand estimates), identical CFG/direction/holdout across architectures; instruments: nearest-PSNR/SSIM, FVD-proxy (`apedp_dyn/video/eval_gate.py`), temporal second-difference, per-clip latency medians. Artifact `benchmarks/jetflow_vs_diffusion_matched_results.json`; the honest verdict (per `docs/jetflow_vs_diffusion_matched_benchmark.md`): diffusion wins every quality instrument at matched budget, the distilled student is the only sub-50 ms/clip option (14.2 ms) and is the serving target — JetFlow's value is a latency envelope, not matched-compute quality. | `audit-only` |
| [`benchmarks/run_decoupled_transport_multiscene_benchmark.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_decoupled_transport_multiscene_benchmark.py) | Multi-scene Decoupled Continuous Spatio-Temporal Transport (DC-STT) benchmark: evaluates continuous PDE transport across vehicular translation, fluid vortex, expanding disocclusion with residual source texture injection, and camera zoom. Measures step latency ($2.25\text{ ms} - 6.91\text{ ms}$, overall 268.1 FPS on CPU), near-zero temporal acceleration flicker ($\|D_t^2 u\|_2 \sim 10^{-5}$ to $10^{-7}$), and 4D Tensor-Train compression ($10.9\times$ to $54.4\times$, up to 98.2% RAM saved) with near-lossless reconstruction ($\text{MSE} \le 8.66 \times 10^{-4}$). These are THROUGHPUT/RECONSTRUCTION claims on scenes with given (analytic) flow, not generation-quality claims. Artifact: `benchmarks/decoupled_transport_multiscene_results.json`. | `audit-only` |
| [`benchmarks/run_dcstt_horizon_decay.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_dcstt_horizon_decay.py) | DC-STT extrapolation-horizon decay audit (the instrument the multi-scene benchmark lacks): transport quality vs rollout horizon $H \in \{16,32,64,128,256\}$ against a 10x-fine-substep reference integrator of the SAME analytic flow (integrator fidelity 18–43 dB), plus appearance/energy and detail (blur-proxy) preservation curves — measured detail energy (blur proxy) falls to 11.3% of initial within $\sim$$H{=}128$ for the fluid-vortex scene, so unbounded-rollout serving quality is NOT supported by current evidence; short-horizon transport/upsampling is. Artifact: `benchmarks/dcstt_horizon_decay_results.json`. | `audit-only` |
| [`apedp_dyn/video/learned_flow_head.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/video/learned_flow_head.py) | Learned flow head for DC-STT (first learned component in the path): small CNN infers per-pixel $\mathbf{v}(\mathbf{x})$ from two context frames, trained self-supervised (advection-consistency + div² + TV) ENTIRELY ON CPU under the repo's wall-clock deadline discipline; torch-warp is sign-convention-parity-pinned to the Rust-bridge NumPy fallback ($\max|\Delta| < 10^{-12}$, `tests/test_dyn_learned_flow_head.py`, 6/6). | `experimental` |
| [`benchmarks/run_learned_flow_dcstt_benchmark.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_learned_flow_dcstt_benchmark.py) | Learned-flow DC-STT evaluation on the shared fluid-domain split (`n_clips=10`, holdouts `fluid_clip_0003/0005` — never train data): 90 s deadline CPU training (1367 steps, loss $4.4\mathrm{e}{-4} \to 6\mathrm{e}{-5}$), then DC-STT rollouts from 2 context frames scored per horizon against the clips' true futures vs persistence and integer-shift-search baselines. HONEST VERDICT (single-run, artifact-backed): the learned head does NOT yet beat persistence in aggregate (38.79/36.06/33.42 dB vs 39.54/37.21/35.24 dB at $H{=}4/8/14$; it wins only on one of two holdout clips) — recorded as-is, no tuning to green. Artifact: `benchmarks/learned_flow_dcstt_results.json`. | `audit-only` |
| [`rust/apedp-kernel/tests/rime_profile.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/tests/rime_profile.rs) | Kernel phase-decomposition profile (graph eval / readout fit / search bookkeeping remainder; median of 7): bookkeeping ≈90% of per-candidate cost at n=150, graph evaluation dominating only at n=12 000. Artifact: `benchmarks/kernel_profile_results.json`. | `audit-only` |
| [`benchmarks/openai_math_sandbox/run_sandbox.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/openai_math_sandbox/run_sandbox.py) | OpenAI-math release pipeline sandbox (`attempt -> certificate -> verification -> retraction accounting`) over 6 machine-checkable tasks plus 8 declared gap entries mirroring the OpenAI 2026-10-06 release (722 manuscripts, 3 retractions, 719 listed); no task is ever `verified` without a checker certificate (readme carries the measured gap analysis). Full-catalog ingestion (`catalog/`): all 372 families / 719 manuscripts from a SHA-256-pinned `openai/math` snapshot, Lean-catalogue join, withdrawal/numbering/formalization-count discrepancies recorded; 372/372 families `uncheckable_proof_level` (no proof engine). | `audit-only` |
| [`benchmarks/known_truth_challenge.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/known_truth_challenge.py) | v0.1.3 known-truth recovery challenge: blind RIME + BIC-OLS control on 11 planted monomial worlds (d ∈ {2,3,6}; grid/noise/coefficient/offset independent) + exp() negative control. Structural verdicts (effective exponent + coefficient ≤5%, reversal-invariant fit). Both-order fit required; abstention never counts as success. In-grammar abstention = miss; out-of-grammar emission recorded. v0.1.4 extensions: `--adjudicated` routes every emission through the referee-side scope refusal; `--repeats N` gives per-world R²/coef bounds over 5 noise seeds. Artifacts: `benchmarks/known_truth_results.json`, `benchmarks/known_truth_results_repeats.json`. | `audit-only` |
| [`specifications/canonical_preregistered_spec_v1.json`](file:///home/psalmprax/ALL_PROJECTS/science/specifications/canonical_preregistered_spec_v1.json) | Canonical frozen specification containing all preregistered thresholds. | `production` |
| [`apedp_dyn/conf/benchmark.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/conf/benchmark.py) | Legacy benchmark configuration and baseline threshold profiles. | `deprecated` |
| [`apedp_dyn/epistemic/transfer_scope.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/transfer_scope.py) | Out-of-distribution transfer scope classifier evaluating empirical domain boundaries. | `production` |
| [`apedp_dyn/epistemic/diagnosis.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/diagnosis.py) | Diagnostic failure attribution determining why candidates were refused at gates. | `production` |
| [`apedp_dyn/epistemic/number_theory_engine.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/number_theory_engine.py) | Discrete/algebraic number theory discovery engine: exact rational reconstruction (Farey/continued fractions), integer relation discovery (PSLQ/lattice null-space for conservation laws and stoichiometry), discrete modular lattice step identification. | `production` |
| [`apedp_dyn/epistemic/coupled_discovery.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/coupled_discovery.py) | Coupled multi-variable non-linear dynamical systems discovery: simultaneous multi-state variable ODE recovery ($\dot{\mathbf{x}} = \mathbf{F}(\mathbf{x})$) with interaction terms and system-level gating. | `production` |
| [`apedp_dyn/epistemic/bootstrap.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/bootstrap.py) | Block-bootstrap and stationary bootstrap statistical confidence interval estimator. | `production` |
| [`apedp_dyn/epistemic/term_necessity.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/term_necessity.py) | Term necessity and coefficient ablation testing framework. | `production` |
| [`apedp_dyn/epistemic/discovery_stability.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/discovery_stability.py) | Discovery stability harness measuring parameter sensitivity across random seeds. | `production` |
| [`apedp_dyn/epistemic/agnostic_adapter.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/agnostic_adapter.py) | Domain-agnostic evidence ingestion adapter standardizing holdout inputs. | `production` |
| [`apedp_dyn/epistemic/calibration.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/calibration.py) | Model probability and uncertainty calibration verification. | `production` |
| [`apedp_dyn/epistemic/pointer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/pointer.py) | Epistemic pointer tracking references to empirical artifacts and receipts. | `production` |
| [`apedp_dyn/epistemic/high_dim_pde.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/high_dim_pde.py) | Hybrid high-dimensional continuous PDE discovery, Jet2/Jet3 Taylor-mode CPU engine, and 5-way epistemic referee taxonomy. | `production` |
| [`apedp_dyn/epistemic/followup.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/followup.py) | Automated followup experiment recommendation and active disambiguation engine. | `experimental` |
| [`apedp_dyn/grammar/mechanical_grammar.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/mechanical_grammar.py) | Mechanical and physical dynamical systems grammar definitions. | `production` |
| [`apedp_dyn/grammar/proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/proposer.py) | Core grammar search proposer generating candidate AST expressions. | `production` |
| [`apedp_dyn/grammar/coupled.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/coupled.py) | Coupled multi-variable differential equation grammar definitions. | `production` |
| [`apedp_dyn/grammar/canonicalize.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/canonicalize.py) | Symbolic expression canonicalization and term ordering. | `production` |
| [`apedp_dyn/quantum/sparrow_photonics.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/quantum/sparrow_photonics.py) | Sparrow Quantum deterministic photonics engine: deterministic QD scaling vs SPDC, passive fusion with Excitation-Based Feedback (EBF), semi-fluid cluster code (sFFCC) resource scaling, and multi-channel fault-tolerance referee. Spec: [`docs/SPARROW_QUANTUM_DETERMINISTIC_PHOTONICS.md`](SPARROW_QUANTUM_DETERMINISTIC_PHOTONICS.md). | `experimental` |

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
| Native Lean 4 Streaming Verification | [`apedp_dyn/epistemic/lean_universal_bridge.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/lean_universal_bridge.py) | `tests/test_dyn_lean_universal_bridge.py` | Verified: Exit code 0, sub-50ms execution via `--stdin`. |
| Full-Lineage Experiment Tracking | [`apedp_dyn/epistemic/experiment_ledger.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/experiment_ledger.py) | `tests/test_dyn_experiment_ledger_and_benchmarks.py` | Verified: Candidate parentage, features, params recorded. |
| Anti-Snooping Holdout Burning | [`apedp_dyn/epistemic/experiment_ledger.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/experiment_ledger.py#L93) | `tests/test_dyn_experiment_ledger_and_benchmarks.py` | Verified: `HoldoutContaminationError` raised on reuse. |
| Deflated Sharpe Ratio Gating ($DSR \ge 0.95$) | [`apedp_dyn/epistemic/trial_ledger.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/trial_ledger.py) | `tests/test_dyn_competing_proposers_and_dsr.py` | Verified: Penalizes trial multiplicity $N$ and skew/kurtosis. |
| Six-Stage Scoped Promotion State Machine | [`apedp_dyn/epistemic/promotion_lifecycle.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/promotion_lifecycle.py) | `tests/test_dyn_experiment_ledger_and_benchmarks.py` | Verified: Stages 1-6 sequential, halts permanently on failure. |
| Decoupled DM Predictive Incrementality ($p < 0.01$) | [`apedp_dyn/epistemic/novelty_adjudication.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/novelty_adjudication.py) | `tests/test_dyn_competing_proposers_and_dsr.py` | Verified: Diebold-Mariano test with HAC standard errors. |
| Structural Literature Orthogonality ($R^2 < 0.85$) | [`apedp_dyn/epistemic/novelty_adjudication.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/novelty_adjudication.py#L150) | `tests/test_dyn_competing_proposers_and_dsr.py` | Verified: Rejects collinear Kyle/Amihud/Biais/Carhart proxies. |
| Stock-Split Scale Invariance ($P/k, kV$) | [`apedp_dyn/sandbox/canonical_ledger.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/canonical_ledger.py#L330) | `tests/test_dyn_competing_proposers_and_dsr.py` | Verified: Conserves dollar volume, detects nominal bias. |
| Tactical Active Fraction Window $[5\%, 90\%]$ (5% to 90%) | [`apedp_dyn/sandbox/canonical_ledger.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/canonical_ledger.py#L250) | `tests/test_dyn_competing_proposers_and_dsr.py` | Verified: Refuses Sharpe for decorative/non-binding gates. |
| RIME graph evaluation is time-consistent ($x[t]$ and $x[t-1]$ are distinct values) | [`rust/apedp-kernel/src/rime/graph.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/graph.rs) | `tests/rime_regression.rs` | Verified: difference/momentum forms retain their magnitude. |
| RIME memory cell is a true recursion | [`rust/apedp-kernel/src/rime/rewrite.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/rewrite.rs) | `tests/rime_regression.rs` | Verified: all-ones input at $d=0.5$ gives $[1, 1.5, 1.75, 1.875]$. |
| RIME scoring fits a readout and charges its degrees of freedom | [`rust/apedp-kernel/src/rime/eval.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/eval.rs) | `tests/rime_regression.rs`, `tests/rime_stage1.rs` | Verified: planted amplitude recovered; raw scoring shown insufficient. |
| RIME abstains on null worlds (with a paired positive control) | [`rust/apedp-kernel/src/rime/search.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/search.rs) | `tests/rime_regression.rs` | Verified: 0 promotions over 24 null worlds, and on 150-row null worlds. |
| RIME proposes through the standard proposer contract | [`apedp_dyn/proposers/rime_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_proposer.py` | Verified: `CandidateProposal` with past-information-only `compute_score` (temporal, not causal); abstains on noise. |
| RIME multi-environment scoring cannot be outvoted by one large frame | [`rust/apedp-kernel/src/rime/search.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/search.rs) | `pooled_dominance_trap_does_not_apply_to_envgate_rule` (Rust) + `test_rime_pooled_dev_scoring_cannot_outvote_a_failing_frame` (Python) | Verified: `MedianWorstGated` refuses a mechanism that wins only in the large env; the consistent-law control passes. |
| RIME Repair v0.1.0 diagnosis only (no equation modification) | [`apedp_dyn/epistemic/rime_repair_diagnosis.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/rime_repair_diagnosis.py) | `tests/test_dyn_rime_repair_v010_diagnosis.py` | Verified: frozen `ResidualReport` (sha256-stable) + finding labels on real proposals; planted bias/AR-noise/curvature/heteroscedasticity detected; null-world abstention and insufficient-rows controls; cutoffs are explicit caller args (`experimental`). |
| RIME Repair v0.1.1–v0.1.4 bounded engine (R1/R2/R3 + 4-role validation) | [`apedp_dyn/epistemic/rime_repair_engine.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/rime_repair_engine.py) | `tests/test_dyn_rime_repair_v011_engine.py` | Verified: planted intercept-shift→R1, missing-square→R2, emergent-lag→R3 each IMPROVED with sealed gain over frozen original; noise→no repair + sealed stays shut (`HoldoutContaminationError` on reopen); adequate original→no false repair; corrupted-fixture (shuffled targets)→NO_IMPROVEMENT, 0 evals, sealed shut; equivalence clustering preserves ambiguity; promotion deferred to APEDP-X track (`experimental`). |
| P0 risk-register dispositions (reviewer 25-finding triage) | tree-wide, measured | `tests/test_dyn_population_extrapolation_v2.py::test_depth_mechanism_budget_bound_not_ignored`, `::test_static_mode_emits_no_temporal_forms`, `tests/test_dyn_rime_known_truth*.py` (12 pass) | Verified: #2 depth NOT ignored — depth 1 evaluates 136 rows vs 200 cap at ≥2 with identical winners (budget-bound convergence); #3 static_mode=True emits 0 temporal forms on lag-planted truth; #9 `recovered_predictive_only` is a benchmark-local label with no promotion path (adjudicate owns verdicts); #10 scope-refusal regression suites green; #21 ridge fix verified real + unit-scale-neutral; REL_FLOOR=0.05 (`adjudicate.py:215`) confirmed NOT spec-pinned — flagged gap, no unilateral change. |
| Flaw-closure round: recall@budget, intervals, chaining, resources, rho sweep | `tests/test_dyn_population_extrapolation_v2.py::test_rime_recall_vs_budget_curve`, `::test_resource_curves_recorded`, `tests/test_dyn_rime_repair_v012_sensitivity.py::test_recall_strong_defects`, `tests/test_dyn_referee_rho_sensitivity.py` | All pinned | Verified: #5 product recall 0/5 budgets (penalty-, not budget-bound); #18 recall 18/18 with 95% lower bound 0.815, single-round termination pinned (`rounds_used==1`; multi-round chaining implemented but unreachable under single-targeted-finding veto semantics — documented extension point); #14 wall-time table recorded with hang-guard bounds; #7 whiteness operating point measured read-only (refuse@0.3/promote@0.5/0.7 on rho1_sqerr 0.413; frozen default untouched and verified equal to explicit 0.5). |
| #11 selective risk/coverage + #7 floor sweep | `tests/test_dyn_selective_risk_ga_rime.py`, `tests/test_dyn_referee_floor_sensitivity.py` | All pinned | Verified: post-hoc refusal policy (fit_R2>0.5 AND selection_R2>0, eval rows never touched) — GA/OLS attempt all signal worlds (eval R2>0.9), abstain 0/2 noise worlds; QEC thirds: OLS refuses flat low-p fit while RIME-repair upgrades state-form to square (sealed 0.987) — complementary honest behaviors. Floor sweep via explicit tau_epi: near-tie flips UNIQUE→MULTI across the bracket (monotone), decisive UNIQUE at 0.05+0.1, identical-predictions-distinct-structures MULTI at all floors; anchor artifact verified on disk. |
| Trading universe sweep (15/30/45/60/90 × daily/5d/21d) | `tests/test_dyn_trading_universe_sweep.py` | Pinned | Verified: all market files are daily bars (no minute data exists — 1/5/15/30/45/60-min frames untestable without inventing data); all 5 daily universes abstain on finite-checked frames; weak 5d/21d emissions (devR2≤0.12) collapse on holdout (R² −0.30/−0.77). Finite-row guard caught and fixed one vacuous loader path (bare-symbol crypto file). RIME on GA's exact 22-signal grammar: 0 proposals at depths 2–3/budgets 200–1000; penalty-off best devR2=0.0010 — absent signal, not missing grammar. Rename-invariance proves semantic blindness (identical graph ids/scores across meaningful vs nonsense names). |
| RIME feature-usage logging | [`apedp_dyn/proposers/rime_usage_log.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/rime_usage_log.py) | `tests/test_dyn_population_extrapolation_v2.py::test_usage_log_planted_composite_uses_only_x3`, `tests/test_dyn_trading_universe_sweep.py::test_usage_log_22_signals_mostly_untouched` | Verified: planted composite emissions reference x3 only (x1/x2 never enter); 22-signal penalty-off emissions touch 3/22 signals (flow/volatility family), 19 untouched. Syntactic presence (reachable nodes), not causal importance; rejected candidates invisible (kernel ships accepted only); zero proposals → zero log (`experimental`). |
| Search-funnel instrumentation + per-position coverage challenge | `rust/apedp-kernel/src/rime/search.rs`, `graph.rs` (`inputs_used()`), [`apedp_dyn/proposers/rime_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_coverage_funnel.py` | Verified: additive-only `rejected_score`/`dropped_dedup`/`dropped_scope` counters + per-proposal `inputs_used` (no score/gate change; invariants `rejected+accepted==evaluated`, `evaluated<=budget` asserted). Coverage on 22-signal frame (planted lag-1 dependence per position): **17/22 exact**, 2/22 collinear-substitute (H-OFI→H-CVD, H-STOCH→H-WILLR — in-grammar twins, MULTI rationale), 3/22 **generation gap**: solo-frame control recovers the same dependence at R²>0.95 while the 22-input frame fails (R² 0.09–0.68) — the lag-1 seed is cut by the 60-form catalogue budget at 22 inputs. Null worlds 0/2. This separates "never generated" (truncation) from "generated but not selected" (`experimental`). GENERATION GAP SUBSEQUENTLY CLOSED — see the seed-order row. |
| Seed-order fix for the measured generation gap (opt-in `seed_order="coverage_first"`) | [`rust/apedp-kernel/src/rime/seeds.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/seeds.rs), `SearchConfig.seed_order`, `RimeProposer(seed_order=...)` | `tests/test_dyn_rime_seed_order_coverage.py` + 3 Rust catalogue tests | Verified: lag-major family-2 keeps every input's shortest lag inside the same 60-form budget. Gap positions H-RV / H-A / H-OBV_MOM now recover the planted column at solo-frame R² (0.9937 / 0.9995 / 0.9597 vs legacy 0.19 / 0.68 / 0.09); all 22 positions name the planted column (only the 2 collinear substitutes remain, a non-identifiability property). Null-world FPR 0/50 unchanged; catalogues identical to legacy whenever the budget holds (Rust-pinned n≤6). Default stays `legacy` — no frozen-spec change (`experimental`). |
| Contemporaneous powers & self-products (opt-in `square_seeds=True`) | [`rust/apedp-kernel/src/rime/seeds.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/seeds.rs), `SearchConfig.square_seeds`, [`apedp_dyn/proposers/rime_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_reachability_and_contemporaneous_qec.py` | Verified: lag-0 self-multiplication (x_i * x_i) admitted into seed catalogue family 5 at depth 0. Closes reachability flaw on QEC: raw [p] discovers p_L = 1000*p^2 directly at depth 0 without requiring manual feature engineering (p_sq) or repair engine intervention. Holdout R² > 0.9995; verified 100% reversal-invariant (|R2_fwd - R2_rev| < 1e-12), whereas legacy lagged proxy (p * p[t-5]) inflates MSE by >10x under row reversal. Systematic residual curvature screen flags d=5 wrong-family (p^2 fit on p^3 truth) while clearing d=3 true law. Default stays `false` (`experimental`). |
| Intraday crypto universe sweep (fetched minute bars) | `tools/fetch_intraday_crypto.py`, `data/market/intraday/` | `tests/test_dyn_intraday_universe_sweep.py` | Verified on genuine Binance klines (1000 bars/symbol/interval; frozen daily corpus untouched): real basket forward returns abstain in **15/15** universe(14/26/39)×interval(1m/5m/15m/30m/1h) cells under both seed orders; null worlds 0 proposals; planted lag-1 control recovered at dev R²>0.98 at every interval; per-position coverage follows the 60-form arithmetic exactly (39 levels + 21 lag-1 → inputs 0–20 recovered under `coverage_first`, 21+ still gaps). Scope recorded: 45m is not an exchange interval; equity legs unfetchable (Yahoo 429) so 60/90-asset intraday universes out of scope; equal bar counts confound interval with calendar span (`experimental`). |
| REAL-data repair benchmark (AME2020 masses, no planting) | `tests/test_dyn_real_data_ame_repair.py` | Pinned, 6 tests | Verified on genuinely measured data: discovery recovers the SEMF **volume term** (`+7426.14*A+93729.3`, dev R² 0.994); diagnosis fires `curvature_suspect` (missing surface term A^2/3); bounded repair at the PREREGISTERED absolute bar 0.02 → `NO_IMPROVEMENT` (bar mis-scaled for a 0.994 baseline); at a scale-appropriate bar → `IMPROVED`, sealed R² **0.9995** vs 0.9947. Grammar-extension experiment (declared `rational_powers`, default off): with the family admitted the validation ranking is proxy (A·Z) 0.99960 > (A·A) 0.99930 > **surface term (A^0.667) 0.99915** > (A^0.333) 0.99898; on the sealed split proxy and mechanism differ by <1e-3 R², so with the identifiability band declared the engine returns `AMBIGUOUS` (set size 4) — predictive success reported separately from identification. Preregistered criteria: `min_sse_reduction` (relative SSE reduction) and `equivalence_r2_tol`, both default-off/frozen. SEMF literature coefficients reach R² 0.9999 on the same frame; nothing is promoted (`experimental`). |
| RIME Repair sensitivity/recall milestone (v0.1.2–v0.1.3) | [`apedp_dyn/epistemic/rime_repair_engine.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/rime_repair_engine.py) | `tests/test_dyn_rime_repair_v012_sensitivity.py`, `tests/test_dyn_rime_repair_v013_realdata.py` | Verified: recall 9/9 strong defects with structure recovery (right operator/variable); weak-defect curve never harms sealed outcome; BTC panel abstention regression (repair-R²=−0.0462, sealed shut); 5/5 walk-forward windows abstain with persistence RMSE reported; chance predictive pass blocked at stability gate (`experimental`). |
| Model-class closure: interaction probes + lagged-product R2 terms | [`apedp_dyn/epistemic/rime_repair_diagnosis.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/rime_repair_diagnosis.py), [`apedp_dyn/epistemic/rime_repair_engine.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/rime_repair_engine.py) | `tests/test_dyn_rime_repair_v011_engine.py::test_repair_composite_product_completed` | Verified: omitted interactions of independent drivers (white residuals) fire `interaction_suspect` via probe family mirrored 1:1 in the R2 catalogue (lag choices cite frozen LAG_SET's two shortest; no new configured values); planted composite completes to sealed R²>0.9 with the lagged product term. QEC-conservative behavior preserved (3 refused R2 rows incl. R² 0.97). No frozen-spec change: probe/term families live in the experimental engine, cutoffs stay explicit caller args (`experimental`). |
| Staged search + coverage/budget curves (grammar-size auditing) | [`apedp_dyn/epistemic/rime_staged_search.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/rime_staged_search.py) | `tests/test_dyn_rime_staged_search.py` | Verified: two-stage search (cheap family-admit pass, then full-budget expansion) that REPORTS coverage instead of asserting it. Measured budget-curve: recovery saturates at 176 evaluated with identical best R² 0.9616 across budgets 96→2000, and reached families DROP 3→2 — depth/budget claims beyond the cap are budget-bound and are pinned as such. Expanded grammar (all ops ON) spends more budget (200 vs 176) for the same winning family and same score: expansion buys reachability, not recall. Staged run is never worse than the straight full-budget run. Structured-null FPR unchanged with all ops ON (proposer 4/12 worlds, system 0) (`experimental`). |
| Noisy multi-dataset repair suite (real measured targets, 2% noise) | `tests/test_dyn_noisy_multidataset_repair.py` | Pinned | Verified on three real AME2020 target families (volume/surface, isobaric parabola, symmetry residual), each with 2% observation noise: D1 IMPROVED with sealed evidence; D2 INSUFFICIENT_EVIDENCE (92 rows, splits below the diagnosis floor); D3 NO_IMPROVEMENT. True no-repair baseline (zero repair operators) never reports a fix. Row-floor limitation recorded: Bertozzi (5 runs), Brown lifter (20 rows), warp-energy records cannot host the 4-role battery without data acquisition (`experimental`). |
| Structural invariance ranking (identification by cross-slice agreement) | [`apedp_dyn/epistemic/structural_invariance.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/structural_invariance.py) | `tests/test_dyn_structural_invariance.py` | Verified: uniqueness is earned by the SAME canonical structure winning >= 4/5 independent dev slices rather than by a thin margin in one pooled fit — the measured failure mode where a correlated proxy (A*Z) outranked the correct term (A^0.667) by ~3e-4. Key is the decomposed variable multiset with the fitted readout excluded and additive-zero decorations stripped (measured: without that, PruneEdge syntactic twins fragmented agreement to 2/5). Negative control: A-only wins 5/5 with the VOLUME term only, surface term absent — agreement does not repair a grammar that cannot express the truth. Rotating winner -> MULTI; empty search -> NONE (`experimental`). | [`apedp_dyn/epistemic/whiteness_calibration.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/whiteness_calibration.py) | `tests/test_dyn_whiteness_flag_calibration.py` | Verified: derives Guard 2's \|rho1\| flag from the target's own serial dependence by block-demeaning instead of a fixed 0.5. Measured both fix directions: on an AR(2) target a residual with \|rho1\|=0.723 is refused by the frozen 0.5 and accepted by the calibrated 0.858 (over-refusal fixed); on a white target a residual with \|rho1\|=0.273 is admitted by the frozen 0.5 and refused by the calibrated 0.226 (under-refusal fixed). Flag is monotone in target persistence (0.114 white → 0.858 AR(2)). Ships `screen_structured_null_worlds`, which reports proposer-level and system-level FPR separately; on 24 structured null worlds: 12 proposer promotions, 0 system promotions. Lives in its own module because `adjudicate.py` is a hash-pinned frozen record (`experimental`). |
| KNOWN LIMITATION — same-driver omitted lags vs rho1 screens | `tests/test_dyn_rime_repair_v012_sensitivity.py::test_same_driver_omitted_lag_sensitivity_map` | Sensitivity map (6 corners) | **Documented boundary**: single-regressor OLS residuals on an AR driver are usually absorbed white (4/6 corners silent). Firing corners (state-cell misfit remainders) yield gated rejection, not false wins. R3's demonstrated domain is emergent lag from an independent driver. Whiteness alone is insufficient for structural identification. |
| KNOWN LIMITATION — whiteness gate under geometric-spacing extrapolation (QEC) | `tests/test_dyn_rime_repair_v014_qec.py::test_qec_degraded_linear_repair_refused_despite_high_r2` | Pinned conservative behavior | **Documented boundary**: on log-spaced p, quadratic extrapolation misfit grows smoothly into validation rows and trips the temporal screen, vetoing an R²=0.95 repair. Refusal is the safe failure direction; gate assumes stationarity across evaluation rows (valid for time series). No gate respecification without its own preregistered benchmark. |
| QEC thirds decisive comparison (GA vs RIME-orig vs RIME-repaired vs OLS) | `tests/test_dyn_rime_repair_v014_qec.py::test_qec_thirds_same_split_ga_vs_rime_repaired` | Pinned, same eval rows, no refits | Verified: RIME-repaired sealed R² 0.9869 wins; OLS-quad 0.8733 (its policy refusal was over-conservative — recorded limitation of the post-hoc rule); RIME-orig −1.5959; GA starved to intercept-only (−2.15) on the flat fit third. Better repair outcome on this test: yes. Better-than-GA generally / transferable law: NOT established. |
| Population extrapolation RIME vs GA (depths, feature space, repair-mediated) | `tests/test_dyn_population_extrapolation_rime_ga.py`, `tests/test_dyn_population_extrapolation_v2.py` | Pinned grids | Verified: RIME depths 1–6 identical (x3[t-2] partial); +sq transforms change nothing; withheld-feature abstains all depths; GA ladder isolates products as the sole gap (squares add nothing); GA-small ties RIME (<0.01 all pops); RIME-large (23 engineered cols) ties RIME-raw (truncation + no-Add reason measured); repair-mediated RIME matches GA-large incl. transfer (sealed/B 0.9998, C 0.8952) (`experimental`). |
| GA on QEC vs OLS vs RIME (method boundary) | `tests/test_dyn_ga_qec.py` | Pinned three-way + incident note | Verified: GA selects {1, p²}, holdout RMSE 0.002220 = same-split OLS (C/p_th²≈1000 recovered); RIME raw-[p] deep search recovers via seeded self-interactions (holdout R² 0.9995, stable); shallow-config single divergent run (lag winner, 1 of ~14) explained by a concurrent kernel rebuild landing mid-measurement (absolute- vs relative-ridge binaries), not search nondeterminism. A concurrent session's scale-invariant ridge fix (`eval.rs`: relative `ridge·Σo²`, commit 9781a3d) was verified real and behavior-neutral on unit-scale tests; an unattributed edit to this row/test asserting it was first reverted, then re-verified and corrected here. Coordination via separate worktrees required going forward. |
| Frozen-equation OOS (QEC Eq1 vs Eq2, no refit) | `tests/test_dyn_rime_qec_oos_frozen.py`, `tests/test_dyn_rime_qec_repaired_transfer.py` | Pinned comparison | Verified: T1 fresh seeds R²>0.999 both; Eq1≡Eq2 to ~2e-5 on log grids (one hypothesis, not two); uniform grid R²>0.99 with doubled MAE; d=5 graceful failure matching frozen OLS-quad; mean baseline negative. Repaired equation (frozen): R² 0.990–0.995 on fresh/uniform/irregular/extended-range/3x-noise d=3 grids, always above frozen OLS-quad; monotone decay d=4→7 (0.953/0.817/0.653/0.497) with OLS-quad marginally above throughout — no transferable law, ordered degradation; square coef 1071 vs truth 1000 (useful structure, not the mechanism). |
| Real-Market Trading Predictive Profitability | N/A (Candidate evaluations in `sandbox/cpmd_*.py`) | None (Empirical holdouts failed) | **Explicitly Unverified / Unproven**: H-LIK and H-XSC failed OOS confirmation. |
| Rational operations & Padé saturation (Rust SafeDiv, Epistemic Repair R5, and Normalized Transforms) | [`rust/apedp-kernel/src/rime/graph.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/graph.rs), [`apedp_dyn/epistemic/rime_repair_engine.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/rime_repair_engine.py), [`apedp_dyn/proposers/rime_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_rational_and_pade.py` | Verified: Option 1: Native Rust `safediv(x, y) = x / (1 + |y|)` enables rational division in symbolic mechanism graph ($R^2 = 1.0000$ on rational target $y = x/(1+x)$, opt-in `rational_ops=True`); Option 2: Epistemic repair engine operator `R5_PADE_SATURATION` fits non-linear Padé saturation $g(z) = a \cdot z / (1 + b \cdot |z|)$ on repair partition, cutting MSE by >93% and reaching $R^2 = 1.0000$ on sealed holdout; Option 3: Normalized algebraic transforms (`sat_norm`, `sat_sq_norm`) scale inputs by domain standard deviation to recover saturating threshold laws ($R^2 > 0.97$ vs un-saturated $p^2$ $R^2 = 0.78$). Zero regressions; defaults off (`experimental`). |
| QEC d=5 cubic discovery & saturating generators | [`apedp_dyn/sandbox/qec_threshold.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/qec_threshold.py), [`tests/test_dyn_rime_qec_d5_and_saturation_pins.py`](file:///home/psalmprax/ALL_PROJECTS/science/tests/test_dyn_rime_qec_d5_and_saturation_pins.py) | `tests/test_dyn_rime_qec_d5_and_saturation_pins.py` | Verified: Domain-spanning split discovers true cubic `(p * (p * p))` at depth 1 via `AddInteraction` with holdout $R^2 > 0.999$ and scale $\approx 99,863.1$ (true $C/p_{\text{th}}^3 = 100,000$), while recovering quadratic proxy `(p * p)` at dev $R^2 \approx 0.9626$; chronological 300/300 split on low-SNR subrange ($\text{SNR} \approx 0.31$) yields linear fit holdout $R^2 \approx -0.435$, where complexity gating correctly abstains; systematic curvature screen flags wrong-family `(p * p)` fit on $d=5$ observations; canonical Padé and Exponential saturating data generators pinned on disk (`experimental`). |
| STEM Grammar Expansion (Native Primitives, Multi-Engine Proposers & Dimensional Homogeneity) | [`rust/apedp-kernel/src/rime/graph.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/graph.rs), [`apedp_dyn/proposers/stem_proposers.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/stem_proposers.py), [`apedp_dyn/epistemic/dimensional_homogeneity.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/dimensional_homogeneity.py) | `tests/test_dyn_stem_grammar_options_123.py` | Verified: Option 1: Native Rust graph nodes (`Sin`, `Cos`, `SafeExp`, `SafeLog`, `Diff`) and rewrite ops (`stem_ops=True`); Option 2: Multi-engine proposers (`HamiltonianProposer`, `SolenoidalFluidProposer`, `CompartmentalSirProposer`) enforcing symplecticity, divergence-free constraints, and stoichiometric conservation; Option 3: Dimensional homogeneity referee gate validating SI unit consistency via Buckingham-$\Pi$ algebra before hypothesis promotion (`experimental`). |
| Preregistered Real-Data Benchmark Suite | `data/mass_1.mas20.txt`, `data/bertozzi_1964.jsonl`, `data/market/` | `tests/test_dyn_preregistered_real_data_benchmark.py` | Verified on genuine real-world measurements: (1) AME2020 nuclear binding energies (1,159 nuclides): RIME discovers Coulomb term $Z^2$ and $A \cdot Z$ with holdout $R^2 > 0.70$ and RMSE $< 0.20$ MeV/nucleon on 559 unseen nuclides, compared against literature Weizsäcker SEMF ($R^2 = 0.9921$); (2) Bertozzi 1964 relativistic electron accelerator: classical mechanics rejected catastrophically (holdout RMSE $\approx 4.88c$), relativistic mechanics promoted with `UNIQUE_SUPPORT` (holdout RMSE $\approx 0.0055c$); (3) Real market orderflow and asset returns: zero false promotions ($\text{FPR} = 0.00\%$), 100% abstention on pure noise (`experimental`). |
| RIME v0.2 Structural Flaw Remediation | [`rust/apedp-kernel/src/rime/graph.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/graph.rs), [`rust/apedp-kernel/src/rime/search.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/search.rs), [`apedp_dyn/proposers/rime_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_v02_flaw_remediation.py` | Verified: (1) Native Rational Powers (`power_ops=True`): admits $x^{2/3}$, $x^{1/3}$, $x^{1/2}$, $x^{-1/3}$ into search, recovering nuclear surface scaling $A^{2/3}$ with exact $R^2 = 1.000000$ and zero error; (2) Small-Sample Laboratory Gating (`min_rows=5`): resolves small-sample paralysis on laboratory experimental data ($N=5$) while preserving frozen $N \ge 30$ default for noisy time-series; (3) Correlated Proxy Trap Remediation (`identifiability_band=0.002` & `check_derivatives=True`): prevents false-certainty promotion on collinear hypotheses by reporting `AMBIGUOUS_EQUIVALENCE_SET` and verifying gradient alignment; (4) Diverse Beam Frontier (`diverse_beam=True`): guarantees round-robin representation across structural archetypes to eliminate greedy beam monoculture (`experimental`). |
| Rational Parameter Identifiability & Decoupled 5-Metric Reporting | [`apedp_dyn/epistemic/rime_repair_engine.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/rime_repair_engine.py) | `tests/test_dyn_rime_rational_parameter_identifiability.py` | Verified: (1) Parameter reconciliation: explicit mapping of physical scales ($c_{\text{num}} = a \cdot k_{\text{fit}} = 1000.0$, $c_{\text{den}} = b \cdot k_{\text{fit}} = 2000.0$) and saturation envelope ratio ($\alpha = b/a = 2.0000$); (2) Noise robustness: verified across $\sigma \in [0.5\%, 2\%, 5\%]$, with parameter errors bounded within 0.43% to 4.29%; (3) Domain range sensitivity: proves parameter identifiability breakdown on narrow intervals ($p \in [0.001, 0.008]$) where $R^2 > 0.999$ coexists with 6.5% parameter error, correctly triggering `PREDICTIVE_ONLY_ILL_CONDITIONED`; (4) Negative controls: verified against pure quadratic ($b \to 0$) and exponential saturation; (5) Decoupled 5-metric reporting: independently emits `structure_match`, `parameter_recovery`, `prediction_score`, `extrapolation_score`, and `identifiability_status` (`experimental`). |
| RIME Native Resolution of Seed Catalog Exhaustion & Grammar Constraint | [`rust/apedp-kernel/src/rime/seeds.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/seeds.rs), [`rust/apedp-kernel/src/rime/search.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/search.rs), [`apedp_dyn/proposers/rime_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_native_additive_and_seeds.py` | Verified: (1) Seed Catalog Exhaustion resolved via `max_seeds` and `lagged_products=True` admitting contemporaneous lagged interactions $x_i[t-k] \cdot x_j[t-k]$ into Family 6b; (2) Grammar Constraint resolved via native `additive_terms >= 2` wired to `search_additive` with joint refit and honest multiplicity/degree-of-freedom penalization; (3) Composite population extrapolation ($y = 2 x_1[t-1] x_2[t-1] + 3 x_3[t-2] + 10$) natively recovered with true parameters; (4) Matches GA-Large across Pop A ($R^2 \approx 0.9998$), Pop B ($R^2 \approx 0.9998$), and Pop C shift ($R^2 \approx 0.896$) without referee repair; (5) Distractor resilience filtering 7 noise features ($R^2 > 0.9999$) and parsimonious single-term retention on null/single-term worlds (`experimental`). |
| Closed-loop prior-guided search (profiler -> search adapter, unsolicited recovery) | [`apedp_dyn/epistemic/prior_guided_search.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/prior_guided_search.py) | `tests/test_dyn_prior_guided_unsolicited.py` | Verified (7/7, fixed seeds): consumes the shape prior + dataset profile on dev rows only and selects a search path through a fixed table (stepped -> dev-exact sign basis + extrapolatable rule form; singular/power -> transmutation fitter; else abstain with reason). Emits `CandidateProposal` objects; never promotes, never sees holdout. Unsolicited Landauer recovery on sealed splits (dev [0.1,4.9] -> holdout [5.0,9.8]): holdout R2 == 1.0 with zero human primitive choice; the module contains no quoted operator tokens usable for per-equation selection (machine-guarded test). Owns a deterministic two-stage floor-rule fit with max-margin tie-break, which was REQUIRED by measurement: several (w, off) classify dev rows identically (SSE 1.5e-29 tie) while holdout separates them (0.974 vs 1.0). Noise rung 1 (2% staircase): sealed holdout >= 0.999. Rung 2 labeled semi-synthetic (BTC 5m returns + injected W=125/OFF=40 staircase at local SNR=5): width 125.02 vs 125, quantum within 0.4%, offset modulo width, holdout 0.8910 vs oracle 0.8917 (noise ceiling); same code abstains on the clean stream; low-SNR boundary (SNR<=4 abstains) pinned. Recorded integration gap: the transmutation fitter declines stepped data under noise that the shape prior identifies (level-clustering cap), so row 1 uses the adapter's uniform floor-fit path. Rung 2 is semi-synthetic by necessity (no real known-truth dataset >= 40 rows in repo); unsolicited discovery on real unknowns remains unclaimed (`experimental`). |
| Report fidelity (verdicts derive from measured rows) | [`apedp_dyn/epistemic/report_fidelity.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/report_fidelity.py) | `tests/test_dyn_landauer_report_fidelity.py` | Verified (4/4): `verdict_for_row` with explicit caller args (tau_extrapolation, rho_max, eps_exact_rmse, delta_exact_r2); exact-recovery bypass checked first so rho1 of numerical noise cannot refuse R2 == 1. The Landauer harness template renders this function's output verbatim (4-line + import edit; NaN rho1 renders `n/a (exact-zero residuals)`, not `0.0000`); hardcoded PASSED/REFUSED labels removed. Committed md == committed JSON cell-for-cell on the main table (machine-asserted); a no-hardcode guard scans the harness source. Robustness-table numerics verified by harness re-execution (all digits reproduce), not by the fidelity test, which covers the main table (`experimental`). |
| Data-driven shape inference (data-driven family prior + step basis, steps 1-2 of unassisted stepped discovery) | [`apedp_dyn/epistemic/shape_inference.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/shape_inference.py) | `tests/test_dyn_shape_inference.py` | Verified (15/15, fixed seeds): infers which structural family the data supports, with change-points estimated from the data rather than declared. Method: exact O(k·n²) DP piecewise-constant and piecewise-linear segmentations; the family verdict is how many PARAMETERS each family needs to reach a common fit target (an achievable×(1−sse_target) R², both families costed per parameter), a shared-square-root-of-zero-free statistic. `fit_step_family` emits a candidate `c₀ + Σ cᵢ·sign(x − tᵢ)` form with data-estimated thresholds and OLS coefficients — needing NO new operator node (`Sign`/`Add`/`Const`/`Mul` already exist). Measured: Landauer staircase recovered EXACTLY (R² 1.0, coefficients 2.5 and 0.5×5 matching the true generating form, 5 change-points within half the local bin gap of truth 1..5), still identified at 2% noise (R² 0.9997) and through 10% noise. Negative controls, all correct: straight line → `smooth` with 0 change-points and NO seeds emitted; saturating Padé → `smooth`, 0; iid noise → `smooth`; BTC forward returns (1094 rows) → `no_structure` (pc R² 0.017 below the caller floor) — no fabricated steps on real market data. Step count is data-estimated (2-step → 2, 5-step → 5). <40 rows → `no_structure`. FOUR statistics were implemented and REJECTED by measurement and are recorded in CHANGELOG.md: "piecewise-constant beats a line" (flags noise and a saturating curve as stepped, only +0.027 on the real staircase); flat-run fraction on the p90 |Δy| scale (collapses 0.975→0.065 under noise); within-segment drift ratio (mislabeling a coarse segmentation of a CURVE as plateaus); and separately-BIC-selected k per family (degenerating because enough constant segments drive SSE to exactly 0). Critical gate-dependence control: `fit_step_family` on a STRAIGHT LINE with forced thresholds returns R² 0.979 and `identifiable=True`, so a high R² from this basis is NOT admissible evidence of steps — only the independent prior's verdict is, which is why `staircase_prior` refuses to emit seeds on a line. LIMITATIONS (stated, not hidden): the step basis is not yet fed to the kernel search, so multi-term composition is unbuilt and no unsolicited end-to-end discovery is claimed; at 20% noise the staircase no longer classifies within `max_segments=32`, a cap limit, recorded rather than tuned away (`experimental`). |
| Landauer Conductance Staircase: Reproducible RIME vs. GA Benchmark | [`benchmarks/run_landauer_rime_vs_ga_benchmark.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_landauer_rime_vs_ga_benchmark.py), [`docs/landauer_conductance_staircase_rime_vs_ga_benchmark.md`](file:///home/psalmprax/ALL_PROJECTS/science/docs/landauer_conductance_staircase_rime_vs_ga_benchmark.md) | `benchmarks/run_landauer_rime_vs_ga_benchmark.py` | Verified from reproducible execution on disk: (1) GA polynomial (degree 5) achieves in-sample $R^2 = 0.9645$ but catastrophically diverges out-of-sample ($R^2 = -11530.5976$, RMSE = 149.32, $\kappa = 4.54 \times 10^4$), refused by Epistemic Referee whiteness ($|\rho_1| = 0.7238 > 0.50$); (2) Unassisted RIME finds smooth linear surrogate ($R^2_{\text{train}} = 0.9598$, $R^2_{\text{extrap}} = 0.9486$), refused by referee whiteness ($|\rho_1| = 0.7386 > 0.50$); (3) Assisted RIME with `floor` recovers exact quantum conductance $G_0 = 1.000000$ ($R^2_{\text{extrap}} = 1.000000$); (4) Autonomous Epistemic Profiler & Transmuter detects discrete plateau topology ($84.8\%$ flat derivatives) and recovers $G_0 = 1.000000$ in $0.35$ ms; (5) Machine-measured across clean, thermal broadening ($k_B T / \Delta E = 0.08$), 2% noise ($R^2_{\text{extrap}} = 0.999991$), and 5% noise ($R^2_{\text{extrap}} = 1.000000$). Results logged to `benchmarks/landauer_benchmark_results.json` (`experimental`). |
| Quantum Equation Discoveries & RIME Physics Remediations | [`rust/apedp-kernel/src/rime/seeds.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/seeds.rs), [`apedp_dyn/epistemic/rime_repair_engine.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/rime_repair_engine.py), [`apedp_dyn/proposers/rime_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/rime_proposer.py) | `tests/test_dyn_rime_quantum_equation_fixes.py` | Verified: (1) Limitation 1 (Negative Powers): Brus quantum dot confinement ($E_g(R) = 1.74 + 2.45 R^{-2} - 0.28 R^{-1}$) natively discovered via power_ops ($x^{-1}, x^{-2}$) and transforms (`inv`, `inv_sq`), raising extrapolation $R^2$ from $-70.27$ to $+0.9959$; (2) Limitation 2 (Continuous Denominator Scale & Lorentzian Resonance): R5_PADE_SATURATION expanded to Form 2 resonance $a / (1 + b |z|)$, recovering true $\Omega_R^2 = 25$ on Detuned Rabi with $R^2 = 1.000000$ (clean) and $R^2 = 0.999968$ (2% noise, 0.76% parameter error, `IDENTIFIED_STABLE`); (3) Limitation 3 (Harmonic Frequency Multipliers): `harmonic_seeds=True` emits $\cos(t+t)=\cos(2t)$ at Depth 0, discovering resonant Rabi dynamics ($P_e(t) = \frac{1-\cos(2t)}{2}$) with holdout $R^2 = 1.000000$ (`experimental`). |
| Autonomous Epistemic Profiler & Coordinate Transmutation | [`apedp_dyn/epistemic/autonomous_profiler.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/autonomous_profiler.py), [`apedp_dyn/epistemic/coordinate_transmuter.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/coordinate_transmuter.py) | `tests/test_dyn_autonomous_epistemic_profiler.py` | Verified: (1) Autonomous Epistemic Profiler diagnoses topological invariants: dynamic range (>100 -> log_mse), Arrhenius tunneling (d ln y / d(1/x) = const), power-law scaling (d ln y / d ln x = const), quantized plateaus (flat fraction > 40%), and dual-asymptotic boundary crossovers (Rayleigh-Jeans vs Wien); (2) Coordinate Transmuter linearizes non-linear manifolds: exact non-perturbative recovery of Schwinger tunneling ($10 E^2 e^{-1/E}$, $R^2 = 1.000000$), Brus confinement ($4.5 R^{-2}$, $R^2 = 1.000000$), Landauer staircase ($2 \lfloor w \rfloor$, $R^2 = 1.000000$), Landau-Zener non-adiabatic transition ($P = \exp(-0.75/v)$, $R^2 = 1.000000$), BCS superconducting energy gap ($\Delta = 3.52 \exp(-1/g)$, $R^2 = 1.000000$), and Integer Quantum Hall Effect staircase ($\sigma_{xy} = \lfloor 1/B \rfloor$, $R^2 = 1.000000$) (`experimental`). |
| RIME-AD Native Rust Engines & Unified Quantum Benchmark | [`rust/apedp-kernel/src/rime/total_variation.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/total_variation.rs), [`rust/apedp-kernel/src/rime/implicit.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/implicit.rs), [`rust/apedp-kernel/src/rime/diff_pade.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/diff_pade.rs), [`rust/apedp-kernel/src/rime/enumerate.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/enumerate.rs) | `tests/test_dyn_rust_all_equations_benchmark.py` | Verified (6/6 tests, 9 physical equations, zero hardcoded verdicts): (1) BV-ROF Total Variation denoising (`total_variation.rs`) recovers clean Landauer ($R^2_{\text{extrap}} = 1.000000$) and IQHE ($R^2_{\text{extrap}} = 0.990926$); (2) Sparse Implicit SVD (`implicit.rs`) solves Detuned Rabi Padé resonance ($1 - y - 0.04 x^2 y = 0$) with holdout $R^2 = 1.000000$ and RMSE $< 10^{-11}$; (3) Differential Padé & Jet Algebra (`diff_pade.rs`) discovers Schwinger, Landau-Zener, and BCS ODEs; Landau-Zener passes exact recovery ($R^2 = 1.000000, \text{RMSE} = 1.22 \times 10^{-9}$), while Schwinger and BCS are honestly refused by the referee's whiteness gate due to smooth numerical RK4 integration drift ($|\rho_1| \approx 0.925 \ge 0.5$); (4) Planck Radiance is honestly recorded as a structural failure ($R^2_{\text{dev}} = -1.27$, holdout collapse, REFUSED) because linear Jet ODEs cannot represent non-linear Riccati crossovers; (5) Brus Confinement recovers exact multi-rational powers ($1.74 - 0.28 R^{-1} + 2.45 R^{-2}$) with holdout $R^2 = 1.000000$. Full machine results logged in `benchmarks/quantum_all_equations_benchmark_results.json` and `docs/quantum_all_equations_benchmark_results.md` (`experimental`). |
| Multi-Population Extrapolation & Distractor Challenge: RIME-AD vs GA/GP | [`benchmarks/run_population_extrapolation_rime_ad_vs_ga.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_population_extrapolation_rime_ad_vs_ga.py), [`docs/population_extrapolation_rime_ad_vs_ga.md`](file:///home/psalmprax/ALL_PROJECTS/science/docs/population_extrapolation_rime_ad_vs_ga.md) | `tests/test_dyn_population_extrapolation_benchmark.py` | Verified (3/3 tests, across Pop A-train, Pop A-test, Pop B, Pop C, Pop D, and 7-noise distractor challenge): (1) In-sample & holdout parity: RIME-AD matches GA-Large ($R^2_{\text{test}} = 0.9998$ vs $0.9998$) while GA-Small collapses ($R^2_{\text{test}} = 0.6450$) due to missing nonlinear interactions; (2) Parameter shift transfer: $c_1 = 2.0 \to 2.5, c_2 = 3.0 \to 2.0$ transfers gracefully ($R^2 = 0.8961$); (3) Heavy-tail Laplace OOD: RIME-AD achieves $R^2 = 1.0000$ (0.999980); (4) Exact formula recovery: $+1 \cdot (2.9956 \cdot x_3[t-2] + 1.9966 \cdot (x_1[t-1] x_2[t-1])) + 9.9988$; (5) Distractor noise immunity: zero false positives out of 7 distractor channels (0% false discovery); (6) Execution wall-time: RIME-AD runs in 8.7 ms vs 2137.7 ms for GA-Large (~246x speedup) (`experimental`). |
| Koopman Spectral Discovery & Lie Point Symmetries (Rust specialists + GA comparison) | [`rust/apedp-kernel/src/rime/koopman.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/koopman.rs), [`rust/apedp-kernel/src/rime/lie_symmetry.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/lie_symmetry.rs) | `tests/test_dyn_rust_koopman_vs_ga.py`, `tests/test_dyn_rust_lie_symmetry.py` | Verified (6/6): (1) Delay-embedded linear spectral discovery (`koopman.rs`, affine AR(2) + companion eigenvalues) recovers harmonic frequency $\omega = 2.0000$ with spectral radius $1.00$ and holdout rollout $R^2 = 1.000000$ in $0.31$ ms; damped oscillator recovers $\omega = 3$, $\sigma = -0.5$; (2) matched-split native-GA comparison on identical train/holdout: GA discovers the exact trig identity ($-2\sin^2(t)+1$, holdout $R^2 = 1.000000$) in $14.71$ ms over 1560 evaluations vs Koopman $0.31$ ms; (3) Lie point-symmetry discovery (`lie_symmetry.rs`, exact polynomial determining-equation algebra + Gram/Jacobi nullspace) verifies the scaling/x-translation symmetries of $y' = y$ (nullity $\ge 2$), the exact 9-dimensional translation algebra of $y' = 0$, and $y$-translation for $y' = x$ (`experimental`). |
| Five-gate validation programme & Planck honesty fix | [`benchmarks/run_five_gate_validation.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_five_gate_validation.py), [`tests/test_dyn_planck_limitation.py`](file:///home/psalmprax/ALL_PROJECTS/science/tests/test_dyn_planck_limitation.py) | `tests/test_dyn_five_gate_validation.py`, `tests/test_dyn_planck_limitation.py` | Verified (all 5 gates pass): Gate 1 exact recovery on 8/9 equations with the Planck limitation held; Gate 2 honest noise degradation (Landauer quantized through 5% noise; Schwinger at 2% noise refuses exact claim with trajectory $R^2 < 0$); Gate 3 four adversarial abstentions (sine, white noise, null world); Gate 4 matched-compute ablation — Rabi tie at $R^2 \approx 1.0$ (RIME 26 ms vs GA 8 ms), Brus GA overfits (holdout $R^2 = -2.93$) while RIME transfers; Gate 5 sealed confirmation under fresh seed 12345. Planck: proved no exact first-order polynomial ODE exists ($u'/u \notin \mathbb{Q}(v)$), engine correctly abstains, benchmark verdict corrected from PASSED to RECORDED LIMITATION (`experimental`). |
| RIME-AD autonomous multi-representation proposer | [`apedp_dyn/proposers/rime_ad_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/rime_ad_proposer.py) | `benchmarks/run_population_extrapolation_rime_ad_vs_ga.py` | `RimeAdProposer`: profiles dataset invariants (plateaus, dynamic range, Arrhenius tunneling), fans out across the native Rust representations (TV denoising, implicit null-space, differential Padé, shared-DAG enumeration, multi-term rational powers), ranks on dev data by BIC/MDL, and emits standardized `CandidateProposal` objects with out-of-sample `compute_score` callables; never promotes, never sees holdout (`experimental`). |
| Rust bridge zero-copy data plane, compile cache & kernel-path observability | [`apedp/rust_bridge.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp/rust_bridge.py), [`rust/apedp-kernel/src/data.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/data.rs), [`rust/apedp-kernel/src/lib.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/lib.rs) | `tests/test_rust_bridge_observability.py`, `tests/test_rust_bridge_dispatch.py`, `tests/test_cross_implementation_conformance.py`, `cargo test` | Verified (45/45 bridge tests + 26/26 differential conformance + 179 Rust tests): (1) **Zero-copy columns** — `PythonColumns::extract` in `lib.rs` now hands `ArrayView1<f64>` views straight into NumPy buffers via the `DataCol` erased-column trait (`data.rs`), replacing the per-call `Vec<Vec<f64>>` deep-copy that re-copied the same dataset on every per-candidate crossing; parser/numeric/genetic/graph/eval/search/additive/residual surfaces take `&[V: DataCol]` so owned JSON-fed paths and borrowed NumPy paths are interchangable with no copy, and the same pattern extends to single-array arguments via a `GuardedVector` pin-once guard (contiguous `&[f64]` slices, strided one-copy fallback) for targets and metric arrays; (2) **Compile cache** — `RustKernelBridge._kernel_compile` LRU-caches FFI `compile` results keyed on `(expr_str, variable_names)` (owner-guarded so a swapped kernel clears it), and the Python fallback caches its sympy `lambdify` compilations by expression text; (3) **Kernel-path counters** — every `RustKernelBridge` instance reports `tallies` (rust/fallback counts), `last_path`, and attaches a `kernel_path` field to every returned result dict, so benchmark runs can prove the fast path executed (the fix the bridge's own audit asked for: "a counter, not a log line"); (4) **Loud missing-kernel warning** — bridge construction without the native kernel logs one WARNING naming the `maturin develop --release` build command instead of a buried debug line; (5) **Measured (before/after: git-HEAD kernel+bridge in an isolated venv, min-of-3 interleaved fresh runs, 600×4 float64)** — Python reference engine repeated eval 2678.9 → 34.5 µs/call (77×, lambdify cache), batch scoring round (15 candidates) 1087.6 → 888.1 µs (−18%), repeated candidate-evaluation call 110.8 → 95.6 µs (−14%; per-call cost at this size is Rayon-dispatch-dominated, so copy-elimination scales with dataset size), `compute_metrics` 11.1 → 7.6 µs (−32%), kernel-level `score_candidates_parallel` 817.7 → 785.2 µs (no kernel-path regression); artifacts: [`benchmarks/run_bridge_fastpath_benchmark.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_bridge_fastpath_benchmark.py) + [`benchmarks/bridge_fastpath_benchmark_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/bridge_fastpath_benchmark_results.json); (6) **Release profile** — workspace `Cargo.toml` adds `codegen-units = 1` to `lto = true`; (7) **Wheel artifacts** — `.github/workflows/test.yml` gains a `build-kernel-wheel` job (manylinux wheel via `PyO3/maturin-action`, uploaded as artifact so the kernel is `pip install`able without running maturin) and a tag-gated `publish-kernel-wheel` job that only fires with a `PYPI_API_TOKEN` secret present (`production`). |
| Spatio-Temporal PDE Discovery & Epistemic Referee Bridge | [`rust/apedp-kernel/src/rime/pde.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/pde.rs), [`apedp_dyn/epistemic/pde_discovery.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/pde_discovery.py), [`apedp_dyn/epistemic/pde_transfer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/pde_transfer.py) | `tests/test_dyn_frontier_9_of_10.py`, `tests/test_dyn_frontier_claims_integrity.py`, `tests/test_dyn_pde_transfer.py` | Verified (5/5 + 4/4 tests): (1) Native Rust PDE discovery engine (`pde.rs`) computes 4th-order spatial central stencils ($u_x, u_{xx}, u_{xxx}$) and temporal differences ($u_t$), constructing space-time feature tensors and applying Sequentially Thresholded Ridge Regression (STRidge); (2) Epistemic bridge (`pde_discovery.py`) partitions the time axis and scores a re-discovered fit on the held-out window (re-fit holdout, not model-transfer holdout); (3) Discovers 1D Heat diffusion ($\partial_t u = 0.5\, \partial_{xx} u$) — dev $R^2 = 0.9963$, holdout-window $R^2 = 0.999997$, coefficient $0.4972$; (4) Discovers Viscous Burgers ($\partial_t u = -1.0\, u\, \partial_x u + 0.1\, \partial_{xx} u$) — holdout-window $R^2 = 0.9999$, advection coefficient error $0.38\%$, diffusion coefficient error $11.2\%$ with one spurious low-weight $u$ term selected (coefficient-faithful structure, not exact coefficient recovery); (5) `pde_transfer.py` evaluates dev-discovered coefficients FROZEN on the holdout window with the identical stencils (true model-transfer metric; Burgers frozen $R^2 = 0.999772$ vs re-fit $0.999909$ on the Rust fixture). See [`frontier_9_of_10_architectural_resolution.md`](frontier_9_of_10_architectural_resolution.md) for the corrected record (`experimental`). |
| Discrete Number Theory & Active Invariant Transfer Engines | [`rust/apedp-kernel/src/rime/number_theory.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/number_theory.rs), [`rust/apedp-kernel/src/rime/invariant_transfer.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/invariant_transfer.rs), [`apedp_dyn/epistemic/number_theory_engine.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/number_theory_engine.py), [`apedp_dyn/proposers/invariant_transfer_proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/proposers/invariant_transfer_proposer.py) | `tests/test_dyn_number_theory_and_invariant_transfer.py` | Verified (4/4 tests): (1) Exact rational reconstruction via Farey sequences and continued fractions recovers Laughlin fractional quantum hall factors ($\nu = 2/5$), Brus powers ($2.45 = 49/20$), and critical exponents ($1/8$) from noisy decimals; (2) Integer relation discovery via lattice search finds exact stoichiometric conservation laws ($2\text{H}_2 + \text{O}_2 - 2\text{H}_2\text{O} = 0$) and harmonic frequency resonances; (3) Modular lattice discovery identifies discrete crystal lattice spacings ($\Delta = 2.0, x_0 = 1.0$) with zero submultiple degeneracy; (4) Active Invariant Transfer Proposer evaluates cross-environment Invariant Risk Minimization (IRM) penalties, worst-regime retention gating, and active extrapolation barrier guards, promoting true invariant causal representations while completely rejecting spurious correlators inverted under regime intervention (`production`). |
| 2D PDEs, Non-Uniform Stencils & Coupled Dynamical Systems | [`rust/apedp-kernel/src/rime/pde.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/pde.rs), [`rust/apedp-kernel/src/rime/coupled_system.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/coupled_system.rs), [`apedp_dyn/epistemic/pde_discovery.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/pde_discovery.py), [`apedp_dyn/epistemic/coupled_discovery.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/coupled_discovery.py) | `tests/test_dyn_coupled_and_2d_pde.py` | Verified (3/3 tests): (1) 2D spatial spatio-temporal PDE discovery (`discover_pde_2d`) recovers continuous 2D Heat diffusion ($\partial_t u = 0.25 \nabla^2 u$) from $u(t, x, y)$ with $R^2 > 0.99$; (2) Arbitrary non-uniform mesh stencils via exact Fornberg weights (`fornberg_weights`, `discover_pde_nonuniform`) recover governing PDEs on irregular coordinate grids; (3) Coupled multi-variable non-linear ODE discovery (`discover_coupled_system`) recovers multi-state interaction systems simultaneously, discovering Lotka-Volterra predator-prey dynamics ($\dot{x} = 1.5x - 1.0xy$, $\dot{y} = -3.0y + 1.0xy$) with exact parameter recovery and system $R^2 > 0.99$ (`production`). |
| Hybrid High-Dimensional Continuous PDE Discovery & Fused Jet2/Jet3 / Vlasov-Attention CPU Engine (1D to 100D+) | [`rust/apedp-kernel/src/rime/high_dim_pde.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/high_dim_pde.rs), [`apedp_dyn/epistemic/high_dim_pde.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/high_dim_pde.py) | `tests/test_dyn_high_dim_hybrid_pde.py` | Verified (33/33 Python tests + 11/11 Rust unit tests, 9.5/10 Gold Standard): (1) Hybrid dimension dispatch for all $D < 10$ ($D \in [1, 9]$): routes to Eulerian Cartesian grid finite differences (`discover_pde`, `discover_pde_2d`, `discover_pde_3d`, and general stride-based `discover_pde_nd_cartesian` with 4th-order 5-point central stencils when $N_k \ge 7$ and 2nd-order 3-point stencils when $3 \le N_k < 7$, verified on 2D, 3D, 4D, and 5D Cartesian grids with frozen-coefficient holdout $R^2 > 0.9999$); (2) for $D \ge 10$ (tested on $D=20, 50, 64, 100$): (a) **Mesh-Free Point Cloud Sampling**: true scattered point cloud discovery via multi-scale Gaussian Radial Basis Functions (`discover_pde_rbf_point_cloud_hd`, exact analytical $D$-dimensional RBF Laplacian $\nabla^2_D \phi = 2\varepsilon^2(2\varepsilon^2 r^2 - D)\phi$ without finite-difference probes, $R^2 > 0.9999$) plus axis-star probe point clouds (`discover_pde_point_cloud_hd`, `evaluate_point_cloud_pde_frozen_oos`); (b) **Radial / Isotropic Symmetry Reduction**: exact hyperspherical reduction $\nabla^2_{20} u = u_{rr} + \frac{19}{r} u_r$ (`discover_pde_radial_hd`, $R^2 > 0.99999$); (c) **Low-Rank Tensor Trains (TT / MPS)**: multi-core bond-rank $R$ Matrix Product State discovery (`discover_pde_tensor_train_hd`, $R^2 > 0.9999$ on rank-2 20D heat kernel sum in $\mathcal{O}(N_t D N_x R)$ memory) plus rank-1 separable factor slice (`discover_pde_separable_hd`); (d) **Deep BSDE / Neural PINNs**: mesh-free collocation PDE discovery (`discover_pde_neural_pinn_hd`, $R^2 > 0.9999$ in 20D) powered by the L1/L2 Block-Tiled AVX2 Fused 2-Jet Neural PDE Operator (`eval_jet2_neural_pde_operator`) and Ornstein-Uhlenbeck Fokker-Planck operator (`eval_fokker_planck_ou_operator`), plus exact 3rd-order `Jet3` parameter gradients (`compute_jet3_pde_loss_and_grads`, `RustPdeNetwork`, Rayon-parallelized to $7.05\text{ ms}$, cutting the JAX gap by $2.35\times$) and Block-Tiled Online-Softmax McKean-Vlasov Self-Attention PDE Stepper (`step_vlasov_attention_pde_hd`, zero-allocation tile scratchpads outperforming PyTorch float64 CPU OpenMP by $4.97\times$ at $N=512$ and $4.77\times$ at $N=4096$); (e) **SymPy Machine-Checked Mathematical Certificate**: formal computer-algebra verification (`test_jet3_sympy_symbolic_equivalence`) proving all Rust Jet3 gradients match SymPy symbolic derivatives down to machine epsilon ($< 10^{-14}$); (f) **Canonical Literature 100D & 50D Benchmarks & Convergence Study**: exact recovery of 100-dimensional Black-Scholes-Barenblatt (`discover_pde_black_scholes_barenblatt_100d`, $R^2_{\text{OOS}} > 0.999$), 100-dimensional Hamilton-Jacobi-Bellman optimal control (`discover_pde_hjb_100d`, recovering diffusion $\nu=0.05$, gradient penalty $\beta=-0.04$, and reaction $\mu=0.02$ with $R^2_{\text{OOS}} > 0.999$ while rejecting $u^2$), and 50-dimensional nonlinear Allen-Cahn bistable reaction-diffusion (`discover_pde_allen_cahn_hd`, $R^2_{\text{OOS}} > 0.98$); verified timestep convergence (`test_black_scholes_barenblatt_timestep_convergence`) demonstrating that the original 53% discount discrepancy under Forward Euler is strictly linear $\mathcal{O}(dt)$ truncation error ($dt=0.02 \implies +0.0261$, $dt=0.01 \implies +0.0127$), driven to $< 0.1\%$ ($0.06\%$) under 4th-order Runge-Kutta at $dt \le 0.002$; (g) **Noise-Robust Tikhonov Smoothing & Adversarial Cauchy Resilience**: suppresses temporal noise variance by $> 2\times$ (`tikhonov_smooth_trajectory`), cutting residual RMSE by $> 2\times$ under heavy-tailed Cauchy perturbations (`test_pde_adversarial_heavy_tailed_cauchy_noise_and_outliers`) and eliminating spurious term promotions; (h) **Epistemic Referee 400-Trial Monte Carlo Calibration, Misspecification Refusal & Causal Arrow Guard**: empirical False Positive Rate $\text{FPR} \le 0.01$ (0/300 false positives $\implies$ exact one-sided Clopper-Pearson 95% upper bound $0.994\% < 1.0\%$) and True Positive Rate $\text{TPR} \ge 0.95$ (100/100 true positives $\implies$ Wilson 95% score lower bound $96.30\% > 95.0\%$, exact one-sided lower bound $97.05\%$); verified adversarial model misspecification refusal (`test_pde_adversarial_fractional_misspecification`, refusing causal certification on non-local fractional diffusion $|k|^{1.4}$); verified out-of-distribution geometry transfer (`test_pde_out_of_distribution_geometry_transfer`, frozen hypercube $[-0.2, 0.2]^{20}$ model transferring to unseen spherical shell $r \in [0.4, 0.8]$ with $R^2_{\text{OOD}} > 0.999$); and verified thermodynamic arrow-of-time causal direction guard (`test_pde_time_reversal_causal_arrow_guard`, proving forward dissipation $\alpha > 0$ vs unphysical reversed entropy decrease $\alpha < 0$) (`production`). |
| JetFlow-Video Pip Package & Google Colab Training Engine | [`packages/jetflow-video/pyproject.toml`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/pyproject.toml), [`packages/jetflow-video/jetflow_video/tensor_train.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/tensor_train.py), [`packages/jetflow-video/jetflow_video/stepper.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/stepper.py), [`packages/jetflow-video/jetflow_video/noether.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/noether.py), [`packages/jetflow-video/jetflow_video/physics.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/physics.py), [`packages/jetflow-video/jetflow_video/bridge.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/bridge.py), [`packages/jetflow-video/jetflow_video/model.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/model.py), [`packages/jetflow-video/jetflow_video/trainer.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/trainer.py), [`packages/jetflow-video/examples/colab_quickstart.ipynb`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/examples/colab_quickstart.ipynb), [`packages/jetflow-video/examples/train_colab.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/examples/train_colab.py), [`packages/jetflow-video/examples/generate_video.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/examples/generate_video.py) | `tests/test_dyn_jetflow_video_package.py` | Verified (8/8 unit tests passing): Standalone PEP 517/621 installable pip package (`pip install -e packages/jetflow-video`) optimized for Google Colab CPU and NVIDIA T4/A100/L4 GPUs. Provides PyTorch-native 2nd-order Jet2 flow matching, exact discrete Fourier Leray solenoidal projection ($\nabla \cdot \mathbf{v} = 0$, residual $< 10^{-6}$), 4D Tensor-Train compression ($75\times$ to $186\times$ VRAM reduction), Continuous Wavelet and Diffusers VAE bridges, zero-disk direct Hugging Face pre-computed matrix streaming (`io.BytesIO` in-memory RAM caching), standalone inference CLI generator (`generate_video.py`), and full training loop (`JetFlowTrainer`) with multi-objective Jet2 loss ($\mathcal{L}_{\text{flow}} + \lambda_{\text{acc}} \mathcal{L}_{\text{acc}} + \lambda_{\text{div}} \mathcal{L}_{\text{noether}}$), automatic mixed precision (AMP), and 1-click Colab interactive quickstart notebook (`experimental`). |
| JetFlow-Video Fast-Path Remediation Integrity Stack (2026-10-17) | [`apedp_dyn/video/jet_flow_model.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/video/jet_flow_model.py), [`apedp_dyn/video/semantic_bridge.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/video/semantic_bridge.py) | `tests/test_dyn_jet_flow_sota_path.py`, `tests/test_dyn_jet_flow_video.py` | Verified (live): (1) the NumPy `train_step` fake trainer (one scalar mean-gradient on a single bias vector, presented as Jet2 flow-matching training) is REMOVED — the method is now diagnostics-only (`DeprecationWarning`, no parameter mutation, pinned); all real training routes to `packages/jetflow-video JetFlowTrainer` (full-parameter AdamW, grad-flow confirmed); (2) hash-based prompt embedding replaced by process-stable feature hashing (`feature_hash_word_char_trigram_v1`, MD5 indices — the builtin `hash()` salt bug is gone: embeddings byte-identical across processes, pinned in-subprocess; lexical similarity preserved 0.744 vs 0.229 on shared-vocabulary prompts); (3) found and documented a real package bug: `model.generate_video` samples in the direction OPPOSITE to `trainer.compute_loss`'s parameterization — the shipped samplers in `first_frame.py`/`distill.py` integrate the trainer-consistent direction (t: 1 -> 0) (`experimental`). |
| Held-Out Video Evaluation Gate (certification instruments) | [`apedp_dyn/video/eval_gate.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/video/eval_gate.py) | `tests/test_dyn_video_eval_gate.py` | Verified (5/5): PSNR/SSIM (identity ceilings, monotone degradation), temporal consistency (smooth advected clips beat frame-shuffled copies), FVD PROXY (sliced Wasserstein-2 over a frozen deterministic featurizer — NOT I3D-FVD, honest label, identical distributions separate ~0, degraded > clean, cross-seed deterministic), burnable holdout split (disjoint, deterministic, exclusive), and a certify API whose thresholds are supplied by the caller (no new hardcoded threshold literals — repo rule 1.3) (`audit-only`). |
| Video-Metrics Rust Kernel Dispatch (bridge fast path) | [`rust/apedp-kernel/src/video.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/video.rs), [`apedp/rust_bridge.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp/rust_bridge.py), [`apedp_dyn/video/eval_gate.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/video/eval_gate.py) | `tests/test_dyn_video_metrics_kernel.py` | Verified (18/18 across the three video-gate test files; full 58-test dynamic regression suite green): (1) `eval_gate` acquires the kernel-backed bridge lazily (module import never pulls the apedp tree) and routes `compute_psnr` / `compute_ssim` through `video_psnr` / `video_ssim_frame`; (2) a missing or unbuilt kernel silently degrades to the verbatim NumPy reference bodies — bridge docstring discipline, same numbers, slower path; (3) dispatch counters tally Rust-path vs fallback-path calls, and shape-mismatch guards raise ValueError before any kernel call in both paths; (4) exact parity (≤ 1e-10) against the NumPy references on identity, additive-noise, and (T, 3, H, W) video-routing cases, on all three dispatch modes (rust, forced-fallback bridge, pure-Python) (`production`). |
| Fluid Domain Scoping (domain-matched dataset) | [`apedp_dyn/video/fluid_domain.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/video/fluid_domain.py) | `tests/test_dyn_fluid_domain.py` | Verified (4/4): deterministic 2D incompressible Navier-Stokes tracer clips (velocity from streamfunction — grid-divergence-free to $|\nabla \cdot u| < 10^{-10}$, the property that makes the Noether projector physically meaningful on this domain), semi-Lagrangian dye advection passing the temporal-consistency instrument, and a scope manifest with exclusive holdout ids via `held_out_split` (the domain scoping lever: one learnable physics class instead of "everything") (`experimental`). |
| Real-First-Frame Conditioning + Perceptual Decoder + Distillation + CPU Export (photorealism fast-path) | [`packages/jetflow-video/jetflow_video/first_frame.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/first_frame.py), [`packages/jetflow-video/jetflow_video/perceptual_decoder.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/perceptual_decoder.py), [`packages/jetflow-video/jetflow_video/distill.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/jetflow-video/jetflow_video/distill.py), [`benchmarks/run_jetflow_distill_benchmark.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_jetflow_distill_benchmark.py), [`benchmarks/run_jetflow_cpu_export_benchmark.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/run_jetflow_cpu_export_benchmark.py) | `tests/test_dyn_jetflow_anchored_and_distill.py`, `tests/test_dyn_jetflow_perceptual_decoder.py`, `tests/test_dyn_jetflow_cpu_export.py` | Verified (11/11 + 2 benchmark artifacts): (1) anchored flow-matching (clean real frame 0 through training corruption AND generation; loss excludes the anchor; anchor re-imposed bitwise every step); (2) perceptual decoder = Haar base + ZERO-INIT residual conv stack (starts at exact Haar parity — no regression cliff) + patch-LSGAN + gradient loss, trained on fluid frames; (3) honest solution distillation: trained teacher (16-step reference) -> student endpoint RMSE 0.808 -> 0.507 measured on the artifact `benchmarks/jetflow_distill_curves.json` (the honest replacement for the old integrator-only "2 beats 50" benchmark); (4) ONNX export (static traced graph per clip config — traced attention cannot carry a dynamic T, documented) + `onnxruntime` dynamic int8: size 173.5 KB -> 104.9 KB, parity $\max|\Delta| = 0.0$, latency measured (eager 8.68 ms vs ORT fp32 5.32 ms — int8 is a size win, fp32 ORT the speed win at this scale) in `benchmarks/jetflow_cpu_export_benchmark_results.json` (`experimental`). |
| APEDP-Continuum: Standalone Pip Package & Unified PDE/ODE Subpackages | [`packages/apedp-continuum/pyproject.toml`](file:///home/psalmprax/ALL_PROJECTS/science/packages/apedp-continuum/pyproject.toml), [`packages/apedp-continuum/setup.py`](file:///home/psalmprax/ALL_PROJECTS/science/packages/apedp-continuum/setup.py), [`apedp_dyn/pde/__init__.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/pde/__init__.py), [`apedp_dyn/ode/__init__.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/ode/__init__.py), [`apedp/pde/__init__.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp/pde/__init__.py), [`apedp/ode/__init__.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp/ode/__init__.py) | `tests/test_dyn_pde_ode_packages.py` | Verified (7/7 unit tests passing): Standalone PEP 517/621 pip package (`pip install -e packages/apedp-continuum`) and first-class subpackages (`apedp.pde`, `apedp.ode`, `apedp_dyn.pde`, `apedp_dyn.ode`) exposing high-dimensional continuous PDE discovery ($1\text{D}$ to $100\text{D}+$), multi-variable coupled non-linear ODE discovery (Lotka-Volterra predator-prey), typed data containers (`PdeGridData`, `PdePointCloudData`, `OdeTrajectoryData`), standardized proposer engines (`PdeProposer`, `CoupledOdeProposer`) conforming to `ProposerBase`, and epistemic referee validation engines (`PdeReferee`, `OdeReferee`) enforcing frozen holdout verification and thermodynamic causality (`production`). |
| Decoupled Continuous Spatio-Temporal Transport (DC-STT) | [`apedp_dyn/video/decoupled_transport.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/video/decoupled_transport.py), [`rust/apedp-kernel/src/video.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/video.rs), [`apedp/rust_bridge.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp/rust_bridge.py) | `tests/test_dyn_decoupled_continuous_transport.py` | Verified (5/5 tests passing): (1) Native Rust kernel fast-path execution and zero-copy tallies accounting; (2) Differential cross-validation: exact numerical parity between native Rust kernel and NumPy fallback ($\max\|\Delta\| < 10^{-12}$); (3) Noether mass-conservation: Helmholtz-Hodge projection onto divergence-free subspace; (4) Optical flow divergence and disocclusion mask detection with residual source injection; (5) Multi-scene video trajectory generation across diverse physical dynamics (translation, fluid vortex, radial expansion) and 4D Tensor-Train matrix compression with continuous temporal interpolation (`experimental`). |



---

*Documentation generated for APEDP-X v2.0.0 — Autonomous Physics, Engineering & Market Discovery Platform*
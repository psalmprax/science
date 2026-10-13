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
| [`apedp_dyn/epistemic/claim_ladder.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/claim_ladder.py) | Epistemic claim ladder calibrating warrant strength from speculative to admitted. | `production` |
| [`apedp_dyn/epistemic/cross_asset_gate.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/cross_asset_gate.py) | Multi-asset cross-sectional transfer gate enforcing zero contamination across symbols. | `production` |
| [`apedp_dyn/epistemic/market_extrapolation.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/market_extrapolation.py) | Out-of-distribution regime transfer gate evaluating volatility and liquidity shocks. | `production` |
| [`apedp_dyn/epistemic/frozen_spec.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/frozen_spec.py) | Cryptographic hash verification of referee source files to prevent silent code drift. | `production` |
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
| [`rust/apedp-kernel/src/genetic.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/genetic.rs) | Rust-accelerated genetic algorithm evaluator with verified minimum-MSE selection. | `production` |
| [`rust/apedp-kernel/src/rime/graph.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/graph.rs) | RIME typed mechanism graph: series-at-a-time topological evaluation, `State` recursion, structural validation, canonical SHA-256 ids. | `experimental` |
| [`rust/apedp-kernel/src/rime/rewrite.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/rewrite.rs) | The five frozen RIME structural rewrites (lag, interaction, state, nonlinearity wrap, prune). | `experimental` |
| [`rust/apedp-kernel/src/rime/eval.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/eval.rs) | RIME scorer: ridge-fitted linear readout, charged degrees of freedom, parallel batch evaluation. | `experimental` |
| [`rust/apedp-kernel/src/rime/seeds.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/seeds.rs) | Deterministic 60-form seed catalogue (levels, delays, differences, lead-lag interactions, memory, wrappers). | `experimental` |
| [`rust/apedp-kernel/src/rime/search.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/search.rs) | Deterministic budgeted beam search over the frozen rewrites; null-model-anchored gate with EBIC selection charge. | `experimental` |
| [`rust/apedp-kernel/src/rime/record.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/src/rime/record.rs) | Immutable content-addressed RIME candidate records (graph id, parent, rewrite op, seed, dev score). | `experimental` |
| [`benchmarks/rime_vs_neural.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/rime_vs_neural.py) | Head-to-head benchmark: RIME vs ridge and MLP baselines across planted mechanisms, sample sizes, and a null arm. | `audit-only` |
| [`benchmarks/matched_compute.py`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/matched_compute.py) | v0.1.3 matched-compute comparison harness: RIME (temporal + static grammars) vs existing proposers vs PySR 2.7 head-to-head (serial-deterministic, binding timeout, lag columns for information parity), at equal wall budget with 5 repeats/utilization/environment metadata, scored on a preregistered grammar-coverage axis (out-of-coverage abstention is correct; emitted-but-failed is failed; static worlds require reversal-invariance so grid-order tracking scores `index_fit`.). Measured outcome: complementarity, not dominance — RIME-temporal 4/5, PySR 3/5, each recovering a family the other misses. | `audit-only` |
| [`rust/apedp-kernel/tests/rime_profile.rs`](file:///home/psalmprax/ALL_PROJECTS/science/rust/apedp-kernel/tests/rime_profile.rs) | Kernel phase-decomposition profile (graph eval / readout fit / search bookkeeping remainder; median of 7): bookkeeping ≈90% of per-candidate cost at n=150, graph evaluation dominating only at n=12 000. Artifact: `benchmarks/kernel_profile_results.json`. | `audit-only` |
| [`specifications/canonical_preregistered_spec_v1.json`](file:///home/psalmprax/ALL_PROJECTS/science/specifications/canonical_preregistered_spec_v1.json) | Canonical frozen specification containing all preregistered thresholds. | `production` |
| [`apedp_dyn/conf/benchmark.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/conf/benchmark.py) | Legacy benchmark configuration and baseline threshold profiles. | `deprecated` |
| [`apedp_dyn/epistemic/transfer_scope.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/transfer_scope.py) | Out-of-distribution transfer scope classifier evaluating empirical domain boundaries. | `production` |
| [`apedp_dyn/epistemic/diagnosis.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/diagnosis.py) | Diagnostic failure attribution determining why candidates were refused at gates. | `production` |
| [`apedp_dyn/epistemic/bootstrap.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/bootstrap.py) | Block-bootstrap and stationary bootstrap statistical confidence interval estimator. | `production` |
| [`apedp_dyn/epistemic/term_necessity.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/term_necessity.py) | Term necessity and coefficient ablation testing framework. | `production` |
| [`apedp_dyn/epistemic/discovery_stability.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/discovery_stability.py) | Discovery stability harness measuring parameter sensitivity across random seeds. | `production` |
| [`apedp_dyn/epistemic/agnostic_adapter.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/agnostic_adapter.py) | Domain-agnostic evidence ingestion adapter standardizing holdout inputs. | `production` |
| [`apedp_dyn/epistemic/calibration.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/calibration.py) | Model probability and uncertainty calibration verification. | `production` |
| [`apedp_dyn/epistemic/pointer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/pointer.py) | Epistemic pointer tracking references to empirical artifacts and receipts. | `production` |
| [`apedp_dyn/epistemic/followup.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/followup.py) | Automated followup experiment recommendation and active disambiguation engine. | `experimental` |
| [`apedp_dyn/grammar/mechanical_grammar.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/mechanical_grammar.py) | Mechanical and physical dynamical systems grammar definitions. | `production` |
| [`apedp_dyn/grammar/proposer.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/proposer.py) | Core grammar search proposer generating candidate AST expressions. | `production` |
| [`apedp_dyn/grammar/coupled.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/coupled.py) | Coupled multi-variable differential equation grammar definitions. | `production` |
| [`apedp_dyn/grammar/canonicalize.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/grammar/canonicalize.py) | Symbolic expression canonicalization and term ordering. | `production` |

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
| RIME Repair v0.1.1–v0.1.4 bounded engine (R1/R2/R3 + 4-role validation) | [`apedp_dyn/epistemic/rime_repair_engine.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/rime_repair_engine.py) | `tests/test_dyn_rime_repair_v011_engine.py` | Verified: planted intercept-shift→R1, missing-square→R2, emergent-lag→R3 each IMPROVED with sealed gain over frozen original; noise→no repair + sealed stays shut (`HoldoutContaminationError` on reopen); adequate original→no false repair; equivalence clustering preserves ambiguity; promotion deferred to APEDP-X track (`experimental`). |
| RIME Repair sensitivity/recall milestone (v0.1.2–v0.1.3) | [`apedp_dyn/epistemic/rime_repair_engine.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/rime_repair_engine.py) | `tests/test_dyn_rime_repair_v012_sensitivity.py`, `tests/test_dyn_rime_repair_v013_realdata.py` | Verified: recall 9/9 strong defects with structure recovery (right operator/variable); weak-defect curve never harms sealed outcome; BTC panel abstention regression (repair-R²=−0.0462, sealed shut); 5/5 walk-forward windows abstain with persistence RMSE reported; chance predictive pass blocked at stability gate (`experimental`). |
| KNOWN LIMITATION — same-driver omitted lags vs rho1 screens | `tests/test_dyn_rime_repair_v012_sensitivity.py::test_same_driver_omitted_lag_sensitivity_map` | Sensitivity map (6 corners) | **Documented boundary**: single-regressor OLS residuals on an AR driver are usually absorbed white (4/6 corners silent). Firing corners (state-cell misfit remainders) yield gated rejection, not false wins. R3's demonstrated domain is emergent lag from an independent driver. Whiteness alone is insufficient for structural identification. |
| KNOWN LIMITATION — whiteness gate under geometric-spacing extrapolation (QEC) | `tests/test_dyn_rime_repair_v014_qec.py::test_qec_degraded_linear_repair_refused_despite_high_r2` | Pinned conservative behavior | **Documented boundary**: on log-spaced p, quadratic extrapolation misfit grows smoothly into validation rows and trips the temporal screen, vetoing an R²=0.95 repair. Refusal is the safe failure direction; gate assumes stationarity across evaluation rows (valid for time series). No gate respecification without its own preregistered benchmark. |
| GA on QEC vs OLS vs RIME (method boundary) | `tests/test_dyn_ga_qec.py` | Pinned three-way + incident note | Verified: GA selects {1, p²}, holdout RMSE 0.002220 = same-split OLS (C/p_th²≈1000 recovered); RIME raw-[p] deep search recovers via seeded self-interactions (holdout R² 0.9995, stable); shallow-config single divergent run (lag winner, 1 of ~14) explained by a concurrent kernel rebuild landing mid-measurement (absolute- vs relative-ridge binaries), not search nondeterminism. A concurrent session's scale-invariant ridge fix (`eval.rs`: relative `ridge·Σo²`, commit 9781a3d) was verified real and behavior-neutral on unit-scale tests; an unattributed edit to this row/test asserting it was first reverted, then re-verified and corrected here. Coordination via separate worktrees required going forward. |
| Frozen-equation OOS (QEC Eq1 vs Eq2, no refit) | `tests/test_dyn_rime_qec_oos_frozen.py` | Pinned comparison | Verified: T1 fresh seeds R²>0.999 both; Eq1≡Eq2 to ~2e-5 on log grids (fitted scales absorbed the same grid ratio — one hypothesis, not two); T2 uniform grid R²>0.99 with doubled MAE; T3/T4 d=5 graceful failure (R² 0.78→0.66, matching frozen OLS-quad); mean baseline negative throughout. Extended: residual profile grows HIGH-end (Eq1 0.0038→0.0137, OLSQ flat ~1.1×); irregular grid distinguishes generator (fresh OLS w1=1001.4, noise floor hit; frozen Eqs 1.3–1.8× floor); transfer R² decays monotonically d=3→6, noise-independent (structural failure). |
| Real-Market Trading Predictive Profitability | N/A (Candidate evaluations in `sandbox/cpmd_*.py`) | None (Empirical holdouts failed) | **Explicitly Unverified / Unproven**: H-LIK and H-XSC failed OOS confirmation. |

---

*Documentation generated for APEDP-X v2.0.0 — Autonomous Physics, Engineering & Market Discovery Platform*
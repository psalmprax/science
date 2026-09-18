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

---

## 1. Overview

APEDP (Autonomous Physics & Engineering Discovery Platform) is an end-to-end scientific computing platform for discovering, validating, and repairing physical equations from experimental data. It combines symbolic regression, dimensional analysis, Bayesian model selection, physical anomaly repair, epistemic tracking, quantum error correction, metrology, and applied defect auditing in a unified framework.

For an in-depth breakdown of system capabilities and why the platform balances autonomous discovery with domain constraints, see [System Capabilities & Architectural Philosophy](SYSTEM_CAPABILITIES_AND_PHILOSOPHY.md).

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

*Documentation generated for APEDP v1.0.0 — Autonomous Physics & Engineering Discovery Platform*
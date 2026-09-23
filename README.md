# APEDP — Autonomous Physics & Engineering Discovery Platform

A scientific computing platform for dimensional assurance, symbolic regression,
Bayesian model selection, physical anomaly repair, epistemic tracking,
metrology analysis, quantum error correction, and applied defect auditing.

APEDP is domain-general at its evidence and adjudication layer, not fully
domain-agnostic at discovery. The core can evaluate provenance, locked
holdouts, dimensional consistency, numerical conditioning, sensitivity, and
refusal across domains; discovery grammars, units, physical constraints,
baselines, and data adapters remain domain-specific where needed.

## Architecture

```
apedp/
├── api               FastAPI REST server (discovery, metrology, repair, epistemic)
├── core              DimensionalAssuranceEngine, ClosedLoopAssuranceSystem, PHYSICAL_CONSTANTS
├── metrology         AnalyticalUncertaintyPropagator, MetrologySensorStream, CalibrationTraceabilityChain
├── discovery         BayesianRefinementEngine, MultiVariableSRE, EvolutionaryDiscoveryEngine
├── repair            PhysicalAnomalyRepairEngine, AgnosticClosedLoopAssuranceSystem
├── epistemic         EpistemicGraph, LifecycleManager, RequirementsTraceabilityMatrix
├── applied_defects   UniversalAppliedDefectEngine, domain-specific auditors
├── qec               Quantum error correction (Knill-Laflamme, surface codes, etc.)
├── scenarios         APEDPScenarioSuite (22 scenarios)
├── ingestion         External data sources (academic APIs, DOCX/PDF, Wikipedia)
├── symbolic_regression SymbolicRegressor with evolutionary search
├── visualization     Plotting utilities (fitness landscapes, uncertainty, calibration, epistemic graphs)
├── persistence       ModelStore, ExperimentLog, serialization
├── pipeline          End-to-end discovery pipeline (ingest → discover → validate → repair → persist)
└── __init__.py       Unified public API
```

## Quick Start

```python
from apedp import DimensionalAssuranceEngine, ClosedLoopAssuranceSystem
from apedp.pipeline import DiscoveryPipeline

engine = DimensionalAssuranceEngine()
platform = ClosedLoopAssuranceSystem(engine)

m = engine.register_variable('m', 'kilogram')
c = engine.register_variable('c_light', 'meter / second')
expr = m * c**2

platform.register_model_spec('Einstein_Energy', expr, 'joule')
val = platform.evaluate_model('Einstein_Energy', {'m': 1.0, 'c_light': 299792458.0})
print(f"E = mc^2 = {val:.4e} J")  # 8.988e+16 J

# End-to-end pipeline
pipeline = DiscoveryPipeline()
result = pipeline.run("data.csv", "target_column")
```

## API Server

```bash
APEDP_API_KEY='replace-with-a-secret' uvicorn apedp.api:create_app --factory --host 0.0.0.0 --port 8000
```

The API requires `APEDP_API_KEY` by default. For local-only development, set
`APEDP_ALLOW_UNAUTHENTICATED=1`; do not use that mode on a network-exposed
deployment. Discovery responses include a data-adequacy report, and production
claims should use `DiscoveryPipeline.run_prospective()` with a locked holdout.

Available endpoints (all verified against the real module APIs; see `docs/DOCUMENTATION.md`):
- `GET /health` — Service health
- `POST /discover/regress` — Symbolic regression (returns penalized MSE, lower is better; not R²)
- `POST /discover/bayes` — Bayesian model comparison
- `POST /metrology/propagate` — GUM uncertainty propagation for a SymPy expression
- `POST /metrology/calibrate/log` — Log a calibration record
- `POST /metrology/calibrate/verify` — Verify traceability for a sensor
- `POST /repair/anomalies` — Boundary anomaly stress-test and regularization
- `POST /epistemic/node` — Add a knowledge-graph node
- `POST /epistemic/edge` — Add a lineage edge
- `POST /epistemic/requirement` — Register a traceability requirement
- `POST /epistemic/tournament` — Run an adversarial deliberation tournament

## Running Tests

```bash
pytest tests/ -v
```

## Features

- **Dimensional Assurance**: Full SI unit checking with SymPy/pint integration
- **Symbolic Regression**: Evolutionary search for dimensionally-consistent equations
- **Bayesian Model Selection**: Occam's complexity penalty for model comparison
- **Physical Anomaly Repair**: Automatic singularity detection and regularization
- **Epistemic Tracking**: Adversarial deliberation tournaments, requirements traceability
- **Quantum Error Correction**: Knill-Laflamme conditions, surface code thresholds
- **Applied Defect Auditing**: Physics, chemistry, materials science, nanotechnology
- **Metrology**: GUM uncertainty propagation, calibration traceability chains
- **REST API**: Full FastAPI server with OpenAI-compatible endpoints
- **Visualization**: Fitness landscapes, uncertainty plots, calibration curves, epistemic graphs
- **Persistence**: Save/load discovered models with versioning
- **Pipeline**: End-to-end ingest → discover → validate → repair → persist
- **Evidence adjudication**: `ADMIT`, `DOMAIN_LIMITED`, `CONDITIONAL`, `PREDICTIVE_ONLY`, `REFUSE`, and `INCONCLUSIVE` outcomes
- **Benchmark protocol**: Locked datasets, information firewalls, separated prior tracks, adapters, and documented sensitivity variants
- **Identifiability diagnostics**: Rolling-origin parameter drift reported separately from forecast accuracy
- **Real-data inverse validation**: Real-RINEX (real-rinex) GPS structural recovery with grouped satellite holdout and grouped satellite validation diagnostics
- **Docker**: Containerized deployment
- **CI/CD**: GitHub Actions with testing, linting, type checking, security scanning

## Documentation

- Full documentation is available at [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).
- System capabilities and architectural philosophy: [docs/SYSTEM_CAPABILITIES_AND_PHILOSOPHY.md](docs/SYSTEM_CAPABILITIES_AND_PHILOSOPHY.md).
- Discovery engine ownership and evidence boundaries: [docs/DISCOVERY_ENGINE_BOUNDARIES.md](docs/DISCOVERY_ENGINE_BOUNDARIES.md).
- Publication-data discovery findings: [docs/PUBLICATION_DATA_FINDINGS.md](docs/PUBLICATION_DATA_FINDINGS.md).

Current scientific claims should distinguish candidate generation from
admission. A low-error equation is not automatically a scientific law:
prospective holdout validation, physical admissibility, robustness, and
provenance determine the qualified outcome.

## Small Physics LLM (Proposed)

A planned extension: a small 1.5–3.8B parameter domain-specific LLM fine-tuned on physics equations that collaborates with APEDP's verification engine. The LLM handles creative reasoning while APEDP handles dimensional validation, Bayesian scoring, and anomaly repair. See [Small Physics LLM section](docs/DOCUMENTATION.md#13-small-physics-llm) for details.

## License

MIT — see `LICENSE` file.

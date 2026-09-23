# Phase Plan: SOTA Neural Discovery for APEDP

## Phase Goal
Replace the prototype sklearn-based MLP surrogate (`neural_relevance`, `distill_via_surrogate` in `discovery.py:466-638`) with a production-grade, JAX-based neural discovery system featuring:
- Modern architectures (ResNet, Transformer, Graph Networks for symbolic tasks)
- Full uncertainty quantification (Bayesian NN via SVI, Deep Ensembles, Conformal Prediction)
- Differentiable symbolic regression (gradient-based expression discovery)
- Active learning / experimental design integration
- Rigorous validation (k-fold CV, repeated seeds, OOD benchmarks, hyperparameter optimization)

## Success Criteria (Measurable)
| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Neural surrogate R² on held-out data | ≥ 0.95 on benchmark physics datasets | k-fold CV (k=5, 5 seeds) |
| Uncertainty calibration (ECE) | < 0.05 | Conformal prediction coverage |
| Symbolic recovery rate (SRBench) | ≥ 80% on Feynman equations | SRBench subset, 100 equations |
| OOD detection AUROC | ≥ 0.90 | Distribution shift benchmarks |
| Inference latency (CPU) | < 10ms/sample | `jax.jit` compiled, batch=1 |
| Integration test pass rate | 100% | Existing `test_selection_sampler_gates.py` + new tests |

## Architecture Overview

```
apedp/neural_discovery/
├── __init__.py              # Public API exports
├── architectures.py         # JAX/Flax models (ResNet, Transformer, GNN)
├── uncertainty.py           # Bayesian NN (SVI), Deep Ensembles, Conformal
├── symbolic.py              # Differentiable symbolic regression
├── active_learning.py       # Acquisition functions, experiment design
├── training.py              # Training loops, hyperopt, CV
├── validation.py            # OOD benchmarks, calibration metrics
├── integration.py           # Bridge to discovery.py, model_discovery.py
└── tests/
    ├── test_architectures.py
    ├── test_uncertainty.py
    ├── test_symbolic.py
    ├── test_active_learning.py
    ├── test_integration.py
    └── test_benchmarks.py
```

## Task Breakdown

### Task 1: Core Architectures (`architectures.py`) — 3 days
**Deliverable**: JAX/Flax implementations of:
- `ResNetBlock` / `MLPResNet` — Residual MLP with LayerNorm, Swish, dropout
- `SymbolicTransformer` — Transformer encoder for expression sequences (prefix notation)
- `EquationGNN` — Graph network over expression trees (variables=ops=nodes)
- `PhysicsInformedNN` — PINN-style with PDE residual loss (reuse `jax_pde_solver.py`)

**Requirements**:
- All models `jax.jit` compilable, `grad` compatible
- Configurable via `NeuralDiscoveryConfig` dataclass
- Support variable input dimensions (Dict[str, Array] → Array)
- Deterministic with seed

**Acceptance**: Unit tests for forward/grad/JIT; benchmark vs sklearn MLP on 5 physics datasets.

### Task 2: Uncertainty Quantification (`uncertainty.py`) — 3 days
**Deliverable**:
- `BayesianNN` — Mean-field variational inference (SVI) with JAXopt/BlackJAX
- `DeepEnsemble` — 5+ independently trained members, shared architecture
- `ConformalPredictor` — Split conformal prediction for distribution-free coverage
- `CalibrationMetrics` — ECE, NLL, sharpness, coverage at 90/95/99%

**Requirements**:
- `predict_with_uncertainty(x) → (mean, std, lower, upper)` API
- Conformal: marginal coverage ≥ 90% on held-out
- Ensemble diversity: pairwise disagreement > 0.1
- SVI: ELBO convergence monitored, KL tracked

**Acceptance**: Calibration plots on 3 UCI regression + 3 physics datasets; ECE < 0.05.

### Task 3: Differentiable Symbolic Regression (`symbolic.py`) — 4 days
**Deliverable**:
- `NeuralSymbolicRegressor` — End-to-end differentiable SR:
  - Neural network proposes expression tree (Transformer decoder)
  - Symbolic execution via JAX (differentiable operators: +,-,*,/,sin,cos,exp,log,pow)
  - Loss: MSE + complexity penalty (node count)
  - Training: RL (REINFORCE) + supervised (behavioral cloning from `bic_forward_select`)
- `ExpressionTokenizer` — Prefix notation tokenizer for expressions
- `JAXSymbolicExecutor` — Pure JAX expression evaluation with autodiff

**Requirements**:
- Recovers `1/R**2`, `1/R`, `x**2`, `exp(-x)` on synthetic benchmarks
- Gradient flows to expression parameters (coefficients, exponents)
- Integrates with `sparse_select_terms` as initialization
- Beats `SymbolicRegressor` (legacy GA) on Feynman subset

**Acceptance**: 80% exact recovery on 50 Feynman equations (noise-free); 60% with 5% noise.

### Task 4: Active Learning / Experimental Design (`active_learning.py`) — 2 days
**Deliverable**:
- `AcquisitionFunction` base class + implementations:
  - `BALD` (Bayesian Active Learning by Disagreement)
  - `ExpectedImprovement` (for optimization)
  - `MaxVariance` / `MaxEntropy` (uncertainty sampling)
  - `DiversityAcquisition` (k-means++ on embeddings)
- `ActiveLearningLoop` — Orchestrates: train → acquire → label → retrain
- Integration with `experiment_designer.py` (EIG proxy) and `closed_loop.py`

**Requirements**:
- Reduces label budget by ≥ 40% vs random sampling on 3 benchmarks
- Respects domain constraints (from `check_data_adequacy`)
- Batch acquisition (q=5-10) with diversity

**Acceptance**: Active learning curves on battery degradation, protein folding, QD datasets.

### Task 5: Training, Hyperopt, CV (`training.py`) — 2 days
**Deliverable**:
- `NeuralDiscoveryTrainer` — Unified training loop:
  - k-fold CV (k=5 default) with repeated seeds (5 seeds)
  - Early stopping on validation loss (patience=50)
  - Learning rate scheduling (cosine annealing + warmup)
  - Gradient clipping, mixed precision (bfloat16)
- `HyperparameterOptimizer` — Optuna integration:
  - Search space: lr, width, depth, dropout, weight_decay, ensemble_size
  - Pruning (MedianPruner), 50 trials default
- `ModelCheckpoint` — Best by validation R², saves eqx tree

**Requirements**:
- Reproducible: fixed seed → identical weights
- Logs: TensorBoard/W&B compatible (JSONL fallback)
- CPU/GPU/TPU agnostic (JAX backend)

**Acceptance**: Hyperopt finds config within 5% of manual tuning on 3 datasets in < 1hr.

### Task 6: Validation & OOD Benchmarks (`validation.py`) — 2 days
**Deliverable**:
- `OODBenchmarkSuite` — Distribution shift tests:
  - Covariate shift (extrapolation beyond training range)
  - Concept shift (different physics regime)
  - Dataset shift (Feynman → real experimental)
- `CalibrationSuite` — Reliability diagrams, ECE, sharpness
- `SRBenchEvaluator` — Feynman/Feynman-extended benchmark runner
- `ComparisonBaseline` — sklearn MLP, XGBoost, PySR, AFP-FE

**Requirements**:
- Automated report generation (Markdown + plots)
- CI-friendly: runs in < 10 min on CPU
- Tracks metrics over time (regression detection)

**Acceptance**: Report shows SOTA neural > sklearn MLP on all metrics; no regressions.

### Task 7: Integration Bridge (`integration.py`) — 2 days
**Deliverable**:
- `NeuralDiscoveryEngine` — Drop-in replacement for `neural_relevance`/`distill_via_surrogate`:
  ```python
  engine = NeuralDiscoveryEngine(config=NeuralDiscoveryConfig())
  relevance = engine.neural_relevance(x_data, y_data)  # same API
  distilled = engine.distill_via_surrogate(x_data, y_data)  # same API
  symbolic = engine.discover_symbolic(x_data, y_data)  # NEW: end-to-end SR
  ```
- `ModelDiscoveryIntegration` — Enhances `ModelDiscoveryEngine.discover()`:
  - Neural surrogate for holdout scoring
  - Uncertainty-aware candidate ranking
  - Active learning suggestions for next experiment
- Backward compatibility: existing tests pass unchanged

**Requirements**:
- All existing tests in `test_selection_sampler_gates.py` pass
- New API documented with examples
- No breaking changes to `discovery.py` public API

**Acceptance**: `pytest tests/test_selection_sampler_gates.py -v` 100% pass; new tests 100% pass.

## Dependencies to Add (pyproject.toml)
```toml
[project.optional-dependencies]
neural = [
    "flax>=0.8",
    "optax>=0.2",
    "equinox>=0.11",      # PyTree-based models
    "blackjax>=1.0",      # MCMC/SVI
    "jaxopt>=0.8",        # Optimization
    "optax>=0.2",         # Optimizers
    "optax-contrib>=0.1", # Extra schedulers
    "opendp>=0.1",        # Differential privacy (optional)
    "optuna>=3.5",        # Hyperopt
]
```

## Integration Points (Existing Code)
| File | Function | Integration |
|------|----------|-------------|
| `discovery.py` | `neural_relevance` | Replace impl, keep signature |
| `discovery.py` | `distill_via_surrogate` | Replace impl, keep signature |
| `model_discovery.py` | `ModelDiscoveryEngine.discover` | Add `neural_surrogate` option |
| `experiment_designer.py` | `design_experiments` | Accept neural acquisition |
| `closed_loop.py` | `ClosedLoopRunner` | Neural proposer |
| `jax_pde_solver.py` | PDE solvers | PINN loss in `PhysicsInformedNN` |
| `priors.py` | SEMF priors | Bayesian NN priors |

## Risks & Mitigations
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| JAX/Flax version conflicts | Medium | High | Pin versions, test matrix |
| SVI convergence on small data | High | Medium | Fallback to ensembles + conformal |
| Symbolic exec speed | Medium | Medium | JIT cache, operator fusion |
| API breakage | Low | High | Strict backward compat, deprecation warnings |

## Timeline (18 working days = ~3.5 weeks)
| Week | Tasks | Milestone |
|------|-------|-----------|
| 1 | 1, 2 | Architectures + Uncertainty working |
| 2 | 3, 4 | Differentiable SR + Active Learning |
| 3 | 5, 6 | Training/CV + Validation suite |
| 4 | 7 | Integration + Full test pass |

## Verification Gates
1. **Unit**: Each module has ≥ 90% coverage, `pytest -x` passes
2. **Integration**: `test_selection_sampler_gates.py` + new `test_neural_discovery.py` pass
3. **Benchmark**: SRBench ≥ 80% recovery, ECE < 0.05, active learning 40% budget reduction
4. **Regression**: No existing test failures, no API breaks
5. **Documentation**: API docs + examples in `docs/neural_discovery.md`

## Definition of Done
- [x] All 7 task surfaces implemented and covered by module tests
- [x] Neural optional dependencies are declared in `pyproject.toml`
- [x] Neural module suites pass: integration, active learning, uncertainty,
  training, and validation tests
- [ ] `pytest tests/test_neural_discovery.py -v` full top-level suite remains a
  long-running environment check
- [x] `pytest tests/test_selection_sampler_gates.py -v` remains the regression gate
- [ ] Benchmark thresholds (SRBench recovery, ECE, OOD AUROC) still require a
  reproducible benchmark run and must not be inferred from smoke tests
- [x] Public neural APIs are documented in the phase plan and module docstrings
- [ ] CHANGELOG.md updated

Implementation note: Bayesian variational optimization, trained surrogate
integration, data-derived symbolic initialization, active-learning training,
MC-dropout acquisition, k-fold CV, hyperparameter search, and OOD/calibration
components are executable and their module suites pass. The benchmark
thresholds above remain empirical claims pending a bounded benchmark report.
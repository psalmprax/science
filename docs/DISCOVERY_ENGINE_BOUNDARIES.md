# Discovery Engine Boundaries

APEDP contains several discovery surfaces. They are complementary, not interchangeable.

| Surface | Primary role | Use when |
|---|---|---|
| `MultiVariableSRE` | Physics-aware evolutionary symbolic regression with dimensional filtering, guided seeds, parameter fitting, and Pareto tracking | A bounded symbolic search over registered variables is required |
| `SymbolicRegressor` | Legacy tournament/crossover symbolic regression | Backward compatibility with the older API is required |
| `sparse_select_terms` | Deterministic feature-library prefilter using BIC or Lasso | A fast sparse initialization is needed before symbolic search |
| `NeuralDiscoveryEngine` | Neural surrogate, uncertainty, relevance, and surrogate distillation | The data is high-dimensional or a neural proposal/relevance stage is useful |
| `HypothesisComposer` | Data-free theoretical candidate composition | Generating speculative priors, not validating laws |
| `hybrid_discover` | Orchestrates probes, priors, bounded GP, and assurance reporting | A complete bounded discovery workflow is needed |
| `run_adaptive_discovery` | Stateful controller-managed `MultiVariableSRE` search | Search policy should react to telemetry during evolution |

## Authority rules

1. `MultiVariableSRE` is the canonical data-driven symbolic search engine.
2. `SymbolicRegressor` remains legacy and should not receive new capabilities unless a compatibility bug requires it.
3. Neural outputs are proposals, relevance signals, or surrogate approximations. They do not bypass dimensional, domain, holdout, or falsification gates.
4. `HypothesisComposer` outputs theoretical specifications. Its score is not evidence of truth.
5. `run_adaptive_discovery` changes search policy only. It cannot override scientific refusal gates.
6. `JevCompatibleBackend` is optional. Local fallback decisions are recorded explicitly, and no network service is required.

## Evidence status

A module test demonstrates executable behavior. It does not establish the benchmark claims in the neural discovery plan. Claims about SRBench recovery, calibration, OOD AUROC, label savings, or physical truth require a measured report produced from independent datasets.

# APEDP-X — Autonomous Physics, Engineering & Market Discovery Platform

A scientific discovery and epistemic referee platform combining formal mathematical verification (Lean 4), multi-proposer candidate generation, dimensional analysis, rigorous multiple-testing control (Deflated Sharpe Ratio), anti-snooping holdout guards, and scoped non-fungible promotion state machines.

---

## What APEDP-X Does

- **Epistemic Referee Discipline:** Enforces a strict separation between hypothesis generation and adjudication. Hypotheses cannot judge their own validity or define their own promotion criteria.
- **Formal Mathematical Verification:** Native zero-disk Lean 4 compiler verification (`--stdin` streaming) proving algebraic identities, dimensional consistency, and mathematical bounds.
- **Competing Multi-Proposer Architecture:** Independent proposers (e.g. `SparseMicrostructureProposer`, `OrderBookResiliencyProposer`, genetic AST regression) submit standardized `CandidateProposal` objects conforming to explicit strategy archetypes (`TAIL_RISK` $\ge 30$ events, `CONTINUOUS_FACTOR` $\ge 85\%$, and `TACTICAL` active fraction strictly in $[5\%, 90\%]$).
- **Deflated Sharpe Ratio (DSR) Trial Accounting:** Implements Bailey & López de Prado (2014) closed-form DSR, tracking cumulative trial counts, variance, skewness, and kurtosis to penalize data dredging and search multiplicity.
- **Full-Lineage Experiment Ledger & Anti-Snooping Guard:** Every candidate trial logs its feature set, hyperparameters, seeds, and dataset slices. Any confirmation holdout slice failed by a candidate is permanently burned; iterative replacements attempting to reuse it trigger `HoldoutContaminationError`.
- **Six-Stage Scoped Promotion:** Enforces non-fungible sequential verification:
  1. `MATHEMATICAL_CONSISTENCY` (Lean 4 + dimensional invariants)
  2. `PREDICTIVE_RECOVERY` (Cross-validated out-of-sample error reduction)
  3. `UNIQUE_IDENTIFICATION` (Diebold-Mariano test $p < 0.01$ against Kyle/Amihud/Biais/Carhart baselines; literature $R^2 < 0.85$)
  4. `OUT_OF_SAMPLE_TRANSFER` (Un-snooped temporal regimes and cross-asset transfer)
  5. `ECONOMIC_VALIDITY` (Net Sharpe after fees, breakeven fee hurdles, $DSR \ge 0.95$)
  6. `REAL_MARKET_PROMOTION` (Live sandbox/prospective execution without degradation)

---

## What APEDP-X Does NOT Establish

To maintain scientific integrity, the platform explicitly acknowledges its empirical and theoretical boundaries:
1. **No Proof of Live Trading Profitability:** While development-period hypotheses (e.g. H-LIK, H-XSC) exhibited strong in-sample statistical recovery, they failed or degraded during prospective real-market confirmation. The platform's gates are designed to catch and reject such false discoveries, not to guarantee alpha.
2. **No Conservative Hamiltonian Physics in Economics:** Financial order flow does not conserve mechanical energy or obey Hamiltonian mechanics. Financial signals are modeled under dissipative, regime-switching microstructure dynamics, not conservative physical laws.
3. **No Domain-Agnostic Equation Generation Without Adapters:** While the epistemic evidence and holdout-gating core is domain-general, actual hypothesis discovery requires domain-specific adapters, units, operational constraints, and realistic baseline registries.

---

## System Architecture

```
                    ┌───────────────────────────────────────────────┐
                    │            Competing Proposer Pool            │
                    │  • Sparse Microstructure (ElasticNet)         │
                    │  • Resiliency (Biais/Large Exponential Decay) │
                    │  • Genetic AST Symbolic Regression            │
                    └───────────────────────┬───────────────────────┘
                                            │ Emits CandidateProposal
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │      FullLineageExperimentLedger (v1)         │
                    │  • Tracks features, params, seeds, slices     │
                    │  • Cumulative trial accounting for DSR        │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                          Six-Stage Promotion Lifecycle                            │
│                                                                                   │
│  [Stage 1: Lean 4 Kernel] ──► [Stage 2: OOS Recovery] ──► [Stage 3: Novelty DM]   │
│            │                                                    │                 │
│            ▼                                                    ▼                 │
│  [Stage 6: Real Market]   ◄── [Stage 5: Economic DSR] ◄── [Stage 4: OOS Transfer]│
│                                                             (Anti-Snooping Guard) │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Running the Automated Test Suite

Run all dynamic tests, canonical ledger verifications, and epistemic benchmarks:
```bash
pytest tests/test_dyn_competing_proposers_and_dsr.py \
       tests/test_audit_open_flaws_remediated.py \
       tests/test_dyn_blind.py \
       tests/test_dyn_canonical_ledger.py \
       tests/test_dyn_native_ga_best.py \
       tests/test_dyn_experiment_ledger_and_benchmarks.py -v
```

### 2. Running the Three-Tier Benchmark Suite

Execute the calibrated benchmark evaluating referee discipline (FPR/CAR/VRR with Wilson CIs), discovery power across effect sizes ($IC \in [0.02, 0.05, 0.15]$), and out-of-distribution transfer:
```python
from apedp_dyn.sandbox.calibrated_benchmark_suite import (
    run_calibrated_referee_benchmark,
    run_calibrated_discovery_benchmark,
    run_transfer_benchmark,
)

# Tier 1: Referee discipline
ref_report = run_calibrated_referee_benchmark(my_referee_fn, n_trials=30)
print(f"FPR: {ref_report.fpr:.2%} (CI: {ref_report.fpr_ci}), CAR: {ref_report.car:.2%}")

# Tier 2: Discovery statistical power
disc_report = run_calibrated_discovery_benchmark(my_proposer_fn, trials_per_condition=15)
print(f"Power at IC=0.15: {disc_report.strong_signal_power:.2%}")
```

### 3. Verifying In-Memory Lean 4 Proofs via Streaming `--stdin`

Run formal verification directly through Lean 4's kernel without disk overhead:
```python
from apedp_dyn.epistemic.lean_universal_bridge import verify_lean4_code_stdin

code = "theorem tautology (p : Prop) (h : p) : p := h"
result = verify_lean4_code_stdin(code)
print(f"Verified: {result.verified}, Kernel exit code: {result.exit_code}")
```

### 4. Inspecting Canonical Preregistered Settings

The authoritative source of truth for preregistered thresholds is frozen at [`specifications/canonical_preregistered_spec_v1.json`](specifications/canonical_preregistered_spec_v1.json) and guarded by SHA-256 verification in [`apedp_dyn/conf/canonical_spec.py`](apedp_dyn/conf/canonical_spec.py).

---

## Key Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md): Component boundaries, evidence flow, and holdout firewalls.
- [`CHANGELOG.md`](CHANGELOG.md): User-visible improvements, bug fixes, and audit remediations.
- [`AGENTS.md`](AGENTS.md): Repository guidelines and integrity rules for AI coding assistants.
- [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md): Complete module-by-module reference and status index.
- [`docs/apedp_discovery_engine_ledger_and_benchmarks_audit.md`](docs/apedp_discovery_engine_ledger_and_benchmarks_audit.md): Engineering audit mapping guarantees to tests.

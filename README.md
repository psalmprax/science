# APEDP-X — Autonomous Physics, Engineering & Market Discovery Platform

[![APEDP CI/CD](https://github.com/psalmprax/science/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/psalmprax/science/actions/workflows/test.yml)

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

## Getting Started in 5 minutes

Prerequisites: Git, Python 3.12 with `venv` and pip, and network access for
dependencies. Python 3.10–3.12 is tested in CI. From the repository root, use
this lightweight setup and verification path (POSIX shell):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest tests/test_smoke_imports.py::test_every_source_file_parses tests/test_ci_quality_contract.py -q
```

On Windows, activate with `.venv\Scripts\activate` instead. This verifies source
syntax and CI configuration contracts without Rust, Redis, GPU, or Lean setup;
it does **not** certify native or scientific behavior. First-time dependency
downloads may take longer than five minutes. Do not install `requirements.lock`
(a historical workstation snapshot).

For the full test environment, pinned Rust toolchain, native build/import check,
and Redis integration setup, follow [Contributing: local setup](CONTRIBUTING.md#local-setup).
The existing [Makefile](Makefile) provides `make lint`, `make typecheck`,
`make lint-rust`, `make kernel`, and `make test` once their prerequisites are installed.

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

## Required checks and CI status matrix

Expected checks, not a live pass/fail report. Commands run from the repository
root unless noted; see [CONTRIBUTING.md](CONTRIBUTING.md#checks-matching-ci) for
environment setup and scope limitations.

| Check | Command / scope | Actual workflow and job |
| --- | --- | --- |
| Python lint | `make lint`: Ruff `E9,F` on `apedp/`, `apedp_dyn/`, `tests/`; configured rules on `apedp_dyn/conf/` | [APEDP CI/CD](.github/workflows/test.yml), `python-quality` (Python 3.12) |
| Type checks | `make typecheck`: strict mypy on `apedp_dyn/conf/paths.py` and `canonical_spec.py`, not the entire repository | [APEDP CI/CD](.github/workflows/test.yml), `python-quality` |
| Rust quality/unit tests | `make lint-rust`; `cargo test --workspace --locked` | [APEDP CI/CD](.github/workflows/test.yml), `rust-quality` |
| Native build and smoke | `maturin develop --locked --manifest-path rust/apedp-kernel/Cargo.toml`, bridge availability assertion, `python -m pytest tests/test_smoke_imports.py -q -x`, then test collection | [APEDP CI/CD](.github/workflows/test.yml), `test` (Python 3.10/3.11/3.12) |
| Scientific evidence | `python benchmarks/check_scientific_drift.py`; `python -m pytest tests/test_scientific_recovery.py tests/test_dyn_rime_known_truth.py -q -m scientific_regression` | [APEDP CI/CD](.github/workflows/test.yml), `test` |
| Regression/integration | `python -m pytest tests/ -v --tb=short -n auto -x -m "not integration and not scientific_regression"`; separate `-m integration` run with Redis | [APEDP CI/CD](.github/workflows/test.yml), `test` |
| Security | `bandit -r apedp/ -ll`; CodeQL over detected repository languages | [APEDP CI/CD](.github/workflows/test.yml), `test`; [CodeQL](.github/workflows/codeql.yml), `Analyze` |
| Distribution builds | Locked maturin builds for Linux/macOS/Windows; `python -m build` from each `packages/<package>/` | [Cross-Platform Native Wheels](.github/workflows/wheels.yml), `linux-wheels`, `macos-wheels`, `windows-wheels`, `sdist`, `python-packages` |

For pull requests, configure branch protection to require `Python quality`,
`Rust quality`, every Python-version leg of `test`, and `mutation-battery` from
the **APEDP CI/CD** workflow, plus `Analyze` from **CodeQL** after its first
successful run. The `test` matrix depends on both quality jobs.
`build` and `deploy` are main-branch jobs, not pull-request checks. The
**Cross-Platform Native Wheels** workflow has PR path filters (Rust, Cargo
manifests/lockfile, packages, or its own workflow), so it is not expected on a
documentation-only PR; it also runs on master/main pushes, version tags, or
manual dispatch. Workflow files do not configure branch protection: repository
administrators select required checks in GitHub settings.

### Coverage policy

There is no repository-wide numeric coverage target or coverage gate in CI today.
Branch coverage is enabled for ad hoc Python coverage reports, and
`tools/coverage_audit.py` is an advisory report rather than a pass/fail gate.
Keep or add focused tests for changed behavior; coverage improvements should be
reviewed alongside test quality rather than represented as an enforced
percentage.

### Flaky-test handling

CI does not automatically retry failing tests. Capture the workflow run URL,
failing step, and relevant environment details before rerunning a suspected
transient failure. Reproduce it locally when possible; file an issue with the
evidence if it recurs, and route it to the current review owner in
[CODEOWNERS](.github/CODEOWNERS). Do not treat a rerun as a fix or suppress a
deterministic regression.

## Repository layout and module ownership

See the [repository structure policy and cleanup guide](CONTRIBUTING.md#repository-structure-policy)
for canonical code, docs/specs, data/benchmarks, outputs, and experiment zones.
The [module ownership/responsibility map](CONTRIBUTING.md#module-ownership-and-responsibilities)
explains `packages/`, `crates/`, the active `rust/` kernel, and `apedp*` boundaries.
Reviews route to `@psalmprax` via [CODEOWNERS](.github/CODEOWNERS).
Existing imports, persisted evidence, and directory paths are unchanged.

## Key Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md): Component boundaries, evidence flow, and holdout firewalls.
- [`CHANGELOG.md`](CHANGELOG.md): Keep a Changelog history and unreleased changes.
- [`RELEASING.md`](RELEASING.md): Release checklist, SemVer, tagging, and rollback guidance.
- [`SECURITY.md`](SECURITY.md): Vulnerability reporting and supported-version policy.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): Setup, checks, repository structure policy, cleanup guide, and module ownership.
- [`docs/adr/README.md`](docs/adr/README.md): Architecture decision records and review process.
- [`AGENTS.md`](AGENTS.md): Repository guidelines and integrity rules for AI coding assistants.
- [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md): Complete module-by-module reference and status index.
- [`docs/apedp_discovery_engine_ledger_and_benchmarks_audit.md`](docs/apedp_discovery_engine_ledger_and_benchmarks_audit.md): Engineering audit mapping guarantees to tests.

## Releases and versioning

Published artifacts follow Semantic Versioning independently; the current
metadata versions are not a promise that all packages share a release number.
See [RELEASING.md](RELEASING.md) for preflight checks, changelog updates, and
tagging policy.

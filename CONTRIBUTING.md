# Contributing

Keep changes focused and preserve scientific test intent. Read [AGENTS.md](AGENTS.md)
before changing proposers, referees, holdouts, or frozen specifications.

## Repository structure policy

Keep reusable runtime code separate from research inputs and generated evidence.
Use these existing zones for new files; this policy does not rename directories.

| Zone | Canonical locations | What belongs here |
| --- | --- | --- |
| Code/runtime | `apedp/`, `apedp_dyn/`, `packages/`, `rust/`, `crates/` | Importable implementations and package manifests; see the module map below before choosing a native path. |
| Tests/tooling | `tests/`, `scripts/`, `tools/`, `.github/workflows/` | Regression contracts, maintained entrypoints, developer utilities, and automation. |
| Docs/specs | `docs/`, `specifications/`; root README, architecture and contributor guides | Contributor documentation and reviewed scientific contracts, not per-run reports. Update [the documentation index](docs/DOCUMENTATION.md) for new modules, proposers, or gates. |
| Data/benchmarks | `data/`, `benchmarks/` | Reviewed input fixtures, benchmark protocols, manifests, and intentionally retained baselines. Preserve dataset provenance and train/holdout boundaries. |
| Generated outputs | `outputs/<run-id>/` | New per-run reports, plots, logs, and result/provenance bundles. `outputs/` is ignored; so are existing `artifacts/`, `checkpoints/`, `logs/`, `build/`, `dist/`, and Cargo `target/` paths. |
| Experiments/scratch | `notebooks/`, `scratch/`, `apedp_dyn/experiments/` | Exploratory notebooks and disposable investigations; importable experimental harnesses stay with their package. These directories are not blanket-ignored: keep local results under `outputs/`. |

Do not add a new top-level result directory for each experiment. Promote a
reusable experiment into the appropriate runtime module only with tests and a
documented boundary; an exploratory result is not a production capability.
Use the [standard module status tags](AGENTS.md#3-module-status-labels) when
indexing modules.

### Cleanup and migration guide

- **Move now:** redirect new, unreferenced local output to `outputs/<run-id>/`;
  remove only reproducible, untracked build/cache files. Check references before
  deleting or relocating anything tracked.
- **Keep for compatibility:** existing `benchmark_corpus*`, `benchmark_n10/`,
  `*_results/`, `tournament_results*`, `provenance/`, `apedp_persistence/`,
  root reports, and research documents can be fixtures, persisted state, or
  published evidence. Their current paths remain valid; do not bulk-move them.
- **Migrate separately:** inventory readers, imports, scripts, links, checksums,
  and published references first. Propose one bounded move with updated readers,
  tests, and documented compatibility aliases where needed. Preserve original
  evidence and provenance; never reset ledgers or rename burned holdout slices.
- **Retain deliberately:** copy only reviewed, reproducibility-critical output
  into `data/` or `benchmarks/`, with its command, seed/configuration, dataset hash,
  and provenance sidecar. Do not force-add ignored output trees or model blobs.
  Frozen records and preregistered specifications still follow [AGENTS.md](AGENTS.md);
  this policy is not permission to regenerate or change them.

## Module ownership and responsibilities

[@psalmprax](https://github.com/psalmprax) is the current review owner for all
areas, as recorded in [.github/CODEOWNERS](.github/CODEOWNERS).

| Area | Responsibility and contributor boundary |
| --- | --- |
| `packages/` | Separately packaged domain APIs (continuum, quantum, robotics, molecular, audio, distributed) depend on the `apedp` core and expose domain functionality. Change their public API, dependencies, and packaging here; shared implementations belong in core. `jetflow-video` is standalone and must not acquire an `apedp` dependency. |
| `crates/` | Contains the separate `apedp_kernel` Rust source tree, **not** a member of the active root Cargo workspace. Do not assume changes here affect the installed Python kernel or migrate it implicitly. |
| `rust/apedp-kernel/` | Active PyO3 numerical kernel selected by root `Cargo.toml`, maturin, and CI. Native contributors must preserve Python fallback semantics and run conformance tests against a built extension. |
| `apedp/` | Core scientific APIs, ingestion, equation models, pipeline/API services, and Python/native bridge. Shared runtime behavior belongs here rather than in result scripts. |
| `apedp_dyn/` | Dynamic discovery, domain adapters, proposers, epistemic referees, and experiment orchestration. Proposers emit `CandidateProposal`; only referees adjudicate, and confirmation holdouts remain isolated. Import admission thresholds from `apedp_dyn.conf.canonical_spec`. |
| `apedp_kernel`, `apedp_persistence/` | The former is a development symlink to the installed native package, not a place to edit source; the latter stores persisted JSON state, not Python implementation. Preserve these compatibility paths. |

For deeper boundaries see [ARCHITECTURE.md](ARCHITECTURE.md); for module status
and references see [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

## Local setup

CI tests Python 3.10, 3.11, and 3.12; quality and packaging jobs use 3.12.
Use Python 3.12 for the commands below. Rust is pinned in `rust-toolchain.toml`
to 1.99.0, including rustfmt and Clippy; install Rust with rustup.

From the repository root (on Windows, activate `.venv\Scripts\activate`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install CPU Torch first so pip does not pull the CUDA runtime into CI.
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,test,orchestration]"
python -m pip install -e packages/jetflow-video
rustup show
maturin develop --locked --manifest-path rust/apedp-kernel/Cargo.toml
python -c "from apedp.rust_bridge import is_rust_kernel_available; assert is_rust_kernel_available()"
```

Do not install `requirements.lock`: it is a historical workstation snapshot
containing nonportable editable paths, not the CI dependency contract.
Development tools are pinned in `pyproject.toml`; Rust builds use `Cargo.lock`.

## Checks matching CI

With the virtual environment activated, run:

```bash
make lint
make typecheck
make lint-rust
cargo test --workspace --locked
python -m pytest tests/test_ci_quality_contract.py -q
python -m pytest tests/test_smoke_imports.py -q -x
python -m pytest tests/ --collect-only -q
python -m pytest tests/ -v --tb=short -n auto -x -m "not integration"
```

The `Python quality` and `Rust quality` jobs are blocking dependencies of all
three Python test jobs. Repository administrators should make these checks,
the test matrix, and wheel jobs required in branch protection; workflow
changes alone cannot configure GitHub merge rules.

Integration tests need Redis listening on port 6379. Start your local Redis
service, then run:

```bash
APEDP_REDIS_URL=redis://localhost:6379/0 \
  python -m pytest tests/ -v --tb=short -n auto -x -m integration
```

## Reproducibility and provenance

`NeuralDiscoveryConfig` and `TrainingConfig` default to `seed=42` and
`deterministic=True`. A trainer seeds Python, NumPy, and available Torch RNGs;
JAX uses explicit keys (`apedp.reproducibility.jax_key(seed)`). Deterministic
mode requests deterministic Torch algorithms and disables cuDNN benchmarking.
The standalone evolutionary `SymbolicRegressor` accepts a `seed` in
`RegressionConfig` and uses an isolated, per-run random generator.
It does not promise bitwise identity across accelerator models, drivers, or
framework versions; retain the device/runtime versions with reported results.

When `NeuralDiscoveryConfig.log_dir` is set, each training call writes a JSON
record at `log_dir/provenance/training-<dataset-sha256>.json`. Other discovery
entrypoints can use `write_provenance_artifact(path, dataset=..., config=...,
seed=..., deterministic=..., metrics=..., results=...)`. Schema `1.0` includes
the best-effort Git commit, dataset SHA-256, complete config snapshot and hash,
seed/determinism flags, key metrics, and optional results. Records are replaced
atomically; treat the referenced dataset and result artifacts as part of the
same run record.
Both known-truth challenge entrypoints write a `<results-file>.provenance.json`
sidecar for each run, including the generated dataset hash and recovery metrics.

## Benchmark drift workflow and CI failure categories

`benchmarks/scientific_benchmark_manifest.json` pins the benchmark checksum
inventory and baseline recovery metrics. CI runs
`python benchmarks/check_scientific_drift.py` and a separate
`scientific_regression` known-truth test gate. A baseline change is intentional
only after reviewing the protocol and rerunning the full challenge; update the
baseline and maximum drift together in a reviewed change. Do not loosen a
threshold to hide an unexplained regression.

The CI step named **Scientific regression gate** (and its explicit error
annotation and job summary) indicates a failed recovery criterion. The job
summary separately labels general-suite regression, integration/runtime
failures, and infrastructure/tooling failures (setup, build, quality, or
security); use the named step and its logs for the specific cause.

Run the complete dynamic regression battery listed in [AGENTS.md](AGENTS.md)
before submitting scientific changes. Native conformance tests must run
against the built kernel, not silently skip.

### Phase 1 baseline and follow-ups

- Ruff checks `E9,F` (syntax, undefined names, unused imports/assignments)
  across `apedp/`, `apedp_dyn/`, and `tests/`, retaining existing frozen-file
  exemptions. The full configured rule set gates `apedp_dyn/conf/`.
- Mypy is **strict** for the repository path resolver and canonical spec
  loader. Imported legacy packages are opaque (`follow_imports = "skip"`).
  Expanding strictness across the full packages needs a separate cleanup:
  the initial baseline found over 600 errors with current dependency stubs.
- `cargo fmt --all --check` gates the entire active workspace.
- Clippy gates every workspace target with compiler warnings denied and
  `clippy::correctness` enabled. Other Clippy categories are deferred rather
  than refactoring numerical loops, PyO3 signatures, and recorded precision
  fixtures in this pass. Two intentional rounded search constants have
  local `approx_constant` allowances to preserve search behavior.
- Follow up by widening Ruff and mypy coverage package by package, enabling
  further Clippy categories, and validating wheel matrices on hosted runners.
  This pass does not certify an engineering score or erase legacy debt.

## CI troubleshooting

- **No jobs started:** inspect workflow validation diagnostics. `secrets`
  cannot appear in a job-level `if`; use a job environment and a step condition.
- **Missing pytest or wrong interpreter:** activate `.venv`, reinstall the
  extras above, and use `python -m pytest`. CI checks tool versions during setup.
- **Missing Torch/JAX/Flax:** use the CPU Torch install and test extras
  above. Some test modules import these runtimes during collection. The broad
  experimental `neural` extra still references unavailable `optax-contrib`;
  fixing that optional dependency surface is deferred.
- **Missing tqdm/imageio:** install `packages/jetflow-video` as above; its
  manifest owns the standalone video package's runtime dependencies.
- **Kernel unavailable or native tests skipped:** run `maturin develop` in
  the activated environment, then repeat the import assertion. Installing
  the Python project alone does not build the Rust extension.
- **Formatting/Clippy failure:** run `cargo fmt --all`, then `make lint-rust`
  with the pinned toolchain. Do not blanket-allow compiler warnings.
- **Redis refused connection:** check the service health and
  `APEDP_REDIS_URL`; do not mark integration tests as passing without a service.
- **Wheel failure:** inspect the first maturin/compiler error, target, and
  interpreter in the job log. Builds pin Rust/maturin and use `--locked`.
- **Frozen-file hash failure:** do not autofix or regenerate the pinned
  scientific records to make lint pass; follow the integrity protocol.

CI exits on the first test failure with short tracebacks, gates quality before
the expensive regression matrix, and has bounded job timeouts. Reproduce the
failing command locally before retrying a run.

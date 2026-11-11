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
and references see [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md). Record
significant architecture or process choices in [docs/adr/README.md](docs/adr/README.md).
Release changes use [RELEASING.md](RELEASING.md) and the
[changelog](CHANGELOG.md); versions follow SemVer per published artifact.

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
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
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
python -m pytest tests/test_expression_evaluator.py --cov --cov-branch --cov-report= -q
python -m coverage report --include='*/apedp/expression_evaluator.py' --fail-under=95
python -m pytest tests/ --collect-only -q
python -m pytest tests/ -v --tb=short -n auto -x -m "not integration"
```

The `python-quality` CI job installs its tools from the hash-pinned
`requirements-quality.lock`. Regenerate that lock with
`uv pip compile requirements-quality.in --universal --generate-hashes --output-file requirements-quality.lock`.
The Python test job installs the development tools through the `dev` extra.

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
- Mypy is **strict** for the repository path resolver, canonical spec loader,
  D5-CONF noise-sweep benchmark, extracted FAROS runtime boundary, and native
  build-integrity registry. Imported legacy packages are opaque
  (`follow_imports = "skip"`).
- Bandit checks medium and higher findings in `apedp/` and high findings in
  `apedp_dyn/`; the latter scope is narrower until its intentional lower-severity
  findings are individually assessed.
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

### FAROS native tests

`python -m pytest tests/test_faros_isq.py tests/test_faros_runtime.py -q` includes
numerical tests and native integration tests. Native tests require the approved
KickMix extension at
`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/kickmix.so`.
An arbitrary rebuilt extension is not sufficient: `load_contract` verifies its
hash against the approved build registry.

Restore the pinned upstream sources with `git submodule update --init`.
Create the pinned native environment and check it explicitly:

```bash
uv venv --python 3.13.14 build/qvcc-pinned313
uv pip install --python build/qvcc-pinned313/bin/python --require-hashes -r requirements-native.lock
python scripts/check_verifier_environment.py --python-executable build/qvcc-pinned313/bin/python
```

Preflight refuses missing or unapproved binaries before loading them. The
source build is toolchain-sensitive and is not guaranteed to match the approved
hash on another machine. Installing the lock does not supply the approved
binary or historical receipt. FAROS honors an explicit `python_executable` first,
then `APEDP_Q_PYTHON`, then repository-local `build/qvcc-pinned313` and
`build/qvcc-venv313` environments. Invalid explicit overrides fail rather than
being silently replaced. Resource measurement failures also fail rather than
producing hard-coded resource counts.

The historical receipt test additionally requires
`artifacts/quantum_compilation/invsqrt/receipt_faros_r2_86q.json` with its original
circuit digest. A new run is separate evidence, not a replacement for that
missing historical artifact. Do not fabricate the receipt or change the binary
pin simply to make these tests pass.

The manually dispatched `Approved native research checks` workflow accepts an
artifact run ID and name from this repository, verifies the downloaded binary's
approved hash and interpreter, and runs native phase regression checks. It is
not a full-domain correctness certification. No approved binary artifact is
currently provided by these changes.

The blocking `Research model regressions` job installs
`requirements-research.lock` and runs numerical normalization, rounded Horner,
and parity regressions without native dependencies. Its explicit exclusion of
the native parity check is paired with the separate approved-native workflow,
not presented as evidence that native tests passed. Regenerate the numerical
and native locks with `uv pip compile requirements-research.in --python-version 3.12 --generate-hashes --output-file requirements-research.lock`
and `uv pip compile requirements-native.in --python-version 3.13 --generate-hashes --output-file requirements-native.lock`.

The FAROS builder now runs a fresh verifier when `--verify` is supplied and
propagates failure; without it, the pipeline status is `built_not_verified`.
It does not reuse historical receipts or subtract a fixed Toffoli discount.
Resource statistics are measured from the final circuit, while synthesis cost
and the historical margin remain explicitly labeled estimates/assumptions.
Calibration replays fixed deltas and reports `lp_converged: null` and
`margin_verified: false`; it does not execute or claim convergence of an LP.
`tests/test_faros_evidence_pipeline.py` checks these reporting contracts.

`scripts/prototype_streamed_invsqrt.py` explores a separate 50-qubit-capped,
bit-sliced phase lookup. Run it with the KickMix-compatible interpreter and its
extension directory on `PYTHONPATH`, passing `--output` and optionally
`--chunk-bits`. It preserves the 22-bit input, computes a phase fragment into
reusable workspace, applies its rotations, and reversibly clears the fragment.
It is explicitly a prefix prototype: outside the declared prefix, the lookup
repeats periodically and is **not** a correct full-domain implementation.

An experimental build with 40-bit phase precision, inputs 1–4095, and 17-bit
fragments generated 50 qubits, 98,256 static Toffolis, and 37 non-Clifford
rotations. The verifier passed 1,048,320 prefix shots with score 4,923,955.31
using the unapproved local KickMix rebuild. Four-bit fragments generated 37
qubits but 327,520 Toffolis. Both are substantially worse in total non-Clifford
cost than the 90-qubit FAROS circuit: fewer rotations do not compensate for
the repeated lookup work. These results demonstrate the workspace/cost
tradeoff, not a successful 50-qubit, lower-T benchmark solution.

`scripts/prototype_direct_phase.py` is an independent, shared-predicate phase
oracle without FAROS coefficient loading or polynomial arithmetic. Its
reversible input-predicate tree applies rounded phase rotations directly,
sharing the tree traversal across contributions. It uses all 22 input bits;
inputs beyond its declared prefix receive zero phase rather than aliasing.
Run `scripts/check_direct_phase.py` with the same native interpreter and
`PYTHONPATH` to check exact fixed-point phases, input restoration, clean
scratch, high-bit/out-of-prefix behavior, and invalid inputs.

For inputs 1–4096, this direct prototype generated 43 qubits, 16,418 static
Toffolis, and 4,092 non-Clifford rotations. Experimental prefix verification
passed 262,144 shots with score 1,933,034.64. This improves on the streamed
lookup's cost but is still much worse than FAROS's measured prefix score
157,751.58; it does not meet the lower-total-T goal or implement the complete
benchmark domain. Both measurements use the unapproved local KickMix rebuild.

`scripts/prototype_parity_phase.py` tests a third approach: retain the
largest-magnitude Walsh coefficients of a smooth lift of `1/sqrt(x)`, then
compute each input parity with CNOTs, rotate, and undo the parity. A single
clean ancilla supplies the state-independent phase offset, which cannot be
discarded because the verifier checks absolute phase. Run
`scripts/check_parity_phase.py` to check the transform against explicit
parities, reconstruction, sparse failure reporting, and exact native phases.

The script audits every one of the 3,133,452 benchmark inputs for each requested
term budget before generating the selected experimental circuit. Budgets of
64, 256, 1024, and 4096 terms all failed the numerical tolerance audit. The
256-term circuit used 23 qubits and zero Toffolis, but failed prefix verification
at x=1 (phase error 0.01560 turns versus tolerance 5.960e-8).
These are approximation failures, not simulator failures: native regression
checks confirm the circuit implements its selected parity model. Largest-term
truncation optimizes neither relative worst-case error nor rotation-synthesis
cost; this experiment is not evidence that every possible parity decomposition
is unsuitable. The audit uses floating-point arithmetic, excludes rotation
synthesis error, and is not a formal correctness certificate.

`scripts/audit_normalized_invsqrt.py` investigates approximation feasibility
before attempting another circuit. It audits piecewise Chebyshev approximations
on normalized mantissas `[1,2)` at every benchmark input, including a separate
model that rounds coefficients, mantissas, and Clenshaw recurrence operations.
Run `scripts/check_normalized_invsqrt.py` for boundary and rounding checks.

Four segments with degree five and 28 fractional bits passed the full-domain
numerical audit with worst error/tolerance 0.392764. One segment with degree
nine and 28 fractional bits also passed (0.379256). All four tested 24-bit
rounded candidates failed. These results exclude reversible normalization,
overflow, reversible rounding/garbage cleanup, and rotation synthesis; exponent
scaling still uses floating-point arithmetic followed by 48-bit phase rounding.
They are numerical evidence, not a proof or a measured quantum implementation.
Straightforward Clenshaw evaluation holds two fractional states: at 28
fractional bits, those states alone exceed the 28 workspace qubits remaining
after the 22-bit input in a 50-qubit circuit. A different reversible evaluation
schedule or representation is still needed, with its gate cost measured.

`scripts/audit_horner_invsqrt.py` tests a single classical accumulator with
integer, ties-to-even Horner arithmetic. Exponent parity is folded into two
coefficient rows; mantissa normalization and 48-bit phase scaling use exact
integer shifts in the model. Coefficient fitting and comparison with the target
still use float64. Run `scripts/check_horner_invsqrt.py` for independent
`Fraction`-based rounding and recurrence checks.

All four tested 27-fractional-bit candidates passed the numerical comparison
over the full domain. Two segments of degree seven had worst error/tolerance
0.645662; the accumulator's observed signed width was 28 bits. That already
uses 50 bits together with the input, **before** reversible normalization,
multiply scratch, coefficient selection, or cleanup. All tested 25- and
26-fractional-bit candidates failed accuracy. Crucially, rounded Horner updates
are not injective: at multiplier 1/2, old states 0 and 1 both round to 0.
A quantum circuit cannot simply overwrite the old accumulator as this classical
model does. Discarded information must be retained, reversibly recomputed, or
otherwise handled by a justified circuit construction. No 50-qubit reversible
implementation or T-cost improvement follows from these arithmetic results.

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

## Support and triage

Issue and pull-request triage is best effort; no response-time SLA is promised.
Include reproducible steps, relevant workflow links, and environment details
when reporting a bug. The current review owner is listed in
[CODEOWNERS](.github/CODEOWNERS). Report security vulnerabilities privately
using the process in [SECURITY.md](SECURITY.md), not in a public issue.

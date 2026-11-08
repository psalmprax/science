.PHONY: install test lint typecheck lint-rust format clean clean-cache kernel vqc-native vqc-parity wheel reproduce

# Interpreter used by gates that need one. Override for a venv:
#   make lint PYTHON=.venv/bin/python
PYTHON ?= python

install:
	pip install -e .

# Build the Rust kernel extension into the active environment.
# Requires: Rust toolchain + `pip install maturin`.
kernel:
	maturin develop --release --manifest-path rust/apedp-kernel/Cargo.toml

# Build the Rust VQC execution core (pauli algebra, stabilizer tableau, Clifford
# images) into the active environment. Optional: `apedp_dyn/vqc/native.py` falls
# back to the Python tier when the module is absent, so this is a speed path, not
# a dependency.
vqc-native:
	maturin develop --release --manifest-path rust/apedp-vqc/Cargo.toml

# Re-measure the native core against the Python tier on identical frozen inputs
# and rewrite benchmarks/quantum_compilation/vqc_native_parity.json.
vqc-parity: vqc-native
	python scripts/measure_vqc_rust_parity.py

# Build a distributable wheel (rust/apedp-kernel via maturin) into dist/.
wheel:
	maturin build --release --manifest-path rust/apedp-kernel/Cargo.toml -o dist/

# One command that regenerates and verifies every headline number a reader
# of the docs would quote: frontier claims integrity, Planck/Riccati claims,
# canonical ledger, the frozen external round, trig recovery, PDE transfer.
reproduce: kernel
	pytest tests/test_dyn_frontier_claims_integrity.py \
	       tests/test_dyn_planck_limitation.py \
	       tests/test_dyn_canonical_ledger.py \
	       tests/test_external_benchmarks.py \
	       tests/test_dyn_native_ga_trig_templates.py \
	       tests/test_dyn_pde_transfer.py -q

test:
	pytest tests/ -v

# The full configured ruleset (pyproject [tool.ruff.lint]) on the two trees that
# are held clean, then the pinned baseline for the trees that are not yet. The
# old gate ran --select E9,F, which let the configured rule set accumulate
# findings while reporting success.
lint:
	ruff check apedp/ tests/
	$(PYTHON) scripts/check_quality_baseline.py --ruff-only

# Strict mypy on the conf/ integrity boundary (pyproject [tool.mypy] `files`),
# then the pinned full-package baseline. Bare `mypy` alone checks two modules
# and says nothing about the other ~1,200 files.
typecheck:
	mypy
	$(PYTHON) scripts/check_quality_baseline.py --mypy-only

lint-rust:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --locked -- -A clippy::all -W clippy::correctness -D warnings

format:
	ruff format apedp/ tests/

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/

clean-cache:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

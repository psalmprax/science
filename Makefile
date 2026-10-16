.PHONY: install test lint format clean clean-cache kernel wheel reproduce

install:
	pip install -e .

# Build the Rust kernel extension into the active environment.
# Requires: Rust toolchain + `pip install maturin`.
kernel:
	maturin develop --release --manifest-path rust/apedp-kernel/Cargo.toml

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

lint:
	ruff check apedp/ tests/

format:
	ruff format apedp/ tests/

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/

clean-cache:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

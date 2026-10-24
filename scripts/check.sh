#!/usr/bin/env bash
# Full local verification gate: everything CI runs that is cheap enough to run
# here, plus the complete dynamic regression battery. Run before pushing.
#
#   bash scripts/check.sh            # full battery
#   bash scripts/check.sh --quick    # fast tier only (ruff + smoke + guards)
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

echo "== ruff (changed files: blocking) =="
CHANGED=$(git diff --name-only --diff-filter=ACMRT HEAD | rg '\.py$' || true)
if [[ -n "$CHANGED" ]]; then
  .venv/bin/ruff check $CHANGED
else
  echo "no changed python files vs HEAD"
fi

echo "== ruff (whole package: informational, pre-existing debt) =="
.venv/bin/ruff check apedp_dyn/ tests/ --output-format concise || echo "(pre-existing findings above are not blocking)"

echo "== frozen referee integrity =="
.venv/bin/python -c "from apedp_dyn.sandbox.blind_benchmark import verify_referee; verify_referee(); print('referee integrity OK')"

echo "== import smoke tier =="
.venv/bin/python -m pytest tests/test_smoke_imports.py -q

if [[ $QUICK -eq 1 ]]; then
  echo "== quick tier complete =="
  exit 0
fi

echo "== dynamic regression battery =="
.venv/bin/python -m pytest \
  tests/test_dyn_competing_proposers_and_dsr.py \
  tests/test_audit_open_flaws_remediated.py \
  tests/test_dyn_blind.py \
  tests/test_dyn_blind_protocol.py \
  tests/test_dyn_canonical_ledger.py \
  tests/test_dyn_native_ga_best.py \
  tests/test_dyn_experiment_ledger_and_benchmarks.py \
  tests/test_documentation_and_spec_integrity.py \
  tests/test_dyn_expanded_proposers_and_calibrated_power.py \
  tests/test_dyn_dmrg_quantum.py \
  tests/test_dyn_lob_transport.py \
  tests/test_dyn_audio_field_transport.py \
  tests/test_dyn_molecular_se3_flow.py \
  tests/test_dyn_robotics_trajectory.py \
  tests/test_dyn_number_theory_and_invariant_transfer.py \
  tests/test_dyn_trading_sandboxes_epistemic.py \
  -q

echo "== targeted mutation battery (REG-MUT gate: survived mutant fails the run) =="
.venv/bin/python scripts/run_mutation_battery.py || {
  echo "MUTATION BATTERY FAILED: a pinned referee/leakage mutant SURVIVED its pins (see benchmarks/mutation_battery_report.json)"
  exit 1
}

echo "== rust kernel battery =="
if command -v cargo >/dev/null 2>&1; then
  (cd rust/apedp-kernel && cargo test --lib -q)
else
  echo "cargo not installed - Rust kernel tests SKIPPED (reported explicitly, not silently hidden)"
fi

echo "== full check complete =="

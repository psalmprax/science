#!/usr/bin/env bash
set -euo pipefail

# Default paths for APEDP-X science workspace if not set in environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${SCIENCE_ROOT:=$REPO_DIR}"
: "${QCOMP_ROOT:=$REPO_DIR/apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation}"
: "${PYTHON_BIN:=/dev/shm/qvcc-venv313/bin/python}"
: "${PATCH_ROOT:=$REPO_DIR}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$QCOMP_ROOT/verifiers/dqi/verify_submission_dqi.py" ]]; then
  echo "ERROR: official DQI verifier not found under $QCOMP_ROOT/verifiers/dqi" >&2
  exit 2
fi
if [[ ! -f "$SCIENCE_ROOT/apedp_dyn/quantum_compilation/dqi_reference_circuit.py" && \
      ! -f "$QCOMP_ROOT/verifiers/dqi/dqi_reference_circuit.py" ]]; then
  echo "ERROR: dqi_reference_circuit.py not found in expected source locations" >&2
  exit 2
fi

# The adapter belongs in the science checkout. Copy only if it is absent;
# never silently overwrite a user-maintained implementation.
DEST="$SCIENCE_ROOT/apedp_dyn/quantum_compilation/dqi_builder.py"
if [[ ! -f "$DEST" ]]; then
  cp "$PATCH_ROOT/apedp_dyn/quantum_compilation/dqi_builder.py" "$DEST"
  echo "Installed new adapter at $DEST"
else
  echo "Using existing adapter at $DEST"
fi

cd "$SCIENCE_ROOT"
export PYTHONPATH="$SCIENCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
OUT="$SCIENCE_ROOT/artifacts/quantum_compilation/dqi/iteration_sweep_receipts"
mkdir -p "$OUT"

# Include the 222-iteration reference build in the same run as the 216 candidate.
# Each row is persisted after each candidate by the sweep harness.
"$PYTHON_BIN" "$PATCH_ROOT/scripts/sweep_dqi_eea_bound.py" \
  --competition-root "$QCOMP_ROOT" \
  --builder apedp_dyn.quantum_compilation.dqi_builder:build_candidate \
  --iterations 222 216 \
  --shots 102400 \
  --workers 6 \
  --output "$OUT"

"$PYTHON_BIN" "$PATCH_ROOT/scripts/audit_dqi_sweep.py" "$OUT/results.json" \
  | tee "$OUT/audit.txt"

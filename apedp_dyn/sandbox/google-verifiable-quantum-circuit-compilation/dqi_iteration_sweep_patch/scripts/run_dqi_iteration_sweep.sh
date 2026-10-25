#!/usr/bin/env bash
set -euo pipefail

# Required: paths from the actual local environment. Do not assume these exist
# in a different container or machine.
: "${SCIENCE_ROOT:?Set SCIENCE_ROOT to the root of your APEDP-X science checkout}"
: "${QCOMP_ROOT:?Set QCOMP_ROOT to the official quantum-compilation bundle}"
: "${PYTHON_BIN:=/dev/shm/qvcc-venv313/bin/python}"
: "${PATCH_ROOT:?Set PATCH_ROOT to this patch-pack directory}"

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

# Include the 222-iteration reference build in the same run as all shorter caps.
# Each row is persisted after each candidate by the sweep harness.
"$PYTHON_BIN" "$PATCH_ROOT/scripts/sweep_dqi_eea_bound.py" \
  --competition-root "$QCOMP_ROOT" \
  --builder apedp_dyn.quantum_compilation.dqi_builder:build_candidate \
  --iterations 222 180 164 156 152 \
  --shots 102400 \
  --workers 4 \
  --output "$OUT"

"$PYTHON_BIN" "$PATCH_ROOT/scripts/audit_dqi_sweep.py" "$OUT/results.json" \
  | tee "$OUT/audit.txt"

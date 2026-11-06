# DQI iteration-sweep adapter patch

This patch makes the documented `ProosAndZalkaDecoder.build_syndrome_decoding_mbuc(..., n_iters=...)` interface callable by the provided verifier-driven sweep harness and adds a receipt audit.

## Important boundaries

- This is a **builder adapter**, not a new decoder or a verifier.
- The adapter uses the existing `dqi_reference_circuit.ProosAndZalkaDecoder` with 70 12-bit syndrome registers and 4,095 12-bit error-mask registers, following the construction shown in the supplied project materials.
- The sweep harness imports the official `dqi.verify_submission_dqi` verifier and records each serialized artifact's SHA-256 plus its complete verifier report.
- The audit uses the verifier-reported score as authoritative and checks that it reconciles to `repetition_factor * num_qubits**2 * toffoli_count` when those report fields are present. It reports rebuilt-baseline vs pinned-ledger-reference separately; it does not overwrite the pinned value.
- This patch was syntax-checked in the available environment, but **not run through Kickmix or the official verifier**, because this execution session does not contain the project checkout, `kickmix`, competition verifier tree, or `/dev/shm/qvcc-venv313/bin/python`.

## Run in the equipped project environment

Copy this patch folder to a known path and set:

```bash
export SCIENCE_ROOT=/home/psalmprax/ALL_PROJECTS/science
export QCOMP_ROOT=/path/to/google-verifiable-quantum-circuit-compilation
export PATCH_ROOT=/path/to/dqi_iteration_sweep_patch
export PYTHON_BIN=/dev/shm/qvcc-venv313/bin/python
```

Then run:

```bash
bash "$PATCH_ROOT/scripts/run_dqi_iteration_sweep.sh"
```

Outputs are written to:

`$SCIENCE_ROOT/artifacts/quantum_compilation/dqi/iteration_sweep_receipts/`

Review `results.json`, `summary.json`, and `audit.txt`. Do not promote a candidate unless its official verifier report is `PASS`, the score formula reconciles, and the baseline row `n_iters=222` has a valid receipt under the same seed/configuration.

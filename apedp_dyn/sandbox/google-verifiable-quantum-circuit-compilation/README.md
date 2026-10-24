# APEDP-Q alternatives: Python + Rust starter

This is a research scaffold for the quantum-circuit alternatives discussed in the chat. It is **not** a claim that the included runners were executed against the full competition environment. The competition `kickmix` extension, official verifiers, and certain helper files from the source notebooks are external dependencies.

## What is included

- `apedp_q/models.py`: typed contracts, candidate identity and verifier receipts.
- `apedp_q/ledger.py`: append-only JSONL log; promotion requires an authoritative PASS, matching artifact hash and lower measured cost.
- `apedp_q/driver.py`: generic injectable experiment driver for proposer, compiler, estimator and verifier adapters.
- `apedp_q/costs.py`: task cost formula helpers and geometric-mean portfolio score.
- `apedp_q/phase_calibration.py`: generic linear-program phase calibration helper with post-quantization recheck. It does not edit quantum circuits itself. `scripts/calibrate_phase_npz.py` runs it against exported arrays.
- `scripts/nr_qrom.py` + `scripts/run_invsqrt_qrom_sweep.py`: piecewise-linear QROM seed precision search using the reference reversible arithmetic and official inverse-square-root verifier.
- `scripts/run_direct_poly_layout_sweep.py`: runs layout/cache variants from the Direct Phases notebook's exported helper cells and coefficient table.
- `scripts/run_chemistry_ensemble.py`: wrapper for the supplied chemistry rotation/compression helpers.
- `scripts/sweep_dfthcsos_batch.py`: controlled starter-kit batch-size sweep plus official chemistry verifier.
- `scripts/verify_dqi_candidate.py`: runs the official DQI verifier on a compiled candidate; it is a verifier adapter, **not** a decoder generator.
- `scripts/score_portfolio.py`: calculates a projected aggregate score from three per-task costs.
- `rust-core/`: deterministic JSON CLI for precision-grid generation, Pareto filtering, and portfolio score computation. The Rust tool screens/ranks configurations; Python/official verifiers still decide correctness.

## Set up Python package

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,calibrate]"
pytest
```

The tests cover the generic cost and ledger invariants. They do not test `kickmix` or a competition circuit.

## Point to the competition data

Set `QCOMP_ROOT` to the directory containing `kickmix/` and `verifiers/`.

Kaggle-style example:

```bash
export QCOMP_ROOT=/kaggle/input/competitions/google-verifiable-quantum-circuit-compilation
python scripts/run_invsqrt_qrom_sweep.py --output artifacts/qrom_sweep --max-full-verifications 2
```

The QROM script uses the precision configurations from the supplied verifier-guided sweep, performs a full-domain fixed-point screen, compiles viable circuits, then full-verifies the cheapest candidates up to the configured limit. Full verifier runs can be expensive.

## Run direct-polynomial register/cache variants

Export the source notebook's helper cells to a directory with these four files:

- `direct_phase.py`
- `recycled.py`
- `low_memory_adder.py`
- `selected_coefficients.csv`

Then run:

```bash
python scripts/run_direct_poly_layout_sweep.py \
  --helper-dir /path/to/exported_direct_phase_helpers \
  --output artifacts/direct_poly_layouts \
  --cache-values 0 4 8 12 16 --include-reused-zero \
  --full-verifications 1
```

It first checks the supplied coefficient table with its full-domain integer model, validates circuit/model equality on spot-check batches, and invokes the official verifier on the lowest-estimated-cost layout(s). Do not treat the model certificate as a substitute for verifier PASS.

## Run chemistry alternatives

The batch-size sweep needs the separate DFTHCSOS starter kit, whose path contains `starter/task3_circuits/build_circuits.py`:

```bash
export QCOMP_STARTER_KIT=/path/to/starter_kit
python scripts/sweep_dfthcsos_batch.py --batch-sizes 6 7 8 9 --brot 12
```

For ensemble rotations/compression, export `rotate_ensemble.py`, `chemistry_runner.py`, `tune_identity.py`, and `ensemble_compression.py` from the supplied Direct Phases notebook. Point `--source-dfthcsos` at a directory containing `submission.npz` and the matching angle circuit family:

```bash
python scripts/run_chemistry_ensemble.py \
  --helper-dir /path/to/exported_chemistry_helpers \
  --source-dfthcsos /path/to/pinned_dfthcsos \
  --output artifacts/chemistry_ensemble
```

The wrapper requires the competition's task-specific Python dependencies and starter helper APIs.

## Verify DQI candidates

This does not construct a decoder. Once a new circuit generator produces a `.kmb`, run:

```bash
python scripts/verify_dqi_candidate.py path/to/dqi_candidate.kmb \
  --out artifacts/dqi_candidate_verification.json
```

A decoder specialized to one fixed mask is not eligible unless it meets the competition's dynamic `E0`--`E4094` interface and passes the official verifier.

## Rust core

Build requires a Rust toolchain and Cargo. From `rust-core/`:

```bash
cargo test
cargo run --release -- score score.example.json
cargo run --release -- grid precision-grid.example.json > precision_grid.json
cargo run --release -- pareto pareto.example.json
```

`grid` emits proposed precision tuples only. It does not test overflow, accuracy or build circuits; feed its configurations to Python's exact model and official-verifier flow.

## Research protocol

1. Freeze contract and reference hashes.
2. Reproduce the baseline.
3. Import/pin prior candidate artifact and report hashes before using them as incumbents.
4. Keep screening results separate from authoritative verification receipts.
5. A promotion requires a PASS for the exact artifact and a strictly lower measured cost under the same cost formula.
6. Record all failures. Never replace exceptions with artificial scores.
7. Treat a composite score calculated from local task costs as a projection, not as an official leaderboard result.

## Current limitations

- A new DQI decoder algorithm is not implemented here; only its official verifier runner is provided.
- The exact gate-level rewrite and Kickmix circuit rewriting need to be implemented against the source circuit IR/API. No guessed text rewrite is included.
- The phase calibration helper solves the mathematical LP only. Conversion to exact circuit-angle edits and a full verifier run remain adapter work.
- The supplied notebook components have not been fetched or independently run here. Paths to exported helper cells are explicit because the chat did not provide a complete repository checkout or all binary artifacts.

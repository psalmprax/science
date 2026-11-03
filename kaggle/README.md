# Submitting the 89-qubit inverse square root to Kaggle

`kaggle/invsqrt_89q/notebook.ipynb` verifies the champion circuit and writes the submission
artifact. This file covers the two things the notebook cannot do for you: attach the circuit,
and push the notebook.

## What is being submitted

| | |
|---|---|
| benchmark | I — inverse square root |
| logical qubits | 89 |
| max Toffoli-like | 1,466 |
| worst error / tolerance | 0.999623 |
| artifact | `invsqrt_89q_shared_pddd.kmb`, 245,010 bytes |
| sha256 | `ce04c9d54dc42e42455444022b6652a6efb0be2630aca7b485d3efd4e58fa9de` |

Benchmarks II and III are **not** submitted. The Benchmark II decoder exists as a circuit but
cannot be compiled against the pinned `kickmix` build — its ledger clears need an `MR` opcode the
build does not have. Benchmark III is a different case, and the earlier note here was wrong: its
submission is already `kickmix` circuits (`.kmb`/`.kmx`), and the bundle ships
`verifiers/dfthcsos/example_submission/angle_k_reference_circuit.py`, which writes them with stock
`kickmix` from `submission.npz`. No opcode is missing for III; this repository's signed 4-bank
design missed the 128-qubit cap on the angle registers at 132, and sat 4,363 Toffolis over.
Submitting only Benchmark I caps the portfolio score at the cube root of its single-benchmark
ratio — see the notebook.

## Step 1 — the circuit has to travel as a dataset

A notebook cannot reach into this repository, and `enable_internet` is off for competition
notebooks, so the circuit must arrive as an input. Kaggle needs a `dataset-metadata.json`:

```json
{
  "title": "APEDP-Q 89q inverse square root circuit",
  "id": "psalmprax/apedp-q-89q-circuit",
  "licenses": [{"name": "CC0-1.0"}]
}
```

Put it next to the circuit in a scratch directory, then create and upload:

```bash
mkdir -p /tmp/apedp_kaggle_ds
cp artifacts/quantum_compilation/invsqrt/invsqrt_89q_shared_pddd.kmb /tmp/apedp_kaggle_ds/
cp kaggle/dataset-metadata.json /tmp/apedp_kaggle_ds/   # after filling in the username
kaggle datasets create -p /tmp/apedp_kaggle_ds
```

`artifacts/` is gitignored, so the bytes exist only where they were built. On a fresh clone,
regenerate them first — this is the only code path that writes them — and check the hash
before uploading:

```bash
APEDP_Q_KICKMIX_PYTHON=/path/to/python3.13 python scripts/test_shared_pddd_circuit.py --json
```

Confirm the hash survived the upload before trusting anything downstream:

```bash
sha256sum /tmp/apedp_kaggle_ds/invsqrt_89q_shared_pddd.kmb
```

It must print `ce04c9d5…4e58fa9de`. The notebook checks this itself and refuses to run
otherwise, so a corrupted upload fails at cell 3 rather than producing a receipt for a
different circuit.

## Step 2 — attach both, then push

Edit `kaggle/invsqrt_89q/kernel-metadata.json` and replace `psalmprax` in both
`id` and `dataset_sources`. Then:

```bash
kaggle kernels push -p kaggle/invsqrt_89q
kaggle kernels status psalmprax/apedp-q-a-verified-89-qubit-inverse-square-root
kaggle kernels output psalmprax/apedp-q-a-verified-89-qubit-inverse-square-root -p /tmp/apedp_out
```

`competition_sources` mounts the competition bundle read-only at `/kaggle/input/`, which is
where the notebook looks for `kickmix` and the official verifiers. If the mount path differs
on the day, the notebook prints every `kickmix` directory it can see before failing — paste
that into `QCOMP_ROOT` rather than guessing.

## Step 3 — the submission route

Two routes exist and the competition's Submit page decides which applies:

- **file submission** — upload `submission/invsqrt.kmb` from the notebook output under
  whatever filename the Submit page names;
- **code competition** — publish the notebook version and submit that.

The notebook writes both the `.kmb` and its receipt, so either route works without re-running
the verifier. Check the required filename and route on the Submit page before the graded run;
the notebook does not rename the artifact to guess.

## Runtime and shot budget

`SHOTS_PER_X` in §4 is 64, the graded default. It cannot be lowered: `num_shots` is
`shots_per_x x 3,133,452` and the verifier refuses anything that is not a multiple of its
256-shot batch. Since 3,133,452 = 4 x 783,363 with 783,363 odd, `shots_per_x` must itself be a
multiple of 64. Values such as 3200 are legal but are 50x the graded budget, which is why they
take hours rather than minutes. Both apply identical pass/fail logic; only the
sample size differs, and the receipt records which was used. Set `SHOTS_PER_X = None` when the
run time is affordable and you want the receipt to match the graded configuration exactly.

## If something fails

| symptom | cause |
|---|---|
| `Competition bundle not found` | competition data not attached, or `competition_sources` missing from the metadata |
| `not the pinned champion artifact` | wrong dataset, truncated upload, or a rebuilt circuit — the rebuild changed the bytes |
| `verifier exited 1` | the circuit itself; the printed JSON carries the reason and the failing input |
| `shutil.SameFileError` on the copy | the circuit was found inside `/kaggle/working`, so the output path is the input path |

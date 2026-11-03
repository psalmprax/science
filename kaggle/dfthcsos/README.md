# Benchmark III notebook: the 123-qubit angle family

`notebook.ipynb` builds dfthcsos `ANGLE_k` circuits at `brot = 10`, `J = 10`, checks them with
the competition's own `scoreinfra.circuits`, and reports what it measures. It claims no score:
the pinned `FeMoCo76o113e.fcidump` is not distributable, so `verify.py` cannot run without the
competition data mounted.

## What it produces

| | |
|---|---|
| peak qubits per `ANGLE_k` | **123** (cap 128, five spare) |
| circuits for 75 angles | **8** (`ceil(75/10)`) |
| Toffoli per circuit | **3,974** (3,975-node tree, one early-break) |
| Toffoli, forward + reverse | **63,584** |
| word recovery | 298,200 of 298,200 words **exact** |

The measured comparison is inside the notebook and is about precision, not about any other
submission: `brot = 12, J = 8` fits (119 qubits) and costs 79,480 Toffolis; `brot = 10, J = 10`
costs 63,584 for 4 more qubits and two fewer bits of precision.

## Uploading

1. Put the competition bundle (`kickmix/`, `verifiers/`, `sample_submission/`) in a Kaggle
   **dataset** and attach it to the notebook; the first code cell searches `/kaggle/input` for
   `verifiers/dfthcsos` and stops if it is absent, rather than working off a stand-in.
2. Set `dataset_sources` and `id` in `kernel-metadata.json`.
3. `kaggle kernels push -p kaggle/dfthcsos`.

With the FCIDUMP added to the same dataset, replace the last cell's report with
`python dfthcsos/verify.py submission/ --tree-edges submission/submission.npz` and the score
exists; nothing else in the notebook changes.

## Known limits

- `shots_per_address = 2` in the verification cell, so recovery is checked against two
  independent measurement trajectories per address, not the verifier's 64. The circuits are
  Clifford+Toffoli with no randomness in the data path, but this is weaker evidence than the
  official run and is stated as such.
- The notebook does not re-derive `lambda_eff`, `E_gap` or the score.

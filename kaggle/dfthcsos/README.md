# Benchmark III notebook: the 123-qubit angle family, checked and scored

`notebook.ipynb` builds dfthcsos `ANGLE_k` circuits at every `(brot, J)` the 128-qubit cap
admits, checks each family with the competition's own `scoreinfra.circuits`, and scores it
with `dfthcsos/verify.py` against the `FeMoCo76o113e.fcidump` in the competition mount.

## What it measures

| | |
|---|---|
| peak qubits per `ANGLE_k` | **123** (cap 128, five spare) at `brot = 10, J = 10` |
| circuits for 75 angles | **8** (`ceil(75/10)`) |
| Toffoli per circuit | **3,974** (3,975-node tree, one early-break) |
| Toffoli, forward + reverse | **63,584** |
| word recovery | 298,200 of 298,200 words **exact** |

## What it scores (kernel version 4)

| `brot` | `J` | `K` | `Q` | `T` | `error_1` | `error_2` | score |
|---|---|---|---|---|---|---|---|
| 12 | 8 | 10 | 119 | 79,480 | 3.185 mHa | 18.640 mHa | **3.3556e9** PASS |
| 11 | 9 | 9 | 122 | 71,532 | 221.29 mHa | 51.74 mHa | `inf` FAIL |
| 10 | 10 | 8 | 123 | 63,584 | 242.30 mHa | 59.38 mHa | `inf` FAIL |

The cap admits all three and the 20 mHa accuracy gate admits one, so the cheapest family —
the one this notebook was first written to promote — is inadmissible. Its 19.98% Toffoli
saving is real and cannot be spent. The 11-bit row is what makes that a measurement rather
than a guess: it sits between the two in every resource and still fails by 201 mHa, so the
floor is 12 bits and the gate, not the qubit cap, sets it.

The 12-bit family built here is byte-identical to the bundle's own sample submission
(`submission sha256 276d464f…`, same score), which is the check that the builder is sound:
two independent implementations, one submission.

## Uploading

1. The circuits are checked by the competition bundle itself, so the competition must be
   attached as `competition_sources`; the first code cell searches `/kaggle/input` for
   `verifiers/dfthcsos` and stops if it is absent, rather than working off a stand-in.
2. `id` in `kernel-metadata.json` must be the slug Kaggle assigns; `dataset_sources` stays
   empty because the bundle (FCIDUMP included) comes from the competition mount.
3. `kaggle kernels push -p kaggle/dfthcsos`.

## Known limits

- Both failing rows are produced by requantising the sample's 12-bit words. A coarser table
  would need words refit at that precision — rounding moves the words without moving the
  integrals they represent — and no refit exists in this checkout.
- `PASS` is admissibility, not a ranking: the leaderboard also holds submissions on different
  factorizations, which move `lambda_eff` independently of these circuits.

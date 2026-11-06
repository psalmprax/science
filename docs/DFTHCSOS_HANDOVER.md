# Benchmark III handover: what is measured, what is missing, what finishes it

**Status:** circuits built and verified for word recovery, then scored by the real verifier: the bundle's 12-bit family **passes** (3.19 / 18.64 mHa) and every coarser family this repo can build **fails** the 20 mHa budget · **Detail:**
[`docs/DFTHCSOS_SUB128_SCOPE.md`](DFTHCSOS_SUB128_SCOPE.md) · **Notebook:**
[`kaggle/dfthcsos/notebook.ipynb`](../kaggle/dfthcsos/notebook.ipynb) · **Pins:**
[`tests/test_dyn_dfthcsos_sub128_family.py`](../tests/test_dyn_dfthcsos_sub128_family.py)

This is the handover for one benchmark, written so the next person does not have to
reconstruct which numbers exist, which do not, and why.

## The input that is no longer missing

`FeMoCo76o113e.fcidump`, 128,128,348 bytes, sha256
`e5a48af746fd075026dec9e8f5df2d363802a121e8c53af9a0b984bde695e1bd`, at
`verifiers/dfthcsos/FeMoCo76o113e.fcidump`.

It ships in the competition's own data, so this checkout can obtain it:

```bash
kaggle competitions download -c google-verifiable-quantum-circuit-compilation \
    -f verifiers/dfthcsos/FeMoCo76o113e.fcidump -p /tmp/qcc
```

It still cannot be generated, approximated or substituted. `scoreinfra/reference.py` pins both its
size and its hash, `check_fcidump` has no override argument, and
`dfthc_math.load_inputs` calls `load_reference(check_fcidump())` unconditionally — so every
scoring path needs it. The submission `npz` does **not** stand in for it: it carries `U`,
`W`, `H2_I`, `H1_I`, `Hsym_lower`, and `load_inputs` *replaces* the submitted `H1_I` /
`Hsym_lower` with the baseline's own before deriving `S` and `beta_1`. Nothing shipped here
carries the one- and two-electron integrals.

## What is measured, and reproducible now

| quantity | value | how |
|---|---|---|
| peak qubits per `ANGLE_k` | **123** (cap 128, 5 spare) | `circuit_costs` via the builder |
| circuits for 75 angles | **8** (`J = 10`) | same |
| Toffoli per circuit | **3,974** | same; 3,975-node bound, padding-trimmed |
| two-pass charge | **63,584** = 2 × 8 × 3,974 | checker's own `num_toffoli_like` |
| word recovery | **298,200 / 298,200 exact** | `scoreinfra.circuits.recover_words_from_circuits` |
| angle error at `brot = 10` | max 3.068e-3 rad, rms 1.878e-3 | requantisation of the 12-bit table |
| angle error at `brot = 9` | max 6.136e-3 rad, rms 3.600e-3 | same |
| max rotation perturbation | 4.339e-3 (`brot=10`), 8.677e-3 (`brot=9`) | `2√2·sin(δ/2)` |

Reproduce any row with:

```bash
python3.13 scripts/setup_verifier_env.py                 # once; the venv that loads kickmix
python3.13 scripts/build_dfthcsos_sub128.py --brot 10 --batch-size 10 --verify
python3 -m pytest tests/test_dyn_dfthcsos_sub128_family.py -q
```

## The command that now runs, and what it said

With the FCIDUMP in place at the path above and the circuits written to `submission/`:

```bash
cd apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers
python dfthcsos/verify.py /path/to/submission --tree-edges /path/to/submission/submission.npz
```

It runs, and it rejects this family:

| submission | `Q` | `T` | `error_1` | `error_2` | verdict |
|---|---|---|---|---|---|
| sample, 12-bit words | 730 | 93,237 | **3.19 mHa** | **18.64 mHa** | PASS, `lambda_eff` 49.301, cost 3.356e9 |
| `brot = 11, J = 9` | 733 | 84,989 | 221.3 mHa | 51.7 mHa | FAIL |
| `brot = 10, J = 10` | 123 | 63,584 | **242.3 mHa** | 59.4 mHa | FAIL |
| `brot = 9, J = 11` | 122 | 55,636 | 338.3 mHa | 89.6 mHa | FAIL |
| `brot = 8, J = 13` | 127 | 47,688 | 568.2 mHa | 159.2 mHa | FAIL |

The tolerance is 20 mHa. Everything the family was checked for before still holds -- the table
loads exactly, the register contract passes, the phase and workspace checks pass, the width is
inside the cap -- and none of it touches the chemistry error. Rounding the sample's 12-bit words
to 10 bits is what costs 242 mHa, twelve times the budget.

`brot = 11, J = 9` is the row that closes the precision question: it is the only submission
between the two (9 circuits, 71,532 Toffolis, 122 qubits) and it fails by 201 mHa, so the
floor is 12 bits and it is the accuracy gate that sets it -- not the 128-qubit cap, which
admits all three. The one-body residual is 3.19 mHa at twelve bits and 221 mHa at eleven: a
70x jump is not rounding, it is the overlap-based pairing of submitted rows onto exact
eigenvectors switching assignment once the decoded two-body block moves far enough.

The 12-bit family built by `kaggle/dfthcsos/notebook.ipynb` reproduces the sample
**byte for byte** (`submission sha256 276d464f…`, same score 3.3556e9): two independent
builders, the same words, the same submission.

## The question that was waiting on that input, and its answer

`brot = 9, J = 11` costs **55,636** Toffolis (122 qubits, 7 circuits), 12.5% below the 10-bit
family, and the open question was whether a refit of its words could pay for the coarser grid:
the 9-bit family wins while `lambda_eff(9) / lambda_eff(10) >= 0.875`.

It cannot. `scripts/refit_dfthcsos_words.py --oracle 12 11` builds the best submission any
grid allows: it encodes the unquantised vectors on the Givens tree and then rebuilds the
one-body rows from the *decoded* two-body block, which is the order the shipped table uses.
At 12 bits it reproduces the shipped family exactly (`3.185 / 18.640 mHa`, `lambda_eff`
49.301), which is the check that the construction is the real one; at 11 bits it lands at
`6.480 / 25.035 mHa` — still 5 mHa over the gate. No search can beat an oracle, so the
11-bit and 10-bit families are unreachable, not merely unfitted.

The reason is one level up. The unquantised vector set is itself at **16.13 mHa**, so the
whole factorisation has **3.87 mHa** of headroom under a 20 mHa gate, and one bit of grid
costs 8.9 of it. The floor is 12 bits because the factorisation is near the gate, not
because the words were rounded.

## Ideas that were tried and rejected, with the measurement

| idea | verdict | evidence |
|---|---|---|
| 9-bit delta payload from the Givens sign invariant | **false on this table** | `bit9 == bit8` on 78.59% of words; the table spans θ/π ∈ [0, 2) |
| share the tree's 11 ancillas | **impossible** | builder refuses: parent control and child target are the same qubit, 7,943 times |
| fuse the deepest leaves into multi-controlled X | **costs more than it saves** | frees 1 ancilla for ~46× Toffolis; the fused build also failed to reproduce the table |
| replace the reverse pass with measurement-based clearing | **contract forbids it** | checker requires restore-to-previous-batch after every reverse step; `R` also randomises phase |
| share one tree between the two passes | **impossible** | between forward and reverse of one circuit nothing differs; `T` = 2 × Σ exactly |
| early-break over delta tables | **nothing to collapse** | 31,808 address-loads, 31,808 distinct delta rows |
| `O(sqrt(A))` shared comparator ladder | **addressing only, total loses** | §5.1: 183 addressing Toffolis vs 3,974 (21.7x cheaper), but 40.3x the tree on the shipped `J=10` family (16.1x at `J=4`) because the payload loads become `ccx`; §5.2 measured the crossing at **0.87 set bits per address**, and one set bit on every address still loses at 1.11x |

## What is not known

- `lambda_eff` is now measured for the reference at **49.301** (`E_gap` 3.965, cost
3.356e9), and the failing quantity is not the metric but the words. A refit at the coarser
precision was measured (`scripts/refit_dfthcsos_words.py --oracle 12 11`) and does not
rescue it: the oracle 11-bit encoding is 25.0 mHa, against a 20 mHa gate and a 16.13 mHa
continuous floor.
- Whether a sub-`A` Toffoli tree exists: **no, not for this payload.** Two constructions were
  built and measured. The chunked read (§5) costs 17.2x and does not recover the words at any
  block size. The shared comparator ladder the `O(sqrt(A))` model actually describes (§5.1)
  does recover, and it does cut addressing from 3,974 Toffolis to 183 -- 21.7x. It still
  totals 40.3x worse on the shipped family, because a two-control load turns all 159,068
  payload bits into Toffolis. §5.2 then asked how sparse a table would have to be, on real
  circuits: the crossing is 0.87 set bits per address, and one set bit on every address is
  already 1.11x too many. The 13x figure stays withdrawn and is now replaced by that
  decomposition rather than by a guess.
- Which density of *real* integral data would reach 0.87 set bits per address: the delta
  words of an FCIDUMP-derived table are not known here, so §5.2 measures the threshold and
  not the candidate's own density.
- Word recovery was verified at 2 shots per address here, not the verifier's 64.

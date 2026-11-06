# Upstream submission: `MR`, a Z-basis read for `kickmix`

**Status:** draft, ready to file · **Patch:** [`scripts/kickmix_mr_opcode.patch`](../scripts/kickmix_mr_opcode.patch)
(1,422 lines, +940/−9, 19 files) · **Build:** [`scripts/build_kickmix_mr.py`](../scripts/build_kickmix_mr.py) ·
**Spec:** [`docs/MR_INSTRUCTION_SPEC.md`](MR_INSTRUCTION_SPEC.md) · **Request:**
[`docs/MR_PATCH_REQUEST.md`](MR_PATCH_REQUEST.md) · **Decision it unblocks:**
[`docs/BENCHMARK2_STATUS_DECISION.md`](BENCHMARK2_STATUS_DECISION.md) §D2

Nothing here has been sent. The repository holds no upstream remote, issue tracker or
maintainer address, so this document is the submission body and filing it is a delivery
step this checkout cannot perform. Everything in the patch is written against sources
this repository already vendors, so it needs no discovery work from a reviewer.

## The PR body

> **Add `MR` (Z-basis measure-and-reset) as a first-class instruction**
>
> The instruction set can measure only in the X basis (`HMR`), and on `|1>` that returns
> a random bit: 24 ones in 64 shots. There is no way to read a computational-basis
> predicate into a classical bit and then keep going, so no circuit can make its work
> conditional on data it has just computed. `R` clears a qubit but randomises the
> tracked phase — `x(q); reset(q)` reports a random `phase_flipped` on 37 of 64 shots,
> in a circuit that contains no phase at all — so it cannot be used as a read-and-clear
> either.
>
> This adds `MR` and `MR_IF`: measure in Z, write the outcome to a classical bit, leave
> the target in `|0>`, leave the shot's accumulated phase untouched. It is `HMR` with
> the basis changed and the outcome reported. In the Reed-Solomon decoder that
> motivated it, six of these reads take the official instance from 5,725,108.4 to
> 1,221,408.0 Toffolis per shot at equal error rate — a **4.5898× score ratio** under
> the competition verifier at 102,400 shots, 0 value and 0 phase errors.

## What the gap is, in one measurement

Everything the gated decoders do is conditional on a predicate the circuit itself
computes: is `sigma(alpha^j)` zero, did this EEA round quiesce, is `c3` clear. The IR can
compose conditions (`push_condition` nests and costs nothing) but cannot read one, and
the four ways around it are all closed:

| route | why it fails |
|---|---|
| `push_condition(qubit)` | `TypeError` — classical bit only |
| `bit_store(control=qubit)` | `ValueError` |
| conjugate into X with `H` | `h` is not on the builder; `{X, Z, R, CX, CZ, SWAP, CSWAP, CCX, CCZ}` is closed under the Z-diagonal group |
| `R` then copy its discarded outcome | `R` randomises the tracked phase: 11/16 random `phase_flipped`, and the arms built on it are wrong on 35 of 64 official shots while costing what the correct arms cost |

## The change

| component | what it does | added / removed |
|---|---|---|
| `circuit/op_type.h`, `op_type.cc` | `OpType::MR = 0x4 \| OP_HAS_1Q \| OP_HAS_1B` (0x54), `MR_IF` (0xD4), names, `OP_TYPE_TABLE`, `.kmx` `define` | +8 / −2 |
| `build/circuit_builder.{h,cc}` | `mr` / `mr_if` / `broadcast_mr`, coercion and broadcast identical to `hmr` | +60 / −0 |
| `sim/sim.h` | Z-basis branch: `bits[t] = (bits[t] & ~cond) \| (q & cond)`, `qubits[target] &= ~cond`, phase untouched | +51 / −0 |
| `circuit/circuit*.cc`, `circuit_util.cc`, `mutable_circuit.cc` | `MR q%u b%u` (and ` if b%u`) in the writer, `str`, `MR=b` text diagram, `MZ` box in the HTML diagram, reaction depth | +57 / −4 |
| `py/circuit/circuit_builder.pybind.cc` | `CircuitBuilder.mr(target, output)` | +91 / −0 |
| `py/circuit/circuit.pybind.cc`, `py/interactive_simulator.pybind.{h,cc}` | `CIRCUIT.kmb` round-trip and the interactive simulator surface | +172 / −0 |
| `py/operation_statistics.pybind.cc` | its own `MR` group, never `num_toffoli_like` | +2 / −1 |

Three API decisions a reviewer will want stated rather than inferred:

1. **`MR` is not Toffoli-like.** It is a measurement and a Clifford reset, priced at zero
   exactly as `HMR` is. Counting it as Toffoli-like would make every circuit above look
   worse than the reference it improves.
2. **The phase contract is the whole point.** `_reset_z` must not touch the frame that
   `read_phase_flipped_across_shots` reports. Implementing `MR` as `R` plus a copied bit
   reproduces the 11/16 failure and passes every other test.
3. **`R` is unchanged.** This adds the variant that reports what it discarded; it does
   not redefine the existing opcode, so no shipped circuit changes meaning.

No verifier change is required: the junk-qubit check reads all non-syndrome qubits at the
end and `MR` leaves them at `|0>`, so a circuit using it verifies untouched.

## What the patch is worth, measured under the official verifier

Both runs are the competition's own `verify_submission_dqi.py` at the official instance
(m=4095, n=70, GF(2¹²)), 102,400 shots, batch 256, `log2_delta` −128 — the full scoring
path, no stubs, on a local `MR` build:

| | reference | six-lever arm | campaign arm |
|---|---|---|---|
| status | PASS | PASS | PASS |
| score | 2.443946358367293e13 | **5.324728209826311e12** | 2.990338795715691e12 |
| value / phase errors | 0 / 0 | **0 / 0** | 0 / 0 |
| `T` per shot | 5,725,108.382 | **1,221,408.045** | 685,936.205 |
| `Q` | 1,893 | **1,913** | 1,913 |
| circuit sha256 | `959471d0…` | `749e347b…` | `13f24dee…` |
| ratio vs reference | 1.0000× | **4.5898×** | 8.1728× |

Records: [`benchmarks/quantum_compilation/dqi_arm_submission_score.json`](../benchmarks/quantum_compilation/dqi_arm_submission_score.json),
[`dqi_campaign_arm_submission_score.json`](../benchmarks/quantum_compilation/dqi_campaign_arm_submission_score.json).

### The 4.5898×, decomposed

The score is `R(eps_plus) · Q² · T`, so the ratio is not the Toffoli ratio:

```text
T ratio            5,725,108.382 / 1,221,408.045 = 4.6873
Q ratio            1,913 / 1,893               = 1.0106
Q² penalty                                     = 1.0212
score ratio        4.6873 / 1.0212             = 4.5898
```

The `MR` reads cost 20 qubits — one shared read ancilla reused by every read, plus the
registers the gates keep live — and that 2.1% is why the headline is 4.5898× and not
4.687×. It is quoted here rather than buried because a reviewer reproducing the Toffoli
count and finding 4.6873× should know which number the patch actually earns. `R` is
identical in both runs: 0 errors at 102,400 shots on each side, so the comparison carries
no error-rate advantage either way.

## What it does not buy

The Benchmark II target is 1.6975e12, and the six-lever arm is 5.3247e12 — still **3.14×
away**. This patch removes the part of the decoder's cost that its own predicates make
avoidable; the rest is the algorithm. Stated plainly so nobody files it expecting the
benchmark to close: it does not, and §D3 of the decision record explains why the residual
is not field arithmetic either.

## Reproducing it

```bash
# build a local MR extension (pybind11 + setuptools for CPython 3.13)
python3.13 scripts/build_kickmix_mr.py          # prints build/kickmix_mr/lib

# correctness: phase is the discriminator, not the value
python3.13 - <<'PY'
import kickmix as km
b = km.CircuitBuilder(); q = b.create_qubit(); bit = b.create_bit()
b.x(q); b.mr(q, bit)
print(b.finish_circuit().simulate(64).read_bit(bit))   # 64/64 ones
PY

# cost, official instance, 64 shots
python3.13 scripts/measure_mr_arm_costs_kickmix.py

# full official score at 102,400 shots against a real build
python3.13 scripts/measure_dqi_arm_submission_score.py
```

Acceptance is four checks, and the third is the one that separates a correct `MR` from a
plausibly wrong one:

1. `scripts/probe_dqi_predicate_gating.py --m 15 --n 6 --degree 4 --shots 64` → 2.998×
   with `value_errors = phase_errors = 0`.
2. The same at `--m 255 --n 14 --degree 8 --shots 512` → 3.709×, 0/0.
3. `x(q); mr(q, b)` reports `b = 1` and `phase_flipped = 0` on every shot, where
   `x(q); reset(q)` reports a mixture — 37/64 in this build.
4. `pytest tests/test_dqi_predicate_gating.py -q`.

## Patch mechanics, verified

`s scripts/kickmix_mr_opcode.patch` applies to `git archive HEAD` of the vendored
`kickmix` sources with `patch -p1 --forward --batch`: 19 files, no rejects, and the two
`sim.h` hunks land at offset −17 lines, so the tree the patch was written against is 17
lines longer than the vendored one. That offset is stable and harmless; a reviewer on
current `main` may see different offsets again, which is the only porting work expected.

## Split before filing

The patch is 940 added lines for "one opcode", which no reviewer should accept without
asking why. It contains two independent changes:

| | files | added / removed |
|---|---|---|
| **PR 1 — `MR`** | `op_type.{h,cc}`, `sim/sim.h`, `circuit_builder.{h,cc}`, `circuit*.cc`, `circuit_util.cc`, `mutable_circuit.cc`, `py/circuit/circuit_builder.pybind.cc`, `py/circuit/circuit.pybind.cc`, `py/interactive_simulator.pybind.{h,cc}`, `py/operation_statistics.pybind.cc` | **382 / −7**, 15 files (`sim.h` lands in two blocks) |
| **PR 2 — `gf_product = 'and54'`** | `gen/gf_arithmetic/gen_gf_mul54.{cc,h}`, `gen_gf_mul.cc`, `py/circuit/subs/gf_arithmetic.pybind.cc`, plus the `gf_product` property hunks in `circuit_builder.{h,cc}` and its pybind | **558 / −2** (6 files) |

PR 2 is a Karatsuba base-case change (`mu(12) = 54` instead of 63 ANDs, 75 → 66 Toffolis
per controlled call) that happens to ride in the same tree. It is worth filing, but it is
reviewable on its own terms and it should not gate the opcode: `MR` alone earns 4.144×
and PR 2 takes it to 4.5898×.

The split is already made and checked: `scripts/split_kickmix_patch.py` classifies each
hunk by whether its changed lines mention the policy, writes `build/kickmix_split/mr.patch`
and `gf54.patch`, and proves both apply. Both are also **compiled here**: the `MR`-only
extension imports with `mr` present and no `gf_product`, and the stacked one imports with
both, defaulting to `karatsuba` and switching to `and54` on assignment. Files cannot be
split instead of hunks, because the `gf_product` enum and its pybind live in files `MR`
also edits; that is why the classification is at hunk granularity and why the two patches
apply in order rather than independently.

## Review risk

One failure mode is silent, which is why the acceptance tests lean on phase: an `MR` that
clears like `R` returns the right syndromes and fails only on half the shots. The
`operation_statistics` key is the second: counting `MR` as Toffoli-like would make every
circuit in this document look worse than the reference it improves. Rollback is trivial —
the opcode is additive, and removing it restores the current instruction set with every
existing circuit still verifying.

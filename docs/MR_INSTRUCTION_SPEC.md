# `MR`: the one instruction between the DQI reference circuit and 3.599×

**Status:** patch written and measured on a rebuilt `kickmix`; the pinned competition
binary is unchanged (§6) · **Scope:** `kickmix` (Benchmark II only) ·
**Evidence:** `benchmarks/quantum_compilation/dqi_predicate_gating.json`,
`scripts/probe_dqi_predicate_gating.py`, `tests/test_dqi_predicate_gating.py` ·
**Projected result (oracle-gated):** Benchmark II `T_mean` 5,725,071.61 → 1,382,207.08
(4.142×) at `Q` = 1,893 with 0 value and 0 phase errors on 64 shots.
**Measured with the implementation (§6a):** 5,725,071.6 → 1,381,390.0, **4.144×** — the
projection is reproduced by the gated circuits, every stage within 0.06% of it. A sixth
lever (§6b) takes it to 1,325,230.0, **4.320×**, at `Q` = 1,894 against the reference's
1,893, and the 54-AND field multiply of §7 to 1,230,640.0, **4.652×**, at `Q` = 1,897.

This note specifies exactly what `kickmix` would have to gain. It is written from
the outside: the repository carries `kickmix.so` and its `LICENSE.txt`, not the
sources, so the patch points below are API-level and behavioural, not line numbers.
The instruction set is readable from the extension itself and is
`{NEG, BIT_INVERT, BIT_STORE0, BIT_STORE1, X, Z, R, HMR, CX, CZ, SWAP, CSWAP, CCX,
CCZ, Z_POW, INFO_TAG, DEBUG_PRINT, PUSH_CONDITION, POP_CONDITION}` — everything the
design below needs is already there except one opcode.

## 1. Why one opcode is the whole gap

Three optimisations were measured on the official DQI instance, and all three are
blocked by the same missing primitive.

| Lever | What it needs | Worth |
|---|---|---|
| Skip the Forney body where `sigma(alpha^j) != 0` | a classical bit equal to a computational-basis predicate | 1.332× |
| Clear the EEA instead of running its adjoint | a phase-safe read-then-clear of 1,803 scratch qubits | 2.652× (with the above) |
| Skip the multiply sweep in rounds where `c0 = c3 = 0` | a classical bit per round | 3.599× (with the above) |
| Skip the register swap in rounds where `c3 = 0` | the same round bits carry `c3` | 3.929× (with the above) |
| Skip the shift step on its true liveness | `flag` *after* `_update_flag`, same round bits | 4.142× (with the above) |

The IR can skip work — `push_condition` composes, nests, and costs nothing — but it
cannot *read* a qubit in the computational basis:

- `hmr` measures in the **X** basis. On `|1>` it returns a random bit (measured:
  31/64 ones). A `sigma(alpha^j) == 0` flag lives in the Z basis and is unreachable.
- `push_condition(qubit)` raises `TypeError`; `bit_store(control=qubit)` raises
  `ValueError`. Neither accepts a quantum control.
- No Hadamard can be synthesized: `{X, Z, R, CX, CZ, SWAP, CSWAP, CCX, CCZ}` is
  closed under the Z-diagonal group, so the predicate cannot be conjugated into the
  X basis either.
- `R` (Z-basis reset) is not a substitute. It clears the qubit but **randomizes the
  tracked phase**: a circuit with no phase in it at all, `x(q); reset(q)`, reports a
  random `phase_flipped` on 11/16 shots, while the same circuit without the reset
  reports 0/16. The arm that clears with `R` is wrong on 35 of 64 official-instance
  shots while costing exactly what the correct arm costs.

`del_and`-style Clifford+measurement gadgets do preserve the phase, but they
uncompute an AND rather than clearing an arbitrary dirty qubit, so they cannot
stand in for a general reset either.

## 2. The instruction

```python
CircuitBuilder.mr(target, output)
```

| | |
|---|---|
| `target` | `km.q`, `km.array` or `Sequence[km.q]` — qubits to measure and clear |
| `output` | `km.b`, `km.array` or `Sequence[km.b]` — matching classical bits |
| broadcasts | yes, like `hmr` |

Semantics, per shot:

1. measure `target[i]` in the **Z** basis and write the outcome to `output[i]`;
2. leave `target[i]` in `|0>`;
3. leave the shot's accumulated phase **unchanged** — the distinction from `R`,
   and the only behavioural difference that matters.

Serialised form, next to `HMR q%u b%u`: `MR q%u b%u` (and `HMR` already carries a
classical guard in its formatter — `MR` should take the same `if b%u` suffix).

Not Toffoli-like: `MR` must not enter `num_toffoli_like`. It is a measurement and a
Clifford reset, priced as zero, exactly as `HMR` is priced today.

`R` stays as it is; this proposal does not change its semantics, only adds the
variant that reports what it discarded.

## 3. Where it plugs in

1. **Instruction enum** (`Circuit.__getitem__` keys and the `.kmb` parser/formatter):
   add `'MR'` beside `'R'` and `'HMR'`.
2. **`CircuitBuilder.mr`**, with the same argument coercion/broadcast path as `hmr`
   (both take a qubit sequence and a bit sequence of equal length).
3. **Simulator**: the Z-basis measurement must feed the same per-shot classical
   state that `bit_store`/`hmr` outcomes feed, and the reset must *not* disturb
   `read_phase_flipped_across_shots`. Today `R` does disturb it; that is the defect
   this instruction routes around. Implementing `MR` as `R` plus a copied bit will
   fail every acceptance test in §4.
4. **`Simulator.use_same_registers_as` / `write_across_shots`**: no change — `MR`
   writes bits that `write_across_shots` can already overwrite for oracle runs.
5. **`operation_statistics`**: `MR` counted under its own key, not
   `num_toffoli_like`.
6. **`verifiers/dqi/verify_submission_dqi.py`**: no change. The junk-qubit check
   reads all non-syndrome qubits at the end and `MR` leaves them at `|0>`, so a
   circuit using it verifies unchanged.

## 4. Acceptance tests for the rebuild

A rebuild is correct when all four hold. Each is already mechanised.

1. `python scripts/probe_dqi_predicate_gating.py --m 15 --n 6 --degree 4 --shots 64`
   reports `T` 33,232.22 → 12,850.47 with `value_errors = phase_errors = 0`
   (2.586×, 46.98% dead sweeps).
2. The same probe at `--m 255 --n 14 --degree 8 --shots 512` reports 3.205× with 0/0.
3. A phase-only regression: `x(q); mr(q, b)` must report `phase_flipped = 0` on
   every shot where `x(q); reset(q)` reports a mixture. This is the single test that
   distinguishes `MR` from `R`.
4. `tests/test_dqi_predicate_gating.py` still passes; its recorded numbers are the
   ones above.

Once the opcode exists, the three levers reduce to substitutions the probe script
already performs:

- `push_condition(mr_bit_of(flip_if_equal(sigma, 0, ...)))` around the Forney body;
- `mr(q, b)` on each scratch qubit where the probe writes `cx(b, q)` from an oracle
  readout, and on each syndrome bit where it re-prepares from `C{i}`;
- `mr(ctrl[0], b0); mr(ctrl[3], b3); push_condition(b0 or b3)` around
  `_multiply_and_add_or_sub`, and `push_condition(b3)` around `_swap_a_and_b`;
- `mr(flag, bf)` **after** `_update_flag` and `push_condition(b1 or b2 or (not bf and (b0 or b3)))`
  around `_shift_b_array_right` — the post-update flag is required: reading it at the top
  of the round fails on every shot (512/512 value errors at m=255), because `_update_flag`
  toggles it first.

## 4a. Review request

This note is the change request for the `kickmix` rebuild. Nothing has been sent
externally: the repository has no outbound channel authorised for it, and the package is
self-contained so whoever holds the `kickmix` sources can work from it directly. To
review, reproduce the three numbers in §4 with

```bash
python scripts/probe_dqi_predicate_gating.py --m 15 --n 6 --degree 4 --shots 64
python scripts/probe_dqi_predicate_gating.py --m 255 --n 14 --degree 8 --shots 512
pytest tests/test_dqi_predicate_gating.py -q
```

and check §2 against the existing `hmr` implementation — `MR` is an `hmr` that measures
`Z` instead of `X` and reports the outcome.

## 5. What this does not buy

3.599× is not the 12.86× the Benchmark II target needs. `T_mean` lands at 1.591M
against 445,229 — a 3.57× gap — and what remains is one multiply-accumulate sweep
per live round, which the algorithm requires. That 445,229 figure is a target, not a
derivation: `scripts/measure_dqi_recycling_offsets.py` shows degree recycling can remove
one `logm`-bit register at most (8 qubits at `n = 70`; supported peak **1,885**, not
1,789) and cannot remove a gate, so the gap below is closed by arithmetic rather than by
register work. **That register is now closed as a lever too:** the 1,885 build fails 64-shot
equivalence and is score-negative, so nothing is scored at 1,885 — see
[`DQI_RECYCLED_PASS_BODIES.md`](DQI_RECYCLED_PASS_BODIES.md). `Q` stays at 1,893: 1,752 persistent
registers (`2 × 73 × 12`, both factors frozen by the instance) plus 141 transient.
The honest reading is that this instruction removes the part of the reference
circuit's cost that is *avoidable*, and the rest is the decoder.

Later and measured, so that this section is not read as the last word: reading each
movement and product predicate and gating the work on it (the same `mr` gadget, at
per-position grain) took the six-lever arm **1,221,466.8 → 686,014.8 per shot, 1.541×** the
target when stacked with the flag/metadata gate this instruction was specified for —
[`EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md). The avoidable part was much larger
than this specification could see; the rest is still the decoder, and 686,014.8 is still
240,785.8 above 445,229.

## 6. Status: the patch exists, and the arms run on a rebuilt `kickmix`

The instruction is no longer only a specification. `scripts/kickmix_mr_opcode.patch`
implements it against the sources vendored at
`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src`,
and `scripts/build_kickmix_mr.py` compiles a local extension from those sources with
the patch applied (into `build/kickmix_mr/lib`, which is gitignored).

What landed, per §3:

| Component | Where |
|---|---|
| opcodes | `OpType::MR = 0x4 \| OP_HAS_1Q \| OP_HAS_1B` (0x54) and `MR_IF` (0xD4), with names, the `OP_TYPE_TABLE` entry and the `.kmx` parser's `define` |
| builder | `CircuitBuilder::mr` / `mr_if` / `broadcast_mr`, plus `CircuitBuilder.mr(target, output)` in the Python bindings |
| simulator | a Z-basis branch in `sim/sim.h`: `bits[t] = (bits[t] & ~cond) \| (q & cond)`, `qubits[target] &= ~cond`, phase untouched |
| statistics | its own `MR` group; not Toffoli-like |
| `reaction_depth` | outcome depth from the target's bit-flip depth (`x_depth + 1`), target left at depth 0 |
| printing | `MR q%u b%u` (and ` if b%u`) in the file writer and `str`, `MR=b` in text diagrams, `MZ` box in the HTML diagram |

Semantics are pinned by the four checks §4 asks for:
`x(q); mr(q, b)` reports `b = 1` and `phase_flipped = 0` on 64/64 shots, while the same
circuit with `reset` reports 37/64 — the distinction the instruction exists for.

**Measured on the rebuilt extension** (`scripts/run_mr_gated_arms_kickmix.py`, which
builds the arms with a real `kickmix.CircuitBuilder` and simulates them with a real
`kickmix.Simulator`):

| Instance | Result |
|---|---|
| m=15, n=6, GF(2⁴), 8 shots | `none`, `swap`, `sweep`, `forney`, `clear`, `shift`, `quotient`, `all`, `full`: reference-identical syndromes and phase, syndromes restored, every non-syndrome qubit `|0>` |
| m=255, n=14, GF(2⁸), 16 shots | `all`: PASS on the same four criteria |
| m=4095, n=70, GF(2¹²), 4 shots (the official instance) | `all`, `full`: PASS; reference 1,893 qubits, arm 1,894 allocated ids, phase matches (1 one each) |
| falsifier `--clear-with r` | FAIL (phase 6 ones vs 3, junk dirty) |
| falsifier `--shift-under` | FAIL (syndromes not restored) |

This is correctness, not a score. The extension is a local build: no receipt, and
`apedp_dyn/quantum_compilation/contracts.py` still pins the official binary, which has
no `MR`. The T-count reduction in §1 still needs the upstream rebuild plus the official
verifier.

### 6a. What the gated circuits actually cost

Correctness is not the projection. `scripts/measure_mr_arm_costs_kickmix.py` prices the
same circuits with `kickmix`'s conditional-aware per-shot counting (record:
`benchmarks/quantum_compilation/kickmix_mr_arm_costs.json`, official instance, 64 shots,
seed 20261028):

| Stage | Measured `T_mean` | §1 projection | Measured ratio | Projected ratio | Gap |
|---|---|---|---|---|---|
| reference | 5,725,071.6 | 5,725,071.61 | 1.000× | 1.000× | −0.00% |
| Forney gate | 4,298,268.3 | 4,298,955.87 | 1.332× | 1.332× | −0.02% |
| + clear EEA | 2,157,906.3 | 2,158,651.45 | 2.653× | 2.652× | −0.03% |
| + sweep gate | 1,590,308.3 | 1,590,938.28 | 3.600× | 3.599× | −0.04% |
| + swap gate | 1,456,286.3 | 1,457,103.45 | 3.931× | 3.929× | −0.06% |
| + shift gate | 1,381,390.0 | 1,382,207.08 | 4.144× | 4.142× | −0.06% |
| + quotient gate (§6b) | **1,325,230.0** | — (no projection) | **4.320×** | — | — |

The reference matching the pinned row to 0.01 Toffoli says the rebuilt builder prices
identically, so the whole comparison is one measurement against another. Every stage now
lands within 0.06% of its projection, and the residue is the same on every row: the
gated circuits are a couple of Toffolis *cheaper* per live position than the oracle
build, because the oracle still pays two `degree - 1`-Toffoli equality tests to produce
`sigma(alpha^j) == 0` where the implemented arm reads that value out of `sigma` (0
Toffoli). Per-lever, alone: `swap` 1.049×, `sweep` 1.247×, `forney` 1.332×, `clear`
1.597×, `shift` 1.027×, `quotient` 1.279×; the shift lever inside the chain is worth the
5.14% the oracle accounted for it (1,456,286.3 → 1,381,390.0).

Three defects had to be fixed before the gates delivered what they claimed, and the
first measurement (3.556×) is preserved in git history rather than in the record:

| Defect | Effect | Fix |
|---|---|---|
| the shift gate's `¬flag` was a second `bit_store(…, True)` instead of a store of 0 under the flag, so the bit was constant and the gate fired on `c0 ∨ c1 ∨ c2 ∨ c3` | the lever skipped only rounds with every control clear: 1.002× alone, 0.23% of `T` | store 0 under the flag |
| the sweep gate wrapped only the position sweep, leaving the quotient divisions and the `n_a` add/undo outside it | the +10.0% overshoot on that row, all of it work that cancels in a dead round | gate the whole step on `c0 ∨ c3` |
| the Forney predicate was an 11-Toffoli equality circuit run twice per position, plus a copy qubit allocated per position | the +2.3% overshoot on the Forney row and 4,095 leaked qubit ids | read `sigma(alpha^j)` with `mr`, restore it with classically controlled `x`, and assemble the predicate from two classical stores |

Together they take the full stack from 1,609,925.7 to 1,381,390.0 per shot.

### 6b. The sixth lever, and the qubits the reads no longer cost

Two further changes, measured the same way (same record, same 64 shots):

* **Quotient gate.** The sweep gate establishes that a round is live when `c0 ∨ c3`;
  the four counter bits are a one-hot decode, so at most one of the two is set. Each of
  the sweep's two polynomial divisions is consumed by an add controlled on its own
  counter bit, so in every live round exactly one division is executed under a false
  add — it computes a quotient nothing reads. `accumulate_quotient` now pushes that bit,
  so the dead division is skipped rather than computed and then uncomputed. Worth
  **4.07%** of the five-lever count, 1,381,390.0 → **1,325,230.0, 4.320×**; as an arm
  alone 1.279× (4,477,555.6). It has no projected row because the oracle build priced
  both divisions as live.
* **Shared read ancilla.** Each read was allocating a fresh copy qubit, and the
  sweep's `q_i` was not freed at all, so the id count grew by roughly a register per
  round: the five-lever arm allocated 4,503 ids against the reference's 1,893. One
  ancilla now serves every read — `mr` clears its target, so reuse is free — and `q_i`
  is freed in the round that allocates it: both stacks allocate **1,894** ids, one more
  than the reference, and the extra id is the shared ancilla.So the measured statement at the official instance is **4.320×** at `Q` = 1,894, and it is
now **4.652×** with the 54-AND field multiply of §7 at `Q` = 1,897; the 4.142× is no longer
an upper bound the design cannot reach, since it is what the five gates cost, to within the
0.06% above, and the sixth lever goes below it. What still holds
from §5 is the part that does not depend on this: 1.325M per shot is not the 445,229
Benchmark II needs — it is 2.98× above it — and the reads cost one qubit, not the 2,610
they used to leak. What is left is the field multiply itself, and after §7 it is **59.48%** of this arm's
executed Toffolis (788,250.0 of 1,325,230.0 per shot, from 10,510 executed calls of 75
Toffolis), not the 86.33% an earlier ablation reported. `docs/CIRCUIT_OPTIMISATION_RULES.md`
§5 records the correction and prices the rest.

## 7. The multiply is 54 ANDs now, measured

`ixor_gf2_mul` is the Karatsuba recursion to the gate — 63 ANDs at GF(2¹²), matching
`K(12)`, with no recursion split beating it. What beats it is a better *base case*:
`mu(2) = 3` and `mu(3) = 6` were established by exhausting the bilinear search, and
`12 = 3 × 3 × 6 = 54`. `scripts/gf54_mul.py` emits that algorithm as gates and
`scripts/measure_gf54_mul_kickmix.py` prices it (`benchmarks/quantum_compilation/gf54_mul_kickmix.json`):

| | built-in | 54-AND |
|---|---|---|
| Toffolis per controlled call | 75 (63 + 12) | **66 (54 + 12)** |
| six-lever arm, `T_mean` per shot | 1,325,230.0 | **1,230,640.0** |
| ratio against the reference | 4.320× | **4.652×** |
| `Q` | 1,894 | 1,897 |

The saving is 7.14%, not the 14.3% the AND count suggests and not the 12.3% a projection
that scaled the whole 75 would give: the 12-Toffoli controlled add is in both circuits. The
emitted multiply was checked on random basis states (values, restored operands, clean
workspace, untouched phase) before the arm was run with it, and the arm passes the same
six-lever verification as §6.

That emitter is a Python gate stream, and it reaches exactly one instruction. The same
multiply is now also a builder policy in the `MR` patch — `gen_gf54_mul` behind
`builder.gf_product = 'and54'` — so it reaches the inversion (315 → **270**) and the
division (378 → **324**) as well as the multiply (75 → 66 controlled).
`scripts/measure_gf54_native_opcode.py` prices it (`gf54_native_opcode.json`): the
**reference** decoder goes **5,725,071.6 → 5,201,222.5 per shot (9.15%)** at `Q` 1,893 →
1,912, and the six-lever arm **1,325,230.0 → 1,221,466.8 (4.687× against the unchanged
reference; 4.258× against a reference carrying the same opcode)**. Of the reference's
523,849.1 Toffolis saved per shot, **35.2% (184,225.1) is the Forney inversion block** —
20,469.5 executed products across 4,093.9 of the 4,095 positions — measured by flipping the
policy around `init_gf2_inverse_with_scaffold` and `gf2_phase_by_product` only, not inferred
from the AND count.

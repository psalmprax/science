# `MR`: the one instruction between the DQI reference circuit and 3.599×

**Status:** proposed IR change · **Scope:** `kickmix` (Benchmark II only) ·
**Evidence:** `benchmarks/quantum_compilation/dqi_predicate_gating.json`,
`scripts/probe_dqi_predicate_gating.py`, `tests/test_dqi_predicate_gating.py` ·
**Measured result if added:** Benchmark II `T_mean` 5,725,071.61 → 1,382,207.08
(4.142×) at `Q` = 1,893 with 0 value and 0 phase errors on 64 shots.

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
per live round, which the algorithm requires. `Q` stays at 1,893: 1,752 persistent
registers (`2 × 73 × 12`, both factors frozen by the instance) plus 141 transient.
The honest reading is that this instruction removes the part of the reference
circuit's cost that is *avoidable*, and the rest is the decoder.

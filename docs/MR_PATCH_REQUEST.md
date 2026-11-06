# Change request: add `MR` (Z-basis measure-and-reset) to kickmix

**Target:** the `kickmix` extension built for Benchmark II · **Spec:**
[`docs/MR_INSTRUCTION_SPEC.md`](MR_INSTRUCTION_SPEC.md) · **Size:** one opcode,
one builder method, one simulator branch.

## What this buys

| | reference | with `MR` | factor |
|---|---|---|---|
| official `T_mean` | 5,725,071.61 | **1,382,207.08** | **4.142×** |
| `Q` | 1,893 | 1,893 | — |
| value / phase errors | 0 / 0 | **0 / 0** (64 shots) | — |
| extra classical bits | — | 3,135 | not scored |
| extra qubits / Toffolis | — | 0 / 0 | — |

The four levers it unlocks, in the order they were measured:

| Lever | Factor | Cumulative |
|---|---|---|
| skip the Forney body where `sigma(alpha^j) != 0` | 1.332× | 1.332× |
| clear the EEA instead of running its adjoint | 1.994× | 2.652× |
| skip the multiply sweep in rounds where `c0 = c3 = 0` | 1.357× | 3.599× |
| skip the register swap in rounds where `c3 = 0` | 1.092× | 3.929× |
| skip the shift step on its true liveness | 1.054× | **4.142×** |

Scaled instances agree: 2.998× at m=15/GF(2⁴), 3.709× at m=255/GF(2⁸), 4.142× at
m=4095/GF(2¹²). Receipt arithmetic: the DQI receipt moves 24.439T → 5.900T and the
portfolio score 2.361982 → 3.79. Benchmark II's 12.86× target is still 3.10× away — this
request closes the avoidable part, not the target.

## Why it cannot be done in the IR as it stands

`push_condition` accepts only a classical bit, and the only readout today is `hmr`, an
**X-basis** measurement — on `|1>` it returns a random bit. `push_condition(qubit)` and
`bit_store(control=qubit)` are rejected, and no Hadamard can be synthesized to conjugate
the predicate into X. `R` clears a qubit but randomizes the tracked phase: a phase-free
circuit `x(q); reset(q)` reports a random `phase_flipped` on 11/16 shots. `MR` is the
missing operation, and the design needs it in three places (predicate read, scratch
clear, per-round control read).

## Diff summary

Against the API surface observable from `kickmix.so`; no source tree is checked out here,
so the file names are the components rather than paths. Line counts are estimates for
review sizing.

| Component | Change | ~size |
|---|---|---|
| instruction enum / `.kmb` opcode table | add `MR`, keep `R` and `HMR` untouched | +2/−0 |
| `.kmb` formatter and parser | `MR q%u b%u` (optionally `if b%u`, matching `HMR`) | +12/−2 |
| `CircuitBuilder` bindings | `mr(target, output)` mirroring `hmr`'s coercion and broadcast | +18/−0 |
| simulator | Z-basis measurement into the same classical state `HMR` writes, reset to `|0>`, **no effect on `read_phase_flipped_across_shots`** | +25/−0 |
| `operation_statistics` | count `MR` under its own key, never `num_toffoli_like` | +2/−0 |
| docstring / `__doc__` examples | `mr` entry beside `hmr`, including the phase contract | +20/−0 |
| verifiers | **none** — `verify_submission_dqi.py` and `verify_submission_invsqrt.py` need no change | +0 |

Reference sketch for the simulator branch, which is the only subtle part:

```python
def mr(self, target, output):
    """Measure Z, write the outcome, clear the qubit. Phase must survive."""
    for q, b in broadcast(target, output):
        outcome = self._read_z(q)          # deterministic: states are Z-basis up to phase
        self._write_bit(b, outcome)
        self._reset_z(q)                   # X if outcome, but no phase bookkeeping
```

The one line that makes or breaks it: `_reset_z` must not touch the frame that
`read_phase_flipped_across_shots` reports. Implementing `MR` as `R` followed by copying
the bit out of `R`'s discarded measurement will reproduce the 11/16 random-phase failure.

## Test plan

Four acceptance tests, all already mechanised:

1. `python scripts/probe_dqi_predicate_gating.py --m 15 --n 6 --degree 4 --shots 64`
   → `T` 33,232.22 → 11,084.36 (2.998×), `value_errors = phase_errors = 0`.
2. Same probe at `--m 255 --n 14 --degree 8 --shots 512` → 3.709×, 0/0.
3. Phase regression: `x(q); mr(q, b)` must report `phase_flipped = 0` on every shot,
   where `x(q); reset(q)` reports a mixture. This is the single test that separates
   `MR` from `R`.
4. `pytest tests/test_dqi_predicate_gating.py -q` → 12 passed, 1 skipped (the skip is
   the kickmix-interpreter re-measurement; run it under the built interpreter to make it
   12/0).

## Risk and rollback

- **Risk:** a miscompiled `MR` that clears like `R` is *silent* — the circuit still
  returns the right syndromes and fails only on the phase, which is half the shots. Test
  3 exists for exactly this.
- **Risk:** counting `MR` as Toffoli-like would make every arm above look worse than the
  reference; the `operation_statistics` change is one key.
- **Rollback:** the opcode is additive. Removing it restores the current instruction set
  and every existing receipt still verifies; nothing in the repository is built on `MR`.

## What I need to write the actual patch

The repository ships `kickmix.so` and `LICENSE.txt`, not the sources, so the diff above is
against the observable API. Point me at the source tree (path or clone URL) and I will
turn the table into a real patch with the tests wired in; until then the numbers here are
oracle upper bounds, not receipts.

## What is actually missing (two things, neither of them analysis)

1. **The source tree.** `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/`
   contains `kickmix.so` and `LICENSE.txt` and nothing else. A compiled extension cannot be
   patched: the opcode table, the `.kmb` parser, the builder bindings and the simulator
   branch all live inside the binary. The licence is Apache-2.0, so a patch is permissible
   — it is not available to write. Everything in this request is therefore written against
   the API surface observable from the extension.
2. **A channel.** There is no upstream remote, issue tracker or maintainer contact in this
   repository, so nothing has been filed. This document is the request; filing it is a
   delivery step, not an analysis step.

## Why a replacement circuit designer does not close this gap

Building our own builder/simulator that has `MR` would let the gated arms run instead of
being supplied an oracle truth table, which is real work, but it cannot produce a score.
The contract in §6 of [`PHOTONIC_ACCELERATOR_BOUNDARY.md`](PHOTONIC_ACCELERATOR_BOUNDARY.md)
accepts only the official `kickmix` compilation plus `verify_submission_dqi` /
`verify_submission_invsqrt`, and `apedp_dyn/quantum_compilation/contracts.py` pins both by
SHA-256 before an official run. A local designer's T count is a surrogate, and a surrogate
may rank candidates but must never be written into a receipt. So a replacement engine
changes which numbers we *can measure*, not which numbers *count*; until the opcode
exists upstream, 4.142× can only be measured on a local rebuild, which is a surrogate.

If the goal is instead to make the arms executable for correctness testing (not scoring),
a local simulator with `MR` was the cheaper route, and it now exists:
[`apedp_dyn/quantum_compilation/local_mr_sim.py`](../apedp_dyn/quantum_compilation/local_mr_sim.py).
It reproduces `kickmix` shot for shot on the reference decoder, and
`scripts/run_mr_gated_arms.py` runs all six levers with the predicate read by a real
`mr` rather than an oracle. Each arm reproduces the reference's values **and** phase at
m=15/GF(2⁴) and at m=255/GF(2⁸), while the same arms read with `r` fail on the phase;
the shift arm additionally fails when its gate drops the `¬flag ∧ (c0 ∨ c3)` branch —
see `benchmarks/quantum_compilation/local_mr_arms.json`.

What that does and does not change: the levers' **correctness** is no longer untested,
but their **cost** still cannot be scored here, because only the official `kickmix`
build plus the official verifiers produce a receipt. The 4.142× is now a measured
Toffoli count on a local rebuild — 4.144× for the five oracle-gated levers and **4.320×**
with the quotient gate, at `Q` = 1,894 against the reference's 1,893
(`docs/MR_INSTRUCTION_SPEC.md` §6a–§6b) — a surrogate, not a receipt, until `MR` exists
upstream; this section's request stands unchanged.

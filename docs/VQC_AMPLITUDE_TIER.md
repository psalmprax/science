# The amplitude tier: what the read-then-gate licence actually covers

**Status:** experimental module + measured boundary · **Scope:** the
read-then-gate gadget every gate in the DQI arm is built from, at the sizes where
amplitudes are representable · **Evidence:**
`benchmarks/quantum_compilation/vqc_licence_boundary.json`, produced by
`scripts/measure_vqc_licence_boundary.py`; module
`apedp_dyn/quantum_compilation/statevector_sim.py`.

Every Toffoli figure in this repository is a **basis-state** figure. The engine
behind it, `kickmix`, stores computational-basis states; it is exact for the job
it was built for, because the decoder's input *is* a basis state. What it cannot
say -- and what the schedule campaign's licence quietly depends on -- is what the
arm's gates do to anything else. This note measures that boundary instead of
asserting it, and it is careful about what the measurement is worth: the tier is
an exact statevector simulator for *gadgets*, not a competition engine, and the
arm is nowhere near representable in it.

## 1. Why a basis-state count needs a licence at all

`docs/EEA_SCHEDULE_CAMPAIGN.md` §2.1 licenses all five gates on **P1**: every
register the campaign reads holds a computational-basis value at the point it is
read. That is a claim about the *input ensemble* (a sampled syndrome plus a
classical MBUC register), not about the circuit. Given P1, a read is
deterministic, the source is untouched, and skipping the work under a false
predicate is the identity — which is why the executed Toffoli count can fall
without the circuit changing what it computes.

P1 is true of the benchmark's inputs and false in general. The consequence is
that the arm is a *different map* from the reference outside the basis, and the
number 686,014.8 is a count for basis inputs. Both halves of that sentence are
measurable, and they are what this tier measures.

## 2. The gadget, and the two maps

The gate the whole campaign rests on replaces a controlled operation

```
reference:  cswap(control, x, y)
```

with

```
gated:      cx(control, ancilla); mr(ancilla -> bit); bit ? swap(x, y) : nothing
```

where `mr` is a Z-basis measurement and reset: the ancilla is cleared and its
value reported, so `control` and the tracked phase are untouched. Four qubits:
the control, the readout ancilla, and the exchanged pair.

| | reference | gated |
|---|---|---|
| instructions | 1 controlled swap | 1 `cx`, 1 `mr`, 1 conditional swap |
| branches | 1 (unitary) | 2 on a superposed control |
| Toffolis | 1 per exchanged qubit | 0 for the read, the same for the swap when it fires |

## 3. On every basis input, the same state

Eight inputs (control, left, right ∈ {0,1}, ancilla = 0), compared state by
state:

- all eight are **equal up to global phase**;
- the classical bit equals the control in all eight;
- the ancilla returns to |0> in all eight, so it is reusable.

That is P1 as a measurement rather than as a precondition: on the ensemble the
benchmark actually feeds the arm, the gated map and the controlled map are the
same computation, and the basis-state counts are counts of *that* computation.

## 4. Outside the basis, a different map — and the size of the difference

Input: control in |+>, pair = |1>|0>. The reference entangles the control with
the pair and leaves the control coherent. The gated map measures the control,
so it branches:

| quantity | reference | gated |
|---|---|---|
| branches | 1 | **2**, at 0.500 / 0.500 |
| control off-diagonal (|⟨ψ₀|ψ₁⟩|) | 0.500 | **0.000** |
| fidelity of the mixture against the reference state | — | **0.500** |

The fidelity is exactly 1/2, and the off-diagonal is exactly 0: the gadget is a
**complete dephasing of the control**, not a small perturbation. There is no
parameter to tune — the read is a measurement, and a measurement of a
superposed control destroys the coherence the coherent gate preserves.

So the honest form of the campaign's claim is: *on the benchmark's basis-state
inputs the gated arm is the reference computation, and its executed-Toffoli
saving is a saving on that computation; off that ensemble the arm is a different
map, and the count is not a count of the reference.* A submission that claimed
the gated counts for a superposition input would be claiming a different circuit.

## 5. `HMR` is not `MR`, and `H` then `HMR` is — which is why the builder matters

Experiment D asks whether the pinned competition binary can compile the arm. The
obvious substitution for the missing `mr` is the `hmr` it *does* have: `hmr` is
`H` followed by a Z-basis measurement and reset, so `H` followed by `hmr` is
`H·H·MR = MR`. Measured on this tier:

| map | on a basis input | on a superposed input |
|---|---|---|
| `hmr` alone | **two branches at 0.500 each** — it randomises a basis-state qubit | same |
| `h; hmr` | one branch, **identical to `mr`** | **identical to `mr`**, branch for branch |
| `h; h; hmr` | identical to `mr` (the extra H is the identity) | same |

The algebra is exact: the substitution is correct, and it needs a Hadamard gate.
The pinned builder has `hmr` and no `h`, so the identity holds and cannot be
emitted — which is the fact Experiment D records, and the reason this tier does
not rescue the receipt.

## 6. What the tier cannot do, in arithmetic

| register | qubits | amplitudes | log₁₀ amplitudes |
|---|---|---|---|
| the arm | 1,913 | **576 decimal digits** | 575.87 |
| the reference | 1,893 | 570 digits | 569.85 |

A dense amplitude-level verification of the arm would need one complex number per
basis state of a 1,913-qubit register — 576 digits of them, at 16 bytes each.
`dimension_report` refuses to give that a float and reports the log instead. This
is why the tier's reach is the gadget, and why every count in this repository is
a basis-state count: the alternative is not a bigger machine, it is a different
question.

## 7. Cross-check, so the tier is not self-certifying

The statevector tier is validated against `kickmix` on the gated gadget's eight
basis inputs: identical output register values on all eight, on the local `MR`
rebuild (the only engine that can execute the gadget at all — the pinned binary
cannot). Agreement is recorded in the JSON, not asserted here. Where the two
engines can both run, they agree; where they cannot, the statevector tier is the
one that can, and its scope is stated in §6.

## 8. Pins

`tests/test_statevector_licence.py` pins: the tier's gate algebra (`x`, `cx`,
`ccx`, `cswap`, `h`, `cz`, measurement-and-reset), the eight-input equality, the
0.500 fidelity and 0.000 off-diagonal on the superposed control, `hmr`'s
randomisation and `h; hmr == mr`, the refusal above 26 qubits, and the recorded
boundary (`benchmarks/quantum_compilation/vqc_licence_boundary.json`).

A future edit that claims the arm's counts for a superposition input, or that
calls the gated map "equivalent" without the basis restriction, fails there.

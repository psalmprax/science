# APEDP-VQC Core: the first acceptance milestone, measured

**Status:** experimental · **Scope:** a versioned circuit IR, an exact statevector
backend, a stabilizer backend, density-matrix noise channels, the bit- and
phase-flip repetition codes at any odd distance, the perfect five-qubit code, a
unitary/phase equivalence referee with a Clifford branch above the dense bound, a
decoder interface with a second (minimum-weight matching) implementation, a
discovery admission layer, a certified Clifford rewrite search that runs the whole
loop at 100 qubits, and adjudication receipts · **Evidence:** `apedp_dyn/vqc/`,
`tests/test_vqc_core.py` (63 tests), `tests/test_vqc_codes.py` (46),
`tests/test_vqc_equivalence.py` (20), `tests/test_vqc_clifford_equivalence.py` (19),
`tests/test_vqc_stabilizer.py` (40), `tests/test_vqc_decoders.py` (18),
`tests/test_vqc_discovery.py` (13), `tests/test_vqc_clifford_discovery.py` (15),
`tests/test_vqc_qasm.py` (15), `tests/test_vqc_stabilizer_scaling.py` (7),
`tests/test_vqc_synthesis.py` (56), `tests/test_vqc_placement.py` (34),
`tests/test_vqc_external_comparison.py` (51), `tests/test_vqc_corpus.py` (11),
`tests/test_vqc_foreign_referee.py` (11), `tests/test_vqc_marginals.py` (26),
`tests/test_vqc_memory.py` (9),
`scripts/measure_vqc_stabilizer_ghz.py` and its record,
`scripts/compare_vqc_external_tools.py` and its record,
`scripts/measure_vqc_stabilizer_scaling.py` and its record,
`scripts/measure_vqc_clifford_discovery.py` and its record,
`scripts/measure_vqc_synthesis.py` and its record, and the Phase 0 inventory in
[`VQC_PHASE0_CAPABILITY_INVENTORY.md`](VQC_PHASE0_CAPABILITY_INVENTORY.md); this note
is the record of what they measure.

The DQI work is one workload: a real circuit, resource-accounted to the gate and
verified by a competition's own verifier. It is not a platform. This milestone is
the smallest piece of machinery that *is* one — a circuit language every component
can share, simulators that actually evolve a state (dense, and stabilizer past the
dense tier's reach), noise that can be checked analytically, codes that survive a
single error with a prediction written down first, a referee that compares two
circuits as unitaries instead of as distributions, and a receipt that says whether
a claim stands.

Everything below is measured by the test suite named at the end of each section.
Nothing in this note is a projection.

## 1. The representation core

`apedp_dyn/vqc/circuit.py` is the IR. A circuit is a schema version, numbered
qubits and classical bits, declared ancillas, optional metadata, and a tuple of
operations:

| operation | what it carries |
|---|---|
| `Gate(name, targets, params, condition)` | a name from one gate table, its target order (controls first), parameters, and an optional classical condition |
| `Measure(qubit, bit, condition)` | Z-basis measurement **with reset** — the competition's `MR` semantics — writing the outcome to a classical bit |
| `Reset(qubit, condition)` | return a qubit to `|0>` and discard the outcome; not the same operation as `Measure` |
| `Noise(qubit, kind, probability, condition)` | a stochastic single-qubit error, executed per shot by the backend |

`condition` is a conjunction of `(bit, expected)` pairs, which is what makes
teleportation and syndrome decoding expressible without a control-flow language:
`Z` on the target iff bit 0 measured 1.

Validation is not advisory. It refuses an unknown gate, wrong arity, a repeated
qubit, an out-of-range qubit or bit, a condition read before the bit is written, a
non-finite parameter, an out-of-range probability, an unknown noise kind, an
unknown schema version, and a circuit with no qubits. `circuit.validate()` is
called on every construction path and by every backend.

Identity is canonical: `to_json()` sorts keys, drops whitespace, and fixes float
repr, so `Circuit.from_json(c.to_json()).to_json() == c.to_json()` and
`sha256()` is stable. The tests execute the original and the reconstruction at the
same seed and require identical counts **and** identical per-shot samples.

Reset is its own operation rather than a disguised measurement: `X; Reset; Measure`
reads `0` on every shot, and resetting one half of a Bell pair leaves the other
maximally mixed (0.503 / 0.497 at 4,096 shots) instead of leaving it in `|0>` — the
reset map cuts the correlation without collapsing the partner.

### 1.1 Interchange

`apedp_dyn/vqc/qasm.py` imports and exports **OpenQASM 2**, and refuses what the
format cannot carry rather than approximating it:

| construct | behaviour |
|---|---|
| `h x y z s sdg t tdg rx ry rz u1 cx cz swap ccx cswap`, `measure`, `reset` | round-trips with the same operations, metadata, ancilla contract and hash |
| `if(c==k)` | round-trips when the condition tests **every** declared bit |
| a condition on a subset of bits | **refused**: `if(c==k)` compares the whole register, and `if (c[1] == 1)` is QASM 3 |
| noise operations | **refused**: OpenQASM 2 has no noise construct |
| user-defined `gate`, `u2`/`u3`, malformed parameters | **refused** with the reason |

Teleportation is the instructive case: its canonical JSON round-trip is exact, and
its QASM export is refused because two of its gates are conditioned on one bit of a
two-bit register. Metadata and declared ancillas travel as `// vqc-...` header
comments — a convention of this module, not the standard, and what lets a QASM
round-trip preserve the hash instead of dropping the ancilla contract.

## 2. Exact simulation

`StatevectorBackend` runs a dense state through
`apedp_dyn/quantum_compilation/statevector_sim.py` (which already implemented `MR`
collapse-and-reset and classical bits), with the gate set completed in
`apedp_dyn/vqc/gates.py` and gate application shared by every backend.

| experiment | independent check | measured |
|---|---|---|
| Bell | state equals `(|00> + |11>)/sqrt(2)` up to global phase; outcomes perfectly correlated at 50/50 | passes at 4,096 shots |
| GHZ | only `000` and `111`, each 0.5 | passes at 4,096 shots |
| Teleportation | **every shot's** post-state equals `|00> ⊗ RY(π/4)|0>` up to global phase | 64/64 shots exact, and X-basis `P(0)` = 0.85596 against an analytic 0.85355 (worst outcome z = 1.98) |

Teleportation is the test that measurement and feedforward are real rather than
decorative: with the conditional `Z` removed, the two syndrome branches disagree
on phase and `P(0)` collapses to **0.500** (analytic coin flip), which the
adjudicator rejects at z > 60.

## 3. Noise

`apedp_dyn/vqc/noise.py` is the exact ensemble tier: `rho -> Σ K_k rho K_k†` for
bit flip, phase flip, depolarising, amplitude damping and dephasing, with
`is_valid()` recomputing Hermiticity, unit trace and `min eigenvalue >= -1e-9`
rather than assuming them.

| check | analytic expectation | measured |
|---|---|---|
| depolarising one half of a Bell pair, `p` | fidelity to the Bell state `1-p`; purity `(1-p)² + p²/3` | matches to 1e-12 on `p ∈ {0, 0.05, 0.2, 0.5, 1}` |
| bit flip on `|0>` at `p` | `P(1) = p` | 0.29999999999999993 at p = 0.3 |
| amplitude damping on `|1>` at `γ` | `P(0) = γ` | 0.4 at γ = 0.4 |
| dephasing `|+>` at `p` | purity `1 - p/2` | 0.75 at p = 0.5 |
| every channel, every `p` | Kraus completeness `Σ K†K = I` | matches to 1e-12 |

The IR's `Noise` operation is a **trajectory** model instead: the backend samples
the Pauli error per shot. That is exact for the ensemble and sampled per shot, so
it carries its seed and shot count. Measured: a bit flip at `p = 0.2` on one half
of a Bell pair leaves the pair correlated with probability 0.7979 ± 0.0045
(expected `1 - p = 0.8` at 8,000 shots); the same seed reproduces the run counts
exactly, a different seed does not.

## 4. A logical qubit

`apedp_dyn/vqc/logical.py` builds the distance-3 repetition code *as a circuit*:
encode with two CNOTs, optional deterministic or stochastic bit flips, two parity
syndromes into ancillas 3 and 4 (measured, so the ancillas are reset), a
classically conditioned correction per syndrome, and a three-bit readout.

| injected error | syndrome | readout after correction |
|---|---|---|
| `X` on qubit 0 | `10` | `000` (corrected) |
| `X` on qubit 1 | `11` | `000` (corrected) |
| `X` on qubit 2 | `01` | `000` (corrected) |
| `X` on qubits 0 and 1 | `01` | **`111`** — miscorrected into a logical flip |

Measured logical failure rates at 8,000 shots, seed 20261030, against the exact
prediction `3p² - 2p³`:

| `p` | measured | standard error | exact | uncorrected |
|---|---|---|---|---|
| 0.05 | 0.006625 | 0.000907 | 0.00725 | 0.05 |
| 0.10 | 0.024875 | 0.001741 | 0.028 | 0.10 |

**What this is not.** No threshold is claimed: one distance cannot demonstrate
one. Syndrome extraction is assumed perfect (noiseless ancillas, ideal readout),
the code protects only against bit flips, and the failure formula is the exact
combinatorial prediction for *this* code, not a hardware forecast. Those limits
are in the module docstring and in every measured row's `as_dict()`
(`"syndrome_extraction": "perfect-ancilla"`, `"protects_against": "bit-flip only"`).

### 4.1 The phase-flip code: the same code, transposed

The conjugate of the above: encode in the X basis, measure `X_i X_j`, correct with
`Z`s. It shares the arithmetic on purpose — the same `3p² - 2p³` prediction, the
same syndrome table with the roles of `X` and `Z` exchanged — so a disagreement
between the two would be an implementation error rather than a new result.

| injected error | syndrome | readout |
|---|---|---|
| `Z` on qubit 0 / 1 / 2 | `10` / `11` / `01` | corrected on **every** shot, both preparations |
| `Z` on qubits 0 and 1 | `01` | misdiagnosed as a single error on qubit 2, logical value flipped |

At `p = 0.1`, 4,000 shots, seed 20261030: measured **0.023** ± 0.0024 against the
exact 0.028 (uncorrected 0.10). At `p = 0` it is 0 failures / 512 shots; at `p = 1`
every qubit is flipped and it reports > 0.6, because at saturation a code is not a
code.

### 4.2 The perfect five-qubit code

`apedp_dyn/vqc/five_qubit.py` is a real distance-3 code rather than a repetition
code: four stabilizers `X Z Z X I`, `I X Z Z X`, `X I X Z Z`, `Z X I X Z`, four
ancillas, an encoder that projects `|00000>` onto the code space with a conditional
Pauli frame, syndrome extraction, conditional correction, and the logical `Z`
(`ZZZZZ`) read as the parity of all five data qubits.

The prediction is **not a fit**: `exact_failure_probability(p)` enumerates all
`4**5 = 1024` Pauli patterns, applies the decoder to each, and weights the ones
whose residual anticommutes with `Z_L`. At `p = 1` it is 0.5432098765 — a rational
with denominator `3**5`, which is what summing weights over 1024 patterns gives and
what no curve fit would produce.

| `p` | shots | measured | standard error | exact (enumerated) |
|---|---|---|---|---|
| 0.05 | 2,000 | 0.0130 | 0.0025 | 0.014888 |
| 0.10 | 2,000 | 0.0540 | 0.0051 | 0.053005 |

Structure checks, all measured rather than asserted: all **15** single-qubit Paulis
map to distinct nonzero syndromes out of the 16 patterns (the code is perfect —
`2**4 = 16 = 1 + 15`), the decoder returns exactly the injected error for each, the
encoder prepares the exact encoded state on every shot with all four ancillas back
in `|0>`, and a weight-2 error (`X` on one qubit, `Z` on another) does flip the
logical value for some pairs — a distance-3 code is not a distance-5 code.

**What this is not.** Depolarising noise, perfect syndrome extraction, and a
single logical observable; the exact figure is the closed form for *this* channel
and *this* decoder. The `as_dict()` of every run says so
(`"counts_phase_errors": false`, `"syndrome_extraction": "perfect-ancilla"`).

## 5. Equivalence: one global phase free, nothing else

`apedp_dyn/vqc/equivalence.py` materialises both unitaries — column `j` is the image
of basis state `j`, computed through the same gate application the backends use, so
this is the circuit's map and not a second implementation of it — and compares them
by the Frobenius distance to the closest global-phase multiple, normalised per
amplitude:

```
distance = min_phi ||U_c - e^(i·phi) U_r||_F / sqrt(2^n),   phi = arg <U_r, U_c>
```

That minimum is attained in closed form — `||U_c - e^(i·phi) U_r||_F^2 = 2·2^n −
2·Re(e^(i·phi)<U_r, U_c>)` — so the reported phase *is* the minimiser, and the value
takes its maximum `sqrt(2)` exactly when the two unitaries have zero
Hilbert–Schmidt overlap. A global phase is free by construction; every relative
phase changes the distance and is caught.

The maximum *single-entry* deviation at that phase is reported separately as
`entry_deviation`, and it is explicitly not a minimised quantity
(`"entry_deviation_is_minimised": false` in the receipt). Those are two objectives:
minimising the max-entry (infinity-norm) deviation picks a different phase in
general. An earlier version of this table reported a max-entry number taken at the
Frobenius-optimal phase and called it a minimum, which is why `CCX` versus `CCX; Z`
appeared as 2.0; the min-max value for that pair is `sqrt(2)`, and both metrics now
report `sqrt(2)` for it with the 2.0 entry deviation labelled as such. The verdicts
below did not change.

| candidate | distance | entry deviation | verdict |
|---|---|---|---|
| `CX` versus `H CZ H` | 0.0 | 0.0 | **VALID** |
| `RX(2π)` versus `I` | 0.0, phase π | 0.0 | **VALID** — one global phase is not an error |
| `CCX` versus `CCX; Z` | 1.414214 | 2.0 | **INVALID** |
| `CCX` versus `CX(0,2); CX(1,2)` | 1.224745 | 1.0 | **INVALID** |
| `CX` versus `SWAP` | 1.224745 | 1.0 | **INVALID** |
| `Z` versus `I` | 1.414214 | 2.0 | **INVALID** — orthogonal states on `|+>`, identical Z probabilities |
| a candidate on four qubits where the reference has three | — | — | **INVALID** |
| a candidate that measures and feeds forward | — | — | **INVALID**, with the reason |
| a candidate no tier can decide (20 qubits, non-Clifford) | — | — | **UNRESOLVED**, not INVALID |

**The `CCX; Z` row is the reason this module exists.** The two circuits return
identical Z-basis probabilities on every input — the test asserts that elementwise
*before* it asserts the rejection — and they are different maps, because the extra
`Z` is a relative phase. A distribution test passes that candidate; this one does
not, which is precisely the DQI phase-error failure mode at circuit level.

The other half of the DQI discipline is in the same receipt, as a *separate*
statement: `claims` names the resource measures the candidate is offered under, and
the report carries `claim_verdict: MET | UNMET | NOT_ASSERTED` beside the semantic
verdict. A candidate that computes exactly the reference unitary but claims
`operations: 1` while it has three is `verdict: VALID`, `claim_verdict: UNMET`,
`promotable: false` — "correct" and "better" are two facts, and `promote()` refuses
the second while leaving the first standing. Folding them into one INVALID verdict
would put the wrong statement in the ledger.

**Independence of the referee.** Both paths above run gate application, so a bug in
one gate rule would be invisible to a comparison of two circuits that use it.
`tests/test_vqc_equivalence.py` closes that loop with a second construction that
shares no code with either: gate matrices built by bit arithmetic on basis indices,
with no simulator and no tensor slicing. The suite requires the execution path's
unitary to equal the oracle's for random Clifford circuits on 1–4 qubits, and
requires the stabilizer tier's images to equal `U P U†` computed from the oracle's
`U`. This is an *independent* check of the small-system tier, not a second referee
for large ones: above ten qubits the only exact path is still the tableau one.

**When a global phase is *not* free: the controlled scope.** A global phase is free
for a standalone operation and not for a controlled subroutine, where it becomes a
relative phase between control branches. Both claims are available:
`adjudicate_equivalence(..., as_subroutine=True)` compares `diag(I, U_c)` against
`diag(I, U_r)` instead of the matrices themselves, which removes the phase allowance.
Measured: `RX(2π)` versus `I` is **VALID** standalone (distance 0, phase π) and
**INVALID** as a subroutine (distance `sqrt(2)`), while `CX` versus `H CZ H` is
VALID in both scopes because it is an identity and not merely a phase-equivalent.
The report carries `scope: standalone | controlled` and says which question it
answered. Above the dense bound the controlled question is `UNRESOLVED`: the
Clifford tier's conjugation images cannot see a global phase at all, so it cannot
decide whether that phase survives under control, and it is not asked to.

Refusal is explicit: `unitary()` raises `NotAUnitaryCircuitError` for a
measurement, a reset, a noise operation or a classically conditioned gate. A channel
has no unitary, and the report says so instead of comparing amplitudes that do not
exist. The dense matrix bounds this tier near ten qubits; the stabilizer tier below
covers the Clifford cases at sizes the matrix cannot reach.

## 6. A second backend: 100 qubits of GHZ, cross-checked

`apedp_dyn/vqc/stabilizer.py` holds `n` commuting signed Pauli generators instead of
`2**n` amplitudes, so a 100-qubit GHZ chain — `2**100` amplitudes, which is 576
decimal digits of them — fits in 100 strings of 100 letters. Phases are integers mod
four and every product, conjugation and sign comes from a table built once at import
by multiplying matrices and matching the result against `i**p · P`; a gate whose
conjugated Pauli is not a Pauli is refused rather than approximated.

Measurement is exact in both cases: if some generator anticommutes with `Z_q` the
outcome is a coin flip and the generators are rewritten so `±Z_q` replaces the one
that anticommuted; if none does, the outcome is already decided and is read off by
Gaussian elimination over GF(2) on the `2n`-bit characteristic vectors. The sign is
part of the answer, which is why the 100-qubit numbers below are numbers.

The referee for this tier is the dense tier, and the cross-check runs the **same**
circuit builder on both:

| check | dense tier | stabilizer tier |
|---|---|---|
| GHZ outcome probabilities, `n = 2..12` | the reference | 8,188 probabilities, max deviation **1.1e-16** |
| Pauli expectations, `n = 6` (both GHZ stabilizers plus 16 random strings) | the reference | 22 strings, max deviation **2.2e-16** |
| 100-qubit GHZ, Z basis | `2**100` amplitudes, i.e. nothing | 100 generators equal to the textbook set; only `0…0` and `1…1` at 0.5 each; 512 shots → 248 / 264 in about six seconds on this host (the record stores the measured wall clock) |
| 100-qubit GHZ, X basis | — | 2**99 even-parity strings exist; 128 shots, **0 collisions**, every observed string even |
| non-Clifford gate, or any `Noise` op | executes | **refused by name** |

The 100-qubit row is only usable because of the first two: the closed-form GHZ
predictions used at `n = 100` are the ones the dense tier confirmed at `n ≤ 12`, and
the test asserts the 100-qubit circuit is that same ladder gate for gate
(`large.ops[:len(small.ops)] == small.ops`).

What the X-basis row does **not** claim: 128 samples out of `2**99` strings are
consistent with the parity constraint the stabilizers impose and show no collisions,
which is what a uniform distribution over that support would look like. It is not
evidence that the simulator enumerates or represents all `2**99` outcomes — nothing
does, and the parity constraint is the whole test.

The record is
`benchmarks/quantum_compilation/vqc_stabilizer_ghz.json`, produced by
`scripts/measure_vqc_stabilizer_ghz.py`; it carries the deviations, the counts, the
timings, the checks and the limitations below.

**Boundaries.** Clifford gates only (`h x y z s sdg cx cz swap`); pure states only,
so every `Noise` operation is refused — a stochastic channel is not a stabilizer
state; and no amplitudes, so `keep_states=True` is refused and entropy or amplitude
readout is not available here. `Reset` follows the same one-trajectory semantics as
the dense tier (measure, then correct), which is right per shot and is not a mixed
state.

`StabilizerState.validate()` is the structural check the arithmetic cannot make on
its own, and it is `O(n²)` so it is a check for hand-built tableaus and tests rather
than a per-gate guard: it refuses generators that **anticommute** (no simultaneous
eigenspace, so there is no state) and generators that are **rank-deficient** (the
group has fewer than `2**n` elements, so the state is not pure). A dependency whose
phases do not cancel — the state would be its own negative — is already refused by
the elimination. The tests also pin that a repeated measurement returns the
collapsed outcome rather than re-randomising, that a `Y`-basis state (`|+i>`) is
undetermined in `Z` and yields a certain, valid branch on each trajectory, and that
`Y`/`±Y` expectations follow the `H`/`S` conjugations.

## 6.1 Scaling past 100 qubits: three families, measured

`scripts/measure_vqc_stabilizer_scaling.py` and its record
(`benchmarks/quantum_compilation/vqc_stabilizer_scaling.json`) measure 100, 200, 500
and 1,000 qubits in four circuit families. One family is not an experiment: GHZ has a
closed form, so the chain is checked against it entry for entry, the star is required
to be in the *same* stabilizer group by a different route, the ring graph state is
required to stabilise its own defining generators `X_q Z_(q-1) Z_(q+1)`, and the
random Clifford layers have no closed form at all, so they are checked structurally
(`validate()`, every generator an exact eigenstate) and by equivalence at scale.

| family, 1,000 qubits | gates | build | `validate()` | one Z sample | check |
|---|---|---|---|---|---|
| GHZ chain | 1,000 | 0.47 s | 0.43 s | 0.62 s | generators equal the closed form |
| GHZ star | 1,000 | 0.49 s | 0.61 s | 0.60 s | textbook generators all stabilise the state |
| ring graph state | 2,000 | 0.77 s | 0.45 s | 0.28 s | all 1,000 defining generators stabilise it |
| random Clifford layers | 7,500 | 14.2 s | 0.65 s | **127 s** | valid pure tableau, exact eigenstates |

The tableau costs `n²` labels: 1,000,000 bytes at 1,000 qubits, against
`2**1,000 ≈ 10**301` amplitudes for the dense tier. `validate()` is `O(n²)` symplectic
products on packed bit masks, so it is 0.4–0.6 s at 1,000 qubits rather than the 99 s
the per-qubit version took before the masks — the record is why that number exists.

The 127 s for a single Z-basis sample of a maximally mixed-looking random Clifford
state is the honest cost of the operation, not an error: a sample measures all 1,000
qubits and each measurement runs the GF(2) elimination, so sampling is where this
representation is expensive even though building it is not. GHZ and graph states
sample in well under a second because their measurement outcomes are almost all
determined.

**Equivalence at scale**, same record: a 7,500-gate random Clifford circuit at 1,000
qubits and the same unitary rewritten with cancelling `h h` / `cx cx` pairs (8,190
gates) are decided **VALID** from 2,000 Pauli images in 61 s, and the same circuit with
one `s` replaced by `sdg` is **INVALID** with the offending image named (`Z_0`, a sign
flip) in 54 s. No matrix is built at either size. Chain versus star is **VALID** at
state scope and **INVALID** at unitary scope at every size, which is the same
distinction the 100-qubit record shows, now reproducible at 1,000.

**What this does not claim.** Clifford only; pure states only, so every `Noise`
operation is still refused and sampled Pauli trajectories are not measured here; the
dense comparison stops where the dense tier stops; and wall clock on one host is a
measurement of this implementation, not a complexity theorem. 1,000 qubits of GHZ is
a reachable structured circuit, not a general 1,000-qubit simulator.

### 5.1 Above the dense bound: Clifford circuits, exactly

`adjudicate_clifford_equivalence` compares the *conjugation action* instead of a
matrix: the images of ``X_q`` and ``Z_q``, ``2n`` signed Pauli strings. If two
Cliffords agree on all of them, their quotient commutes with the whole Pauli group
and is therefore a scalar -- so this is exactly unitary equivalence up to one global
phase, with no ``2**n`` anywhere and no tolerance to tune.

| candidate, 100 qubits | unitary scope | state scope |
|---|---|---|
| GHZ chain, plus `XZXZ` on qubit 5 (a global phase of −1) | **VALID**, 0/200 images differ | **VALID** |
| star construction: the same prepared state, a different unitary | **INVALID**, 196/200 differ | **VALID** |
| chain with `cx(0,1)` replaced by `cz(0,1)` | **INVALID**, 3/200 differ, mismatches named | **INVALID** |

The dense tier returns `UNRESOLVED` on the same pairs rather than calling them wrong:
being unable to compute a `2**100 x 2**100` matrix is not evidence about the circuit.
At one qubit the distinction is sharpest: `S` and `Z S = S†` prepare the **same
state** from `|0>` and are different unitaries (dense: INVALID, `X_0` image differs),
which is why the report carries a `scope` field and why state scope is never the
default.

The same table is decided at **1,000 qubits** in §6.1: 2,000 images, 61 s for a
7,500-gate pair and 54 s for its mutated twin, with the mutation named on the image
it changes. The number that grows is wall clock and the `2n × depth` work, not memory:
the image tables are 2,000 strings of 1,000 letters.

The **controlled scope** is dense-only (§5): the images are a complete invariant up to
one global phase, which means they are blind to exactly the phase a controlled use
cares about. A controlled question above the dense bound is `UNRESOLVED`.

## 7. Decoders: the interface, the table, and the matching

`apedp_dyn/vqc/decoders.py` names what a decoder is (`name`, `code`, `distance`,
`decode(syndrome) -> Correction`) and supplies two implementations: the distance-3
syndrome table the repetition code started with, and exact minimum-weight perfect
matching on a defect graph. The graph belongs to one error basis, and a CSS code
needs one per basis; the non-CSS five-qubit code keeps its own decoder rather than
pretending a graph covers it.

Matching is T-join dynamic programming over subsets of defects: a defect pairs with
another defect, or terminates on a boundary, at minimum total path weight. It is
exact, not greedy, and exponential in the number of defects — the honest price, and
why the tests stop at distance 7.

`apedp_dyn/vqc/logical.py` gained `repetition_code_circuit(distance, decoder=...)`,
which builds the correction gates **from the decoder**: a lookup table and a matching
decoder are two different circuits for the same code, and the tests compare them on
the same noise.

| check | result |
|---|---|
| distance 3: matching versus the old table | identical correction on all 4 syndromes, identical counts at the same seed |
| distance 7: matching versus brute force | minimum weight matches a brute-force matching over all 64 syndromes |
| every error of weight ≤ `(d−1)/2`, d = 3, 5, 7 | decoded exactly |
| weight 4 on distance 7 | miscorrected for some patterns, as a distance-3 code must be |
| logical rate at `p = 0.1`, 1,200 shots, seed 20261030 | d = 3: 0.0258 vs 0.028 exact; d = 5: 0.0050 vs 0.00856; d = 7: 0.0033 vs 0.002728 |
| the prediction | `repetition_exact_failure_probability` enumerates all `2**d` patterns and applies the decoder; at distance 3 it reproduces `3p² - 2p³` to 1e-15 |
| refusals | a distance-5 syndrome given to the distance-3 table, a graph with an isolated defect, an even distance, an unknown syndrome width |

The two error types are the same experiment in a rotated basis, and the measured
rates are equal at the same seed — which is the kind of agreement that catches a
missing Hadamard rather than hiding it.

## 8. Discovery admission

`apedp_dyn/vqc/discovery.py` is the interface a discovery engine has to go through:
`CircuitCandidate(candidate_id, circuit, reference, claims, expected, shots, seed)`
and `adjudicate_candidate`, which routes to the referee that can decide the pair —
stabilizer tier when both circuits are Clifford, dense tier otherwise, distribution
receipt when the candidate declares one — and seals the result as a
`CandidateVerdict`.

| candidate | verdict | why |
|---|---|---|
| `CX` as `H CZ H`, claim `two_qubit_gates: 1` | **VALID**, `claim_verdict: MET` | equivalent, and the claim holds |
| the same candidate, claim `operations: 1` | **VALID**, `claim_verdict: UNMET` | equivalent, and larger than claimed: refused as non-improving, not as wrong |
| `CX; Z` against `CX` | **INVALID** | relative phase; identical statistics |
| measured Bell, expected `{00: 0.5, 11: 0.5}`, 2,000 shots | **VALID** | a measured circuit has no unitary claim; the receipt decides |
| measured Bell, expected `{00: 0.5, 11: 0.5}`, 300 shots | **UNRESOLVED** | underpowered: the receipt abstains and promotion is refused |
| a measured circuit that declares nothing | **UNRESOLVED** | no claim was checked at all |
| 100-qubit Clifford with a global phase | **VALID**, tier `stabilizer` | decided above the dense bound |
| 20-qubit non-Clifford pair | **UNRESOLVED** | neither tier can decide it; not promoted |

Four questions are answered in four fields, because collapsing them loses a
distinction the ledger needs:

| field | values | question |
|---|---|---|
| `verdict` (`semantic_verdict`) | `VALID` / `INVALID` / `UNRESOLVED` | does it compute the reference operation? |
| `claim_verdict` (`resource_verdict`) | `MET` / `UNMET` / `NOT_ASSERTED` | is it the improvement it is offered as? |
| `evidence` (`evidence_sufficiency`) | `SUFFICIENT` / `UNDECIDED` / `INSUFFICIENT` | was anything checked, and did every check decide? |
| `promotable` (`promotion_decision`) | `PROMOTED` / `REFUSED` | may it be promoted? |

`promote(verdict)` refuses unless all three hold: `VALID`, no `UNMET` claim, and
`SUFFICIENT` evidence. The refusals say which one failed — "is INVALID, not VALID",
"is VALID but not an improvement (UNMET)", or "has insufficient evidence: no check was
run". `evidence` is derived from the checks in `__post_init__`, so a caller cannot set
it, and **an empty check set is `INSUFFICIENT`, never a pass**: a candidate nothing
was asked about is refused with that reason rather than promoted on a vacuous
verification. `tests/test_vqc_discovery.py` pins exactly that.

An equivalent-but-larger candidate keeps `verdict: VALID` and carries
`claim_verdict: UNMET` — the table above reports the second fact, not a wrong circuit.

### 8.1 The loop, run end to end at 100 qubits

`apedp_dyn/vqc/clifford_search.py` is the first engine that emits this IR, and
`scripts/measure_vqc_clifford_discovery.py` runs the whole workflow on one workload:
**propose → certify → rank → admit → promote**, with the refusals recorded beside the
promotion.

The search space is local rewrites that preserve the unitary *by construction*:
cancelling self-inverse pairs (`h h`, `cx cx`, `cz cz`, `swap swap`, …), the same
cancellation when the pair only *commutes* past disjoint gates, the `S` algebra
(`s sdg` vanishes, `s s = z`), and the Pauli conjugations `h z h = x`, `h x h = z`.
That list is a real limitation, and the record states it: this is not a synthesiser,
and it cannot propose a circuit that these rules cannot reach.

Every accepted step is certified by the Clifford referee **against the original
reference**, not against the previous step, so the loop cannot drift, and a wrong rule
is caught rather than trusted. `tests/test_vqc_clifford_discovery.py` turns that into a
property test: for five random Clifford circuits, every rewrite the module proposes
must come back `VALID`.

| measured, 100 qubits | result |
|---|---|
| reference: GHZ chain with cancelling work spliced in | 222 operations, depth 108 |
| after the search | **100 operations**, depth 100 |
| steps | 76, every one `VALID`; 0 rejected; converged before the 1,000-step cap |
| search wall clock | 7.1 s |
| admission | `VALID`, claim `MET`, evidence `SUFFICIENT`, tier `stabilizer`, **promoted** |
| six mutated candidates (delete a gate, flip a phase, `cx`↔`cz`) | all **INVALID**, each with the differing image named |
| the cheaper star circuit (100 vs 222 operations) | **INVALID** at unitary scope (196/200 images differ), **VALID** at state scope, **refused** |
| sampling, 64 shots on the stabilizer backend | identical support, maximum count difference 0 |
| the same workload at 8 qubits, dense referee | **VALID**, distance `1.6e-16` |

The star row is the one to read twice. It is *cheaper* than the reference and it
prepares the *same state*, which is exactly what a resource-only search would
promote. It is not the same unitary, so the referee refuses it, and the report says
why. That is the difference between optimising a circuit and optimising a
distribution, and it is the reason the referee and the resource counter are separate
officials in this loop.

## 8.2 Where the execution runs: Rust first, Python as the referee

The execution core is native (`rust/apedp-vqc`, module `apedp_vqc`) and the Python
tier in `stabilizer.py` is both its fallback and its reference. The split is an
inventory decision, not a rewrite: Pauli algebra, the stabilizer tableau, sampling,
exact distributions and the `2n` Clifford images are bit-vector and integer work
with a predictable cost; the referee, the discovery loop and every verdict stay in
Python, because a fast path must not certify itself.

`apedp_dyn/vqc/native.py` is the bridge. It parses the **same canonical circuit
JSON** and derives its tables from the **same gate matrices**, so there is one
specification with two readers. Every entry point returns `None` when the fast path
cannot answer, and `None` means the Python tier decides; with the wheel absent the
package is unchanged, only slower. A run records which engine executed it
(`RunResult.engine`), a Clifford report records who conjugated the Paulis and who
checked them (`image_engine`, `image_verified_by`), and `engine="rust"` raises rather
than quietly downgrading.

Measured on identical frozen circuits, same process, median of five with the
interquartile range recorded (`scripts/measure_vqc_rust_parity.py`; Python 3.12.3,
NumPy 2.5.3, Linux x86-64, rustc 1.99.0, release build compiled before the run so no
compilation time is in these numbers): images at 100 qubits 17.84 ms -> **4.65 ms**
(3.8x), at 400 qubits 481.94 ms -> **66.96 ms** (7.2x); 100-qubit sampling at 256
shots 3045.63 ms -> **700.11 ms** (4.4x); the exact distribution at 10 qubits
37.37 ms -> **2.52 ms** (14.8x); a 100-qubit Clifford adjudication 41.16 ms ->
**11.63 ms** (3.5x). At 1,000 qubits the two-circuit
adjudication takes **1.94 s** against about a minute on the Python image path.
Agreement is exact, and it is not
sampled: the record's `parity_corpus` re-derives images, generators and (where
enumerable) the whole distribution in Python and compares them entry by entry.
The ratios move between runs on this host (the 100-qubit image row has been measured
from 3.0x to 4.8x), so the record keeps the raw samples and the IQR for each side; the
IQR is a spread over five repeats in one process, not a confidence interval, and the
record's `hardware` block names the machine, the compiler and the build so a reader can
see what the number is a number *about*.

Above `PARITY_LIMIT` (24) qubits the referee does not re-derive the table -- that
would cost exactly what the fast path saves -- so the report says
`image_verified_by: "suite"` and the correctness argument is the conformance corpus.
The dense statevector tier, the decoders, the noise engine and the discovery loop are
not ported, and [`VQC_RUST_CORE.md`](VQC_RUST_CORE.md) says why.

## 8.3 Synthesis from a specification: no reference circuit

The loop in §8.1 starts from a circuit. `apedp_dyn/vqc/synthesis.py` starts from the
*mathematics*: a `TargetSpecification` is a qubit count and `n` signed Pauli
stabilizers, validated as a pure state before any search runs, with no circuit
anywhere in the object. Two referees decide a construction without a reference — the
stabilizer tier by requiring `phase_of(generator) == +1` at any width, and the dense
tier by comparing the prepared amplitudes against the state *projected out of the
specification's stabilizers*, for non-Clifford constructions up to ten qubits.

At 100 qubits, five construction families are built from a GHZ specification and all
five verify: chain (100 operations, depth 100, span 1), star (100/100/99),
balanced binary tree (100/**12**/50), reverse chain and a seeded random tree
(100/14/90). The promoted candidate is the balanced tree, admission
`VALID`/`MET`/`SUFFICIENT` → **PROMOTED**; the Pareto frontier keeps the chain beside
it, because depth and connectivity trade off and the ranking is a stated preference
rather than a measurement of quality. Four wrong constructions (drop the Hadamard,
reverse the middle `cx`, add a Hadamard, `cx`→`cz`) are refused with the failing
generator named, and a second target — a 100-qubit ring graph state — is built and
promoted while the linear cluster, the same construction missing the closing edge, is
refused with its two absent stabilizers named.

Experiment B repeats the synthesis at 10, 50, 100, 250, 500 and 1,000 qubits: 24 rows,
every one verified against the specification, the 1,000-qubit constructions verified in
about 1.3–1.7 s each. The construction space is a **grammar** (chain, star, balanced
tree, seeded random tree, ring, and the graph-state schedules of §8.3.3), not a general
synthesiser, and non-Clifford *search* does not exist: the dense specification referee decides a `ry`-built GHZ up to ten
qubits (at 3 qubits the distance is 3.9e-17, which is rounding) and returns
`UNRESOLVED` above it.
[`VQC_SYNTHESIS.md`](VQC_SYNTHESIS.md) is the full write-up; the record is
`benchmarks/quantum_compilation/vqc_synthesis.json`.

### 8.3.1 Two referees, one circuit

A specification can be decided two ways, and until §8.3.1 only one of them ran per
circuit: the referee picked the stabilizer tier for a Clifford circuit and the dense
tier otherwise, so nothing ever compared them. `referee_agreement` now runs **both** on
the same circuit — `adjudicate_specification(..., tier="stabilizer")` and
`tier="dense"` — and reports the two verdicts side by side. A named tier that cannot
decide a circuit raises rather than falling back: a quiet fallback would make the
comparison a referee against itself.

Measured on 23 circuits — GHZ chains and balanced trees at every width from 2 to 10
qubits, ring graph states at 3–6, and one `cx`→`cz` mutation of a chain — **22 accepted
by both and 1 refused by both**, in exact agreement on every row. The accepted rows are
decided *exactly*: the dense tier's phase-free distance is at most `1e-12` (rounding),
not a tolerance being spent. On the refused row the two tiers fail differently and
still agree on the verdict: the stabilizer tier reports the `ZZ` generator whose sign
flipped and the declared support that no longer matches, and the dense tier reports an
amplitude distance of `0.125`.

The two-tier corpus stops where the dense tier does, so its wide rows are agreed with a
referee outside the package instead: the 400-qubit `cluster_20x20` and 900-qubit
`cluster_30x30` minimum schedules are `VALID` at the stabilizer tier (400 and 900
generators) and `VALID` again for **Stim**, which simulates the same circuits as tableaux
and evaluates the same generators as observables. The three wrong graph constructions the
record refuses are decided again by Stim at 400 qubits, and both referees refuse them —
a third party only ever seen accepting would not be shown to refuse anything. The dense
tier's refusals at both widths (it raises rather than quietly falling back) are stored
beside the rows, and a *disagreement* is not stored at all: the script raises and writes
no record, so an agreeing row is the only kind that can exist.

That corpus is a conformance sample, not a proof. The two tiers share the circuit parser
and one gate table — `gates.matrix_for` is what the stabilizer tier derives its Clifford
conjugation tables from and what the dense tier applies — so a wrong gate matrix or an
indexing convention error would fool both. Only the *representation* is independent:
integer group membership over `n` Pauli strings against `2**n` amplitudes. Randomised
stabilizer circuits, longer sequences, phase-sensitive mutations and adversarial
specifications are the corpus that would test the shared layer, and they are not
claimed here; the closed-form checks in the stabilizer and synthesis suites are what
pins the shared layer today.

The four mathematical conditions on a target are checked before any referee runs:
`n` generators for `n` qubits, mutual commutativity (the symplectic product is even for
every pair), independence over GF(2) (the echelon basis has rank `n`, so the group has
`2**n` elements and the `+1` eigenspace is unique), and Hermitian phases (only `+1` and
`-1`, since a projector onto an `i**q` eigenspace is not a state). Normalisation and
global-phase invariance are properties of the dense referee and are pinned where they
are used: the state is projected and renormalised at every step, and the comparison
minimises over one phase, so `X Y Z = iI` applied to a GHZ still verifies.

### 8.3.2 Architecture-aware depth: what a topology charges

Logical depth assumes any qubit may interact with any other. Experiment D places the
same constructions on four connectivity graphs and reports what routing costs
(`apedp_dyn/vqc/placement.py`). Two routers run on every row: **greedy** (move the
second operand along a shortest path, no scheduling, no lookahead) and **lookahead**
(each swap goes into the earliest layer its two nodes are free, then the candidate
that minimises the wire distance of the next eight two-qubit gates involving a moved
qubit). At 100 qubits:

| construction | topology | logical | greedy mapped | lookahead mapped | greedy swaps | lookahead swaps |
|---|---|---|---|---|---|---|
| chain | line | 100 | 100 | 100 | 0 | 0 |
| chain | 10x10 lattice | 100 | 191 | **117** | 246 | 225 |
| chain | all-to-all | 100 | 100 | 100 | 0 | 0 |
| balanced tree | 10x10 lattice | 12 | 220 | **84** | 547 | 511 |
| balanced tree | ring | 12 | 253 | **212** | 2,151 | 1,832 |
| balanced tree | line | 12 | 311 | **278** | 2,415 | 2,165 |
| balanced tree | all-to-all | 12 | 12 | 12 | 0 | 0 |

Every one of the 16 routed circuits is decided against the same specification and every
row is `VALID`. That is possible because routing is a **relabelling**, not an
approximation: the mapping from logical qubit to node is tracked exactly, and the
emitted swaps are undone in reverse order at the end, so the swap product telescopes to
the identity and the mapped circuit is the original unitary on the same wires. The
restore ladder is part of the reported cost and is reported separately
(`depth_before_restore`, `restore_swap_gates`), because a compiler that tracks its
final layout can absorb it and a verifiable circuit cannot.

The finding is the review's, and it survives the better router: **depth 12 is not depth
12 on hardware.** On a line the balanced tree is still nearly three times deeper than
the nearest-neighbour chain it was supposed to beat (278 against 100), on a ring 212
against 100, and on a 10x10 lattice 84 against 117 — while on all-to-all it is the 12 it
claims. The lookahead improves every row that needs routing (lattice tree 220 → 84,
−62%) and leaves every free row untouched, which is why the Pareto frontier keeps both
circuits instead of picking a winner. It is a heuristic, not a theorem: at small widths
it can lose (a 9-qubit tree on a line is two layers deeper), and neither router
reorders gates. The numbers are routing bounds, not a floor on what hardware must pay,
and no device calibration, error rate or classical-control cost is in them.

The `mapped_depth` above is the **total** including the restore ladder;
`depth_before_restore` is the circuit before that ladder is appended, and
`restore_depth_cost` is the difference (94 layers for the greedy lattice tree, 35 for
the lookahead one, 0 wherever routing was unnecessary). Two-qubit counts are reported
in three bases rather than one: **abstract operations** (a swap is one instruction —
1,193 for the greedy lattice tree, 1,121 for the lookahead), **native entanglers** in a
declared basis (one per swap here, so the same numbers) and **CNOT-equivalent** (three
per swap — **3,381** and **3,165**; the record quotes `99 + 2(547)` and `99 + 6(547)`
beside each other). On a CNOT-only backend the tree costs 2.8× the headline count, and
the lookahead saves 3,756 CNOT-equivalents across the eight rows.

### 8.3.3 Graph states: the specification is the wiring

A graph state is `X_v Z_N(v)` on every vertex, so its specification *is* an edge set.
`cluster_state_target(rows, cols)` describes a two-dimensional cluster — the grid-graph
state with qubits on the lattice **vertices**, not the surface-code cluster whose qubits
sit on lattice edges — and `seeded_graph_target(n, seed=...)` an irregular graph with no
regular structure. `graph_of_state` reads the edges back out of the stabilizers, so the
families are handed no graph, and a specification with a `Y`, two `X`s or an asymmetric
edge is refused by name.

Because the edges are the target's, the operation count, the two-qubit count and the
connectivity are fixed and **only the schedule is free**. Five schedules are built:
wire order, transposed order, breadth-first tree then closures, a **minimum edge
colouring** (`colouring.py`), and a seeded random order as a control. A layer of `cz`
gates is a matching, so a schedule is an edge colouring, and the minimum is the edge
chromatic number `χ′`, bounded by Vizing's theorem to `Δ ≤ χ′ ≤ Δ + 1`. The scheduler
finds a `Δ`-colouring by complete search — optimal by the degree lower bound — or
*refutes* `Δ`, which proves the graph class 2 and makes the constructed `Δ + 1`
colouring optimal. Two refutations exist and the cheaper one is checked first: a layer
is a matching, so it holds at most `⌊V/2⌋` edges, and when the edge count exceeds
`Δ·⌊V/2⌋` the counting bound refutes `Δ` outright (`status:
delta_refuted_by_layer_capacity`); otherwise the search must exhaust its space
(`status: delta_refuted`). The search prunes on a necessary condition (no vertex may
have more uncoloured edges than free colours) and carries a node budget, because
deciding the `Δ` case is NP-complete: a run that only exhausts the budget classifies
nothing, so it reports a `Δ + 1` colouring — at most one layer above the minimum by
Vizing — and never calls it minimal. That colouring is *constructed*, not searched:
`vizing_colouring` runs the fan argument behind Vizing's theorem (Misra–Gries form, one
fan, one `c`/`d` path inversion and one rotation per edge in a single pass), verifies
the result and returns it, so a budget-exhausted instance still gets a schedule. The
counting bound is what makes the odd complete graph family measurable: `K9`
(`Δ = 8`, 36 edges) and `K11` (`Δ = 10`, 55) cannot be refuted by search at any usable
budget, and are now proved class 2 at zero nodes — the peel schedules both in 15
layers, the construction in **9** and **11**.

| target | qubits | edges | Δ | χ′ (proven) | tree-first (layers) | random (layers) | raster (layers) |
|---|---|---|---|---|---|---|---|
| cluster 4x4 | 16 | 24 | 4 | **4** | 8 | 7 | 12 |
| cluster 10x10 | 100 | 180 | 4 | **4** | 20 | 10 | 36 |
| cluster 20x20 | 400 | 760 | 4 | **4** | 40 | 13 | 76 |
| irregular, seed 7 | 25 | 68 | 12 | **12** | 23 | 20 | 23 |
| irregular, seed 23 | 64 | 271 | 16 | **16** | 43 | 31 | 44 |

All 25 valid rows verify. The 400-qubit lattice is the case where the schedule's cost
is measurable rather than negligible: building a schedule takes 0.02-0.03 s and deciding
it 0.22-0.26 s, per row, and the five schedules' times are within a few percent of each
other — same 1,160 operations to build, same 400 stabilizer checks to decide
([`VQC_SYNTHESIS.md`](VQC_SYNTHESIS.md) §5.3 records the per-row wall clock). Replacing the greedy peel changes **none** of
these five depths — the peel already happened to attain `Δ`, and the record says the saving is
0 rather than presenting the new scheduler as an improvement on Experiment E. What it
adds is proof, and the class-2 case: on the complete graph `K5` (`Δ = 4`, `χ′ = 5`)
the old peel used **7** layers and the minimum is **5**; on `K9` (`χ′ = 9`) and `K11`
(`χ′ = 11`) the peel used **15** layers against **9** and **11**, both refuted by the
layer-capacity bound rather than by a search that would never finish; on an odd cycle
and the Petersen graph the peel was already optimal, and the scheduler now proves why.
Three
wrong constructions (a missing edge, an extra edge, one `cz`→`cx`) are refused with
exactly the two endpoint stabilizers named, and a GHZ chain against a 3x4 cluster is
`INVALID` at a shared width, in both directions: the referee is deciding the state,
not accepting a family. The Pareto frontier collapses to the minimum schedule here —
unlike §8.3, where depth traded against connectivity — because the connectivity is the
specification's, so the shallowest schedule dominates at equal operations.

**What routing does to the advantage (§5.3.1 of `VQC_SYNTHESIS.md`).** The five
schedules were then placed on a line, a ring and the cluster's own lattice by both
routers, at 16, 100 and 400 qubits — 90 placements, 72 comparisons against the minimum
schedule, every placed circuit `VALID` under the same referee. On the lattice the
target's edges *are* lattice edges, so no schedule needs a SWAP and the advantage is
exactly the logical one (1.00x, all 24 lattice comparisons). On a line or ring it
**grows**: the minimum schedule pays a real routing bill (12,316 swaps at 400 qubits on
a line, longest hop 40) but the localized orders pay 144,400 with a longest hop of 362,
so the surviving fraction is 1.00x-219x (median 4.67x) and no comparison inverts — the
minimum schedule is still the shallowest after routing, by a wider margin than before.

Experiment E also found an implementation bug worth recording: the dense tier compared
`StatevectorSimulator` amplitudes (bit `i` = qubit `i`) against a Kronecker product
(qubit 0 most significant). Every earlier specification was invariant under reversing
its wires, so the two orders were the same state; an irregular graph is not, and the
naive comparison reported `0.0605` on a state that was exactly right. One re-index
(`_specification_order`) makes it `2.8e-17`, and the test pins both numbers.

### 8.3.4 What one non-Clifford decision costs

The dense tier is the only referee that decides a non-Clifford construction against a
specification, and until this measurement `DENSE_SPEC_LIMIT = 10` was a policy with no
price attached. The tier materialises each stabilizer as a dense `2**n x 2**n` Pauli
operator, so one operator costs `16 * 4**n` bytes — quadratic in the Hilbert dimension —
while the state is `16 * 2**n`. Each width is measured in a fresh process (RSS is a
process high-water mark) with a declared 8 GiB address-space cap:

| qubits | decision | peak traced | one Pauli operator | referee |
|---|---|---|---|---|
| 6 | 0.02 s | 0.2 MiB | 0.06 MiB | VALID |
| 8 | 0.07 s | 1.5 MiB | 1.0 MiB | VALID |
| **10** (bound) | **0.25 s** | **20 MiB** | 16 MiB | VALID |
| 11 | 1.31 s | 80 MiB | 64 MiB | UNRESOLVED |
| 12 | 5.1 s | 320 MiB | 256 MiB | UNRESOLVED |
| 13 | 20.3 s | 1.25 GiB | 1.0 GiB | UNRESOLVED |
| 14 | 88.4 s | 5.0 GiB | 4.0 GiB | UNRESOLVED |
| 15 | — | — | **16 GiB** | UNRESOLVED (wall) |

The peak grows by 3.96–4.00x per qubit, which is the operator's own scaling, and the
time by 3.9–5.3x. The wall under the declared budget is **15 qubits**, where one
operator is twice the budget and the process raises `MemoryError` after building the
4 GiB intermediates; 14 fits at 88 s and 5 GiB. Inside its bound the tier is cheap —
0.2 s and 20 MiB at ten qubits — so memory, not arithmetic, is what stops it.

The bound is not lifted: `adjudicate_specification` still answers `UNRESOLVED` above
ten, and `dense_tier_cost` returns seconds, peak bytes and a distance with **no verdict
field**, so a caller allowed to exceed the bound cannot turn a price into an
adjudication. The wall is also a property of this implementation: applying a Pauli
string as single-qubit gates and index permutations would use `O(2**n)` memory and move
it, and that is stated rather than priced.

## 8.4 Against an external tool: the same specification, our referee on their circuit

Every claim above is measured against this package's own reference implementations,
which is precisely the objection the external review raised: nobody outside cares how
a tool compares with itself. `apedp_dyn/vqc/external.py` is the matched comparison, and
its rule is that **a tool that did not run is not a tool that lost**: every row is
either a measurement on the same frozen specification or an abstention naming the
reason.

Qiskit 2.6.1 is wired because `synth_circuit_from_stabilizers` takes the same signed
Pauli stabilizer strings this package already holds, so both engines can be given *the
same specification* rather than two spellings of one. Its output is exported to QASM 2
and re-imported, then decided by the specification referee — so its gate count is only
quoted for a circuit that was independently verified as preparing the requested state.
At 100 qubits (`benchmarks/quantum_compilation/vqc_external_comparison.json`):

| target | engine | operations | two-qubit gates | depth | verdict |
|---|---|---|---|---|---|
| GHZ 10 | grammar: chain / tree | 10 | 9 | 10 / 5 | VALID |
| GHZ 10 | Qiskit | 64 | 45 | 35 | VALID |
| GHZ 100 | grammar: chain / tree | 100 | 99 | 100 / **12** | VALID |
| GHZ 100 | Qiskit | 5,149 | **4,950** | 395 | VALID |
| ring 100 | grammar: ring | 200 | **100** | **101** | VALID |
| ring 100 | Qiskit | 200 | **100** | 199 | VALID |
| random 30 | grammar | *abstains: no construction family* | — | — | — |
| random 30 | Qiskit | 643 | 222 | 250 | VALID |

The GHZ row is a 50× difference in two-qubit gates, and the ring row is **none at
all**: a general Gaussian-elimination synthesizer reaches the same two-qubit count as
the hand-written ring construction and loses only on depth. That is why the record says
the advantage is target-specific and does not claim a general one — a grammar that
selects among constructions it already knows is not a synthesizer that finds one, and
beating it on the target it was written for is not a general advantage. The random
stabilizer row is the honest complement: there the grammar has nothing to offer and
Qiskit does. On a line, the same three 100-qubit circuits cost 100 / 311 / 1,272
mapped depth (0 / 2,415 / 19,208 routing swaps) under §8.3.2's **greedy** router; the
external record is a matched comparison against one fixed router, not a claim that
§8.3.2's lookahead leaves every foreign circuit as it found it.

**A finding worth its own paragraph: the two ecosystems disagree about string order.**
Qiskit reads the first character of a stabilizer string as the *highest*-indexed qubit;
this package reads it as qubit 0. Handed over unconverted, the result is the mirror
image of the requested state — which verifies for a GHZ and for a ring, because both
are palindromic, and fails for every other target. That is exactly the class of defect
differential testing exists to find, and it was invisible until a random 30-qubit
state entered the corpus; `test_the_string_order_difference_is_pinned_on_a_non_
palindromic_target` pins it with a two-qubit `X`-on-0 / `Z`-on-1 target that no
palindrome can hide. The adapter reverses the strings, and the referee — not the
conversion code — is what confirms the reversal.

**The shared corpus: a sampled distribution of graph states, not a chosen set.** The
rows above share a weakness the record itself names: the GHZ and ring targets are
exactly what the grammar was written for, so beating a general synthesizer there is a
measurement about the target. `apedp_dyn/vqc/corpus.py` is the correction, and after
the review of that correction it is a *sample* rather than a set: every family at five
widths (12, 20, 32, 48, 64 qubits) with five independent seeded draws per cell, plus
one random stabilizer anchor at 12, 20 and 32 qubits. Seventy-eight instances in all,
seventy-five of them matched, each reproducible from the arithmetic seed rule
`seed = SEED_BASE[family] + 10 * width + replicate`, so a row's label names the cell
that produced it.

The reason for the cells is that one draw per family cannot distinguish a family's
behaviour from that draw's. Every family is therefore reported as a distribution, and
the width trend is part of the report rather than an average over it:

| family, width | grammar depth (median) | Qiskit depth (median) | ratio q/g, median [IQR over the 5 draws] | depth wins g/q/t |
|---|---|---|---|---|
| regular 3, 12q | 4 | 13 | 3.25 [3.00–3.25] | 5/0/0 |
| regular 3, 20q | 4 | 12 | 3.00 [2.75–3.50] | 5/0/0 |
| regular 3, 32q | 4 | 15 | 3.75 [3.25–4.00] | 5/0/0 |
| regular 3, 48q | 4 | 15 | 3.75 [3.50–3.75] | 5/0/0 |
| regular 3, 64q | 4 | 14 | 3.50 [3.50–3.75] | 5/0/0 |
| regular 4, 12q | 5 | 16 | 3.20 [3.00–3.60] | 5/0/0 |
| regular 4, 20q | 5 | 18 | 3.60 [2.60–4.40] | 5/0/0 |
| regular 4, 32q | 5 | 21 | 4.20 [3.60–4.40] | 5/0/0 |
| regular 4, 48q | 5 | 23 | 4.60 [4.20–4.80] | 5/0/0 |
| regular 4, 64q | 5 | 20 | 4.00 [4.00–4.20] | 5/0/0 |
| irregular, 12q | 6 | 15 | 2.50 [1.75–3.00] | 5/0/0 |
| irregular, 20q | 8 | 20 | 2.38 [2.25–2.78] | 5/0/0 |
| irregular, 32q | 11 | 26 | 2.50 [2.36–2.67] | 5/0/0 |
| irregular, 48q | 14 | 45 | 3.21 [2.60–3.50] | 5/0/0 |
| irregular, 64q | 17 | 53 | 2.88 [2.65–3.20] | 5/0/0 |
| random stabilizer 12q | *abstains* | 34 | — | — |
| random stabilizer 20q | *abstains* | 94 | — | — |
| random stabilizer 32q | *abstains* | 126 | — | — |

On all 75 matched instances the two engines use the **same number of two-qubit gates**
— one entangler per edge — and the grammar's only saving is depth, because it colours
the edge set into the minimum number of parallel matchings (§8.3.3) while Qiskit emits
the edges in sequence. Across the sample the grammar is shallower on 75 of 75
instances; the median ratio is **3.25×** with quartiles 2.75–3.75 and a range of
1.50–5.00, and per family regular 3 is 3.50× [3.00–3.75], regular 4 is 4.00×
[3.20–4.40] and the irregular family is 2.65× [2.38–3.20]. The bootstrap interval on
those medians (regular 3 [3.00, 3.75], regular 4 [3.60, 4.40], irregular [2.50, 3.00])
resamples these instances and says how far a median could have moved under another
draw from the same generator; it is not a statement about graph space, and the record
says so.

The width trend has a mechanical cause worth stating, because it is the reason a single
width would have been misleading: the grammar's depth is set by the degree (4 layers
for a cubic state, 5 for a quartic one) and does not grow with the qubit count, while
Qiskit's elimination emits edges in sequence and its depth tracks the width. The ratio
therefore widens with width even though neither engine's result changes shape. That is
a scheduling result a transpiler pass can reproduce, not a gate-count advantage. On the
three random stabilizer states the grammar abstains and Qiskit synthesizes; the corpus
keeps that asymmetry in the same table instead of only reporting the favourable graph
cases.

**The routed sample: each engine's own router, not ours on both sides.** The synthesis
rows measure circuits in the abstract, and the line-topology task routes *both* circuits
with this package's router — which measures this package's router. The routed corpus is
the correction: all 75 matched instances are routed onto the recorded `fake_sherbrooke`
coupling map by **Qiskit's own transpiler** (initial-layout search, SABRE, layout kept,
seed 1234, optimisation 1) and by **this package's lookahead router** (identity
placement, restore ladder reported separately). Every routed circuit is decided by the
same referee: Qiskit's against the target relabelled by its final layout, this
package's against the target padded with idle wires.

| family | routed depth, ours / Qiskit (median) | ratio q/a, median [IQR] | wins a/q/t | swaps ours / Qiskit |
|---|---|---|---|---|
| all 75 | 51 / 49 | **0.97** [0.88–1.10] | 31/40/4 | 350 / 206 |
| random 3-regular | 44 / 32 | 0.88 [0.82–0.98] | 5/19/1 | 257 / 135 |
| random 4-regular | 56 / 54 | 1.00 [0.91–1.06] | 12/12/1 | 350 / 206 |
| seeded irregular | 75 / 79 | 1.05 [0.96–1.11] | 14/9/2 | 423 / 253 |

**The 3.25× abstract depth gap does not survive routing.** At the median the two
routers land within 3% of each other (bootstrap interval on the median ratio
[0.93, 1.04]), and Qiskit is the shallower of the two on 40 of 75 instances against this
package's 31. The width trend is where the two disagree with each other: Qiskit is
ahead at 12–48 qubits (median ratios 0.95, 0.93, 0.96, 0.99) and this package is ahead
at 64 qubits (1.09 [1.01–1.13], 11 of 15 instances). The swap counts say part of why:
this package's router swaps more in every family (350 against 206 at the median across
the sample), because Qiskit's transpiler searches an initial layout while this package
starts from the identity placement on a sparse device graph and then restores. That is
each tool as shipped, which is the comparison the review asked for, and it is not the
same optimisation problem on both sides — the pre-restore routed depth is the
comparable number and the total is reported beside it.

**A third-party referee: the corpus is decided twice.** A comparison one side of which
is this package, adjudicated only by this package, is self-referential — and that was
the last limitation the record carried. `apedp_dyn/vqc/foreign_referee.py` is the second
opinion: it hands each circuit to **Stim**, simulates it as a stabilizer tableau, and
evaluates every generator of the frozen specification as a Pauli observable. The
specification is the same signed Pauli strings both engines were given, so Stim decides
the *same question*, not a looser one, and a pure stabilizer state is the unique +1
eigenstate of `n` independent stabilizers — no amplitude is ever computed, which is why
the same referee decides a 127-qubit routed circuit in a hundredth of a second.

| circuit set | decisions | VALID | INVALID | REFUSED | disagreements with our referee |
|---|---|---|---|---|---|
| corpus synthesis (ours and Qiskit's) | 150 | 150 | 0 | 0 | 0 |
| corpus routed (ours and Qiskit's) | 150 | 150 | 0 | 0 | 0 |

Every matched instance carries Stim's verdict for both circuits beside our referee's,
and only `VALID`/`INVALID` count as verdicts: an absent Stim, a non-Clifford gate or a
measured circuit is `UNAVAILABLE`/`REFUSED` with a reason and is never scored as a loss.
That matters more here than anywhere else in the record, because the referee is the one
component that decides both sides.

What the third party does *not* settle, and the record says so: it decides the prepared
**state**, not gate counts, depth or wall clock, which remain this package's accounting;
it is **Clifford-only**, so a non-Clifford circuit (the dense-tier syntheses elsewhere in
this package) is refused rather than verified; and the bridge from this package's IR to
Stim's circuit is ours, so a translation bug would be invisible to both referees. A
second third-party implementation that does not share our IR, and a third-party resource
accounting, are still missing.

What the corpus does *not* establish: the widths are five fixed tiers, the draws come
from one pairing-model generator and one tree-plus-edges generator, and instances at
the same width share that width — so the spread is over *these* 78 instances, not over
graph space. The routed rows are one router run per instance (SABRE is stochastic and
one seed is recorded).

**Stim: the execution comparison, on the same specification.** Stim does not compete on
synthesis, so it is not asked to. It is handed the same signed stabilizer strings, told
the same shot count, and both samplers are decided against the **exact** outcome
distribution — the referee the 12-qubit sampling workload exists to make possible. This
package samples by cloning its tableau and measuring qubit by qubit; Stim builds its
own tableau, synthesizes a preparation circuit and compiles a measurement sampler. The
two sample sets are also compared with each other under a null in which both are
correct, so agreement is not inferred from either side's clock or from a shared bug in
one implementation.

| 12-qubit target | shots | grammar TV / band | Stim TV / band | pair TV / band | verdict |
|---|---|---|---|---|---|
| GHZ | 40,000 | 0.0000 / 0.0079 | 0.0025 / 0.0079 | 0.0025 / 0.0096 | both `CONSISTENT`, pair `AGREE` |
| ring graph | 40,000 | 0.1297 / 0.1323 | 0.1247 / 0.1323 | 0.1774 / 0.1866 | both `CONSISTENT`, pair `AGREE` |
| random stabilizer | 40,000 | 0.0428 / 0.0497 | 0.0460 / 0.0497 | 0.0638 / 0.0701 | both `CONSISTENT`, pair `AGREE` |

Median throughput differs by about **three orders of magnitude** (thousands against
millions of samples per second; the record carries the run's two clocks and rates).
That ratio is an implementation comparison, not a ceiling: a Python per-shot tableau
clone against Stim's compiled C++ sampler, with no parallelism on this side and no
memory measurement on either.

**Above twelve qubits: exact marginals as the referee.** The 12-qubit workload exists
because the *full* distribution is enumerable there, and that is a property of the
referee rather than of either sampler. `apedp_dyn/vqc/marginals.py` removes it: the
Z-basis distribution of a stabilizer state is uniform over an affine subspace of
GF(2)^n, so the marginal on a chosen subset of wires is Gaussian elimination over GF(2)
plus `2**|T|` outcomes — nothing exponential in the *state's* width — and each
sampler's projected samples are decided against it by a chi-square goodness-of-fit,
Bonferroni-corrected over the subsets tested. The wide block runs both engines at 16,
24 and 32 qubits on every adjacent pair of wires plus three 10-wire projections
(contiguous, strided, seeded draw), 20,000 shots each:

| target | qubits | subsets | uniform | smallest family *p* | grammar | Stim | pair |
|---|---|---|---|---|---|---|---|
| GHZ | 16 | 18 | 0 | 0.064 | `CONSISTENT` | `CONSISTENT` | `AGREE` |
| random stabilizer | 16 | 18 | 15 | 1.000 | `CONSISTENT` | `CONSISTENT` | `AGREE` |
| GHZ | 24 | 26 | 0 | 1.000 | `CONSISTENT` | `CONSISTENT` | `AGREE` |
| random stabilizer | 24 | 26 | 23 | 0.159 | `CONSISTENT` | `CONSISTENT` | `AGREE` |
| GHZ | 32 | 34 | 0 | 1.000 | `CONSISTENT` | `CONSISTENT` | `AGREE` |
| random stabilizer | 32 | 34 | 30 | 1.000 | `CONSISTENT` | `CONSISTENT` | `AGREE` |
| ring graph | 32 | 34 | 34 | 1.000 | `CONSISTENT` | `CONSISTENT` | `AGREE` |

190 subset decisions, all inside the corrected p-value on both engines, and no
subset on which the two engines disagree. The margins are not uniform: GHZ's pair
marginals keep the correlation (two outcomes at `p = 1/2`), so a mis-wired qubit is
caught by the pairs that contain it and named by them.

What the block does **not** decide is stated with it. A marginal is a projection, not
the distribution: a state that differs from the target only on wires no tested subset
covers passes, which is why every row lists the subsets it was decided on. And every
graph state is uniform in the Z basis, so no Z-basis marginal tells one graph state
from another — the ring-graph row is in the block to show that (34 of 34 marginals
uniform) rather than to claim a verdict it cannot support. The full distribution is
still only decided at 12 qubits; what the wide rows add is an exact referee at widths
where the enumeration does not exist, not a weaker referee wearing its name.

**Memory, on the same workloads.** Throughput without memory is half a comparison, and
peak memory cannot be read the obvious way: `VmHWM` never falls, so two engines measured
in one interpreter cannot be separated, and a whole child's peak is mostly the ~165 MiB
of interpreter and numpy both engines stand on. `apedp_dyn/vqc/memory.py` runs each
engine in a fresh child and **resets the peak counter** after the engine's imports and
after rebuilding the specification, so the window covers the sampling step and nothing
before it. The child returns the count hash of the samples it drew, and the record's
checks require it to equal the hash of the recorded run — the memory belongs to these
samples, not to a similar workload.

| block | engine | import floor | median growth | largest row |
|---|---|---|---|---|
| 12-qubit, 40,000 shots | grammar | 164 MiB | 0.8 MiB | 1.2 MiB |
| 12-qubit, 40,000 shots | Stim | 169 MiB | 3.0 MiB | 3.0 MiB |
| 16–32 qubits, 20,000 shots | grammar | 164 MiB | 1.4 MiB | 8.6 MiB |
| 16–32 qubits, 20,000 shots | Stim | 169 MiB | 3.4 MiB | 10.0 MiB |

What moves the number is the **outcome table**, not the width: a 32-qubit GHZ keeps two
outcomes and grows 0.2 MiB here, while a 32-qubit random state with 20,000 distinct
outcomes grows 8.6 MiB (this package) and 10.0 MiB (Stim) — at which point both engines
are measuring the same 20,000-entry table and the difference is a megabyte of
implementation. So the honest reading is that memory is comparable, slightly in this
package's favour on the smaller tables, and dominated by the outcome table on the
larger ones. Peak RSS is resident growth rather than bytes allocated, one run rather
than a median, and no ceiling claim: Stim's 20,000-qubit workload is not measured here.
Stim's published 20,000-qubit surface-code workload is not addressed anywhere in this
record, and the 1,000-qubit rows in §6.1 remain evidence about this package's own
scaling rather than a state-of-the-art claim.

**Each tool's own router, on a recorded device.** The line topology task above still
routes Qiskit's circuit with *our* router; that is the "our router applied to their
circuit" caveat the record used to carry. The device task closes it: Qiskit is given
the coupling map of `fake_sherbrooke` — a recorded 127-qubit IBM Eagle snapshot shipped
with `qiskit-ibm-runtime`, not a synthetic lattice — and its own transpiler (SABRE),
while both of this package's routers run on the same device graph. Qiskit keeps its
final layout rather than restoring it, so its circuit is decided against the target
relabelled by that layout; this package restores, and both its routed and total depths
are reported.

| logical circuit | Qiskit routed depth | greedy routed / total | lookahead routed / total |
|---|---|---|---|
| GHZ chain | 239 | **119** / 217 | **116** / 130 |
| GHZ balanced tree | **83** | 131 / 240 | 95 / 165 |
| ring 100 | 251 | **121** / 219 | **117** / 173 |
| Qiskit's synthesised GHZ (5,149 gates) | **1,416** | 29,660 / 54,859 | 7,253 / 12,323 |

All eight routed circuits are `VALID` under the same referee, and Qiskit's own
`depth()` agrees with the depth measured on the re-imported circuit. The result is
split, and the record says so: **Qiskit wins 2 of 4** against each router (median
routed-depth ratio 1.47× against the lookahead, 1.32× against greedy), because on its
own 5,149-gate GHZ the grammar's edge ordering is exactly the schedule the router then
has to undo, while on the small grammar circuits the minimum colouring wins. A single
transpiler seed and optimisation level are recorded because SABRE is stochastic; this
is one router run, not a median over seeds.

**MQT QCEC is wired for the equivalence task** (§8.5), which is the task it actually
overlaps with: it is handed the same pairs the equivalence referee in §5/§5.1 decides,
on a corpus built so the two disagree.

**What is not compared.** The device comparison prices a SWAP as one two-qubit gate on
both sides and contains no calibration, error model or measurement; routed depth is not
fidelity. The two samplers also drew independent seeds, so the pair comparison is two
independent samples of one state rather than paired trajectories.

Qiskit, Stim and MQT QCEC are optional dependencies of the harness only
(`pip install apedp[compare]`): nothing in `apedp_dyn` imports any of them at import
time, and without one the harness records that task as an abstention with a reason
while the rest of this suite is unchanged. The comparison records are
`benchmarks/quantum_compilation/vqc_external_comparison.json` (synthesis, routing,
sampling, memory and equivalence) and, for the tiers it exercises,
`vqc_synthesis.json`, `vqc_stabilizer_scaling.json` and `vqc_native_parity.json`.

### 8.5 The equivalence task: pairs built so the two referees disagree

An equivalence comparison on pairs both tools call equivalent measures nothing, which
is what the earlier record said when it left QCEC out: *the comparison that would
matter is a corpus where the two disagree, and it does not exist yet*. That corpus is
`apedp_dyn/vqc/equivalence_corpus.py`, and its rule is that **each pair's relation is
fixed by the construction that built it**, not by either referee. Eighteen pairs, six
families; every row in the record carries its construction, both circuits' hashes, and
every scope this package can be asked about.

| family | n | what fixes the answer |
|---|---|---|
| `rewrite` | 3 | `X X = I`; two preparation blocks on disjoint wires commute; `SWAP = CX CX CX` — exactly the same unitary |
| `global_phase` | 2 | `X Y Z = i I` and `RZ(2*pi) = -I` — equal standalone, a relative phase under control |
| `relative_phase` | 2 | `S` vs `Sdg`; `CZ` on one branch, which is diagonal and leaves every Z-basis probability identical |
| `state` | 1 | `Z_0 Z_1` stabilises a GHZ — the same prepared state from a different unitary |
| `tolerance` | 7 | `U_c = RX(theta) U_r` for `theta` from `1e-13` to `1e-3` — exactly inequivalent at every point, distance `theta/2` |
| `wide` | 3 | 12 qubits, above the dense bound: a Clifford rewrite (decided here), `T Tdg` (equal), `T` vs `S` (unequal) |

**What each answered.** The rows below are the ones that carry the comparison; the
whole table is in `benchmarks/quantum_compilation/vqc_external_comparison.json`.

| pair | question | this package | QCEC 3.11.0 |
|---|---|---|---|
| `rewrite-cancelled-pair` | standalone | `VALID`, d = 0 | `equivalent` |
| `phase-rz-2pi-is-minus-one` | standalone | `VALID`, d = 1.7e-16 | `equivalent_up_to_global_phase` |
| `phase-rz-2pi-is-minus-one` | controlled | `INVALID`, d = 1.41 | not decided: no controlled mode |
| `relative-diagonal-cz` | standalone | `INVALID`, d = 1.00 | `not_equivalent` |
| `state-ghz-stabiliser` | state | `VALID` | `not_equivalent` — it decides the unitary, and our unitary verdict agrees |
| `tolerance-1e-09` | standalone | `VALID`, d = 5e-10 | `equivalent` |
| `tolerance-1e-08` | standalone | `INVALID`, d = 5e-9 | `equivalent` |
| `tolerance-1e-03` | standalone | `INVALID`, d = 5e-4 | `no_information` or `not_equivalent`, varying between identical calls |
| `wide-non-clifford-differing` | standalone | `UNRESOLVED` (`'t' is not Clifford`) | `not_equivalent` |
| `wide-clifford-rewrite` | standalone | `VALID`, stabilizer tier | `equivalent` |

**Three axes, kept apart.** *Tolerance*: both tools apply a threshold and neither is
the arbiter — this package at `1e-9` on the per-amplitude Frobenius distance, QCEC at
a numerical tolerance of its own (`2.27e-13` as its configuration declares it, which is
not the number its verdicts flip at). The ladder is the measurement: `equivalent` up
to `theta = 1e-9`, `equivalent` still at `1e-8` where this package has said `INVALID`
since `5e-9`, and from `1e-7` up the criterion changed between *identical* calls. Each
ladder point is asked fifteen times and the record carries every criterion, so a
boundary answer is a distribution and not a verdict. *Scope*: `diag(I, U)` is not
`diag(I, e^(i phi) U)` and a state is not a unitary, so the controlled and
prepared-state verdicts are recorded beside the standalone one and never compared
with an answer to a question the other tool does not decide. *Abstention*: a 12-qubit
pair carrying a `T` is refused here — the reason names the gate and the tier — and
decided there; a 12-qubit Clifford pair is decided by both, which is why the dense
bound is a bound on the dense tier and not on the referee.

The record's classifier has four ways to explain a difference and one way to report
the absence of an explanation, and the run refuses to write itself if any row lands in
the last one. It also refuses to write if any pair's verdict differs from what its
construction requires: the corpus is a test of this package first and a comparison
with QCEC second.

**A false `VALID` came out of writing this.** The `state` family is what exposed it:
`adjudicate_clifford_equivalence(on="state")` reported a GHZ state and its
`CZ`-phase-flipped twin as the *same state*, with zero overlap between them. The group
hash two conventions folded together — the mask represents `X^a Z^b` while the argument
is a string of Pauli matrices (`Y` is `i X Z`, not `X + Z`), and `X^a Z^b · X^c Z^d`
carries a `(-1)^{b.c}` that adding the two phases drops. The sign therefore depended on
which rows the elimination happened to pick: `+YYX` was read as a stabilizer of a GHZ
state it is orthogonal to, and `Y_0 Y_1` was not found in the group of a Bell pair at
all. Both are fixed in `apedp_dyn/vqc/stabilizer.py` (`_product_phase` plus the Y
count), and `tests/test_vqc_stabilizer.py` now enumerates every Pauli string of a random
3-qubit state against the group built by multiplying generators, and against the dense
expectation — 64 strings per state, both directions, `None` included.

**What this does not show.** Eighteen constructed pairs measure the *edges* of two
decision rules, not a general agreement rate over circuit pairs; the widths are one
ladder of one angle and one 12-qubit family; QCEC is given one corpus and no timeout,
and nothing here is a claim about its ceiling; and it has not been run on synthesis,
routing or sampling, just as Qiskit has not been run on equivalence. No clock ratio is
reported: QCEC answers in milliseconds on every row, and a single wall-clock
measurement on a 3-qubit pair is not a benchmark.

## 9. Receipts

`apedp_dyn/vqc/receipts.py` seals one adjudication into machine-readable JSON:
identity (circuit SHA-256, schema version), provenance, per-check evidence,
resources (qubits, bits, gates, two-qubit gates, measurements, noise operations,
depth), ancilla cleanup, reproducibility (backend, shots, seed, Python and NumPy
versions) and a verdict from `{VALID, INVALID, UNRESOLVED}`.

| candidate | verdict | why |
|---|---|---|
| correct teleportation, 4,096 shots | **VALID** | all outcomes within 4 standard errors; ancilla 1 returned to `|0>`; 4,096 ≥ the 1,600 shots needed to see a 0.05 shift at 4σ |
| teleportation without the conditional `Z` | **INVALID** | worst outcome off by z = 62; the circuit is a different map on a superposition |
| teleportation with `X` applied to the consumed qubit after its measurement | **INVALID** | ancilla excitation 0.99999999999 — workspace abandoned, not cleaned |
| correct teleportation, 8 shots | **UNRESOLVED** | nothing failed; 8 shots cannot detect a 0.05 shift, so no pass is issued |
| a circuit with an unknown gate | **INVALID** | validation fails, and a receipt is still produced |

The `UNRESOLVED` path is the point of the closed verdict set: an underpowered run
must not become a pass.

## 10. What is not built

Stated so the next milestone is not mistaken for this one:

- **No MPS or tensor-network tier.** Two backends exist (dense, stabilizer) and they
  are compared on every circuit both can hold; a circuit that is neither dense nor
  Clifford has no fast tier, only the exponential one.
- **No general synthesiser and no non-Clifford search.** §8.3 constructs from a
  specification inside a grammar of named wireings; it cannot invent a topology
  outside it, and no rewrite rule or search move crosses the Clifford boundary. The
  Toffoli/arithmetic/QFT workloads need their own engine, cost model and referee
  story — the dense specification referee can *verify* a non-Clifford construction up
  to ten qubits, but nothing proposes one.
- **No surface code.** The matching decoder runs on repetition-code defect graphs, so
  no two-dimensional lattice, no crossing defects, no decoding threshold. The
  distance scaling measured here is a repetition code's.
- **No fault-tolerance experiment.** No encoded gate, no repeated syndrome
  measurement (each run extracts one round), no threshold, and syndrome extraction is
  assumed perfect in every code measurement above.
- **No hardware adapter.** The IR is not a device model, and no backend emits pulses
  or receives them.
- **Equivalence above the dense bound is Clifford-only.** A 20-qubit non-Clifford
  candidate gets `UNRESOLVED` from every tier, which is why discovery admission
  refuses it rather than passing it.
- **One discovery engine, and it is a rewriter.** `apedp_dyn/vqc/clifford_search.py`
  proposes from a fixed list of semantics-preserving Clifford rewrites, and it cannot
  propose anything those rules cannot reach (§8.1). It is not a synthesiser and not a
  genetic algorithm: nothing in `apedp/` or `apedp_dyn/quantum_compilation/search.py`
  emits these candidates, so a GA, GP or RIME engine over this IR is still unbuilt.
- **No general circuit discovery integration.** The DQI search
  (`apedp_dyn/quantum_compilation/search.py`) still emits its own candidate type and
  its own IR; the two stacks are joined only by this rewrite engine, not by a common
  candidate type.
- **No general noise.** The IR's noise operations are Pauli/Pauli-mixture
  trajectories; non-Pauli channels are available only on the exact density-matrix
  tier, which is capped at 12 qubits.

## 11. Pins

`tests/test_vqc_core.py` (63 tests) is the acceptance suite: the IR round-trip and
the twelve malformed-program rejections; Bell, GHZ and teleportation against
independently computed states; reset semantics including the entangled case; the
feedforward-dropped counterexample; five channels on five probabilities with
analytic values and Kraus completeness; seeded trajectory reproducibility and the
expected decorrelation; every single-error syndrome and both preparations; the
two-error miscorrection; the logical rate against `3p² - 2p³`; and the five receipt
verdicts above. `tests/test_vqc_qasm.py` (15 tests) covers the interchange table,
including every refusal; `tests/test_vqc_phase0_inventory.py` (6 tests) checks that
every path the inventory names exists and that the frozen DQI baseline is unchanged.

`tests/test_vqc_codes.py` (46 tests) is the code suite: every single `X`/`Z` error
on every qubit of both repetition codes, the two-error miscorrection each of them
fails, the phase-flip rate against `3p² - 2p³`, the five-qubit encoder against a
stabilizer-projection reference, all fifteen single-qubit Paulis to distinct
syndromes, every single error corrected on both preparations, a weight-2 logical
flip, and the enumerated prediction (including that it is a rational with
denominator `3**5`, i.e. arithmetic rather than a fit).

`tests/test_vqc_equivalence.py` (20 tests) pins the table in §5, including the
assertion that the rejected phase candidate reproduces every Z-basis probability
first, the metric itself (`I` vs `Z` = `sqrt(2)`, the 2.0 entry deviation reported
separately, the `[0, sqrt(2)]` range), the claim separation, the controlled scope
(`RX(2π)` VALID standalone and INVALID under control, an identity VALID in both), and
the independent bit-arithmetic oracle both paths must match. `tests/test_vqc_stabilizer.py` (40
tests) pins §6: the same builder on both tiers for every outcome probability and
random-Clifford Pauli expectations, the 100-qubit generators against the closed form,
the sampled counts and their five-sigma band, the X-basis even-parity distribution
with no collisions, teleportation with feedforward on both tiers, each named refusal
in the domain table, the structural validation (`validate()`, repeated
measurement, `Y` branches), and the exact-distribution limit: past twelve qubits both
engines refuse to enumerate rather than allocate, and the refusal points at
`probability_of_bitstring` and sampling, which stay exact.

`tests/test_vqc_clifford_equivalence.py` (19 tests) pins §5.1: the images against the
dense `U P U†` on random Clifford circuits, the two branches agreeing on every pair
both can decide, the 100-qubit global phase accepted and the 100-qubit mutation named,
chain-versus-star decided oppositely at the two scopes, `S` versus `S†` as the
one-qubit version of the same distinction, and the domain edges (non-Clifford gate,
measured candidate, wrong qubit count, unmet claim, machine-readable report).
`tests/test_vqc_decoders.py` (18 tests) pins §7: matching against the lookup table and
against a brute-force minimum over all 64 distance-7 syndromes, exactness up to
`(d-1)/2` and miscorrection at distance-7 weight 4, the refusals, the circuits built
from two decoders producing identical counts, and the measured rates against the
enumerated prediction for distances 3, 5 and 7. `tests/test_vqc_discovery.py` (13
tests) pins §8, including that an empty check set is `INSUFFICIENT` evidence and
explicitly unpromotable, that the four questions are four fields, that an underpowered
distribution claim and an uncheckable candidate pair are both refused rather than
passed, and that a phase which is free standalone is refused as a subroutine at 2
qubits and `UNRESOLVED` above the dense bound. `tests/test_vqc_clifford_discovery.py`
(15 tests) pins §8.1: every proposed rewrite certified for five random Clifford
circuits, the search reaching the 12-qubit ground truth, mutation refusals carrying
the named image, and the 100-qubit record's totals, admission and hashes. `tests/test_vqc_stabilizer_scaling.py` (7 tests) pins §6.1: the record's
hashes, its four families at four sizes, every check passed, the representation cost
(`n²` labels against `2**n` amplitudes) and a re-derivation of the 100-qubit rows
from the script that produced them.

`tests/test_vqc_qcec_equivalence.py` (33 tests) pins §8.5: every pair's construction
and relation, this package's verdict against every declared expectation on all three
scopes, the tolerance ladder's distances as the halved angles and its two edges
around `1e-9`, the state pair decided `VALID` as a state and `INVALID` as a unitary
with zero overlap, the diagonal pair whose Z-basis probabilities agree to 1e-12 and
whose unitaries do not, the classifier's five outcomes and the one it must never
invent (`CONTRADICTION` on a pair both tools should call equal), the summary
recomputing from the rows and noticing a tampered verdict, the Qiskit gate table being
total over `GATE_SPECS`, and the record's own rows: hashes, three scopes, the corpus
SHA-256, the repeats, and no contradiction.

`tests/test_vqc_marginals.py` (26 tests) pins the wide referee: every marginal against
a dense partial trace of a state built by projecting a full-support vector onto the
specification's eigenspaces (an independent route through real Pauli matrices, which is
what caught the first version's Y-generator sign), the generator signs honoured rather
than read as labels, the graph-state uniformity as arithmetic, a 32-qubit marginal
cheap enough to pin a cost bound on (linear algebra, not enumeration), the chi-square
survival function against textbook critical values, the Bonferroni product as
arithmetic (`min(1, m * p)`), a mis-wired wire caught by the subsets that contain it and
only those, a subset with too few expected counts abstaining as `UNRESOLVED` rather
than rejecting, a sample set from another specification refused, two correct engines
agreeing while a mis-wired one disagrees, and the blind spot measured: samples of a
different graph state pass this referee.

`tests/test_vqc_memory.py` (9 tests) pins the memory harness: the counter is a real peak,
the reset makes a window measurable, a target round-trips hash-identically through its
printed stabilizers (declared support included), a measured engine's count hash equals
the calling process's own, Stim's library sits inside the floor rather than in the
sample, a platform that cannot reset the counter abstains with a reason, a child that
sampled a different specification is refused, and two runs agree on everything but the
bytes and the clock.

`tests/test_vqc_external_comparison.py` (51 tests) pins §8.4: a foreign circuit is
decided by our referee before its count is quoted, a foreign claim the circuit does not
meet is `UNMET` and not `INVALID`, a wrong circuit is `INVALID` with the failing
generator named, the four-field policy is the same one (compared against the local
path), the string-order difference is pinned on a non-palindromic target *and* the
naive order is shown to produce the mirror image, an absent package is an abstention
with a reason and never a comparison, and the record's own ratios are recomputable from
its rows — including the shared corpus, whose rows share one specification hash, verify
on both engines, keep the abstentions unscored, and whose summary is recomputed from
the rows in the test rather than trusted: win counts, medians, quartiles, ranges, the
width breakdown, the structure spread and the bootstrap interval on each median ratio.
That test also pins the sample itself — every family cell drawn at every declared width
with the declared replicates, and the record's `design` block equal to the rows it
produced — so a summary cannot be reported for a sample that was not taken. The routed
sample has its own pins: 75 instances whose routed circuit is hash-identical to the
circuit the synthesis row measured, each engine's own router named on its own row,
Qiskit's seed and optimisation level equal to the task's, both routed circuits `VALID`,
and every routed median, quartile, win count, swap figure and share recomputed from the
rows (per family and per width). The third-party referee has its own pins in
`tests/test_vqc_foreign_referee.py` (11): it decides the same specification, it fails a
dropped gate, a circuit for a different target, an inverted phase and a mis-wired qubit,
it refuses a non-Clifford gate, a measurement, a width mismatch and an absent Stim with a
reason and never counts one as a loss, and it decides a 127-wire routed circuit. On the
record side the pins require every matched row to carry both referees' verdicts, the
two to agree, the summary to recompute from the rows, and the abstention paths to carry
their reasons.
`tests/test_vqc_corpus.py` (11 tests) pins the corpus itself: reproducible from its
seeds hash for hash, every seed equal to the one the documented rule yields, one label
per cell, a smaller sample being the same sample restricted, actually regular and
connected when read back out of the stabilizers, not a ring or lattice, and split
between buildable graph states and stabilizer states for which the grammar refuses with
a reason. The Stim rows are pinned
the way the synthesis rows are: same specification hash on both sides, both sample sets
`CONSISTENT` with the exact distribution on a reproducibility-checked count hash, the
pair `AGREE`, a deliberately skewed sample set reported `INCONSISTENT` rather than
smoothed over, a width where the full distribution is not enumerable refused by that
path rather than compared against an approximation (and refereed by exact marginals
instead, below), and a missing Stim refused with a reason instead of silently skipped.
The wide rows are pinned as narrowly as they are claimed: every row above 12 qubits,
every subset decision carrying the Bonferroni product it was taken at, no subset
abstaining silently, GHZ's pair marginals keeping the two-outcome correlation, the
ring-graph row showing all-uniform marginals instead of being left out, and the prose
pinned not to claim the whole distribution. The routing rows are
pinned the same way: Qiskit's transpiler on a given coupling map produces a circuit
that verifies against the target relabelled by its final layout, its reported depth
equals the re-imported depth, a circuit wider than the device is refused, and the
record's device graph is re-derived from the same snapshot and compared hash for hash.

`tests/test_vqc_synthesis.py` (57 tests) pins §8.3, §8.3.1 and §8.3.4: a specification that is not a pure
state is refused (anticommuting, rank-deficient, imaginary phase, support not summing
to one), the tableau and projected-vector readings of a target agree at every width
both reach, all five GHZ construction families and the ring family reproduce their
targets, the registered near-miss and four mutated constructions are refused with the
failing generator named, the ranking and the Pareto frontier follow the measured rows,
a valid construction with an unmet claim stays `VALID`/`UNMET` and is not promoted, the
claim rule is pinned against the equivalence referee's own implementation, the dense
tier decides a non-Clifford construction and abstains above its bound, a 1,000-qubit
specification is decided exactly, and both referees reach the same verdict at every
width both can decide — including on a wrong construction and on one differing by a
single global phase — the two-dimensional cluster and irregular graph targets read
their wiring out of their stabilizers, every valid graph family prepares its target,
the minimum schedule attains the degree floor at every size, the wrong graph
constructions are refused with the failing endpoint stabilizers named, the two basis
orders are pinned on a state that is not symmetric under reversing its wires, and the
cost probe is pinned as measuring the referee's own decision exactly inside the bound,
as quoting a price and no verdict above it, and as refusing a circuit of the wrong
width.
`tests/test_vqc_colouring.py` (14 tests) pins §8.3.3's scheduler: a `Δ`-colouring is
reported optimal with the degree argument and a real matching per layer, the odd cycle
and `K5` are *proved* class 2 by the layer-capacity bound at **zero** search nodes, the
Petersen graph is proved class 2 by an **exhausted** search (the bound is tight there),
`K5` saves **2** layers against the greedy peel (7 → 5), `K9` and `K11` are refuted at
budget 1 and at budget 200 alike and return **9** and **11** layers against the peel's
15 with `search_nodes == 0`, a one-node budget on the Petersen graph returns the
constructed `Δ + 1` schedule marked `optimal=False` rather than the peel and without
claiming a minimum, the layer bound is cross-checked against the exhaustive `Δ` search
on **every** graph of at most five vertices so a refutation is never wrong, the
construction is checked to be a valid bounded colouring of every one of those graphs
(1,094) and of the structured families (`K9`, a 9-cycle, the Petersen graph, a 4-cube,
`K5,5`), a construction that fails its own check falls back to the peel labelled
`construction_refused`, self-loops, duplicate and out-of-range edges are refused, and
every reported layer is checked to be a matching whose union is the edge set.
`tests/test_vqc_placement.py` (34 tests) pins §8.3.2: every
two-qubit gate of a routed circuit is between adjacent nodes, a routed circuit still
prepares the specification by both tiers, all-to-all reproduces the logical depth, a
line reproduces a chain for free, a tree pays for its connectivity, the restore ladder
is reported rather than hidden, an unreachable graph or an unplaceable three-qubit gate
is refused, no placement is ever cheaper than the abstract circuit, and — for the
lookahead router — every gate is still between neighbours, every routed circuit still
prepares the specification by both tiers, the circuit is deterministic, and a chain on
a lattice is shallower and cheaper than the greedy router's own result.
`tests/test_vqc_synthesis_record.py` (32 tests) pins the record: no reference circuit
in Experiment A, every 1,000-qubit row verified, the four refusals named, Experiment D's
both routers (**the lookahead never worse on depth, swaps or two-qubit gates on any of
the eight comparisons, with the savings recomputable from the two columns**), **Experiment E's six graph specifications** (edge counts,
degree floors, schedules that tie on every resource except depth, the minimum's floor, the
three edge-level refusals, the cross-target discrimination and the referees' agreement
on asymmetric graphs), **the dense tier's measured ceiling** (every width inside the
bound decided `VALID`, every width above it `UNRESOLVED`, the `16 * 4**n` operator
model against the traced peak, the fourfold growth per qubit, the declared 8 GiB budget
and the width at which it is exceeded), **the scheduler's own accounting** (every
Experiment E target `optimal` with `layers_saved == greedy - minimum`, the total
recomputable and equal to **0** on those six, and the class-2 rows proved `Δ + 1` with
`K5`'s 2-layer saving recomputable), **the routing block** (90 placements, 72
comparisons recomputable from the rows, every placed circuit `VALID`, the lattice at
zero swaps and exactly 1.00x, and no comparison inverted), **the per-schedule wall clock** (every row's
`synthesis_seconds` and `verification_seconds` present and positive, each the median of
three recorded samples, with three timed lattice sizes and verification dearer than
synthesis at every one), and the SHA-256 of the script and six
modules that produced it.

`tests/test_vqc_native_parity.py` (45 tests) pins §8.2: the derived tables entry by
entry in both languages, the images and generators of a fourteen-circuit corpus,
the exact distributions, a relative phase seen identically by both engines, the
reverse (a global phase seen by neither), the fallback with the extension removed,
the refusal to downgrade a requested native run, a corrupted table refused rather
than adjudicated, the parity-limit split, and a sampler corpus of measured circuits —
teleportation, a mid-circuit reset with feedforward and a two-qubit condition, and both
condition polarities on one bit — whose allowed outcome *sets* are deterministic, so a
reset that is not applied or a condition read in the wrong order is visible rather than
statistical. A classically conditioned gate is refused by `assert_parity` (it is not a
unitary, so it has no image table) and a `ccx` is refused by both engines by name. `tests/test_vqc_native_parity_record.py`
(7 tests) pins the record's schema, its `agrees` flags, the SHA-256 of the script,
the seven Rust sources and the three Python modules it measured, and that every
case's circuit can be rebuilt to its recorded hash.

A future edit that lets a serialisation round-trip change a hash, makes a channel
non-trace-preserving, promotes an underpowered run to `VALID`, accepts a
candidate with a dirty ancilla, or lets the native engine disagree with the
reference without failing fails there.

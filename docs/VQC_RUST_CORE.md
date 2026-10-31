# APEDP-VQC: what is Rust-first, what is Python, and what was measured

**Status:** experimental · **Crate:** `rust/apedp-vqc` (module `apedp_vqc`) ·
**Bridge:** `apedp_dyn/vqc/native.py` · **Record:**
`benchmarks/quantum_compilation/vqc_native_parity.json` · **Pins:**
`tests/test_vqc_native_parity.py` (45), `tests/test_vqc_native_parity_record.py` (7)

This document answers one question with an inventory: **which parts of VQC are
Rust-first, which parts are deliberately still Python, and what evidence exists
for the split.** It is the answer to the architecture review whose five
engineering rules appear below as checked items rather than aspirations.

## 1. The split

| capability | language | why there | state |
|---|---|---|---|
| Pauli strings, phases mod 4, the derived conjugation tables | **Rust**, Python fallback | bit vectors and integer phase arithmetic, called once per generator per gate | done (`pauli.rs`, `bits.rs`) |
| Stabilizer tableau: gates, measurement, reset, sampling | **Rust**, Python fallback | the shot loop is the dominant cost at 100+ qubits | done (`stabilizer.rs`, `runtime.rs`) |
| Exact Z-basis distribution and Pauli expectations | **Rust**, Python fallback | `2**n` chained measurements, no amplitudes | done |
| Matrix-free Clifford equivalence (the `2n` images) | **Rust**, Python fallback | `2n x ops` label updates; the review's "suitable for Rust" module | done (`equivalence.rs`) |
| Circuit IR parsing | **Rust**, against the *Python* canonical JSON | one specification, two readers | done (`ir.rs`) |
| The referee and every verdict | **Python** | independence: the fast path must not mark its own homework | unchanged |
| Discovery, GA/GP/RIME, candidate generation | **Python** | research surface, changes weekly | unchanged |
| Dense statevector tier | **Python** | 26-qubit cap, NumPy is already vectorised | not ported |
| Decoders (lookup, MWPM) | **Python** | dependencies, not yet hot | milestone 3, not built |
| Noise engine (density matrices, trajectories) | **Python** | correctness-first work | milestone 5, not built |

Two rows matter more than the rest. The referee stayed in Python on purpose: Rust
computes images, `equivalence.py` compares them and writes `VALID`/`INVALID`/
`UNRESOLVED`. And **nothing is Rust-only**: with the wheel absent, every entry
point returns `None` and the Python tier answers, so the package still works on a
machine with no Rust toolchain.

## 2. One specification, not two

`ir.rs` parses the JSON that `Circuit.to_json()` already emits -- the same schema
version, the same operation tags, the same gate names. The Rust crate has no
builder, no gate table of its own and no second notion of a condition; a gate it
does not know is refused by name at execution time, which is when
`stabilizer.py` refuses it too. A parity test caught the first version of the
condition parser guessing a `[[bit, value]]` shape when the writer emits
`{"bit": i, "value": bool}` -- which is the argument for reading the real format
rather than a remembered one.

The conjugation tables are *derived* on both sides from the same 2x2 and 4x4 gate
matrices at first use. They are not transcribed sign patterns, and
`test_the_native_conjugation_tables_are_the_python_tiers_tables` compares all 24
one-qubit rules and all 48 two-qubit rules as integers.

## 3. What was measured

Identical frozen circuits, both engines, same process, warm-up call excluded,
median of five with the interquartile range recorded (`scripts/measure_vqc_rust_parity.py`;
Python 3.12.3 and NumPy 2.5.3, Linux x86-64, rustc 1.99.0, release build compiled
before the run so no compilation is in these numbers):

| case | Python | Rust | ratio | Rust IQR | agrees |
|---|---|---|---|---|---|
| images, 100-qubit GHZ ladder (200 images) | 17.84 ms | **4.65 ms** | 3.8x | 0.50 ms | exact |
| images, 400-qubit ladder (800 images) | 481.94 ms | **66.96 ms** | 7.2x | 12.40 ms | exact |
| sampling, 100 qubits x 256 shots | 3045.63 ms | **700.11 ms** | 4.4x | 117.19 ms | same support |
| exact distribution, 10 qubits (1024 outcomes) | 37.37 ms | **2.52 ms** | 14.8x | 0.08 ms | exact |
| Clifford adjudication, 100-qubit pair | 41.16 ms | **11.63 ms** | 3.5x | 3.22 ms | same verdict |

The ratios move between runs on this host -- the 100-qubit image row has been
measured anywhere from 3.0x to 4.8x -- so the record carries the raw samples and
the IQR for each side rather than a single ratio to quote. The 400-qubit row (ten
times longer, IQR 12 ms) is the stable one for scaling; the 100-qubit row is there
because that width is the one the platform's own claims are stated at. The IQR is
a spread over five repeats in one process on one host: it is not a confidence
interval, and it says nothing about another machine.

The sampling row is 4.4x and not 40x because the Python tier's cost is dominated
by Python-level per-shot object churn, which Rust removes, but the shot loop still
touches every op: the honest statement is "4.4x at 100 qubits with 256 shots on
this machine", not "Rust makes stabilizer simulation fast".

The scaling record shows the same effect where it matters most. Regenerated with
the native engine, deciding two **1,000-qubit** circuits (7,500 and 8,190 gates,
2,000 images each) takes **1.94 s**, against about a minute on the Python image
path, and the discovery loop at 100 qubits dropped from 5.9 s to 2.6 s with the
same 222 -> 100 operations and the same 76 certified steps
(`benchmarks/quantum_compilation/vqc_stabilizer_scaling.json`,
`vqc_clifford_discovery.json`).

Exact agreement is not sampled: `parity_corpus` in the record re-derives the
images, generators and (where enumerable) the whole distribution in Python and
compares them entry by entry, and the corpus is run in the test suite too
(45 tests), including measured circuits, conditional feedforward, the
teleportation circuit, five seeded random Clifford circuits, a non-Clifford
refusal, a noise refusal and the 100-qubit GHZ chain. The measured part of the
corpus is a *sampler* corpus with deterministic allowed-outcome sets: a mid-circuit
reset followed by feedforward and a two-qubit condition, and both condition
polarities on one measured bit. Those sets are what a reset that is not applied or a
condition read in the wrong order changes, so the comparison does not need a shot
budget to be decisive. A classically conditioned gate is refused by `assert_parity`
as well — it is not a unitary, so it has no image table, and tagging it as an
ordinary gate is how the harness would have compared two different circuits.

## 4. The five rules, as code

| rule | how it is implemented | where |
|---|---|---|
| one canonical circuit specification | Rust parses the Python canonical JSON; unknown operations refused by name | `ir.rs`, `test_the_native_conjugation_tables_are_the_python_tiers_tables` |
| no duplicated undocumented semantics | tables derived from matrices on both sides; the gate set is asserted equal in both languages | `pauli.rs`, `test_the_clifford_gate_set_is_the_same_in_both_languages` |
| Rust speed must not compromise correctness | every entry point returns `None` when it cannot answer; `assert_parity()` recomputes in Python and raises on disagreement; a corrupted table is refused, not adjudicated | `native.py`, `test_a_corrupted_native_table_is_refused_rather_than_adjudicated` |
| verification stays independent | images come from Rust, verdicts from Python; below `PARITY_LIMIT` (24) qubits the table is re-derived in Python, above it the report says `image_verified_by: "suite"` | `equivalence.py`, `test_above_the_parity_limit_the_report_says_suite_not_verified` |
| benchmark before claiming superiority | the record is regenerated by one command, pins both implementations' SHA-256, asserts the Python timings never touch the fast path, and refuses to write if any case disagrees | `scripts/measure_vqc_rust_parity.py` |

## 5. Provenance is recorded, not asserted

A run says which engine executed it. `RunResult.engine` is `"rust"` or
`"python"`; `CliffordEquivalenceReport.image_engine` and `image_verified_by` say
who conjugated the Paulis and who checked the result. `engine="rust"` is a
*request*: if the extension is missing or declines, the call raises rather than
quietly downgrading, so a report cannot label a Python run native.
`native.tallies()` counts `rust`/`python`/`fallback` calls, `APEDP_VQC_PYTHON_ONLY=1`
pins the reference path in-process, and the benchmark asserts the fast-path
counter is zero around every Python timing.

## 6. Building and verifying

```bash
make vqc-native       # maturin develop --release --manifest-path rust/apedp-vqc/Cargo.toml
cargo test --manifest-path rust/apedp-vqc/Cargo.toml --no-default-features   # 31 pure-Rust tests
make vqc-parity       # regenerate benchmarks/quantum_compilation/vqc_native_parity.json
pytest tests/test_vqc_native_parity.py tests/test_vqc_native_parity_record.py
```

`cargo test --no-default-features` runs the Pauli algebra, the tableau, the IR
parser, the runtime and the images without a Python interpreter, because PyO3 is
behind the optional `python` feature. That is the layer where a sign error is
cheapest to catch.

## 7. What is not built

* **Decoders (milestone 3)** and the **noise engine (milestone 5)** are Python
  only. MWPM is exponential in the number of defects and is stated as such in
  `decoders.py`; porting it is a separate piece of work with its own benchmarks.
* **The dense statevector tier (milestone 4)** is not ported. It is capped at
  ~26 qubits and already backed by NumPy; a Rust port would be a rewrite for a
  narrow gain, and the review's sequence puts it after the decoders for the same
  reason.
* **The Rust crate is one crate with seven modules**, not the eight crates the
  review sketched. Splitting `vqc-ir`/`pauli-core`/`stabilizer`/
  `clifford-equivalence` into crates that share one build and one test corpus
  adds packaging work without changing a single measurement; the module
  boundaries are in place for a later split.
* **No memory ceiling is claimed.** Peak RSS was not measured for the native
  path, so the record reports time only. Adding a measured memory column is the
  obvious next step for the resource-accounting half of the review.
* **Parallelism is not used.** The shot loop is single-threaded; `rayon` and
  per-shot work splitting are an obvious 4-8x and an equally obvious source of
  nondeterminism, so they are deliberately left out until a record can pin seeds
  and ordering.

## 8. Cross-references

* [`VQC_CORE.md`](VQC_CORE.md) -- the platform's own account of the tiers, the
  referee and the discovery loop, with §6.1 for the stabilizer measurements.
* [`VQC_PHASE0_CAPABILITY_INVENTORY.md`](VQC_PHASE0_CAPABILITY_INVENTORY.md) --
  the capability inventory this split is classified against.
* [`DQI_OFFICIAL_COMPATIBILITY.md`](DQI_OFFICIAL_COMPATIBILITY.md) -- the DQI
  evidence that the native work does not touch, frozen by SHA-256 in
  `benchmarks/quantum_compilation/vqc_phase0_baseline.json`.

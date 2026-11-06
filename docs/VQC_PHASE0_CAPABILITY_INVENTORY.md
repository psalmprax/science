# APEDP-VQC Phase 0: what exists, what is reused, what has to be built

**Status:** inventory, taken before further Phase-1 work · **Scope:** the DQI
subsystem in this repository, classified against the seven VQC capability areas ·
**Evidence:** the file paths below, plus
`benchmarks/quantum_compilation/vqc_phase0_baseline.json` and
`tests/test_vqc_phase0_inventory.py`.

Phase 0 has one rule that matters: **the DQI evidence does not move.** This
inventory changed no record, no script and no threshold. The thirteen DQI records
are frozen by SHA-256 in `benchmarks/quantum_compilation/vqc_phase0_baseline.json`,
and the test recomputes every hash, so a later change that reaches into the DQI
baseline fails loudly instead of silently re-basing it.

## 1. Decisions, and what each one means

| decision | meaning |
|---|---|
| **reuse** | the existing code is used as is, through its current interface |
| **adapt** | the existing code or its conventions are kept, but wrapped or generalised for VQC |
| **build** | nothing usable existed; it was written in `apedp_dyn/vqc/` |
| **retain** | kept in place as one backend or one evidence source, deliberately *not* made the architecture |
| **not connected** | exists, is relevant, and is not wired to VQC yet |

## 2. The inventory

| capability | existing artifact | decision | where it stands now |
|---|---|---|---|
| DQI circuit construction | `scripts/gf54_mul_reduced.py`, `apedp_dyn/quantum_compilation/dqi_builder.py`, `scripts/run_mr_gated_arms*.py` | **build** a real IR; **port the semantics** | the emitters write into `kickmix`'s C++ builder, not into a neutral representation, so they cannot be reused as an IR. Their *semantics* are what the IR reproduces: measurement-with-reset clears, classically conditioned correction gates, declared workspace (`apedp_dyn/vqc/circuit.py`) |
| Gate vocabulary | `kickmix` opcodes (`HMR`, `CCX`, `CSWAP`, …), `statevector_sim.py` gate set | **build** one table | `apedp_dyn/vqc/gates.py` defines the unitaries; both backends apply gates through it, so a name cannot mean two things |
| Resource accounting | `keepalive`-free `kickmix` `operation_statistics`, the `measure_*` scripts, `docs/NONMULTIPLY_LEDGER.md` | **adapt** | `Circuit.resources()` counts IR operations (gates by name, two-qubit gates, measurements, resets, noise, depth). Toffoli-like accounting stays `kickmix`'s: the pinned extension is the authority for the competition workload, and VQC does not re-derive it |
| Candidate verification | `apedp_dyn/quantum_compilation/contracts.py`, `candidates.py` (`CircuitCandidate`, `EvidenceIncompleteError`), `referee.py`, `ledger.py` | **adapt** the discipline | `apedp_dyn/vqc/receipts.py` keeps the three properties that made the DQI version work: hash-pinned identity, explicit evidence fields, and an abstention state (`UNRESOLVED`, the analogue of `INSUFFICIENT_EVIDENCE`) |
| Experiment records | `benchmarks/quantum_compilation/*.json` and the `scripts/measure_*` producers | **reuse the convention** | VQC receipts record identity, provenance, per-check statistics, resources, reproducibility and a verdict; the DQI records are untouched and frozen |
| Pinned competition compiler | `apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix` | **retain** as one backend | it stays the only thing that may produce a DQI receipt, and it is not the platform's execution model. VQC does not import it and its absence does not break VQC |
| Exact amplitude tier | `apedp_dyn/quantum_compilation/statevector_sim.py` | **reuse** | `apedp_dyn/vqc/backends.py`'s `StatevectorBackend` executes IR ops through it; the `MR` collapse-and-reset semantics and the classical-bit machinery came from there rather than being rewritten |
| Noise | none existed for general circuits (only the amplitude-tier licence probe) | **build** | `apedp_dyn/vqc/noise.py`: exact density-matrix channels with trace/positivity/Kraus checks; the IR's `Noise` op is the sampled trajectory counterpart |
| Logical qubit | none | **build** (done, three codes + a decoder interface) | `apedp_dyn/vqc/logical.py`: the distance-3 bit- and phase-flip repetition codes with the exact `3p² - 2p³`, plus a distance-parameterised builder; `apedp_dyn/vqc/five_qubit.py`: the perfect five-qubit code (all 15 single-qubit Paulis to distinct syndromes, prediction enumerated over all 1,024 patterns); `apedp_dyn/vqc/decoders.py`: the decoder interface with a lookup table and **exact minimum-weight perfect matching** (distances 3/5/7 measured against enumerated predictions: 0.0258/0.0050/0.0033 at p = 0.1) |
| Circuit discovery | `apedp_dyn/quantum_compilation/search.py`, `multi_structure_search.py`, `apedp/evolutionary_search.py`, RIME proposers | **interface built; one engine connected** | `apedp_dyn/vqc/discovery.py` accepts a `CircuitCandidate` (candidate, reference, resource claims, optional declared distribution), routes it through the equivalence referee and the distribution receipt, and refuses `promote()` for anything that is not VALID — including `UNRESOLVED` — and anything with `INSUFFICIENT` evidence (an empty check set). `apedp_dyn/vqc/clifford_search.py` is the first engine that emits this IR: local Clifford rewrites, every accepted step certified by the referee against the original reference, ranked by `(operations, depth)`, then admitted. The DQI search and the equation engines still do not emit this IR |
| Optimisation / Pareto cost | DQI-specific optimisers and the `(T, Q, D, G₂, M)` accounting scattered across records | **rewrite engine built; no Pareto front** | `apedp_dyn/vqc/clifford_search.py` applies semantics-preserving rewrites (self-inverse cancellation, commutation past disjoint gates, `s`/`Sdg` algebra, `h z h = x`, `h x h = z`) and certifies each step, so an "optimisation" here cannot change the unitary — measured at 100 qubits: 222 operations → 100 in 7.1 s, 76 certified steps, promoted on `MET` + `SUFFICIENT` evidence. Greedy, local, Clifford-only; no cost vector, no Pareto front, and no rule that changes the gate vocabulary |
| Equivalence and phase referee | verifier's per-shot value/phase rule; `VQC_AMPLITUDE_TIER.md` amplitude comparisons | **build** (done, two tiers) | receipts still compare *distributions* against a declared expectation at 4σ; `apedp_dyn/vqc/equivalence.py` adds unitary equivalence up to **one global phase** — dense (materialise both unitaries, near ten qubits) and Clifford (compare the conjugation action on the Pauli generators, exact at **any** size, so 100-qubit pairs are decided). `CCX` versus `CCX; Z` is identical in statistics and INVALID at distance 1.414214 (`min_phi ||U_c - e^(i·phi) U_r||_F / sqrt(2^n)`, minimised in closed form; the max-entry deviation at that phase, 2.0, is reported separately and is not a minimised quantity); a resource claim is reported as its own `claim_verdict` (`MET`/`UNMET`) so an equivalent-but-larger candidate stays VALID and is refused as non-improving rather than as wrong; a 100-qubit global phase is VALID and a 100-qubit changed gate is INVALID with the image named; a size neither tier can decide is `UNRESOLVED`, not `INVALID`; and two scopes are offered — `as_subroutine=True` compares `diag(I, U)` instead, so `RX(2π)` is VALID standalone and INVALID under control (dense-only; above ten qubits the controlled question is `UNRESOLVED`). At **1,000 qubits** two different circuits of 7,500 and 8,190 gates are decided from 2,000 images (VALID) and a one-gate mutation is INVALID with the image named — **1.9 s each** with the native image engine, down from about a minute on the Python path, recorded in `benchmarks/quantum_compilation/vqc_stabilizer_scaling.json` |
| Bigger simulation backends | none (dense statevector only, 26-qubit cap) | **build** (stabilizer done) | `apedp_dyn/vqc/stabilizer.py`: exact Clifford execution over `n` signed Pauli generators, cross-checked against the dense tier on 8,188 outcome probabilities and 22 Pauli strings (max deviation `1.1e-16`), and carrying a 100-qubit GHZ chain the dense tier cannot hold. Scaling is measured to **1,000 qubits** in four families (`benchmarks/quantum_compilation/vqc_stabilizer_scaling.json`): GHZ chain and star, a ring graph state against its defining generators, and seeded random Clifford layers; 1,000-qubit GHZ builds in 0.47 s with 0.62 s per sample, a 7,500-gate random Clifford in 14.2 s, and the `n²` label cost is recorded against `2**n` amplitudes. MPS / tensor-network tiers do not exist and nothing in VQC claims them |
| Standard interchange | none | **build** (Phase 1, task 6) | `apedp_dyn/vqc/qasm.py`: OpenQASM 2 import/export, with explicit refusals for noise, sub-register conditions and user-defined gates |
| Native execution core | `apedp_dyn/vqc/stabilizer.py`, `equivalence.py` (the reference implementations) | **build** in Rust, Rust-first with the Python tier as fallback | `rust/apedp-vqc` (crate `apedp-vqc`, import name `apedp_vqc`) implements `pauli-core`, `stabilizer`, `clifford-equivalence` and `resource/IR` parsing; `apedp_dyn/vqc/native.py` is the bridge. It parses the **same canonical circuit JSON** and derives its conjugation tables from the same gate matrices, so there is one specification rather than two. `StabilizerBackend` and `adjudicate_clifford_equivalence` use it by default and fall back to Python when it is absent or declines; the **verdict stays in Python**, which re-derives the image table below `PARITY_LIMIT` (24) qubits and reports `image_verified_by: "suite"` above it. Measured on identical frozen inputs: 3.5x to 14.8x with the hardware, compiler, build and per-side IQR recorded (and a 1,000-qubit adjudication in 1.94 s against about a minute) (`benchmarks/quantum_compilation/vqc_native_parity.json`); the dense statevector tier, decoders, noise engine and discovery loop remain Python |
| Synthesis from a specification | none (every engine above needs a reference circuit) | **build** (first experiment: GHZ and ring targets) | `apedp_dyn/vqc/synthesis.py`: a `TargetSpecification` is a qubit count plus signed Pauli stabilizers, validated as a pure state before any search runs, with **no reference circuit in the object**. Two referees decide a construction: the stabilizer tier by `phase_of(generator) == +1` at any width, and the dense tier by comparing amplitudes against the state **projected out of the specification's stabilizers** (non-Clifford constructions, ≤ 10 qubits). At 100 qubits five GHZ constructions verify (chain 100 operations/depth 100/span 1, star 100/100/99, balanced tree 100/**12**/50, random tree 100/14/90), the balanced tree is promoted, four wrong constructions and one near-miss family are refused with the failing generator named, and a second target (ring graph state) is built and promoted. Scaled to 1,000 qubits: 24 rows, all verified in 1.3-1.7 s each. Graph states are specifications too: a two-dimensional cluster (`cluster_state_target`, the grid-graph state on lattice vertices) and an irregular graph (`seeded_graph_target`) name every edge, so the families fix the wiring and choose only the schedule — five schedules at 9, 16, 25, 64, 100, 400 and 900 qubits, all verified (the 9-qubit one is `K9`, the class-2 case inside Experiment E: 36 edges, minimum 9 layers one above the degree floor, the peel's 15 measured against it) (the 400-qubit target is a 20x20 lattice with 760 edges and the 900-qubit one a 30x30 lattice with 1,740; each schedule records its own synthesis and verification wall clock, the **median of three repeats** of each phase on the same rebuilt circuit (all three samples kept on the row), and the three large lattices form a timing ladder — build 0.001-0.002 s and decide 0.015-0.017 s at 100 qubits, 0.020-0.032 / 0.233-0.288 at 400, 0.101-0.160 / 1.269-1.369 at 900, with deciding about ten times building at every size), with a **minimum edge colouring** (`apedp_dyn/vqc/colouring.py`: a complete `Δ`-colouring search, optimal by the degree bound, or a refutation of `Δ` that proves the graph class 2 and makes the `Δ + 1` colouring optimal by Vizing's bound — the refutation being either a layer-capacity count (more edges than `Δ` matchings of `⌊V/2⌋` can hold: `K9`'s 36 edges do not fit into eight layers of four) or an exhausted search, and the `Δ + 1` colouring being *constructed* by Vizing's fan argument rather than searched for) attaining the proven degree floor (`d` neighbours need `d` `cz` layers) and three edge-level wrong constructions refused with the failing endpoint stabilizers named. Those schedules are also placed on a line, a ring and the cluster's own lattice at the three smaller lattices (16, 100 and 400 qubits) by both routers: the lattice needs no SWAP (so the minimum's advantage is exactly the logical one), while on a line or ring the minimum schedule routes far better than the localized orders and the advantage widens (1.00x-219x, median 4.67x, no comparison inverted). The scheduler saves 0 layers on six of those seven targets — the greedy peel it replaced already attained `Δ` — and **6** on `K9` (15 → 9), plus **14** across the class-2 rows it adds: 2 on `K5` (7 → 5), 6 on `K9`, 4 on `K11` and 2 on `K13`, both odd complete graphs whose 15-layer peel is proved down to 9 and 11 by the layer-capacity bound (their `Δ` search needs more than five million nodes, so without that count the family could only be reported as undecided). The construction space is a **grammar**, not a general synthesiser; see `docs/VQC_SYNTHESIS.md` |
| Architecture-aware depth | none (logical depth was reported as if every qubit could interact with every other) | **build** (Experiment D) | `apedp_dyn/vqc/placement.py`: line, ring, lattice and all-to-all connectivity graphs, plus two SWAP routers. A routed circuit is the **same unitary on the same wires**, because the routing swaps are undone in reverse order at the end (the swap product telescopes to the identity), so the mapped circuit goes through the same specification referee as the abstract one. At 100 qubits the balanced tree's logical depth 12 becomes **220** on a 10x10 lattice (547 routing swaps), 253 on a ring and 311 on a line under the **greedy** router (second operand along a shortest path, no lookahead); the **lookahead** router schedules each candidate swap into the earliest layer its nodes are free and breaks ties by the next eight two-qubit gates, giving **84** on the lattice, 212 on the ring and 278 on a line, while the chain is depth 100 for free and 191 → **117** on a lattice. Both are reported row by row and neither reorders gates, so these are routing bounds, not floors; see `benchmarks/quantum_compilation/vqc_synthesis.json` (`experiment_d`) |
| External comparison | none (every earlier claim was measured against this package's own reference implementation) | **build** (matched specification, foreign circuit decided by our referee) | `apedp_dyn/vqc/external.py`: a `Submission` is another tool's circuit plus that tool's own operation count, and `adjudicate_submission` runs it through the same specification referee and the same four-field policy a local candidate faces. Qiskit 2.6.1 is wired because its `synth_circuit_from_stabilizers` takes the same signed Pauli strings this package holds; its output comes back through QASM 2 and re-imports. At 100 qubits: GHZ 99 two-qubit gates (ours, depth 12) against **4,950** (Qiskit, depth 395), ring state **100 against 100** two-qubit gates with depth 101 against 199, and on a random 30-qubit stabilizer state this package **abstains** while Qiskit synthesizes. **Qiskit's own transpiler is wired for routing**: on the recorded `fake_sherbrooke` coupling map (127 qubits, 144 edges), four logical circuits are routed by Qiskit's SABRE and by both of this package's routers on the same device graph, with all eight results decided `VALID` by the same referee (Qiskit's rows against the target relabelled by its final layout). Qiskit wins 2 of 4 routed-depth comparisons against each router — median ratio 1.47× against the lookahead and 1.32× against greedy — and its own synthesised GHZ circuit is one of its wins, so the result is split rather than a sweep. **Stim is wired for the execution task** (not synthesis): both engines are handed the same signed stabilizers and the same 40,000 shots, and both sample sets are decided against the exact 12-qubit distribution — all three targets `CONSISTENT`, both sampler pairs `AGREE`, with the throughput ratio (about three orders of magnitude at the median) recorded as an implementation comparison of a Python per-shot tableau clone against Stim's compiled C++ sampler, not a ceiling. **That ceiling is now removed for marginals** (`apedp_dyn/vqc/marginals.py`): the exact Z-basis distribution of a chosen subset of wires is derived from the specification by Gaussian elimination over GF(2), so the same two engines are sampled at 16, 24 and 32 qubits and each sample set is decided on 190 subsets (every adjacent pair plus three 10-wire projections) by a chi-square goodness-of-fit with a Bonferroni-corrected p-value — every subset `CONSISTENT` on both engines, no subset of disagreement, and the graph-state blind spot (a uniform Z-marginal cannot tell two graph states apart) recorded as the ring-graph row rather than omitted. A marginal decides a projection, not the whole distribution, which the record states beside the numbers. **Memory is measured on the same workloads** by `apedp_dyn/vqc/memory.py`: each engine samples in a fresh child interpreter whose peak-RSS counter is reset after its imports and after the specification is rebuilt, so the recorded number is the sampling step's own resident growth (median 0.8 MiB here against Stim's 3.0 at 12 qubits, 1.4 against 3.4 above the enumerable width) over an import floor of 164/169 MiB charged to neither engine; the child's count hash must equal the recorded run's, so the memory belongs to the samples in the record. The number that moves is the outcome table rather than the width — a 32-qubit GHZ grows 0.2 MiB, a 32-qubit random state with 20,000 distinct outcomes 8.6 against 10.0 MiB — so memory is comparable and not an order of magnitude either way, and it is not a ceiling claim. **QCEC is now wired for the equivalence task** rather than recorded as unavailable: `apedp_dyn/vqc/equivalence_corpus.py` holds eighteen pairs whose relation is fixed by the construction that built them, and both referees are asked on each — this package's on all three scopes it offers (standalone, controlled, prepared state), QCEC's on the standalone unitary one, which is the only question it decides. The families isolate where two decision rules can differ: a tolerance ladder of one angle from `1e-13` to `1e-3` (this package's edge at `1e-9` on the per-amplitude distance, QCEC still answering `equivalent` at `1e-8` and returning *different criteria on identical calls* from `1e-7` up, so those rows carry the criteria and not a verdict), two global-phase pairs (free standalone, `INVALID` under control, where QCEC has no controlled mode), a diagonal `CZ` pair that every Z-basis probability cannot separate, a GHZ pair that is the same *state* from a different *unitary* (our state verdict `VALID`, our unitary verdict `INVALID`, QCEC's `not_equivalent` agreeing with the second), a 12-qubit Clifford pair both decide, and two 12-qubit non-Clifford pairs this package refuses and QCEC decides. Writing it exposed a false `VALID` in the state scope of `adjudicate_clifford_equivalence` — `+YYX` was read as a stabilizer of a GHZ state it is orthogonal to — which is fixed in `apedp_dyn/vqc/stabilizer.py` and now pinned by an exhaustive enumeration. No state-of-the-art claim rests on the large-width rows (`benchmarks/quantum_compilation/vqc_external_comparison.json`). The record also carries a **shared corpus** (`apedp_dyn/vqc/corpus.py`) that is a **sampled distribution**, not a set: three graph families at five widths (12, 20, 32, 48, 64 qubits) with five independent seeded draws per cell, plus one random stabilizer anchor at each of 12, 20 and 32 qubits — 78 instances, 75 matched, every row on one specification hash, every seed the documented arithmetic rule (`SEED_BASE[family] + 10 * width + replicate`) and every label naming its cell. On all 75 matched instances both engines use the **same two-qubit count** (one entangler per edge); the grammar's only saving is depth under its minimum edge colouring, reported as a spread rather than a row — shallower on 75 of 75, median ratio **3.25×** [quartiles 2.75–3.75, range 1.50–5.00], per family regular 3 3.50× [3.00–3.75], regular 4 4.00× [3.20–4.40], irregular 2.65× [2.38–3.20], and per width: the grammar's depth is set by the degree (4 layers cubic, 5 quartic) while Qiskit's tracks the qubit count, so the ratio widens with width on an unchanged schedule. On the three stabilizer states the grammar abstains and Qiskit synthesizes. **Each engine's own router is now run on the same sample**: all 75 matched instances are routed onto the recorded `fake_sherbrooke` coupling map by Qiskit's transpiler (its own layout search, layout kept, seed 1234, optimisation 1) and by this package's lookahead router (identity placement, restore reported separately), every routed circuit decided `VALID` by the same referee. The 3.25× abstract depth gap **does not survive routing**: median ratio 0.97× [quartiles 0.88–1.10], Qiskit shallower on 40 of 75 against this package's 31, with Qiskit ahead at 12–48 qubits and this package ahead at 64 (1.09× [1.01–1.13]); this package's router also swaps more (350 against 206 at the median) because it starts from the identity placement on a sparse device graph. The widths are five fixed tiers and the draws come from two generators, so the spread is over these instances and not over graph space. **Every corpus circuit is now decided twice**, so adjudication is not self-referential: by this package's referee and by a third-party referee (`apedp_dyn/vqc/foreign_referee.py` — Stim's tableau simulator, which evaluates every generator of the same frozen specification) — 150 synthesis and 150 routed decisions, all `VALID`, zero disagreements. The third party decides the prepared state and not the resource counts, is Clifford-only (a non-Clifford circuit is `REFUSED`, not verified), and the IR-to-Stim bridge is this package's |
| Discovery-to-verification loop | the pieces in the three rows above, unjoined | **build** (first complete workload) | `apedp_dyn/vqc/clifford_search.py` plus `scripts/measure_vqc_clifford_discovery.py`: a 100-qubit junked GHZ chain is rewritten to 100 operations, every step certified from 2,000 Pauli images with no matrix, the result promoted, the six mutated candidates refused with the differing image named, and the cheaper star circuit — same prepared state, different unitary — refused at unitary scope while VALID at state scope |

## 2.1 What has landed since the inventory was taken

This section exists so the table above is not reread as a plan: the referee row,
the backends row, the logical-qubit row, the discovery row and the native-execution row
have all moved, and the evidence is in [`VQC_CORE.md`](VQC_CORE.md) §4.1, §4.2, §5,
§5.1, §6, §7 and §8 and in [`VQC_RUST_CORE.md`](VQC_RUST_CORE.md).
Nothing in §2 was edited to retro-fit a claim — the rows say what is now true.

## 3. Phase 1, task by task

| task | status | evidence |
|---|---|---|
| 1. Define the gate vocabulary | done | `gates.py`: `h x y z s sdg t tdg rx ry rz phase cx cz swap ccx cswap`; arity and parameter counts are part of the spec |
| 2. Define circuit and operation structures | done | `Circuit`, `Gate`, `Measure`, `Reset`, `Noise`, conditions, ancillas, metadata, schema version |
| 3. Validation for invalid operations | done | twelve rejection cases in `tests/test_vqc_core.py` (unknown gate, arity, repeated qubit, out-of-range qubit/bit, condition read before written, probability range, noise kind, missing parameter, no qubits, bad ancilla, schema version) |
| 4. Canonical serialisation | done | `to_json()` sorts keys, drops whitespace, fixes float repr; re-serialising is byte-identical |
| 5. Deterministic circuit hash | done | `sha256()` over the canonical bytes; tests require the rebuild to match the original's counts **and** per-shot samples |
| 6. Import/export | done, with named limits | `qasm.py`: export/import `OPENQASM 2.0` including `measure`, `reset`, `if(c==k)`; refuses noise operations and sub-register conditions, because `if(c==k)` compares the whole register and per-bit tests are QASM 3 |

Acceptance circuits (Bell, GHZ, teleportation) each serialise, deserialise and
retain their operations; teleportation's canonical JSON round-trip is exact, while
its **QASM export is refused by design** because two of its gates are conditioned on
one bit of a two-bit register. That is the kind of gap this inventory exists to
record rather than paper over.

## 4. Where the proposed tree actually lives

The proposal sketched `apedp_vqc/ir/`, `apedp_vqc/simulator/statevector/`,
`apedp_vqc/noise/`, and so on. The implementation is flatter, and inside the
existing package:

```
apedp_dyn/vqc/
├── __init__.py     facade + stated boundaries
├── circuit.py      the IR            (proposal: ir/)
├── gates.py        gate table        (proposal: ir/gates.py)
├── qasm.py         interchange       (proposal: ir/serialization.py)
├── backends.py     execution         (proposal: simulator/statevector/)
├── noise.py        channels + exact density matrices (proposal: noise/)
├── logical.py      bit- and phase-flip repetition codes (proposal: logical/)
├── five_qubit.py   the perfect five-qubit code         (proposal: logical/)
├── decoders.py     decoder interface + MWPM            (proposal: decoders/)
├── equivalence.py  unitary + phase referee            (proposal: verification/)
├── discovery.py    discovery admission                (proposal: discovery/)
├── stabilizer.py   a second backend, Clifford only    (proposal: simulator/stabilizer/)
├── receipts.py     adjudication      (proposal: verification/)
├── placement.py    connectivity-aware depth        (proposal: compilation/)
├── external.py     foreign circuits, our referee   (proposal: comparison/)
├── corpus.py       a sampled graph-state corpus     (proposal: comparison/)
│                   neither engine specialises to
├── marginals.py    exact marginals at any width,    (proposal: comparison/)
│                   the referee above 12 qubits
├── memory.py       peak resident growth per engine, (proposal: comparison/)
│                   measured in its own child process
├── foreign_referee.py  Stim as a third-party
│                   referee for both engines' output (proposal: comparison/)
├── equivalence_corpus.py  pairs built so the two
│                   equivalence referees disagree    (proposal: comparison/)
└── qasm.py, docs, tests as above
```

Two deviations, both deliberate:

- **No second top-level package.** Every module in this repository lives under
  `apedp_dyn/` or `apedp/`; a parallel `apedp_vqc/` would split the import graph and
  the documentation index for no gain. If VQC ever needs to be installable
  separately, `apedp_dyn/vqc/` is already a package boundary.
- **Flat instead of three-deep subpackages.** `ir/`, `simulator/statevector/` and
  `verification/` would each hold one or two files at this size. The proposal's
  structure becomes worth its import depth when there are several backends and
  several decoders — i.e. in Phase 6 — not before.

## 5. What is genuinely absent

Named so no phase is mistaken for another:

- **Block 2 (reliability)** has noise on both tiers, three codes (bit-flip and
  phase-flip repetition codes, the perfect five-qubit code), a decoder interface with
  an exact matching implementation, and repetition-code logical rates at distances 3,
  5 and 7 against enumerated predictions. Missing: a surface code (the matching graph
  is one-dimensional), encoded gates, repeated syndrome measurement, any threshold
  claim, and non-perfect syndrome extraction.
- **Block 3 (intelligence)** is started at the referee end — unitary/phase
  equivalence in two tiers, a stabilizer backend, and a discovery admission layer —
  and not at the proposer end: nothing emits this IR yet, there is no rewrite-based
  optimiser and no Pareto comparison. MPS / tensor-network backends and hardware
  adapters do not exist.
- **Phase 7** has distribution-level adjudication, a hard abstention rule, exact
  unitary equivalence inside the dense bound, and exact Clifford equivalence above it.
  A non-Clifford candidate above the dense bound is `UNRESOLVED`, and discovery
  admission refuses to promote it rather than treating the gap as a pass.

## 6. The next task, in order

1. A distance-3 surface-code memory experiment on the decoder interface that exists:
   the matching decoder is graph-driven, so the work is the lattice and its
   stabilizer schedule, not the decoder.
2. Discovery adapters: emit `CircuitCandidate` objects from the search engines and the
   RIME proposers, so a proposed circuit cannot be promoted without a
   `CandidateVerdict` — the admission rule exists, the producers do not.
3. MPS / tensor-network backend: the stabilizer tier now covers Clifford circuits,
   and everything else still needs `2**n` amplitudes.
4. Repeated syndrome measurement with noisy ancillas, which is the assumption every
   logical rate in this repository currently makes.

Each of these gets its own acceptance test before it is described as done.

# APEDP-VQC: the designer, end to end

**Status:** map of an `experimental` subsystem · **Package:** `apedp_dyn/vqc/` ·
**Canonical repo:** `/home/psalmprax/ALL_PROJECTS/science` (see
[`REPOSITORY_CANONICAL_DECISION.md`](REPOSITORY_CANONICAL_DECISION.md)) ·
**Records:** `benchmarks/quantum_compilation/vqc_*.json`

This page answers "where is the quantum circuit designer, what are its parts, and what
happens to a circuit on the way through it". The *claims* it makes are in
[`VQC_SYNTHESIS.md`](VQC_SYNTHESIS.md) and [`VQC_CORE.md`](VQC_CORE.md); this is the
route map, and it deliberately carries no new measurement.

## 1. One sentence

> The designer is handed the **mathematics of a target** (its stabilizer generators, or a
> graph's edge set) and **no circuit at all**, and it returns one or more circuits that it
> has built, decided against that target by two independent methods, ranked, and admitted
> through the same policy the rest of the platform uses.

It is not a compiler for a supplied circuit (that is `clifford_search.py`), and it is not a
general synthesiser: it selects among construction families it knows how to build.

## 2. The route a circuit takes

```
                         TargetSpecification          (name, n qubits, signed Paulis, support)
                                   │
              ┌────────────────────┴────────────────────┐
              │ __post_init__ validates the target:     │  refuses anticommuting
              │ stabilizers, rank, phase, support       │  generators before any search
              └────────────────────┬────────────────────┘
                                   │  specification_state_vector / dense_oracle_check
                                   ▼
    build(family, target) ─── chain │ star │ balanced_tree │ random_tree │ ring │
                                   │ graph_raster │ graph_column_major │
                                   │ graph_tree_first │ graph_layers │ graph_random_order
                                   ▼
                              Circuit (IR)          canonical, versioned, hashable
                                   │  SHA-256 names the circuit in every record
                                   ▼
             adjudicate_specification(circuit, target, tier="stabilizer" | "dense")
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │ stabilizer tier: phase_of(generator) == +1 for all  │  any width
        │ dense tier: amplitudes vs the state projected out   │  ≤ 10 qubits
        │             of the specification, up to one phase   │
        └──────────────────────────┬──────────────────────────┘
                                   │  referee_agreement() runs BOTH and compares
                                   ▼
                        SpecificationReport      VALID / INVALID / UNRESOLVED
                                   │           + which generators failed, by name
                                   ▼
                   synthesize()  ── rank (depth, operations, max_edge_span)
                                   │  + Pareto frontier reported separately
                                   │  + adjudicate_specification_candidate() → admission
                                   ▼
                            SynthesisOutcome      PROMOTED / REFUSED, with the claim
                                   │              checked separately from validity
                                   ▼
                    place(circuit, connectivity)  line │ ring │ grid │ all_to_all
                                   │  greedy or lookahead router, swaps scheduled into
                                   │  the earliest free layer, restore ladder included
                                   ▼
                     Placement  (mapped_depth, swaps, two-qubit gates in 3 bases)
                                   │  the placed circuit is re-decided against the
                                   │  relabelled specification — routing is a relabelling,
                                   │  not an approximation, and must verify as one
                                   ▼
                              record written only if every check passes
```

**The scheduling step** is where graph states differ from GHZ. A graph state's edge set is
the target, so the only free variable is the *order* the commuting `cz` gates are emitted.
`graph_layers` computes a minimum edge colouring (`colouring.py`): `Δ` by a complete search
under a node budget, refuted by a counting bound where the search cannot finish, and
`Δ + 1` always available from Vizing's own construction. The number of colours is the number
of layers.

## 3. The modules

| module | lines | what it is |
|---|---|---|
| `synthesis.py` | 1,675 | The engine: targets, families, both in-package referees, ranking, admission, dense-tier pricing. |
| `external.py` | 1,139 | Foreign circuits against our referee, and ours against Stim / Qiskit / MQT QCEC — one task-matched harness. |
| `stabilizer.py` | 726 | Exact stabilizer tableau: chain lengths a dense simulator cannot reach. |
| `equivalence.py` | 723 | Unitary equivalence up to exactly one global phase. |
| `equivalence_corpus.py` | 622 | Pairs built so two equivalence referees can be compared, including the pairs that must be refused. |
| `colouring.py` | 620 | Minimum edge colouring: `χ′`, the capacity bound, the Vizing construction, class-2 proof. |
| `placement.py` | 600 | Connectivity models, greedy and lookahead routers, the swap-accounting bases. |
| `marginals.py` | 594 | Exact Z-basis marginals of a specification at any width, without a statevector. |
| `logical.py` | 460 | A logical qubit built out of the IR: bit-flip and phase-flip repetition codes. |
| `decoders.py` | 390 | Decoder interface, lookup table, exact minimum-weight decoder. |
| `five_qubit.py` | 382 | The perfect distance-3 code, as a runnable construction. |
| `memory.py` | 377 | Peak memory of a sampling engine, measured in its own process (no shared high-water mark). |
| `circuit.py` | 365 | The IR: one canonical, hashable circuit object. |
| `native.py` | 351 | Rust-first execution with a bit-identical Python fallback. |
| `corpus.py` | 339 | The 78-instance shared corpus of Qiskit-produced graph states, for like-for-like comparison. |
| `clifford_search.py` | 326 | Discovery over Clifford circuits: propose, certify, rank. The rewrites loop. |
| `discovery.py` | 361 | Admission policy shared with the rest of the platform; claim checked separately from validity. |
| `qasm.py` | 275 | OpenQASM 2 in and out, with the parts it cannot carry refused by name. |
| `receipts.py` | 263 | What a claim is allowed to state, and what evidence it needs. |
| `__init__.py` | 252 | The public surface (`Circuit`, `Backend`, `adjudicate*`, `StabilizerBackend`, …). |
| `foreign_referee.py` | 251 | Adjudication of a preparation circuit by an outside simulator. |
| `noise.py` | 239 | Density matrices and channels: the exact ensemble, not a sampled stand-in. |
| `gates.py` | 140 | **The one gate table.** Both in-package referees read `matrix_for` here. |
| `backends.py` | 183 | One backend interface, so a circuit is the same circuit everywhere. |

`gates.py` is the single most important line in this table: it is the shared layer, which is
why agreement between the two tiers is a conformance sample rather than a proof
(§5.2 of [`VQC_SYNTHESIS.md`](VQC_SYNTHESIS.md)).

## 4. The public entry points

| entry point | module | returns |
|---|---|---|
| `ghz_target`, `ring_graph_target`, `cluster_state_target`, `seeded_graph_target`, `irregular_graph_target`, `watts_strogatz_target`, `graph_state_target` | `synthesis.py` | a validated `TargetSpecification` |
| `build(family, target, seed=0)` | `synthesis.py` | one `Construction` (circuit + how it was wired) |
| `adjudicate_specification(circuit, target, tier=…)` | `synthesis.py` | `SpecificationReport`; a named tier that cannot decide raises, it does not fall back |
| `referee_agreement(circuit, target)` | `synthesis.py` | both tiers side by side, or `UNAVAILABLE`/`null` — never an abstention recorded as agreement |
| `synthesize(target, families=…)` | `synthesis.py` | `SynthesisOutcome` with the ranking and the frontier |
| `entangling_depth_lower_bound(target)` | `synthesis.py` | the bound a schedule is judged against |
| `dense_tier_cost(circuit, target)` | `synthesis.py` | the price of one non-Clifford decision, measured in a child process |
| `place(circuit, connectivity, router=…)` | `placement.py` | `Placement` with `mapped_depth`, swaps, gates in three bases |
| `line`, `ring`, `grid`, `all_to_all` | `placement.py` | connectivity models |
| `minimum_edge_colouring`, Vizing construction | `colouring.py` | the schedule, with the proof that carries it |

## 5. The artifacts

| kind | files |
|---|---|
| **records** | `vqc_synthesis.json` (Experiments A–E, placement, agreement, QCEC, depth context), `vqc_external_comparison.json` (shared corpus, device routing, Qiskit depth histogram), `vqc_clifford_discovery.json`, `vqc_stabilizer_ghz.json`, `vqc_stabilizer_scaling.json`, `vqc_native_parity.json`, `vqc_licence_boundary.json`, `vqc_phase0_baseline.json` |
| **producers** | `measure_vqc_synthesis.py`, `compare_vqc_external_tools.py`, `measure_vqc_clifford_discovery.py`, `measure_vqc_stabilizer_ghz.py`, `measure_vqc_stabilizer_scaling.py`, `measure_vqc_rust_parity.py`, `measure_vqc_licence_boundary.py` |
| **docs** | `VQC_SYNTHESIS.md` (the claims), `VQC_CORE.md` (the core and the rewrites loop), `VQC_RUST_CORE.md` (native execution and parity), `VQC_AMPLITUDE_TIER.md` (amplitude-level semantics), `VQC_PHASE0_CAPABILITY_INVENTORY.md` (capability rows), this map |
| **pins** | `tests/test_vqc_{synthesis,synthesis_record,placement,colouring,stabilizer,equivalence,clifford_equivalence,external_comparison,corpus,codes,discovery,native_parity,native_parity_record,phase0_inventory,licence_boundary}.py` |

## 6. Where the designer refuses, and what it refuses

The refusals are the design, not the error handling:

| the designer will not | because |
|---|---|
| accept a target that is not a pure state | `TargetSpecification` checks the generators before any search runs |
| decide a non-Clifford circuit at 12 qubits | the dense tier stops at 10; it answers `UNRESOLVED` and names the bound instead of guessing |
| call a schedule minimal when the search hit its node budget | it reports the `Δ+1` Vizing construction and says the optimum is unproven |
| record a third-party disagreement as a measurement | `referee_agreement` raises and writes no record |
| record an abstention as agreement | `UNAVAILABLE` and `agree: null` are their own values |
| report an unmet claim as a wrong circuit | validity and claim are separate fields; a valid circuit offered as `{"depth": 1}` is `VALID` and `UNMET` |
| route a circuit and call it approximate | the swaps are undone in reverse, so the placed circuit is the same unitary on the same wires — and it is re-decided to prove it |

## 7. What is *not* here

Stated so the map is not mistaken for a capability list:

* **No non-Clifford synthesis search.** The dense tier can *decide* a non-Clifford
  construction within 10 qubits; nothing *proposes* one. An arithmetic/QFT/Toffoli designer
  would need its own engine, cost model and referee story.
* **No invented topology.** The grammar is named wireings; a target outside it needs a new
  family written by hand.
* **No 900-qubit routing.** The 900-qubit lattice is timed and refereed but not routed;
  the greedy router costs tens of seconds per placement at that size.
* **No hardware cost model.** Every topology is abstract: uniform edge cost, no calibration,
  no error rates, no classical control. These are routing bounds, not device prices.

## 8. Cross-references

* [`VQC_SYNTHESIS.md`](VQC_SYNTHESIS.md) — the experiments and their measured results.
* [`VQC_CORE.md`](VQC_CORE.md) §8 — the rewrites loop, and §8.3 for where synthesis sits.
* [`VQC_RUST_CORE.md`](VQC_RUST_CORE.md) — the native core and the parity record.
* [`REPOSITORY_CANONICAL_DECISION.md`](REPOSITORY_CANONICAL_DECISION.md) — which checkout
  this package is canonical in, and why.

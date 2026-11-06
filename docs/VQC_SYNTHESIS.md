# APEDP-VQC: synthesis from a specification, with no reference circuit

**Status:** experimental · **Module:** `apedp_dyn/vqc/synthesis.py` ·
**Record:** `benchmarks/quantum_compilation/vqc_synthesis.json` ·
**Pins:** `tests/test_vqc_synthesis.py` (56), `tests/test_vqc_synthesis_record.py` (27),
`tests/test_vqc_placement.py` (34)

The rewrites loop (`clifford_search.py`) *optimises a supplied circuit*. This document
is the next capability: the platform is given the **mathematics of a target** and no
circuit at all, and it constructs one, decides whether it is right, ranks the
alternatives and admits the best through the same policy the rewrites loop uses.

## 1. What a specification is

A `TargetSpecification` is a name, a qubit count, `n` signed Pauli stabilizers and
(optionally) the exact outcome support:

```python
ghz_target(100)                  # X on every qubit, Z_q Z_{q+1} on every pair
ring_graph_target(100)           # X_q Z_{q-1} Z_{q+1} on every qubit
cluster_state_target(10, 10)     # the grid-graph state: X_v Z_N(v) on every vertex
seeded_graph_target(64, seed=23) # an irregular graph, reproducible from its seed
```

There is no reference circuit in that object, and it is not merely absent by
convention: `TargetSpecification.__post_init__` feeds the stabilizers to the
stabilizer tableau's structural check, so a specification that is not a pure state
(anticommuting generators, a rank-deficient set, an imaginary phase, a support that
does not sum to one) is refused before any search runs. A target is a *state
description*, and it is checked as one.

## 2. Two referees, neither of which needs a reference

| tier | what it compares | width |
|---|---|---|
| **stabilizer** | `phase_of(stabilizer) == +1` for every specified generator, in the tableau's exact integer arithmetic | any size |
| **dense** | the prepared amplitudes against the state *projected out of the specification's stabilizers*, up to one global phase | ≤ `DENSE_SPEC_LIMIT` (10) |

The dense tier is what makes a specification a first-class oracle rather than a
Clifford trick: `specification_state_vector` starts from `|0...0>` and projects onto
`(I + (-1)^p P)/2` for each stabilizer, so the expected state is derived from the
Pauli matrices of the *specification*, never from a circuit. `dense_oracle_check`
compares the two readings of a target at every width both can reach, and the record
asserts they agree (`all_agree: true`).

A third check runs when the target declares a support and the state is small enough to
enumerate exactly: the whole measurement distribution, outcome by outcome, with no
tolerance (these are powers of two).

A graph state is the general case of the ring: `X_v Z_{N(v)}` on every vertex, written
as its edge set. `graph_of_state` reads that edge set back out of the stabilizers, so
the construction families are handed no graph — they read one — and a specification
that is not of that form (a `Y`, two `X`s, an asymmetric edge) is refused by name.
`cluster_state_target(rows, cols)` is the graph state of a square lattice with qubits
on its **vertices** (not the surface-code cluster, whose qubits sit on lattice edges),
and `irregular_graph_target` / `seeded_graph_target` supply a graph with no regular
structure.

One implementation finding belongs here because Experiment E produced it: the dense
tier compares amplitudes built by `StatevectorSimulator` (bit `i` = qubit `i`) against
a state built as a Kronecker product (qubit 0 most significant), and for three
experiments every specification was invariant under reversing its wires, so the two
orders were the same state. An irregular graph state is not, and the naive comparison
reported a distance of `0.0605` on a state that was exactly right. `_specification_order`
is the single re-index that fixes it (`2.8e-17` on the same state), and
`test_the_two_basis_orders_are_pinned_on_an_asymmetric_state` pins both numbers so the
bug cannot return silently.

## 3. Experiment A — construct at 100 qubits, no reference circuit

Five construction families are built from the GHZ specification at 100 qubits. Each
is decided against the specification alone; all five verify.

| family | operations | depth | max edge span | nearest-neighbour | verdict |
|---|---|---|---|---|---|
| chain | 100 | 100 | 1 | yes | VALID |
| reverse_chain | 100 | 100 | 1 | yes | VALID |
| star | 100 | 100 | 99 | no | VALID |
| **balanced_tree** | 100 | **12** | 50 | no | VALID |
| random_tree (seeded) | 100 | 14 | 90 | no | VALID |

* **Ranking** is `(depth, operations, max_edge_span)`; the promoted candidate is
  `spec-ghz-100q-balanced_tree`, with admission
  `VALID` / `MET` / `SUFFICIENT` → **PROMOTED**.
* **The Pareto frontier is reported instead of a single winner**, because depth and
  connectivity genuinely trade off: `chain` is 100 deep and needs only neighbour
  wires; `balanced_tree` is 12 deep and reaches across half the register. `star` is on
  neither axis's frontier — and note that it is *not* shallow, because all of its
  `cx` gates share qubit 0 as control and therefore serialise. That is a measurement,
  not an assumption: at 100 qubits a star is depth 100, not depth 2.
* **Four wrong constructions are refused**, each with the failing generator named:
  drop the Hadamard (the product state fails `X...X`), reverse the middle `cx`,
  add an extra Hadamard on the last qubit, and replace a `cx` with a `cz`. All four
  are `INVALID` with `failed_count >= 1` and the Pauli string recorded.
* **A second target is run** so the result cannot be read as "GHZ works": a 100-qubit
  **ring graph state**, built by the `h`+`cz` ring (VALID, promoted) while the linear
  cluster — the same construction with the closing edge missing — is refused with
  `+XZIIZ...` and `+ZIIZX...` named as absent from the prepared state's group.

## 4. Experiment B — the same synthesis at 10 to 1,000 qubits

| qubits | family | ops | depth | span | verification (s) | verdict |
|---|---|---|---|---|---|---|
| 10 | chain | 10 | 10 | 1 | 0.048 | VALID |
| 10 | balanced_tree | 10 | 5 | 5 | 0.063 | VALID |
| 50 | chain | 50 | 50 | 1 | 0.004 | VALID |
| 50 | balanced_tree | 50 | 10 | 25 | 0.004 | VALID |
| 100 | chain / balanced | 100 / 100 | 100 / 12 | 1 / 50 | 0.018 / 0.025 | VALID |
| 250 | chain / balanced | 250 / 250 | 250 / 14 | 1 / 125 | 0.115 / 0.122 | VALID |
| 500 | chain / balanced | 500 / 500 | 500 / 16 | 1 / 250 | 0.479 / 0.775 | VALID |
| 1,000 | chain / balanced | 1000 / 1000 | 1000 / 18 | 1 / 500 | 1.332 / 1.547 | VALID |
| 1,000 | star / random | 1000 / 1000 | 1000 / 21 | 999 / 948 | 1.604 / 1.716 | VALID |

Every one of the 24 rows is verified against the specification, with the number of
stabilizers checked equal to the qubit count. The 1,000-qubit rows cost about 1.3–1.7 s
each, and that is dominated by building the 1,000-gate circuit and walking 1,000
`phase_of` queries, not by any exponential term: the stabilizer representation is
linear in `n` gates and the referee is `O(n²)` bit operations.

## 5. Experiment C — not built, and its boundary measured

The review's third experiment is expansion beyond Clifford circuits. That search does
not exist. What does exist is the **verification** boundary, and the record measures it
rather than implying more:

| target | circuit | qubits | tier | verdict |
|---|---|---|---|---|
| GHZ | `ry(pi/2)` + `cx` ladder (not Clifford) | 3 | dense | VALID (distance 3.9e-17) |
| GHZ | same | 8 | dense | VALID (distance 6.9e-18) |
| GHZ | same | 12 | none | **UNRESOLVED** — "not Clifford and 12 qubits is above the dense specification bound of 10" |

So a specification can already be checked against a non-Clifford construction within
the dense bound, and the tier refuses to guess above it. The missing half is a search
that *proposes* non-Clifford moves: no rewrite rule in `clifford_search.py` can cross
that line, and the Toffoli/arithmetic/QFT workloads the review lists need their own
engine, their own cost model and their own referee story. That is the next experiment,
not a claim of this one.

## 5.1 Experiment D — the same construction on a real connectivity graph

Logical depth is what §3 reports, and it assumes any qubit may interact with any
other. `apedp_dyn/vqc/placement.py` places the same circuit on a line, a ring, a
rectangular lattice and an all-to-all device, and reports what routing costs. Both
routers run on every row: the **greedy** one this record started with, and a
**lookahead** one added since.

| construction | topology | logical | greedy mapped | lookahead mapped | greedy swaps | lookahead swaps |
|---|---|---|---|---|---|---|
| chain | line | 100 | 100 | 100 | 0 | 0 |
| chain | 10x10 lattice | 100 | 191 | **117** | 246 | 225 |
| chain | all-to-all | 100 | 100 | 100 | 0 | 0 |
| balanced tree | 10x10 lattice | 12 | 220 | **84** | 547 | 511 |
| balanced tree | ring | 12 | 253 | **212** | 2,151 | 1,832 |
| balanced tree | line | 12 | 311 | **278** | 2,415 | 2,165 |
| balanced tree | all-to-all | 12 | 12 | 12 | 0 | 0 |

All 16 rows are decided against the same specification and all are `VALID`, because
routing here is a *relabelling* rather than an approximation: the logical-to-node map is
tracked exactly, and the emitted swaps are undone in reverse order at the end, so the
swap product telescopes to the identity and the mapped circuit is the original unitary
on the same wires. The restore ladder is part of the reported cost and is reported
separately (`depth_before_restore`, `restore_swap_gates`), since a compiler that tracks
its final layout can absorb it and a verifiable circuit cannot. **`mapped_depth` above is
the total depth including that ladder** — the depth a device executes — and
`restore_depth_cost` is how many layers the ladder adds: 94 for the greedy lattice tree,
35 for the lookahead one, and 0 on every row that needed no routing. Neither number is
the other, and the record carries both.

**Two-qubit counts are reported in three bases**, because they answer different
questions and the first version of this table only showed the first:

| basis | what a SWAP costs | lattice tree, greedy | lattice tree, lookahead |
|---|---|---|---|
| abstract two-qubit operations | one instruction | 1,193 | 1,121 |
| native entanglers (declared basis) | one, this record | 1,193 | 1,121 |
| CNOT-equivalent | three CNOTs | **3,381** | **3,165** |

The arithmetic is quoted in the record rather than left to the reader: `1,193 = 99 +
2(547)` and `3,381 = 99 + 6(547)`, where 547 is the forward swap count, and the totals
double it because the restore ladder repeats every swap. On a CNOT-only backend the
lattice tree costs **3,381 entanglers**, not 1,193 — a factor of 2.8 on the headline
number, and 3,756 fewer CNOT-equivalents for the lookahead across the eight rows. The
same three bases are carried for every row, and `place(..., swap_entanglers=3)` declares
a CNOT-only basis explicitly so the native column and the CX-equivalent column agree.

**The finding survives the better router, and gets sharper.** The balanced tree's
advantage is conditional on connectivity: on all-to-all it is the depth 12 it claims,
and on every other topology the routed tree is deeper than the nearest-neighbour chain
it was meant to beat — 278 against 100 on a line, 212 against 100 on a ring, 84 against
117 on the lattice. The lookahead improves every row that needed routing, and leaves
every free row untouched:

- lattice chain: 191 → **117** depth (−39%), 246 → 225 swaps, 591 → 549 two-qubit gates;
- lattice tree: 220 → **84** depth (−62%), 547 → 511 swaps, 1,193 → 1,121 two-qubit gates;
- ring tree: 253 → **212** (−16%); line tree: 311 → **278** (−11%).

That is what a scheduler buys. The greedy router moves the second operand along a
shortest path with no view of anything else. The lookahead router considers both
operands and every shortest-path step, schedules each candidate swap into the earliest
layer its two nodes are free (so routing fills idle wires instead of lengthening the
critical path), and breaks ties by the total wire distance of the next eight two-qubit
gates that involve a qubit the swap moves. The window and the tie-break order are
choices, not measurements, and the record carries both routers row by row rather than a
single "improved" number.

It is a heuristic, not a theorem. At small widths it can lose (a 9-qubit balanced tree
on a line routes two layers deeper than greedy), neither router reorders gates — the
program order is routed as written — and the topologies are abstract (every node holds a
qubit, every edge costs the same, no calibration, no error rate, no classical control).
These are routing bounds, not a floor on what hardware must pay, and a better router
would improve them again.

The record also carries the two results a reader should not have to take on trust. **The
counterexample is pinned**: a 9-qubit balanced tree on a line, where the greedy router
uses 17 layers and the lookahead **19** — two layers worse, because the window sees too
little to pay for the swap it chooses. **The placements are sorted into a Pareto
frontier** over (mapped depth, CNOT-equivalent two-qubit gates): it collapses to **one**
point, the balanced tree on all-to-all, because in this corpus the two costs are ordered
identically — every routed alternative is deeper *and* dearer. A frontier of one point is
reported as that result rather than as a trade-off, and it is why the table above can
quote a single "best" placement without hiding a cost the way a score would.

**SABRE, swept over five seeds.** The same six routed cases are handed to Qiskit's own
router, on the same coupling map and optimisation level, under seeds 0–4. All 30 routed
circuits are decided `VALID` against the specification relabelled by Qiskit's final
layout, and the seed-to-seed interval is reported per case. It is not zero: the balanced
tree on a ring spans **120–144 layers** (spread 24), on a line 166–176, on the lattice
68–73, and the chain on a lattice 188–192; a chain on a line or ring needs no routing and
is identical under all five seeds. A single-seed SABRE number therefore understates the
depth by up to **24 layers** on this corpus — which is why the sweep, not one run, is the
measurement. This package's lookahead router is quoted beside it at its pre-restore depths
(124, 158, 49 and 100 on those cases): the comparison is against that column because
Qiskit keeps its final layout while this record's router restores the permutation.

## 5.2 Two referees, one circuit

§2 describes the stabilizer tier and the dense tier. Until this slice they were never
run against the same circuit — the referee picked one, so "two referees" was a claim
about code layout. `referee_agreement()` runs **both** and reports the verdicts side by
side; `adjudicate_specification(..., tier=...)` is the override it uses, and a named
tier that cannot decide a circuit raises rather than falling back.

**What the two referees share, and what they do not.** They share the circuit object
and its parser (gate names, targets, parameters, conditions), and both derive gate
semantics from one table, `gates.matrix_for` — the stabilizer tier builds its Clifford
conjugation tables from those matrices and the dense tier applies them directly. A
wrong `h` matrix or a qubit-indexing convention error would therefore fool both. What
they do not share is the *representation*: group membership in exact integer arithmetic
over `n` Pauli strings versus `2**n` amplitudes and a phase-free distance. The closed
forms in `tests/test_vqc_stabilizer.py` and `tests/test_vqc_synthesis.py` are the
independent check on the shared layer, and the corpus below is a conformance sample,
not a proof of agreement.

**An independence ledger, so that "three referees" is auditable.** The table names what
each tier shares with the others, because a shared component is a shared failure mode:

| tier | what it shares | what is genuinely independent | what it is blind to |
|---|---|---|---|
| stabilizer | parser, `gates.matrix_for`, the circuit object | exact integer group membership over `n` Pauli strings | anything non-Clifford (it refuses, it does not guess) |
| dense | parser, `gates.matrix_for`, the circuit object | `2**n` amplitudes and a phase-free distance | widths above `DENSE_SPEC_LIMIT`; it refuses there |
| Stim | **this package's IR→Stim bridge** | an external tableau simulator and its own observable evaluation | Clifford only; it decides the prepared state, not the schedule |
| MQT QCEC | a QASM export written here | an external unitary comparison | sees the whole operator, so it cannot see a schedule that is right up to a wire permutation any better than the stabilizer tier can |

So at 400 and 900 qubits there is exactly **one** in-package decision on the schedule plus one
outsider reached through this package's own bridge, and the disagreements that would matter
most — a translation bug, a wrong gate matrix — are the ones no row here can catch. The
closed forms in `tests/test_vqc_stabilizer.py` and `tests/test_vqc_synthesis.py` are the only
check on the shared layer, and they are the reason it is stated as a limit rather than a gap.

23 circuits (GHZ chain and balanced tree at every width 2..10, ring graph states at
3..6, and one `cx`→`cz` mutation): **22 accepted by both, 1 refused by both, no
disagreement.** That is a sample, not a theorem: randomised stabilizer circuits, longer
sequences, phase-sensitive mutations and adversarial specifications are the next corpus
and are not claimed here. The accepts are decided exactly — the dense tier's phase-free distance is
≤ 1e-12, which is rounding rather than a tolerance being spent. On the refused row the
tiers fail for different reasons and still agree on the verdict: the stabilizer tier
names the `ZZ` generator whose sign flipped and the declared support that no longer
matches, the dense tier reports an amplitude distance of 0.125.

**Two wide rows, and the widths where a second tier does not exist.** The two-tier
corpus stops at ten qubits because that is where the dense tier stops, so agreement at
400 or 900 qubits cannot be an agreement between those two referees. The corpus's wide
rows are the Experiment E minimum schedules for `cluster_20x20` (400 qubits, 760 edges)
and `cluster_30x30` (900 qubits, 1,740 edges) — the same circuits Experiment E times,
compared by SHA-256 — decided by the stabilizer tier (`VALID` on all 400 and all 900
generators) and again by **Stim**, which simulates each circuit as a tableau and
evaluates every generator of the same frozen specification as a Pauli-observable
expectation: `VALID`, 400/400 and 900/900, no disagreement.

A third party that is only ever seen accepting has not been shown to refuse anything. The
same three wrong graph constructions Experiment E refuses — a dropped edge, an added
edge, and one `cz` replaced by `cx` — are therefore decided again at the 400-qubit width,
and both referees answer `INVALID`, each naming the generators it failed.

Beside the wide rows the corpus records the rows that cannot exist: asking the dense tier
for either target raises *"the dense tier is bounded at 10 qubits and this construction
acts on 400"* (and the same at 900), and both refusals are stored
(`referee_agreement.wide.dense_tier`) rather than left as absences — so the corpus
documents where two independent in-package decisions are impossible, and not only where
they succeed. A *disagreement* is not stored at all: if a comparative third-party
verdict differs from the stabilizer tier's, the script raises and writes no record,
because two independent decisions contradicting each other is not a measurement to
publish. When Stim is absent the rows read `UNAVAILABLE` and `agree` is `null`: an
abstention is never recorded as an agreement, and it is not scored as a disagreement
either. What the Stim rows do not buy is the list `foreign_referee.py` states: it is
Clifford-only, it decides the prepared state rather than the schedule, and the IR→Stim
bridge is this package's, so a shared translation bug would fool both referees.

## 5.3 Experiment E — a 2-D cluster and an irregular graph, as specifications

Experiment A's targets are permutation-symmetric enough that its constructions are a
wiring choice. A graph state is not: the specification names every edge, so the edge
set, the operation count, the two-qubit count and the connectivity are all the
target's, and **the only free variable left is the schedule** — the order the
commuting `cz` gates are emitted in. Five schedules are built for each target:
ascending wire order (`graph_raster`), transposed order (`graph_column_major`), a
breadth-first spanning tree then the closing edges (`graph_tree_first`), a **minimum
edge colouring** (`graph_layers`), and a seeded random order as a control
(`graph_random_order`).

| target | qubits | edges | max degree (Δ) | minimum layers (χ′) | tree-first (layers) | random (layers) | raster / column (layers) |
|---|---|---|---|---|---|---|---|
| cluster 4x4 | 16 | 24 | 4 | **4** | 8 | 7 | 12 |
| cluster 10x10 | 100 | 180 | 4 | **4** | 20 | 10 | 36 |
| cluster 20x20 | 400 | 760 | 4 | **4** | 40 | 13 | 76 |
| cluster 30x30 | 900 | 1,740 | 4 | **4** | 60 | 13 | 116 |
| irregular, 25 q, seed 7 | 25 | 68 | 12 | **12** | 23 | 20 | 23 |
| irregular, 64 q, seed 23 | 64 | 271 | 16 | **16** | 43 | 31 | 44 |
| complete K9 | 9 | 36 | 8 | **9** | 15 | 15 | 15 |

All 35 valid rows are `VALID` at the stabilizer tier, with `n` stabilizers checked.

The K9 row is the class-2 case **inside** Experiment E. Its 36 edges cannot fit into
eight layers of four, so the measured minimum is nine — one above the degree floor, and
proved so by the capacity count at zero search nodes — and the peel's 15 layers are shown
against a built, verified 9. That is the six-layer saving the colouring block proves,
now appearing in the same timing table as the clusters rather than in a separate one.
The three largest lattices — 100, 400 and 900 qubits at the same degree-4 wiring — are
the timing ladder: each family is timed while it is built and while the referee decides
it (`synthesis_seconds`, `verification_seconds` on the row), so the schedule's cost can
be read as a function of width rather than as a single point.

The depth table above is the measured schedule; the seconds are per row in the record
(`synthesis_seconds` to build the circuit from the specification, `verification_seconds`
for the stabilizer referee to decide it):| lattice | qubits | edges | operations | build (s) | decide (s) | decide / build |
|---|---|---|---|---|---|---|
| 10x10 | 100 | 180 | 280 | 0.001-0.002 | 0.015-0.017 | 7-12× |
| 20x20 | 400 | 760 | 1,160 | 0.020-0.032 | 0.233-0.288 | 7-14× |
| 30x30 | 900 | 1,740 | 2,640 | 0.101-0.160 | 1.269-1.369 | 8-13× |

Each pair is the range of the five row medians at that size. Every clock is the
**median of three repeats** of that phase on the same rebuilt circuit, with all three
samples recorded on the row: at 400 qubits the three deciding samples span up to 43% of
their median (median span 11%), which is the size of the differences these columns are
read for, and the median is what the table quotes rather than whichever run came first.

**Deciding costs about ten times building at every size** — the ratio is 7-14× across
all fifteen rows and does not widen with the lattice — because the referee runs one
stabilizer check per qubit while the builder emits one `h` per qubit and one `cz` per
edge, and both are linear in the operation count. What grows is the whole target:
**0.14 s at 100 qubits, 2.36 s at 400 and 11.34 s at 900**, each covering five schedules
and three refusals. The deciding clocks are flat at a given size — at 900 qubits the
**5-layer** schedule costs **1.37 s** to decide and the **117-layer** walk **1.31 s** —
so **depth is not bought with time**, and no timing advantage is claimed for the minimum
at any of the three: every family emits the same operations (2,640 at 900 qubits) and
every decision is the same 900 stabilizer checks. Building does vary, by up to 2×
between orders at that size (the tree-first walk runs a breadth-first pass before
emitting), but not with depth.

These are wall clocks on this host, each measured on a circuit
rebuilt in a warm process (the family has already been built and decided once for the
ranking), not a benchmark against another implementation; individual samples still move
by tens of percent between runs under load — which is why the median is quoted and the
samples are kept beside it — and the script refuses to write the record if a rebuilt
circuit does not hash to the row it is attached to. The record carries the committed
values; this page quotes the range rather than a number that would be stale as soon as
the record was regenerated.

### 5.3.1 What routing does to that advantage

The depths above are logical: they assume any qubit may interact with any other. A
schedule's advantage is not settled until the circuit is routed, because the ordering
that packs gates into the fewest layers is not obviously the one that minimises SWAPs.
Every schedule was therefore placed on a **line**, a **ring** and the cluster's **own
lattice**, by both routers, at the three smaller lattices (16, 100 and 400 qubits) — 90
placements, 72 comparisons against the minimum schedule, and every placed circuit
decided `VALID` by the same
referee (it is the same unitary on the same wires, because the routing swaps are undone
in reverse). The 900-qubit lattice is timed but **not routed**: the greedy router needs
tens of seconds per placement at that size, so the record names the target it did not
route rather than shrinking its row count.

| qubits | topology | minimum schedule, mapped (greedy) | best other schedule | logical gap | mapped gap | survives |
|---|---|---|---|---|---|---|
| 100 | line | 109 | column-major 360 | 32 | 251 | 7.8× |
| 100 | ring | 109 | raster 444 | 32 | 335 | 10.5× |
| 100 | own lattice | 5 | random 11 | 6 | 6 | **1.00×** |
| 400 | line | 229 | column-major 1,520 | 72 | 1,291 | 17.9× |
| 400 | ring | 229 | raster 1,894 | 72 | 1,665 | 23.1× |
| 400 | own lattice | 5 | random 14 | 9 | 9 | **1.00×** |

**The advantage survives everywhere, and on the lattice it survives exactly.** The grid
row is the control: the target's edges *are* lattice edges, so the identity placement
needs no SWAP at all, every family keeps its logical depth, and the gap is the logical
gap (0 swaps, 1.00× for all 24 lattice comparisons).

**On a line or a ring it grows.** Routing costs the minimum schedule too — 12,316 swaps
at 400 qubits on a line, longest hop 40 — but the localized orders pay far more:
raster, column-major and tree-first need **144,400** swaps with a longest hop of **362**
(they emit gates across the whole register in an order no line can serve locally), and
the random order 30,362. So the mapped gap is larger than the logical one: across all 72
comparisons the surviving fraction is **1.00×-219×, median 4.67×**, and **no comparison
inverts** — the minimum schedule is still the shallowest circuit after routing, by a
wider margin than before. The strongest alternative is column-major at 100 and 400
qubits on a line, and raster on a ring, which is the nearest-neighbour order of a
different axis: it is the best routeable rival and still 7.8× to 23.1× further behind
than it was logically.
`graph_layers` is no longer a greedy peel: it computes the edge chromatic number
`χ′` (`apedp_dyn/vqc/colouring.py`) and is optimal by Vizing's bound
(`Δ ≤ χ′ ≤ Δ + 1`). On these six targets the old peel happened to attain `Δ`, so
**the replacement saves 0 layers on every Experiment E target** — the record says so
rather than implying an improvement. Its value is that the optimum is now *proved* and
that class-2 graphs are handled, which the peel could not do.

The `Δ + 1` half of Vizing's theorem is *constructed*, not searched. `vizing_colouring`
implements the fan argument (in its Misra–Gries form): one fan, one `c`/`d` path
inversion and one rotation colour a single edge in a single pass with `Δ + 1` colours,
so a `(Δ + 1)`-schedule is always available and the scheduler never has to fall back on
the peel. The construction is verified before it is returned (proper, complete, no more
than `Δ + 1` colours) and is exercised exhaustively on every simple graph of up to five
vertices in `tests/test_vqc_colouring.py`.

**Class 2 is proved by whichever argument finishes first.** `Δ` is refuted by a
counting bound when the edges cannot fit into `Δ` layers — a layer is a matching, so it
holds at most `⌊V/2⌋` edges — and otherwise by exhausting the `Δ`-colouring search. The
bound is what makes the odd complete graphs measurable: their search cannot finish
(`K9` needs more than five million nodes, `K11` more), so before it existed the whole
family could only be reported as undecided. It now refutes `Δ` in one multiplication,
and the constructed `Δ + 1` schedule is minimal by Vizing.

| class-2 graph | qubits | Δ | χ′ | proof | greedy peel | minimum | saved |
|---|---|---|---|---|---|---|---|
| odd cycle C5 | 5 | 2 | 3 | layer capacity (5 > 2·2) | 3 | **3** | 0 |
| Petersen | 10 | 3 | 4 | exhausted search (31 nodes) | 4 | **4** | 0 |
| complete K5 | 5 | 4 | 5 | layer capacity (10 > 4·2) | 7 | **5** | **2** |
| complete K9 | 9 | 8 | 9 | layer capacity (36 > 8·4) | 15 | **9** | **6** |
| complete K11 | 11 | 10 | 11 | layer capacity (55 > 10·5) | 15 | **11** | **4** |
| complete K13 | 13 | 12 | 13 | layer capacity (78 > 12·6) | 15 | **13** | **2** |

The Petersen graph is the case the bound cannot decide — 15 edges fit exactly into
three layers of five — so its refutation is still the search's own exhaustion, and the
record keeps both kinds of proof side by side rather than reducing them to one. The odd
complete graphs are the family where that would have been impossible: six circuits,
all `VALID` under the same referee, and **14 layers saved across the class-2 rows**
(2 on `K5`, 6 on `K9`, 4 on `K11`, 2 on `K13`). The 100-qubit lattice is still **more than seven times shallower**
than the row-major walk (5 against 37) at exactly the same 280 operations and 180
two-qubit gates, and at 400 qubits it is **fifteen times shallower** than the row-major
walk (5 against 77) at the same 1,160 operations and 760 two-qubit gates.

**A third referee, on the unitary question.** Every referee above decides a *prepared
state*. MQT QCEC answers a different and stricter question — do the two circuits compute
the same unitary up to one global phase — so a schedule that permuted its wires would
prepare the right state and fail this one. The comparison is therefore run where the two
questions coincide by construction: every schedule of one edge set emits the same `h` and
the same commuting `cz` set on the same wires, making them one operator, while the wrong
constructions are not. The reference is the specification's own preparation — `h` on
every wire, one `cz` per edge in sorted order — built in the script rather than by a
construction family, so nothing is compared with itself. It decided all 11 pairs: the
five 16-qubit schedules are `equivalent` (one unitary, five schedules), the 25-qubit
irregular graph and the **900-qubit lattice** are `equivalent`, and the three wrong
constructions are `not_equivalent` — agreeing with this package's referee on every row.
No pair in this corpus was beyond QCEC's reach, so the record holds **no abstention**:
the abstention path is implemented and pinned by tests, and the corpus does not exercise
it. That is stated rather than quietly counted as a win.

**The lookahead window, swept.** Window 8 was a choice, so all five are measured: 40 rows
over windows 2, 4, 8, 16 and 32 on every Experiment D placement, each with depth, swaps,
two-qubit gates and wall clock. Window 8 reproduces the recorded lookahead rows exactly,
by circuit SHA-256; all-to-all is window-independent (the control — if the window moved
it, the measurement would be contaminated); and depth is **not** monotone in the window.
The best-depth total is 969 against 1,003 at window 8: the balanced tree on the lattice
gains at 32 (84 → 74) while the chain prefers 2. No ordering between windows is claimed.

**Two counting bounds on every target, and the construction's overshoot rate.** Each
Experiment E target now carries the layer-capacity bound `Δ·⌊V/2⌋` and the matching bound
`⌈|E|/ν⌉` with `ν` a maximum matching — never weaker than the capacity bound when a
graph has no perfect matching. On the six class-1 targets neither refutes `Δ` (4 against
760/200 on the 400-qubit lattice), so the schedule's minimality there rests on the search
or the peel rather than on a bound that happened not to fire; on K9 the capacity bound is
exactly what refutes it (36 > 8·4), which is why both bounds are reported even where they
are silent, and the target that fires is named. Separately, Vizing's `Δ + 1` construction is called directly on 41 random graphs
whose `Δ`-colouring this module proves: it spends the extra layer on **27 of them
(66%)**, never more than one. So on a class-1 graph the construction alone would
overshoot two times in three, which is exactly why the `Δ` search runs first.

Three findings the record states rather than implying:

* **The Pareto frontier collapses to one family here**, and that is the correct answer.
  In Experiment A depth traded against connectivity, so the frontier kept the chain
  beside the tree. A graph state fixes the connectivity, so the five schedules tie on
  operations and edge span and differ only in depth: `graph_layers` dominates, and the
  promoted candidate is `spec-cluster_2d-400q-graph_layers` on the largest target (and
  the minimum schedule for each of the other four).
* **Three wrong graph constructions are refused edge by edge.** Drop a lattice edge,
  add an edge the specification does not name, or replace one `cz` with `cx`, and
  exactly the two stabilizers of that edge's endpoints are reported absent from the
  prepared state's group (`failed_count: 2`, the Pauli strings named) — a refusal that
  says *which* edge, not just that something is wrong.
* **The target really is the graph.** A 12-qubit GHZ chain decided against a 3x4
  cluster is `INVALID`, and the cluster decided against the GHZ is `INVALID`, at a
  shared width — so "every family verifies" is a statement about the constructions,
  not about the referee's tolerance.

Both referees are run on the graph targets at the widths both can reach: two clusters
(8 and 10 qubits), two seeded irregular graphs (9 and 10), a three-qubit path and one
dropped-edge refusal — **five accepted by both, one refused by both**, the accepted
distances at 1e-17 (rounding) and the refusal at `0.125`. The irregular cases are the
point of that corpus: a cluster is invariant under reversing its wires, so it can hide
a basis-order mistake, and an asymmetric graph cannot.

### 5.3.2 Where those depths sit in the Qiskit corpus's depth histogram

The depths in the table above are this package's. The same kind of specification handed
to Qiskit's `synth_circuit_from_stabilizers` produces the 78-instance corpus in
[`apedp_dyn/vqc/corpus.py`](../apedp_dyn/vqc/corpus.py) — random 3- and 4-regular graph
states and seeded irregular graphs at 12, 20, 32, 48 and 64 qubits, five independent
draws per cell, plus one random stabilizer state at each of 12, 20 and 32 — and that
distribution is what each Experiment E depth is reported against. The block is written
by `scripts/compare_vqc_external_tools.py` (`qiskit_depth_context` in
[`vqc_external_comparison.json`](../benchmarks/quantum_compilation/vqc_external_comparison.json)),
not typed here.

| Qiskit depth | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 90-99 | 120-129 |
|---|---|---|---|---|---|---|---|
| corpus rows (n = 78) | 43 | 20 | 4 | 6 | 3 | 1 | 1 |

Median **18**, quartiles **14-24.75**, range **10-126**. No row falls between 60 and 89
or between 100 and 119. The three random stabilizer anchors (34, 94, 126) are the whole
tail; every graph-state row is between 10 and 55, and the shallowest circuit Qiskit
produces anywhere in the corpus is **10**, at 12 qubits.

| target | qubits | edges | this package (χ′ + 1) | corpus cell | n | cell min / median / max | at or below |
|---|---|---|---|---|---|---|---|
| cluster 4x4 | 16 | 24 | **5** | 12q* | 16 | 10 / 14.5 / 34 | 0 (0%) |
| cluster 10x10 | 100 | 180 | **5** | 64q* | 15 | 11 / 20 / 55 | 0 (0%) |
| cluster 20x20 | 400 | 760 | **5** | 64q* | 15 | 11 / 20 / 55 | 0 (0%) |
| cluster 30x30 | 900 | 1,740 | **5** | 64q* | 15 | 11 / 20 / 55 | 0 (0%) |
| irregular, 25 q, seed 7 | 25 | 68 | **13** | 20q* | 16 | 11 / 18 / 94 | 5 (31%) |
| irregular, 64 q, seed 23 | 64 | 271 | **17** | 64q | 15 | 11 / 20 / 55 | 5 (33%) |

`*` marks a cell the corpus does not run at that width: the corpus's widths are 12, 20,
32, 48 and 64, so **only the 64-qubit irregular target is compared at its own width**
and the other five are placed against the nearest cell, with the ratio recorded
(12/16 = 0.75, 20/25 = 0.80, 64/100 = 0.64, 64/400 = 0.16, 64/900 = 0.07). A depth at
900 qubits and a depth at 64 are not the same quantity, and the table says which row is
which rather than averaging the gap away.

Read that way, three things follow, and none of them is a depth advantage:

* **The five-layer cluster depths are below every corpus row**, not merely below the
  median: 0 of 78 instances have a Qiskit circuit of five layers or fewer, and the
  corpus's own minimum is 10. But a cluster is bipartite, so its `χ′` is its maximum
  degree (4) and its depth is `1 + χ′ = 5` **by the target's edge set**, not by a
  search: the number is a property of the specification that any schedule attaining
  the floor would report.
* **The irregular depths are inside the corpus, not under it.** At 25 qubits the
  13-layer schedule is at or below 5 of the 16 rows in the 20-qubit cell (31%), and the
  64-qubit target's 17 layers are at or below 5 of the 15 rows in its own cell (33%),
  whose median is 20. Both of those are class-1 graphs whose depth is `Δ + 1`
  (12 + 1 and 16 + 1), so again the number tracks a property of the graph.
* **The corpus engine was never asked for depth.** `synth_circuit_from_stabilizers` is
  a general elimination that builds a circuit for any stabilizer state; its output is
  one engine's construction, not a floor, and a corpus cell's minimum is not the
  shortest circuit that exists for those instances. The instances differ too — no
  corpus row is a lattice. So this block is the width-by-width context for the depths
  in the table above, and the like-for-like engine-against-engine measurement on the
  *same* instances (with the routing that mostly erases it) is the shared-corpus and
  device-routing tasks of `vqc_external_comparison.json`, not a second claim here.

### 5.3.3 Theorem, measurement, or engineering

Three kinds of claim appear in this document and they carry different weight. Separating
them is what stops a restatement of a 1916 result from reading as a discovery:

| claim | status | whose result |
|---|---|---|
| a bipartite graph's edge chromatic number equals its maximum degree | **theorem** (König, 1916) | the cluster's 5-layer depth *is* `1 + Δ`; any schedule attaining the floor reports it |
| every graph has a `(Δ+1)`-edge-colouring | **theorem** (Vizing, 1964) | the constructed `Δ+1` schedule is optimal-by-bound, not optimal-by-search |
| K9 needs 9 layers, not 8 | **theorem applied + measured** | refuted by `36 > 8·4` at zero search nodes, then built and verified at 9 |
| the Petersen graph needs 4 | **measurement** | the counting bound is silent (15 edges fit three layers of five), so the search exhausts in 31 nodes |
| a balanced tree is 12 deep where a chain is 100 | **textbook construction** | a property of the chosen wiring, not an optimisation |
| the minimum schedule stays shallowest after routing, by 7.8×–23.1× | **measurement** | 72 comparisons, no inversion, on three topologies |
| deciding a circuit costs ~10× building it | **measurement (this host)** | 7–14× across 15 rows; wall clocks, medians of three |
| `Δ+1` overshoots on 66% of class-1 graphs | **measurement** | 41 random graphs; never by more than one layer |

Nothing in the theorem rows is a contribution of this package. What the package contributes is
that the optimum is **proved** rather than assumed, that the proofs are wired to the same
referee that decides the circuits, and that a counterexample is pinned where one exists
(the 9-qubit tree on a line, 19 layers against greedy's 17).

## 5.4 Where the dense tier stops: the price of one non-Clifford decision

The dense tier is the only referee that can decide a non-Clifford construction against
a specification, and its bound — `DENSE_SPEC_LIMIT`, ten qubits — was a policy with no
price attached. It now has one. The tier materialises every stabilizer as a dense
`2**n x 2**n` Pauli operator (`_pauli_matrix` is a chain of `np.kron`) and applies it
to the `2**n` amplitudes, so **one stabilizer costs `16 * 4**n` bytes — quadratic in the
Hilbert dimension** — while the state itself is `16 * 2**n`. A decision runs `n` such
projections.

Every width was measured in a fresh process, because `ru_maxrss` is a process
high-water mark and several widths in one process would report the largest peak for all
of them. Each child caps its address space at a declared **8 GiB** over its post-import
baseline (`RLIMIT_AS`), runs the tier's own arithmetic, and attaches **no verdict**:
at or below the bound the referee's own answer is recorded beside the price, above it
the referee still returns `UNRESOLVED` and only the price is new.

| qubits | decision (s) | peak traced | peak RSS | one Pauli operator | referee |
|---|---|---|---|---|---|
| 6 | 0.02 | 0.2 MiB | 196 MiB | 0.06 MiB | VALID |
| 8 | 0.07 | 1.5 MiB | 196 MiB | 1.0 MiB | VALID |
| **10** (the bound) | **0.25** | **20 MiB** | 196 MiB | 16 MiB | VALID |
| 11 | 1.31 | 80 MiB | 254 MiB | 64 MiB | UNRESOLVED |
| 12 | 5.1 | 320 MiB | 510 MiB | 256 MiB | UNRESOLVED |
| 13 | 20.3 | 1.25 GiB | 1.44 GiB | 1.0 GiB | UNRESOLVED |
| 14 | 88.4 | 5.0 GiB | 5.2 GiB | 4.0 GiB | UNRESOLVED |
| 15 | — | — | 5.2 GiB | **16 GiB** | UNRESOLVED (wall) |

At the declared budget the wall is at **15 qubits**: one operator is 16 GiB, twice the
budget, so the process raises `MemoryError` after building the 4 GiB intermediates
(13 s, 5.2 GiB of RSS) rather than being killed by the kernel. Everything below the
wall fitted. Between consecutive widths the peak grows by **3.96–4.00x** — the four is
not fitted, it is the arithmetic — and the time by **3.9–5.3x**.

The practical reading: 13 qubits is the last comfortable width (20 s, 1.25 GiB), 14 is
out of interactive reach (88 s, 5 GiB) but fits the budget, and 15 does not fit. Below
11 the traced peak is 0.2–20 MiB and peak RSS is just the interpreter baseline, which
is the useful part of the measurement: **inside its declared bound the tier is cheap**
— 0.2 s and 20 MiB at ten qubits — so what stops it is memory, not arithmetic.

Two limits on that statement. The 8 GiB budget is the record's, not the host's: this
host had 15 GiB available and 29 GiB total, and a larger budget reaches 14 with room
to spare, which the `16 * 4**n` model predicts. And the wall is a property of **this
implementation**: a Pauli string can be applied in `O(2**n)` memory as single-qubit
gates plus index permutations (no dense operator), which would move the wall by
several qubits. The record prices the tier that exists, and says that none of these
numbers is a claim about what the question costs in principle.

## 6. The admission policy is the same one, and it is checked here too

The synthesis candidate satisfies the `Candidate` protocol in `discovery.py`
(identity, circuit, claims) and goes through the same sealed verdict with the same four
separate answers. The record includes the case that matters:

* a valid construction offered under `{"depth": 1}` comes out **`VALID` and `UNMET`** —
  the circuit is right, it is simply not the improvement it was offered as — and the
  promotion is refused with "does not improve its resource claim (depth: claimed 1,
  measured 12)". An unmet claim is never reported as a wrong circuit.

Writing that test caught a real bug in the first draft of this module: the claim check
was being folded into the semantic verdict, so an unmet claim produced `INVALID`, which
is a false statement about the circuit. The two are separate fields for exactly this
reason.

## 7. What this is not

* **Not a general synthesiser.** The construction space is a grammar of named wireings
  (chain, star, balanced binary tree, seeded random tree, ring). The engine selects
  among constructions it knows how to build; it cannot invent a topology outside the
  grammar, and it cannot reach a target whose construction family has not been written.
* **Four targets, not a family of them.** GHZ, the ring graph state, the two-dimensional
  cluster and an irregular graph (the last two through one edge-set-driven family set).
  A target outside those still needs its own construction family, which is the honest
  statement of the current generality.
* **The scheduler is exact only within its search budget.** Deciding whether a
  `Δ`-colouring exists is NP-complete, so `graph_layers` runs a complete backtracking
  search under a node budget, pruned by a necessary condition (a vertex cannot have more
  uncoloured edges than free colours). A search that finishes gives a proven optimum; a
  search that hits the budget decides nothing, so the result is **not** called minimal;
  it is still the `Δ + 1` colouring Vizing's own argument *constructs* (fan rotations,
  Misra–Gries form), which the theorem bounds to at most one layer above the minimum, so
  the schedule stays usable where the peel's would not be. Only if that construction
  failed its own verification would the peel stand, and the result would say so
  (`status: construction_refused`). The seven Experiment E targets never search at all
  (the peel already attains `Δ` on six of them, and K9 is refuted by the capacity count);
  of the six class-2 cases five are refuted at zero
  nodes by the layer-capacity bound and the Petersen graph takes 31 nodes to exhaust.
  A pathological graph would be reported as an unproven schedule rather than an optimum.
* **The dense tier's ceiling is a price for one implementation.** It stops at 15 qubits
  under an 8 GiB budget because each stabilizer is a dense `2**n x 2**n` operator. An
  `O(2**n)`-memory projection would move the wall, and that is not measured here.
* **The ranking is a stated preference.** `(depth, operations, max_edge_span)` orders
  the candidates, and the Pareto frontier is reported alongside it, because collapsing
  depth and connectivity into one number would be a preference dressed as a
  measurement.
* **No resource advance over a known-optimal preparation is claimed.** Every GHZ
  family here is `n` operations and `n-1` two-qubit gates — the same count as the
  textbook chain — so what is demonstrated is *construction from a specification and
  exact verification of the alternatives*, including their depth and connectivity
  costs, not a cheaper GHZ.

## 8. Reproducing

```bash
python scripts/measure_vqc_synthesis.py            # writes the record
pytest tests/test_vqc_synthesis.py tests/test_vqc_placement.py \
       tests/test_vqc_synthesis_record.py
```

The record pins the SHA-256 of `synthesis.py`, `placement.py`, `discovery.py`,
`equivalence.py`, `stabilizer.py` and the script, and refuses to be written at all if
any of its own checks fail.

## 9. Cross-references

* [`VQC_CORE.md`](VQC_CORE.md) §8 for the rewrites loop this builds past, §8.3 for
  where synthesis sits in the pipeline, §8.3.1 for the two referees' agreement,
  §8.3.2 for the topology measurement, §8.3.3 for the graph-state schedules and
  §8.3.4 for the dense tier's measured ceiling.
* [`VQC_RUST_CORE.md`](VQC_RUST_CORE.md) for the native execution core the referee
  calls, and what is deliberately still Python.
* [`VQC_DESIGNER_MAP.md`](VQC_DESIGNER_MAP.md) for the route map — entry points, the 24
  modules, the records and producers, and how a specification becomes a verified
  schedule — as distinct from this page, which carries the measured claims.
* [`REPOSITORY_CANONICAL_DECISION.md`](REPOSITORY_CANONICAL_DECISION.md) for which
  checkout is canonical and what is stranded in the other one.
* [`VQC_PHASE0_CAPABILITY_INVENTORY.md`](VQC_PHASE0_CAPABILITY_INVENTORY.md) for the
  capability row this experiment moves.

# Benchmark III — decision note: no ladder read in the submission

**Status:** `experimental`. The prototype
([`scripts/prototype_dfthcsos_dirty_ladder.py`](../scripts/prototype_dfthcsos_dirty_ladder.py))
is measured and word-checked but not shipped. This note records the decision and the
measurements behind it, so the question does not get reopened without new data.

**Question.** Can a sublinear-Toffoli read — a shared comparator ladder plus a block tree —
beat the shipped segment-tree QROM on the Benchmark III angle family?

**Decision.** No. Not on the shipped family (`brot = 10`, `J = 10`), and not because the
construction is wrong: its addressing really is 21.7x cheaper. It loses because the read is
charged in Toffolis and the payload is not, so the ladder's saving is spent on `ccx` loads,
and the 128-qubit cap then forbids the split wide enough to make the addressing cheap.

## 1. The four regimes

`T` is per circuit against the shipped tree's 3,974 Toffolis. "In cap" means the read fits
128 qubits. Every entry is a built circuit that passes the competition's own
`recover_words_from_circuits`.

| regime | delta table | `J = 4`, `low = 5` (in cap) | `J = 10` (shipped; cap admits `low <= 2`) |
|---|---|---|---|
| 1. dense, real | 40.0 set bits/address, no uniform block | 64,115 = **16.1x** | 160,064 = **40.3x** |
| 2. sparse, random | up to ~0.87 set bits/address | wins below **0.87** | wins at 0.10–0.50 (`low = 2`, 0.79–0.96x), loses from 0.75 |
| 3. sparse, fully populated | 1 bit on every address | 3,638 vs 3,453 = **1.05x** | 4,765 vs 3,770 = **1.26x** at `low = 2`; `low = 0` is the tree |
| 4. structured | payload in 8 runs / 1 run | 280 vs 131 (**2.14x**) / 20 vs 14 (**1.43x**) | 122 vs 122 and 15 vs 15 at `low = 0`; every in-cap split loses (2.0x to 4.4x) |

The ladder wins in exactly one regime: payload sparse enough to leave **empty addresses**
(below 0.87), at a split wide enough to amortise (`low >= 5`, which needs ~60 free ancillas).
That combination exists at `J = 4`, `brot = 10` — not at the shipped `J = 10`, where 100 payload
qubits leave `low = 2`.

An earlier version of this note claimed an 8% win on a populated table. That was measured against
a tree whose break only collapsed single repeated cells; with row-equality breaking on both reads
the ladder is 5% dearer at `J = 4` and 26% dearer at `J = 10`, so the claim is withdrawn. Every
tree count in this note is the post-port one.

## 2. Why the cap, not the table, settles the shipped family

`--low-scan` costs every split with its qubit count. At `J = 10` the in-cap set is `{0, 1, 2}`
and `low = 3` already needs 134 qubits. `--low-matrix` walks all eight densities on all four layouts:

| layout | tree `T` (0.10 -> 1.00) | best in-cap `low` | ladder `T` there | verdict |
|---|---|---|---|---|
| random | 1,406 -> 3,691 | 2 up to 0.50, then 0 | 1,105 -> 2,952, then the tree | wins 0.79–0.96x, only where sparsity leaves empty addresses |
| per-address | 386 -> 3,770 | 0 at every density | = tree (1.00x) | never wins |
| blocked | 89 -> 122 | 0 at every density | = tree (1.00x) | never wins |
| contiguous | 15 at every density | 0 at every density | = tree (1.00x) | never wins |
| flat comparator (`low = 12`, `S > rows`) | 6,980 / 7,548 | out of cap (7,094 / 7,662 qubits) | — | dominated: 58x the tree's qubits for more Toffolis |

`--low-matrix` also runs the same grid at `J = 4`, where `low = 5` is in cap — the one
configuration that could have rescued the structured layouts. It does not:

| layout | tree `T` | best in-cap `low` | ladder `T` there | vs tree |
|---|---|---|---|---|
| random | 1,400 -> 3,637 | **5** up to 0.75, then 0 | 576 -> 3,067 | **0.41–0.89x** |
| per-address | 355 -> 3,453 | 0 at every density | = tree | 1.00x |
| blocked | 103 -> 131 | 0 at every density | = tree | 1.00x |
| contiguous | 14 -> 15 | 0 at every density | = tree | 1.00x |

Two readings, and neither is about the table being hard. The wide split is worth 2.4x where the
payload is random and sparse (576 against 1,400 at 0.10 bits/address), and worth nothing to a
structured table, because those trees are 14–131 Toffolis — the ladder's 183-Toffoli addressing
alone exceeds the read it would replace. Structure is not sparsity plus order; it is the case the
tree is already near-optimal on. So the earlier "crossing of 0.56 at `J = 10`" describes
`low = 2`, which is not the split to choose: at `low = 0` the read is the tree. **No in-cap split
beats the tree on the shipped family.**

The 0.56 figure itself is real arithmetic — 996 addressing Toffolis at `low = 2` against a
0.75 fixed-payload break-even — it is simply not the best admissible configuration.

## 3. What the early-break port changed: nothing, on the real table

`build_delta_circuit` compared each cell of a block to the scalar `delta[lo, 0]`, so it only
collapsed blocks that were one repeated 0/1 pattern. It now compares each row to `delta[lo]`,
which is a strict generalisation (an old match is still a match) and so can only remove nodes.

Measured with `--break-audit`:

| table | collapsible ranges | family Toffolis |
|---|---|---|
| shipped `submission.npz`, all 8 delta tables | **0, 0, 0, 0, 0, 0, 0, 0** | 3,974 x 8, unchanged |
| synthetic, 1 bit on every address, `J = 10` | 198 | **3,770** (204 saved) |

So the 204 Toffolis are a property of a *synthetic* table that repeats rows, not of this one.
The real delta tables have no uniform range at any size, which is also why the tree is exactly
`rows - 2` and why the read is impossible to improve by collapsing. The port is kept because it
is strictly better and identical in cost on any table without repeats.

The vendored reference (`example_submission/angle_k_reference_circuit.py` inside the bundle) still
uses the scalar form. It is third-party code and is left untouched; our builder matches its
semantics wherever the scalar rule fires, and only differs where it finds a second collapse.

## 4. What is left, and what is not

- **`J` through precision.** `brot = 9, J = 11` is 122 qubits, `K = 7`, 55,636 Toffolis across
  both passes — 12.5% below the shipped family — at a measured angular cost of 6.136e-3 rad.
  Whether that bit is affordable is a `lambda_eff` question and needs the FCIDUMP.
- **The two-pass factor is structural.** The checker re-executes the same bytes in reverse and
  charges both, so `T = 2 * sum(circuits)` is not a bookkeeping choice. There is no second
  payload to fuse and no tree to share between passes.
- **The read is not a lever.** Four regimes, one failure mode: the ladder moves addressing out
  of the Toffoli count only by moving payload into it, and this table's payload is 40x the
  addressing it saves.

## 5. Reproducing every number here

```bash
export PYTHONPATH="$PWD/apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix:$PYTHONPATH"

# the shipped family, and whether any block can collapse
build/qvcc-venv313/bin/python scripts/build_dfthcsos_sub128.py --brot 10 --batch-size 10 --break-audit

# the four regimes (J=4 split that fits, and the J=10 in-cap answer)
build/qvcc-venv313/bin/python scripts/prototype_dfthcsos_dirty_ladder.py --sweep --rows 3976 --batch-size 4
build/qvcc-venv313/bin/python scripts/prototype_dfthcsos_dirty_ladder.py --low-scan --rows 3976 --batch-size 10 --set-bits 1.0 --placement per-address
build/qvcc-venv313/bin/python scripts/prototype_dfthcsos_dirty_ladder.py --low-matrix --rows 3976 --batch-size 10

# the real table's density, which is what puts it in regime 1
build/qvcc-venv313/bin/python scripts/prototype_dfthcsos_dirty_ladder.py --density
```

Pinned so a builder change cannot move them silently:
`tests/test_dyn_dfthcsos_sub128_family.py` (family dimensions and digest, the ladder's
addressing counts and both break-evens, the measured crossing, the structured pair, and the
`J = 10` in-cap split table). Analysis in `docs/DFTHCSOS_SUB128_SCOPE.md` §5.1–§5.7;
changelog in `CHANGELOG.md`.

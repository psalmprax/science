"""Pin the Benchmark III angle family so a builder change cannot move it silently.

The family's dimensions *are* the result: `brot = 10`, `J = 10` is the only configuration
that fits the 128-qubit cap with a useful margin and the smallest circuit count, so a
change to the builder that quietly trades one against the other is a change to the answer,
not a refactor. This test states the answer and fails if it moves:

    peak qubits 123 (cap 128, five spare)
    circuits K 8 for 75 angles at J = 10
    Toffoli-like gates 3,974 per circuit
    two-pass charge 63,584 = 2 x 8 x 3,974

It also pins the arithmetic the numbers rest on — the selection width, the cap identity
`12 + J*brot + (12-1) <= 128`, and the fact that the reported total is exactly twice the
per-circuit sum, which is the two-pass structure rather than a coincidence.

`kickmix` is a CPython 3.13 extension, and this suite is usually run by a different
interpreter, so the family is built in a subprocess using the interpreter that can load it
(`build/qvcc-venv313`, created by `scripts/setup_verifier_env.py`). When the extension *is*
importable in-process the same assertions run directly. If neither is available the test
skips with the command that fixes it, rather than passing on nothing.

Word recovery over all 3,976 addresses is not repeated here — it takes minutes and is what
`python scripts/build_dfthcsos_sub128.py --verify` is for. These pins are about the size of
the circuits, which is the quantity a refactor is most likely to move.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
SCRIPTS = ROOT / "scripts"
VENV_PYTHON = ROOT / "build" / "qvcc-venv313" / "bin" / "python"

BROT, BATCH = 10, 10
PINNED = {
    "selection_qubits": 12,
    "peak_qubits": 123,
    "qubit_cap": 128,
    "headroom": 5,
    "circuits_K": 8,
    "toffoli_total_two_passes": 63584,
}
PINNED_TOFFOLI_PER_CIRCUIT = 3974
# sha256 of the canonical size summary: any change in widths or per-circuit counts changes it,
# even one that leaves every total in PINNED intact.
PINNED_SUMMARY_SHA256 = "3957afa368c79e7ec52ccddec4ae768cb988539d78418a7276136917e945c88e"


def _in_process() -> bool:
    for path in (BUNDLE / "kickmix", BUNDLE / "verifiers", SCRIPTS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import kickmix  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - ImportError or OSError from an ABI
        # mismatch; either way this interpreter cannot use the module.
        return False


def _report_key(name: str) -> dict:
    """The size summary for one family, built by whichever interpreter can load kickmix."""
    if _in_process():
        import build_dfthcsos_sub128 as B
        import numpy as np

        with np.load(B.DEFAULT_NPZ, allow_pickle=False) as archive:
            words = np.array(archive["angle_words"], dtype=np.int64)
        report = B.report(words, BROT, BATCH, compressed=False)
        report.pop("circuits")
        return report

    if not VENV_PYTHON.exists():
        pytest.skip(
            "kickmix is not importable by this interpreter and "
            f"{VENV_PYTHON.relative_to(ROOT)} does not exist; "
            "run `python3.13 scripts/setup_verifier_env.py` to build it"
        )
    code = (
        "import json, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import build_dfthcsos_sub128 as B\n"
        "with np.load(B.DEFAULT_NPZ, allow_pickle=False) as z:\n"
        "    words = np.array(z['angle_words'], dtype=np.int64)\n"
        f"r = B.report(words, {BROT}, {BATCH}, compressed=False)\n"
        "r.pop('circuits')\n"
        "print(json.dumps(r))\n"
    )
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BUNDLE / "kickmix"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    completed = subprocess.run(
        [str(VENV_PYTHON), "-c", code], capture_output=True, text=True, env=env
    )
    if completed.returncode != 0:
        pytest.fail(f"building the family failed:\n{completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def family() -> dict:
    return _report_key("primary")


def test_family_dimensions_are_pinned(family: dict) -> None:
    """The width, the circuit count and the Toffoli counts must not drift."""
    for key, expected in PINNED.items():
        assert family[key] == expected, (
            f"{key} moved: {family[key]!r} != {expected!r}. "
            "This is a result, not an implementation detail — update the docs and "
            "docs/DFTHCSOS_SUB128_SCOPE.md deliberately if the change is intended."
        )
    counts = family["toffoli_per_circuit"]
    assert len(counts) == PINNED["circuits_K"], counts
    assert set(counts) == {PINNED_TOFFOLI_PER_CIRCUIT}, counts


def test_cap_identity_holds(family: dict) -> None:
    """`selection + J*brot + (selection-1) == peak`, and it is inside the cap."""
    selection = family["selection_qubits"]
    peak = selection + BATCH * BROT + (selection - 1)
    assert peak == family["peak_qubits"]
    assert family["peak_qubits"] + family["headroom"] == family["qubit_cap"] == 128


def test_two_pass_charge_is_exactly_double(family: dict) -> None:
    """The reported total is two applications of the family, not a rounded Sum."""
    assert family["toffoli_total_two_passes"] == 2 * sum(family["toffoli_per_circuit"])


@pytest.mark.slow
def test_family_recovers_every_angle_word() -> None:
    """The circuits load the table exactly, checked by the competition's own recovery.

    Size pins cannot see a read that is the right size and the wrong function, so this runs
    `recover_words_from_circuits` over all 3,976 addresses: forward and reverse, every shot,
    selection unchanged, workspace clean, phase unflipped, padding zero. It is marked `slow`
    because it simulates the whole family twice; the fast pins above are what the inner loop
    runs.
    """
    if _in_process():
        import build_dfthcsos_sub128 as B
        import numpy as np

        with np.load(B.DEFAULT_NPZ, allow_pickle=False) as archive:
            words12 = np.array(archive["angle_words"], dtype=np.int64)
        words = B.requantise(words12, 12, BROT)
        family = B.build_family(words, BROT, BATCH, compressed=False)
        recovered, layout, _toffoli = B._scoreinfra().recover_words_from_circuits(
            family, num_vectors=words.shape[0], num_angles=words.shape[1],
            shots_per_address=2, seed=20261102)
        assert layout.brot == BROT and layout.batch_size == BATCH
        assert recovered.shape == words.shape
        assert np.array_equal(recovered, words), "the family does not load the table"
        return

    if not VENV_PYTHON.exists():
        pytest.skip(f"kickmix needs {VENV_PYTHON.relative_to(ROOT)}; run scripts/setup_verifier_env.py")
    code = (
        "import json, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import build_dfthcsos_sub128 as B\n"
        "with np.load(B.DEFAULT_NPZ, allow_pickle=False) as z:\n"
        "    w12 = np.array(z['angle_words'], dtype=np.int64)\n"
        f"words = B.requantise(w12, 12, {BROT})\n"
        f"family = B.build_family(words, {BROT}, {BATCH}, compressed=False)\n"
        "rec, layout, toffoli = B._scoreinfra().recover_words_from_circuits(\n"
        "    family, num_vectors=words.shape[0], num_angles=words.shape[1],\n"
        "    shots_per_address=2, seed=20261102)\n"
        "print(json.dumps({'exact': bool(np.array_equal(rec, words)),\n"
        "                  'shape': list(rec.shape), 'brot': layout.brot,\n"
        "                  'J': layout.batch_size, 'toffoli': toffoli}))\n"
    )
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BUNDLE / "kickmix"), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    completed = subprocess.run(
        [str(VENV_PYTHON), "-c", code], capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        pytest.fail(f"word recovery failed to run:\n{completed.stderr[-3000:]}")
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["brot"] == BROT and result["J"] == BATCH, result
    assert result["shape"] == [3976, 75], result
    assert result["exact"], f"the family does not load the table: {result}"
    assert result["toffoli"] == PINNED["toffoli_total_two_passes"], result


# ── the O(sqrt(A)) ladder: addressing counts and the break-even, pinned ──────


def test_ladder_addressing_and_break_even_are_pinned() -> None:
    """The ladder's addressing counts, and the break-even that decides who wins.

    `ladder_rule` is what the full-table numbers rest on, so it is pinned rather than
    trusted: addressing at five splits, the payload's gate type, and the break-even at
    each `J`. The gate type is the whole story -- a load with two controls is a `ccx`,
    which the checker counts as Toffoli-like, while `low = 0` loads by `cx` and so
    degenerates to exactly the tree (`3,974` Toffolis at any payload).

    The measured crossing is a *different* number and is pinned by
    `test_ladder_density_crossing_is_measured`: the tree's early-break collapses
    all-zero subtrees, so the tree cheapens as the payload thins and a fixed-payload
    break-even overstates the ladder's room.
    """
    sys.path.insert(0, str(SCRIPTS))
    from prototype_dfthcsos_dirty_ladder import ladder_rule

    rows = 3976
    addressing = {low: ladder_rule(rows, low, 0)["addressing"] for low in (0, 2, 4, 5, 6)}
    assert addressing == {0: 3974, 2: 996, 4: 275, 5: 183, 6: 185}, addressing
    assert ladder_rule(rows, 5, 0)["ladder_ancillas"] == 62

    degenerate = ladder_rule(rows, 0, 63932)
    assert degenerate["payload_gate"] == "x", degenerate
    assert degenerate["total"] == 3974, "low = 0 must cost exactly the shipped tree"
    assert ladder_rule(rows, 5, 63932)["payload_gate"] == "ccx"
    assert ladder_rule(rows, 5, 63932)["total"] == 64115

    # Break-even: the payload the tree can no longer amortise, at the best in-cap split.
    # J = 4 -> low = 5 (183 addressing); J = 10, the shipped family -> low = 2 (996).
    assert rows - 2 == 3974
    assert 3974 - 183 == 3791 and round(3791 / rows, 3) == 0.953
    assert 3974 - 996 == 2978 and round(2978 / rows, 2) == 0.75
    # Both are far below what the table actually carries (16.1 and 40.0 per address).
    assert 63932 / rows == 16.08 or abs(63932 / rows - 16.1) < 0.05
    assert abs(159068 / rows - 40.0) < 0.05


@pytest.mark.slow
def test_ladder_density_crossing_is_measured() -> None:
    """The ladder wins only below ~0.87 set bits/address, and 1.0 already loses.

    Every circuit here is costed and then checked with the competition's own word
    recovery, so a cheaper-looking read that does not load the table fails. This is the
    pin behind the `--sweep` table in docs/DFTHCSOS_SUB128_SCOPE.md §5.2.
    """
    if not VENV_PYTHON.exists():
        pytest.skip(
            f"kickmix needs {VENV_PYTHON.relative_to(ROOT)}; run scripts/setup_verifier_env.py"
        )
    code = (
        "import json, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import build_dfthcsos_sub128 as B\n"
        "import prototype_dfthcsos_dirty_ladder as L\n"
        "C = B._scoreinfra()\n"
        "BROT, J, ROWS = 10, 4, 3976\n"
        "out = {}\n"
        "for d in (0.10, 0.75, 1.00):\n"
        "    delta = L.sparse_table(ROWS, J, d)\n"
        "    row = {'payload': L.popcount(delta)}\n"
        "    for name, circ in (('tree', B.build_delta_circuit(delta, BROT, compressed=False)),\n"
        "                      ('ladder', L.build_ladder(delta, BROT, low=5,\n"
        "                                                   early_break=False))):\n"
        "        row[name] = C.circuit_costs((circ,))[0]['max_magic']\n"
        "        rec, _, _ = C.recover_words_from_circuits((circ,), num_vectors=ROWS,\n"
        "            num_angles=J, shots_per_address=2, seed=20261102)\n"
        "        row[name + '_ok'] = bool(np.array_equal(rec, delta))\n"
        "    out[str(d)] = row\n"
        "print(json.dumps(out))\n"
    )
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BUNDLE / "kickmix"), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    completed = subprocess.run(
        [str(VENV_PYTHON), "-c", code], capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        pytest.fail(f"density sweep failed to run:\n{completed.stderr[-3000:]}")
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    # Ladder counts are three lower than the first sweep: the builder now traverses the
    # 125 blocks the table has, not the 128 the address space allows (see
    # `test_rule_is_exact_for_non_power_of_two_block_counts`).
    # Tree counts are the post-port ones: the shipped builder now breaks on row equality,
    # not cell equality, so it collapses repeated rows on these synthetic tables.
    pinned = {"0.1": (1400, 581), "0.75": (3447, 3165), "1.0": (3637, 4159)}
    for key, (tree_t, ladder_t) in pinned.items():
        row = result[key]
        assert row["tree"] == tree_t and row["ladder"] == ladder_t, (key, row)
        assert row["tree_ok"] and row["ladder_ok"], (key, row)
    # The sparsest non-empty table there is: one set bit on every address. It loses.
    assert result["1.0"]["ladder"] > result["1.0"]["tree"], result["1.0"]
    assert result["0.75"]["ladder"] < result["0.75"]["tree"], result["0.75"]


# ── block counts, and where the payload sits ─────────────────────────────────


def _run_venv(code: str) -> list:
    """Run a snippet under the interpreter that can import kickmix, and parse its JSON."""
    if not VENV_PYTHON.exists():
        pytest.skip(
            f"kickmix needs {VENV_PYTHON.relative_to(ROOT)}; run scripts/setup_verifier_env.py"
        )
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BUNDLE / "kickmix"), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    completed = subprocess.run(
        [str(VENV_PYTHON), "-c", code], capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        pytest.fail(f"snippet failed:\n{completed.stderr[-3000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_rule_is_exact_for_non_power_of_two_block_counts() -> None:
    """`ceil(rows/S)` blocks, not the power of two the address space could hold.

    100 addresses at `low = 2` is 25 blocks, so the block tree costs 23 Toffolis; the
    address space would allow 32 blocks and 30. The ladder builder used to walk the
    padding, which is why every full-table count was three Toffolis high.
    """
    sys.path.insert(0, str(SCRIPTS))
    from prototype_dfthcsos_dirty_ladder import ladder_rule

    addressing = {low: ladder_rule(100, low, 0)["addressing"] for low in range(5)}
    assert addressing == {0: 98, 1: 48, 2: 27, 3: 23, 4: 33}, addressing
    # The full table: 3,976 rows at low = 5 is 125 blocks, and the padded tree would
    # have been 126 nodes = three Toffolis more than the table needs.
    assert ladder_rule(3976, 5, 0)["block_tree"] == 123
    assert 3976 // 32 == 124


@pytest.mark.slow
def test_built_ladder_matches_the_rule_on_uneven_blocks() -> None:
    """The rule and the circuit agree where the block count is not a power of two.

    A rule that only matches on powers of two would pass on the 64-address slice and
    fail on the shipped table, so this pins the built circuits at 100 addresses, where
    `low = 2` gives 25 blocks and the padding cannot be walked unnoticed.
    """
    code = (
        "import json, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import build_dfthcsos_sub128 as B\n"
        "import prototype_dfthcsos_dirty_ladder as L\n"
        "C = B._scoreinfra()\n"
        "delta = L.sparse_table(100, 4, 1.0)\n"
        "payload = L.popcount(delta)\n"
        "out = []\n"
        "for low in range(5):\n"
        "    circ = L.build_ladder(delta, 10, low=low, early_break=False)\n"
        "    measured = C.circuit_costs((circ,))[0]['max_magic']\n"
        "    out.append([low, measured, L.ladder_rule(100, low, payload)['total']])\n"
        "print(json.dumps(out))\n"
    )
    for low, measured, rule in _run_venv(code):
        assert measured == rule, f"low={low}: built {measured}, rule {rule}"


@pytest.mark.slow
def test_placement_decides_the_tree_not_the_density() -> None:
    """At equal payload, clustering it makes the tree cheaper, so the ladder loses.

    The ladder is layout-invariant -- its `ccx` count is the payload count -- while the
    tree's early-break collapses whole blocks of identical cells. Same density (0.5),
    same 1,988 set bits, three layouts: trees of 3,160 / 122 / 15 Toffolis.
    """
    code = (
        "import json, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import build_dfthcsos_sub128 as B\n"
        "import prototype_dfthcsos_dirty_ladder as L\n"
        "C = B._scoreinfra()\n"
        "out = []\n"
        "for placement in ('random', 'blocked', 'contiguous'):\n"
        "    delta = L.sparse_table(3976, 4, 0.5, placement=placement)\n"
        "    tree = C.circuit_costs((B.build_delta_circuit(delta, 10, compressed=False),))\n"
        "    ladder = C.circuit_costs((L.build_ladder(delta, 10, low=5, early_break=False),))\n"
        "    out.append([placement, L.popcount(delta), tree[0]['max_magic'],\n"
        "                ladder[0]['max_magic']])\n"
        "print(json.dumps(out))\n"
    )
    by_layout = {name: (payload, tree, ladder) for name, payload, tree, ladder
                 in _run_venv(code)}
    assert set(by_layout) == {"random", "blocked", "contiguous"}, by_layout
    payloads = {layout[0] for layout in by_layout.values()}
    ladders = {layout[2] for layout in by_layout.values()}
    assert payloads == {1988}, payloads
    assert ladders == {2171}, "the ladder must not depend on where the payload sits"
    assert by_layout["random"][1] == 3104, by_layout["random"]
    assert by_layout["blocked"][1] == 122 and by_layout["contiguous"][1] == 15, by_layout


def test_real_table_has_no_uniform_block() -> None:
    """Early-break needs a uniform block; the shipped angle table has none.

    This is what makes the random-placement sweep the representative layout and the
    contiguous one an unrepresentative bound: a real table's tree cannot collapse, so
    its cost stays at one node per address (3,974 = rows - 2).
    """
    import numpy as np

    npz = BUNDLE / "sample_submission" / "dfthcsos" / "submission.npz"
    with np.load(npz, allow_pickle=False) as archive:
        words = np.array(archive["angle_words"], dtype=np.int64)
    assert words.shape[0] == 3976, words.shape
    for width in (16, 62, 124, 256):
        usable = (words.shape[0] // width) * width
        blocks = words[:usable].reshape(-1, width, words.shape[1])
        uniform = (blocks == blocks[:, :1, :]).all(axis=(1, 2))
        assert not uniform.any(), f"{int(uniform.sum())} uniform {width}-address blocks"


@pytest.mark.slow
def test_early_break_moves_the_crossing_to_0_91() -> None:
    """The ladder with the tree's own collapsing rule: 0.87 -> 0.91, and still 8% worse.

    Also pins the per-address regime, where every address carries a bit and the tree is
    exactly `rows - 2` because it has nothing to collapse: the ladder is 8% cheaper at
    one bit per address, which is why the real table's 40 bits per address settles it.
    """
    code = (
        "import json, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import build_dfthcsos_sub128 as B\n"
        "import prototype_dfthcsos_dirty_ladder as L\n"
        "C = B._scoreinfra()\n"
        "out = []\n"
        "for placement, density in (('random', 0.75), ('random', 1.0),\n"
        "                           ('per-address', 1.0)):\n"
        "    delta = L.sparse_table(3976, 4, density, placement=placement)\n"
        "    tree = C.circuit_costs((B.build_delta_circuit(delta, 10,\n"
        "        compressed=False),))[0]['max_magic']\n"
        "    brk = C.circuit_costs((L.build_ladder(delta, 10, low=5,\n"
        "        early_break=True),))[0]['max_magic']\n"
        "    plain = C.circuit_costs((L.build_ladder(delta, 10, low=5,\n"
        "        early_break=False),))[0]['max_magic']\n"
        "    out.append([placement, density, tree, brk, plain, L.popcount(delta)])\n"
        "print(json.dumps(out))\n"
    )
    rows = {(row[0], row[1]): row for row in _run_venv(code)}
    assert set(rows) == {("random", 0.75), ("random", 1.0), ("per-address", 1.0)}, rows
    # The dense tables carry one bit per address; 0.75 carries 2,982 of them.
    assert (rows[("random", 0.75)][5], rows[("random", 1.0)][5],
            rows[("per-address", 1.0)][5]) == (2982, 3976, 3976), rows
    assert (rows[("random", 0.75)][2], rows[("random", 0.75)][3],
            rows[("random", 0.75)][4]) == (3447, 3067, 3165), rows[("random", 0.75)]
    assert (rows[("random", 1.0)][2], rows[("random", 1.0)][3],
            rows[("random", 1.0)][4]) == (3637, 4032, 4159), rows[("random", 1.0)]
    # Every address populated: the tree collapses the repeated rows it finds (3,453, not
    # 3,974), and with the same rule on both reads the ladder is now 5% dearer, not 8%
    # cheaper. That withdrawal is the point of this pin.
    placed = rows[("per-address", 1.0)]
    assert placed[2] == 3453, placed
    assert (placed[3], placed[4]) == (3638, 4159), placed
    assert placed[3] > placed[2], "with the ported break the ladder must lose here too"


@pytest.mark.slow
def test_structured_and_j10_reads_are_measured_like_for_like() -> None:
    """Structured layouts, and the shipped `J = 10` family, at the same density.

    The first two rows are the like-for-like comparison the block-range break makes
    possible: 296 against 131 with the payload in eight runs, 74 against 14 with it in
    one. The third is the family that ships, where 100 payload qubits leave only
    `low = 2`, so the ladder's addressing is 996 Toffolis and one bit per address costs
    it 20% more than the tree rather than 8% less.
    """
    code = (
        "import json, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import build_dfthcsos_sub128 as B\n"
        "import prototype_dfthcsos_dirty_ladder as L\n"
        "C = B._scoreinfra()\n"
        "out = []\n"
        "for placement, J, low in (('blocked', 4, 5), ('contiguous', 4, 5),\n"
        "                          ('per-address', 10, 2)):\n"
        "    delta = L.sparse_table(3976, J, 1.0, placement=placement)\n"
        "    tree = C.circuit_costs((B.build_delta_circuit(delta, 10,\n"
        "        compressed=False),))[0]['max_magic']\n"
        "    lad = C.circuit_costs((L.build_ladder(delta, 10, low=low,\n"
        "        early_break=True),))[0]['max_magic']\n"
        "    out.append([placement, J, low, tree, lad, L.popcount(delta)])\n"
        "print(json.dumps(out))\n"
    )
    rows = {row[0]: row for row in _run_venv(code)}
    assert set(rows) == {"blocked", "contiguous", "per-address"}, rows
    assert all(row[5] == 3976 for row in rows.values()), rows
    # The offset ladder is built only where a load addresses it, so the structured reads
    # carry no fixed `2(S - 1)` tax: 280 (2.14x) and 20 (1.43x).
    assert rows["blocked"][3:5] == [131, 280], rows["blocked"]
    assert rows["contiguous"][3:5] == [14, 20], rows["contiguous"]
    shipped = rows["per-address"]
    assert (shipped[1], shipped[2]) == (10, 2), shipped
    assert shipped[3] == 3770 and shipped[4] == 4765, shipped
    assert shipped[4] > shipped[3], "at J=10 the ladder must lose at one bit per address"


@pytest.mark.slow
def test_j10_admits_only_narrow_splits_and_none_of_them_wins() -> None:
    """At `J = 10` the cap admits `low <= 2`, and the best of those is `low = 0`.

    The 0.56 crossing quoted for the shipped family is a statement about `low = 2`, which
    is not the split to choose: `low = 0` -- the plain tree -- is cheaper, 3,770 against
    4,765 at one bit per address. The ladder's advantage only appears at `low = 3`, which
    needs 134 qubits and so is out of cap. Pinning the qubit counts pins the reason.
    """
    code = (
        "import json, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import build_dfthcsos_sub128 as B\n"
        "import prototype_dfthcsos_dirty_ladder as L\n"
        "C = B._scoreinfra()\n"
        "delta = L.sparse_table(3976, 10, 1.0, placement='per-address')\n"
        "out = []\n"
        "for low in (0, 1, 2, 3):\n"
        "    circ = L.build_ladder(delta, 10, low=low, early_break=True)\n"
        "    cost = C.circuit_costs((circ,))[0]\n"
        "    out.append([low, cost['num_qubits'], cost['max_magic']])\n"
        "print(json.dumps(out))\n"
    )
    rows = {row[0]: (row[1], row[2]) for row in _run_venv(code)}
    assert rows[0] == (123, 3770) and rows[1] == (124, 5557), rows
    assert rows[2] == (127, 4765) and rows[3] == (134, 4279), rows
    in_cap = {low: toffoli for low, (qubits, toffoli) in rows.items() if qubits <= 128}
    assert in_cap == {0: 3770, 1: 5557, 2: 4765}, in_cap
    assert min(in_cap, key=in_cap.get) == 0, "the tree, at low = 0, must be the in-cap best"


@pytest.mark.slow
def test_the_decision_note_regimes_do_not_move() -> None:
    """One pin per regime of `docs/BENCHMARK3_LADDER_DECISION.md`.

    Each row is `(tree, best in-cap read)` for the case the note names, so a builder change
    that quietly improves one regime fails here instead of quietly invalidating the note.
    Regime 1 needs no circuit (`ladder_rule` is the arithmetic the note quotes); the rest are
    built and costed, and every build here is word-checked elsewhere in this file.
    """
    sys.path.insert(0, str(SCRIPTS))
    from prototype_dfthcsos_dirty_ladder import ladder_rule

    # Regime 1: the real table, J=4 and the shipped J=10, plus low=0 being the tree.
    assert ladder_rule(3976, 5, 63932)["total"] == 64115, "regime 1 at J=4 moved"
    assert ladder_rule(3976, 2, 159068)["total"] == 160064, "regime 1 at J=10 moved"
    assert ladder_rule(3976, 0, 63932)["total"] == 3974, "low=0 must cost the tree"

    code = (
        "import json, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import build_dfthcsos_sub128 as B\n"
        "import prototype_dfthcsos_dirty_ladder as L\n"
        "C = B._scoreinfra()\n"
        "cases = [('random', 4, 1.0, 5), ('random', 4, 0.75, 5),\n"
        "         ('per-address', 4, 1.0, 5), ('per-address', 10, 1.0, 2),\n"
        "         ('per-address', 10, 1.0, 0), ('blocked', 4, 1.0, 5),\n"
        "         ('contiguous', 4, 1.0, 5), ('blocked', 10, 1.0, 0),\n"
        "         ('contiguous', 10, 1.0, 0), ('blocked', 10, 0.1, 0),\n"
        "         ('contiguous', 10, 0.1, 0)]\n"
        "out = []\n"
        "for placement, J, density, low in cases:\n"
        "    delta = L.sparse_table(3976, J, density, placement=placement)\n"
        "    tree = C.circuit_costs((B.build_delta_circuit(delta, 10,\n"
        "        compressed=False),))[0]['max_magic']\n"
        "    lad = C.circuit_costs((L.build_ladder(delta, 10, low=low,\n"
        "        early_break=True),))[0]['max_magic']\n"
        "    out.append([placement, J, density, low, tree, lad])\n"
        "print(json.dumps(out))\n"
    )
    rows = {(row[0], row[1], row[2], row[3]): (row[4], row[5]) for row in _run_venv(code)}
    assert rows[("random", 4, 0.75, 5)] == (3447, 3067), rows[("random", 4, 0.75, 5)]
    assert rows[("random", 4, 1.0, 5)] == (3637, 4032), rows[("random", 4, 1.0, 5)]
    assert rows[("per-address", 4, 1.0, 5)] == (3453, 3638), rows[("per-address", 4, 1.0, 5)]
    assert rows[("per-address", 10, 1.0, 2)] == (3770, 4765), rows[("per-address", 10, 1.0, 2)]
    assert rows[("per-address", 10, 1.0, 0)] == (3770, 3770), rows[("per-address", 10, 1.0, 0)]
    assert rows[("blocked", 4, 1.0, 5)] == (131, 280), rows[("blocked", 4, 1.0, 5)]
    assert rows[("contiguous", 4, 1.0, 5)] == (14, 20), rows[("contiguous", 4, 1.0, 5)]
    assert rows[("blocked", 10, 1.0, 0)] == (122, 122), rows[("blocked", 10, 1.0, 0)]
    assert rows[("contiguous", 10, 1.0, 0)] == (15, 15), rows[("contiguous", 10, 1.0, 0)]
    # The structured verdict is not a property of one density: at 0.10 the trees are 89 and
    # 15 and `low = 0` matches each exactly, which is the matrix row the note quotes.
    assert rows[("blocked", 10, 0.1, 0)] == (89, 89), rows[("blocked", 10, 0.1, 0)]
    assert rows[("contiguous", 10, 0.1, 0)] == (15, 15), rows[("contiguous", 10, 0.1, 0)]

    # The verdicts, not just the counts: regime 2 wins sparse and loses dense, regimes 3
    # and 4 lose at every split, and `low = 0` is exactly the tree wherever the cap leaves
    # nothing wider.
    wins = rows[("random", 4, 0.75, 5)]
    loses = rows[("random", 4, 1.0, 5)]
    assert wins[1] < wins[0] and loses[1] > loses[0], (wins, loses)
    for key in (("per-address", 4, 1.0, 5), ("per-address", 10, 1.0, 2),
                ("blocked", 4, 1.0, 5), ("contiguous", 4, 1.0, 5)):
        assert rows[key][1] > rows[key][0], key
    for key in (("per-address", 10, 1.0, 0), ("blocked", 10, 1.0, 0),
                ("contiguous", 10, 1.0, 0), ("blocked", 10, 0.1, 0),
                ("contiguous", 10, 0.1, 0)):
        assert rows[key][1] == rows[key][0], key


def test_size_summary_digest_is_pinned(family: dict) -> None:
    """A digest over widths and per-circuit counts, so a reshuffle cannot hide in totals."""
    summary = json.dumps(
        {
            "selection": family["selection_qubits"],
            "peak": family["peak_qubits"],
            "k": family["circuits_K"],
            "per_circuit": family["toffoli_per_circuit"],
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(summary.encode("utf-8")).hexdigest()
    assert digest == PINNED_SUMMARY_SHA256, (
        f"family size summary changed: {digest}\nsummary: {summary}"
    )

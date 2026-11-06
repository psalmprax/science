"""Pins the ``w2``-parameterised curvature checkpoint in the Benchmark I champion builder.

Status: audit-only

``scripts/test_shared_pddd_circuit.py`` used to hard-code the curvature checkpoint for
``cfg.w2 == 7`` in five places (``checkpoint_start = 4``, the Gray-code span ``range(..., 6)``,
``alloc_qubits(3)``, the literal ``targets_dict`` keys, and ``if j < 6``). Any other ``w2``
mis-assembled the measurement records and crashed inside ``phase_from_records`` with an
``IndexError`` — which is a *mathematical* failure, not a software one, because the
record-index correspondence is what makes the HMR-based uncompute phase-exact.

Three things are pinned:

* the refactor is semantics-preserving — the ``w2 == 7`` champion still rebuilds
  byte-identical to ``ce04c9d5…``, so the parameterisation cannot have changed the shipped
  circuit;
* the checkpoint tracks ``w2`` — ``curve_checkpoint_values`` returns the top
  ``CURVE_CHECKPOINT_BITS`` Gray-coded K2 bits for whatever ``w2`` the config declares, and the
  ``w2 == 8`` build assembles, compiles and **passes the official verifier**;
* the ``f2 = 9`` / ``w2 = 8`` candidate that motivated the change is measured **worse** than
  the champion, with the reason it cannot be rescued — so a screen-positive margin does not
  get recorded as a win.

The kickmix-dependent tests are opt-in, following the convention of
``tests/test_dyn_invsqrt_champion_rebuild.py``:

    APEDP_Q_KICKMIX_PYTHON=/path/to/python3.13 pytest tests/test_dyn_invsqrt_w2_checkpoint.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(
    0,
    str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"),
)
sys.path.insert(
    0,
    str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt"),
)

KICKMIX_PYTHON = os.environ.get("APEDP_Q_KICKMIX_PYTHON")
PINNED_CHAMPION_SHA256 = "ce04c9d54dc42e42455444022b6652a6efb0be2630aca7b485d3efd4e58fa9de"

# Measured on the official verifier at 25,600 shots, this session. The rotation calibration
# is the champion's pinned one (only valid at the champion's rotation count), so this is a
# structural receipt rather than a score claim — see the note's §9.
PINNED_W2_8 = {
    "f2": 9,
    "num_qubits": 93,
    "max_toffoli_count": 1474,
    "score": 165427.86369320287,
    "verified_shots": 25600,
}
CHAMPION_SCORE = 157601.08


def _kickmix_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


# The kickmix extension lives outside the package tree, so inline scripts must put it (and the
# Benchmark I verifier) on ``sys.path`` themselves, exactly as ``scripts/`` does.
_PATH_SETUP = (
    "sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts');"
    "sys.path.insert(0, 'apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix');"
    "sys.path.insert(0, 'apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt');"
)


@pytest.mark.skipif(not KICKMIX_PYTHON, reason="set APEDP_Q_KICKMIX_PYTHON to a CPython 3.13 interpreter with kickmix")
def test_curve_checkpoint_values_tracks_w2() -> None:
    """The plan's target bits must follow ``cfg.w2``, not stay at curve4/5/6."""
    if not Path(KICKMIX_PYTHON).exists():
        pytest.skip(f"APEDP_Q_KICKMIX_PYTHON does not exist: {KICKMIX_PYTHON}")

    script = (
        "import sys;"
        + _PATH_SETUP
        + "import scheduled_arithmetic90 as H;"
        "import signed_digit_invsqrt_v3 as core;"
        "import test_shared_pddd_circuit as T;"
        "out = {};\n"
        "for f2 in (8, 9, 10):\n"
        "    cfg = H.Cfg(f2, 19, 27, 1, (5,));\n"
        "    tabs, _ = core.regenerate_tables_exact(cfg);\n"
        "    out[f2] = sorted(T.curve_checkpoint_values(cfg, tabs));\n"
        "print(out)\n"
    )
    result = subprocess.run(
        [KICKMIX_PYTHON, "-c", script],
        cwd=REPO_ROOT,
        env=_kickmix_env(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    plans = eval(result.stdout.strip().splitlines()[-1])
    assert plans[8] == ["curve4", "curve5", "curve6"], plans
    assert plans[9] == ["curve5", "curve6", "curve7"], plans
    assert plans[10] == ["curve6", "curve7", "curve8"], plans


@pytest.mark.skipif(not KICKMIX_PYTHON, reason="set APEDP_Q_KICKMIX_PYTHON to a CPython 3.13 interpreter with kickmix")
def test_w2_7_champion_still_rebuilds_byte_identical() -> None:
    """The refactor must be semantics-preserving: same bytes, not merely the same counts."""
    if not Path(KICKMIX_PYTHON).exists():
        pytest.skip(f"APEDP_Q_KICKMIX_PYTHON does not exist: {KICKMIX_PYTHON}")

    script = (
        "import sys, json, hashlib;" + _PATH_SETUP + "import scheduled_arithmetic90 as H;"
        "import signed_digit_invsqrt_v3 as core;"
        "import test_shared_pddd_circuit as T;"
        "cfg, tabs, _ = core.load_or_build_tables();"
        "cands = json.loads(open('src/shared_davio_results.json').read())['candidates'];\n"
        "plan = T.curve_checkpoint_values(cfg, tabs);\n"
        "sp = T.build_shared_plan(plan, order=cands[0]['order'], polarity=cands[0]['polarity']);\n"
        "c6 = json.loads(open('src/boolean_choices90.json').read())['coarse6'];\n"
        "raw, _ = T.build_candidate(cfg, tabs, coarse_choice=c6, shared_curve_plan=sp);\n"
        "opt, _ = T.domain_compiler.optimize_circuit(raw, linear=True);\n"
        "fixed = json.loads(open('src/invsqrt_calibration.json').read());\n"
        "t = T.cal._serialize(opt, 'kmx').decode(); lines, rots = T.cal.parse_kmx(t);\n"
        "values = fixed['deltasPhaseUnits']; dropped = set(fixed['droppedOriginalRotationIndices']);\n"
        "rep = T.uf.rewrite(lines, rots, {j: int(v) for j, v in enumerate(values) if j not in dropped}, dropped);\n"
        "cand = T.km.Circuit(rep); cand.to_file('/tmp/_w2_7_golden.kmb', format='kmb');\n"
        "print(hashlib.sha256(open('/tmp/_w2_7_golden.kmb','rb').read()).hexdigest())\n"
    )
    result = subprocess.run(
        [KICKMIX_PYTHON, "-c", script],
        cwd=REPO_ROOT,
        env=_kickmix_env(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().splitlines()[-1] == PINNED_CHAMPION_SHA256


@pytest.mark.skipif(not KICKMIX_PYTHON, reason="set APEDP_Q_KICKMIX_PYTHON to a CPython 3.13 interpreter with kickmix")
def test_w2_8_build_passes_the_official_verifier() -> None:
    """``w2 = 8`` must assemble, compile and verify — the failure the refactor removes."""
    if not Path(KICKMIX_PYTHON).exists():
        pytest.skip(f"APEDP_Q_KICKMIX_PYTHON does not exist: {KICKMIX_PYTHON}")

    script = (
        "import sys, json, hashlib;" + _PATH_SETUP + "import scheduled_arithmetic90 as H;"
        "import signed_digit_invsqrt_v3 as core;"
        "import test_shared_pddd_circuit as T;"
        "cfg = H.Cfg(9, 19, 27, 1, (5,));"
        "tabs, _ = core.regenerate_tables_exact(cfg);"
        "cands = json.loads(open('src/shared_davio_results.json').read())['candidates'];\n"
        "sp = T.build_shared_plan(T.curve_checkpoint_values(cfg, tabs), order=cands[0]['order'], polarity=cands[0]['polarity']);\n"
        "c6 = json.loads(open('src/boolean_choices90.json').read())['coarse6'];\n"
        "raw, _ = T.build_candidate(cfg, tabs, coarse_choice=c6, shared_curve_plan=sp);\n"
        "opt, _ = T.domain_compiler.optimize_circuit(raw, linear=True);\n"
        "fixed = json.loads(open('src/invsqrt_calibration.json').read());\n"
        "t = T.cal._serialize(opt, 'kmx').decode(); lines, rots = T.cal.parse_kmx(t);\n"
        "values = fixed['deltasPhaseUnits']; dropped = set(fixed['droppedOriginalRotationIndices']);\n"
        "assert len(rots) == len(values), 'pinned calibration only valid at the champion rotation count';\n"
        "rep = T.uf.rewrite(lines, rots, {j: int(v) for j, v in enumerate(values) if j not in dropped}, dropped);\n"
        "cand = T.km.Circuit(rep); cand.to_file('/tmp/_w2_8.kmb', format='kmb');\n"
        "base = T.ProblemSpecInvSqrt.for_n_p(10, shots_per_x=1);\n"
        "spec = T.ProblemSpecInvSqrt(width=base.width, max_x=25600, beta=base.beta,"
        " precision_bits=base.precision_bits, shots_per_x=1, system_qubits=base.system_qubits);\n"
        "r = T.verify(cand, spec);\n"
        "print(json.dumps({k: r.get(k) for k in ('status','num_qubits','max_toffoli_count','score','verified_shots')}))\n"
    )
    result = subprocess.run(
        [KICKMIX_PYTHON, "-c", script],
        cwd=REPO_ROOT,
        env=_kickmix_env(),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["status"] == "PASS", receipt
    assert receipt["num_qubits"] == PINNED_W2_8["num_qubits"]
    assert receipt["max_toffoli_count"] == PINNED_W2_8["max_toffoli_count"]
    assert receipt["verified_shots"] == PINNED_W2_8["verified_shots"]
    assert receipt["score"] == pytest.approx(PINNED_W2_8["score"], rel=1e-9)


def test_w2_8_candidate_is_measured_worse_than_the_champion() -> None:
    """The candidate that motivated the refactor loses, and cannot be rescued.

    ``w2 = 8`` buys a better *screen* margin (0.90 of tolerance against the champion's 1.00),
    but the build costs four more qubits and eight more Toffolis, and the verifier prices a
    rotation at ``0.56 log2(1/eps) + 5.3`` with ``eps = pi * min_margin / 2**64 / n_rot``. For
    the wider circuit to beat the champion it would have to remove ~337 T from its rotation
    synthesis, which needs ``min_margin`` to grow ~1.3e4x — about 4.9x the tolerance itself.
    A margin cannot exceed the tolerance, so no calibration can close the gap.
    """
    w2_8 = PINNED_W2_8
    assert w2_8["score"] > CHAMPION_SCORE, "the w2=8 candidate is no longer worse: re-measure it"

    def eps_from_t(t: float) -> float:
        return 2.0 ** (-((t - 5.3) / 0.56))

    t_now = 1260.0 / 44  # measured rotation-synthesis T per rotation
    needed_te = CHAMPION_SCORE / w2_8["num_qubits"]
    t_new = (1260.0 - 4 * (1778.8 - needed_te)) / 44
    growth = eps_from_t(t_new) / eps_from_t(t_now)
    champion_margin_fraction = 1 - 0.9996227434818049  # pinned worst error / tolerance
    assert champion_margin_fraction * growth > 1.0, (
        "the required margin now fits inside the tolerance: the candidate may be winnable "
        "after all — re-derive its calibration and measure it"
    )

"""Registered pins for RIME §16e (PREREG-RIME-SCOPE-MV-2026-10-17). audit-only.

Pins the NEGATIVE registered outcome recorded in
`benchmarks/scope_refusal_mv_validation_results.json` and
`docs/RIME_SCOPE_MV_REPORT.md`:

  * the two-sided criteria fail on the first valid run; the criteria flags and
    family-set outcome stay exactly as recorded (raw and final-set readings);
  * the exact in-family control recovery is wrongfully refused by the
    periodic-quadratic and amplitude-modulated probes on every seed — the
    small-frequency linearization mechanism (frequency at the ~0.1 refinement
    floor, dev fit at the noise floor) is pinned live;
  * every G2 guard cell is `undefined_no_probe_buildable` (the frozen variance
    guard against §16c axis scales), so "all G promoted" is unevaluable as
    preregistered;
  * `abs_linear2` is the only guard-clean family and the frozen family-set rule
    has no retention assignment covering W1/W3.

A failure here means the registered outcome moved; it must be re-reported, and
any design change requires a v2 prereg — never patched (§16e deviation rule).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from apedp_dyn.epistemic.scope_refusal_mv import (
    PROBE_MV_IDS,
    _select_am_product,
    _select_periodic_quadratic,
    build_family_probes_mv,
)
from benchmarks.known_truth_challenge import build_worlds
from benchmarks.run_scope_refusal_mv_validation import (
    FEATS,
    NOISE,
    aligned_frame,
    emit_list_top,
    law_control,
    referee,
)

_JSON = Path(_REPO) / "benchmarks" / "scope_refusal_mv_validation_results.json"


@pytest.fixture(scope="module")
def results():
    return json.loads(_JSON.read_text(encoding="utf-8"))


def _cell(results, name, seed):
    for r in results["cells"]:
        if r["cell"] == name and r["seed"] == seed:
            return r
    raise KeyError((name, seed))


def _decliners(rec):
    return {p["probe"] for p in (rec["adjudication"] or {}).get("probes", [])
            if p["verdict"] == "declined"}


def test_registered_run_failed_and_flags_frozen(results) -> None:
    assert results["prereg_id"] == "PREREG-RIME-SCOPE-MV-2026-10-17"
    assert results["pass"] is False
    assert results["errors"] == []
    assert len(results["cells"]) == 27
    assert results["criteria"] == {
        "w_all_declined_final_set": False,
        "w_all_declined_raw": True,
        "g1_promoted_final_set": True,
        "g1_promoted_raw": False,
        "g2_monomials_promoted_final_set": False,
        "g2_monomials_promoted_raw": False,
        "g2_cells_defined": "0/12",
        "g3_no_promotion": True,
    }
    assert results["determinism_ok"] is True


def test_control_refused_on_every_seed_linearization_pin(results) -> None:
    for seed in (700, 701, 702):
        rec = _cell(results, "control", seed)
        assert rec["adjudication"]["scope_verdict"] == "declined", seed
        decl = _decliners(rec)
        assert "probe_mv_periodic_quadratic" in decl and "probe_mv_am_product" in decl, seed
    # mechanism pin (seed 700): both refusals are observational
    # indistinguishability against an exact control recovery
    rec = _cell(results, "control", 700)
    status = {p["probe"]: p["status"] for p in rec["adjudication"]["probes"]}
    assert status["probe_mv_periodic_quadratic"] == "multi_hypothesis"
    assert status["probe_mv_am_product"] == "multi_hypothesis"
    assert status["probe_mv_abs_linear2"] == "unique_support"
    assert "2.000092 * (x1_l1 * x2_l1)" in rec["formula"]


def test_w_declined_raw_and_family_set_collapse(results) -> None:
    for seed in (700, 701, 702):
        for name in ("w1_sin_interaction", "w2_max_switch", "w3_am_product"):
            rec = _cell(results, name, seed)
            assert rec["adjudication"]["scope_verdict"] == "declined", (name, seed)
    fs = results["family_set"]
    assert fs["final"] == ["probe_mv_abs_linear2"]
    assert sorted(fs["excluded"]) == sorted(
        [PROBE_MV_IDS[0], PROBE_MV_IDS[2]])
    for fam, info in fs["excluded"].items():
        assert info["guard_failures"] == ["control"] * 3, fam
    # final-set reading: W2 covered by abs; W1/W3 lose their only decliners
    for key, ok in results["final_w_declined"].items():
        if key.startswith("w2_"):
            assert ok is True, key
        else:
            assert ok is False, key


def test_g2_guards_undefined(results) -> None:
    g2 = [r for r in results["cells"] if r["kind"] == "g2_mv_p1"]
    assert len(g2) == 12
    for rec in g2:
        assert rec["status"] == "undefined_no_probe_buildable", rec["cell"]
        assert rec["adjudication"] is None and rec["probe_families"] == []


def test_g3_abstained(results) -> None:
    g3 = [r for r in results["cells"] if r["kind"] == "g3"]
    assert len(g3) == 3
    for rec in g3:
        assert rec["classification"] == "ABSTAINED" and rec["n_props"] == 0


def test_live_linearization_floor_on_control() -> None:
    dev, hold = aligned_frame(700, law_control)
    Xd = dev[FEATS].to_numpy(dtype=float)
    y = dev["target"].to_numpy(dtype=float)
    _mse, key, _w = _select_periodic_quadratic(Xd, y)
    assert key[0] == "pq" and key[1:3] == (0, 1)
    assert key[3] <= 0.1001, key          # frequency pinned at the refinement floor
    assert _mse < 0.003, _mse             # dev fit at the declared noise floor
    _mse_am, key_am, _w_am = _select_am_product(Xd, y)
    assert key_am[0] == "am" and key_am[3] <= 0.1001, key_am


def test_live_probe_build_contracts() -> None:
    dev, hold = aligned_frame(700, law_control)
    probes = build_family_probes_mv(dev, hold, feature_cols=FEATS, noise_sigma=NOISE)
    assert [p.candidate.candidate_id for p in probes] == list(PROBE_MV_IDS)

    r = np.random.RandomState(3)
    n = 80
    p = np.linspace(1.0, 30.0, n)
    f1 = pd.DataFrame({"p": p, "target": p ** 2 + 0.05 * r.randn(n)})
    probes1 = build_family_probes_mv(f1.iloc[:50].reset_index(drop=True),
                                     f1.iloc[50:].reset_index(drop=True),
                                     feature_cols=["p"])
    assert [q.candidate.candidate_id for q in probes1] == [PROBE_MV_IDS[0]]

    worlds = {bw.world.name: bw for bw in build_worlds(noise_seed=700)}
    bw = worlds["d2_C1000_uniform_1x"]
    with pytest.raises(ValueError, match="no multivariate probe family"):
        build_family_probes_mv(bw.dev, bw.hold, feature_cols=["p"],
                               noise_sigma=bw.sigma)


def test_live_control_adjudication_declined() -> None:
    dev, hold = aligned_frame(700, law_control)
    emitted, _ms, info = emit_list_top(dev, hold)
    assert emitted is not None and info["hold_r2"] >= 0.99
    probes = build_family_probes_mv(dev, hold, feature_cols=FEATS, noise_sigma=NOISE)
    res = referee(dev, hold, emitted, probes, NOISE)
    assert res["scope_verdict"] == "declined"
    assert "probe_mv_periodic_quadratic" in _decliners({"adjudication": res})

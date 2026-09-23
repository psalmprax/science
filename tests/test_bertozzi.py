"""Bertozzi 1964: classical refuted 5/5, relativistic matches, regime tag works."""
import json
import math

PATH = "data/bertozzi_1964.jsonl"


def _data():
    with open(PATH) as fh:
        return json.loads(fh.read())


def test_classical_superluminal_everywhere():
    d = _data()
    m = d["electron_rest_MeV"]
    for run in d["runs"]:
        v_cl = math.sqrt(2 * run["KE_MeV"] / m)
        assert v_cl > 1.0, (run, v_cl)


def test_relativistic_matches():
    d = _data()
    m = d["electron_rest_MeV"]
    for run in d["runs"]:
        gamma = 1 + run["KE_MeV"] / m
        v_rel = math.sqrt(max(0.0, 1 - 1 / gamma ** 2))
        assert abs(v_rel - run["v_over_c"]) < 0.035, (run, v_rel)


def test_regime_tag_saturates():
    d = _data()
    betas = [r["v_over_c"] ** 2 / 2 for r in d["runs"]]
    assert max(betas) <= 0.5 + 1e-9, betas
    gammas = [r["KE_MeV"] / d["electron_rest_MeV"] for r in d["runs"]]
    assert gammas[-1] / max(betas) > 50.0, (gammas, betas)


def test_full_loop_classical_vs_relativistic():
    """End-to-end on real data: fit both forms, bakeoff, report."""
    import numpy as np

    from apedp.scientific_layers import bakeoff, candidate_report
    d = _data()
    m = d["electron_rest_MeV"]
    ke = np.array([r["KE_MeV"] for r in d["runs"]])
    v = np.array([r["v_over_c"] for r in d["runs"]])
    xtr = {"ke": ke[:3]}
    ytr = v[:3]
    xte = {"ke": ke[3:]}
    yte = v[3:]
    def classical(dd):
        return np.sqrt(2 * np.asarray(dd["ke"]) / m)

    def relativ(dd):
        return np.sqrt(np.maximum(
            0.0, 1 - 1 / (1 + np.asarray(dd["ke"]) / m) ** 2))
    out = bakeoff(xtr, ytr, xte, yte, classical,
                  {"relativistic": lambda Xt, Xte: (
                      relativ(Xte),
                      {"form": "sqrt(1-1/(1+K/m)^2)"})})
    rows = {r["form"]: r for r in out["rows"]}
    assert rows["relativistic"]["mae"] < rows["candidate"]["mae"], out
    rep = candidate_report(
        "classical-v", "sqrt(2K/m)", "Bertozzi-1964",
        {"KE_MeV": (0.5, 15.0)}, len(yte),
        rows["candidate"]["mae"], rows["relativistic"]["mae"], "PASS",
        {"status": "FAIL", "reason": "predicts v>c on all runs"},
        "real-measurement", "none", "not established")
    assert "PHYSICAL DOMAIN: FAIL" in rep

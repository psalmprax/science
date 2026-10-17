"""Canonical ledger pins: accounting identity, determinism, validity masks.

All metrics aggregate from one per-observation ledger; no competing harness.
"""


from apedp_dyn.sandbox.canonical_ledger import ledger_hash, run_ledger
from apedp_dyn.sandbox.multi_asset_signals_sandbox import load_universe

W4 = {"H-BB": 0.811068, "H-OBV_MOM": -1.427209,
      "NL_TAKER_MOM": 0.598382, "NL_MASTER_SNR": 0.673950}


def _mini_universe():
    uni, syms = load_universe("30")
    syms = syms[:6]
    data = {s: uni[s].iloc[-120:].reset_index(drop=True) for s in syms}
    return syms, data


def test_identity_gross_minus_fees_equals_net():
    syms, data = _mini_universe()
    _, s = run_ledger(syms, data, W4, "base", 1.0, 0.5, slice(60, 119))
    assert abs((s["gross"] - s["fees"]) - s["net"]) < 1e-9


def test_deterministic_replay_identical_hash():
    syms, data = _mini_universe()
    r1, _ = run_ledger(syms, data, W4, "hyst", 1.0, 0.5, slice(60, 119))
    r2, _ = run_ledger(syms, data, W4, "hyst", 1.0, 0.5, slice(60, 119))
    assert ledger_hash(r1) == ledger_hash(r2)


def test_invalid_scores_force_flat_with_reason():
    syms, data = _mini_universe()
    rows, s = run_ledger(syms, data, W4, "base", 1.0, 0.5, slice(0, 119))
    inv = [r for r in rows if not r["valid"]]
    assert len(inv) > 0  # warmup windows must exist
    assert all(r["pos"] == 0.0 and r["reason"] == "invalid_score" for r in inv)
    assert s["n_invalid"] == len(inv)


def test_hysteresis_never_exceeds_baseline_turnover():
    syms, data = _mini_universe()
    _, sb = run_ledger(syms, data, W4, "base", 1.0, 0.5, slice(60, 119))
    _, sh = run_ledger(syms, data, W4, "hyst", 1.0, 0.5, slice(60, 119))
    assert sh["turnover"] <= sb["turnover"]

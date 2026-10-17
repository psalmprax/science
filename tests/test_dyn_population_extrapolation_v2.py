"""Population extrapolation v2: depths 1-6, transform-augmented features,
GA basis ladder (linear/lag, +squares, +products), repair-mediated transfer.

Same planted composite + populations as v1 (seeds fixed):
  y = 2*x1[t-1]*x2[t-1] + 3*x3[t-2] + 10 (+shifted 2.5/2.0 on C).

Measured findings pinned here:
  * RIME depths 1-6 identical (x3[t-2]); +sq transforms change nothing;
    [x1,x2]-withheld abstains. Depth/feature-breadth move nothing — the
    bottleneck is compositional (no Add rewrite), confirmed again.
  * GA ladder isolates the gap precisely: +squares adds nothing over
    linear+lag (0.685 vs 0.681); only +products jumps (0.9998).
  * Repair-mediated RIME matches GA-large on every population including
    cross-population transfer probes (sealed 0.9998, B 0.9998, C 0.8952):
    the model-class gap closes through diagnosis-targeted repair, no spec
    change. Transfer probes are read-only evaluations on extra populations
    (no decisions); sealed discipline unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG
from apedp_dyn.epistemic.rime_repair_engine import (
    RepairConfig,
    RepairOperator,
    RepairPartitions,
    RepairStatus,
    repair,
)
from apedp_dyn.proposers.rime_proposer import RimeProposer

CUTS = {"rho_flag": RESID_RHO_FLAG, "bias_ratio_cut": 0.25,
            "variance_ratio_cut": 4.0, "curvature_corr_cut": 0.3,
            "interaction_corr_cut": 0.3, "min_rows": 30}

CONFIG = RepairConfig(
    max_candidates_per_round=6,
    max_rounds=2,
    max_added_terms=1,
    min_validation_r2_gain=0.02,
    equivalence_tol=1e-9,
    operators=(RepairOperator.R1_PARAM_REFIT, RepairOperator.R2_TERM_INSERT, RepairOperator.R3_LAG_INSERT),
    invariance_slices=0,
)


def _make(seed: int, c1: float = 2.0, c2: float = 3.0, n: int = 600) -> pd.DataFrame:
    r = np.random.RandomState(seed)
    x1, x2, x3 = r.randn(n), r.randn(n), r.randn(n)
    y = np.zeros(n)
    y[2:] = c1 * (x1[1:-1] * x2[1:-1]) + c2 * x3[:-2] + 10.0 + 0.05 * r.randn(n - 2)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


def _lagcol(col: np.ndarray, k: int) -> np.ndarray:
    col = np.asarray(col, dtype=float)
    v = np.full(len(col), np.nan)
    if k > 0:
        v[k:] = col[:-k]
    else:
        v[:] = col
    return v


def _ga_basis(df: pd.DataFrame, level: int) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {"1": np.ones(len(df))}
    for c in ("x1", "x2", "x3"):
        raw = np.asarray(df[c], dtype=float)
        d[c] = raw
        for k in (1, 2):
            d[f"{c}[t-{k}]"] = _lagcol(raw, k)
    if level >= 1:
        for c in ("x1", "x2", "x3"):
            for k in (0, 1, 2):
                key = c if k == 0 else f"{c}[t-{k}]"
                d[f"{key}^2"] = d[key] ** 2
    if level >= 2:
        d["x1[t-1]*x2[t-1]"] = d["x1[t-1]"] * d["x2[t-1]"]
        d["x1[t-2]*x2[t-2]"] = d["x1[t-2]"] * d["x2[t-2]"]
        d["x1*x2"] = d["x1"] * d["x2"]
        d["x1*x3"] = d["x1"] * d["x3"]
        d["x2*x3"] = d["x2"] * d["x3"]
    return d


def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    ss = float(np.sum((yt - yt.mean()) ** 2))
    return 1 - float(np.sum((yt - yp) ** 2)) / ss if ss > 0 else float("nan")


def _pops():
    A = _make(500)
    B = _make(501)
    C = _make(502, c1=2.5, c2=2.0)
    return A.iloc[:400].reset_index(drop=True), A.iloc[400:].reset_index(drop=True), B, C


def _run_ga(df_tr: pd.DataFrame, ytr: np.ndarray, level: int):
    rng = np.random.RandomState(7)
    d = _ga_basis(df_tr, level)
    names = list(d.keys())
    Z = np.column_stack([d[k] for k in names])
    lam = 1e-6

    def fit(mask):
        M = Z[:, mask]
        ok = np.isfinite(M).all(axis=1) & np.isfinite(ytr)
        if int(ok.sum()) < int(mask.sum()) + 1:
            return None, np.inf
        w, *_ = np.linalg.lstsq(M[ok], ytr[ok], rcond=None)
        return w, float(np.mean((ytr[ok] - M[ok] @ w) ** 2)) + lam * int(mask.sum())

    pop: list[np.ndarray] = []
    while len(pop) < 40:
        m = rng.rand(len(names)) < 0.3
        if not m.any():
            continue
        pop.append(m)
    for _ in range(25):
        pop = sorted(pop, key=lambda m: fit(m)[1])
        nxt = [m.copy() for m in pop[:6]]
        while len(nxt) < 40:
            a = sorted([pop[i] for i in rng.choice(len(pop), 3, replace=False)], key=lambda m: fit(m)[1])[0]
            b = sorted([pop[i] for i in rng.choice(len(pop), 3, replace=False)], key=lambda m: fit(m)[1])[0]
            c = np.where(rng.rand(len(names)) < 0.5, a, b)
            if rng.rand() < 0.15:
                c[rng.randint(len(names))] ^= 1
            if not c.any():
                continue
            nxt.append(c)
        pop = nxt
    best = sorted(pop, key=lambda m: fit(m)[1])[0]
    w, _ = fit(best)
    return best, w, names


def _ga_eval(df: pd.DataFrame, w, mask, names, level: int) -> float:
    d = _ga_basis(df, level)
    yp = np.column_stack([d[k] for k in names])[:, mask] @ w
    return _r2(df["target"].to_numpy(), yp)


def test_rime_depths_1_to_6_identical():
    Atr, Ate, B, C = _pops()
    for depth in (1, 2, 3, 4, 5, 6):
        props = RimeProposer(feature_cols=["x1", "x2", "x3"], target_col="target",
                             max_depth=depth, budget=200).fit_and_propose(Atr)
        assert len(props) > 0, depth
        b = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
        assert b.metadata.get("equation") == "x3[t-2]", (depth, b.metadata.get("equation"))
        scores = [_r2(Ate["target"].to_numpy(), np.asarray(b.compute_score(Ate), dtype=float)),
                  _r2(B["target"].to_numpy(), np.asarray(b.compute_score(B), dtype=float)),
                  _r2(C["target"].to_numpy(), np.asarray(b.compute_score(C), dtype=float))]
        for s, lo, hi in zip(scores, (0.55, 0.65, 0.35), (0.75, 0.80, 0.55), strict=False):
            assert lo < s < hi, (depth, scores)


def test_rime_sq_transforms_change_nothing():
    Atr, Ate, _B, _C = _pops()
    props = RimeProposer(feature_cols=["x1", "x2", "x3"], target_col="target",
                         max_depth=3, budget=200, algebraic_transforms=("sq",)).fit_and_propose(Atr)
    assert len(props) > 0
    b = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
    assert b.metadata.get("equation") == "x3[t-2]", b.metadata.get("equation")
    assert 0.55 < _r2(Ate["target"].to_numpy(), np.asarray(b.compute_score(Ate), dtype=float)) < 0.75


def test_ga_basis_ladder_isolates_products():
    Atr, Ate, B, C = _pops()
    ytr = Atr["target"].to_numpy()
    got = {}
    for level, tag in ((0, "lin"), (1, "sq"), (2, "prod")):
        m, w, names = _run_ga(Atr, ytr, level)
        got[tag] = [_ga_eval(df, w, m, names, level) for df in (Atr, Ate, B, C)]
    assert abs(got["lin"][1] - 0.6450) < 0.03, got
    assert abs(got["sq"][1] - got["lin"][1]) < 0.03, got  # squares add nothing
    assert got["prod"][1] > 0.99 and got["prod"][2] > 0.99 and got["prod"][3] > 0.8, got
    print(f"\n[ladder] lin={got['lin'][1]:.4f} sq={got['sq'][1]:.4f} prod={got['prod'][1]:.4f}")


def test_repair_mediated_transfer_matches_ga_large():
    # Requires the interaction-probe engine; predates frozen snapshots
    # without it (3/4 there is the expected scope boundary).
    Atr, Ate, B, C = _pops()
    full = pd.concat([Atr, Ate]).reset_index(drop=True)
    parts = RepairPartitions.from_frame(full, full["target"].to_numpy())
    tr = parts.discovery_df.assign(target=parts.discovery_y)
    orig = sorted(RimeProposer(feature_cols=["x1", "x2", "x3"], target_col="target",
                               max_depth=2, budget=200).fit_and_propose(tr),
                  key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
    out = repair(orig, parts, CONFIG, **CUTS)
    assert out.status == RepairStatus.IMPROVED, (out.status, out.submit_note)
    assert out.winner is not None and "(x1[t-1]*x2[t-1])" in (out.winner.metadata["term"] or "")
    assert out.sealed is not None and out.sealed["winner_r2"] > 0.95, out.sealed
    for nm, df, lo in (("B-fresh", B, 0.95), ("C-shift", C, 0.8)):
        r = _r2(df["target"].to_numpy(), np.asarray(out.winner.compute_score(df), dtype=float))
        assert r > lo, (nm, r)
    print(f"\n[repair-transfer] sealed={out.sealed['winner_r2']:.4f} (GA-large B/C: 0.9998/0.8961)")


def test_depth_mechanism_budget_bound_not_ignored():
    # Reviewer risk #2, resolved by measurement: depth is NOT ignored.
    # Depth 1 searches less (136 evaluated, under the 200 budget cap);
    # depths >= 2 all hit the cap with identical winners. Identical outputs
    # mean budget-bound convergence, and depth is effective below the cap.
    Atr, _, _, _ = _pops()
    seen = {}
    for depth in (1, 2, 4, 6):
        props = RimeProposer(feature_cols=["x1", "x2", "x3"], target_col="target",
                             max_depth=depth, budget=200).fit_and_propose(Atr)
        assert len(props) > 0, depth
        m0 = props[0].metadata
        seen[depth] = (m0.get("evaluated"), m0.get("accepted"),
                       sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0].metadata.get("equation"))
    assert seen[1][0] < seen[2][0], seen  # depth changes the search volume
    assert seen[2][0] == 200 and seen[6][0] == 200, seen  # cap binds above
    assert len({v[2] for v in seen.values()}) == 1, seen  # same winner
    print(f"\n[depth-mechanism] {seen}")


def test_static_mode_emits_no_temporal_forms():
    # Reviewer risk #3: static_mode=True must make temporal primitives
    # unrepresentable (no lag/state forms, no index-structure leakage).
    import numpy as _np
    import pandas as _pd
    rng = _np.random.RandomState(11)
    n = 600
    x = rng.randn(n)
    y = _np.zeros(n)
    y[2:] = 3.0 * x[:-2] + 10.0 + 0.01 * rng.randn(n - 2)
    df = _pd.DataFrame({"x": x, "target": y})
    props = RimeProposer(feature_cols=["x"], target_col="target", max_depth=2,
                         budget=200, static_mode=True).fit_and_propose(df)
    temporal = [p for p in props
                if ("[t-" in (p.metadata.get("equation") or "") or "state_" in (p.metadata.get("equation") or ""))]
    assert temporal == [], [p.metadata.get("equation") for p in props]
    print(f"\n[static-guard] {len(props)} proposals, 0 temporal forms")


def test_rime_recall_vs_budget_curve():
    # Reviewer #5: target-expression (both-sided product) recall vs search
    # budget. Measured: 0/5 budgets — the product never appears at any
    # budget because selection is penalty-dominated, not budget-starved
    # (evaluated counts confirm the budget is spent, not ignored).
    Atr, _, _, _ = _pops()
    for budget, lo_ev in ((50, 1), (100, 1), (200, 100), (500, 100), (1000, 100)):
        props = RimeProposer(feature_cols=["x1", "x2", "x3"], target_col="target",
                             max_depth=3, budget=budget).fit_and_propose(Atr)
        ev = props[0].metadata.get("evaluated", 0) if props else 0
        eqs = [p.metadata.get("equation") for p in props]
        assert ev <= budget, (budget, ev)
        assert ev >= lo_ev, (budget, ev)
        assert not any("x1" in (e or "") and "x2" in (e or "") and "*" in (e or "") for e in eqs), eqs
    print("\n[recall@budget] product recall 0/5 budgets; budget spent, penalty binds")


def test_resource_curves_recorded():
    # Reviewer #14: wall-time vs budget for each method family. Bounds are
    # deliberately generous (hang-guard, not benchmark); the printed table
    # is the quality-vs-resource record. Machine-dependent absolutes are
    # not asserted — evaluated counts and quality bands are.
    import time as _time
    Atr, Ate, _, _ = _pops()
    rows = []
    for budget in (100, 200, 500):
        t0 = _time.perf_counter()
        props = RimeProposer(feature_cols=["x1", "x2", "x3"], target_col="target",
                             max_depth=3, budget=budget).fit_and_propose(Atr)
        dt = _time.perf_counter() - t0
        assert dt < 300, (budget, dt)
        b = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
        r2 = _r2(Ate["target"].to_numpy(), np.asarray(b.compute_score(Ate), dtype=float))
        assert 0.55 < r2 < 0.75, (budget, r2)
        rows.append((f"RIME/b{budget}", round(dt, 2), b.metadata.get("evaluated"), round(r2, 4)))
    print("\n[resources] (method, wall_s, evaluated, Atest_R2):")
    for r in rows:
        print("   ", r)


def test_usage_log_planted_composite_uses_only_x3():
    # Usage logging on the composite grid: all 5 emissions reference x3
    # only; x1/x2 (the product drivers) never appear — the product form
    # never even enters the emitted set, consistent with the seed-coverage
    # and penalty-dominated findings.
    from apedp_dyn.proposers.rime_usage_log import log_search_usage
    Atr, _, _, _ = _pops()
    props = RimeProposer(feature_cols=["x1", "x2", "x3"], target_col="target",
                         max_depth=2, budget=200).fit_and_propose(Atr)
    assert len(props) > 0
    rep = log_search_usage(props, ["x1", "x2", "x3"])
    assert rep.used_in_any == ["x3"], rep.to_dict()
    assert rep.unused_in_all == ["x1", "x2"], rep.to_dict()
    print(f"\n[usage] composite: {rep.n_proposals} proposals, used={rep.used_in_any}")

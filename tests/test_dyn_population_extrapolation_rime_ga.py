"""Population extrapolation: RIME (depth x feature-space) vs GA (basis size).

Planted composite truth (seeds fixed):
  y[t] = 2*x1[t-1]*x2[t-1] + 3*x3[t-2] + 10 + noise
Populations: A-train (discovery, rows 0:400), A-test (400:600, same
mechanism), B (fresh realization, same mechanism), C (shifted coefficients
2.5/2.0 — graceful transfer, not identity).

Measured findings pinned here:
  * RIME [x1,x2] (x3 withheld): abstains at ALL depths 1-4 — the
    both-sided product is unreachable within budget (seed-coverage result).
  * RIME [x1,x2,x3]: x3[t-2] linear part at every depth 1-4, identical
    metrics (depth changes nothing here).
  * GA-small (linear+lag basis): matches RIME nearly exactly — with
    comparable feature spaces the methods tie.
  * GA-large (basis hand-given the planted interaction): near-perfect
    recovery and transfer. The gap opens only where GA is handed terms
    RIME's fixed grammar cannot reach — a representability gap, not a
    search gap.
  * C-shift: all approaches degrade gracefully (R2 stays positive).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.proposers.rime_proposer import RimeProposer


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


def _ga_basis(df: pd.DataFrame, large: bool) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {"1": np.ones(len(df))}
    for c in ("x1", "x2", "x3"):
        raw = np.asarray(df[c], dtype=float)
        d[c] = raw
        for k in (1, 2):
            d[f"{c}[t-{k}]"] = _lagcol(raw, k)
    if large:
        for c in ("x1", "x2", "x3"):
            for k in (0, 1, 2):
                key = c if k == 0 else f"{c}[t-{k}]"
                d[f"{key}^2"] = d[key] ** 2
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


def _run_ga(df_tr: pd.DataFrame, ytr: np.ndarray, large: bool):
    rng = np.random.RandomState(7)
    d = _ga_basis(df_tr, large)
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


def _ga_eval(df: pd.DataFrame, w, mask, names) -> float:
    d = _ga_basis(df, "x1[t-1]*x2[t-1]" in names)
    yp = np.column_stack([d[k] for k in names])[:, mask] @ w
    return _r2(df["target"].to_numpy(), yp)


def _populations():
    A = _make(500)
    B = _make(501)
    C = _make(502, c1=2.5, c2=2.0)
    return A.iloc[:400].reset_index(drop=True), A.iloc[400:].reset_index(drop=True), B, C


def test_rime_depth_sweep_identical_and_partial():
    Atr, Ate, B, C = _populations()
    for depth in (1, 2, 3, 4):
        props = RimeProposer(feature_cols=["x1", "x2", "x3"], target_col="target",
                             max_depth=depth, budget=200).fit_and_propose(Atr)
        assert len(props) > 0, depth
        b = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
        assert b.metadata.get("equation") == "x3[t-2]", (depth, b.metadata.get("equation"))
        scores = [_r2(Ate["target"].to_numpy(), np.asarray(b.compute_score(Ate), dtype=float)),
                  _r2(B["target"].to_numpy(), np.asarray(b.compute_score(B), dtype=float)),
                  _r2(C["target"].to_numpy(), np.asarray(b.compute_score(C), dtype=float))]
        for s, lo, hi in zip(scores, (0.55, 0.65, 0.35), (0.75, 0.80, 0.55)):
            assert lo < s < hi, (depth, scores)
    print("\n[rime-depth] depths 1-4 identical: x3[t-2] partial recovery; C-shift graceful")


def test_rime_withheld_feature_abstains_all_depths():
    Atr, _, _, _ = _populations()
    for depth in (1, 2, 3, 4):
        props = RimeProposer(feature_cols=["x1", "x2"], target_col="target",
                             max_depth=depth, budget=200).fit_and_propose(Atr)
        assert len(props) == 0, (depth, [p.formula for p in props])


def test_ga_basis_size_and_rime_parity():
    Atr, Ate, B, C = _populations()
    ytr = Atr["target"].to_numpy()
    # small basis: ties RIME's linear-part recovery on every population
    m_s, w_s, n_s = _run_ga(Atr, ytr, False)
    rime_props = RimeProposer(feature_cols=["x1", "x2", "x3"], target_col="target",
                              max_depth=2, budget=200).fit_and_propose(Atr)
    assert len(rime_props) > 0
    rime_best = sorted(rime_props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
    r_at = _r2(Ate["target"].to_numpy(), np.asarray(rime_best.compute_score(Ate), dtype=float))
    g_at = _ga_eval(Ate, w_s, m_s, n_s)
    assert abs(g_at - 0.6450) < 0.03, g_at
    assert abs(g_at - r_at) < 0.05, (g_at, r_at)  # parity with matched spaces
    # large basis (hand-given the interaction): near-perfect + transfer
    m_l, w_l, n_l = _run_ga(Atr, ytr, True)
    r_b = _ga_eval(B, w_l, m_l, n_l)
    r_c = _ga_eval(C, w_l, m_l, n_l)
    assert r_b > 0.99, r_b
    assert r_c > 0.8, r_c
    print(f"\n[parity] GA-small Atest={g_at:.4f} ~= RIME {r_at:.4f}; GA-large B={r_b:.4f} C={r_c:.4f}")


def _bigframe(df: pd.DataFrame) -> pd.DataFrame:
    def lagcol(col, k):
        col = np.asarray(col, dtype=float)
        v = np.full(len(col), np.nan)
        if k > 0:
            v[k:] = col[:-k]
        else:
            v[:] = col
        return v
    d = {"x1": np.asarray(df["x1"]), "x2": np.asarray(df["x2"]), "x3": np.asarray(df["x3"])}
    for c in ("x1", "x2", "x3"):
        for k in (1, 2):
            d[f"{c}L{k}"] = lagcol(d[c], k)
    for c in ("x1", "x2", "x3"):
        for k in (0, 1, 2):
            key = c if k == 0 else f"{c}L{k}"
            d[f"{key}sq"] = d[key] ** 2
    d["x1L1_x2L1"] = d["x1L1"] * d["x2L1"]
    d["x1L2_x2L2"] = d["x1L2"] * d["x2L2"]
    d["x1x2"] = d["x1"] * d["x2"]
    d["x1x3"] = d["x1"] * d["x3"]
    d["x2x3"] = d["x2"] * d["x3"]
    out = pd.DataFrame(d)
    out["target"] = np.asarray(df["target"])
    return out


def test_rime_large_matches_rime_raw_not_ga_large():
    # RIME handed the SAME 23 engineered columns GA-large had. Measured:
    # identical linear-part recovery (x3L2) at every depth/budget — the
    # precomputed product column scores only R2~0.31 alone, and the
    # combination is unreachable: 23 levels + partial delays exhaust the
    # 60-form seed budget before cross-input sums, and no frozen rewrite
    # introduces Add. GA-large wins via mask+OLS additive fitting — a model
    # class gap (additive regression vs single-program search), not effort.
    Atr, Ate, B, C = _populations()
    AtrB, AteB = _bigframe(Atr), _bigframe(Ate)
    feats = [c for c in AtrB.columns if c != "target"]
    assert len(feats) == 23, len(feats)
    for depth, budget in ((2, 200), (3, 1000)):
        props = RimeProposer(feature_cols=feats, target_col="target",
                             max_depth=depth, budget=budget).fit_and_propose(AtrB)
        assert len(props) > 0
        b = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
        assert b.metadata.get("equation") == "x3L2", (depth, budget, b.metadata.get("equation"))
        scores = [_r2(AteB["target"].to_numpy(), np.asarray(b.compute_score(AteB), dtype=float)),
                  _r2(B["target"].to_numpy(), np.asarray(b.compute_score(_bigframe(B)), dtype=float)),
                  _r2(C["target"].to_numpy(), np.asarray(b.compute_score(_bigframe(C)), dtype=float))]
        for s, lo, hi in zip(scores, (0.55, 0.65, 0.35), (0.75, 0.80, 0.55)):
            assert lo < s < hi, (depth, budget, scores)
    print("\n[rime-large] 23 engineered inputs, still x3L2 partial: model-class gap, not feature count")

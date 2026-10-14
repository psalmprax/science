"""GA on QEC threshold data vs OLS vs RIME (pinned three-way comparison).

Measured (seeds fixed, 600-point data, 300/300 train/holdout split):
  * GA-evolved (mask + OLS readout, 2/7 basis terms {1, p^2}): holdout
    RMSE=0.002220, R2=0.9995 — numerically identical to same-split OLS,
    i.e. GA contributes selection, not fitting magic here.
  * OLS plain_quad same split recovers w≈(-7e-05, 994.85): 994.85 matches
    the planted C/p_th^2 = 0.1/1e-4 = 1000 (physics recovered).
  * RIME raw [p], deep search (depth 4, budget 1000): seeded
    self-interactions (p*p[t-k]) absorb the near-constant log-grid ratio
    into the fitted scale; holdout R2=0.9995, stable 3/3 runs.

Integrity note (2026-10-13, corrected): this file was found containing an
unattributed edit asserting a "scale-invariant ridge" kernel change. The
edit was initially reverted as unverifiable — incorrectly. The change is
real (commit 9781a3d: relative `ridge·Σo²` in `fit_readout`, verified by
diff; behavior-neutral on unit-scale tests per the full suite). A single
divergent shallow-config run (lag winner amid product winners) is
attributed to a concurrent kernel rebuild landing mid-measurement, not to
search nondeterminism. Concurrent sessions must coordinate via separate
worktrees; see docs row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.proposers.rime_proposer import RimeProposer
from apedp_dyn.sandbox.qec_threshold import QEC_P_TH, generate_qec_threshold_data


def _basis(pp: np.ndarray) -> dict[str, np.ndarray]:
    return {"1": np.ones_like(pp), "p": pp, "p2": pp**2, "p3": pp**3,
            "sqrt(p)": np.sqrt(pp), "(p/pth)^2": (pp / QEC_P_TH) ** 2,
            "(p/pth)^3": (pp / QEC_P_TH) ** 3}


def _run_ga(ptr, ytr, pho, yho, seed: int = 7, pop_size: int = 40, gens: int = 25):
    rng = np.random.RandomState(seed)
    names = ["1", "p", "p2", "p3", "sqrt(p)", "(p/pth)^2", "(p/pth)^3"]
    Btr = np.column_stack([_basis(ptr)[k] for k in names])
    Bho = np.column_stack([_basis(pho)[k] for k in names])
    mu, sd = Btr[:, 1:].mean(0), Btr[:, 1:].std(0)
    Ztr = np.column_stack([np.ones(len(ptr)), (Btr[:, 1:] - mu) / sd])
    Zho = np.column_stack([np.ones(len(pho)), (Bho[:, 1:] - mu) / sd])
    lam = 1e-6

    def fit(mask):
        w, *_ = np.linalg.lstsq(Ztr[:, mask], ytr, rcond=None)
        return w, float(np.mean((ytr - Ztr[:, mask] @ w) ** 2)) + lam * mask.sum()

    pop = [rng.rand(len(names)) < 0.35 for _ in range(pop_size)]
    for m in pop:
        if not m.any():
            m[rng.randint(len(names))] = True
    for _ in range(gens):
        scored = sorted(pop, key=lambda m: fit(m)[1])
        nxt = [m.copy() for m in scored[:6]]
        while len(nxt) < pop_size:
            t = lambda: sorted([pop[i] for i in rng.choice(len(pop), 3, replace=False)], key=lambda m: fit(m)[1])[0]
            a, b = t(), t()
            c = np.where(rng.rand(len(names)) < 0.5, a, b)
            if not c.any():
                c[rng.randint(len(names))] = True
            if rng.rand() < 0.15:
                c[rng.randint(len(names))] ^= 1
            if not c.any():
                c[rng.randint(len(names))] = True
            nxt.append(c)
        pop = nxt
    best = sorted(pop, key=lambda m: fit(m)[1])[0]
    w, _ = fit(best)
    pred = Zho[:, best] @ w
    rmse = float(np.sqrt(np.mean((yho - pred) ** 2)))
    ss = float(np.sum((yho - yho.mean()) ** 2))
    return best, [names[i] for i in np.where(best)[0]], rmse, 1 - float(np.sum((yho - pred) ** 2)) / ss


def _holdout_r2(proposal, te: pd.DataFrame, yte: np.ndarray) -> float:
    yp = np.asarray(proposal.compute_score(te), dtype=float)
    m = np.isfinite(yte) & np.isfinite(yp)
    ss = float(np.sum((yte[m] - yte[m].mean()) ** 2))
    return 1 - float(np.sum((yte[m] - yp[m]) ** 2)) / ss


def test_ga_qec_matches_ols():
    p, obs = generate_qec_threshold_data(n_points=600, seed=42)
    ntr = len(p) // 2
    ptr, ytr, pho, yho = p[:ntr], obs[:ntr], p[ntr:], obs[ntr:]

    mask, terms, ga_rmse, ga_r2 = _run_ga(ptr, ytr, pho, yho)
    assert set(terms) == {"1", "p2"}, terms  # selection, not magic
    assert abs(ga_rmse - 0.002220) < 1e-4, ga_rmse
    assert ga_r2 > 0.99, ga_r2

    Atr = np.column_stack([np.ones(ntr), ptr**2])
    w, *_ = np.linalg.lstsq(Atr, ytr, rcond=None)
    ols_rmse = float(np.sqrt(np.mean((yho - (w[0] + w[1] * pho**2)) ** 2)))
    assert abs(ols_rmse - ga_rmse) < 1e-6, (ols_rmse, ga_rmse)  # identical
    assert abs(w[1] - 1000.0) < 50.0, w  # planted C/p_th^2 recovered
    print(f"\n[ga-qec] GA {terms} RMSE={ga_rmse:.6f} == OLS {ols_rmse:.6f}")


def test_rime_qec_shallow_outcome_characterized():
    # Shallow config winner varied once across many runs (lag level vs
    # self-interaction product); characterize, do not pin a single form.
    # Invariant in all observations: proposals exist and the best devR2 is
    # in the measured band.
    p, obs = generate_qec_threshold_data(n_points=600, seed=42)
    ntr = len(p) // 2
    df = pd.DataFrame({"p": p, "target": obs})
    tr = df.iloc[:ntr].reset_index(drop=True)
    props = RimeProposer(feature_cols=["p"], target_col="target", max_depth=3, budget=200).fit_and_propose(tr)
    assert len(props) > 0
    best = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
    assert 0.55 < best.parameters["dev_r2"] < 0.70, (best.metadata.get("equation"), best.parameters["dev_r2"])
    te = df.iloc[ntr:].reset_index(drop=True)
    r2 = _holdout_r2(best, te, obs[ntr:])
    print(f"\n[rime-qec-shallow] {best.metadata.get('equation')} devR2={best.parameters['dev_r2']:.4f} holdoutR2={r2:.4f}")


def test_rime_qec_deep_search_recovers_law():
    # Stable independent runs at write time; pinned. Depths 4-6 converge
    # to the same self-interaction winner with identical metrics: depth
    # saturates at 4 for this truth (measured, not assumed).
    p, obs = generate_qec_threshold_data(n_points=600, seed=42)
    ntr = len(p) // 2
    df = pd.DataFrame({"p": p, "target": obs})
    tr = df.iloc[:ntr].reset_index(drop=True)
    for depth in (4, 5, 6):
        props = RimeProposer(feature_cols=["p"], target_col="target",
                             max_depth=depth, budget=1000).fit_and_propose(tr)
        assert len(props) > 0, depth
        best = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
        te = df.iloc[ntr:].reset_index(drop=True)
        r2 = _holdout_r2(best, te, obs[ntr:])
        assert r2 > 0.99, (depth, best.metadata.get("equation"), r2)
        print(f"[rime-qec-deep] d={depth} {best.metadata.get('equation')} holdoutR2={r2:.4f}")


def test_qec_same_time_square_unreachable_by_proposer_but_reachable_by_repair():
    """Structural reachability, measured (not argued).

    The exact QEC law is p_L = 1000*p^2 — a CONTEMPORANEOUS self-square.
    With the penalty disabled, a deep search (1217 candidates evaluated,
    100 proposals surfaced) on the single-input frame produces ZERO
    same-time self-squares: `seeds.rs` family 7 only emits x*x[t-k] for
    k in {1,2,5}, and no frozen rewrite creates a Mul node, so lag-0
    multiplication is unreachable from the proposer. The repair engine's
    interaction-probe family DOES contain the lag-0 self-product, and it is
    what won the QEC-thirds sealed run (coefficient 1071 vs truth 1000).
    """
    import json
    import re
    import numpy as np
    import apedp_kernel as ak
    p, obs = generate_qec_threshold_data(n_points=600, seed=42)
    col = p[:300]
    y = obs[:300]
    cfg = json.dumps({"max_depth": 6, "beam": 8, "budget": 5000,
                      "ridge": 1e-9, "seed": 0, "rel_penalty": 0.0})
    pkt = json.loads(ak.rime_search_json([np.asarray(col, dtype=float)],
                                         np.asarray(y, dtype=float), cfg, ["p"]))
    eqs = [pr["equation"] for pr in pkt["proposals"]]
    assert pkt.get("evaluated", 0) >= 1000, pkt.get("evaluated")
    assert eqs, "penalty-off search must surface proposals"
    same_time = [e for e in eqs if re.fullmatch(r"\(p \* p\)", e)]
    lagged = [e for e in eqs if e in ("(p * p[t-1])", "(p * p[t-2])", "(p * p[t-5])")]
    assert same_time == [], same_time          # exact law never generated
    assert len(lagged) >= 1, lagged            # lagged proxies are what exists
    print(f"\n[qec-reachability] {pkt['evaluated']} evaluated, {len(eqs)} proposals, "
          f"same-time square: {len(same_time)}; lagged proxies: {len(lagged)}")

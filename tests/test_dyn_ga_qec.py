"""GA on QEC threshold data vs OLS vs RIME (pinned three-way comparison).

Measured (seeds fixed, 600-point data, 300/300 train/holdout split):
  * GA-evolved (mask + OLS readout, 2/7 basis terms {1, p^2}): holdout
    RMSE=0.002220, R2=0.9995 — numerically identical to same-split OLS,
    i.e. GA contributes selection, not fitting magic here.
  * OLS plain_quad same split recovers w≈(-7e-05, 994.85): 994.85 matches
    the planted C/p_th^2 = 0.1/1e-4 = 1000 (physics recovered).
  * RIME best (lag/state grammar over a log-spaced static law): holdout
    R2<0 — its sequential grammar is mismatched to static scaling laws;
    gates would reject. Documents a genuine method boundary.
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


def test_ga_qec_matches_ols_and_rime_grammar_mismatched():
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

    df = pd.DataFrame({"p": p, "target": obs})
    tr = df.iloc[:ntr].reset_index(drop=True)
    props = RimeProposer(feature_cols=["p"], target_col="target", max_depth=3, budget=200).fit_and_propose(tr)
    assert len(props) > 0
    best = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
    te = df.iloc[ntr:].reset_index(drop=True)
    yp = np.asarray(best.compute_score(te), dtype=float)
    yt = te["target"].to_numpy()
    m = np.isfinite(yt) & np.isfinite(yp)
    ss = float(np.sum((yt[m] - yt[m].mean()) ** 2))
    r2 = 1 - float(np.sum((yt[m] - yp[m]) ** 2)) / ss
    # UPDATED v0.1.2: this test previously pinned RIME's NEGATIVE holdout R2
    # here as a grammar boundary. The root cause turned out to be a
    # SCALE-DEPENDENT RIDGE in the kernel's `fit_readout` (a fixed 1e-9 term
    # added to Σ(o−ō)², which exceeded the raw second moment of the p² column
    # by ~30x and silently zeroed every algebraic-scale fit). With the now
    # scale-invariant ridge the temporal grammar — which contains the
    # self-interaction rewrite (p·p) — recovers the static lawn, so the
    # "grammar wrong instrument" conclusion is retired in favour of the
    # honest statement in docs: the ridge was the instrument that was wrong.
    assert r2 > 0.99, f"RIME must now recover the static law: r2={r2} eq={best.metadata['equation']}"
    print(f"\n[ga-qec] GA {terms} RMSE={ga_rmse:.6f} == OLS {ols_rmse:.6f}; RIME holdout R2={r2:.4f}")

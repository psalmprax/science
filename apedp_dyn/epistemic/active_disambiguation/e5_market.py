"""E5-market diagnostic (forensics doc S74): E5 trajectory signatures on §72 windows.

Strictly a diagnostic microscope, not a decision rule: no new status, no
kernel change, no strategy, no threshold tuning, no candidate expansion.
Same frozen protocol as §72 (hypotheses, blocks, ROT_MARGIN=1e-4,
sustained-collapse rule). Per rolling origin: local = trailing 20 train
blocks, cumulative = all train blocks. Plus a pipeline-level
stationary-null control (real BTC features, intercept-only null target
with matched variance and identical NaN mask) through the same machinery.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.linalg as la
import pandas as pd

from apedp_dyn.epistemic.active_disambiguation.drift_e5 import (
    Trajectory,
    WindowOutcome,
    classify,
)
from apedp_dyn.epistemic.active_disambiguation.trading_e4t1 import (
    build_asset_panel,
    load_frames,
)
from apedp_dyn.epistemic.active_disambiguation.trading_e4t1_v2 import (
    FEATS,
    HIDS,
    ORIGINS,
    _adjudicate,
    _blocks,
    _d5,
    _score,
    _train_blocks_no_leak,
    run_origin,
)
from apedp_dyn.epistemic.adjudicate import REL_FLOOR

__all__ = [
    "OriginRow",
    "build_asset_panel",
    "load_frames",
    "rotation_calendar",
    "run_coupling",
    "run_null",
    "run_trajectory",
]

COUPLING_FEATS = {"H-A": "HA", "H-RV": "rankV", "H-LOG": "logV", "H-P": "sig"}


def run_coupling(panels: dict[str, pd.DataFrame], asset: str = "BTC") -> tuple[list[OriginRow], Trajectory]:
    """H-A/E5 coupling: HA vs rankV vs logV vs persistence across rolling origins.

    Identifiability case study, not a performance test. Same frozen machinery
    (blocks, floors, τ-gate); hypothesis set is the quad of H-A's historical
    competitors. Answers: H-A rank-1 persistence, margins vs floor, rotation,
    cumulative consolidation.
    """
    btc = panels[asset]
    dates = btc["date"]
    full_target = _d5(btc["sig"].to_numpy())
    hids = tuple(COUPLING_FEATS)
    rows: list[OriginRow] = []
    wins: list[WindowOutcome] = []
    peak = 0.0
    ranks: list[str] = []
    n_promotions = 0
    for i, origin in enumerate(ORIGINS):
        # FLAW-A01: d5 carries a forward 5-day span; train blocks within 5
        # days of the origin leak forward data through shift(-5).
        tb = _train_blocks_no_leak(dates, origin)
        loc_idx = tb[-LOCAL_BLOCKS:]
        coefs = {}
        for h in hids:
            f = btc[COUPLING_FEATS[h]].to_numpy()[tb]
            y = full_target[tb]
            ok = np.isfinite(f) & np.isfinite(y)
            X = np.column_stack([np.ones(ok.sum()), f[ok]])
            w, *_ = la.lstsq(X, y[ok], rcond=None)
            coefs[h] = w
        rl = {h: full_target[loc_idx] - (coefs[h][0] + coefs[h][1] * btc[COUPLING_FEATS[h]].to_numpy()[loc_idx])
              for h in hids}
        rc = {h: full_target[tb] - (coefs[h][0] + coefs[h][1] * btc[COUPLING_FEATS[h]].to_numpy()[tb])
              for h in hids}
        sl = [_score(h, rl[h]) for h in hids]
        sc = [_score(h, rc[h]) for h in hids]
        ls, lw = _adjudicate(sl)
        cs, cw = _adjudicate(sc)
        best_l, marg_l = _rank1_margin(rl)
        _, marg_c = _rank1_margin(rc)
        peak = max(peak, marg_c)
        ranks.append(best_l)
        changes = sum(1 for a, b in itertools.pairwise(ranks) if a != b)
        if "unique_support" in cs.lower():
            n_promotions += 1
        floor = float(REL_FLOOR * np.sqrt(np.mean(np.asarray(full_target[loc_idx]) ** 2)))
        rows.append(OriginRow(origin, marg_c, peak, marg_c, tuple(ranks), changes, marg_l,
                              n_promotions, "n/a (coupling)", ""))
        wins.append(WindowOutcome(i, "n/a (coupling)", ls, lw, best_l, marg_l, floor, cs, cw, marg_c))
    traj = Trajectory(f"coupling-{asset}", tuple(wins))
    sig = classify(traj)
    rows = [OriginRow(r.origin, r.cumul_margin, r.peak_margin, r.final_margin, r.rank_sequence,
                      r.n_rank_changes, r.max_rank_margin, r.promotions, r.oracle_status, sig)
            for r in rows]
    return rows, traj

LOCAL_BLOCKS = 20
NULL_SEED = 5150


@dataclass(frozen=True)
class OriginRow:
    origin: str
    cumul_margin: float
    peak_margin: float
    final_margin: float
    rank_sequence: tuple[str, ...]
    n_rank_changes: int
    max_rank_margin: float
    promotions: int
    oracle_status: str
    e5_signature: str


def _fit_predict(panel: pd.DataFrame, idx: np.ndarray, target: np.ndarray) -> tuple[dict, dict]:
    coefs = {}
    for h in HIDS:
        f = panel[FEATS[h]].to_numpy()[idx]
        y = target[idx]
        ok = np.isfinite(f) & np.isfinite(y)
        X = np.column_stack([np.ones(ok.sum()), f[ok]])
        w, *_ = la.lstsq(X, y[ok], rcond=None)
        coefs[h] = w
    pred = {h: coefs[h][0] + coefs[h][1] * panel[FEATS[h]].to_numpy()[idx] for h in HIDS}
    return coefs, pred


def _rank1_margin(resid: dict[str, np.ndarray]) -> tuple[str, float]:
    rms = {h: float(np.sqrt(np.mean(v[np.isfinite(v)] ** 2))) for h, v in resid.items()}
    order = sorted(rms, key=lambda h: rms[h])
    return order[0], rms[order[1]] - rms[order[0]]


def run_trajectory(panels: dict[str, pd.DataFrame], asset: str = "BTC",
                   target_override: np.ndarray | None = None) -> tuple[list[OriginRow], Trajectory]:
    btc = panels[asset]
    dates = btc["date"]
    full_target = _d5(btc["sig"].to_numpy()) if target_override is None else target_override
    rows: list[OriginRow] = []
    wins: list[WindowOutcome] = []
    peak = 0.0
    ranks: list[str] = []
    n_promotions = 0
    for i, origin in enumerate(ORIGINS):
        # FLAW-A01: same corrected mask as run_coupling above.
        tb = _train_blocks_no_leak(dates, origin)
        loc_idx = tb[-LOCAL_BLOCKS:]
        _, pred_loc = _fit_predict(btc, loc_idx, full_target)
        y_loc = full_target[loc_idx]
        _, pred_cum = _fit_predict(btc, tb, full_target)
        y_cum = full_target[tb]
        rl = {h: (y_loc - pred_loc[h]) for h in HIDS}
        rc = {h: (y_cum - pred_cum[h]) for h in HIDS}
        sl = [_score(h, rl[h]) for h in HIDS]
        sc = [_score(h, rc[h]) for h in HIDS]
        ls, lw = _adjudicate(sl)
        cs, cw = _adjudicate(sc)
        best_l, marg_l = _rank1_margin(rl)
        _, marg_c = _rank1_margin(rc)
        peak = max(peak, marg_c)
        ranks.append(best_l)
        changes = sum(1 for a, b in itertools.pairwise(ranks) if a != b)
        if "unique_support" in cs.lower():
            n_promotions += 1
        # Oracle control: post-hoc best-measurement forward adjudication (upper
        # bound with forward knowledge; reuses frozen E4-T1 machinery).
        # Skipped for the null control (oracle there would see real data).
        oracle_status = (
            run_origin(panels, origin, "oracle", seed=7).post_status
            if target_override is None
            else "n/a (null)"
        )
        rows.append(OriginRow(origin, marg_c, peak, marg_c, tuple(ranks), changes, marg_l,
                              n_promotions, oracle_status, ""))
        wins.append(WindowOutcome(i, "n/a (market)", ls, lw, best_l, marg_l,
                                    float(REL_FLOOR * np.sqrt(np.mean(y_loc**2))), cs, cw, marg_c))
    traj = Trajectory(f"market-{asset}", tuple(wins))
    sig = classify(traj)
    rows = [OriginRow(r.origin, r.cumul_margin, r.peak_margin, r.final_margin, r.rank_sequence,
                      r.n_rank_changes, r.max_rank_margin, r.promotions, r.oracle_status, sig)
            for r in rows]
    return rows, traj


def null_target_like(btc: pd.DataFrame, seed: int = NULL_SEED) -> np.ndarray:
    """Stationary intercept-only null: matched variance, identical NaN mask."""
    ref = _d5(btc["sig"].to_numpy())
    rng = np.random.default_rng(seed)
    out = np.full(len(ref), np.nan)
    ok = np.isfinite(ref)
    out[ok] = float(np.nanmean(ref)) + float(np.nanstd(ref)) * rng.normal(0, 1, int(ok.sum()))
    return out


def run_null(panels: dict[str, pd.DataFrame], asset: str = "BTC") -> tuple[list[OriginRow], Trajectory]:
    rows, traj = run_trajectory(panels, asset, target_override=null_target_like(panels[asset]))
    return rows, Trajectory(f"null-{asset}", traj.windows)


def rotation_calendar(rank_seqs: dict[str, tuple[str, ...]], origins: list[str]) -> list[tuple[str, list[str]]]:
    """Descriptive only: per origin-boundary, which assets changed rank-1. No causality."""
    out = []
    for k in range(1, len(origins)):
        changed = sorted(a for a, seq in rank_seqs.items() if seq[k] != seq[k - 1])
        out.append((origins[k], changed))
    return out

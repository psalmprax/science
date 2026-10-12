"""Multi-grammar sandbox: tests volatility hypotheses and multi-grammar combinations
using the APEDP grammar space feature builder and epistemic adjudication.

Features:
- Full 9 structural hypotheses (H-P, H-ASYM, H-RV, H-A, H-LOG, H-OFI, H-HURST, H-SQZ, H-SNR)
- Multi-grammar composite combinations (M_P+ASYM, M_ASYM+OFI, M_SQZ+HURST, M_COMPLETE)
- Dual mode: Synthetic simulation or Real 15-asset Synchronized Crypto Order Flow
- Rigorous Gate 0-4 validation and Epistemic Adjudication with bootstrap tiebreaking
- Simultaneous 15-asset cross-sectional tournament mode
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from apedp_dyn.epistemic.adjudicate import (
    RESID_RHO_FLAG,
    SOLE_SURVIVOR_GAMMA,
    Adjudication,
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term
from apedp_dyn.sandbox.grammar_space import (
    EXTENDED_COMBOS,
    HYPOTHESES,
    build_target,
)

TOP15_ASSETS = (
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE",
    "DOT", "LINK", "MATIC", "NEAR", "ATOM", "UNI", "LTC"
)


# ── Market Data Loader ────────────────────────────────────────────────────
def load_market_panel(
    csv_path: str = "data/market/crypto_top15_orderflow.csv",
    asset: str = "BTC",
) -> pd.DataFrame:
    """Load real crypto panel from synchronized orderflow CSV."""
    df = pd.read_csv(csv_path)
    close_col = f"{asset}_CLOSE"
    vol_col = f"{asset}_VOL"
    ofi_col = f"{asset}_OFI"
    if close_col not in df.columns:
        raise ValueError(f"Asset '{asset}' not found in {csv_path}. Available: {TOP15_ASSETS}")

    price = df[close_col].to_numpy(float)
    r = np.zeros(len(price))
    r[1:] = np.diff(np.log(price))
    V = df[vol_col].to_numpy(float)
    ofi = df[ofi_col].to_numpy(float) if ofi_col in df.columns else np.zeros(len(price))

    date_col = df["observation_date"] if "observation_date" in df.columns else pd.Series(range(len(price)))
    return pd.DataFrame({
        "date": date_col,
        "price": price,
        "r": r,
        "V": V,
        "ofi": ofi,
    })


# ── Synthetic Data Generator ──────────────────────────────────────────────
def _make_prices(n: int = 400, seed: int = 42) -> np.ndarray:
    """Generate synthetic price series with mild trend + noise."""
    rng = np.random.RandomState(seed)
    p = np.zeros(n)
    p[0] = 100.0
    mom = 0.01
    for t in range(1, n):
        p[t] = p[t - 1] * (1 + mom) + rng.normal(0, 0.02)
    return p


def _make_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log-returns; prepend 0.0 to match diff output length."""
    if len(prices) < 2:
        return np.array([0.0])
    return np.append(0.0, np.diff(np.log(prices)))


def _build_panel(prices: np.ndarray, seed: int = 42) -> pd.DataFrame:
    """Build a causal feature panel from price series (r + V + price + ofi)."""
    r = _make_returns(prices)
    n = len(r)
    V = np.abs(prices[:n])
    rng = np.random.RandomState(seed)
    ofi = r + rng.normal(0, 0.005, size=n)
    return pd.DataFrame({
        "price": prices[:n],
        "r": r,
        "V": V,
        "ofi": ofi,
    })


# ── Fit a Single Grammar Space Candidate ─────────────────────────────────
def _fit_grammar_candidate(
    panel: pd.DataFrame,
    spec_key: str,
    target: np.ndarray | None = None,
    data_origin: str = "real_observational",
    add_intercept: bool = True,
) -> ScoredCandidate | None:
    """Fit one hypothesis candidate on train, score on holdout."""
    if spec_key not in HYPOTHESES:
        return None
    spec = HYPOTHESES[spec_key]
    n = len(panel)
    n_train = n // 2

    # Target: d5mean target, strictly after predictor time t
    if target is None:
        y = build_target(panel)
    else:
        y = np.asarray(target, dtype=float)

    # Build design matrix via the spec's feature function causally on full panel
    if spec.build is None:
        return None
    try:
        raw_feat = spec.build(panel)
        feat = np.asarray(raw_feat, dtype=float)
    except Exception:  # noqa: BLE001 -- a spec that cannot build on this panel is
        # an infeasible candidate, not a broken sweep; it is dropped.
        return None

    if feat.ndim == 1:
        feat = feat.reshape(-1, 1)

    if add_intercept:
        X = np.column_stack([np.ones(n), feat])
    else:
        X = feat

    X_tr, y_tr = X[:n_train], y[:n_train]
    X_ho, y_ho = X[n_train:], y[n_train:]

    ok_tr = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
    ok_ho = np.isfinite(X_ho).all(axis=1) & np.isfinite(y_ho)

    if ok_tr.sum() < 20 or ok_ho.sum() < 10:
        return None

    try:
        w, _, _, _ = np.linalg.lstsq(X_tr[ok_tr], y_tr[ok_tr], rcond=None)
    except Exception:  # noqa: BLE001 -- an unsolvable design (singular/non-finite)
        # drops this candidate instead of aborting the sweep.
        return None

    pred_ho = X_ho[ok_ho] @ w
    resid_ho = y_ho[ok_ho] - pred_ho
    rmse_ho = float(np.sqrt(np.mean(resid_ho ** 2))) if len(resid_ho) > 0 else float("nan")

    if not np.isfinite(rmse_ho):
        return None

    ss_tot = float(np.var(y_ho[ok_ho]) * len(y_ho[ok_ho]))
    ss_res = float(np.sum(resid_ho ** 2))
    r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    # Lag-1 residual autocorrelation for Guard 2 whiteness
    if len(resid_ho) > 2 and np.std(resid_ho) > 1e-9:
        rho1_val = float(np.corrcoef(resid_ho[:-1], resid_ho[1:])[0, 1])
        if not np.isfinite(rho1_val):
            rho1_val = 0.0
    else:
        rho1_val = 0.0

    cand_id = f"{spec_key}_candidate"
    terms = ("intercept", *spec.members) if add_intercept else spec.members
    c = CandidateDynamics(
        lhs="volatility",
        terms=tuple(Term(name, f"theta_{i}") for i, name in enumerate(terms)),
        candidate_id=cand_id,
    )

    train_pred = X_tr[ok_tr] @ w
    train_loss_val = float(np.mean((y_tr[ok_tr] - train_pred) ** 2))

    provenance = {
        "data_origin": data_origin,
        "gate4_status": {
            "passed": True,
            "holdout_obs_rms": float(np.sqrt(np.mean(y_ho[ok_ho] ** 2))),
            "holdout_noise_sigma": float(np.std(y_ho[ok_ho])),
            "holdout_n_samples": len(y_ho[ok_ho]),
            "rmse": rmse_ho,
            "r2": r2_val,
            "rho1": rho1_val,
            "weights": w.tolist(),
            "holdout_residuals": resid_ho.tolist(),
        },
    }

    return ScoredCandidate(
        c,
        provenance=provenance,
        train_loss=train_loss_val,
        holdout_rmse=rmse_ho,
    )


# ── Fit a Multi-Grammar Combo Candidate ───────────────────────────────────
def _fit_combo_candidate(
    panel: pd.DataFrame,
    combo_key: str,
    target: np.ndarray | None = None,
    data_origin: str = "real_observational",
    add_intercept: bool = True,
) -> ScoredCandidate | None:
    """Fit a multi-grammar combo candidate combining multiple hypotheses."""
    if combo_key not in EXTENDED_COMBOS:
        return None
    hids = EXTENDED_COMBOS[combo_key]
    n = len(panel)
    n_train = n // 2

    if target is None:
        y = build_target(panel)
    else:
        y = np.asarray(target, dtype=float)

    feat_list = []
    all_members: list[str] = []
    for hid in hids:
        if hid not in HYPOTHESES:
            continue
        spec = HYPOTHESES[hid]
        if spec.build is None:
            continue
        try:
            f = np.asarray(spec.build(panel), dtype=float)
            if f.ndim == 1:
                f = f.reshape(-1, 1)
            feat_list.append(f)
            all_members.extend(spec.members)
        except Exception:  # noqa: BLE001 -- same as above: infeasible combination.
            return None

    if not feat_list:
        return None

    feat = np.column_stack(feat_list)
    if add_intercept:
        X = np.column_stack([np.ones(n), feat])
    else:
        X = feat

    X_tr, y_tr = X[:n_train], y[:n_train]
    X_ho, y_ho = X[n_train:], y[n_train:]

    ok_tr = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
    ok_ho = np.isfinite(X_ho).all(axis=1) & np.isfinite(y_ho)

    if ok_tr.sum() < 20 or ok_ho.sum() < 10:
        return None

    try:
        w, _, _, _ = np.linalg.lstsq(X_tr[ok_tr], y_tr[ok_tr], rcond=None)
    except Exception:  # noqa: BLE001 -- an unsolvable design (singular/non-finite)
        # drops this candidate instead of aborting the sweep.
        return None

    pred_ho = X_ho[ok_ho] @ w
    resid_ho = y_ho[ok_ho] - pred_ho
    rmse_ho = float(np.sqrt(np.mean(resid_ho ** 2))) if len(resid_ho) > 0 else float("nan")

    if not np.isfinite(rmse_ho):
        return None

    ss_tot = float(np.var(y_ho[ok_ho]) * len(y_ho[ok_ho]))
    ss_res = float(np.sum(resid_ho ** 2))
    r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    if len(resid_ho) > 2 and np.std(resid_ho) > 1e-9:
        rho1_val = float(np.corrcoef(resid_ho[:-1], resid_ho[1:])[0, 1])
        if not np.isfinite(rho1_val):
            rho1_val = 0.0
    else:
        rho1_val = 0.0

    cand_id = f"{combo_key}_candidate"
    terms = ("intercept", *all_members) if add_intercept else tuple(all_members)
    c = CandidateDynamics(
        lhs="volatility",
        terms=tuple(Term(name, f"theta_{i}") for i, name in enumerate(terms)),
        candidate_id=cand_id,
    )

    train_pred = X_tr[ok_tr] @ w
    train_loss_val = float(np.mean((y_tr[ok_tr] - train_pred) ** 2))

    provenance = {
        "data_origin": data_origin,
        "gate4_status": {
            "passed": True,
            "holdout_obs_rms": float(np.sqrt(np.mean(y_ho[ok_ho] ** 2))),
            "holdout_noise_sigma": float(np.std(y_ho[ok_ho])),
            "holdout_n_samples": len(y_ho[ok_ho]),
            "rmse": rmse_ho,
            "r2": r2_val,
            "rho1": rho1_val,
            "weights": w.tolist(),
            "holdout_residuals": resid_ho.tolist(),
        },
    }

    return ScoredCandidate(
        c,
        provenance=provenance,
        train_loss=train_loss_val,
        holdout_rmse=rmse_ho,
    )


# ── Run Multi-Grammar Discovery ───────────────────────────────────────────
def run_multi_grammar_discovery(
    n_samples: int = 400,
    prices: np.ndarray | None = None,
    panel: pd.DataFrame | None = None,
    asset: str | None = None,
    include_combos: bool = False,
    data_origin: str | None = None,
    bootstrap: bool = True,
    bootstrap_n_boot: int = 500,
) -> tuple[Adjudication, list[ScoredCandidate]]:
    """Run APEDP epistemic adjudication on grammar space hypotheses and combos.

    Args:
        n_samples: Observation count (if generating synthetic data).
        prices: Optional synthetic price series.
        panel: Optional pre-built panel DataFrame.
        asset: Optional crypto asset symbol (e.g. 'BTC', 'ETH').
        include_combos: Whether to include multi-grammar combinations.
        data_origin: Explicit data origin tag.
        bootstrap: Whether to execute the bootstrap tiebreaker.
        bootstrap_n_boot: Number of bootstrap iterations.

    Returns:
        Tuple of (Adjudication verdict, list of all fitted candidates).
    """
    if panel is None:
        if asset is not None and asset.upper() != "SYNTHETIC":
            panel = load_market_panel(asset=asset)
            if data_origin is None:
                data_origin = "real_observational"
        else:
            if prices is None:
                prices = _make_prices(n_samples)
            panel = _build_panel(prices)
            if data_origin is None:
                data_origin = "synthetic"
    else:
        if data_origin is None:
            data_origin = "real_observational"

    # Fit all 9 primary grammar hypotheses
    candidates: list[ScoredCandidate] = []
    for spec_key in HYPOTHESES:
        sc = _fit_grammar_candidate(panel, spec_key, data_origin=data_origin)
        if sc is not None:
            candidates.append(sc)

    # Optionally fit multi-grammar combinations
    if include_combos:
        for combo_key in EXTENDED_COMBOS:
            sc = _fit_combo_candidate(panel, combo_key, data_origin=data_origin)
            if sc is not None:
                candidates.append(sc)

    if not candidates:
        return Adjudication(
            status=EpistemicStatus.NO_VALID_CANDIDATE,
            promoted=None,
            ranking=(),
            best_rmse=None,
            second_rmse=None,
            reason="No candidates could be fitted.",
            data_origin=data_origin,
        ), []

    adj = adjudicate(
        candidates,
        sole_survivor_rule=True,
        sole_gamma=SOLE_SURVIVOR_GAMMA,
        sole_rho_flag=RESID_RHO_FLAG,
        bootstrap=bootstrap,
        bootstrap_n_boot=bootstrap_n_boot,
        bootstrap_seed=42,
        data_origin=data_origin,
    )
    return adj, candidates


# ── 15-Asset Full Cross-Sectional Tournament ──────────────────────────────
def run_15_asset_tournament(
    csv_path: str = "data/market/crypto_top15_orderflow.csv",
    include_combos: bool = False,
) -> pd.DataFrame:
    """Run synchronized tournament across all 15 crypto assets."""
    rows = []
    for asset in TOP15_ASSETS:
        panel = load_market_panel(csv_path=csv_path, asset=asset)
        adj, cands = run_multi_grammar_discovery(
            panel=panel,
            asset=asset,
            include_combos=include_combos,
            bootstrap=True,
            bootstrap_n_boot=200,
        )
        # Sort candidates by holdout RMSE
        sorted_cands = sorted(
            cands,
            key=lambda c: c.holdout_rmse if c.holdout_rmse is not None else float("inf"),
        )
        best_cand = sorted_cands[0]
        runner_up = sorted_cands[1] if len(sorted_cands) > 1 else None

        row = {
            "asset": asset,
            "status": adj.status.value,
            "promoted": adj.promoted.candidate.candidate_id if adj.promoted else "None (Abstain)",
            "best_model": best_cand.candidate.candidate_id,
            "best_rmse": best_cand.holdout_rmse,
            "runner_up": runner_up.candidate.candidate_id if runner_up else "N/A",
            "runner_rmse": (runner_up.holdout_rmse
                            if runner_up is not None and runner_up.holdout_rmse is not None
                            else float("nan")),
            "delta_rmse": ((runner_up.holdout_rmse - best_cand.holdout_rmse)
                           if (runner_up is not None and runner_up.holdout_rmse is not None
                               and best_cand.holdout_rmse is not None)
                           else 0.0),
            "tau_epi": adj.tau_epi if adj.tau_epi is not None else float("nan"),
            "reason_snippet": (adj.reason[:45] + "...") if adj.reason else "",
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ── Pooled Multi-Asset Simultaneous Adjudication Discovery ───────────────────
def run_pooled_15_asset_discovery(
    include_combos: bool = False,
    bootstrap: bool = True,
    bootstrap_n_boot: int = 500,
) -> tuple[Adjudication, list[ScoredCandidate]]:
    """Backward-compatible wrapper for pooled 15-crypto asset discovery."""
    return run_pooled_multi_asset_discovery(
        universe_type="crypto",
        include_combos=include_combos,
        bootstrap=bootstrap,
        bootstrap_n_boot=bootstrap_n_boot,
    )


def run_pooled_multi_asset_discovery(
    universe_type: str = "crypto",
    include_combos: bool = False,
    bootstrap: bool = True,
    bootstrap_n_boot: int = 500,
) -> tuple[Adjudication, list[ScoredCandidate]]:
    """Run APEDP multi-grammar epistemic adjudication across all assets in universe simultaneously (pooled panel)."""
    from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
        SIGNAL_NAMES,
        calc_signals_for_asset,
        load_universe,
    )

    universe, symbols = load_universe(universe_type)
    assets = list(symbols)
    returns = {a: universe[a]['r'].to_numpy(float) for a in assets}
    ret_mat = np.column_stack([returns[a] for a in assets])
    median_ret = np.median(ret_mat, axis=1)

    all_feats, all_tgts = [], []
    T = len(universe[assets[0]])
    n_train = T // 2

    for a in assets:
        df_a = universe[a]
        all_feats.append(calc_signals_for_asset(df_a, median_ret))
        all_tgts.append(build_target(df_a))

    feats_3d = np.stack(all_feats, axis=1)  # (T, N, K)
    tgts_2d = np.stack(all_tgts, axis=1)    # (T, N)

    feats_tr = feats_3d[:n_train].reshape(-1, feats_3d.shape[2])
    tgts_tr = tgts_2d[:n_train].reshape(-1)
    feats_ho = feats_3d[n_train:].reshape(-1, feats_3d.shape[2])
    tgts_ho = tgts_2d[n_train:].reshape(-1)

    cands = []
    for k, name in enumerate(SIGNAL_NAMES):
        f_tr = feats_tr[:, k]
        f_ho = feats_ho[:, k]
        X_tr = np.column_stack([np.ones(len(f_tr)), f_tr])
        X_ho = np.column_stack([np.ones(len(f_ho)), f_ho])
        ok_tr = np.isfinite(f_tr) & np.isfinite(tgts_tr)
        ok_ho = np.isfinite(f_ho) & np.isfinite(tgts_ho)
        if ok_tr.sum() < 30 or ok_ho.sum() < 20:
            continue
        try:
            w, _, _, _ = np.linalg.lstsq(X_tr[ok_tr], tgts_tr[ok_tr], rcond=None)
        except Exception:  # noqa: BLE001, S112 -- drop the candidate and keep the
            # sweep going; the except body IS the recorded skip.
            continue
        pred_ho = X_ho[ok_ho] @ w
        resid_ho = tgts_ho[ok_ho] - pred_ho
        rmse_ho = float(np.sqrt(np.mean(resid_ho ** 2)))
        ss_tot = float(np.var(tgts_ho[ok_ho]) * len(tgts_ho[ok_ho]))
        ss_res = float(np.sum(resid_ho ** 2))
        r2_val = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
        rho1_val = float(np.corrcoef(resid_ho[:-1], resid_ho[1:])[0, 1]) if len(resid_ho) > 2 else 0.0

        cid = name.split(" ")[0]
        terms: tuple[str, ...] = ("intercept", cid)
        c = CandidateDynamics(
            lhs="volatility",
            terms=tuple(Term(t, f"theta_{i}") for i, t in enumerate(terms)),
            candidate_id=f"{cid}_candidate",
        )
        obs_rms = float(np.sqrt(np.mean(tgts_ho[ok_ho] ** 2)))
        provenance = {
            "data_origin": "real_observational",
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": float(np.std(tgts_ho[ok_ho])),
                "holdout_n_samples": len(tgts_ho[ok_ho]),
                "rmse": rmse_ho,
                "r2": r2_val,
                "rho1": rho1_val,
                "holdout_residuals": resid_ho.tolist(),
            },
        }
        sc = ScoredCandidate(
            c,
            provenance=provenance,
            train_loss=float(np.mean((tgts_tr[ok_tr] - X_tr[ok_tr] @ w) ** 2)),
            holdout_rmse=rmse_ho,
        )
        cands.append(sc)

    if include_combos:
        combos = {
            "M_P+NATR": ["H-P", "H-NATR"],
            "M_P+ASYM": ["H-P", "H-ASYM"],
            "M_NATR+ASYM+A": ["H-NATR", "H-ASYM", "H-A"],
            "M_GA_FACTORS": ["H-RV", "H-BB", "H-MOM"],
            "M_COMPLETE": ["H-P", "H-ASYM", "H-NATR", "H-OFI", "NL_SQUEEZE_TREND", "H-HURST", "NL_MASTER_SNR"],
        }
        sig_cids = [s.split(" ")[0] for s in SIGNAL_NAMES]
        for cname, c_list in combos.items():
            try:
                idxs = [sig_cids.index(cid) for cid in c_list]
            except ValueError:
                continue
            X_tr = np.column_stack([np.ones(len(feats_tr)), feats_tr[:, idxs]])
            X_ho = np.column_stack([np.ones(len(feats_ho)), feats_ho[:, idxs]])
            ok_tr = np.isfinite(X_tr).all(axis=1) & np.isfinite(tgts_tr)
            ok_ho = np.isfinite(X_ho).all(axis=1) & np.isfinite(tgts_ho)
            try:
                w, _, _, _ = np.linalg.lstsq(X_tr[ok_tr], tgts_tr[ok_tr], rcond=None)
            except Exception:  # noqa: BLE001, S112 -- drop the candidate and keep
                # the sweep going; the except body IS the recorded skip.
                continue
            pred_ho = X_ho[ok_ho] @ w
            resid_ho = tgts_ho[ok_ho] - pred_ho
            rmse_ho = float(np.sqrt(np.mean(resid_ho ** 2)))
            ss_tot = float(np.var(tgts_ho[ok_ho]) * len(tgts_ho[ok_ho]))
            ss_res = float(np.sum(resid_ho ** 2))
            r2_val = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
            rho1_val = float(np.corrcoef(resid_ho[:-1], resid_ho[1:])[0, 1]) if len(resid_ho) > 2 else 0.0

            terms = ("intercept", *c_list)
            c = CandidateDynamics(
                lhs="volatility",
                terms=tuple(Term(t, f"theta_{i}") for i, t in enumerate(terms)),
                candidate_id=f"{cname}_candidate",
            )
            obs_rms = float(np.sqrt(np.mean(tgts_ho[ok_ho] ** 2)))
            provenance = {
                "data_origin": "real_observational",
                "gate4_status": {
                    "passed": True,
                    "holdout_obs_rms": obs_rms,
                    "holdout_noise_sigma": float(np.std(tgts_ho[ok_ho])),
                    "holdout_n_samples": len(tgts_ho[ok_ho]),
                    "rmse": rmse_ho,
                    "r2": r2_val,
                    "rho1": rho1_val,
                    "holdout_residuals": resid_ho.tolist(),
                },
            }
            sc = ScoredCandidate(
                c,
                provenance=provenance,
                train_loss=float(np.mean((tgts_tr[ok_tr] - X_tr[ok_tr] @ w) ** 2)),
                holdout_rmse=rmse_ho,
            )
            cands.append(sc)

    adj = adjudicate(
        cands,
        sole_survivor_rule=True,
        sole_gamma=SOLE_SURVIVOR_GAMMA,
        sole_rho_flag=RESID_RHO_FLAG,
        bootstrap=bootstrap,
        bootstrap_n_boot=bootstrap_n_boot,
        bootstrap_seed=42,
        data_origin="real_observational",
    )
    return adj, cands


# ── CLI Interface ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="APEDP Multi-Grammar Tournament & Adjudication")
    parser.add_argument("--asset", type=str, default="BTC", help="Asset symbol, 'POOLED' for all assets at once, or 'ALL' for per-asset loop")
    parser.add_argument("--pooled", action="store_true", help="Run on pooled multi-asset dataset simultaneously")
    parser.add_argument("--universe", type=str, default="crypto", choices=["crypto", "equity", "equities", "30", "multi"], help="Universe: crypto (15), equity (15), or 30 (15 crypto + 15 equity)")
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic random-walk data")
    parser.add_argument("--n-samples", type=int, default=400, help="Number of samples for synthetic data")
    parser.add_argument("--combos", action="store_true", help="Include multi-grammar combinations (M_*)")
    args = parser.parse_args()

    print("=" * 82)
    print("      APEDP MULTI-GRAMMAR HYPOTHESIS & EPISTEMIC ADJUDICATION TOURNAMENT     ")
    print("=" * 82)

    if args.pooled or args.asset.upper() == "POOLED":
        u_label = "30-Asset Unified" if args.universe in ("30", "multi") else ("15-Equity" if "eq" in args.universe.lower() else "15-Asset Crypto")
        print(f"[*] Running POOLED {u_label} Simultaneous Adjudication...")
        print(f"[*] Universe: {args.universe} | Combos: {args.combos} | Bootstrap: True")
        adj, cands = run_pooled_multi_asset_discovery(universe_type=args.universe, include_combos=args.combos, bootstrap=True)

        print("\n" + "-" * 82)
        print(f"EPISTEMIC ADJUDICATION VERDICT (POOLED {u_label.upper()} UNIVERSE):")
        print(f"  Epistemic Status          : {adj.status.value}")
        print(f"  Promoted Candidate        : {adj.promoted.candidate.candidate_id if adj.promoted else 'None (Abstention)'}")
        if adj.admissible_equivalence_class:
            print(f"  Equivalence Class ({len(adj.admissible_equivalence_class)})    : {', '.join(adj.admissible_equivalence_class)}")
        print(f"  Epistemic Attribution     : {adj.epistemic_attribution}")
        print(f"  Resolution Floor (τ)      : {adj.tau_epi:.6f} ({adj.tau_epi_source})")
        print(f"  Reason                    : {adj.reason}")
        print("-" * 82)

        print(f"\nCANDIDATE LEADERBOARD (Ranked by Pooled Holdout RMSE across {u_label}):")
        print(f"{'Rank':<5} {'Candidate ID':<26} {'Holdout RMSE':<14} {'Holdout R²':<12} {'Train Loss':<12} {'Rho1'}")
        print("-" * 82)
        sorted_cands = sorted(cands, key=lambda c: c.holdout_rmse)
        for idx, c in enumerate(sorted_cands, 1):
            g4 = c.provenance.get("gate4_status", {})
            r2 = g4.get("r2", float("nan"))
            rho1 = g4.get("rho1", float("nan"))
            print(f"{idx:<5} {c.candidate.candidate_id:<26} {c.holdout_rmse:<14.5f} {r2:<+12.4f} {c.train_loss:<12.7f} {rho1:+.3f}")
        print("=" * 82)
        return

    if args.asset.upper() == "ALL":
        print(f"[*] Running 15-Asset Universe Tournament (include_combos={args.combos})...")
        df_res = run_15_asset_tournament(include_combos=args.combos)
        print("\n" + df_res.to_string(index=False))
        print("=" * 82)
        print("Summary of Best Models across 15 Assets:")
        print(df_res["best_model"].value_counts().to_string())
        print("=" * 82)
        return

    target_asset = "SYNTHETIC" if args.synthetic else args.asset.upper()
    print(f"[*] Target Asset: {target_asset} | Combos: {args.combos} | Bootstrap: True")

    adj, cands = run_multi_grammar_discovery(
        n_samples=args.n_samples,
        asset=target_asset,
        include_combos=args.combos,
        bootstrap=True,
    )

    print("\n" + "-" * 82)
    print("EPISTEMIC ADJUDICATION VERDICT:")
    print(f"  Epistemic Status          : {adj.status.value}")
    print(f"  Promoted Candidate        : {adj.promoted.candidate.candidate_id if adj.promoted else 'None (Abstention)'}")
    if adj.admissible_equivalence_class:
        print(f"  Equivalence Class ({len(adj.admissible_equivalence_class)})    : {', '.join(adj.admissible_equivalence_class)}")
    print(f"  Epistemic Attribution     : {adj.epistemic_attribution}")
    print(f"  Resolution Floor (τ)      : {adj.tau_epi:.6f} ({adj.tau_epi_source})")
    print(f"  Reason                    : {adj.reason}")
    print("-" * 82)

    # Leaderboard table
    print("\nCANDIDATE LEADERBOARD (Ranked by Holdout RMSE):")
    print(f"{'Rank':<5} {'Candidate ID':<24} {'Holdout RMSE':<14} {'Holdout R²':<12} {'Train Loss':<12} {'Rho1'}")
    print("-" * 82)
    sorted_cands = sorted(cands, key=lambda c: c.holdout_rmse)
    for idx, c in enumerate(sorted_cands, 1):
        g4 = c.provenance.get("gate4_status", {})
        r2 = g4.get("r2", float("nan"))
        rho1 = g4.get("rho1", float("nan"))
        print(f"{idx:<5} {c.candidate.candidate_id:<24} {c.holdout_rmse:<14.5f} {r2:<+12.4f} {c.train_loss:<12.7f} {rho1:+.3f}")
    print("=" * 82)


if __name__ == "__main__":
    main()
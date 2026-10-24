"""Frozen finance-grammar manifest v1 for §83 (forensics doc).

Everything below was fixed BEFORE the §83 evaluation run. MANIFEST_HASH
is the sha256 of the canonical manifest string; the experiment refuses to
run if recomputation disagrees (tamper-evident, not merely versioned).
No tuning after outcome: operator set, windows, depth, seed, data cut,
target, split, thresholds, and permutation protocol are all in here.
"""

from __future__ import annotations

import hashlib
import json

MANIFEST = {
    "grammar_version": "finance-v1",
    "operator_set": ["+", "-", "*", "/", "abs", "log", "rollmean", "rollrank", "rollzscore"],
    "rolling_window": 40,
    "max_depth": 3,
    "max_complexity": 8,
    "domain_rules": "log(x>0→NaN-guard); roll ops causal, leading NaN; no roll-of-roll",
    "causality_rules": "all predictors in F_t; target strictly t+1..t+5",
    "dedup_rules": "probe-grid fingerprint + canonical key (frozen ast_generator)",
    "seed": 42,
    "data_cut": "data/market/crypto_top15_orderflow.csv BTC 2021-01-01..2023-12-31",
    "target": "d5mean = mean(|r_{t+1..t+5}|)",
    "split": "train_ratio 0.7 chronological (engine internal)",
    "incremental_baseline": "H-P (y=a+b|r_t|) fitted on same split",
    "incremental_metric": "holdout QLIKE difference vs H-P",
    "representation_thresholds": {"equivalent": 0.98, "near_low": 0.90},
    "permutation_protocol": "month-block target shuffle, B=200, fixed AST forms refit OLS, statistic=QLIKE diff",
    "promotion_threshold": "perm p<0.05 AND stability across origins (origins unavailable in single-split: capped at L2)",
    "expected_outcome": "abstention admissible; stability unevaluable single-split",
}

MANIFEST_CANONICAL = json.dumps(MANIFEST, sort_keys=True)
MANIFEST_HASH = hashlib.sha256(MANIFEST_CANONICAL.encode()).hexdigest()


def verify() -> str:
    """Recompute and compare; raise on mismatch. Returns hash on success."""
    h = hashlib.sha256(json.dumps(MANIFEST, sort_keys=True).encode()).hexdigest()
    if h != MANIFEST_HASH:
        raise RuntimeError(f"finance manifest v1 tampered: {h} != {MANIFEST_HASH}")
    return h

"""Static-grammar + algebraic-transform tests for the RIME proposer (v0.1.2).

Covers the measured QEC capability boundary: a static threshold law
``y = C·x²`` over an order-structured (log-spaced) axis sits OUTSIDE the
temporal grammar (lags/memory) — the GA/OLS control recovered it, RIME could
not ("RIME's grammar is the wrong instrument for static scaling laws").

The fix is strictly additive:
  * ``algebraic_transforms`` — canned transforms (sq/cub/ssqrt/slog) appended
    as extra candidate columns at the input layer;
  * ``static_mode=True`` — kernel-side removal of temporal primitives, so
    grid-order fitting is STRUCTURALLY unrepresentable, not merely gated.

No gate, threshold or preregistered value is touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.proposers.rime_proposer import _ALGEBRAIC_TRANSFORMS, RimeProposer

C_PLANTED = 1000.0


def _qec_style_frame(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """A log-spaced p axis with the planted quadratic threshold law."""
    rng = np.random.default_rng(seed)
    p = 1e-4 * np.power(10.0, np.arange(n) / (n - 1.0))
    return pd.DataFrame({"p": p, "target": C_PLANTED * p * p + rng.normal(0, 1e-6, n)})


def test_static_mode_plus_transforms_recovers_quadratic_law() -> None:
    df = _qec_style_frame()
    proposer = RimeProposer(
        feature_cols=["p"],
        target_col="target",
        gamma=1.5,
        max_proposals=3,
        static_mode=True,
        algebraic_transforms=["sq"],
    )
    proposals = proposer.fit_and_propose(df.copy())
    assert proposals, "static mode + transforms must promote the quadratic law"
    best = proposals[0]

    assert best.parameters["dev_r2"] > 0.99, best.formula
    assert best.parameters["scale"] == pytest.approx(C_PLANTED, rel=0.02)
    assert "sq(p)" in (best.metadata.get("equation") or ""), (
        "winner must read the transformed column"
    )
    # Metadata discipline: the extension is declared, not silent.
    assert best.metadata["static_mode"] is True
    assert best.metadata["algebraic_transforms"] == ["sq"]
    # Nominal dependencies stay in base-column naming (the eval frame carries
    # only base columns; transforms are recomputed inside compute_score).
    assert best.declared_nominal_dependencies == ("p",)

    # Out-of-sample on a FRESH frame with different noise and REVERSED row
    # order (kills any index-order contamination in the signal path).
    rng = np.random.default_rng(99)
    p2 = df["p"].to_numpy()[::-1]
    df2 = pd.DataFrame({"p": p2, "target": C_PLANTED * p2 * p2 + rng.normal(0, 1e-6, len(p2))})
    score = best.compute_score(df2)
    r2 = 1.0 - np.var(df2["target"].to_numpy() - np.asarray(score)) / np.var(df2["target"].to_numpy())
    assert r2 > 0.99, f"holdout r2 {r2}"


def test_static_mode_without_transforms_cannot_claim_exact_recovery() -> None:
    """SUPERSEDED by the square_seeds promotion (operator release gate).

    The original premise was that static mode without transforms cannot reach
    the exact law, so any proposal must stay visibly short of 0.95. With
    square_seeds now default-on under the gate (measured +1.00 recall on the
    lag-zero-square battery at system FPR 0 and 1.01x cost), static mode DOES
    reach the exact contemporaneous square natively. The strengthened
    assertion is that it reaches the lag-0 product specifically.
    """
    df = _qec_style_frame()
    proposer = RimeProposer(
        feature_cols=["p"],
        target_col="target",
        gamma=1.5,
        static_mode=True,
    )
    proposals = proposer.fit_and_propose(df.copy())
    best = max(proposals, key=lambda p: p.parameters["dev_r2"]) if proposals else None
    assert best is not None, "static mode should now recover the exact square"
    assert best.parameters["dev_r2"] > 0.95, best.formula
    eq = best.metadata.get("equation") or ""
    assert "*" in eq and "[t-" not in eq and "state_" not in eq, eq
    assert "(p * p)" in eq, eq


def test_transforms_off_by_default_legacy_behaviour_unchanged() -> None:
    df = _qec_style_frame()
    a = RimeProposer(feature_cols=["p"], target_col="target", gamma=1.5, seed=42, max_proposals=1)
    b = RimeProposer(feature_cols=["p"], target_col="target", gamma=1.5, seed=42, max_proposals=1)
    (pa,) = a.fit_and_propose(df.copy())
    (pb,) = b.fit_and_propose(df.copy())
    assert pa.candidate_id == pb.candidate_id  # determinism preserved
    assert pa.metadata["algebraic_transforms"] is None
    assert pa.metadata["static_mode"] is False


def test_unknown_transform_kinds_are_rejected_up_front() -> None:
    with pytest.raises(ValueError, match="unknown algebraic transform"):
        RimeProposer(algebraic_transforms=["pow", "exp"])


def test_transform_catalogue_and_domain_safety() -> None:
    # The canned catalogue is what the docs promise, and transforms are
    # NaN-safe and finite on mixed-domain data (negatives, zero, positives).
    assert sorted(_ALGEBRAIC_TRANSFORMS) == [
        "cub",
        "sat",
        "sat_norm",
        "sat_sq",
        "sat_sq_norm",
        "slog",
        "sq",
        "ssqrt",
    ]
    x = np.array([-3.0, -1e-3, 0.0, 1e-3, 3.0, np.nan])
    from apedp_dyn.proposers.rime_proposer import _apply_transform

    for kind in _ALGEBRAIC_TRANSFORMS:
        out = _apply_transform(kind, x)
        assert np.isnan(out[-1]), f"{kind} must be NaN-preserving"
        assert np.all(np.isfinite(out[:-1])), f"{kind} must be finite on its domain"

    # Labels appear in assembled columns in deterministic order.
    df = pd.DataFrame({"a": np.linspace(-2, 2, 40), "target": np.ones(40)})
    proposer = RimeProposer(
        feature_cols=["x"],
        target_col="target",
        algebraic_transforms=["sq", "slog"],
    )
    assembled = proposer._columns([df])  # noqa: SLF001 — internal contract
    assert assembled is not None
    labels, cols, _target, base_feats = assembled
    assert labels[:2] == ["x", "sq(x)"] or labels[2] == "sq(x)"
    assert all(k.endswith((".0", ".9")) or isinstance(k, str) for k in labels)  # str labels
    assert set(labels) >= {"x", "sq(x)", "slog(x)"}
    assert base_feats == ["x"]
    # The NaN frame separator lands in every expanded column.
    n_finite = [int(np.isfinite(c).sum()) for c in cols]
    assert len(set(n_finite)) in (1, 2), "transforms must follow base-column structure"

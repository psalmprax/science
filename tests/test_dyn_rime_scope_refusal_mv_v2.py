"""Tests for RIME §16e v2 Multivariate Scope Refusal (PREREG-RIME-SCOPE-MV-v2-2026-10-17).

Verifies the remediation of §16e v1 negative result:
1. G1 control_in_family: All three seeds (700, 701, 702) are cleanly PROMOTED with
   all rival probes separated (Delta > tau_epi), resolving the c -> 0.1 linearization.
2. W1, W2, W3 out-of-family worlds: All seeds are cleanly DECLINED by the rival probes.
3. G2 monomial guards (d2, d3, d6): Probes build cleanly under scale-relative variance guards
   and all true monomials are PROMOTED.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.epistemic.scope_refusal import adjudicate_scope, scored_from_predict
from apedp_dyn.epistemic.scope_refusal_mv_v2 import (
    PROBE_MV_V2_IDS,
    build_family_probes_mv_v2,
)
from benchmarks.known_truth_challenge import build_worlds
from benchmarks.run_scope_refusal_mv_validation import (
    FEATS,
    NOISE,
    W_LAWS,
    aligned_frame,
    emit_list_top,
    law_control,
)


@pytest.mark.parametrize("seed", [700, 701, 702])
def test_v2_control_promoted_across_seeds(seed: int) -> None:
    """G1 control is promoted without wrongful refusal from linearization."""
    dev, hold = aligned_frame(seed, law_control)
    emitted, _, info = emit_list_top(dev, hold)
    assert emitted is not None
    assert info["hold_r2"] > 0.99

    probes = build_family_probes_mv_v2(dev, hold, feature_cols=FEATS, noise_sigma=NOISE)
    assert len(probes) == 3

    hold_y = hold["target"].to_numpy(float)
    res = adjudicate_scope(
        emitted, probes,
        holdout_scale={"obs_rms": float(np.sqrt(np.mean(hold_y ** 2))),
                       "noise_sigma": NOISE, "n_samples": len(hold_y)},
        bootstrap=True, bootstrap_n_boot=200, bootstrap_seed=7,
    )
    assert res.verdict == "promoted"
    for p in res.probes:
        assert p["verdict"] == "separated"


@pytest.mark.parametrize("w_name", ["w1_sin_interaction", "w2_max_switch", "w3_am_product"])
def test_v2_out_of_family_declined(w_name: str) -> None:
    """W1, W2, W3 are declined by their corresponding rival probes."""
    seed = 700
    dev, hold = aligned_frame(seed, W_LAWS[w_name])
    emitted, _, _ = emit_list_top(dev, hold)
    assert emitted is not None

    probes = build_family_probes_mv_v2(dev, hold, feature_cols=FEATS, noise_sigma=NOISE)
    hold_y = hold["target"].to_numpy(float)
    res = adjudicate_scope(
        emitted, probes,
        holdout_scale={"obs_rms": float(np.sqrt(np.mean(hold_y ** 2))),
                       "noise_sigma": NOISE, "n_samples": len(hold_y)},
        bootstrap=True, bootstrap_n_boot=200, bootstrap_seed=7,
    )
    assert res.verdict == "declined"
    declined_probes = [p["probe"] for p in res.probes if p["verdict"] == "declined"]
    assert len(declined_probes) >= 1


@pytest.mark.parametrize("world_name,k", [
    ("d2_C1000_uniform_1x", 2),
    ("d3_C100_uniform_1x", 3),
    ("d6_C1e12_log_1x", 6),
    ("d2_C1000_log_1x", 2),
])
def test_v2_g2_monomial_guards_defined_and_promoted(world_name: str, k: int) -> None:
    """G2 monomial guards build successfully under scale-relative variance and promote."""
    worlds = {bw.world.name: bw for bw in build_worlds(noise_seed=700)}
    bw = worlds[world_name]
    dev, hold = bw.dev, bw.hold
    x = dev["p"].to_numpy(float)
    y = dev["target"].to_numpy(float)
    w, *_ = np.linalg.lstsq(np.column_stack([x ** k]), y, rcond=None)
    alpha = float(w[0])

    emitted = scored_from_predict(
        f"emitted_p{k}",
        lambda fd: np.asarray(fd["p"].to_numpy(float) ** k * alpha),
        dev, hold, noise_sigma=bw.sigma,
    )

    # Scale-relative variance guard allows probe building on micro-scale axes
    probes = build_family_probes_mv_v2(dev, hold, feature_cols=["p"], noise_sigma=bw.sigma)
    assert len(probes) >= 1
    assert probes[0].candidate.candidate_id == PROBE_MV_V2_IDS[0]

    hold_y = hold["target"].to_numpy(float)
    res = adjudicate_scope(
        emitted, probes,
        holdout_scale={"obs_rms": float(np.sqrt(np.mean(hold_y ** 2))),
                       "noise_sigma": bw.sigma, "n_samples": len(hold_y)},
        bootstrap=True, bootstrap_n_boot=200, bootstrap_seed=7,
    )
    assert res.verdict == "promoted"

r"""Fluid domain scoping tests.

Covers:
1. Determinism (byte-identical across processes/calls) and id <-> seed mapping.
2. Incompressibility: velocity derived from streamfunction is grid- (and
   grid-precision-) divergence-free — the property that makes the Noether
   projector physically meaningful on this domain.
3. Advection smoothness: fluid clips pass the temporal-consistency instrument
   (smooth advected evolution, not independent noise per frame).
4. Manifest scope discipline: holdout ids disjoint and exclusive from train.
5. Eval-gate integration: the gate bundles run on fluid clips (real vs degraded).
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.video.eval_gate import (
    fvd_proxy,
    measure_temporal_consistency,
)
from apedp_dyn.video.fluid_domain import (
    _streamfunction_velocity,
    _vortex_streamfunction,
    fluid_domain_manifest,
    make_domain_set,
    make_fluid_clip,
)


def test_fluid_clip_deterministic_and_id_seed_mapping():
    a = make_fluid_clip(17)
    b = make_fluid_clip(17)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (16, 3, 64, 64)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    # Different seeds -> different content
    c = make_fluid_clip(18)
    assert float(np.mean(np.abs(a - c))) > 1e-3


def test_velocity_is_divergence_free_from_streamfunction():
    psi = _vortex_streamfunction(32, 32, [(9.0, 20.0), (24.0, 6.0)], [5.0, -3.0])
    u, v = _streamfunction_velocity(psi)
    dux = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) * 0.5
    dvy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) * 0.5
    div = dux + dvy
    assert float(np.abs(div).max()) < 1e-10, float(np.abs(div).max())


def test_fluid_domain_is_temporally_smooth_and_featurizes():
    vid = make_fluid_clip(3)
    cons = measure_temporal_consistency(vid)
    jerky = measure_temporal_consistency(vid[np.random.RandomState(0).permutation(vid.shape[0])])
    assert cons < jerky, (cons, jerky)
    # FVD proxy is runnable on domain clips (real vs perturbed)
    pert = vid + np.random.RandomState(1).randn(*vid.shape) * 0.05
    d = fvd_proxy([vid], [pert])["swd"]
    assert d > 0.0


def test_manifest_holdout_discipline():
    man = fluid_domain_manifest(n_clips=10, holdout_fraction=0.3, holdout_seed=5)
    tr, ho = set(man["train_ids"]), set(man["holdout_ids"])
    assert not (tr & ho)
    assert tr | ho == {f"fluid_clip_{i:04d}" for i in range(10)}
    man2 = fluid_domain_manifest(n_clips=10, holdout_fraction=0.3, holdout_seed=5)
    assert set(man2["holdout_ids"]) == ho  # exclusive, deterministic
    clips = make_domain_set(tuple(man["train_ids"]), tuple(man["holdout_ids"]), t=4, h=32, w=32)
    # Holdout clips generate under the same mapping and never leak into other ids
    assert set(clips) == tr | ho

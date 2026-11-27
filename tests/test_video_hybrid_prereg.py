"""Pins for PREREG-VIDEO-HYBRID-2026-11 (scripts/run_prereg_video_hybrid.py)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "packages" / "jetflow-video")):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location("run_prereg_video_hybrid", ROOT / "scripts" / "run_prereg_video_hybrid.py")
assert _spec is not None and _spec.loader is not None
rh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rh)


def test_the_size_is_set_on_the_comparison_module() -> None:
    rh.set_size()
    assert (rh.h.FRAMES, rh.h.SIZE, rh.h.JETFLOW["hidden_dim"]) == (8, 32, 64)
    train, held = rh._toy("collision", "dev")
    assert train.shape[1:] == (8, 3, 32, 32) and len(train) == 400 and len(held) == 100


def test_every_arm_builds_trains_and_samples_at_the_larger_size() -> None:
    rh.set_size()
    train, held = rh._toy("fluid", "dev")
    latents = rh.h.encode(train[:8])
    for arm in rh.ARMS:
        torch.manual_seed(0)
        model = rh.h.build(arm)
        trainer = rh.h.train_meanflow if arm.startswith("meanflow") else rh.h.train_flow_matching
        trainer(model, latents, 2.0, lr=1e-3, seed=0)
        how, steps = rh.GRID[arm][0]
        row = rh.h.evaluate(model, how, steps, 4, 1, train[:16], held[:8])
        assert np.isfinite(row["frechet_pca"]) and row["calls_per_clip"] == steps


def test_toy_ranges_and_ucf_groups_do_not_overlap() -> None:
    final, dev = set(range(500, 1000)), set(range(1000, 1500))
    assert not final & dev
    (ftr, fhe), (dtr, dhe) = rh.UCF_GROUPS["final"], rh.UCF_GROUPS["dev"]
    assert not set(ftr) & set(fhe) and not set(dtr) & set(dhe)
    assert not (set(dtr) | set(dhe)) & set(fhe)       # development never reads the final held-out groups


def test_beats_needs_the_margin_and_two_seeds() -> None:
    assert rh.beats([0.8, 0.8, 0.8], [1.0, 1.0, 1.0])["held"]
    assert not rh.beats([0.95, 0.95, 0.95], [1.0, 1.0, 1.0])["held"]
    assert not rh.beats([0.1, 1.2, 1.2], [1.0, 1.0, 1.0])["held"]      # mean passes, one seed lower


def test_each_expectation_reads_on_its_own() -> None:
    def job(arm: str, seed: int, fd: float) -> dict:
        return {"samples": [{"sampler": s, "steps": k, "frechet_pca": fd, "temporal_roughness": 0.05}
                            for s, k in rh.GRID[arm]], "floor_frechet_pca": 1.0}
    jobs = {(d, a, s): job(a, s, 1.0 if a == "meanflow-jetflow" else 2.0) for d in rh.DATA for a in rh.ARMS
            for s in rh.SEEDS}
    del jobs[("ucf101", "fm-unet", 1)]
    out = rh.expectations(jobs)
    assert out["fluid"]["H1 the hybrid beats MeanFlow on the UNet at one call"]["held"] is True
    assert out["ucf101"]["H3 the hybrid at four calls beats plain flow matching at four calls"]["held"] is None
    assert out["ucf101"]["H1 the hybrid beats MeanFlow on the UNet at one call"]["held"] is True


def test_the_final_mode_refuses_a_changed_plan(tmp_path: Path) -> None:
    if not rh.DIGEST.exists():
        pytest.skip("not frozen yet")
    rh.check_frozen()

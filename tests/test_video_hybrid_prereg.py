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


def test_every_arm_and_sampler_of_the_grid_builds_trains_and_samples_at_the_larger_size() -> None:
    rh.set_size()
    train, held = rh._toy("fluid", "dev")
    latents = rh.h.encode(train[:8])
    for arm in rh.ARMS:
        torch.manual_seed(0)
        model = rh.h.build(arm)
        trainer = rh.h.train_meanflow if arm.startswith("meanflow") else rh.h.train_flow_matching
        trainer(model, latents, 2.0, lr=1e-3, seed=0)
        for how, steps in rh.GRID[arm]:
            row = rh._sampler_row(model, how, steps, 4, 1, train[:16], held[:8])
            calls = 2 * steps if how == "jetflow-sampler" else steps
            assert row["frechet_pca"] is not None and not row["non_finite"], (arm, how, steps)
            assert row["calls_per_clip"] == calls, (arm, how, steps, row["calls_per_clip"])


def test_toy_ranges_and_ucf_groups_do_not_overlap() -> None:
    (fbase, fseed), (dbase, dseed) = rh.TOY["final"], rh.TOY["dev"]
    final, dev = set(range(fbase, fbase + 500)), set(range(dbase, dbase + 500))
    assert not final & dev and fseed != dseed
    first_test = set(range(0, 500)) | set(range(3000, 3500))          # the clips RESULT-VIDEO-FLOW-2026-11 used
    assert not (final | dev) & first_test
    (ftr, fhe), (dtr, dhe) = rh.UCF_GROUPS["final"], rh.UCF_GROUPS["dev"]
    assert not set(ftr) & set(fhe) and not set(dtr) & set(dhe)
    assert not (set(dtr) | set(dhe)) & set(fhe)       # development never reads the final held-out groups


def test_beats_needs_the_margin_and_two_seeds() -> None:
    assert rh.beats([0.8, 0.8, 0.8], [1.0, 1.0, 1.0])["held"]
    assert not rh.beats([0.95, 0.95, 0.95], [1.0, 1.0, 1.0])["held"]
    assert not rh.beats([0.1, 1.2, 1.2], [1.0, 1.0, 1.0])["held"]      # mean passes, one seed lower


def test_a_breakdown_of_the_hybrid_is_a_loss_and_one_of_a_comparator_is_a_lost_seed() -> None:
    inf = float("inf")
    assert not rh.beats([0.5, 0.5, inf], [1.0, 1.0, 1.0])["held"]     # the hybrid's blown-up seed is not excused
    assert not rh.no_worse([0.5, inf, 0.5], [1.0, 1.0, 1.0])["held"]
    assert not rh.rougher([0.05, inf, 0.05], [0.05, 0.05, 0.05])["held"]
    # The reviewer's case: a comparator's breakdown does not hand the hybrid a win it did not earn.
    assert not rh.beats([100.0, 50.0, 60.0], [40.0, 60.0, inf])["held"]
    assert not rh.rougher([0.2, 0.2, 0.2], [0.05, 0.05, inf])["held"]
    assert rh.beats([0.5, 0.5, 0.5], [1.0, 1.0, inf])["seeds_compared"] == 2
    with pytest.raises(rh.NotRead):
        rh.beats([0.5, 0.5, 0.5], [1.0, inf, inf])                  # one comparable seed: not read
    with pytest.raises(rh.NotRead):
        rh.rougher([0.05, 0.05, 0.05], [inf, inf, 0.05])

    rh.set_size()
    train, held = rh._toy("fluid", "dev")
    for arm in ("meanflow-jetflow", "fm-unet"):          # weights gone to NaN, as when training blows up
        torch.manual_seed(0)
        model = rh.h.build(arm)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float("nan"))
        how, steps = rh.GRID[arm][0]
        row = rh._sampler_row(model, how, steps, 4, 0, train[:16], held[:8])
        assert row["non_finite"] and row["frechet_pca"] is None, row


def _jobs(fd_hybrid: float, mode: str = "dev") -> dict:
    def job(d: str, a: str, s: int, fd: float) -> dict:
        return {"data": d, "arm": a, "seed": s, "mode": mode, "complete": True, "floor_frechet_pca": 1.0,
                "samples": [{"sampler": h, "steps": k, "frechet_pca": fd, "temporal_roughness": 0.05}
                            for h, k in rh.GRID[a]]}
    return {(d, a, s): job(d, a, s, fd_hybrid if a == "meanflow-jetflow" else 2.0) for d in rh.DATA for a in rh.ARMS
            for s in rh.SEEDS}


def test_each_expectation_reads_on_its_own_and_the_outcome_follows_the_order() -> None:
    jobs = _jobs(1.0)
    del jobs[("ucf101", "fm-unet", 1)]
    out = rh.expectations(jobs)
    assert out["fluid"]["H1"]["held"] is True and out["fluid"]["H2"]["jetflow_best"] == "euler 1"
    assert out["ucf101"]["H3"]["held"] is None and out["ucf101"]["H4-4"]["held"] is None
    assert out["ucf101"]["H1"]["held"] is True
    told = rh.outcome(out)["proposal"]
    assert "H4-1 not read" in told and "H3 not read" in told                   # not read is not "does not hold"
    whole = rh.outcome(rh.expectations(_jobs(1.0)))
    assert whole["rules"] == [3, 4] and "few-step model at one call" in whole["proposal"]
    assert "model at four calls" in whole["proposal"]
    assert rh.outcome(rh.expectations(_jobs(3.0)))["rules"] == [1]             # the hybrid loses: listed as retired
    jobs = _jobs(1.0)
    del jobs[("ucf101", "meanflow-unet", 0)]
    assert rh.outcome(rh.expectations(jobs))["rules"] == [2, 4]                # H1 not read on ucf101
    # The reviewer's case: H2 read and failed on ucf101 while H1 is not read retires the hybrid all the same.
    jobs = _jobs(1.0)
    del jobs[("ucf101", "meanflow-unet", 0)]
    for s in rh.SEEDS:
        for row in jobs[("ucf101", "fm-jetflow", s)]["samples"]:
            row["frechet_pca"] = 0.5
    assert rh.outcome(rh.expectations(jobs))["rules"] == [1]
    jobs = _jobs(1.0)
    for s in rh.SEEDS:                                                        # flicker at one call only
        for row in jobs[("ucf101", "meanflow-jetflow", s)]["samples"]:
            if row["steps"] == 1:
                row["temporal_roughness"] = 0.2
    told = rh.outcome(rh.expectations(jobs))["proposal"]
    assert "H4-1 does not hold" in told and "model at four calls" in told


def test_euler_32_is_the_last_sampler_of_every_grid() -> None:
    assert all(grid[-1] == ("euler", 32) for grid in rh.GRID.values())


def test_only_numerical_errors_count_as_a_breakdown() -> None:
    rh.set_size()
    torch.manual_seed(0)
    model = rh.h.build("fm-unet")
    assert rh._breakdown(np.linalg.LinAlgError("Eigenvalues did not converge"), model)
    assert rh._breakdown(RuntimeError("linalg.eigh: the algorithm failed to converge"), model)
    assert not rh._breakdown(RuntimeError("DefaultCPUAllocator: can't allocate memory"), model)
    assert not rh._breakdown(ValueError("shapes (3,) and (4,) not aligned"), model)


def test_collect_checks_each_final_job() -> None:
    rec = _jobs(1.0, "final")[("fluid", "fm-unet", 0)]
    assert "frozen digests differ" in rh.check_job(rec, "final")
    rec = {**rec, "frozen": rh.frozen_digests(), "seconds_budget": rh.SECONDS, "n_gen": rh.N_GEN,
           "train_clips": 400, "held_clips": 100, "frames": rh.FRAMES, "size": rh.SIZE,
           "jetflow_hidden": rh.HIDDEN, "threads": rh.THREADS, "loaded": rh._loaded_digests()}
    assert rh.check_job(rec, "final") == []
    assert rh.check_job({**rec, "seconds_budget": 240.0}, "final")
    assert rh.check_job({**rec, "threads": 2}, "final") == ["clip size, width or threads"]
    assert rh.check_job({**rec, "complete": False}, "final") == [rh.PARTIAL]
    assert rh.check_job(rec, "dev") == ["mode final"]
    changed = {**rec, "loaded": {**rec["loaded"], "scripts/run_prereg_video_hybrid.py": "0" * 64}}
    assert any("loaded files" in p for p in rh.check_job(changed, "final"))


def test_collect_rejects_duplicates_and_mixed_commits(tmp_path: Path) -> None:
    import json

    base = dict(_jobs(1.0)[("fluid", "fm-unet", 0)])
    base.update(frames=rh.FRAMES, size=rh.SIZE, jetflow_hidden=rh.HIDDEN, threads=rh.THREADS, commit="a")
    (tmp_path / "x").mkdir()
    (tmp_path / "job_fluid_fm-unet_s0.json").write_text(json.dumps(base))
    (tmp_path / "x" / "job_fluid_fm-unet_s0.json").write_text(json.dumps(base))
    rh.collect(rh.argparse.Namespace(mode="dev", out=tmp_path))
    got = json.loads((tmp_path / "results.json").read_text())
    assert "more than one record for this job" in got["rejected_or_partial_jobs"]["fluid/fm-unet/0"]


def test_the_final_mode_refuses_a_changed_plan(tmp_path: Path) -> None:
    if not rh.DIGEST.exists():
        pytest.skip("not frozen yet")
    rh.check_frozen()

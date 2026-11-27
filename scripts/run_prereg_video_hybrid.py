#!/usr/bin/env python3
"""PREREG-VIDEO-HYBRID-2026-11: MeanFlow on JetFlow's network against its two parents, larger, once.

Status: audit-only

The larger test RESULT-VIDEO-FLOW-2026-11 proposed for its hybrid, on the owner's word ("go ahead to test
all", 27 November 02:32). The models and samplers are those of
``packages/jetflow-video/jetflow_video/head_to_head.py`` (frozen by DIGEST-VIDEO-FLOW and not edited); this
runner sets that module's clip size before it builds anything: 8 frames of 32 x 32 instead of 6 of 16, and
JetFlow's network 64 wide instead of 48 (the UNet is matched to it, as before). Training is 2,700 seconds.

Data: ``fluid`` and ``collision`` (toy worlds; ``final`` clips 500 to 999, ``dev`` 1000 to 1499, each split
80/20 by the repository's manifest split with seeds 15 and 16; neither range was used by the earlier test)
and ``ucf101`` (the UCF101 subset, both of its splits pooled and split by group, so no actor or scene is in
both: ``final`` trains on groups 1 to 19 and scores on groups 20 to 25; ``dev`` trains on 1 to 15 and scores on
16 to 19, never reading 20 to 25). ``--mode final`` checks the digest first.

    python scripts/run_prereg_video_hybrid.py job --data ucf101 --arm meanflow-jetflow --seed 0 --mode final --out OUT
    python scripts/run_prereg_video_hybrid.py collect --mode final --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "packages" / "jetflow-video")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from jetflow_video import head_to_head as h  # noqa: E402

PREREGISTRATION_ID = "PREREG-VIDEO-HYBRID-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-VIDEO-HYBRID.json"
FROZEN = ("preregistration/PREREG-VIDEO-HYBRID-2026-11.md", "scripts/run_prereg_video_hybrid.py",
          ".github/workflows/video-hybrid.yml", "tests/test_video_hybrid_prereg.py",
          "packages/jetflow-video/jetflow_video/head_to_head.py",
          *(f"packages/jetflow-video/jetflow_video/{m}.py" for m in ("__init__", "model", "stepper", "noether", "physics",
                                                                    "diffusion_baseline", "bridge", "tensor_train",
                                                                    "text_encoder", "depth_se3", "trainer")),
          "apedp_dyn/video/fluid_domain.py", "apedp_dyn/video/collision_domain.py", "apedp_dyn/video/eval_gate.py")
FRAMES, SIZE, HIDDEN = 8, 32, 64
DATA = ("fluid", "collision", "ucf101")
ARMS = ("meanflow-jetflow", "meanflow-unet", "fm-jetflow", "fm-unet")   #: the hybrid, its two parents, the reference
SEEDS = (0, 1, 2)
GRID: dict[str, tuple[tuple[str, int], ...]] = {
    "meanflow-jetflow": (("meanflow", 1), ("meanflow", 2), ("meanflow", 4), ("euler", 32)),
    "meanflow-unet": (("meanflow", 1), ("meanflow", 2), ("meanflow", 4), ("euler", 32)),
    "fm-jetflow": (("euler", 1), ("euler", 4), ("euler", 32), ("jetflow-sampler", 2)),
    "fm-unet": (("euler", 1), ("euler", 4), ("euler", 32)),
}
UCF_REPO, UCF_FILE = "sayakpaul/ucf101-subset", "UCF101_subset.tar.gz"
UCF_SHA256 = "e9fcc76af48d320be88c5265f2e0576ecd615956976f6ce4742fdf2b042b71eb"   #: as pinned by the earlier test
UCF_GROUPS = {"final": (range(1, 20), range(20, 26)), "dev": (range(1, 16), range(16, 20))}
TOY = {"final": (500, 15), "dev": (1000, 16)}
MARGIN, SEEDS_NEEDED = 0.9, 2     #: "A beats B": mean FD of A at most 0.9 of B's, and lower on 2 of 3 seeds
ROUGHNESS_LIMIT = 1.5             #: the hybrid's roughness at four calls at most 1.5 times flow matching's


def set_size() -> None:
    """The comparison module's clip size and JetFlow's width for this test (module globals, read at call time)."""
    h.FRAMES, h.SIZE = FRAMES, SIZE
    h.JETFLOW = {**h.JETFLOW, "hidden_dim": HIDDEN}


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")


def _toy(kind: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    from apedp_dyn.video.eval_gate import held_out_split

    base, seed = TOY[mode]
    ids = [str(base + i) for i in range(500)]
    train, held = held_out_split(ids, holdout_fraction=0.2, seed=seed)
    return (h.toy_clips(kind, [int(i) for i in train]), h.toy_clips(kind, [int(i) for i in held]))


def _group(name: str) -> int:
    m = re.search(r"_g(\d+)_", name)
    if m is None:
        raise ValueError(f"no group in {name}")
    return int(m.group(1))


def _ucf(mode: str, cache: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    archive = Path(hf_hub_download(UCF_REPO, UCF_FILE, repo_type="dataset", cache_dir=str(cache / "hub")))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != UCF_SHA256:
        raise SystemExit(f"{UCF_FILE} does not match its pinned SHA-256")
    root = cache / "ucf101"
    if not root.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
    base = next(p.parent for p in root.rglob("train") if p.is_dir())
    clips, names = [], []
    for split in ("train", "test"):
        c, n = h.video_folder(base, split)
        clips.append(c)
        names += n
    allc = np.concatenate(clips)
    groups = np.array([_group(n) for n in names])
    train_g, held_g = UCF_GROUPS[mode]
    tr = np.isin(groups, list(train_g))
    he = np.isin(groups, list(held_g))
    info = {"archive_sha256": digest, "train_groups": sorted(set(groups[tr].tolist())),
            "held_groups": sorted(set(groups[he].tolist())),
            "train_videos": len({n for n, k in zip(names, tr, strict=True) if k}),
            "held_videos": len({n for n, k in zip(names, he, strict=True) if k})}
    return allc[tr], allc[he], info


def data(kind: str, mode: str, cache: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if kind == "ucf101":
        return _ucf(mode, cache)
    train, held = _toy(kind, mode)
    return train, held, {}


def run_job(args: argparse.Namespace) -> int:
    if args.mode == "final":
        check_frozen()
    set_size()
    torch.set_num_threads(args.threads)
    train, held, info = data(args.data, args.mode, args.cache)
    latents = h.encode(train)
    torch.manual_seed(args.seed)
    model = h.build(args.arm)
    trainer = h.train_meanflow if args.arm.startswith("meanflow") else h.train_flow_matching
    fit = trainer(model, latents, args.seconds, lr=1e-3, seed=args.seed)
    print(f"{args.data} {args.arm} seed {args.seed}: {fit}", flush=True)
    rows = []
    for how, steps in GRID[args.arm]:
        row = h.evaluate(model, how, steps, args.n_gen, 20_000 + args.seed, train, held)
        rows.append(row)
        print(json.dumps(row), flush=True)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(train), size=min(args.n_gen, len(train)), replace=False)
    record: dict[str, Any] = {
        "data": args.data, "arm": args.arm, "seed": args.seed, "mode": args.mode, "training": fit,
        "parameters": sum(p.numel() for p in model.parameters()), "n_gen": args.n_gen,
        "frames": FRAMES, "size": SIZE, "jetflow_hidden": HIDDEN,
        "train_clips": len(train), "held_clips": len(held), "data_info": info, "samples": rows,
        "floor_frechet_pca": round(h.frechet_pca(train, held, train[pick]), 4),
        "held_temporal_roughness": round(h.temporal_roughness(held), 5),
        "threads": args.threads, "torch": torch.__version__, "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "cpu": platform.processor() or _cpu_model(),
        "commit": os.environ.get("GITHUB_SHA"), "numpy": np.__version__}
    if args.mode == "final":
        record["frozen"] = frozen_digests()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"job_{args.data}_{args.arm}_s{args.seed}.json").write_text(json.dumps(record, indent=1) + "\n")
    return 0


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _values(jobs: dict[tuple[str, str, int], dict[str, Any]], kind: str, arm: str, how: str, steps: int,
            field: str = "frechet_pca") -> list[float]:
    out = []
    for seed in SEEDS:
        row = next(r for r in jobs[(kind, arm, seed)]["samples"] if r["sampler"] == how and r["steps"] == steps)
        out.append(float(row[field]))
    return out


def beats(a: list[float], b: list[float], margin: float = MARGIN) -> dict[str, Any]:
    wins = sum(x < y for x, y in zip(a, b, strict=True))
    held = float(np.mean(a)) <= margin * float(np.mean(b)) and wins >= SEEDS_NEEDED
    return {"held": held, "mean_a": round(float(np.mean(a)), 4), "mean_b": round(float(np.mean(b)), 4),
            "seeds_lower": wins}


def _read(fn: Any) -> dict[str, Any]:
    try:
        out: dict[str, Any] = fn()
        return out
    except (KeyError, StopIteration) as missing:
        return {"held": None, "not_read": f"job or sampler missing: {missing}"}


def expectations(jobs: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in DATA:
        def fd(arm: str, how: str, steps: int, kind: str = kind) -> list[float]:
            return _values(jobs, kind, arm, how, steps)

        def rough(kind: str = kind) -> dict[str, Any]:
            a = _values(jobs, kind, "meanflow-jetflow", "meanflow", 4, "temporal_roughness")
            b = _values(jobs, kind, "fm-unet", "euler", 4, "temporal_roughness")
            return {"held": float(np.mean(a)) <= ROUGHNESS_LIMIT * float(np.mean(b)),
                    "mean_hybrid": round(float(np.mean(a)), 5), "mean_flow_matching": round(float(np.mean(b)), 5)}

        def floor(kind: str = kind) -> dict[str, Any]:
            vals = [jobs[(kind, a, s)]["floor_frechet_pca"] for a in ARMS for s in SEEDS if (kind, a, s) in jobs]
            if not vals:
                raise KeyError(kind)
            return {"value": round(float(np.mean(vals)), 4)}
        out[kind] = {
            "H1 the hybrid beats MeanFlow on the UNet at one call":
                _read(lambda: beats(fd("meanflow-jetflow", "meanflow", 1), fd("meanflow-unet", "meanflow", 1))),
            "H2 the hybrid beats JetFlow's flow matching at one call":
                _read(lambda: beats(fd("meanflow-jetflow", "meanflow", 1), fd("fm-jetflow", "euler", 1))),
            "H3 the hybrid at four calls beats plain flow matching at four calls":
                _read(lambda: beats(fd("meanflow-jetflow", "meanflow", 4), fd("fm-unet", "euler", 4))),
            "H4 the hybrid's clips at four calls are at most 1.5 times as rough as flow matching's": _read(rough),
            "floor_mean": _read(floor),
        }
    return out


def collect(args: argparse.Namespace) -> int:
    if args.mode == "final":
        check_frozen()
    jobs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for p in sorted(args.out.glob("job_*.json")):
        rec = json.loads(p.read_text())
        jobs[(rec["data"], rec["arm"], int(rec["seed"]))] = rec
    missing = [f"{d}/{a}/{s}" for d in DATA for a in ARMS for s in SEEDS if (d, a, s) not in jobs]
    record: dict[str, Any] = {"preregistration": PREREGISTRATION_ID, "mode": args.mode, "missing_jobs": missing,
                              "jobs": [jobs[k] for k in sorted(jobs)]}
    if jobs:
        record["expectations"] = expectations(jobs)
    if args.mode == "final":
        record["frozen"] = frozen_digests()
    (args.out / "results.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k in ("missing_jobs", "expectations")}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("job", "collect"))
    parser.add_argument("--data", choices=DATA)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=0)
    parser.add_argument("--mode", choices=("dev", "final"), required=True)
    parser.add_argument("--seconds", type=float, default=2700.0)
    parser.add_argument("--n-gen", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=Path("video_cache"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "job":
        if not (args.data and args.arm):
            parser.error("--data and --arm are required")
        return run_job(args)
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())

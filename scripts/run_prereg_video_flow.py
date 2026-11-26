#!/usr/bin/env python3
"""PREREG-VIDEO-FLOW-2026-11: JetFlow, plain flow matching and MeanFlow at equal training time, once.

Status: audit-only

``job`` trains one arm of ``packages/jetflow-video/jetflow_video/head_to_head.py`` on one data set with
one seed for ``--seconds``, then generates ``--n-gen`` clips with each sampler of its grid and scores them
against the held-out real clips. ``collect`` reads every job, adds each data set's floor (the training
clips' own distance to the held-out ones) and applies the expectations of
``preregistration/PREREG-VIDEO-FLOW-2026-11.md``.

Data: ``fluid`` and ``collision`` (the repository's toy worlds, 16 x 16, six frames; ``final`` uses the
manifest split of clips 0 to 499, ``dev`` clips 3000 to 3499 split the same way with another seed) and
``ucf101`` (real footage: the UCF101 subset on the Hugging Face Hub, ten action classes, every third
frame, resized to 16 x 16; ``final`` trains on its train split and scores on its test split, ``dev``
uses the train split only, holding out groups 8 to 10). ``--mode final`` checks the digest first.

    python scripts/run_prereg_video_flow.py job --data ucf101 --arm fm-jetflow --seed 0 --mode final --out OUT
    python scripts/run_prereg_video_flow.py collect --mode final --out OUT
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

PREREGISTRATION_ID = "PREREG-VIDEO-FLOW-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-VIDEO-FLOW.json"
FROZEN = ("preregistration/PREREG-VIDEO-FLOW-2026-11.md", "scripts/run_prereg_video_flow.py",
          "packages/jetflow-video/jetflow_video/head_to_head.py", ".github/workflows/video-flow.yml",
          "tests/test_video_flow_head_to_head.py",
          *(f"packages/jetflow-video/jetflow_video/{m}.py" for m in ("model", "stepper", "noether", "physics",
                                                                    "diffusion_baseline", "bridge", "tensor_train")),
          "apedp_dyn/video/fluid_domain.py", "apedp_dyn/video/collision_domain.py", "apedp_dyn/video/eval_gate.py")
DATA = ("fluid", "collision", "ucf101")
ARMS = ("fm-jetflow", "fm-unet", "meanflow-unet", "meanflow-jetflow")   #: the last is a hybrid
SEEDS = (0, 1, 2)
GRID: dict[str, tuple[tuple[str, int], ...]] = {
    "fm-jetflow": (("euler", 1), ("euler", 2), ("euler", 4), ("euler", 8), ("euler", 32),
                   ("jetflow-sampler", 1), ("jetflow-sampler", 2), ("jetflow-sampler", 4),
                   ("jetflow-sampler-corrected", 1), ("jetflow-sampler-corrected", 2), ("jetflow-sampler-corrected", 4)),
    "fm-unet": (("euler", 1), ("euler", 2), ("euler", 4), ("euler", 8), ("euler", 32),
                ("jet2", 2), ("jet2-corrected", 1), ("jet2-corrected", 2), ("jet2-corrected", 4),
                ("jetflow-sampler", 2), ("jetflow-sampler-corrected", 2)),                  # the last two: hybrids
    "meanflow-unet": (("meanflow", 1), ("meanflow", 2), ("meanflow", 4), ("euler", 8), ("euler", 32),
                      ("jet2-corrected", 2)),                                             # the last: a hybrid
    "meanflow-jetflow": (("meanflow", 1), ("meanflow", 2), ("meanflow", 4), ("euler", 8), ("euler", 32)),
}
UCF_REPO, UCF_FILE = "sayakpaul/ucf101-subset", "UCF101_subset.tar.gz"
UCF_SHA256: str | None = "e9fcc76af48d320be88c5265f2e0576ecd615956976f6ce4742fdf2b042b71eb"   #: from the rehearsal
MARGIN, SEEDS_NEEDED = 0.9, 2            #: "A beats B": mean FD of A at most 0.9 of B's, and lower on 2 of 3 seeds


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")


def _toy(kind: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    from apedp_dyn.video.eval_gate import held_out_split

    base, seed = (0, 13) if mode == "final" else (3000, 14)
    ids = [str(base + i) for i in range(500)]
    train, held = held_out_split(ids, holdout_fraction=0.2, seed=seed)
    return (h.toy_clips(kind, [int(i) for i in train]), h.toy_clips(kind, [int(i) for i in held]))


def _ucf(mode: str, cache: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    archive = Path(hf_hub_download(UCF_REPO, UCF_FILE, repo_type="dataset", cache_dir=str(cache / "hub")))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if UCF_SHA256 is not None and digest != UCF_SHA256:
        raise SystemExit(f"{UCF_FILE} does not match its pinned SHA-256")
    root = cache / "ucf101"
    if not root.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
    base = next(p.parent for p in root.rglob("train") if p.is_dir())
    train, names = h.video_folder(base, "train")
    info: dict[str, Any] = {"archive_sha256": digest, "train_videos": len(set(names))}
    if mode == "final":
        held, held_names = h.video_folder(base, "test")
        info["held_videos"] = len(set(held_names))
        info["train_groups"] = sorted({int(re.search(r"_g(\d+)_", n).group(1)) for n in names})  # type: ignore[union-attr]
        info["held_groups"] = sorted({int(re.search(r"_g(\d+)_", n).group(1)) for n in held_names})  # type: ignore[union-attr]
        return train, held, info
    group = np.array([int(re.search(r"_g(\d+)_", n).group(1)) for n in names])  # type: ignore[union-attr]
    dev_held = (group >= 8) & (group <= 10)
    info.update(dev_held_videos=len({n for n, d in zip(names, dev_held, strict=True) if d}))
    return train[~dev_held], train[dev_held], info


def data(kind: str, mode: str, cache: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if kind == "ucf101":
        return _ucf(mode, cache)
    train, held = _toy(kind, mode)
    return train, held, {}


def run_job(args: argparse.Namespace) -> int:
    if args.mode == "final":
        check_frozen()
    torch.set_num_threads(args.threads)
    train, held, info = data(args.data, args.mode, args.cache)
    latents = h.encode(train)
    model = h.build(args.arm)
    trainer = h.train_meanflow if args.arm.startswith("meanflow") else h.train_flow_matching
    fit = trainer(model, latents, args.seconds, lr=1e-3, seed=args.seed)
    print(f"{args.data} {args.arm} seed {args.seed}: {fit}", flush=True)
    rows = []
    for how, steps in GRID[args.arm]:
        row = h.evaluate(model, how, steps, args.n_gen, 10_000 + args.seed, train, held)
        rows.append(row)
        print(json.dumps(row), flush=True)
    rng = np.random.default_rng(0)                       # the floor's clips: the same for every arm and seed
    pick = rng.choice(len(train), size=min(args.n_gen, len(train)), replace=False)
    record: dict[str, Any] = {
        "data": args.data, "arm": args.arm, "seed": args.seed, "mode": args.mode, "training": fit,
        "parameters": sum(p.numel() for p in model.parameters()), "n_gen": args.n_gen,
        "train_clips": len(train), "held_clips": len(held), "data_info": info, "samples": rows,
        "floor_frechet_pca": round(h.frechet_pca(train, held, train[pick]), 4),
        "held_temporal_roughness": round(h.temporal_roughness(held), 5),
        "threads": args.threads, "torch": torch.__version__, "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "commit": os.environ.get("GITHUB_SHA")}
    if args.mode == "final":
        record["frozen"] = frozen_digests()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"job_{args.data}_{args.arm}_s{args.seed}.json").write_text(json.dumps(record, indent=1) + "\n")
    return 0


def _fd(jobs: dict[tuple[str, str, int], dict[str, Any]], kind: str, arm: str, how: str, steps: int) -> list[float]:
    out = []
    for seed in SEEDS:
        row = next(r for r in jobs[(kind, arm, seed)]["samples"] if r["sampler"] == how and r["steps"] == steps)
        out.append(float(row["frechet_pca"]))
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
    """Each expectation per data set, read on its own: one whose jobs are missing is marked not read."""
    out: dict[str, Any] = {}
    for kind in DATA:
        def fd(arm: str, how: str, steps: int, kind: str = kind) -> list[float]:
            return _fd(jobs, kind, arm, how, steps)

        def e3(kind: str = kind) -> dict[str, Any]:
            mf1, jet = fd("meanflow-unet", "meanflow", 1), fd("fm-jetflow", "jetflow-sampler", 2)
            best_jet = min(float(np.mean(jet)), float(np.mean(fd("fm-jetflow", "jetflow-sampler-corrected", 2))))
            return {"held": float(np.mean(mf1)) <= (1 / MARGIN) * best_jet,
                    "mean_meanflow_1": round(float(np.mean(mf1)), 4), "best_jetflow_4_calls": round(best_jet, 4)}

        def floor(kind: str = kind) -> float:
            vals = [jobs[(kind, a, s)]["floor_frechet_pca"] for a in ARMS for s in SEEDS if (kind, a, s) in jobs]
            if not vals:
                raise KeyError(kind)
            return round(float(np.mean(vals)), 4)
        out[kind] = {
            "E1 JetFlow beats flow matching at four calls, as packaged":
                _read(lambda: beats(fd("fm-jetflow", "jetflow-sampler", 2), fd("fm-unet", "euler", 4))),
            "E1 JetFlow beats flow matching at four calls, its step corrected":
                _read(lambda: beats(fd("fm-jetflow", "jetflow-sampler-corrected", 2), fd("fm-unet", "euler", 4))),
            "E2 MeanFlow beats flow matching at one call":
                _read(lambda: beats(fd("meanflow-unet", "meanflow", 1), fd("fm-unet", "euler", 1))),
            "E3 MeanFlow at one call is no worse than JetFlow at four": _read(e3),
            "E4 the corrected Jet2 step beats Euler at four calls, UNet":
                _read(lambda: beats(fd("fm-unet", "jet2-corrected", 2), fd("fm-unet", "euler", 4))),
            "E4 the corrected Jet2 step beats Euler at four calls, JetFlow":
                _read(lambda: beats(fd("fm-jetflow", "jetflow-sampler-corrected", 2), fd("fm-jetflow", "euler", 4))),
            "E5 JetFlow's network beats the UNet, Euler 32":
                _read(lambda: beats(fd("fm-jetflow", "euler", 32), fd("fm-unet", "euler", 32))),
            "E6a hybrid: MeanFlow helps JetFlow's network at one call":
                _read(lambda: beats(fd("meanflow-jetflow", "meanflow", 1), fd("fm-jetflow", "euler", 1))),
            "E6b hybrid: MeanFlow on JetFlow's network beats MeanFlow on the UNet at one call":
                _read(lambda: beats(fd("meanflow-jetflow", "meanflow", 1), fd("meanflow-unet", "meanflow", 1))),
            "E6c hybrid: the corrected Jet2 step on MeanFlow's velocity beats MeanFlow's own four calls":
                _read(lambda: beats(fd("meanflow-unet", "jet2-corrected", 2), fd("meanflow-unet", "meanflow", 4))),
            "E6d hybrid: JetFlow's corrected sampler on the UNet beats Euler at four calls":
                _read(lambda: beats(fd("fm-unet", "jetflow-sampler-corrected", 2), fd("fm-unet", "euler", 4))),
            "floor_mean": _read(lambda: {"value": floor()}),
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
    parser.add_argument("--seconds", type=float, default=1200.0)
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

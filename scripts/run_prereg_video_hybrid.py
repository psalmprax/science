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
and ``ucf101`` (the UCF101 subset, its three splits pooled and split by group, so no actor or scene is in
both: ``final`` trains on groups 1 to 19 and scores on groups 20 to 25; ``dev`` trains on 1 to 15 and scores on
16 to 19, never reading 20 to 25). ``--mode final`` checks the digest first.

A sampler that raises or gives a non-finite score is a scored outcome, worse than any finite one; only a job
that is absent (or rejected by ``collect``'s checks) is missing. The record is written after training and after
every sampler, so a job stopped part way keeps what it finished.

    python scripts/run_prereg_video_hybrid.py job --data ucf101 --arm meanflow-jetflow --seed 0 --mode final --out OUT
    python scripts/run_prereg_video_hybrid.py collect --mode final --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import tarfile
import time
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
          "apedp_dyn/video/fluid_domain.py", "apedp_dyn/video/collision_domain.py", "apedp_dyn/video/eval_gate.py",
          # importing eval_gate runs the package's __init__, which imports these
          "apedp_dyn/video/__init__.py",
          *(f"apedp_dyn/video/{m}.py" for m in ("continuous_interpolator", "decoupled_transport", "jet_flow_model",
                                               "jet_flow_stepper", "noether_engine", "physics_knobs", "semantic_bridge",
                                               "tensor_train_video", "vae_bridge")),
          "pyproject.toml")
FRAMES, SIZE, HIDDEN = 8, 32, 64
DATA = ("fluid", "collision", "ucf101")
ARMS = ("meanflow-jetflow", "meanflow-unet", "fm-jetflow", "fm-unet")   #: the hybrid, its two parents, the reference
SEEDS = (0, 1, 2)
GRID: dict[str, tuple[tuple[str, int], ...]] = {
    "meanflow-jetflow": (("meanflow", 1), ("meanflow", 2), ("meanflow", 4), ("euler", 32)),
    "meanflow-unet": (("meanflow", 1), ("meanflow", 2), ("meanflow", 4), ("euler", 32)),
    "fm-jetflow": (("euler", 1), ("euler", 4), ("jetflow-sampler", 2), ("euler", 32)),   # the slow one last
    "fm-unet": (("euler", 1), ("euler", 4), ("euler", 32)),
}
UCF_REPO, UCF_FILE = "sayakpaul/ucf101-subset", "UCF101_subset.tar.gz"
UCF_SHA256 = "e9fcc76af48d320be88c5265f2e0576ecd615956976f6ce4742fdf2b042b71eb"   #: as pinned by the earlier test
UCF_GROUPS = {"final": (range(1, 20), range(20, 26)), "dev": (range(1, 16), range(16, 20))}
TOY = {"final": (500, 15), "dev": (1000, 16)}
MARGIN, SEEDS_NEEDED = 0.9, 2     #: "A beats B": mean FD of A at most 0.9 of B's, and lower on 2 of 3 seeds
ROUGHNESS_LIMIT = 1.5             #: the hybrid's roughness at most 1.5 times flow matching's at four calls
JETFLOW_FEW = (("euler", 1), ("euler", 4), ("jetflow-sampler", 2))   #: JetFlow's samplers with at most four calls
SECONDS, N_GEN = 2700.0, 256      #: the final run's training seconds and clips per sampler
NUMERICAL_ALWAYS = (FloatingPointError, ArithmeticError, np.linalg.LinAlgError)
NUMERICAL_IF_SAID = (RuntimeError, ValueError)   #: counted only when the message names a numerical breakdown
NUMERICAL_WORDS = re.compile(r"\bnan\b|\binf|not finite|non-finite|converge|singular", re.IGNORECASE)
THREADS = 4
EXPECTATIONS = {
    "H1": "the hybrid at one call beats MeanFlow on the UNet at one call",
    "H2": "the hybrid at one call is no worse than JetFlow's flow matching at its best sampler with at most four calls",
    "H3": "the hybrid at four calls beats plain flow matching at four calls",
    "H4-1": "the hybrid's clips at one call are at most 1.5 times as rough as plain flow matching's at four calls",
    "H4-4": "the hybrid's clips at four calls are at most 1.5 times as rough as plain flow matching's at four calls",
}


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
    for split in ("train", "val", "test"):
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


def _finite(value: Any) -> float | None:
    """A score as recorded: a finite float, or ``None`` when it is not finite."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _benchmark(threads: int) -> float:
    """Seconds for a fixed piece of work (float32 matrix products) at the job's thread count: the runner's speed."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(384, 384, generator=g)
    started = time.perf_counter()
    for _ in range(200):
        a = torch.tanh(a @ a / 20.0)
    return round(time.perf_counter() - started, 3)


def _weights_finite(model: Any) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def _breakdown(exc: BaseException, model: Any) -> bool:
    """Whether an error is a numerical breakdown of the model (scored), not a fault of the machine or the code."""
    if isinstance(exc, NUMERICAL_ALWAYS) or not _weights_finite(model):
        return True
    return isinstance(exc, NUMERICAL_IF_SAID) and NUMERICAL_WORDS.search(str(exc)) is not None


def _broken_row(how: str, steps: int, why: str) -> dict[str, Any]:
    return {"sampler": how, "steps": steps, "frechet_pca": None, "temporal_roughness": None, "non_finite": True,
            "error": why[:300]}


def _sampler_row(model: Any, how: str, steps: int, n_gen: int, seed: int, train: np.ndarray,
                 held: np.ndarray) -> dict[str, Any]:
    try:
        row = h.evaluate(model, how, steps, n_gen, 20_000 + seed, train, held)
    except (*NUMERICAL_ALWAYS, *NUMERICAL_IF_SAID) as exc:
        if not _breakdown(exc, model):
            raise                       # not the model's breakdown: the job fails and is missing
        return _broken_row(how, steps, f"{type(exc).__name__}: {exc}")
    for key in ("frechet_pca", "temporal_roughness", "memorisation_ratio"):
        row[key] = _finite(row.get(key))
    row["non_finite"] = row["frechet_pca"] is None or row["temporal_roughness"] is None
    return row


def run_job(args: argparse.Namespace) -> int:
    if args.mode == "final":
        check_frozen()
    set_size()
    torch.set_num_threads(args.threads)
    import cv2

    train, held, info = data(args.data, args.mode, args.cache)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(train), size=min(args.n_gen, len(train)), replace=False)
    record: dict[str, Any] = {
        "data": args.data, "arm": args.arm, "seed": args.seed, "mode": args.mode, "seconds_budget": args.seconds,
        "n_gen": args.n_gen, "frames": FRAMES, "size": SIZE, "jetflow_hidden": HIDDEN,
        "train_clips": len(train), "held_clips": len(held), "data_info": info, "samples": [], "complete": False,
        "floor_frechet_pca": _finite(round(h.frechet_pca(train, held, train[pick]), 4)),
        "held_temporal_roughness": _finite(round(h.temporal_roughness(held), 5)),
        "benchmark_seconds": _benchmark(args.threads),
        "threads": args.threads, "torch": torch.__version__, "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "cpu": _cpu_model() or platform.processor(), "opencv": cv2.__version__,
        "commit": os.environ.get("GITHUB_SHA"), "numpy": np.__version__}
    if args.mode == "final":
        record["frozen"] = frozen_digests()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"job_{args.data}_{args.arm}_s{args.seed}.json"

    def write() -> None:
        path.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")

    latents = h.encode(train)
    torch.manual_seed(args.seed)
    model = h.build(args.arm)
    record["parameters"] = sum(p.numel() for p in model.parameters())
    trainer = h.train_meanflow if args.arm.startswith("meanflow") else h.train_flow_matching
    broke: str | None = None
    try:
        fit = trainer(model, latents, args.seconds, lr=1e-3, seed=args.seed)
        fit["final_loss"] = _finite(fit.get("final_loss"))
    except (*NUMERICAL_ALWAYS, *NUMERICAL_IF_SAID) as exc:
        if not _breakdown(exc, model):
            raise
        fit = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        broke = f"training broke down: {fit['error']}"
    if broke is None and not _weights_finite(model):
        broke = "training broke down: weights not finite"
    record["training"] = {**fit, "breakdown": broke is not None}
    write()
    print(f"{args.data} {args.arm} seed {args.seed}: {fit}", flush=True)
    for how, steps in GRID[args.arm]:
        # A model whose training broke down scores a breakdown on every sampler (worse than any finite score).
        row = (_broken_row(how, steps, broke) if broke is not None
               else _sampler_row(model, how, steps, args.n_gen, args.seed, train, held))
        record["samples"].append(row)
        write()
        print(json.dumps(row), flush=True)
    record["loaded"] = _loaded_digests()
    record["complete"] = True
    write()
    return 0


def _loaded_digests() -> dict[str, str]:
    """SHA-256 of every file of this repository the job loaded (the frozen list cannot name them all)."""
    out = {}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix == ".py" and path.is_file() and ROOT in path.parents:
            out[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(out.items()))


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
    """One value per seed; a non-finite value (recorded as ``None``) is read as infinity, worse than any finite one."""
    out = []
    for seed in SEEDS:
        row = next(r for r in jobs[(kind, arm, seed)]["samples"] if r["sampler"] == how and r["steps"] == steps)
        value = row.get(field)
        out.append(math.inf if value is None else float(value))
    return out


def _mean(values: list[float]) -> float:
    return math.inf if not values or any(not math.isfinite(v) for v in values) else float(np.mean(values))


def _shown(x: float) -> float | str:
    return round(x, 5) if math.isfinite(x) else "not finite"


class NotRead(KeyError):
    """Too few seeds on which the comparator gave a finite score."""


def _paired(a: list[float], b: list[float]) -> tuple[list[int], int]:
    """Seeds where both are finite, and seeds where A broke down and B did not.

    A breakdown of A where B was finite fails A's expectation (a breakdown is a loss). A breakdown of the
    comparator B is no evidence for A: that seed is left out, not won; with fewer than two seeds left the
    expectation is not read.
    """
    both = [k for k, (x, y) in enumerate(zip(a, b, strict=True)) if math.isfinite(x) and math.isfinite(y)]
    lost = sum(1 for x, y in zip(a, b, strict=True) if not math.isfinite(x) and math.isfinite(y))
    return both, lost


def beats(a: list[float], b: list[float], margin: float = MARGIN) -> dict[str, Any]:
    both, lost = _paired(a, b)
    wins = sum(a[k] < b[k] for k in both)
    if len(both) < SEEDS_NEEDED and lost == 0:
        raise NotRead(f"{len(both)} seeds with both scores finite")
    ma, mb = _mean([a[k] for k in both]), _mean([b[k] for k in both])
    held = lost == 0 and len(both) >= SEEDS_NEEDED and ma <= margin * mb and wins >= SEEDS_NEEDED
    return {"held": held, "mean_a": _shown(ma), "mean_b": _shown(mb), "seeds_lower": wins,
            "seeds_compared": len(both), "seeds_lost_to_breakdown": lost}


def no_worse(a: list[float], b: list[float], margin: float = MARGIN) -> dict[str, Any]:
    """The first test's form: A's mean at most 1/margin times B's, over the seeds where both are finite; a
    breakdown of A where B was finite fails it."""
    both, lost = _paired(a, b)
    if len(both) < SEEDS_NEEDED and lost == 0:
        raise NotRead(f"{len(both)} seeds with both scores finite")
    ma, mb = _mean([a[k] for k in both]), _mean([b[k] for k in both])
    held = lost == 0 and len(both) >= SEEDS_NEEDED and ma <= mb / margin
    return {"held": held, "mean_a": _shown(ma), "mean_b": _shown(mb), "seeds_compared": len(both),
            "seeds_lost_to_breakdown": lost}


def rougher(a: list[float], b: list[float], limit: float = ROUGHNESS_LIMIT) -> dict[str, Any]:
    """H4: A's mean roughness at most ``limit`` times B's, over seeds where both are finite; A's breakdown fails."""
    both, lost = _paired(a, b)
    if len(both) < SEEDS_NEEDED and lost == 0:
        raise NotRead(f"{len(both)} seeds with both roughnesses finite")
    ma, mb = _mean([a[k] for k in both]), _mean([b[k] for k in both])
    held = lost == 0 and len(both) >= SEEDS_NEEDED and ma <= limit * mb
    return {"held": held, "mean_hybrid": _shown(ma), "mean_flow_matching": _shown(mb), "seeds_compared": len(both),
            "seeds_lost_to_breakdown": lost}


def _read(fn: Any) -> dict[str, Any]:
    try:
        out: dict[str, Any] = fn()
        return out
    except NotRead as few:
        return {"held": None, "not_read": f"too few comparable seeds: {few}"}
    except (KeyError, StopIteration) as missing:
        return {"held": None, "not_read": f"job or sampler missing: {missing}"}


def expectations(jobs: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in DATA:
        def fd(arm: str, how: str, steps: int, kind: str = kind) -> list[float]:
            return _values(jobs, kind, arm, how, steps)

        def h2(kind: str = kind) -> dict[str, Any]:
            # JetFlow's best: the lowest mean over its finite seeds, among samplers finite on at least two seeds.
            options = {f"{how} {steps}": _values(jobs, kind, "fm-jetflow", how, steps) for how, steps in JETFLOW_FEW}
            usable = {k: v for k, v in options.items() if sum(math.isfinite(x) for x in v) >= SEEDS_NEEDED}
            if not usable:
                raise NotRead("no sampler of fm-jetflow finite on two seeds")
            best = min(usable, key=lambda k: float(np.mean([x for x in usable[k] if math.isfinite(x)])))
            return {**no_worse(fd("meanflow-jetflow", "meanflow", 1), usable[best]), "jetflow_best": best}

        def rough(steps: int, kind: str = kind) -> dict[str, Any]:
            return rougher(_values(jobs, kind, "meanflow-jetflow", "meanflow", steps, "temporal_roughness"),
                           _values(jobs, kind, "fm-unet", "euler", 4, "temporal_roughness"))

        def floor(kind: str = kind) -> dict[str, Any]:
            vals = [jobs[(kind, a, s)]["floor_frechet_pca"] for a in ARMS for s in SEEDS if (kind, a, s) in jobs]
            vals = [v for v in vals if v is not None]
            if not vals:
                raise KeyError(kind)
            return {"value": round(float(np.mean(vals)), 4)}
        out[kind] = {
            "H1": _read(lambda: beats(fd("meanflow-jetflow", "meanflow", 1), fd("meanflow-unet", "meanflow", 1))),
            "H2": _read(h2),
            "H3": _read(lambda: beats(fd("meanflow-jetflow", "meanflow", 4), fd("fm-unet", "euler", 4))),
            "H4-1": _read(lambda: rough(1)),
            "H4-4": _read(lambda: rough(4)),
            "floor_mean": _read(floor),
        }
    return out


def outcome(exp: dict[str, Any]) -> dict[str, Any]:
    """What the results propose, in the plan's order (the owner decides; nothing goes in force).

    1. H1 or H2 read and failed on ucf101: the hybrid is proposed for retirement; nothing else is read.
    2. Otherwise, H1 or H2 not read on ucf101: no one-call proposal (a rerun is put to the owner).
    3. Otherwise, the one-call proposal: H1 and H2 on ucf101 and together on a toy world, with H4-1 on ucf101.
    4. Unless rule 1 applied, the four-call proposal: H3 on ucf101 and on a toy world, with H4-4 on ucf101.
    """
    def held(kind: str, key: str) -> bool | None:
        value: bool | None = exp[kind][key]["held"]
        return value

    toys = ("fluid", "collision")
    if held("ucf101", "H1") is False or held("ucf101", "H2") is False:
        return {"rules": [1], "proposal": "list the hybrid as retired (deprecated, kept for replication; nothing in "
                                          "force changes), under the owner's standing word of 27 November 00:41"}
    rules: list[int] = []
    proposals: list[str] = []
    if held("ucf101", "H1") is None or held("ucf101", "H2") is None:
        rules.append(2)
        proposals.append("no one-call proposal: H1 or H2 not read on ucf101; a rerun of the missing jobs is put "
                         "to the owner")
    else:
        rules.append(3)
        if not any(held(t, "H1") and held(t, "H2") for t in toys):
            proposals.append("H1 and H2 hold on ucf101 but on no toy world together: a further test is proposed")
        elif held("ucf101", "H4-1") is True:
            proposals.append("name the hybrid the video package's few-step model at one call")
        elif held("ucf101", "H4-1") is None:
            proposals.append("no one-call proposal: H4-1 not read on ucf101")
        else:
            proposals.append("no one-call proposal: H4-1 does not hold on ucf101; the flicker is reported")
    rules.append(4)
    if held("ucf101", "H3") is True and any(held(t, "H3") for t in toys):
        if held("ucf101", "H4-4") is True:
            proposals.append("name the hybrid the video package's model at four calls")
        elif held("ucf101", "H4-4") is None:
            proposals.append("no four-call proposal: H4-4 not read on ucf101")
        else:
            proposals.append("no four-call proposal: H4-4 does not hold on ucf101; the flicker is reported")
    elif held("ucf101", "H3") is True:
        proposals.append("H3 holds on ucf101 alone: a further test is proposed")
    elif held("ucf101", "H3") is None:
        proposals.append("no four-call proposal: H3 not read on ucf101")
    else:
        proposals.append("no four-call proposal: H3 does not hold on ucf101")
    return {"rules": rules, "proposal": "; ".join(proposals)}


def check_job(rec: dict[str, Any], mode: str) -> list[str]:
    """Why a job's record cannot be counted (an empty list when it can)."""
    problems = []
    if rec.get("mode") != mode:
        problems.append(f"mode {rec.get('mode')}")
    if (rec.get("frames"), rec.get("size"), rec.get("jetflow_hidden"), rec.get("threads")) != (FRAMES, SIZE, HIDDEN,
                                                                                              THREADS):
        problems.append("clip size, width or threads")
    if mode == "final":
        if rec.get("frozen") != frozen_digests():
            problems.append("frozen digests differ")
        if rec.get("seconds_budget") != SECONDS or rec.get("n_gen") != N_GEN:
            problems.append(f"budget {rec.get('seconds_budget')} s, {rec.get('n_gen')} clips")
        if rec.get("data") in ("fluid", "collision") and (rec.get("train_clips"), rec.get("held_clips")) != (400, 100):
            problems.append(f"clips {rec.get('train_clips')}/{rec.get('held_clips')}")
        if rec.get("data") == "ucf101" and rec.get("data_info", {}).get("archive_sha256") != UCF_SHA256:
            problems.append("archive digest")
        loaded = rec.get("loaded") or {}
        changed = [f for f, d in loaded.items() if not (ROOT / f).is_file()
                   or hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != d]
        if changed:
            problems.append(f"loaded files differ from this checkout: {changed[:5]}")
    if not rec.get("complete"):
        problems.append(PARTIAL)
    return problems


PARTIAL = "job stopped before every sampler ran"


def collect(args: argparse.Namespace) -> int:
    if args.mode == "final":
        check_frozen()
    jobs: dict[tuple[str, str, int], dict[str, Any]] = {}
    rejected: dict[str, list[str]] = {}
    records = [json.loads(p.read_text()) for p in sorted(args.out.rglob("job_*.json"))]
    keys = [(r["data"], r["arm"], int(r["seed"])) for r in records]
    commits = {r.get("commit") for r in records}
    ucf_counts = {(r.get("train_clips"), r.get("held_clips")) for r in records if r.get("data") == "ucf101"}
    for rec, key in zip(records, keys, strict=True):
        name = "/".join(map(str, key))
        problems = check_job(rec, args.mode)
        if keys.count(key) > 1:
            problems.append("more than one record for this job")
        if args.mode == "final" and len(commits) > 1:
            problems.append(f"jobs ran on {len(commits)} commits")
        if args.mode == "final" and rec.get("data") == "ucf101" and len(ucf_counts) > 1:
            problems.append(f"ucf101 jobs disagree on their clip counts: {sorted(ucf_counts)}")
        if problems and problems != [PARTIAL]:
            rejected[name] = problems
            continue
        if problems:
            rejected[name] = problems      # partial: its finished samplers are read, the rest are missing
        jobs[key] = rec
    missing = [f"{d}/{a}/{s}" for d in DATA for a in ARMS for s in SEEDS if (d, a, s) not in jobs]
    record: dict[str, Any] = {"preregistration": PREREGISTRATION_ID, "mode": args.mode, "missing_jobs": missing,
                              "rejected_or_partial_jobs": rejected, "expectations_wording": EXPECTATIONS,
                              "jobs": [jobs[k] for k in sorted(jobs)]}
    if jobs:
        record["expectations"] = expectations(jobs)
        record["outcome"] = outcome(record["expectations"])
    if args.mode == "final":
        record["frozen"] = frozen_digests()
    (args.out / "results.json").write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k in ("missing_jobs", "rejected_or_partial_jobs",
                                                                 "expectations", "outcome")}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("job", "collect"))
    parser.add_argument("--data", choices=DATA)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=0)
    parser.add_argument("--mode", choices=("dev", "final"), required=True)
    parser.add_argument("--seconds", type=float, default=SECONDS)
    parser.add_argument("--n-gen", type=int, default=N_GEN)
    parser.add_argument("--threads", type=int, default=THREADS)
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

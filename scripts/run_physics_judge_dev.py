#!/usr/bin/env python3
"""Development runs of the physics judge on PhysInOne (docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md, part 1a).

  fetch  --kind K --count N [--seed S] [--pool dev] --out tracks.json
         Picks N single-physics Train scenes of the kind's phenomenon, reads from each scene's archive on
         the Hugging Face Hub only what the judge needs (by range requests: the static camera list, one
         static camera's segmentation masks, the frame rate), and turns the masks into object tracks
         (centre and the square root of the area per frame, for the dynamic objects that move most).
  judge  --tracks tracks.json --out verdicts.json
         The judge's verdict on each true clip and on each development perturbation of it.
  summary --dir D --out summary.json
         Pass rates by kind and variant over every verdicts file in D.

Data: PhysInOne (Zhou et al., CVPR 2026), CC BY-NC-SA 4.0. The tracks are made at run time and are not
committed; records keep scene names, counts and verdicts.

Pools: a scene whose background number is a multiple of five is reserved for the blind set and is never
read in development (``--pool dev``); the blind set is drawn from those (``--pool blind``) only by the
frozen run.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apedp_dyn.epistemic.physics_judge import Track, judge  # noqa: E402
from apedp_dyn.video.physics_perturbations import PERTURBATIONS, perturb  # noqa: E402

INDEX_REPO = "vLAR/PhysInOne"
INDEX_FILE = "Utils/scripts/download_lists/repo_assignment.txt"
MAP_FILE = "Utils/scripts/download_lists/repo_map.json"
PHENOMENON = {"free_fall": "VerticalFall", "throw": "ObliqueProjectile", "roll": "RollDownSlope",
              "bounce": "ElasticFall", "pendulum": "SimplePendulum", "collision": "MovingHitsMoving"}
OBJECTS = {"collision": 2}
SCENE = re.compile(r"/Scenes/Train/SinglePhysics/(?P<scene>(?P<ph>[A-Za-z]+)__bg(?P<bg>\d+)__[A-Za-z0-9]+)\.")


def _hub():
    from huggingface_hub import HfFileSystem, hf_hub_download

    return HfFileSystem(), hf_hub_download


def select(kind: str, count: int, seed: int, pool: str) -> list[tuple[str, str]]:
    _, download = _hub()
    index = Path(download(INDEX_REPO, INDEX_FILE, repo_type="dataset")).read_text().splitlines()
    repo_map = json.loads(Path(download(INDEX_REPO, MAP_FILE, repo_type="dataset")).read_text())
    want = PHENOMENON[kind]
    found = []
    for line in index:
        path, _, part = line.partition("\t")
        m = SCENE.search(path)
        if not m or m["ph"] != want:
            continue
        blind = int(m["bg"]) % 5 == 0
        if (pool == "blind") != blind:
            continue
        found.append((m["scene"], repo_map[part.strip()]))
    found.sort()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(found), size=min(count, len(found)), replace=False) if found else []
    return [found[i] for i in sorted(pick)]


def _first_array(blob: bytes) -> np.ndarray:
    with np.load(io.BytesIO(blob)) as z:
        return np.asarray(z[z.files[0]])


def tracks_from_scene(scene: str, repo: str) -> dict:
    fs, _ = _hub()
    path = f"datasets/{repo}/Train/SinglePhysics/{scene}_trajectory.zip"
    with fs.open(path, "rb", block_size=1 << 20) as handle, zipfile.ZipFile(handle) as archive:
        names = archive.namelist()
        cams_file = next((n for n in names if n.endswith("static_camera_list.txt")), None)
        cams = []
        if cams_file:
            cams = [c.strip() for c in archive.read(cams_file).decode().split() if c.strip()]
        seg_dirs = sorted({n.rsplit("/seg/", 1)[0] for n in names if "/seg/" in n and n.endswith(".npz")})
        static = [d for d in seg_dirs if not d.endswith("CineCamera_Moving")]
        chosen = None
        for c in cams:
            hit = [d for d in static if d.endswith(c) or d.endswith(f"CineCamera_{c}")]
            if hit:
                chosen = hit[0]
                break
        if chosen is None and static:
            chosen = static[0]
        if chosen is None:
            raise RuntimeError("no static camera with masks")
        fps = 30.0
        actors: list[dict] = []
        traj = next((n for n in names if n.endswith("_trajectory.json") and "/" in n), None)
        if traj:
            doc = json.loads(archive.read(traj)) or {}
            info = doc.get("sequence_info") or {}
            fps = float(info.get("fps") or fps)
            actors = _actor_paths(doc)
        frames = sorted(n for n in names if n.startswith(chosen + "/seg/") and n.endswith(".npz"))
        per_id: dict[int, dict[int, tuple[float, float, float]]] = defaultdict(dict)
        shape = None
        for k, name in enumerate(frames):
            mask = _first_array(archive.read(name))
            if mask.ndim == 3:
                mask = mask[..., 0]
            shape = mask.shape
            for obj in np.unique(mask):
                if obj < 128:
                    continue
                ys, xs = np.nonzero(mask == obj)
                per_id[int(obj)][k] = (float(xs.mean()), float(ys.mean()), float(np.sqrt(xs.size)))
    n = len(frames)
    objects = []
    for obj, seen in per_id.items():
        xy = np.full((n, 2), np.nan)
        size = np.full(n, np.nan)
        for k, (x, y, s) in seen.items():
            xy[k] = (x, y)
            size[k] = s
        ok = np.all(np.isfinite(xy), axis=1)
        extent = float(np.max(np.linalg.norm(xy[ok] - xy[ok][0], axis=1))) if ok.any() else 0.0
        objects.append({"id": obj, "extent": extent, "visible": float(ok.mean()),
                        "xy": [[round(a, 2), round(b, 2)] if np.isfinite(a) else None for a, b in xy],
                        "size": [round(float(v), 2) if np.isfinite(v) else None for v in size]})
    objects.sort(key=lambda o: -o["extent"])
    return {"scene": scene, "camera": chosen.rsplit("/", 1)[-1], "fps": fps, "frames": n,
            "shape": list(shape) if shape else None, "objects": objects, "actors": actors}


def _actor_paths(doc: dict) -> list[dict]:
    """World positions per frame of the trajectory file's solid actors (development diagnostics only)."""
    out = []
    raw = doc.get("actors") or {}
    items = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for name, actor in items:
        if not isinstance(actor, dict):
            continue
        entries = actor.get("transform_data") or []
        if not entries:
            comps = actor.get("components") or {}
            first = next(iter(comps.values()), {}) if isinstance(comps, dict) else {}
            entries = first.get("transform_data") or [] if isinstance(first, dict) else []
        path = []
        for e in entries:
            loc = (e.get("transform") or {}).get("location")
            if isinstance(loc, dict):
                loc = [loc.get("x"), loc.get("y"), loc.get("z")]
            path.append([e.get("frame"), e.get("time_seconds"), loc])
        out.append({"name": str(name), "type": actor.get("actor_type"), "path": path[:120]})
    return out[:4]


def _to_tracks(clip: dict, kind: str) -> list[Track]:
    n = clip["frames"]
    t = np.arange(n) / clip["fps"]
    out = []
    for obj in clip["objects"][:OBJECTS.get(kind, 1)]:
        xy = np.array([p if p is not None else [np.nan, np.nan] for p in obj["xy"]], dtype=float)
        size = np.array([s if s is not None else np.nan for s in obj["size"]], dtype=float)
        out.append(Track(t, xy, size))
    return out


def cmd_fetch(args: argparse.Namespace) -> None:
    picked = select(args.kind, args.count, args.seed, args.pool)
    clips = []
    for scene, repo in picked:
        start = time.time()
        try:
            clip = tracks_from_scene(scene, repo)
            clip["seconds"] = round(time.time() - start, 1)
            clips.append(clip)
            print(f"{scene}: {clip['frames']} frames, {len(clip['objects'])} objects, {clip['seconds']} s", flush=True)
        except (OSError, RuntimeError, ValueError, KeyError, zipfile.BadZipFile) as exc:  # recorded, not hidden
            clips.append({"scene": scene, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{scene}: {type(exc).__name__}: {exc}", flush=True)
    Path(args.out).write_text(json.dumps({"kind": args.kind, "pool": args.pool, "seed": args.seed,
                                          "clips": clips}))


def cmd_judge(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.tracks).read_text())
    kind = data["kind"]
    rows = []
    for i, clip in enumerate(data["clips"]):
        if "error" in clip:
            rows.append({"scene": clip["scene"], "variant": "true", "outcome": "unreadable", "reasons": [clip["error"]]})
            continue
        need = OBJECTS.get(kind, 1)
        if len(clip["objects"]) < need:
            rows.append({"scene": clip["scene"], "variant": "true", "outcome": "untestable",
                         "reasons": [f"{len(clip['objects'])} moving objects in the masks"]})
            continue
        tracks = _to_tracks(clip, kind)
        v = judge(kind, tracks)
        rows.append({"scene": clip["scene"], "variant": "true", "outcome": v.outcome, "reasons": v.reasons,
                     "stats": v.stats})
        for how in PERTURBATIONS:
            broken = perturb(kind, tracks, how, np.random.default_rng(1000 + i))
            if broken is None:
                rows.append({"scene": clip["scene"], "variant": how, "outcome": "not applicable"})
                continue
            v = judge(kind, broken)
            rows.append({"scene": clip["scene"], "variant": how, "outcome": v.outcome, "reasons": v.reasons})
    Path(args.out).write_text(json.dumps({"kind": kind, "rows": rows}, default=float))
    tally = Counter((r["variant"], r["outcome"]) for r in rows)
    for key in sorted(tally):
        print(kind, *key, tally[key])


def cmd_summary(args: argparse.Namespace) -> None:
    table: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for f in sorted(Path(args.dir).glob("verdicts_*.json")):
        data = json.loads(f.read_text())
        for r in data["rows"]:
            table[data["kind"]][r["variant"]][r["outcome"]] += 1
    out = {k: {v: dict(c) for v, c in d.items()} for k, d in table.items()}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--kind", required=True, choices=sorted(PHENOMENON))
    f.add_argument("--count", type=int, default=30)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--pool", choices=("dev", "blind"), default="dev")
    f.add_argument("--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--tracks", required=True)
    j.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--dir", required=True)
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if getattr(args, "pool", "dev") == "blind":
        raise SystemExit("the blind pool is read only by the frozen run")
    {"fetch": cmd_fetch, "judge": cmd_judge, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()

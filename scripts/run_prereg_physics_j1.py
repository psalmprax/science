#!/usr/bin/env python3
"""PREREG-PHYSICS-J1-2026-11: the physics judge's blind test on clips rendered here (gate J, part 1), once.

Status: audit-only

On the owner's word of 27 November ("go ahead to test all"), which took up the proposal to test the judge mainly
on clips simulated and rendered here (``apedp_dyn/video/physics_clips_pybullet.py``) and to keep PhysInOne as a
second check beside gate J. Part 1 of gate J (docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md): true clips and broken
ones. Part 2, agreement with human labels, needs an object tracker on generated video and is a later test.

    python scripts/run_prereg_physics_j1.py rendered --mode dev --shard 0/1 --out DIR     # development seeds 0..29
    python scripts/run_prereg_physics_j1.py rendered --mode final --shard k/N --out DIR   # blind seeds, frozen
    python scripts/run_prereg_physics_j1.py physinone --kind KIND --out DIR               # PhysInOne's blind pool, beside
    python scripts/run_prereg_physics_j1.py collect --mode final --out DIR

Records hold verdicts, reasons and counts, with PhysInOne's scene names; no PhysInOne track is written
(CC BY-NC-SA 4.0). The rendered clips are this repository's own.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from apedp_dyn.epistemic.physics_judge import PASS, judge  # noqa: E402
from apedp_dyn.video.physics_perturbations import PERTURBATIONS, perturb  # noqa: E402

PREREGISTRATION_ID = "PREREG-PHYSICS-J1-2026-11"
JUDGE_DIGEST = ROOT / "preregistration" / "DIGEST-PHYSICS-J1-JUDGE.json"   #: stage 1: the judge, frozen first
DIGEST = ROOT / "preregistration" / "DIGEST-PHYSICS-J1.json"               #: stage 2: with the reviewer's breakages
JUDGE_FILES = (
    "preregistration/PREREG-PHYSICS-J1-2026-11.md", "scripts/run_prereg_physics_j1.py",
    ".github/workflows/physics-j1.yml", "tests/test_physics_j1_prereg.py",
    "apedp_dyn/epistemic/physics_judge.py", "apedp_dyn/video/physics_perturbations.py",
    "apedp_dyn/video/physics_clips_pybullet.py", "scripts/run_physics_judge_dev.py",
)
UNSEEN_MODULE = "apedp_dyn.video.physics_perturbations_unseen"   #: written by the independent reviewer after stage 1
UNSEEN_FILE = "apedp_dyn/video/physics_perturbations_unseen.py"
GATE_KINDS = ("free_fall", "throw", "roll", "bounce", "pendulum")
BESIDE_KINDS = ("collision",)                     #: judged and reported, outside the gate (the owner's choice)
PER_KIND = {"dev": 30, "final": 60}
TRUE_PASS_MIN = 0.80        #: gate J (the plan's proposed numbers, kept by the owner on 27 November 00:07)
BROKEN_PASS_MAX = 0.10      #: per kind of breakage, pooled over the gate's kinds
MIN_APPLIED = 20            #: a kind of breakage applied to fewer clips is reported, not read (this test's number)
PHYSINONE_COUNT = {"collision": 60}               #: else 30 scenes per kind, as in development


def seeds(mode: str) -> list[int]:
    from apedp_dyn.video.physics_clips_pybullet import BLIND_FROM

    base = BLIND_FROM if mode == "final" else 0
    return [base + i for i in range(PER_KIND[mode])]


def digests(files: tuple[str, ...]) -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}


def check_frozen() -> None:
    """Both stages: the judge as frozen first, and the reviewer's breakages frozen with it."""
    stage1 = json.loads(JUDGE_DIGEST.read_text(encoding="utf-8"))["sha256"]
    if stage1 != digests(JUDGE_FILES):
        raise SystemExit("the judge changed since it was frozen (stage 1)")
    stage2 = json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"]
    if stage2 != digests((*JUDGE_FILES, UNSEEN_FILE)):
        raise SystemExit("the plan or the reviewer's breakages changed since they were frozen (stage 2)")


def unseen() -> tuple[tuple[str, ...], Any]:
    """The reviewer's kinds of breakage and their function, or none in development (they do not exist yet)."""
    try:
        module = importlib.import_module(UNSEEN_MODULE)
    except ModuleNotFoundError:
        return (), None
    return tuple(module.UNSEEN_KINDS), module.perturb_unseen


def verdicts(kind: str, tracks: list[Any], rng_seed: int, kinds_unseen: tuple[str, ...],
             perturb_unseen: Any) -> list[dict[str, Any]]:
    rows = []
    v = judge(kind, tracks)
    rows.append({"variant": "true", "outcome": v.outcome, "reasons": v.reasons[:3]})
    for group, hows, fn in (("development", PERTURBATIONS, perturb), ("unseen", kinds_unseen, perturb_unseen)):
        for how in hows:
            broken = fn(kind, tracks, how, np.random.default_rng([rng_seed, len(how)]))
            if broken is None:
                rows.append({"variant": how, "group": group, "outcome": "not applicable"})
                continue
            w = judge(kind, broken)
            rows.append({"variant": how, "group": group, "outcome": w.outcome, "reasons": w.reasons[:3]})
    return rows


def cmd_rendered(args: argparse.Namespace) -> int:
    from apedp_dyn.video.physics_clips_pybullet import make_clip

    if args.mode == "final":
        check_frozen()
    kinds_unseen, perturb_unseen = unseen() if args.mode == "final" else ((), None)
    if args.mode == "final" and not kinds_unseen:
        raise SystemExit("the reviewer's breakages are missing")
    k, n = (int(v) for v in args.shard.split("/"))
    units = [(kind, s) for kind in (*GATE_KINDS, *BESIDE_KINDS) for s in seeds(args.mode)]
    rows = []
    for i, (kind, seed) in enumerate(units):
        if i % n != k:
            continue
        tracks, facts = make_clip(kind, seed, allow_blind=args.mode == "final")
        for row in verdicts(kind, tracks, seed, kinds_unseen, perturb_unseen):
            if row["variant"] == "true":
                row["facts"] = facts             # the scene and camera, this repository's own
            rows.append({"source": "rendered", "kind": kind, "seed": seed, **row})
        print(kind, seed, Counter(r["outcome"] for r in rows if r["seed"] == seed and r["kind"] == kind), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"rendered-{args.mode}-{k:03d}-of-{n:03d}.json").write_text(json.dumps(rows) + "\n")
    return 0


def cmd_physinone(args: argparse.Namespace) -> int:
    import run_physics_judge_dev as dev

    check_frozen()
    kinds_unseen, perturb_unseen = unseen()
    count = PHYSINONE_COUNT.get(args.kind, 30)
    picked = dev.select(args.kind, count, 0, "blind")      # the blind pool: read here, by the frozen run, only
    rows: list[dict[str, Any]] = []
    tried = judged = 0
    for scene, repo in picked:
        if judged >= count or tried >= 4 * count:
            break
        tried += 1
        try:
            clip = dev.tracks_from_scene(scene, repo)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:   # recorded, not hidden
            rows.append({"source": "physinone", "kind": args.kind, "scene": scene, "variant": "true",
                         "outcome": "unreadable", "reasons": [f"{type(exc).__name__}: {exc}"[:200]]})
            continue
        if args.kind == "bounce" and not dev.rebounds(clip.get("actors", [])):
            continue
        if len(clip["objects"]) < dev.OBJECTS.get(args.kind, 1):
            rows.append({"source": "physinone", "kind": args.kind, "scene": scene, "variant": "true",
                         "outcome": "untestable", "reasons": ["too few moving objects in the masks"]})
            judged += 1
            continue
        tracks, _ = dev._to_tracks(clip, args.kind)
        if args.kind == "collision" and not dev.side_view(tracks):
            rows.append({"source": "physinone", "kind": args.kind, "scene": scene, "variant": "true",
                         "outcome": "excluded", "reasons": ["not seen from the side"]})
            judged += 1
            continue
        seed = int(hashlib.sha256(scene.encode()).hexdigest()[:8], 16)
        for row in verdicts(args.kind, tracks, seed, kinds_unseen, perturb_unseen):
            rows.append({"source": "physinone", "kind": args.kind, "scene": scene, **row})
        judged += 1
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"physinone-{args.kind}.json").write_text(json.dumps(rows) + "\n")
    print(args.kind, Counter((r["variant"], r["outcome"]) for r in rows), flush=True)
    return 0


def tally(rows: list[dict[str, Any]], kinds: tuple[str, ...]) -> dict[str, Any]:
    """Per variant over ``kinds``: passes among the clips it applied to, with the other outcomes beside."""
    out: dict[str, Any] = {}
    for variant in sorted({r["variant"] for r in rows if r["kind"] in kinds}):
        sel = [r for r in rows if r["kind"] in kinds and r["variant"] == variant]
        counts = Counter(r["outcome"] for r in sel)
        applied = sum(v for o, v in counts.items() if o not in ("not applicable", "excluded", "unreadable"))
        out[variant] = {"applied": applied, "pass": counts.get(PASS, 0),
                        "share_passing": round(counts.get(PASS, 0) / applied, 4) if applied else None,
                        "outcomes": dict(sorted(counts.items())),
                        "group": next((r.get("group", "true") for r in sel), "true")}
    return out


def gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pooled = tally([r for r in rows if r["source"] == "rendered"], GATE_KINDS)
    true = pooled.get("true", {"applied": 0, "pass": 0})
    true_share = true["pass"] / true["applied"] if true["applied"] else 0.0
    broken = {v: t for v, t in pooled.items() if v != "true"}
    read = {v: t for v, t in broken.items() if t["applied"] >= MIN_APPLIED}
    each = {v: t["share_passing"] is not None and t["share_passing"] <= BROKEN_PASS_MAX for v, t in read.items()}
    unseen_read = any(t.get("group") == "unseen" for t in read.values())
    return {"true_share_passing": round(true_share, 4), "true_holds": true_share >= TRUE_PASS_MIN,
            "broken_holds": each, "reported_not_read": sorted(set(broken) - set(read)),
            "unseen_kinds_read": unseen_read,
            "part_1_holds": bool(true_share >= TRUE_PASS_MIN and each and all(each.values()) and unseen_read),
            "limits": {"true_pass_min": TRUE_PASS_MIN, "broken_pass_max": BROKEN_PASS_MAX, "min_applied": MIN_APPLIED}}


def cmd_collect(args: argparse.Namespace) -> int:
    if args.mode == "final":
        check_frozen()
    rows = [r for p in sorted(args.out.rglob("*.json")) if p.name.startswith(("rendered-", "physinone-"))
            for r in json.loads(p.read_text())]
    rendered = [r for r in rows if r["source"] == "rendered"]
    record: dict[str, Any] = {
        "preregistration": PREREGISTRATION_ID, "mode": args.mode, "rows": len(rows),
        "gate_j_part_1": gate(rows) if args.mode == "final" else None,
        "rendered_gate_kinds": tally(rendered, GATE_KINDS),
        "rendered_by_kind": {k: tally(rendered, (k,)) for k in (*GATE_KINDS, *BESIDE_KINDS)},
        "physinone_beside": {k: tally([r for r in rows if r["source"] == "physinone"], (k,))
                             for k in (*GATE_KINDS, *BESIDE_KINDS)},
        "misses": [{k: r.get(k) for k in ("source", "kind", "seed", "scene", "variant", "outcome", "reasons")}
                   for r in rows if (r["variant"] == "true") == (r["outcome"] != PASS)
                   and r["outcome"] not in ("not applicable", "excluded")],
    }
    if args.mode == "final":
        record["frozen"] = digests((*JUDGE_FILES, UNSEEN_FILE))
    (args.out / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: record[k] for k in ("gate_j_part_1", "rendered_gate_kinds")}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("rendered", "physinone", "collect"))
    parser.add_argument("--mode", choices=("dev", "final"), default="final")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--kind", choices=(*GATE_KINDS, *BESIDE_KINDS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "rendered":
        return cmd_rendered(args)
    if args.command == "physinone":
        if not args.kind:
            parser.error("--kind is required")
        return cmd_physinone(args)
    return cmd_collect(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""PREREG-PHYSICS-J1-2026-11: the physics judge's blind test on clips rendered here (gate J, part 1), once.

Status: audit-only

On the owner's word of 27 November ("go ahead to test all"), which took up the proposal to test the judge mainly
on clips simulated and rendered here (``apedp_dyn/video/physics_clips_pybullet.py``, own generator, exact masks)
and to keep PhysInOne as a second check beside gate J. Part 1 of gate J (docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md):
true clips and broken ones, read for every scene kind and every kind of breakage, with the pooled reading beside.
Part 2, agreement with human labels, needs an object tracker on generated video and is a later test.

    python scripts/run_prereg_physics_j1.py rendered --mode dev --shard 0/1 --out DIR     # development seeds 0..29
    python scripts/run_prereg_physics_j1.py rendered --mode final --shard k/N --out DIR   # blind seeds, frozen
    python scripts/run_prereg_physics_j1.py physinone --kind KIND --out DIR               # PhysInOne's blind pool, beside
    python scripts/run_prereg_physics_j1.py collect --mode final --out DIR

Records hold verdicts, reasons and counts, with PhysInOne's scene names; no PhysInOne track is written
(CC BY-NC-SA 4.0). The rendered clips are this repository's own.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from apedp_dyn.epistemic.physics_judge import FAIL, PASS, judge  # noqa: E402
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
READING_FILE = ROOT / "preregistration" / "PHYSICS-J1-READING.json"   #: the owner's choice of reading, before stage 2
READINGS = ("per_kind", "pooled")
STAGE2_FILES = (*JUDGE_FILES, "preregistration/DIGEST-PHYSICS-J1-JUDGE.json", UNSEEN_FILE,
                "preregistration/PHYSICS-J1-READING.json")
GATE_KINDS = ("free_fall", "throw", "roll", "bounce", "pendulum", "collision")
PER_KIND = {"dev": 30, "final": 60}
TRUE_PASS_MIN = 0.80        #: gate J (the plan's proposed numbers, kept by the owner on 27 November 00:07)
BROKEN_PASS_MAX = 0.10      #: per kind of breakage (read per scene kind, and pooled beside)
MIN_APPLIED = 30            #: a cell applied to fewer clips is read by count: it fails if more than SMALL_CELL_MISSES ...
SMALL_CELL_MISSES = 2       #: ... of its broken clips are not caught (this test's own numbers)
MIN_UNSEEN_KINDS = 2        #: the reviewer writes at least two kinds of breakage
PHYSINONE_COUNT = {"collision": 60}               #: else 30 scenes per kind, as in development
BOUND_CONFIDENCE = 0.95     #: Clopper-Pearson bounds reported beside every share (not read)
NOT_COUNTED = ("not applicable", "excluded", "unreadable")
ERRORS = (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, IndexError, ZeroDivisionError)


def stream(seed: int, how: str) -> np.random.Generator:
    """A breakage's random stream for one clip, named by a hash of the breakage, not by its length."""
    return np.random.default_rng([seed, int(hashlib.sha256(how.encode()).hexdigest()[:8], 16)])


def seeds(mode: str) -> list[int]:
    """Development: 0 to 29. Final: from a base set by the stage-2 digest, so no blind clip existed before it."""
    from apedp_dyn.video.physics_clips_pybullet import BLIND_FROM

    if mode != "final":
        return list(range(PER_KIND["dev"]))
    offset = int(hashlib.sha256(DIGEST.read_bytes()).hexdigest()[:8], 16) % 1_000_000
    base = BLIND_FROM + 1000 * offset
    return [base + i for i in range(PER_KIND["final"])]


def canonical(record: dict[str, Any]) -> bytes:
    """The one byte form a digest file may have (indent 1, sorted keys, a final newline)."""
    return (json.dumps(record, indent=1, sort_keys=True) + "\n").encode()


def digests(files: tuple[str, ...]) -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}


def check_frozen() -> None:
    """Both stages, chained: the judge as frozen first, then the reviewer's breakages with the stage-1 digest."""
    stage1 = json.loads(JUDGE_DIGEST.read_text(encoding="utf-8"))["sha256"]
    if stage1 != digests(JUDGE_FILES):
        raise SystemExit("the judge changed since it was frozen (stage 1)")
    record = json.loads(DIGEST.read_text(encoding="utf-8"))
    if DIGEST.read_bytes() != canonical(record) or set(record) != {"preregistration", "sha256"}:
        raise SystemExit("the stage-2 digest is not in its canonical form (its bytes set the blind seeds)")
    stage2 = record["sha256"]
    if stage2 != digests(STAGE2_FILES):
        raise SystemExit("the plan, the stage-1 digest or the reviewer's breakages changed since stage 2")
    module = importlib.import_module(UNSEEN_MODULE)
    if getattr(module, "STAGE1_SHA256", None) != hashlib.sha256(JUDGE_DIGEST.read_bytes()).hexdigest():
        raise SystemExit("the reviewer's breakages do not name the stage-1 digest they were written against")


def history(loaded: set[str] | None = None) -> dict[str, Any]:
    """The freeze's history in git: each digest written by exactly one commit; the reviewer's file first added after
    stage 1; stage 2 after the reviewer's last change; and from stage 1 to HEAD no frozen or loaded file changed but
    the reviewer's file and the stage-2 digest."""
    git_bin = shutil.which("git") or "git"

    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - fixed argv (git and this repository's paths), no shell
            [git_bin, "-C", str(ROOT), *args], capture_output=True, text=True, check=check)

    def commits(path: str) -> list[str]:
        return git("log", "--format=%H", "--", path).stdout.split()

    def descends(old: str, new: str) -> bool:
        return git("merge-base", "--is-ancestor", old, new, check=False).returncode == 0

    try:
        stage1_path, stage2_path = (str(p.relative_to(ROOT)) for p in (JUDGE_DIGEST, DIGEST))
        stage1, stage2, unseen_all = commits(stage1_path), commits(stage2_path), commits(UNSEEN_FILE)
        changed = set(git("diff", "--name-only", stage1[-1], "HEAD").stdout.split()) if stage1 else set()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {"checked": False, "why": f"{type(exc).__name__}: {exc}"[:200]}
    watched = set(JUDGE_FILES) | set(loaded or ())
    moved = sorted((changed & watched) - {UNSEEN_FILE, stage2_path, str(READING_FILE.relative_to(ROOT))})
    checks = {
        "stage1_one_commit": len(stage1) == 1,
        "stage2_one_commit": len(stage2) == 1,
        "unseen_after_stage1": bool(stage1 and unseen_all) and descends(stage1[0], unseen_all[-1])
        and unseen_all[-1] != stage1[0],
        "stage2_after_unseen": bool(stage2 and unseen_all) and descends(unseen_all[0], stage2[0]),
        "nothing_else_changed_since_stage1": not moved,
    }
    return {"checked": True, **checks, "holds": all(checks.values()), "changed_since_stage1": moved,
            "stage1_commit": stage1[0] if stage1 else None, "stage2_commit": stage2[0] if stage2 else None}


def unseen() -> tuple[tuple[str, ...], Any, dict[str, str]]:
    """The reviewer's kinds of breakage, their function and the law each breaks; none in development."""
    try:
        module = importlib.import_module(UNSEEN_MODULE)
    except ModuleNotFoundError:
        return (), None, {}
    kinds = tuple(module.UNSEEN_KINDS)
    clash = set(kinds) & {"true", *PERTURBATIONS}
    if clash or len(set(kinds)) != len(kinds):
        raise SystemExit(f"the reviewer's kinds must be new names: {sorted(clash) or kinds}")
    return kinds, module.perturb_unseen, dict(module.LAWS)


def verdicts(kind: str, tracks: list[Any], seed: int, kinds_unseen: tuple[str, ...],
             perturb_unseen: Any) -> list[dict[str, Any]]:
    """The true clip and every breakage of it. A judge error is a row: a true clip that errs does not pass, a
    broken clip that errs is not caught. Each breakage gets its own copy of the tracks."""
    rows = []
    try:
        v = judge(kind, copy.deepcopy(tracks))
        rows.append({"variant": "true", "outcome": v.outcome, "reasons": v.reasons[:3]})
    except ERRORS as exc:
        rows.append({"variant": "true", "outcome": "error", "reasons": [f"{type(exc).__name__}: {exc}"[:200]]})
    for group, hows, fn in (("development", PERTURBATIONS, perturb), ("unseen", kinds_unseen, perturb_unseen)):
        for how in hows:
            try:
                broken = fn(kind, copy.deepcopy(tracks), how, stream(seed, how))
            except ERRORS as exc:          # the breakage itself failed: recorded, not counted against the judge
                rows.append({"variant": how, "group": group, "outcome": "not applicable",
                             "reasons": [f"the breakage failed: {type(exc).__name__}: {exc}"[:200]]})
                continue
            if broken is None:
                rows.append({"variant": how, "group": group, "outcome": "not applicable"})
                continue
            try:
                w = judge(kind, broken)
                rows.append({"variant": how, "group": group, "outcome": w.outcome, "reasons": w.reasons[:3]})
            except ERRORS as exc:
                rows.append({"variant": how, "group": group, "outcome": "error",
                             "reasons": [f"{type(exc).__name__}: {exc}"[:200]]})
    return rows


def _loaded() -> dict[str, str]:
    out = {}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix == ".py" and path.is_file() and ROOT in path.parents:
            out[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(out.items()))


def _versions() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        bullet = version("pybullet")
    except PackageNotFoundError:
        bullet = None
    return {"python": platform.python_version(), "numpy": np.__version__, "pybullet": bullet}


def cmd_rendered(args: argparse.Namespace) -> int:
    from apedp_dyn.video.physics_clips_pybullet import make_clip

    if args.mode == "final":
        check_frozen()
    kinds_unseen, perturb_unseen, _ = unseen() if args.mode == "final" else ((), None, {})
    if args.mode == "final" and len(kinds_unseen) < MIN_UNSEEN_KINDS:
        raise SystemExit("the reviewer's breakages are missing")
    if args.mode == "final" and reading() is None:
        raise SystemExit("the owner has not chosen the reading of gate J (PHYSICS-J1-READING.json)")
    k, n = (int(v) for v in args.shard.split("/"))
    units = [(kind, s) for kind in GATE_KINDS for s in seeds(args.mode)]
    rows: list[dict[str, Any]] = []
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"rendered-{args.mode}-{k:03d}-of-{n:03d}.json"
    for i, (kind, seed) in enumerate(units):
        if i % n != k:
            continue
        try:
            tracks, facts = make_clip(kind, seed, allow_blind=args.mode == "final")
        except (ValueError, RuntimeError, OSError) as exc:     # counted against the judge: no row is left out
            why = [f"rendering: {type(exc).__name__}: {exc}"[:200]]
            for variant in ("true", *PERTURBATIONS, *kinds_unseen):
                rows.append({"source": "rendered", "kind": kind, "seed": seed, "variant": variant, "outcome": "error",
                             "reasons": why, **({} if variant == "true" else {"group": "unseen" if variant in kinds_unseen
                                                                             else "development"})})
            continue
        for row in verdicts(kind, tracks, seed, kinds_unseen, perturb_unseen):
            if row["variant"] == "true":
                row["facts"] = facts             # the scene and camera, this repository's own
            rows.append({"source": "rendered", "kind": kind, "seed": seed, **row})
        path.write_text(json.dumps({"rows": rows, "complete": False}) + "\n")     # kept as it goes
        print(kind, seed, Counter(r["outcome"] for r in rows if r["seed"] == seed and r["kind"] == kind), flush=True)
    path.write_text(json.dumps({"rows": rows, "complete": True, "loaded": _loaded(), "versions": _versions()}) + "\n")
    return 0


def cmd_physinone(args: argparse.Namespace) -> int:
    import run_physics_judge_dev as dev

    check_frozen()
    kinds_unseen, perturb_unseen, _ = unseen()
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
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:   # recorded
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
    (args.out / f"physinone-{args.kind}.json").write_text(json.dumps({"rows": rows, "complete": True}) + "\n")
    print(args.kind, Counter((r["variant"], r["outcome"]) for r in rows), flush=True)
    return 0


def bound(k: int, n: int, upper: bool) -> float | None:
    """One-sided Clopper-Pearson bound on a share k of n, at BOUND_CONFIDENCE."""
    from scipy.stats import beta

    if n == 0:
        return None
    if upper:
        return 1.0 if k >= n else float(beta.ppf(BOUND_CONFIDENCE, k + 1, n - k))
    return 0.0 if k == 0 else float(beta.ppf(1 - BOUND_CONFIDENCE, k, n - k + 1))


def _clip(r: dict[str, Any]) -> tuple[Any, str]:
    return (r.get("seed", r.get("scene")), r["kind"])


def cell(rows: list[dict[str, Any]], true_passed: set[tuple[Any, str]] | None = None) -> dict[str, Any]:
    """One scene kind (or several) and one variant: a true clip counts only if it passes, a broken one only if it
    fails (untestable, pass and error are not caught)."""
    counts = Counter(r["outcome"] for r in rows)
    applied = sum(v for o, v in counts.items() if o not in NOT_COUNTED)
    if rows and rows[0]["variant"] == "true":
        good = counts.get(PASS, 0)
        return {"applied": applied, "passing": good, "share": round(good / applied, 4) if applied else None,
                "lower_bound": bound(good, applied, upper=False), "outcomes": dict(sorted(counts.items()))}
    missed = applied - counts.get(FAIL, 0)
    out: dict[str, Any] = {"applied": applied, "not_caught": missed,
                           "share": round(missed / applied, 4) if applied else None,
                           "upper_bound": bound(missed, applied, upper=True), "untestable": counts.get("untestable", 0),
                           "outcomes": dict(sorted(counts.items())), "group": rows[0].get("group") if rows else None}
    if true_passed is not None:
        kept = [r for r in rows if _clip(r) in true_passed and r["outcome"] not in NOT_COUNTED]
        out["not_caught_where_true_passed"] = f"{sum(r['outcome'] != FAIL for r in kept)} of {len(kept)}"
    return out


def tables(rows: list[dict[str, Any]], kinds: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    passed = {_clip(r) for r in rows if r["variant"] == "true" and r["outcome"] == PASS}
    out: dict[str, dict[str, Any]] = {}
    for kind in (*kinds, "pooled"):
        mine = [r for r in rows if r["kind"] in kinds] if kind == "pooled" else [r for r in rows if r["kind"] == kind]
        out[kind] = {v: cell([r for r in mine if r["variant"] == v], None if v == "true" else passed)
                     for v in sorted({r["variant"] for r in mine})}
    return out


def _misses(variants: dict[str, Any], label: str) -> list[str]:
    failures = []
    true = variants.get("true", {"applied": 0, "passing": 0})
    if not true["applied"] or true["passing"] < TRUE_PASS_MIN * true["applied"]:
        failures.append(f"{label}: {true['passing']} of {true['applied']} true clips pass")
    for v, c in variants.items():
        if v == "true" or not c["applied"]:
            continue
        if c["applied"] >= MIN_APPLIED:
            if c["not_caught"] > BROKEN_PASS_MAX * c["applied"]:
                failures.append(f"{label} {v}: {c['not_caught']} of {c['applied']} not caught")
        elif c["not_caught"] > SMALL_CELL_MISSES:
            failures.append(f"{label} {v}: {c['not_caught']} of {c['applied']} not caught (a small cell)")
    return failures


def reading() -> str | None:
    """The owner's choice of reading for gate J, recorded before stage 2, or ``None`` if not yet given."""
    if not READING_FILE.exists():
        return None
    chosen = json.loads(READING_FILE.read_text(encoding="utf-8")).get("reading")
    if chosen not in READINGS:
        raise SystemExit(f"{READING_FILE.name} must name one of {READINGS}")
    return str(chosen)


def gate(rows: list[dict[str, Any]], expected: dict[str, list[int]] | None = None,
         kinds_unseen: tuple[str, ...] = (), chosen: str | None = None) -> dict[str, Any]:
    """Gate J part 1: holds, fails, or not read; both readings are always reported.

    ``per_kind``: holds only if every scene kind holds and the pooled reading holds (small cells can each hold while
    their sum fails). ``pooled``: holds if the pooled reading holds. Gate J's text does not say which; the owner
    chooses before stage 2. Without a choice the state is "no reading chosen".
    """
    rendered = [r for r in rows if r["source"] == "rendered"]
    if expected is not None:
        seen = Counter((r["kind"], r["seed"], r["variant"]) for r in rendered)
        wanted = {(k, s, v) for k, ss in expected.items() for s in ss for v in ("true", *PERTURBATIONS, *kinds_unseen)}
        if set(seen) != wanted or any(n != 1 for n in seen.values()):
            return {"state": "not read", "why": "the rows are not every clip and every breakage exactly once",
                    "missing": len(wanted - set(seen)), "extra": len(set(seen) - wanted)}
    table = tables(rendered, GATE_KINDS)
    per_kind = [m for kind in GATE_KINDS for m in _misses(table[kind], kind)]
    pooled = _misses(table["pooled"], "pooled")
    read_unseen = [h for h in kinds_unseen if table["pooled"].get(h, {}).get("applied", 0) >= MIN_APPLIED]
    limits = {"true_pass_min": TRUE_PASS_MIN, "broken_pass_max": BROKEN_PASS_MAX, "min_applied": MIN_APPLIED,
              "small_cell_misses": SMALL_CELL_MISSES, "min_unseen_kinds": MIN_UNSEEN_KINDS}
    base = {"per_kind_failures": per_kind, "pooled_failures": pooled, "table": table, "limits": limits}
    if expected is not None and (len(kinds_unseen) < MIN_UNSEEN_KINDS or len(read_unseen) != len(kinds_unseen)):
        return {"state": "not read", "why": f"fewer than two reviewer kinds, or one applied to fewer than {MIN_APPLIED} clips",
                **base}
    by_reading = {"per_kind": "holds" if not per_kind and not pooled else "fails",
                  "pooled": "holds" if not pooled else "fails"}
    state = by_reading[chosen] if chosen in by_reading else "no reading chosen"
    return {"state": state, "reading": chosen, "by_reading": by_reading, **base}


def cmd_collect(args: argparse.Namespace) -> int:
    if args.mode == "final":
        check_frozen()
    files = [p for p in sorted(args.out.rglob("*.json")) if p.name.startswith(("rendered-", "physinone-"))]
    parts = [json.loads(p.read_text()) for p in files]
    rows = [r for part in parts for r in part["rows"]]
    complete = bool(parts) and all(part.get("complete") for part in parts)
    changed = sorted({f for part in parts for f, h in (part.get("loaded") or {}).items()
                      if not (ROOT / f).is_file() or hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h})
    kinds_unseen = unseen()[0] if args.mode == "final" else ()
    expected = {k: seeds(args.mode) for k in GATE_KINDS}
    reading_ = gate(rows, expected, kinds_unseen, reading()) if args.mode == "final" else gate(rows, expected)
    if not complete or changed:
        reading_ = {**reading_, "state": "not read",
                   "why": "a shard did not finish" if not complete else "loaded files differ from the checkout",
                   "changed": changed[:10]}
    record: dict[str, Any] = {
        "preregistration": PREREGISTRATION_ID, "mode": args.mode, "rows": len(rows), "generator": "own, exact masks",
        "gate_j_part_1": reading_, "laws_broken": unseen()[2] if args.mode == "final" else {},
        "physinone_beside": tables([r for r in rows if r["source"] == "physinone"], GATE_KINDS),
        "versions": sorted({json.dumps(part.get("versions"), sort_keys=True) for part in parts if part.get("versions")}),
        "history": history({f for part in parts for f in (part.get("loaded") or {})}) if args.mode == "final" else None,
        "misses": [{k: r.get(k) for k in ("source", "kind", "seed", "scene", "variant", "outcome", "reasons")}
                   for r in rows if (r["variant"] == "true") == (r["outcome"] != PASS)
                   and r["outcome"] not in NOT_COUNTED],
    }
    if args.mode == "final":
        record["frozen"] = digests(STAGE2_FILES)
        if not (record["history"] or {}).get("holds"):
            record["gate_j_part_1"] = {**record["gate_j_part_1"], "state": "not read",
                                       "why": "git history does not show the freeze as the plan requires"}
    (args.out / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    shown = {k: v for k, v in record["gate_j_part_1"].items() if k != "table"}
    print(json.dumps(shown, indent=1)[:6000])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("rendered", "physinone", "collect"))
    parser.add_argument("--mode", choices=("dev", "final"), default="final")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--kind", choices=GATE_KINDS)
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

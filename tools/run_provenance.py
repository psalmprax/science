"""Make a test run's provenance explicit, and detect when it goes stale.

Why this exists
---------------
I once spent 56 minutes running the full suite, let it execute across a commit
that fixed the very bug it was reporting, and then cited its result eight hours
later as evidence -- without checking the run's timestamp against the commit
log. The log showed five failures. All five were artifacts of the code changing
underneath the run, and all five passed when re-run afterwards. I built a whole
conclusion ("these tests fail in the full suite but pass alone") on a stale log.

The failure was not a subtle one. The evidence was wrong in a way any timestamp
comparison would have caught, and I did not make that comparison, because
nothing prompted me to. So the rule is now a tool.

What this can and cannot do
---------------------------
A test result is evidence about the code that produced it. This makes the
provenance of a run recoverable after the fact, and flags the two ways a run
stops being evidence:

1. **Staleness** -- a commit landed, after the run started, touching code the
   run exercised. The run's failures may describe code that no longer exists.

2. **Drift** -- the working tree changed during the run. On a shared tree with
   several concurrent sessions this is the common case, and it is invisible
   unless someone hashes the files before and after.

Deliberately warns rather than blocks
------------------------------------
Like the revert guard, this exits 0. A stale log is usually still worth reading
-- it tells you what the code did at that moment, and a regression it caught may
be real. The point is that nobody should cite it as current without saying so.

Limits, stated rather than hidden
---------------------------------
It cannot tell which session changed a file, only that it changed. It cannot
recover the code a run actually executed if the tree moved mid-run; the manifest
records the *launch* state, so drift is detected, not corrected. And a clean
manifest proves the tree was still at launch, not that the run reached the end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: Files whose content defines a run's meaning. Hashing the whole tree is
#: cheap enough that there is no reason to narrow it, and narrowing it is how a
#: manifest ends up looking authoritative while missing the file that changed.
DEFAULT_SCAN_DIRS = ("apedp", "tests")

MANIFEST_NAME = "run_provenance.json"


class GitCommandError(RuntimeError):
    """A git invocation failed.

    This type exists because the previous ``_git`` returned ``proc.stdout`` and
    dropped both the return code and stderr, which made every git failure
    indistinguishable from a git success that had nothing to say. Measured:
    ``_git(["log", "--since=@NaN"])`` returned ``''`` -- git exited 128, wrote
    the diagnosis to stderr, and the tool reported it as "no commits found".

    That is not a hypothetical. It is the same defect as the ``--since=@1``
    floor this file already documents, one level down: there, a near-epoch
    ``--since`` made git legitimately return nothing and the tool concluded no
    code had changed. Here git was never asked the question at all. Both
    surface as the same silent pass, and fixing only the timestamp floor left
    the mechanism live for every other malformed argument.

    Raising is the right response for a check rather than a report: this tool
    warns about staleness but must not be able to manufacture a clean result
    from a command that failed.
    """


def _git(args: list[str], repo: Path | None = None) -> str:
    """Run git and return stdout, or raise with git's own diagnosis attached."""
    # S607: this tool's entire purpose is to ask git about history, so resolving
    # `git` from PATH is the design. Arguments are fixed revs and paths from this
    # repository, never user input, and no shell is involved.
    proc = subprocess.run(  # noqa: S603
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,  # noqa: S607
    )
    if proc.returncode != 0:
        # Include stderr and the argv, because "git failed" with no argument
        # list is not diagnosable and the whole cost of raising is the lost
        # run. stdout is appended too: `git rev-parse <bad-rev>` exits 128 yet
        # echoes the bad name back on stdout, so dropping stderr alone would
        # have left that call still looking like it succeeded.
        raise GitCommandError(
            f"git {' '.join(args)} exited {proc.returncode}"
            f"\nstderr: {proc.stderr.strip() or '(empty)'}"
            f"\nstdout: {proc.stdout.strip() or '(empty)'}"
        )
    return proc.stdout





def resolve_repo(repo: Path | None = None) -> Path:
    """Find the repo root, refusing to guess.

    A manifest written against the wrong tree is worse than no manifest, because
    it looks authoritative. So this raises rather than falling back to a
    hardcoded path.
    """
    if repo is not None:
        root = repo.expanduser().resolve()
        if not (root / ".git").exists():
            raise FileNotFoundError(f"not a git repository: {root}")
        return root
    here = Path(__file__).resolve().parent
    for cand in (here.parent, *here.parents):
        if (cand / ".git").exists():
            return cand
    cwd = Path.cwd()
    for cand in (cwd, *cwd.parents):
        if (cand / ".git").exists():
            return cand
    raise FileNotFoundError(
        "could not locate a git repository; pass --repo explicitly"
    )


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_tree(
    repo: Path, scan_dirs: tuple[str, ...] = DEFAULT_SCAN_DIRS
) -> dict[str, str]:
    """Content-hash every .py file under the scan dirs.

    Deliberately does not use ``git ls-files``: that would omit untracked
    modules, and on this tree the untracked files are exactly the ones a
    concurrent session is most likely to be editing.
    """
    out: dict[str, str] = {}
    for d in scan_dirs:
        base = repo / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if not p.is_file():
                continue
            out[str(p.relative_to(repo))] = _hash(p)
    return out


def head_commit(repo: Path) -> str:
    return _git(["rev-parse", "HEAD"], repo).strip()


def head_subject(repo: Path) -> str:
    return _git(["log", "-1", "--format=%s"], repo).strip()


def snapshot(
    repo: Path,
    scan_dirs: tuple[str, ...] = DEFAULT_SCAN_DIRS,
) -> dict[str, object]:
    """Record the state a run is about to execute against."""
    return {
        "manifest_version": 1,
        "repo": str(repo),
        "head": head_commit(repo),
        "head_subject": head_subject(repo),
        # Seconds as a float, not an ISO string: comparisons happen in code, and
        # a string here would invite lexicographic date comparison.
        "launched_utc": datetime.now(timezone.utc).timestamp(),
        "launched_iso": datetime.now(timezone.utc).isoformat(),
        "scan_dirs": list(scan_dirs),
        "files": scan_tree(repo, scan_dirs),
    }


def write_manifest(manifest: dict[str, object], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(path: Path) -> dict[str, object]:
    # A hand-edited or truncated manifest should not crash the check with an
    # obscure AttributeError; the accessors treat a malformed field as absent.
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    return data


@dataclass
class Drift:
    """Files that changed between launch and verification."""

    edited: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(self.edited or self.added or self.removed)

    def message(self) -> str:
        lines = []
        for label, items in (
            ("edited during the run", self.edited),
            ("added during the run", self.added),
            ("removed during the run", self.removed),
        ):
            for p in items[:20]:
                lines.append(f"    {label}: {p}")
            if len(items) > 20:
                lines.append(f"    ... and {len(items) - 20} more {label}")
        return "\n".join(lines)


def diff_trees(
    before: dict[str, str], after: dict[str, str]
) -> Drift:
    d = Drift()
    for path, old in before.items():
        if path not in after:
            d.removed.append(path)
        elif after[path] != old:
            d.edited.append(path)
    for path in after:
        if path not in before:
            d.added.append(path)
    return d


# --- typed accessors for the manifest -------------------------------------
# The manifest is JSON, so it arrives as dict[str, object] and every read needs
# a narrowing cast. Funnelling those through four named helpers keeps the casts
# in one place; scattering them inline is how the mypy suppressions grew.

def _manifest_str(m: dict[str, object], key: str, default: str = "") -> str:
    v = m.get(key, default)
    return v if isinstance(v, str) else default


def _manifest_float(m: dict[str, object], key: str, default: float = 0.0) -> float:
    v = m.get(key, default)
    return float(v) if isinstance(v, (int, float)) else default


def _manifest_files(m: dict[str, object]) -> dict[str, str]:
    v = m.get("files")
    if not isinstance(v, dict):
        return {}
    return {k: h for k, h in v.items() if isinstance(k, str) and isinstance(h, str)}


def _manifest_dirs(m: dict[str, object]) -> tuple[str, ...]:
    v = m.get("scan_dirs")
    if not isinstance(v, list) or not all(isinstance(d, str) for d in v):
        return DEFAULT_SCAN_DIRS
    return tuple(v)


@dataclass
class StaleCommit:
    rev: str
    subject: str
    committed: str
    paths: list[str]


def commits_since(
    repo: Path, since: float, scan_dirs: tuple[str, ...] = DEFAULT_SCAN_DIRS
) -> list[StaleCommit]:
    """Commits landing after ``since`` that touch the scanned code.

    Filters by path, not just by recency. A run is only invalidated by changes
    to code it could have exercised, so a commit touching only ``docs/`` says
    nothing about a test result -- and treating it as though it did would train
    people to ignore the warning.
    """
    # The leading \x01 is an explicit record separator. Without one, git
    # separates commits only by the shape of the header line, so parsing by
    # "lines that look like headers" silently merges every commit's file list
    # into the first commit -- which reported 330 files for a 5-file commit.
    # %ct is committer time as integer seconds, which is what we compare
    # numerically below.
    fmt = "%x01%H%x1f%s%x1f%cI%x1f%ct"
    # Ask git for one second of slack, then filter exactly in Python below.
    # The slack is redundant for correctness -- the Python-side comparison is
    # what actually decides -- so two of the three guards here are
    # unkillable by mutation, and that is deliberate. Its job is to bound the
    # query git has to walk: a manifest from a week-old run should not make
    # git diff the entire history on a repo this size.
    #
    # The floor is 1973-03-03, not 1. The previous floor of 1 was chosen to
    # dodge `--since=@0`, and it is a real value, but git 2.43 silently returns
    # NOTHING for `--since` at or near the epoch: on this machine
    # `--since=@1`, `@1000000` and `@31536000` each yield 0 commits while
    # `@100000000` yields all 80. It is timezone-dependent too -- `@1` works
    # under TZ=America/New_York and returns nothing under UTC or Europe/Berlin.
    # So "since the beginning of time" reported zero commits, the tool
    # concluded no code had changed, and
    # test_the_real_last_commit_reports_its_true_file_count failed with "no
    # commits touching scanned code found in history" -- a provenance check
    # reporting that there is nothing to check. 1973 predates any repository
    # this could run against, and is far enough past the epoch that git parses
    # it as the instant it is.
    since_ts = max(100_000_000, int(since) - 1)
    out = _git(["log", f"--format={fmt}", "--name-only", f"--since=@{since_ts}"],
               repo)
    if not out.strip():
        return []

    prefixes = tuple(f"{d}/" for d in scan_dirs)
    commits: list[StaleCommit] = []
    for chunk in out.split("\x01"):
        if not chunk.strip():
            continue
        head, _, rest = chunk.partition("\n")
        fields = head.strip().split("\x1f")
        if len(fields) < 4 or not fields[0]:
            continue
        rev, subject, committed, ctime = fields[0], fields[1], fields[2], fields[3]
        try:
            if float(ctime) <= since:
                continue
        except ValueError:
            continue
        paths = [
            ln.strip()
            for ln in rest.splitlines()
            if ln.strip() and ln.strip().startswith(prefixes)
        ]
        if paths:
            commits.append(StaleCommit(rev, subject, committed, paths))
    return commits


def verify(
    manifest: dict[str, object], repo: Path
) -> tuple[Drift, list[StaleCommit]]:
    """Compare the current tree and history against a launch manifest."""
    scan_dirs = _manifest_dirs(manifest)
    drift = diff_trees(_manifest_files(manifest), scan_tree(repo, scan_dirs))
    stale = commits_since(repo, _manifest_float(manifest, "launched_utc"),
                          scan_dirs)
    return drift, stale


def is_stale(manifest: dict[str, object], repo: Path) -> bool:
    """True if anything at all moved since the run launched."""
    drift, stale = verify(manifest, repo)
    current = head_commit(repo)
    return bool(stale or drift.any
                or (current and current != _manifest_str(manifest, "head")))


def render_report(
    manifest: dict[str, object], drift: Drift, stale: list[StaleCommit]
) -> str:
    out: list[str] = []
    out.append("Run provenance check")
    out.append(f"  launched: {_manifest_str(manifest, 'launched_iso')} (UTC)")
    out.append(f"  head:     {_manifest_str(manifest, 'head')[:12]} "
               f"{_manifest_str(manifest, 'head_subject')}")
    out.append("")

    current = head_commit(Path(_manifest_str(manifest, "repo")))
    head_moved = bool(current) and current != _manifest_str(manifest, "head")
    if head_moved:
        out.append(
            f"  STALE: HEAD is now {current[:12]}, the run started at "
            f"{_manifest_str(manifest, 'head')[:12]}"
        )
        out.append("")

    if stale:
        out.append(
            f"  STALE: {len(stale)} commit(s) touched the code under test after "
            "this run launched:"
        )
        for c in stale[:10]:
            out.append(f"    {c.rev[:12]} {c.committed}  {c.subject}")
            for p in c.paths[:5]:
                out.append(f"        {p}")
            if len(c.paths) > 5:
                out.append(f"        ... and {len(c.paths) - 5} more files")
        out.append("")

    if drift.any:
        out.append(
            "  DRIFT: the working tree changed while the run was in flight "
            "(expected on a shared tree, but it means the run may have imported "
            "a mix of revisions):"
        )
        out.append(drift.message())
        out.append("")

    # The three conditions are independent, so the verdict has to consider all
    # of them. An earlier version keyed "CLEAN" off `stale` and `drift` alone,
    # which printed STALE and CLEAN in the same report whenever HEAD had moved
    # without a commit touching the scanned dirs -- the reader is left unsure
    # which one to believe, and a report that contradicts itself is worthless.
    if stale or drift.any or head_moved:
        out.append(
            "  A result from this run describes the code as it was at launch, "
            "not as it is now."
        )
    else:
        out.append("  CLEAN: no commits and no file changes since launch.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record a test run's provenance and detect stale results.",
    )
    parser.add_argument(
        "command", choices=("launch", "verify"), nargs="?", default="verify",
        help="launch: write a manifest now. verify: check a manifest.",
    )
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help=f"manifest path (default ./{MANIFEST_NAME})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where to write the manifest on launch")
    parser.add_argument("--scan-dirs", nargs="*", default=None)
    args = parser.parse_args(argv)

    try:
        repo = resolve_repo(args.repo)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    scan_dirs = tuple(args.scan_dirs) if args.scan_dirs else DEFAULT_SCAN_DIRS

    if args.command == "launch":
        manifest = snapshot(repo, scan_dirs)  # type: ignore[arg-type]
        target = (
            write_manifest(manifest, args.out_dir)
            if args.out_dir
            else write_manifest(
                manifest, Path(args.manifest).parent
                if args.manifest else Path.cwd()
            )
        )
        print(f"wrote {target}")
        print(f"  head:   {_manifest_str(manifest, 'head')[:12]} "
              f"{_manifest_str(manifest, 'head_subject')}")
        print(f"  files:  {len(_manifest_files(manifest))} hashed")
        print(f"  launch: {_manifest_str(manifest, 'launched_iso')}")
        return 0

    mpath = args.manifest or (Path.cwd() / MANIFEST_NAME)
    if not mpath.exists():
        print(
            f"ERROR: no manifest at {mpath}. Run `launch` before the test run.",
            file=sys.stderr,
        )
        return 2
    manifest = read_manifest(mpath)
    drift, stale = verify(manifest, repo)
    report = render_report(manifest, drift, stale)
    if is_stale(manifest, repo):
        print(report, file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Cite this run only with the caveat above, or re-run it. A result "
            "is evidence about the code that produced it.",
            file=sys.stderr,
        )
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

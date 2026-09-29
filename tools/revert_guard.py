"""Guard against committing a wholesale revert of a file to an older revision.

Why this exists
---------------
This working tree is shared by several concurrent sessions. Early in this
project I ran ``git checkout -- <file>`` intending to revert only my own hunk
and destroyed another session's uncommitted changes to the same file. The work
was recoverable only because part of it had been staged, leaving a dangling
blob.

What this can and cannot do
---------------------------
Git has no ``pre-checkout`` hook, so **nothing can prevent** that command. This
guard therefore does not pretend to be prevention. It does two things that are
actually possible:

1. **Detects a wholesale revert at commit time.** The preventable form of the
   incident is committing a file back to an older revision wholesale
   (``git checkout <old-rev> -- <file> && git add <file> && git commit``). That
   is visible in the object store, so a ``pre-commit`` hook can catch it.

2. **Documents the recovery path** for work already lost. Unstaged work is
   gone with no trace; staged work survives as a dangling blob. The warning
   says which case you are in, because the two have very different
   recoverability and pretending otherwise would be worse than silence.

Deliberately warns rather than blocks
-------------------------------------
Legitimately reverting a file wholesale is a real operation -- a bad commit
does need undoing. A guard that blocks a correct action gets bypassed, and a
routinely bypassed guard is worse than an honest warning. So this exits 0 and
prints what it found plus the exact recovery command.

Limits, stated rather than hidden
---------------------------------
It reads the staged index, not the worktree. It cannot tell which session
authored a line. It cannot see unstaged work at all, and the unstaged case is
precisely the one that cannot be recovered.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: How far back to look for the revision a file was reverted to. Deep enough
#: for a recent bad commit, shallow enough to stay fast.
DEFAULT_LOOKBACK = 25

#: A revert is only worth warning about if the reverted-away content was
#: substantial. One-line churn is not the failure mode that lost work.
MIN_REVERTED_LINES = 5

RECOVERY_HINT = """\
Recover lost work:
    git fsck --no-reflogs --unreachable   # list dangling blobs
    git cat-file -p <blob>               # inspect one
    git show <blob> > <path>             # restore it

Staged work survives a `git checkout --` this way. Unstaged work does not
survive at all, and no tool can recover it. That asymmetry is why the rule
exists: on a shared tree, revert your own hunks, never the whole file."""


@dataclass
class RevertFinding:
    """One file that a commit would restore to an older revision wholesale."""

    path: str
    reverted_to: str
    reverted_from: str
    lines_discarded: int
    dangling_blobs: list[str] = field(default_factory=list)

    def message(self) -> str:
        out = [
            f"  {self.path}: this commit restores the file to {self.reverted_to}, "
            f"discarding {self.lines_discarded} lines added by {self.reverted_from}",
        ]
        if self.dangling_blobs:
            out.append(
                "    discarded content is recoverable from blob(s): "
                + ", ".join(self.dangling_blobs)
            )
        else:
            out.append(
                "    no dangling blob found: the discarded content was never "
                "staged, so it is unrecoverable"
            )
        return "\n".join(out)


def _git(args: list[str], repo: Path | None = None) -> str:
    # S607: this tool's entire purpose is to ask git about the object store,
    # so resolving `git` from PATH is the design, not a risk. The arguments are
    # paths and revisions from this repository, never user input, and the
    # subprocess runs with no shell, so there is nothing to inject through.
    proc = subprocess.run(  # noqa: S603
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,  # noqa: S607
    )
    return proc.stdout


def staged_paths(repo: Path | None = None) -> list[str]:
    return [
        ln for ln in _git(["diff", "--cached", "--name-only"], repo).splitlines()
        if ln.strip()
    ]


def index_blob(path: str, repo: Path | None = None) -> str | None:
    """The blob the commit would record for ``path`` (None if unchanged)."""
    out = _git(["ls-files", "-s", "--", path], repo).split()
    return out[1] if len(out) >= 2 else None


def blob_at(rev: str, path: str, repo: Path | None = None) -> str | None:
    out = _git(["rev-parse", f"{rev}:{path}"], repo).strip()
    return out if out and set(out) != {"\n"} else None


def _lines_between(repo: Path | None, old: str | None, new: str | None,
                   path: str) -> int:
    """Changed-line count between two blobs of the same path."""
    if old is None or new is None:
        return 0
    out = _git(["diff", "--numstat", new, old, "--"], repo)
    total = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] != "-":
            total += int(parts[0]) + int(parts[1])
    return total


def find_reverts(
    paths: list[str] | None = None,
    repo: Path | None = None,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    min_lines: int = MIN_REVERTED_LINES,
) -> list[RevertFinding]:
    """Finds staged files whose content matches an older revision wholesale.

    The signature is the whole signal: a legitimate edit produces a blob that
    exists nowhere in history, while a wholesale revert reproduces a blob
    verbatim. That is a property of the object store, not a heuristic, which
    is why this is detectable at all.
    """
    paths = staged_paths(repo) if paths is None else paths
    head = _git(["rev-parse", "HEAD"], repo).strip()
    if not head:
        return []
    revs = _git(
        ["rev-list", f"-n{lookback}", "HEAD"], repo
    ).split()

    findings: list[RevertFinding] = []
    for path in paths:
        new_blob = index_blob(path, repo)
        if new_blob is None:
            continue
        if new_blob == blob_at("HEAD", path, repo):
            continue  # unchanged relative to HEAD; not a revert
        for rev in revs:
            old_blob = blob_at(rev, path, repo)
            if old_blob is None or old_blob != new_blob:
                continue
            # Found a revision whose content this commit would restore.
            newer = _git(
                ["rev-list", f"-n{lookback}", f"{rev}..HEAD"], repo
            ).split()
            from_rev = newer[-1] if newer else head
            discarded = _lines_between(repo, old_blob,
                                       blob_at(from_rev, path, repo), path)
            if discarded < min_lines:
                continue
            findings.append(RevertFinding(
                path=path,
                reverted_to=rev[:12],
                reverted_from=from_rev[:12],
                lines_discarded=discarded,
            ))
            break
    return findings


def dangling_blobs(repo: Path | None = None) -> list[str]:
    """Unreachable blobs, which are where staged-then-lost work survives.

    Reported without claiming any of them is the right one: only whoever lost
    the work knows what it should have said, and a confident wrong pointer
    would be worse than a list.
    """
    out: list[str] = []
    text = _git(["fsck", "--no-reflogs", "--unreachable"], repo)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "blob" and parts[0] in {
            "dangling", "unreachable",
        }:
            out.append(parts[2])
    return out


def render(findings: list[RevertFinding]) -> str:
    return "\n".join(f.message() for f in findings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Warn on commits that wholesale-restore a file to an "
                    "older revision.",
    )
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--paths", nargs="*", default=None)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--min-lines", type=int, default=MIN_REVERTED_LINES)
    parser.add_argument("--check-blobs", action="store_true")
    args = parser.parse_args(argv)

    findings = find_reverts(
        args.paths, args.repo, lookback=args.lookback, min_lines=args.min_lines,
    )
    if not findings:
        return 0

    if args.check_blobs:
        blobs = dangling_blobs(args.repo)
        for f in findings:
            f.dangling_blobs = blobs[:5]

    print("WARNING: this commit wholesale-reverts a file.", file=sys.stderr)
    print(render(findings), file=sys.stderr)
    print("", file=sys.stderr)
    print(RECOVERY_HINT, file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "If this revert is deliberate and affects only your own changes, "
        "ignore this warning (or re-run the hook with --no-verify).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

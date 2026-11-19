#!/usr/bin/env python3
"""Split the combined `kickmix` patch into an `MR` patch and a `gf_product` patch.

``scripts/kickmix_mr_opcode.patch`` carries two independent changes: the ``MR``
Z-basis read and the 54-AND field-multiply policy. Reviewers should see them
separately -- 441 lines for one opcode is already a lot, and the multiply is
reviewable on its own terms -- but they are not file-separable, because the
``gf_product`` enum and property live in ``circuit_builder.h``/``.cc`` and their
pybind, which ``MR`` also edits.

So the split is at *hunk* granularity: a hunk belongs to the multiply patch if
its added or removed lines mention the policy (``gf_product``, ``GFProduct``,
``and54``, ``gf_mul54``, ``gen_gf54``, ``clean_workspace_for_product``), and to
the ``MR`` patch otherwise. Whole new files (``gen_gf_mul54.*``) go to the
multiply patch.

The output is ``build/kickmix_split/{mr,gf54}.patch``; ``mr.patch`` applies to a
clean tree, ``gf54.patch`` applies on top of it. Both are checked with
``patch --dry-run`` before this script reports success, because a split that does
not apply is worse than the combined patch it came from.

    python3.13 scripts/split_kickmix_patch.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH = ROOT / "scripts" / "kickmix_mr_opcode.patch"
VENDOR = (ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
          / "kickmix" / "_qualtran_src" / "kickmix")
OUT = ROOT / "build" / "kickmix_split"

MULTIPLY_TOKENS = ("gf_product", "GFProduct", "and54", "gf_mul54", "gen_gf54",
                   "clean_workspace_for_product")
MULTIPLY_FILES = ("gen/gf_arithmetic/gen_gf_mul54.cc", "gen/gf_arithmetic/gen_gf_mul54.h",
                  "gen/gf_arithmetic/gen_gf_mul.cc",
                  "py/circuit/subs/gf_arithmetic.pybind.cc")


def split_files(text: str) -> list[tuple[str, list[str]]]:
    """Split a unified diff into ``(path, lines)`` blocks, one per file."""
    blocks, current = [], None
    for line in text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                blocks.append(current)
            current = ("", [line])
        elif current is not None:
            current[1].append(line)
    if current:
        blocks.append(current)
    result = []
    for _, lines in blocks:
        path = re.match(r"diff --git a/(\S+) ", lines[0]).group(1)
        result.append((path, lines))
    return result


def split_hunks(header: list[str], body: list[str]) -> list[list[str]]:
    """Group the body by ``@@`` hunk, keeping ``header`` (the file preamble) first."""
    hunks, current = [], None
    for line in body:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
            current = [line]
        elif current is not None:
            current.append(line)
    if current:
        hunks.append(current)
    return hunks


def classify(hunk: list[str]) -> bool:
    """True if the hunk belongs to the multiply patch."""
    return any(tok in line for line in hunk if line[:1] in "+-"
               for tok in MULTIPLY_TOKENS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch", type=Path, default=PATCH)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    blocks = split_files(args.patch.read_text(encoding="utf-8"))
    mr_blocks, mul_blocks = [], []
    for path, lines in blocks:
        header = [ln for ln in lines if not ln.startswith(("@@", "+", "-", " ", "\\"))
                  or ln.startswith(("+++", "---"))]
        body_start = len(header)
        preamble, hunks = lines[:body_start], split_hunks(header, lines[body_start:])
        if path in MULTIPLY_FILES:
            mul_blocks.append(lines)
            continue
        keep_mr = [h for h in hunks if not classify(h)]
        keep_mul = [h for h in hunks if classify(h)]
        if keep_mr:
            mr_blocks.append(preamble + [ln for h in keep_mr for ln in h])
        if keep_mul:
            mul_blocks.append(preamble + [ln for h in keep_mul for ln in h])

    args.out.mkdir(parents=True, exist_ok=True)
    mr_path, mul_path = args.out / "mr.patch", args.out / "gf54.patch"
    mr_path.write_text("".join("".join(b) for b in mr_blocks), encoding="utf-8")
    mul_path.write_text("".join("".join(b) for b in mul_blocks), encoding="utf-8")

    def stats(blocks: list[list[str]]) -> tuple[int, int, int]:
        added = sum(1 for b in blocks for ln in b
                    if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for b in blocks for ln in b
                      if ln.startswith("-") and not ln.startswith("---"))
        return len({b[0].split(" b/", 1)[1].strip() for b in blocks}), added, removed

    print(f"{'patch':12s} {'files':>5} {'added':>6} {'removed':>8}")
    for name, blocks in (("mr.patch", mr_blocks), ("gf54.patch", mul_blocks)):
        files, added, removed = stats(blocks)
        print(f"{name:12s} {files:5d} {added:6d} {removed:8d}")

    # Both must apply: mr to a clean tree, gf54 on top of it.
    scratch = args.out / "tree"
    if scratch.exists():
        subprocess.run(["rm", "-rf", str(scratch)], check=True)  # noqa: S603,S607
    scratch.mkdir(parents=True)
    archive = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(VENDOR), "archive", "HEAD"], check=True, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(scratch)], input=archive.stdout, check=True)  # noqa: S603,S607

    for label, patch in (("mr", mr_path), ("gf54 on top of mr", mul_path)):
        with patch.open("rb") as handle:
            applied = subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["patch", "-p1", "--forward", "--batch"], cwd=scratch,
                stdin=handle, capture_output=True, text=True)
        rejects = [ln for ln in applied.stdout.splitlines() if "FAILED" in ln or "fuzz" in ln]
        if applied.returncode != 0 or rejects:
            print(f"{label}: DOES NOT APPLY\n{applied.stdout[-1500:]}{applied.stderr[-800:]}")
            return 1
        print(f"{label}: applies cleanly ({len(applied.stdout.splitlines())} files checked)")
    print(f"\nwrote {mr_path.relative_to(ROOT)} and {mul_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
